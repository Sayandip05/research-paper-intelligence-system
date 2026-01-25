from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from typing import List, Optional
import uuid
from app.config import get_settings
from app.models.chunk import Chunk, SearchResult, ChunkMetadata

settings = get_settings()


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
    
    def create_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {self.collection_name}")
        else:
            print(f"✅ Collection exists: {self.collection_name}")
    
    def insert_chunks(self, chunks: List[Chunk]):
        """Insert chunks into Qdrant"""
        points = []
        
        for chunk in chunks:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "paper_id": chunk.metadata.paper_id,
                    "paper_title": chunk.metadata.paper_title,
                    "section_title": chunk.metadata.section_title,
                    "page_start": chunk.metadata.page_start,
                    "page_end": chunk.metadata.page_end
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ Inserted {len(points)} chunks into Qdrant")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks (unfiltered - for backward compatibility)"""
        return self.search_with_filter(query_vector, limit=limit, allowed_sections=None)
    
    def search_with_filter(
        self,
        query_vector: List[float],
        limit: int = 5,
        allowed_sections: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks WITH SECTION FILTERING.
        
        This is the PRIMARY method for intent-aware retrieval.
        
        Args:
            query_vector: Embedding of the query
            limit: Max results to return
            allowed_sections: List of canonical section names to include.
                              If None, searches all sections (unfiltered).
        
        Returns:
            List of SearchResult with metadata
        """
        # Build filter if sections specified
        query_filter = None
        if allowed_sections:
            # Always exclude "Unknown" unless explicitly requested
            filtered_sections = [s for s in allowed_sections if s != "Unknown"]
            
            if filtered_sections:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="section_title",
                            match=MatchAny(any=filtered_sections)
                        )
                    ]
                )
                print(f"  🔍 Filtering to sections: {filtered_sections}")
        
        # Execute search with filter
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        ).points
        
        search_results = []
        for result in results:
            payload = result.payload
            
            search_result = SearchResult(
                text=payload["text"],
                score=result.score,
                metadata=ChunkMetadata(
                    paper_id=payload["paper_id"],
                    paper_title=payload["paper_title"],
                    section_title=payload["section_title"],
                    page_start=payload["page_start"],
                    page_end=payload["page_end"]
                )
            )
            search_results.append(search_result)
        
        if allowed_sections:
            print(f"  📊 Retrieved {len(search_results)} chunks from allowed sections")
        
        return search_results
    
    def count(self) -> int:
        """Get total number of chunks"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
