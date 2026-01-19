from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
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
        """Search for similar chunks"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
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
        
        return search_results
    
    def count(self) -> int:
        """Get total number of chunks"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count