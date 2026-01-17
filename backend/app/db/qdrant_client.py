from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from typing import List, Dict, Any, Optional
from app.config import get_settings
from app.models.chunk import Chunk, SearchResult
import uuid

settings = get_settings()


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dim = settings.embedding_dim
    
    def create_collection(self, dimension: int = None):
        """Create collection for paper chunks"""
        dim = dimension or self.embedding_dim
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")
    
    def insert_chunks(self, chunks: List[Chunk]):
        """Insert chunks with embeddings into Qdrant"""
        if not chunks:
            return
        
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant point ID
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "paper_id": chunk.metadata.paper_id,
                    "paper_title": chunk.metadata.paper_title,
                    "authors": chunk.metadata.authors,
                    "year": chunk.metadata.year,
                    "section_title": chunk.metadata.section_title,
                    "section_id": chunk.metadata.section_id,
                    "page_start": chunk.metadata.page_start,
                    "page_end": chunk.metadata.page_end,
                    "chunk_index": chunk.metadata.chunk_index,
                    "has_table": chunk.metadata.has_table,
                    "has_equation": chunk.metadata.has_equation,
                    "has_figure": chunk.metadata.has_figure
                }
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Inserted {len(points)} chunks into Qdrant")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks
        
        Args:
            query_vector: Query embedding
            limit: Number of results
            filters: Optional filters (e.g., {"year": 2023})
        """
        # Build filter
        qdrant_filter = None
        if filters:
            conditions = []
            
            if "year" in filters:
                conditions.append(
                    FieldCondition(
                        key="year",
                        match=MatchValue(value=filters["year"])
                    )
                )
            
            if "paper_id" in filters:
                conditions.append(
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=filters["paper_id"])
                    )
                )
            
            if "section_title" in filters:
                conditions.append(
                    FieldCondition(
                        key="section_title",
                        match=MatchValue(value=filters["section_title"])
                    )
                )
            
            if "has_table" in filters:
                conditions.append(
                    FieldCondition(
                        key="has_table",
                        match=MatchValue(value=filters["has_table"])
                    )
                )
            
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter
        )
        
        # Convert to SearchResult
        search_results = []
        for result in results:
            payload = result.payload
            
            from app.models.chunk import ChunkMetadata
            metadata = ChunkMetadata(
                paper_id=payload["paper_id"],
                paper_title=payload["paper_title"],
                authors=payload["authors"],
                year=payload.get("year"),
                section_title=payload["section_title"],
                section_id=payload["section_id"],
                page_start=payload["page_start"],
                page_end=payload["page_end"],
                chunk_index=payload["chunk_index"],
                total_chunks=0,  # Not stored in Qdrant
                has_table=payload.get("has_table", False),
                has_equation=payload.get("has_equation", False),
                has_figure=payload.get("has_figure", False)
            )
            
            search_result = SearchResult(
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                score=result.score,
                metadata=metadata
            )
            search_results.append(search_result)
        
        return search_results
    
    def delete_paper_chunks(self, paper_id: str):
        """Delete all chunks for a specific paper"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    )
                ]
            )
        )
        print(f"Deleted chunks for paper: {paper_id}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": info.config.params.vectors.size,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count
        }