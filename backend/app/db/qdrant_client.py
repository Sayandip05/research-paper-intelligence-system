from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny,
    SparseVector, SparseVectorParams, SparseIndexParams
)
from typing import List, Optional, Dict, Any
import uuid
from app.config import get_settings
from app.models.chunk import Chunk, SearchResult, ChunkMetadata
from app.models.image import ImageMetadata, ImageSearchResult
settings = get_settings()


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
    
    def create_collection(self):
        """Create hybrid collection with dense + sparse vectors"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # 🆕 Hybrid configuration: dense + sparse vectors
            # NOTE: LlamaIndex QdrantVectorStore prefixes vector names with "text-"
            vectors_config = {
                "text-dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                ),
            }
            
            # 🆕 Sparse vector config for BM42
            sparse_vectors_config = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,  # Keep in memory for speed
                    )
                )
            }
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            print(f"✅ Created HYBRID collection: {self.collection_name}")
            print(f"   - Dense vectors: {settings.embedding_dim}-dim (BGE)")
            print(f"   - Sparse vectors: BM42")
        else:
            print(f"✅ Collection exists: {self.collection_name}")
    
    def insert_chunks(self, chunks: List[Chunk]):
        """Insert chunks with BOTH dense and sparse embeddings"""
        points = []
        
        for chunk in chunks:
            # Validate embeddings exist
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} missing dense embedding")
            
            # 🆕 Check for sparse embedding
            sparse_embedding = getattr(chunk, 'sparse_embedding', None)
            if sparse_embedding is None and settings.enable_hybrid_search:
                raise ValueError(f"Chunk {chunk.chunk_id} missing sparse embedding")
            
            # 🆕 Build hybrid point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text-dense": chunk.embedding,  # Dense vector (BGE) - matches LlamaIndex naming
                    "sparse": sparse_embedding if settings.enable_hybrid_search else None
                },
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
        print(f"✅ Inserted {len(points)} chunks into Qdrant (hybrid mode)")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Backward-compatible search (dense-only, unfiltered)"""
        return self.search_with_filter(
            query_vector, 
            limit=limit, 
            allowed_sections=None,
            query_sparse_vector=None
        )
    
    def search_with_filter(
        self,
        query_vector: List[float],
        limit: int = 5,
        allowed_sections: Optional[List[str]] = None,
        query_sparse_vector: Optional[SparseVector] = None
    ) -> List[SearchResult]:
        """
        🆕 HYBRID SEARCH with section filtering
        
        Combines dense (semantic) and sparse (keyword) retrieval.
        
        Args:
            query_vector: Dense embedding (BGE)
            limit: Max results
            allowed_sections: Section filter
            query_sparse_vector: Sparse embedding (BM42)
        
        Returns:
            List of SearchResult ranked by hybrid score
        """
        # Build section filter
        query_filter = None
        if allowed_sections:
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
        
        # 🆕 Hybrid search or dense-only
        if settings.enable_hybrid_search and query_sparse_vector:
            print(f"  🔀 Running HYBRID search (dense + sparse)")
            results = self._hybrid_search(
                query_vector, 
                query_sparse_vector, 
                limit, 
                query_filter
            )
        else:
            print(f"  📊 Running DENSE-only search")
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="text-dense",
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            ).points
        
        # Convert to SearchResult
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
        
        print(f"  📊 Retrieved {len(search_results)} chunks")
        return search_results
    
    def _hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: SparseVector,
        limit: int,
        query_filter: Optional[Filter]
    ) -> List[Any]:
        """
        🆕 Internal: Perform hybrid search with RRF fusion
        
        Strategy:
        1. Dense search → top-K results
        2. Sparse search → top-K results
        3. RRF fusion → merged ranking
        """
        # Search 1: Dense vector search
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="text-dense",
            query_filter=query_filter,
            limit=limit * 2,  # Over-fetch for better fusion
            with_payload=True
        ).points
        
        # Search 2: Sparse vector search
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="sparse",
            query_filter=query_filter,
            limit=limit * 2,
            with_payload=True
        ).points
        
        # 🆕 RRF Fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results, limit)
        
        return fused_results
    
    def _rrf_fusion(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        limit: int
    ) -> List[Any]:
        """
        🆕 Reciprocal Rank Fusion (RRF)
        
        Formula: RRF_score = Σ 1/(k + rank_i)
        where k = 60 (standard constant)
        """
        k = settings.rrf_k
        
        # Build rank maps
        dense_ranks = {point.id: rank for rank, point in enumerate(dense_results, 1)}
        sparse_ranks = {point.id: rank for rank, point in enumerate(sparse_results, 1)}
        
        # Collect all unique IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for point_id in all_ids:
            score = 0.0
            if point_id in dense_ranks:
                score += settings.dense_weight / (k + dense_ranks[point_id])
            if point_id in sparse_ranks:
                score += settings.sparse_weight / (k + sparse_ranks[point_id])
            rrf_scores[point_id] = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build result list with fused scores
        fused_results = []
        point_map = {p.id: p for p in dense_results + sparse_results}
        
        for point_id, rrf_score in sorted_ids:
            if point_id in point_map:
                point = point_map[point_id]
                point.score = rrf_score  # Override with RRF score
                fused_results.append(point)
        
        print(f"  🔀 RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused_results)} fused")
        
        return fused_results
    
    def count(self) -> int:
        """Get total number of chunks"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    # ==================== IMAGE METHODS ====================

    def create_image_collection(self):
        """🆕 Create collection for image embeddings"""
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.qdrant_image_collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=settings.qdrant_image_collection_name,
                vectors_config=VectorParams(
                    size=settings.clip_embedding_dim,  # 512-dim for CLIP
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created IMAGE collection: {settings.qdrant_image_collection_name}")
            print(f"   - CLIP vectors: {settings.clip_embedding_dim}-dim (ViT-B/32)")
        else:
            print(f"✅ Image collection exists: {settings.qdrant_image_collection_name}")

    def insert_images(
        self,
        images_data: List[tuple]  # List of (ImageMetadata, embedding)
    ):
        """🆕 Insert image embeddings into Qdrant"""
        points = []
        
        for metadata, embedding in images_data:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "image_id": metadata.image_id,
                    "paper_id": metadata.paper_id,
                    "paper_title": metadata.paper_title,
                    "page_number": metadata.page_number,
                    "caption": metadata.caption,
                    "image_type": metadata.image_type
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=settings.qdrant_image_collection_name,
                points=points
            )
            print(f"✅ Inserted {len(points)} image embeddings into Qdrant")

    def search_images(
        self,
        query_vector: List[float],
        limit: int = 3,
        min_score: float = 0.3
    ) -> List[ImageSearchResult]:
        """
        🆕 Search for images using text query embedding
        
        Args:
            query_vector: CLIP text embedding
            limit: Max results
            min_score: Minimum similarity threshold
        
        Returns:
            List of ImageSearchResult
        """
        results = self.client.query_points(
            collection_name=settings.qdrant_image_collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=min_score
        ).points
        
        search_results = []
        for result in results:
            payload = result.payload
            
            search_result = ImageSearchResult(
                image_id=payload["image_id"],
                paper_title=payload["paper_title"],
                page_number=payload["page_number"],
                caption=payload.get("caption"),
                score=result.score,
                metadata=ImageMetadata(
                    image_id=payload["image_id"],
                    paper_id=payload["paper_id"],
                    paper_title=payload["paper_title"],
                    page_number=payload["page_number"],
                    caption=payload.get("caption"),
                    image_type=payload.get("image_type", "figure")
                )
            )
            search_results.append(search_result)
        
        print(f"  🖼️  Retrieved {len(search_results)} images")
        return search_results

    def count_images(self) -> int:
        """🆕 Get total number of images"""
        try:
            info = self.client.get_collection(settings.qdrant_image_collection_name)
            return info.points_count
        except:
            return 0

    def create_image_collection(self):
        """🆕 Create collection for image embeddings"""
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.qdrant_image_collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=settings.qdrant_image_collection_name,
                vectors_config=VectorParams(
                    size=settings.clip_embedding_dim,  # 512-dim for CLIP
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created IMAGE collection: {settings.qdrant_image_collection_name}")
            print(f"   - CLIP vectors: {settings.clip_embedding_dim}-dim (ViT-B/32)")
        else:
            print(f"✅ Image collection exists: {settings.qdrant_image_collection_name}")

    def insert_images(
        self,
        images_data: List[tuple]  # List of (ImageMetadata, embedding)
    ):
        """🆕 Insert image embeddings into Qdrant"""
        points = []
        
        for metadata, embedding in images_data:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "image_id": metadata.image_id,
                    "paper_id": metadata.paper_id,
                    "paper_title": metadata.paper_title,
                    "page_number": metadata.page_number,
                    "caption": metadata.caption,
                    "image_type": metadata.image_type
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=settings.qdrant_image_collection_name,
                points=points
            )
            print(f"✅ Inserted {len(points)} image embeddings into Qdrant")

    def search_images(
        self,
        query_vector: List[float],
        limit: int = 3,
        min_score: float = 0.3
    ) -> List[ImageSearchResult]:
        """
        🆕 Search for images using text query embedding
        
        Args:
            query_vector: CLIP text embedding
            limit: Max results
            min_score: Minimum similarity threshold
        
        Returns:
            List of ImageSearchResult
        """
        results = self.client.query_points(
            collection_name=settings.qdrant_image_collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=min_score
        ).points
        
        search_results = []
        for result in results:
            payload = result.payload
            
            search_result = ImageSearchResult(
                image_id=payload["image_id"],
                paper_title=payload["paper_title"],
                page_number=payload["page_number"],
                caption=payload.get("caption"),
                score=result.score,
                metadata=ImageMetadata(
                    image_id=payload["image_id"],
                    paper_id=payload["paper_id"],
                    paper_title=payload["paper_title"],
                    page_number=payload["page_number"],
                    caption=payload.get("caption"),
                    image_type=payload.get("image_type", "figure")
                )
            )
            search_results.append(search_result)
        
        print(f"  🖼️  Retrieved {len(search_results)} images")
        return search_results

    def count_images(self) -> int:
        """🆕 Get total number of images"""
        try:
            info = self.client.get_collection(settings.qdrant_image_collection_name)
            return info.points_count
        except:
            return 0