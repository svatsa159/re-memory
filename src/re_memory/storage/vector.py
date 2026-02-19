"""Qdrant vector store client.

Stores and retrieves dense embeddings for similarity search.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)


COLLECTION_NAME = "re_memory_embeddings"


class VectorStore:
    """Qdrant-backed vector store for memory embeddings."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        embedding_dim: int = 768,
        collection_name: str = COLLECTION_NAME,
    ):
        self.url = url
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, timeout=10)
        return self._client

    def init(self):
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, point_id: str, vector: list[float], payload: dict | None = None):
        """Insert or update a vector."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {},
                )
            ],
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Search for similar vectors."""
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def delete(self, point_id: str):
        """Delete a vector by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )

    def status(self) -> dict:
        """Get collection info with a live connectivity check."""
        try:
            # Force a fresh connection check — don't rely on cached client state
            check_client = QdrantClient(url=self.url, timeout=3)
            info = check_client.get_collection(self.collection_name)
            check_client.close()
            return {
                "vectors_count": info.points_count,  # points_count = actual stored vectors
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"vectors_count": 0, "status": "unavailable"}

    def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None
