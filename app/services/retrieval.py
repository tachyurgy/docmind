import uuid

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, Document
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = structlog.get_logger()


class SearchResult:
    def __init__(
        self,
        chunk: Chunk,
        document_name: str,
        vector_score: float,
        keyword_score: float,
        combined_score: float,
    ) -> None:
        self.chunk = chunk
        self.document_name = document_name
        self.vector_score = vector_score
        self.keyword_score = keyword_score
        self.combined_score = combined_score


class RetrievalService:
    """Hybrid retrieval combining pgvector cosine similarity with PostgreSQL full-text search."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embeddings = embedding_service or get_embedding_service()

    async def hybrid_search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 5,
        alpha: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search combining vector similarity and keyword matching.

        Args:
            query: The user's natural language question.
            db: Async database session.
            top_k: Number of results to return.
            alpha: Weight for vector score (1-alpha for keyword score).
            document_ids: Optional list of document IDs to scope the search.

        Returns:
            Ranked list of SearchResult objects.
        """
        query_embedding = await self._embeddings.embed_query(query)

        vector_results = await self._vector_search(
            query_embedding, db, top_k=top_k * 2, document_ids=document_ids
        )
        keyword_results = await self._keyword_search(
            query, db, top_k=top_k * 2, document_ids=document_ids
        )

        return self._merge_results(vector_results, keyword_results, alpha, top_k)

    async def _vector_search(
        self,
        query_embedding: list[float],
        db: AsyncSession,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[tuple[Chunk, str, float]]:
        """Search chunks by cosine similarity against the query embedding."""
        distance_expr = Chunk.embedding.cosine_distance(query_embedding)

        stmt = (
            select(Chunk, Document.filename, distance_expr.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance_expr)
            .limit(top_k)
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        result = await db.execute(stmt)
        rows = result.all()

        return [(row[0], row[1], 1.0 - float(row[2])) for row in rows]

    async def _keyword_search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[tuple[Chunk, str, float]]:
        """Search chunks using PostgreSQL full-text search with ts_rank."""
        ts_query = func.plainto_tsquery("english", query)
        rank_expr = func.ts_rank(Chunk.ts_content, ts_query)

        stmt = (
            select(Chunk, Document.filename, rank_expr.label("rank"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.ts_content.op("@@")(ts_query))
            .order_by(rank_expr.desc())
            .limit(top_k)
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            return []

        max_rank = max(float(row[2]) for row in rows) or 1.0
        return [(row[0], row[1], float(row[2]) / max_rank) for row in rows]

    def _merge_results(
        self,
        vector_results: list[tuple[Chunk, str, float]],
        keyword_results: list[tuple[Chunk, str, float]],
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        """Combine vector and keyword results using weighted scoring."""
        scores: dict[uuid.UUID, dict] = {}

        for chunk, doc_name, score in vector_results:
            scores[chunk.id] = {
                "chunk": chunk,
                "document_name": doc_name,
                "vector_score": score,
                "keyword_score": 0.0,
            }

        for chunk, doc_name, score in keyword_results:
            if chunk.id in scores:
                scores[chunk.id]["keyword_score"] = score
            else:
                scores[chunk.id] = {
                    "chunk": chunk,
                    "document_name": doc_name,
                    "vector_score": 0.0,
                    "keyword_score": score,
                }

        results: list[SearchResult] = []
        for data in scores.values():
            combined = alpha * data["vector_score"] + (1 - alpha) * data["keyword_score"]
            results.append(
                SearchResult(
                    chunk=data["chunk"],
                    document_name=data["document_name"],
                    vector_score=data["vector_score"],
                    keyword_score=data["keyword_score"],
                    combined_score=combined,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]
