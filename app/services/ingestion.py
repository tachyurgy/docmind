import uuid

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Chunk, Document
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.parser import parse_document
from app.utils.chunking import RecursiveChunker

logger = structlog.get_logger()


class IngestionService:
    """Orchestrates document ingestion: parse, chunk, embed, and store."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embeddings = embedding_service or get_embedding_service()
        self._chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        db: AsyncSession,
    ) -> Document:
        """Ingest a document: parse it, chunk the text, generate embeddings, and store everything.

        Returns the created Document object with its chunks.
        """
        logger.info("ingestion_started", filename=filename, size=len(file_bytes))

        parsed = await parse_document(file_bytes, content_type, filename)

        doc = Document(
            id=uuid.uuid4(),
            filename=filename,
            content_type=content_type,
            source_text=parsed.full_text,
            page_count=parsed.page_count,
            file_size_bytes=len(file_bytes),
        )
        db.add(doc)

        chunk_results = self._chunker.chunk_pages(parsed.pages)
        if not chunk_results:
            logger.warning("no_chunks_produced", filename=filename)
            await db.flush()
            return doc

        logger.info("chunks_produced", filename=filename, count=len(chunk_results))

        texts = [c.content for c in chunk_results]
        embeddings = await self._embeddings.embed_texts(texts)

        chunks: list[Chunk] = []
        for i, (chunk_result, embedding) in enumerate(zip(chunk_results, embeddings)):
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=doc.id,
                content=chunk_result.content,
                chunk_index=i,
                page_number=chunk_result.page_number,
                token_count=chunk_result.token_count,
                embedding=embedding,
                metadata_={"source": filename},
            )
            chunks.append(chunk)

        db.add_all(chunks)
        await db.flush()

        for chunk in chunks:
            await db.execute(
                text(
                    "UPDATE chunks SET ts_content = to_tsvector('english', :content) "
                    "WHERE id = :id"
                ),
                {"content": chunk.content, "id": str(chunk.id)},
            )

        logger.info(
            "ingestion_complete",
            filename=filename,
            document_id=str(doc.id),
            chunk_count=len(chunks),
        )

        return doc
