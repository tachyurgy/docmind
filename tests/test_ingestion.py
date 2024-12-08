from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ingestion import IngestionService
from app.services.parser import parse_document


class TestParser:
    @pytest.mark.asyncio
    async def test_parse_pdf(self) -> None:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one content."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("app.services.parser.PdfReader", return_value=mock_reader):
            result = await parse_document(b"fake-pdf-bytes", "application/pdf", "test.pdf")

        assert result.page_count == 1
        assert len(result.pages) == 1
        assert result.pages[0][0] == 1
        assert "Page one content" in result.pages[0][1]

    @pytest.mark.asyncio
    async def test_parse_pdf_multiple_pages(self) -> None:
        pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"Content of page {i + 1}."
            pages.append(mock_page)

        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("app.services.parser.PdfReader", return_value=mock_reader):
            result = await parse_document(b"fake-pdf-bytes", "application/pdf", "multi.pdf")

        assert result.page_count == 3
        assert len(result.pages) == 3
        assert result.pages[2][0] == 3

    @pytest.mark.asyncio
    async def test_parse_plaintext(self) -> None:
        text = "Hello, this is plain text content."
        result = await parse_document(
            text.encode("utf-8"), "text/plain", "readme.txt"
        )
        assert result.page_count == 1
        assert "plain text content" in result.full_text

    @pytest.mark.asyncio
    async def test_parse_markdown(self) -> None:
        md = "# Title\n\nSome markdown content with **bold** text."
        result = await parse_document(
            md.encode("utf-8"), "text/markdown", "notes.md"
        )
        assert result.page_count == 1
        assert "Title" in result.full_text

    @pytest.mark.asyncio
    async def test_parse_unknown_falls_back_to_text(self) -> None:
        content = "Just some bytes that we treat as text."
        result = await parse_document(
            content.encode("utf-8"), "application/octet-stream", "data.xyz"
        )
        assert result.page_count == 1
        assert "treat as text" in result.full_text


class TestIngestionService:
    @pytest.mark.asyncio
    async def test_ingest_document_calls_parse_chunk_embed(
        self, mock_embedding_service: MagicMock, mock_db_session: AsyncMock
    ) -> None:
        service = IngestionService(embedding_service=mock_embedding_service)

        text_content = "This is a test document with enough content to create chunks."
        file_bytes = text_content.encode("utf-8")

        doc = await service.ingest_document(
            file_bytes=file_bytes,
            filename="test.txt",
            content_type="text/plain",
            db=mock_db_session,
        )

        assert doc.filename == "test.txt"
        assert doc.content_type == "text/plain"
        assert doc.file_size_bytes == len(file_bytes)
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_document_generates_embeddings(
        self, mock_embedding_service: MagicMock, mock_db_session: AsyncMock
    ) -> None:
        service = IngestionService(embedding_service=mock_embedding_service)

        long_text = "Sentence about testing. " * 100
        file_bytes = long_text.encode("utf-8")

        await service.ingest_document(
            file_bytes=file_bytes,
            filename="long.txt",
            content_type="text/plain",
            db=mock_db_session,
        )

        mock_embedding_service.embed_texts.assert_called_once()
        call_args = mock_embedding_service.embed_texts.call_args[0][0]
        assert len(call_args) > 0

    @pytest.mark.asyncio
    async def test_ingest_empty_document(
        self, mock_embedding_service: MagicMock, mock_db_session: AsyncMock
    ) -> None:
        service = IngestionService(embedding_service=mock_embedding_service)

        doc = await service.ingest_document(
            file_bytes=b"   ",
            filename="empty.txt",
            content_type="text/plain",
            db=mock_db_session,
        )

        assert doc.filename == "empty.txt"
        mock_embedding_service.embed_texts.assert_not_called()
