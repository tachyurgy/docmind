import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.models.document import Chunk, Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id=uuid.uuid4(),
        filename="test_document.pdf",
        content_type="application/pdf",
        source_text="This is a test document about machine learning and natural language processing.",
        page_count=3,
        file_size_bytes=1024,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    chunks = []
    texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
    ]
    for i, text in enumerate(texts):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            content=text,
            chunk_index=i,
            page_number=i + 1,
            token_count=len(text.split()),
            embedding=[0.1] * 1536,
            metadata_={"source": "test_document.pdf"},
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock()
    service.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.1 * (i + 1)] * 1536 for i in range(len(texts))]
    )
    service.embed_query = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_db_session() -> AsyncMock:
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_docmind",
        openai_api_key="sk-test-key",
        anthropic_api_key="sk-ant-test-key",
        api_key="test-api-key",
        chunk_size=256,
        chunk_overlap=32,
    )


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI application.

    Patches the database dependency and API key verification to allow
    tests to run without a real database or valid API keys.
    """
    from app.api.dependencies import verify_api_key
    from app.database import get_db
    from app.main import app

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    async def override_get_db():
        yield mock_session

    async def override_verify_api_key():
        return "test-key"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = override_verify_api_key

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_session

    app.dependency_overrides.clear()
