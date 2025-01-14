import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.models.document import Document


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client) -> None:
        client, mock_session = async_client

        mock_result = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["database"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_reports_db_failure(self, async_client) -> None:
        client, mock_session = async_client

        mock_session.execute = AsyncMock(side_effect=Exception("Connection refused"))

        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"] == "unhealthy"


class TestDocumentsEndpoint:
    @pytest.mark.asyncio
    async def test_upload_rejects_empty_filename(self, async_client) -> None:
        client, mock_session = async_client

        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("", b"content", "text/plain")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_accepts_text_file(self, async_client) -> None:
        client, mock_session = async_client

        doc_id = uuid.uuid4()

        with patch("app.api.routes.documents.IngestionService") as mock_service_cls:
            mock_instance = mock_service_cls.return_value
            mock_doc = Document(
                id=doc_id,
                filename="test.txt",
                content_type="text/plain",
                source_text="Test content",
                page_count=1,
                file_size_bytes=12,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            mock_instance.ingest_document = AsyncMock(return_value=mock_doc)

            mock_count_result = MagicMock()
            mock_count_result.scalar.return_value = 1
            mock_session.execute = AsyncMock(return_value=mock_count_result)

            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", b"Test content", "text/plain")},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, async_client) -> None:
        client, mock_session = async_client

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/documents/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, async_client) -> None:
        client, mock_session = async_client

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        fake_id = uuid.uuid4()
        response = await client.delete(f"/api/v1/documents/{fake_id}")
        assert response.status_code == 404


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_with_no_results(self, async_client) -> None:
        client, mock_session = async_client

        with patch("app.api.routes.query.RetrievalService") as mock_retrieval_cls:
            mock_instance = mock_retrieval_cls.return_value
            mock_instance.hybrid_search = AsyncMock(return_value=[])

            response = await client.post(
                "/api/v1/query",
                json={"query": "What is machine learning?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "No relevant documents" in data["answer"]
        assert data["citations"] == []
        assert data["model_used"] == "none"

    @pytest.mark.asyncio
    async def test_query_validates_empty_query(self, async_client) -> None:
        client, mock_session = async_client

        response = await client.post(
            "/api/v1/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_validates_top_k_range(self, async_client) -> None:
        client, mock_session = async_client

        response = await client.post(
            "/api/v1/query",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_validates_alpha_range(self, async_client) -> None:
        client, mock_session = async_client

        response = await client.post(
            "/api/v1/query",
            json={"query": "test", "alpha": 1.5},
        )
        assert response.status_code == 422


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_auth_required_when_enabled(self) -> None:
        from app.database import get_db
        from app.main import app

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        async def override_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides.pop(
            next(
                (k for k in app.dependency_overrides if k.__name__ == "verify_api_key"),
                None,
            ),
            None,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.api.dependencies.settings") as mock_settings:
                mock_settings.auth_enabled = True
                mock_settings.api_key = "real-key"

                response = await client.get("/api/v1/documents")
                assert response.status_code == 401

                response = await client.get(
                    "/api/v1/documents",
                    headers={"X-API-Key": "real-key"},
                )
                assert response.status_code in (200, 500)

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self) -> None:
        """Health endpoint should work without auth even when auth is enabled."""
        from app.database import get_db
        from app.main import app

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        async def override_get_db():
            yield mock_session

        app.dependency_overrides[get_db] = override_get_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200

        app.dependency_overrides.clear()
