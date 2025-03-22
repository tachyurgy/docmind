"""Seed script: inserts sample documents into DocMind for demo purposes.

Run with:
    python -m scripts.seed_data

Requires a running PostgreSQL database with the DocMind schema applied.
"""

import asyncio
import uuid

from sqlalchemy import text

from app.config import settings
from app.database import async_session, init_db
from app.models.document import Chunk, Document
from app.services.embeddings import get_embedding_service
from app.utils.chunking import RecursiveChunker

SAMPLE_DOCUMENTS = [
    {
        "filename": "architecture_decisions.md",
        "content_type": "text/markdown",
        "text": """# Architecture Decision Records

## ADR-001: Use PostgreSQL with pgvector for vector storage

### Context
We need a database that supports both traditional relational queries and vector
similarity search. The options considered were Pinecone (managed vector DB),
Weaviate (self-hosted vector DB), and PostgreSQL with the pgvector extension.

### Decision
We chose PostgreSQL with pgvector because it lets us keep all data in a single
database, simplifies operations, and avoids the cost and complexity of a separate
vector database service. The pgvector extension supports cosine similarity, L2
distance, and inner product operations with HNSW and IVFFlat indexing.

### Consequences
We accept slightly lower query performance compared to purpose-built vector
databases at very large scale (millions of vectors). For our expected workload
of tens of thousands of document chunks, pgvector performs well within acceptable
latency bounds.

## ADR-002: Hybrid retrieval with reciprocal rank fusion

### Context
Pure vector search can miss keyword-heavy matches. Pure keyword search misses
semantic similarity. Users expect both behaviors when asking questions.

### Decision
We combine pgvector cosine similarity with PostgreSQL full-text search using
tsvector/tsquery. Results are merged with a configurable alpha parameter that
weights the two signals.

### Consequences
Slightly more complex retrieval logic, but significantly better recall for
queries that mix conceptual and keyword-based intent.
""",
    },
    {
        "filename": "onboarding_guide.md",
        "content_type": "text/markdown",
        "text": """# Engineering Onboarding Guide

Welcome to the engineering team. This document covers everything you need to
get started.

## Development Environment

Install Python 3.12 or later. We use pyenv for version management. Clone the
repository and install dependencies with pip install -r requirements.txt. Copy
.env.example to .env and fill in your local database credentials.

## Database Setup

Install PostgreSQL 16 locally or use Docker. Create a database called docmind.
Install the pgvector extension by running CREATE EXTENSION IF NOT EXISTS vector
inside the database. Run alembic upgrade head to apply all migrations.

## Running the Application

Start the development server with uvicorn app.main:app --reload. The API docs
are available at http://localhost:8000/docs. Health check is at /health.

## Testing

Run the test suite with pytest. Tests use mocked external services so you do
not need API keys configured for testing. Coverage reports are generated with
pytest --cov=app.

## Code Style

We use ruff for linting and formatting. Run ruff check . before committing.
The CI pipeline enforces ruff checks and will reject PRs that fail linting.

## Deployment

We deploy to Render.com using Docker. The render.yaml file defines the service
configuration. Push to main to trigger an automatic deployment. Database
migrations run automatically as part of the deploy process.
""",
    },
    {
        "filename": "api_design_principles.md",
        "content_type": "text/markdown",
        "text": """# API Design Principles

## RESTful Resource Naming

Resources are nouns, not verbs. Use plural names for collections. Nest related
resources under their parent. For example, /api/v1/documents/{id}/chunks is
correct; /api/v1/getDocumentChunks is not.

## Pagination

All list endpoints support cursor-based or offset pagination. Default page size
is 20, maximum is 100. Response includes total count, current page, and page
size metadata.

## Error Handling

Errors return appropriate HTTP status codes with a JSON body containing a detail
field. 400 for validation errors, 401 for auth failures, 404 for not found, 429
for rate limits, 500 for server errors.

## Authentication

API key authentication via X-API-Key header. Optional by default for development.
Required in production. Keys are generated per-deployment and stored as
environment variables.

## Versioning

API versions are included in the URL path: /api/v1/. Breaking changes require
a version bump. Non-breaking additions (new fields, new endpoints) do not.

## Rate Limiting

In-memory token bucket per client IP. Default is 60 requests per minute.
Returns 429 with Retry-After header when exceeded.
""",
    },
]


async def seed() -> None:
    await init_db()

    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedding_service = get_embedding_service()

    async with async_session() as session:
        for doc_data in SAMPLE_DOCUMENTS:
            doc_id = uuid.uuid4()
            doc = Document(
                id=doc_id,
                filename=doc_data["filename"],
                content_type=doc_data["content_type"],
                source_text=doc_data["text"],
                page_count=1,
                file_size_bytes=len(doc_data["text"].encode("utf-8")),
            )
            session.add(doc)

            chunk_results = chunker.chunk_pages([(1, doc_data["text"])])
            if not chunk_results:
                continue

            texts = [c.content for c in chunk_results]
            embeddings = await embedding_service.embed_texts(texts)

            for i, (chunk_result, embedding) in enumerate(zip(chunk_results, embeddings)):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=doc_id,
                    content=chunk_result.content,
                    chunk_index=i,
                    page_number=chunk_result.page_number,
                    token_count=chunk_result.token_count,
                    embedding=embedding,
                    metadata_={"source": doc_data["filename"], "seeded": True},
                )
                session.add(chunk)

            await session.flush()

            for chunk_result in chunk_results:
                await session.execute(
                    text(
                        "UPDATE chunks SET ts_content = to_tsvector('english', :content) "
                        "WHERE document_id = :doc_id AND chunk_index = :idx"
                    ),
                    {
                        "content": chunk_result.content,
                        "doc_id": str(doc_id),
                        "idx": chunk_result.chunk_index,
                    },
                )

            print(f"Seeded: {doc_data['filename']} ({len(chunk_results)} chunks)")

        await session.commit()

    print("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(seed())
