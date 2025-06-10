# DocMind

A document Q&A platform powered by Retrieval-Augmented Generation. Upload PDFs, DOCX files, or Markdown, then ask natural language questions and get cited, streaming answers grounded in your documents.

## Architecture

```
                    +-------------------+
                    |   Client / cURL   |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   FastAPI (async)  |
                    |   CORS / Auth /    |
                    |   Rate Limiting    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     | Upload & Ingest  |          | Query & Answer  |
     |  Parse -> Chunk  |          | Hybrid Search   |
     |  Embed -> Store  |          | Generate + Cite |
     +--------+---------+          +--------+--------+
              |                             |
     +--------v--------+          +--------v--------+
     |  OpenAI Embed    |          | Anthropic Claude|
     | text-embed-3-sm  |          | (primary gen)   |
     +---------+--------+          | OpenAI gpt-4o-  |
               |                   | mini (fallback) |
               |                   +--------+--------+
               |                            |
         +-----v----------------------------v-----+
         |      PostgreSQL 16 + pgvector           |
         |  Vector search (cosine) + Full-text     |
         |  search (tsvector/tsquery)              |
         +----------------------------------------+
```

## Features

- Upload PDF, DOCX, Markdown, and plain text documents
- Automatic text extraction, recursive chunking, and embedding
- Hybrid retrieval combining vector similarity (pgvector) with keyword search (tsvector)
- Configurable alpha weighting between vector and keyword scores
- Streaming answers via Server-Sent Events using Claude or GPT-4o-mini
- Inline source citations with document name, page number, and relevance score
- Optional API key authentication and per-IP rate limiting
- Async throughout (FastAPI, SQLAlchemy 2.0, asyncpg)

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16 with the `pgvector` extension
- OpenAI API key (for embeddings)
- Anthropic API key (for generation, optional if using OpenAI fallback)

### Local Development

```bash
# Clone and enter the project
git clone https://github.com/tachyurgy/docmind.git
cd docmind

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database URL and API keys

# Create the pgvector extension (if not already done)
psql -d docmind -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
alembic upgrade head

# Start the development server
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t docmind .
docker run -p 8000:8000 --env-file .env docmind
```

## API Overview

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload and ingest a document |
| GET | `/api/v1/documents` | List documents (paginated) |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete a document and its chunks |

### Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Ask a question, get answer with citations |
| POST | `/api/v1/query/stream` | Ask a question, stream the answer via SSE |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Application and database health check |

### Query Request Body

```json
{
  "query": "What database did we choose and why?",
  "top_k": 5,
  "alpha": 0.7,
  "document_ids": null
}
```

- `top_k` controls how many chunks to retrieve (1-20, default 5)
- `alpha` weights vector vs keyword search (0.0 = keyword only, 1.0 = vector only, default 0.7)
- `document_ids` optionally scopes the search to specific documents

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/docmind` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | (required) |
| `ANTHROPIC_API_KEY` | Anthropic API key for generation | (optional) |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | Embedding vector dimensions | `1536` |
| `CHUNK_SIZE` | Target chunk size in tokens | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks in tokens | `64` |
| `MAX_UPLOAD_SIZE_MB` | Maximum upload file size | `50` |
| `API_KEY` | API key for authentication (blank = disabled) | (blank) |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000,http://localhost:8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Deployment (Render)

This project includes a `render.yaml` for one-click deployment to Render.com.

1. Push the repository to GitHub
2. Connect the repository in your Render dashboard
3. Render will detect `render.yaml` and create the web service and database
4. Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in the Render environment
5. The API key for client authentication is auto-generated

The Dockerfile uses a multi-stage build to keep the production image small. Health checks are configured at `/health`.

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run a specific test file
pytest tests/test_chunking.py -v
```

Tests mock all external services (OpenAI, Anthropic, PostgreSQL) so no API keys or database are needed to run them.

## Linting

```bash
ruff check .
ruff format --check .
```

## Seeding Demo Data

```bash
python -m scripts.seed_data
```

This inserts a few sample Markdown documents for testing the query pipeline end-to-end.

## License

MIT
