import json
import time

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import check_rate_limit, verify_api_key
from app.database import get_db
from app.models.schemas import QueryRequest, QueryResponse, SourceCitation
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

logger = structlog.get_logger()

router = APIRouter(
    prefix="/api/v1/query",
    tags=["query"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Ask a question about uploaded documents and get an answer with citations.

    Performs hybrid search (vector + keyword), retrieves relevant chunks,
    and generates a cited answer.
    """
    start = time.perf_counter()

    retrieval = RetrievalService()
    results = await retrieval.hybrid_search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        alpha=request.alpha,
        document_ids=request.document_ids,
    )

    if not results:
        return QueryResponse(
            answer="No relevant documents found for your query. Try uploading documents first or rephrasing your question.",
            citations=[],
            query_time_ms=(time.perf_counter() - start) * 1000,
            model_used="none",
        )

    generation = GenerationService()
    answer_text, model_used = await generation.generate_answer(
        query=request.query,
        context_chunks=results,
        stream=False,
    )

    citations = [
        SourceCitation(
            document_name=r.document_name,
            chunk_content=r.chunk.content[:300],
            page_number=r.chunk.page_number,
            relevance_score=round(r.combined_score, 4),
        )
        for r in results
    ]

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("query_completed", query=request.query[:100], duration_ms=round(duration_ms, 2))

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        query_time_ms=round(duration_ms, 2),
        model_used=model_used,
    )


@router.post("/stream")
async def query_documents_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """Ask a question and receive a streaming answer via Server-Sent Events.

    Events:
    - metadata: Contains the model name being used.
    - token: Each text token as it is generated.
    - citations: The source citations for the answer.
    - done: Signals completion.
    - error: Signals an error occurred.
    """
    retrieval = RetrievalService()
    results = await retrieval.hybrid_search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        alpha=request.alpha,
        document_ids=request.document_ids,
    )

    if not results:
        async def empty_stream():
            yield {
                "event": "error",
                "data": '{"error": "No relevant documents found for your query."}',
            }

        return EventSourceResponse(empty_stream())

    generation = GenerationService()

    async def event_stream():
        stream = await generation.generate_answer(
            query=request.query,
            context_chunks=results,
            stream=True,
        )

        async for sse_chunk in stream:
            yield {"data": sse_chunk}

        citations = [
            {
                "document_name": r.document_name,
                "chunk_content": r.chunk.content[:300],
                "page_number": r.chunk.page_number,
                "relevance_score": round(r.combined_score, 4),
            }
            for r in results
        ]

        yield {"event": "citations", "data": json.dumps(citations)}

    return EventSourceResponse(event_stream())
