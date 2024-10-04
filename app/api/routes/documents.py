import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import check_rate_limit, verify_api_key
from app.config import settings
from app.database import get_db
from app.models.document import Chunk, Document
from app.models.schemas import DocumentListResponse, DocumentResponse
from app.services.ingestion import IngestionService

logger = structlog.get_logger()

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a document (PDF, DOCX, Markdown, or plain text) for ingestion.

    The document is parsed, chunked, embedded, and stored in the database.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB",
        )

    content_type = file.content_type or "application/octet-stream"

    service = IngestionService()
    doc = await service.ingest_document(file_bytes, file.filename, content_type, db)

    chunk_count_result = await db.execute(
        select(func.count()).where(Chunk.document_id == doc.id)
    )
    chunk_count = chunk_count_result.scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        page_count=doc.page_count,
        file_size_bytes=doc.file_size_bytes,
        chunk_count=chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """List all uploaded documents with pagination."""
    offset = (page - 1) * page_size

    total_result = await db.execute(select(func.count()).select_from(Document))
    total = total_result.scalar() or 0

    docs_result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    docs = docs_result.scalars().all()

    items: list[DocumentResponse] = []
    for doc in docs:
        chunk_count_result = await db.execute(
            select(func.count()).where(Chunk.document_id == doc.id)
        )
        chunk_count = chunk_count_result.scalar() or 0
        items.append(
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                page_count=doc.page_count,
                file_size_bytes=doc.file_size_bytes,
                chunk_count=chunk_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
        )

    return DocumentListResponse(
        documents=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Get details for a specific document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count_result = await db.execute(
        select(func.count()).where(Chunk.document_id == doc.id)
    )
    chunk_count = chunk_count_result.scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        page_count=doc.page_count,
        file_size_bytes=doc.file_size_bytes,
        chunk_count=chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and all its associated chunks."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(doc)
    logger.info("document_deleted", document_id=str(document_id), filename=doc.filename)
