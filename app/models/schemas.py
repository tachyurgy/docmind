import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    filename: str
    content_type: str
    page_count: int
    file_size_bytes: int
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class SourceCitation(BaseModel):
    document_name: str
    chunk_content: str
    page_number: int | None = None
    relevance_score: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    query_time_ms: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class ErrorResponse(BaseModel):
    detail: str
