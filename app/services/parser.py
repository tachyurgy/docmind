import io

import structlog
from docx import Document as DocxDocument
from PyPDF2 import PdfReader

from app.utils.text import clean_text

logger = structlog.get_logger()

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/markdown": "markdown",
    "text/plain": "text",
}


class ParseResult:
    def __init__(self, pages: list[tuple[int, str]], page_count: int) -> None:
        self.pages = pages
        self.page_count = page_count

    @property
    def full_text(self) -> str:
        return "\n\n".join(text for _, text in self.pages if text.strip())


def parse_pdf(file_bytes: bytes) -> ParseResult:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = clean_text(text)
        if cleaned:
            pages.append((i + 1, cleaned))
    return ParseResult(pages=pages, page_count=len(reader.pages))


def parse_docx(file_bytes: bytes) -> ParseResult:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = clean_text(para.text)
        if text:
            paragraphs.append(text)
    full = "\n\n".join(paragraphs)
    return ParseResult(pages=[(1, full)], page_count=1)


def parse_markdown(file_bytes: bytes) -> ParseResult:
    text = file_bytes.decode("utf-8", errors="replace")
    cleaned = clean_text(text)
    return ParseResult(pages=[(1, cleaned)], page_count=1)


def parse_plaintext(file_bytes: bytes) -> ParseResult:
    text = file_bytes.decode("utf-8", errors="replace")
    cleaned = clean_text(text)
    return ParseResult(pages=[(1, cleaned)], page_count=1)


async def parse_document(file_bytes: bytes, content_type: str, filename: str) -> ParseResult:
    """Parse a document based on its content type and return structured pages."""
    doc_type = SUPPORTED_TYPES.get(content_type)

    if doc_type is None:
        if filename.endswith(".pdf"):
            doc_type = "pdf"
        elif filename.endswith(".docx"):
            doc_type = "docx"
        elif filename.endswith(".md"):
            doc_type = "markdown"
        else:
            doc_type = "text"

    logger.info("parsing_document", filename=filename, doc_type=doc_type, size=len(file_bytes))

    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "markdown": parse_markdown,
        "text": parse_plaintext,
    }

    parser = parsers.get(doc_type, parse_plaintext)
    return parser(file_bytes)
