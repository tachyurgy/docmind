import re
from dataclasses import dataclass

import tiktoken


@dataclass
class ChunkResult:
    content: str
    chunk_index: int
    page_number: int | None
    token_count: int


class RecursiveChunker:
    """Splits text into overlapping chunks using a recursive strategy.

    Tries to split by paragraphs first, then sentences, then words, falling back
    to character-level splits only when necessary. Preserves page boundaries
    when page information is available.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def chunk_text(self, text: str, page_number: int | None = None) -> list[ChunkResult]:
        """Chunk a single block of text (optionally tagged with a page number)."""
        if not text.strip():
            return []

        segments = self._recursive_split(text)
        return self._merge_segments(segments, page_number)

    def chunk_pages(self, pages: list[tuple[int, str]]) -> list[ChunkResult]:
        """Chunk a list of (page_number, text) tuples, preserving page boundaries where possible."""
        all_chunks: list[ChunkResult] = []
        carry_over = ""
        carry_page: int | None = None

        for page_num, page_text in pages:
            page_text = page_text.strip()
            if not page_text:
                continue

            combined = f"{carry_over} {page_text}".strip() if carry_over else page_text
            combined_page = carry_page if carry_over else page_num

            if self.count_tokens(combined) <= self.chunk_size:
                carry_over = combined
                carry_page = combined_page
                continue

            chunks = self.chunk_text(combined, combined_page)
            if chunks:
                all_chunks.extend(chunks[:-1])
                carry_over = chunks[-1].content
                carry_page = combined_page
            else:
                carry_over = ""
                carry_page = None

        if carry_over:
            all_chunks.extend(self.chunk_text(carry_over, carry_page))

        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i

        return all_chunks

    def _recursive_split(self, text: str) -> list[str]:
        """Split text recursively: paragraphs, then sentences, then words."""
        if self.count_tokens(text) <= self.chunk_size:
            return [text]

        separators = [
            r"\n\n+",
            r"\n",
            r"(?<=[.!?])\s+",
            r"\s+",
        ]

        for sep in separators:
            parts = re.split(sep, text)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                return self._split_parts(parts)

        mid = len(text) // 2
        return self._recursive_split(text[:mid]) + self._recursive_split(text[mid:])

    def _split_parts(self, parts: list[str]) -> list[str]:
        """Given a list of text segments, combine them greedily into chunks."""
        results: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if self.count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if self.count_tokens(part) > self.chunk_size:
                    results.extend(self._recursive_split(part))
                    current = ""
                else:
                    current = part

        if current:
            results.append(current)

        return results

    def _merge_segments(
        self, segments: list[str], page_number: int | None
    ) -> list[ChunkResult]:
        """Convert text segments into ChunkResults with overlap applied."""
        if not segments:
            return []

        chunks: list[ChunkResult] = []
        for i, segment in enumerate(segments):
            if i > 0 and self.chunk_overlap > 0:
                prev_tokens = self._encoder.encode(segments[i - 1])
                overlap_tokens = prev_tokens[-self.chunk_overlap :]
                overlap_text = self._encoder.decode(overlap_tokens)
                segment = f"{overlap_text} {segment}"

            token_count = self.count_tokens(segment)
            chunks.append(
                ChunkResult(
                    content=segment,
                    chunk_index=i,
                    page_number=page_number,
                    token_count=token_count,
                )
            )

        return chunks
