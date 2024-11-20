import pytest

from app.utils.chunking import RecursiveChunker


@pytest.fixture
def chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def large_chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=512, chunk_overlap=64)


class TestRecursiveChunker:
    def test_short_text_produces_single_chunk(self, chunker: RecursiveChunker) -> None:
        text = "This is a short sentence."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_empty_text_produces_no_chunks(self, chunker: RecursiveChunker) -> None:
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_whitespace_only_produces_no_chunks(self, chunker: RecursiveChunker) -> None:
        chunks = chunker.chunk_text("   \n\n  ")
        assert len(chunks) == 0

    def test_long_text_is_split_into_multiple_chunks(self, chunker: RecursiveChunker) -> None:
        sentences = [f"Sentence number {i} contains some words for testing." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= chunker.chunk_size + chunker.chunk_overlap + 5

    def test_paragraph_splitting(self, chunker: RecursiveChunker) -> None:
        text = "First paragraph with enough text to matter.\n\nSecond paragraph also with some text.\n\nThird paragraph here too."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        full_text = " ".join(c.content for c in chunks)
        assert "First paragraph" in full_text
        assert "Third paragraph" in full_text

    def test_chunk_indices_are_sequential(self, chunker: RecursiveChunker) -> None:
        text = " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_token_count_is_positive(self, chunker: RecursiveChunker) -> None:
        text = "Every chunk should have a positive token count for tracking purposes."
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_page_number_preserved(self, chunker: RecursiveChunker) -> None:
        chunks = chunker.chunk_text("Some text on page five.", page_number=5)
        assert len(chunks) == 1
        assert chunks[0].page_number == 5

    def test_chunk_pages_preserves_boundaries(self, large_chunker: RecursiveChunker) -> None:
        pages = [
            (1, "Content on page one. " * 10),
            (2, "Content on page two. " * 10),
            (3, "Content on page three. " * 10),
        ]
        chunks = large_chunker.chunk_pages(pages)
        assert len(chunks) >= 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_pages_handles_empty_pages(self, large_chunker: RecursiveChunker) -> None:
        pages = [
            (1, "Some real content here."),
            (2, ""),
            (3, "More content on page three."),
        ]
        chunks = large_chunker.chunk_pages(pages)
        full_text = " ".join(c.content for c in chunks)
        assert "real content" in full_text
        assert "page three" in full_text

    def test_overlap_produces_shared_content(self) -> None:
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunker.chunk_text(text)
        if len(chunks) >= 2:
            last_tokens_first = chunks[0].content.split()[-3:]
            assert any(
                token in chunks[1].content for token in last_tokens_first
            ), "Overlap should share content between adjacent chunks"

    def test_count_tokens_returns_integer(self, chunker: RecursiveChunker) -> None:
        count = chunker.count_tokens("Hello world, this is a test.")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string(self, chunker: RecursiveChunker) -> None:
        count = chunker.count_tokens("")
        assert count == 0
