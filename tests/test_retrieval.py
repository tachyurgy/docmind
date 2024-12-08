import uuid
from unittest.mock import MagicMock

import pytest

from app.models.document import Chunk
from app.services.retrieval import RetrievalService


def _make_chunk(content: str, page: int = 1) -> Chunk:
    chunk = MagicMock(spec=Chunk)
    chunk.id = uuid.uuid4()
    chunk.content = content
    chunk.page_number = page
    chunk.chunk_index = 0
    chunk.token_count = len(content.split())
    chunk.document_id = uuid.uuid4()
    return chunk


class TestMergeResults:
    def test_vector_only_results(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())
        chunk = _make_chunk("Vector result")

        vector_results = [(chunk, "doc.pdf", 0.95)]
        keyword_results = []

        merged = service._merge_results(vector_results, keyword_results, alpha=0.7, top_k=5)

        assert len(merged) == 1
        assert merged[0].vector_score == 0.95
        assert merged[0].keyword_score == 0.0
        assert merged[0].combined_score == pytest.approx(0.7 * 0.95)

    def test_keyword_only_results(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())
        chunk = _make_chunk("Keyword result")

        vector_results = []
        keyword_results = [(chunk, "doc.pdf", 0.8)]

        merged = service._merge_results(vector_results, keyword_results, alpha=0.7, top_k=5)

        assert len(merged) == 1
        assert merged[0].vector_score == 0.0
        assert merged[0].keyword_score == 0.8
        assert merged[0].combined_score == pytest.approx(0.3 * 0.8)

    def test_hybrid_merge_combines_scores(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())
        chunk = _make_chunk("Hybrid result")
        chunk_id = chunk.id

        vector_results = [(chunk, "doc.pdf", 0.9)]

        chunk_kw = MagicMock(spec=Chunk)
        chunk_kw.id = chunk_id
        chunk_kw.content = "Hybrid result"
        keyword_results = [(chunk_kw, "doc.pdf", 0.6)]

        merged = service._merge_results(vector_results, keyword_results, alpha=0.7, top_k=5)

        assert len(merged) == 1
        expected = 0.7 * 0.9 + 0.3 * 0.6
        assert merged[0].combined_score == pytest.approx(expected)

    def test_top_k_limits_results(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())

        vector_results = []
        for i in range(10):
            chunk = _make_chunk(f"Result {i}")
            vector_results.append((chunk, "doc.pdf", 0.9 - i * 0.05))

        merged = service._merge_results(vector_results, [], alpha=0.7, top_k=3)
        assert len(merged) == 3

    def test_results_sorted_by_combined_score(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())

        chunk_low = _make_chunk("Low score")
        chunk_high = _make_chunk("High score")

        vector_results = [
            (chunk_low, "doc.pdf", 0.3),
            (chunk_high, "doc.pdf", 0.9),
        ]

        merged = service._merge_results(vector_results, [], alpha=0.7, top_k=5)

        assert merged[0].combined_score > merged[1].combined_score

    def test_alpha_weighting(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())

        chunk_vec = _make_chunk("Good vector match")
        chunk_kw = _make_chunk("Good keyword match")

        vector_results = [(chunk_vec, "doc.pdf", 0.9)]
        keyword_results = [(chunk_kw, "doc.pdf", 0.9)]

        merged_high_alpha = service._merge_results(
            vector_results, keyword_results, alpha=0.9, top_k=5
        )
        merged_low_alpha = service._merge_results(
            vector_results, keyword_results, alpha=0.1, top_k=5
        )

        vec_result_high = next(r for r in merged_high_alpha if r.chunk.id == chunk_vec.id)
        kw_result_high = next(r for r in merged_high_alpha if r.chunk.id == chunk_kw.id)
        assert vec_result_high.combined_score > kw_result_high.combined_score

        vec_result_low = next(r for r in merged_low_alpha if r.chunk.id == chunk_vec.id)
        kw_result_low = next(r for r in merged_low_alpha if r.chunk.id == chunk_kw.id)
        assert kw_result_low.combined_score > vec_result_low.combined_score

    def test_empty_results(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())
        merged = service._merge_results([], [], alpha=0.7, top_k=5)
        assert len(merged) == 0

    def test_document_name_preserved(self) -> None:
        service = RetrievalService(embedding_service=MagicMock())
        chunk = _make_chunk("Content")

        vector_results = [(chunk, "my_report.pdf", 0.85)]
        merged = service._merge_results(vector_results, [], alpha=0.7, top_k=5)

        assert merged[0].document_name == "my_report.pdf"
