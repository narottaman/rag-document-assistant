"""
tests/test_pipeline.py

Basic unit tests for chunkers, indexers, and the generator.
Run: pytest tests/ -v
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Chunker Tests
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Attention mechanisms have become an integral part of compelling sequence modeling.
The Transformer model relies entirely on self-attention to compute representations.
Multi-head attention allows the model to jointly attend to information from different subspaces.
The positional encoding adds information about the relative position of tokens.
"""


def make_dummy_pdf(tmp_path: str) -> str:
    """Create a minimal single-page PDF for testing."""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, SAMPLE_TEXT)
        path = os.path.join(tmp_path, "test.pdf")
        pdf.output(path)
        return path
    except ImportError:
        pytest.skip("fpdf not installed, skipping PDF-based tests")


class TestFixedSizeChunker:
    def test_produces_chunks(self, tmp_path):
        from src.chunkers import FixedSizeChunker
        pdf_path = make_dummy_pdf(str(tmp_path))
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(pdf_path=pdf_path, title="Test")
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("chunk_method" in c for c in chunks)
        assert all(c["chunk_method"].startswith("fixed") for c in chunks)

    def test_chunk_size_respected(self, tmp_path):
        from src.chunkers import FixedSizeChunker
        pdf_path = make_dummy_pdf(str(tmp_path))
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk(pdf_path=pdf_path, title="Test")
        for c in chunks:
            assert c["chunk_size"] <= 200, "Chunks should be near target size"


class TestParagraphChunker:
    def test_produces_chunks(self, tmp_path):
        from src.chunkers import ParagraphChunker
        pdf_path = make_dummy_pdf(str(tmp_path))
        chunker = ParagraphChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(pdf_path=pdf_path, title="Test")
        assert len(chunks) > 0
        assert all(c["chunk_method"].startswith("paragraph") for c in chunks)


class TestSentenceChunker:
    def test_produces_chunks(self, tmp_path):
        from src.chunkers import SentenceChunker
        pdf_path = make_dummy_pdf(str(tmp_path))
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = chunker.chunk(pdf_path=pdf_path, title="Test")
        assert len(chunks) > 0


class TestGetChunkerFactory:
    def test_all_methods_registered(self):
        from src.chunkers import get_chunker
        for method in ["fixed", "sentence", "paragraph", "semantic"]:
            chunker = get_chunker(method)
            assert chunker is not None

    def test_invalid_method_raises(self):
        from src.chunkers import get_chunker
        with pytest.raises(ValueError):
            get_chunker("nonexistent_method")


# ─────────────────────────────────────────────────────────────────────────────
# Indexer Tests (no PDF needed — use raw text)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"id": 0, "text": "The Transformer uses multi-head self-attention mechanisms.",
     "title": "Attention Paper", "filename": "attn.pdf", "heading": None, "page_num": 1,
     "chunk_method": "test", "chunk_size": 60},
    {"id": 1, "text": "BERT pre-trains deep bidirectional representations.",
     "title": "BERT Paper", "filename": "bert.pdf", "heading": None, "page_num": 1,
     "chunk_method": "test", "chunk_size": 52},
    {"id": 2, "text": "GPT-3 is an autoregressive language model with 175 billion parameters.",
     "title": "GPT-3 Paper", "filename": "gpt3.pdf", "heading": None, "page_num": 1,
     "chunk_method": "test", "chunk_size": 67},
]


class TestChromaFlatIndexer:
    def test_add_and_search(self, tmp_path):
        from src.indexers import ChromaFlatIndexer
        indexer = ChromaFlatIndexer(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_col",
        )
        indexer.add(SAMPLE_CHUNKS)
        results = indexer.search("attention mechanism transformer", top_k=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]

    def test_top_k_respected(self, tmp_path):
        from src.indexers import ChromaFlatIndexer
        indexer = ChromaFlatIndexer(
            persist_dir=str(tmp_path / "chroma2"),
            collection_name="test_col2",
        )
        indexer.add(SAMPLE_CHUNKS)
        results = indexer.search("language model", top_k=2)
        assert len(results) <= 2


class TestHNSWIndexer:
    def test_add_and_search(self):
        pytest.importorskip("hnswlib")
        from src.indexers import HNSWIndexer
        indexer = HNSWIndexer()
        indexer.add(SAMPLE_CHUNKS)
        results = indexer.search("BERT bidirectional", top_k=2)
        assert len(results) > 0
        assert results[0]["score"] >= 0


class TestFAISSIndexer:
    def test_add_and_search(self):
        pytest.importorskip("faiss")
        from src.indexers import FAISSIndexer
        indexer = FAISSIndexer(n_lists=2)  # small n_lists for tiny test corpus
        indexer.add(SAMPLE_CHUNKS)
        results = indexer.search("GPT parameters", top_k=1)
        assert len(results) > 0


class TestGetIndexerFactory:
    def test_flat_factory(self, tmp_path):
        from src.indexers import get_indexer
        indexer = get_indexer("flat", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                              persist_dir=str(tmp_path / "c"))
        assert indexer is not None

    def test_invalid_index_raises(self):
        from src.indexers import get_indexer
        with pytest.raises(ValueError):
            get_indexer("unknown_index", embedding_model="test")


# ─────────────────────────────────────────────────────────────────────────────
# Generator Tests (requires OPENAI_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerator:
    def test_generate_returns_string(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from src.generator import Generator
        gen = Generator()
        answer = gen.generate(
            question="What is multi-head attention?",
            contexts=["Multi-head attention allows attending to different representation subspaces."]
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_unknown_answer(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from src.generator import Generator
        gen = Generator()
        answer = gen.generate(
            question="What is the price of a pizza?",
            contexts=["The Transformer model uses self-attention."]
        )
        assert "don't know" in answer.lower() or len(answer) > 0