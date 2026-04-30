"""
src/chunkers/base.py        - Abstract base class
src/chunkers/fixed.py       - Fixed size chunking
src/chunkers/sentence.py    - Sentence-based chunking
src/chunkers/paragraph.py   - Paragraph-based chunking
src/chunkers/semantic.py    - Semantic similarity chunking
src/chunkers/hybrid.py      - Docling HybridChunker (layout-aware)

All chunkers return a unified list of dicts:
[{
    "id": int,
    "text": str,
    "title": str,
    "filename": str,
    "heading": str | None,
    "page_num": int | None,
    "chunk_method": str,
    "chunk_size": int,      # actual char length
}]
"""

# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseChunker(ABC):
    """Every chunker must implement chunk_file() and chunk_text()."""

    def _make_record(
        self,
        chunk_id: int,
        text: str,
        title: str,
        filename: str,
        heading: str | None = None,
        page_num: int | None = None,
        method: str = "base",
    ) -> Dict[str, Any]:
        return {
            "id": chunk_id,
            "text": text.strip(),
            "title": title,
            "filename": filename,
            "heading": heading,
            "page_num": page_num,
            "chunk_method": method,
            "chunk_size": len(text.strip()),
        }

    @abstractmethod
    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        """Chunk a PDF and return list of records."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FIXED SIZE
# ─────────────────────────────────────────────────────────────────────────────
import os

class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size character windows with overlap.
    Fastest, but ignores semantic boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import CharacterTextSplitter

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="",
        )

        records = []
        chunk_id = 0
        filename = os.path.basename(pdf_path)

        for page in pages:
            splits = splitter.split_text(page.page_content)
            for text in splits:
                if text.strip():
                    records.append(
                        self._make_record(
                            chunk_id=chunk_id,
                            text=text,
                            title=title,
                            filename=filename,
                            page_num=page.metadata.get("page", None),
                            method=f"fixed_{self.chunk_size}",
                        )
                    )
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE-BASED
# ─────────────────────────────────────────────────────────────────────────────

class SentenceChunker(BaseChunker):
    """
    Split on sentence boundaries, group N sentences per chunk.
    Preserves readable units.
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        import nltk
        from langchain_community.document_loaders import PyPDFLoader

        # Download punkt tokenizer if not present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        filename = os.path.basename(pdf_path)

        records = []
        chunk_id = 0

        for page in pages:
            sentences = nltk.sent_tokenize(page.page_content)
            step = max(1, self.sentences_per_chunk - self.overlap_sentences)

            for i in range(0, len(sentences), step):
                group = sentences[i: i + self.sentences_per_chunk]
                text = " ".join(group)
                if text.strip():
                    records.append(
                        self._make_record(
                            chunk_id=chunk_id,
                            text=text,
                            title=title,
                            filename=filename,
                            page_num=page.metadata.get("page", None),
                            method=f"sentence_{self.sentences_per_chunk}s",
                        )
                    )
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPH-BASED (Recursive)
# ─────────────────────────────────────────────────────────────────────────────

class ParagraphChunker(BaseChunker):
    """
    LangChain RecursiveCharacterTextSplitter — splits on \\n\\n → \\n → spaces.
    Best general-purpose chunker for prose text.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        filename = os.path.basename(pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        records = []
        chunk_id = 0

        for page in pages:
            splits = splitter.split_text(page.page_content)
            for text in splits:
                if text.strip():
                    records.append(
                        self._make_record(
                            chunk_id=chunk_id,
                            text=text,
                            title=title,
                            filename=filename,
                            page_num=page.metadata.get("page", None),
                            method=f"paragraph_{self.chunk_size}",
                        )
                    )
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups sentences by embedding cosine similarity.
    Splits when similarity drops below threshold.
    GPU-accelerated if available (run on Sol).
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
    ):
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        import nltk
        import numpy as np
        from langchain_community.document_loaders import PyPDFLoader

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        filename = os.path.basename(pdf_path)
        model = self._get_model()

        all_records = []
        chunk_id = 0

        for page in pages:
            sentences = nltk.sent_tokenize(page.page_content)
            if not sentences:
                continue

            embeddings = model.encode(sentences, show_progress_bar=False)

            # Group sentences greedily by similarity
            groups = []
            current_group = [sentences[0]]
            current_emb = embeddings[0]

            for i in range(1, len(sentences)):
                sim = self._cosine_sim(current_emb, embeddings[i])
                if sim >= self.similarity_threshold:
                    current_group.append(sentences[i])
                    # Update running mean embedding
                    current_emb = np.mean(
                        [embeddings[j] for j in range(len(current_group))], axis=0
                    )
                else:
                    groups.append(current_group)
                    current_group = [sentences[i]]
                    current_emb = embeddings[i]

            groups.append(current_group)

            for group in groups:
                text = " ".join(group)
                if len(text.strip()) >= self.min_chunk_size:
                    all_records.append(
                        self._make_record(
                            chunk_id=chunk_id,
                            text=text,
                            title=title,
                            filename=filename,
                            page_num=page.metadata.get("page", None),
                            method=f"semantic_{self.similarity_threshold}",
                        )
                    )
                    chunk_id += 1

        return all_records


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID (Docling — layout-aware)
# ─────────────────────────────────────────────────────────────────────────────

class HybridDoclingChunker(BaseChunker):
    """
    Uses Docling's HybridChunker — respects headings, tables, captions.
    Best structural fidelity for academic PDFs.
    Requires: pip install docling
    """

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        try:
            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ImportError(
                "docling not installed. Run: pip install docling"
            )

        pdf_pipeline_options = PdfPipelineOptions(
            do_ocr=False, do_table_structure=True
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options
                )
            }
        )
        chunker = HybridChunker()
        filename = os.path.basename(pdf_path)

        doc = converter.convert(pdf_path).document
        records = []
        chunk_id = 0

        for chunk in chunker.chunk(dl_doc=doc):
            d = chunk.model_dump()
            heading = (
                d["meta"]["headings"][0] if d["meta"].get("headings") else None
            )
            page_num = None
            try:
                page_num = d["meta"]["doc_items"][0]["prov"][0]["page_no"]
            except (KeyError, IndexError):
                pass

            text = chunk.text.strip()
            if text:
                records.append(
                    self._make_record(
                        chunk_id=chunk_id,
                        text=text,
                        title=title,
                        filename=filename,
                        heading=heading,
                        page_num=page_num,
                        method="hybrid_docling",
                    )
                )
                chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY — get any chunker by name
# ─────────────────────────────────────────────────────────────────────────────

def get_chunker(method: str, **kwargs) -> BaseChunker:
    """
    Factory function.
    Usage: get_chunker("fixed", chunk_size=256)
           get_chunker("semantic", similarity_threshold=0.6)
    """
    method = method.lower()
    mapping = {
        "fixed":   FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
        "hybrid":  HybridDoclingChunker,
    }
    if method not in mapping:
        raise ValueError(
            f"Unknown chunker '{method}'. Choose from: {list(mapping.keys())}"
        )
    return mapping[method](**kwargs)