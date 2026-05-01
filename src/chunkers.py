"""
src/chunkers.py

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

KEY DESIGN DECISIONS:
- Uses pypdf directly (NOT LangChain PyPDFLoader) to avoid torchvision
  circular import on Sol HPC where system torch+torchvision are mismatched.
- sentence-transformers is imported lazily (only in SemanticChunker)
  so fixed/paragraph/sentence chunkers never touch torch.
- nltk punkt is downloaded via SSL — Sol login nodes block SSL outbound.
  We use a regex fallback sentence splitter that requires zero downloads.
  If nltk punkt IS available locally it uses that; otherwise falls back gracefully.

OVERLAP PER METHOD:
  fixed     → chunk_overlap chars (default 50)
  paragraph → chunk_overlap chars (default 50), recursive separators
  sentence  → overlap_sentences sentences (default 1)
  semantic  → NO overlap — splits on cosine similarity drop
  hybrid    → NO overlap — Docling respects document layout boundaries
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import re


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION (torch-free)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text per page using pypdf directly.
    Returns: [{"page_num": int, "text": str}, ...]
    Never imports torch, transformers, or LangChain.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE TOKENIZER — SSL-safe, works on Sol without any nltk downloads
# ─────────────────────────────────────────────────────────────────────────────

def sent_tokenize_safe(text: str) -> List[str]:
    """
    SSL-safe sentence tokenizer for Sol HPC.

    Strategy:
      1. Check if nltk punkt data EXISTS LOCALLY — if yes, use it (best quality)
      2. Never call nltk.download() — that requires SSL which Sol blocks
      3. Fall back to regex splitter — handles academic PDFs well

    The regex fallback:
      - Protects abbreviations: Fig., et al., vs., Dr., Prof., e.g., i.e., etc.
      - Protects decimal numbers: 3.14, 0.001
      - Splits on '. '/'! '/'? ' followed by capital letter or quote
    """
    # ── Try nltk only if data already exists locally ─────────────────────────
    try:
        import nltk
        found = False
        for resource in ["tokenizers/punkt_tab", "tokenizers/punkt"]:
            try:
                nltk.data.find(resource)
                found = True
                break
            except LookupError:
                pass
        if found:
            return nltk.sent_tokenize(text)
        # Data not found locally — skip to regex, no download attempt
    except Exception:
        pass

    # ── Regex fallback ────────────────────────────────────────────────────────
    # Step 1: protect abbreviation dots with a safe placeholder
    placeholder = "\x00DOT\x00"
    abbreviations = [
        r"Fig\.", r"et al\.", r"vs\.", r"Dr\.", r"Prof\.",
        r"Mr\.", r"Mrs\.", r"Ms\.", r"St\.", r"No\.", r"Eq\.",
        r"Sec\.", r"Tab\.", r"App\.", r"approx\.",
        r"e\.g\.", r"i\.e\.", r"cf\.", r"al\.", r"arXiv\.",
        r"pp\.", r"vol\.", r"dept\.",
    ]
    protected = text
    for abbr in abbreviations:
        protected = re.sub(abbr, lambda m: m.group().replace(".", placeholder), protected)

    # Step 2: protect decimal numbers like 3.14 or 1e-5
    protected = re.sub(r"(\d)\.(\d)", r"\1" + placeholder + r"\2", protected)

    # Step 3: split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])", protected)

    # Step 4: restore placeholders and filter
    sentences = []
    for part in parts:
        restored = part.replace(placeholder, ".").strip()
        if len(restored) > 10:
            sentences.append(restored)

    return sentences if sentences else [text.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────

class BaseChunker(ABC):

    def _make_record(
        self,
        chunk_id: int,
        text: str,
        title: str,
        filename: str,
        heading: Optional[str] = None,
        page_num: Optional[int] = None,
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
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FIXED SIZE CHUNKER
# overlap: chunk_overlap characters (default 50)
# ─────────────────────────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed character windows with overlap.
    Fastest method. Ignores semantic boundaries.
    Overlap: chunk_overlap chars between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        pages = extract_pages_from_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        records = []
        chunk_id = 0

        for page in pages:
            for text in self._split_text(page["text"]):
                if len(text.strip()) > 20:
                    records.append(self._make_record(
                        chunk_id=chunk_id,
                        text=text,
                        title=title,
                        filename=filename,
                        page_num=page["page_num"],
                        method=f"fixed_{self.chunk_size}",
                    ))
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE CHUNKER
# overlap: overlap_sentences sentences (default 1)
# ─────────────────────────────────────────────────────────────────────────────

class SentenceChunker(BaseChunker):
    """
    Split on sentence boundaries using sent_tokenize_safe().
    Works on Sol HPC even when SSL is blocked — no nltk download needed.
    Overlap: overlap_sentences sentences shared between consecutive chunks.
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        pages = extract_pages_from_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        records = []
        chunk_id = 0

        for page in pages:
            sentences = sent_tokenize_safe(page["text"])   # ← SSL-safe
            if not sentences:
                continue

            step = max(1, self.sentences_per_chunk - self.overlap_sentences)

            for i in range(0, len(sentences), step):
                group = sentences[i: i + self.sentences_per_chunk]
                text = " ".join(group).strip()
                if len(text) > 20:
                    records.append(self._make_record(
                        chunk_id=chunk_id,
                        text=text,
                        title=title,
                        filename=filename,
                        page_num=page["page_num"],
                        method=f"sentence_{self.sentences_per_chunk}s_ov{self.overlap_sentences}",
                    ))
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPH CHUNKER (Recursive)
# overlap: chunk_overlap characters (default 50)
# ─────────────────────────────────────────────────────────────────────────────

class ParagraphChunker(BaseChunker):
    """
    Recursive splitting: tries \\n\\n → \\n → '. ' → ' ' → '' in order.
    Best general-purpose chunker for prose/academic text.
    Overlap: chunk_overlap chars shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        splits = text.split(sep) if sep else list(text)
        chunks = []
        current = ""

        for split in splits:
            piece = (current + sep + split) if current else split
            if len(piece) <= self.chunk_size:
                current = piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                if self.chunk_overlap > 0 and current:
                    overlap_text = current[-self.chunk_overlap:]
                    current = overlap_text + sep + split
                else:
                    current = split

                if len(current) > self.chunk_size and remaining_seps:
                    sub = self._recursive_split(current, remaining_seps)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c.strip()) > 20]

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        pages = extract_pages_from_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        records = []
        chunk_id = 0

        for page in pages:
            splits = self._recursive_split(page["text"], self.separators)
            for text in splits:
                records.append(self._make_record(
                    chunk_id=chunk_id,
                    text=text,
                    title=title,
                    filename=filename,
                    page_num=page["page_num"],
                    method=f"paragraph_{self.chunk_size}_ov{self.chunk_overlap}",
                ))
                chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC CHUNKER
# overlap: NONE — splits on cosine similarity drop between sentences
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups sentences by embedding cosine similarity.
    Splits when similarity between adjacent sentences drops below threshold.
    NO character overlap — the split boundary IS the semantic break.
    Uses sent_tokenize_safe() — works on Sol without SSL access.
    GPU-accelerated automatically when running on Sol with CUDA available.
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
        import numpy as np

        pages = extract_pages_from_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        model = self._get_model()
        records = []
        chunk_id = 0

        for page in pages:
            sentences = sent_tokenize_safe(page["text"])   # ← SSL-safe
            if len(sentences) < 2:
                continue

            embeddings = model.encode(sentences, show_progress_bar=False)
            groups: List[List[str]] = []
            current_group = [sentences[0]]
            current_emb = embeddings[0].copy()

            for i in range(1, len(sentences)):
                sim = self._cosine_sim(current_emb, embeddings[i])
                if sim >= self.similarity_threshold:
                    current_group.append(sentences[i])
                    current_emb = np.mean(
                        embeddings[max(0, i - len(current_group) + 1): i + 1],
                        axis=0,
                    )
                else:
                    groups.append(current_group)
                    current_group = [sentences[i]]
                    current_emb = embeddings[i].copy()

            groups.append(current_group)

            for group in groups:
                text = " ".join(group).strip()
                if len(text) >= self.min_chunk_size:
                    records.append(self._make_record(
                        chunk_id=chunk_id,
                        text=text,
                        title=title,
                        filename=filename,
                        page_num=page["page_num"],
                        method=f"semantic_t{self.similarity_threshold}",
                    ))
                    chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID DOCLING CHUNKER
# overlap: NONE — respects document layout (headings, tables, captions)
# ─────────────────────────────────────────────────────────────────────────────

class HybridDoclingChunker(BaseChunker):
    """
    Docling HybridChunker — layout-aware PDF parsing.
    Respects headings, tables, figure captions, and section boundaries.
    NO overlap — structural boundaries are the natural split points.
    Requires: pip install docling
    """

    def chunk(self, pdf_path: str, title: str) -> List[Dict[str, Any]]:
        try:
            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ImportError("docling not installed. Run: pip install docling")

        pipeline_opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )
        chunker = HybridChunker()
        filename = os.path.basename(pdf_path)

        doc = converter.convert(pdf_path).document
        records = []
        chunk_id = 0

        for chunk in chunker.chunk(dl_doc=doc):
            d = chunk.model_dump()
            heading = d["meta"]["headings"][0] if d["meta"].get("headings") else None
            page_num = None
            try:
                page_num = d["meta"]["doc_items"][0]["prov"][0]["page_no"]
            except (KeyError, IndexError):
                pass

            text = chunk.text.strip()
            if text:
                records.append(self._make_record(
                    chunk_id=chunk_id,
                    text=text,
                    title=title,
                    filename=filename,
                    heading=heading,
                    page_num=page_num,
                    method="hybrid_docling",
                ))
                chunk_id += 1

        return records


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

CHUNKER_REGISTRY = {
    "fixed":     FixedSizeChunker,
    "sentence":  SentenceChunker,
    "paragraph": ParagraphChunker,
    "semantic":  SemanticChunker,
    "hybrid":    HybridDoclingChunker,
}


def get_chunker(method: str, **kwargs) -> BaseChunker:
    """
    Factory function.

    Usage:
        get_chunker("fixed", chunk_size=256, chunk_overlap=25)
        get_chunker("sentence", sentences_per_chunk=5, overlap_sentences=1)
        get_chunker("paragraph", chunk_size=512, chunk_overlap=50)
        get_chunker("semantic", similarity_threshold=0.5)
        get_chunker("hybrid")
    """
    method = method.lower()
    if method not in CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunker '{method}'. Choose from: {list(CHUNKER_REGISTRY.keys())}"
        )
    return CHUNKER_REGISTRY[method](**kwargs)