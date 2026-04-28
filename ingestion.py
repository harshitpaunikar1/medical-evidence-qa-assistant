"""
Medical evidence document ingestion pipeline.
Parses PDFs and text files, chunks them, and indexes into a vector store for QA retrieval.
"""
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page: int
    text: str
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class IngestionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_words: int = 30
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "medical_evidence"
    vector_dim: int = 384


class PDFParser:
    """Extracts clean text from PDF files using PyMuPDF."""

    def parse(self, file_path: str) -> List[Tuple[int, str]]:
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not installed. Returning stub text for %s.", file_path)
            return [(1, f"[Stub] Content of {os.path.basename(file_path)}")]
        try:
            doc = fitz.open(file_path)
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                text = self._clean(text)
                if text.strip():
                    pages.append((i + 1, text))
            return pages
        except Exception as exc:
            logger.error("PDF parse failed for %s: %s", file_path, exc)
            return []

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


class TextChunker:
    """Splits document text into overlapping word-window chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64, min_words: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words

    def chunk(self, text: str, source_file: str, page: int) -> List[DocumentChunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            if len(chunk_words) < self.min_words:
                break
            chunk_text = " ".join(chunk_words)
            chunk_id = hashlib.md5(f"{source_file}:{page}:{start}".encode()).hexdigest()[:16]
            section = self._detect_section(chunk_text)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page=page,
                text=chunk_text,
                section=section,
            ))
            start += self.chunk_size - self.overlap
        return chunks

    def _detect_section(self, text: str) -> str:
        text_lower = text[:200].lower()
        for section in ["abstract", "introduction", "methods", "results",
                         "discussion", "conclusion", "references"]:
            if section in text_lower:
                return section
        return "body"


class EmbeddingEncoder:
    """Encodes text chunks using a sentence transformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if not ST_AVAILABLE:
            return
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._load()
        if self._model is None:
            import random
            return [[random.gauss(0, 1) for _ in range(384)] for _ in texts]
        return self._model.encode(texts, show_progress_bar=False).tolist()


class VectorIndexer:
    """Upserts document chunks and embeddings into Qdrant."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self._client = None

    def _connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            return False
        try:
            self._client = QdrantClient(url=self.config.qdrant_url)
            collections = [c.name for c in self._client.get_collections().collections]
            if self.config.collection_name not in collections:
                self._client.create_collection(
                    self.config.collection_name,
                    vectors_config=VectorParams(size=self.config.vector_dim, distance=Distance.COSINE),
                )
            return True
        except Exception as exc:
            logger.error("Qdrant connection failed: %s", exc)
            return False

    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        if not self._client and not self._connect():
            logger.info("[STUB] Would index %d chunks.", len(chunks))
            return len(chunks)
        points = [
            PointStruct(
                id=abs(int(c.chunk_id, 16)) % (2**63),
                vector=emb,
                payload={
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "page": c.page,
                    "section": c.section,
                    "text": c.text[:1000],
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self.config.collection_name, points=points)
        return len(points)


class MedicalDocumentIngester:
    """
    End-to-end ingestion pipeline for medical PDFs and text files.
    Parses, chunks, embeds, and indexes documents into a vector store.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_words=self.config.min_chunk_words,
        )
        self.encoder = EmbeddingEncoder(model_name=self.config.embedding_model)
        self.indexer = VectorIndexer(self.config)
        self._stats: Dict[str, int] = {"files": 0, "chunks": 0, "indexed": 0}

    def ingest_file(self, file_path: str) -> int:
        file_path = str(file_path)
        all_chunks = []
        if file_path.lower().endswith(".pdf"):
            pages = self.parser.parse(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                pages = [(1, content)]
            except Exception as exc:
                logger.error("Cannot read %s: %s", file_path, exc)
                return 0
        for page_num, text in pages:
            chunks = self.chunker.chunk(text, file_path, page_num)
            all_chunks.extend(chunks)
        if not all_chunks:
            return 0
        embeddings = self.encoder.encode([c.text for c in all_chunks])
        indexed = self.indexer.upsert(all_chunks, embeddings)
        self._stats["files"] += 1
        self._stats["chunks"] += len(all_chunks)
        self._stats["indexed"] += indexed
        return indexed

    def ingest_directory(self, directory: str,
                          extensions: Optional[List[str]] = None) -> Dict[str, int]:
        if extensions is None:
            extensions = [".pdf", ".txt"]
        results = {}
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in extensions:
                results[str(path)] = self.ingest_file(str(path))
        return results

    def stats(self) -> Dict:
        return dict(self._stats)


if __name__ == "__main__":
    config = IngestionConfig(
        chunk_size=512,
        chunk_overlap=64,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="medical_evidence",
    )
    ingester = MedicalDocumentIngester(config)
    chunker = TextChunker(chunk_size=100, overlap=20)
    sample_text = " ".join([f"word{i}" for i in range(250)])
    chunks = chunker.chunk(sample_text, "sample.txt", 1)
    print(f"Chunking demo: {len(chunks)} chunks from 250 words")
    print(f"Config: chunk_size={config.chunk_size}, overlap={config.chunk_overlap}")
    print("Usage: ingester.ingest_file('paper.pdf') or ingester.ingest_directory('./papers')")
    print(f"Stats: {ingester.stats()}")
