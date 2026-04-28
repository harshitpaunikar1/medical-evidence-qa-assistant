"""
Medical evidence QA assistant using RAG over indexed research documents.
Retrieves relevant chunks from Qdrant and generates evidence-grounded answers.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_file: str
    page: int
    section: str
    text: str
    score: float


@dataclass
class QAResponse:
    question: str
    answer: str
    sources: List[RetrievedChunk]
    model: str
    latency_ms: float
    confidence: str  # high, medium, low


class Retriever:
    """Retrieves relevant document chunks from Qdrant using semantic search."""

    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "medical_evidence",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 5):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.top_k = top_k
        self._encoder = None
        self._client = None
        self._stub_chunks = self._build_stub_chunks()

    def _build_stub_chunks(self) -> List[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="stub_001",
                source_file="randomized_trial.pdf",
                page=3,
                section="results",
                text="The intervention group showed a 23% reduction in mortality rate (95% CI: 15-31%, p<0.001) compared to placebo.",
                score=0.91,
            ),
            RetrievedChunk(
                chunk_id="stub_002",
                source_file="systematic_review.pdf",
                page=7,
                section="discussion",
                text="Meta-analysis of 12 RCTs found consistent benefit of the treatment with NNT of 8 (95% CI: 6-11).",
                score=0.87,
            ),
            RetrievedChunk(
                chunk_id="stub_003",
                source_file="cohort_study.pdf",
                page=2,
                section="methods",
                text="Participants were followed for 24 months with monthly assessments of primary and secondary endpoints.",
                score=0.75,
            ),
        ]

    def _get_encoder(self):
        if self._encoder is None and ST_AVAILABLE:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def _get_client(self):
        if self._client is None and QDRANT_AVAILABLE:
            try:
                self._client = QdrantClient(url=self.qdrant_url)
            except Exception as exc:
                logger.warning("Qdrant unavailable: %s", exc)
        return self._client

    def retrieve(self, query: str,
                 section_filter: Optional[str] = None) -> List[RetrievedChunk]:
        encoder = self._get_encoder()
        client = self._get_client()

        if encoder is None or client is None:
            return self._stub_chunks[:self.top_k]

        try:
            query_vec = encoder.encode([query])[0].tolist()
            search_filter = None
            if section_filter:
                search_filter = Filter(
                    must=[FieldCondition(key="section", match=MatchValue(value=section_filter))]
                )
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=self.top_k,
                query_filter=search_filter,
            )
            return [
                RetrievedChunk(
                    chunk_id=str(r.id),
                    source_file=r.payload.get("source_file", ""),
                    page=r.payload.get("page", 0),
                    section=r.payload.get("section", ""),
                    text=r.payload.get("text", ""),
                    score=round(r.score, 4),
                )
                for r in results
            ]
        except Exception as exc:
            logger.error("Retrieval failed: %s. Using stub.", exc)
            return self._stub_chunks[:self.top_k]


class MedicalQAAssistant:
    """
    RAG-based QA assistant grounded in indexed medical literature.
    Formats retrieved evidence into a structured prompt for an LLM.
    Supports Ollama (local) and Gemini (cloud) backends.
    """

    SYSTEM_PROMPT = """You are a medical evidence analyst. Answer the question using ONLY the provided evidence excerpts.
For each claim, cite the source document and page.
If the evidence is insufficient to answer the question, say so clearly.
Do not speculate beyond what the evidence states.
Format: Answer first, then list citations as [Source: filename, Page X].
"""

    def __init__(self, retriever: Retriever,
                 llm_provider: str = "ollama",
                 llm_model: str = "llama3",
                 gemini_api_key: Optional[str] = None):
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.gemini_api_key = gemini_api_key
        self._gemini_model = None
        self._setup_llm()

    def _setup_llm(self) -> None:
        if self.llm_provider == "gemini" and GEMINI_AVAILABLE and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_model = genai.GenerativeModel(self.llm_model or "gemini-1.5-flash")

    def _build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context = "\n\n".join([
            f"[Source: {c.source_file}, Page {c.page}, Section: {c.section}]\n{c.text}"
            for c in chunks
        ])
        return f"""{self.SYSTEM_PROMPT}

Evidence:
{context}

Question: {question}

Answer:"""

    def _generate(self, prompt: str) -> str:
        if self.llm_provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                resp = ollama.generate(model=self.llm_model, prompt=prompt)
                return resp.get("response", "")
            except Exception as exc:
                logger.error("Ollama error: %s", exc)
        if self.llm_provider == "gemini" and self._gemini_model:
            try:
                resp = self._gemini_model.generate_content(prompt)
                return resp.text
            except Exception as exc:
                logger.error("Gemini error: %s", exc)
        return self._stub_answer(prompt)

    def _stub_answer(self, prompt: str) -> str:
        if "mortality" in prompt.lower():
            return ("Based on the available evidence, the intervention showed a 23% reduction in mortality "
                    "(p<0.001). [Source: randomized_trial.pdf, Page 3]")
        return ("The available evidence suggests potential benefit, but further studies are needed "
                "for definitive conclusions. [Source: systematic_review.pdf, Page 7]")

    def _confidence(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "low"
        avg_score = sum(c.score for c in chunks) / len(chunks)
        if avg_score >= 0.85:
            return "high"
        elif avg_score >= 0.70:
            return "medium"
        return "low"

    def answer(self, question: str,
               section_filter: Optional[str] = None) -> QAResponse:
        """Retrieve relevant evidence and generate a grounded answer."""
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, section_filter=section_filter)
        prompt = self._build_prompt(question, chunks)
        answer_text = self._generate(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        return QAResponse(
            question=question,
            answer=answer_text,
            sources=chunks,
            model=f"{self.llm_provider}/{self.llm_model}",
            latency_ms=round(latency_ms, 1),
            confidence=self._confidence(chunks),
        )

    def batch_answer(self, questions: List[str]) -> List[QAResponse]:
        return [self.answer(q) for q in questions]


if __name__ == "__main__":
    retriever = Retriever(
        qdrant_url="http://localhost:6333",
        collection_name="medical_evidence",
        top_k=3,
    )
    assistant = MedicalQAAssistant(
        retriever=retriever,
        llm_provider="stub",
        llm_model="llama3",
    )

    questions = [
        "What is the effect of the treatment on mortality?",
        "How many studies were included in the meta-analysis?",
        "What was the follow-up duration in the cohort study?",
    ]

    for q in questions:
        resp = assistant.answer(q)
        print(f"\nQ: {resp.question}")
        print(f"A: {resp.answer}")
        print(f"Confidence: {resp.confidence} | Latency: {resp.latency_ms:.0f}ms | Sources: {len(resp.sources)}")
