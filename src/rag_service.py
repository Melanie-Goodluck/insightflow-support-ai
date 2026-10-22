"""Reusable application service for the InsightFlow RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.generator import GeneratedAnswer, GroundedAnswerGenerator
from src.semantic_retriever import SemanticRetriever


@dataclass(frozen=True)
class RAGResponse:
    """UI-agnostic result returned by the complete RAG pipeline."""

    answer: str
    sources: list[str]
    refused: bool
    decision_reason: str
    grounding_score: float | None
    retrieved_evidence: list[str]
    latency_seconds: float


class RAGService:
    """Coordinate retrieval and grounded answer generation."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: GroundedAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str) -> RAGResponse:
        """Answer one non-empty question using brand-neutral retrieval."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question must not be empty")

        started = perf_counter()
        retrieved = self.retriever.search(
            cleaned_question,
            limit=3,
            remove_product_name=True,
        )
        generated: GeneratedAnswer = self.generator.generate(
            cleaned_question,
            retrieved,
        )
        return RAGResponse(
            answer=generated.answer,
            sources=generated.sources,
            refused=generated.refused,
            decision_reason=generated.decision_reason,
            grounding_score=generated.grounding_score,
            retrieved_evidence=generated.retrieved_evidence,
            latency_seconds=perf_counter() - started,
        )
