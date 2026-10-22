"""Grounded answer generation for the local RAG application."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.retriever import SearchResult


GENERATION_MODEL_NAME = "google/flan-t5-base"
DEFAULT_MINIMUM_SCORE = 0.20
DEFAULT_MINIMUM_GROUNDING_COVERAGE = 0.35
DEFAULT_MINIMUM_LOW_INFORMATION_FALLBACK_SCORE = 0.35
FALLBACK_ANSWER = (
    "I could not find enough relevant information in the InsightFlow "
    "documentation. Please contact human support."
)
WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "do", "does", "for", "from", "how", "i", "if", "in", "is", "it",
    "my", "of", "on", "or", "our", "that", "the", "their", "them",
    "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "insightflow",
}


class TextGenerator(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, str]]:
        """Generate text from a prompt."""


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[str]
    refused: bool
    raw_answer: str | None = None
    decision_reason: str = "answered"
    grounding_score: float | None = None
    retrieved_evidence: list[str] = field(default_factory=list)


class LocalSeq2SeqGenerator:
    """Run a sequence-to-sequence model directly without a pipeline task alias."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, str]]:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=kwargs.get("truncation", True),
            max_length=512,
        )
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 160),
                do_sample=kwargs.get("do_sample", False),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            )
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )
        return [{"generated_text": generated_text}]


def load_generation_pipeline() -> TextGenerator:
    """Load FLAN-T5 explicitly on the CPU, downloading it on first use."""

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The transformers package is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    model.eval()
    return LocalSeq2SeqGenerator(
        tokenizer=tokenizer,
        model=model,
    )


def build_grounded_prompt(question: str, results: list[SearchResult]) -> str:
    """Augment the question with the evidence retrieved from documentation."""

    context_parts = []
    for number, result in enumerate(results, start=1):
        context_parts.append(
            f"SOURCE {number}: {result.chunk.heading} ({result.chunk.source})\n"
            f"{result.chunk.text}"
        )

    context = "\n\n".join(context_parts)
    return f"""You are the InsightFlow technical support assistant.
Answer the QUESTION using only the DOCUMENTATION below.
Do not add facts that are absent from the documentation.
If the documentation does not contain the answer, say: {FALLBACK_ANSWER}
Give a concise, helpful answer. Use numbered steps when the documentation describes a procedure.

DOCUMENTATION:
{context}

QUESTION:
{question}

ANSWER:"""


def _content_words(text: str) -> set[str]:
    """Return meaningful lowercase words used by the grounding guard."""

    return {
        word
        for word in WORD_PATTERN.findall(text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def grounding_coverage(answer: str, results: list[SearchResult]) -> float:
    """Measure how many meaningful answer words appear in retrieved evidence."""

    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0

    evidence = " ".join(
        f"{result.chunk.heading} {result.chunk.text}" for result in results
    )
    evidence_words = _content_words(evidence)
    return len(answer_words & evidence_words) / len(answer_words)


def _is_low_information(answer: str, results: list[SearchResult]) -> bool:
    """Detect headings and vague short labels, while keeping concise facts."""

    answer_words = _content_words(answer)
    normalised_answer = " ".join(WORD_PATTERN.findall(answer.lower()))
    headings = {
        " ".join(WORD_PATTERN.findall(result.chunk.heading.lower()))
        for result in results
    }
    is_heading = normalised_answer in headings
    has_compact_fact = bool(re.search(r"\d", answer)) or "," in answer
    is_vague_short_label = len(answer_words) <= 3 and not has_compact_fact
    return is_heading or is_vague_short_label


class GroundedAnswerGenerator:
    """Generate an answer from retrieved evidence and expose its sources."""

    def __init__(
        self,
        generation_pipeline: TextGenerator,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
        minimum_grounding_coverage: float = DEFAULT_MINIMUM_GROUNDING_COVERAGE,
        minimum_low_information_fallback_score: float = (
            DEFAULT_MINIMUM_LOW_INFORMATION_FALLBACK_SCORE
        ),
    ) -> None:
        self.generation_pipeline = generation_pipeline
        self.minimum_score = minimum_score
        self.minimum_grounding_coverage = minimum_grounding_coverage
        self.minimum_low_information_fallback_score = (
            minimum_low_information_fallback_score
        )

    def generate(
        self, question: str, results: list[SearchResult]
    ) -> GeneratedAnswer:
        retrieved_evidence = [
            f"{result.chunk.heading} ({result.chunk.source})" for result in results
        ]
        if not results or results[0].score < self.minimum_score:
            return GeneratedAnswer(
                FALLBACK_ANSWER,
                [],
                True,
                decision_reason="low_similarity",
                retrieved_evidence=retrieved_evidence,
            )

        prompt = build_grounded_prompt(question, results)
        generated = self.generation_pipeline(
            prompt,
            max_new_tokens=160,
            do_sample=False,
            repetition_penalty=1.1,
            truncation=True,
        )
        answer = generated[0]["generated_text"].strip()

        if not answer:
            return GeneratedAnswer(
                FALLBACK_ANSWER,
                [],
                True,
                raw_answer=answer,
                decision_reason="empty_generation",
                retrieved_evidence=retrieved_evidence,
            )

        coverage = grounding_coverage(answer, results)
        if coverage < self.minimum_grounding_coverage:
            return GeneratedAnswer(
                FALLBACK_ANSWER,
                [],
                True,
                raw_answer=answer,
                decision_reason="low_grounding_coverage",
                grounding_score=coverage,
                retrieved_evidence=retrieved_evidence,
            )

        if _is_low_information(answer, results):
            best_result = results[0]
            if best_result.score < self.minimum_low_information_fallback_score:
                return GeneratedAnswer(
                    FALLBACK_ANSWER,
                    [],
                    True,
                    raw_answer=answer,
                    decision_reason="weak_low_information_fallback",
                    grounding_score=coverage,
                    retrieved_evidence=retrieved_evidence,
                )
            return GeneratedAnswer(
                best_result.chunk.text,
                [f"{best_result.chunk.heading} ({best_result.chunk.source})"],
                False,
                raw_answer=answer,
                decision_reason="low_information_fallback",
                grounding_score=coverage,
                retrieved_evidence=retrieved_evidence,
            )

        return GeneratedAnswer(
            answer,
            retrieved_evidence,
            False,
            raw_answer=answer,
            decision_reason="answered",
            grounding_score=coverage,
            retrieved_evidence=retrieved_evidence,
        )
