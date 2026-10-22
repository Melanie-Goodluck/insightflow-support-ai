"""Evaluation helpers for the complete retrieval-and-generation pipeline."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.generator import GeneratedAnswer


@dataclass(frozen=True)
class RAGEvaluationItem:
    question_id: str
    question: str
    answerable: bool
    expected_heading: str | None
    reference_answer: str


@dataclass(frozen=True)
class RAGEvaluationResult:
    item: RAGEvaluationItem
    answer: str
    refused: bool
    sources: list[str]
    top_similarity_score: float | None
    latency_seconds: float
    raw_answer: str | None = None
    decision_reason: str = "unknown"
    grounding_score: float | None = None
    candidate_evidence: list[str] = field(default_factory=list)

    @property
    def routing_correct(self) -> bool:
        return self.refused != self.item.answerable

    @property
    def expected_source_retrieved(self) -> bool | None:
        if not self.item.answerable or not self.item.expected_heading:
            return None
        expected = self.item.expected_heading.lower()
        return any(expected in source.lower() for source in self.candidate_evidence)


@dataclass(frozen=True)
class RAGMetrics:
    total_questions: int
    answerable_questions: int
    unanswerable_questions: int
    routing_accuracy: float
    answerable_answer_rate: float
    unanswerable_refusal_accuracy: float
    false_refusal_rate: float
    unsupported_answer_rate: float
    expected_source_retrieval_rate: float
    mean_latency_seconds: float
    median_latency_seconds: float


EvaluationRunner = Callable[[str], tuple[GeneratedAnswer, float | None]]


def load_rag_evaluation_items(path: Path) -> list[RAGEvaluationItem]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("RAG evaluation data must contain at least one question")

    items = []
    for record in records:
        answerable = bool(record["answerable"])
        expected_heading = record.get("expected_heading")
        if answerable and not expected_heading:
            raise ValueError("Answerable questions require an expected_heading")
        if not answerable and expected_heading is not None:
            raise ValueError("Unanswerable questions must not have an expected_heading")
        items.append(
            RAGEvaluationItem(
                question_id=record["id"],
                question=record["question"],
                answerable=answerable,
                expected_heading=expected_heading,
                reference_answer=record["reference_answer"],
            )
        )
    return items


def run_rag_evaluation(
    items: list[RAGEvaluationItem], runner: EvaluationRunner
) -> list[RAGEvaluationResult]:
    if not items:
        raise ValueError("At least one RAG evaluation item is required")

    results = []
    for item in items:
        started = time.perf_counter()
        response, top_score = runner(item.question)
        elapsed = time.perf_counter() - started
        results.append(
            RAGEvaluationResult(
                item=item,
                answer=response.answer,
                refused=response.refused,
                sources=response.sources,
                top_similarity_score=top_score,
                latency_seconds=elapsed,
                raw_answer=response.raw_answer,
                decision_reason=response.decision_reason,
                grounding_score=response.grounding_score,
                candidate_evidence=response.retrieved_evidence,
            )
        )
    return results


def calculate_rag_metrics(results: list[RAGEvaluationResult]) -> RAGMetrics:
    if not results:
        raise ValueError("At least one RAG evaluation result is required")

    answerable = [result for result in results if result.item.answerable]
    unanswerable = [result for result in results if not result.item.answerable]
    if not answerable or not unanswerable:
        raise ValueError("Evaluation requires answerable and unanswerable questions")

    routing_correct = sum(result.routing_correct for result in results)
    answerable_answered = sum(not result.refused for result in answerable)
    unanswerable_refused = sum(result.refused for result in unanswerable)
    source_hits = sum(result.expected_source_retrieved is True for result in answerable)
    latencies = [result.latency_seconds for result in results]

    return RAGMetrics(
        total_questions=len(results),
        answerable_questions=len(answerable),
        unanswerable_questions=len(unanswerable),
        routing_accuracy=routing_correct / len(results),
        answerable_answer_rate=answerable_answered / len(answerable),
        unanswerable_refusal_accuracy=unanswerable_refused / len(unanswerable),
        false_refusal_rate=(len(answerable) - answerable_answered) / len(answerable),
        unsupported_answer_rate=(len(unanswerable) - unanswerable_refused)
        / len(unanswerable),
        expected_source_retrieval_rate=source_hits / len(answerable),
        mean_latency_seconds=statistics.fmean(latencies),
        median_latency_seconds=statistics.median(latencies),
    )


def write_rag_results_csv(
    results: list[RAGEvaluationResult], destination: Path
) -> None:
    fieldnames = [
        "id", "question", "expected_answerable", "expected_heading",
        "reference_answer", "system_answer", "refused", "routing_correct",
        "expected_source_retrieved", "top_similarity_score", "latency_seconds",
        "decision_reason", "raw_model_answer", "grounding_score",
        "answer_sources", "candidate_evidence", "correctness_0_to_2", "faithfulness_0_to_2",
        "helpfulness_0_to_2", "review_notes",
    ]

    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result.item.question_id,
                    "question": result.item.question,
                    "expected_answerable": result.item.answerable,
                    "expected_heading": result.item.expected_heading or "",
                    "reference_answer": result.item.reference_answer,
                    "system_answer": result.answer,
                    "refused": result.refused,
                    "routing_correct": result.routing_correct,
                    "expected_source_retrieved": (
                        "" if result.expected_source_retrieved is None
                        else result.expected_source_retrieved
                    ),
                    "top_similarity_score": (
                        "" if result.top_similarity_score is None
                        else f"{result.top_similarity_score:.3f}"
                    ),
                    "latency_seconds": f"{result.latency_seconds:.3f}",
                    "decision_reason": result.decision_reason,
                    "raw_model_answer": result.raw_answer or "",
                    "grounding_score": (
                        "" if result.grounding_score is None
                        else f"{result.grounding_score:.3f}"
                    ),
                    "answer_sources": " | ".join(result.sources),
                    "candidate_evidence": " | ".join(result.candidate_evidence),
                    "correctness_0_to_2": "",
                    "faithfulness_0_to_2": "",
                    "helpfulness_0_to_2": "",
                    "review_notes": "",
                }
            )
