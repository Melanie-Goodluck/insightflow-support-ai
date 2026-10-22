"""Reusable metrics for evaluating ranked document-retrieval results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.retriever import SearchResult


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_heading: str
    style: str


@dataclass(frozen=True)
class QuestionResult:
    question: EvaluationQuestion
    rank: int | None
    retrieved_headings: list[str]


@dataclass(frozen=True)
class EvaluationMetrics:
    top_1_accuracy: float
    top_3_accuracy: float
    mean_reciprocal_rank: float


def find_expected_rank(results: list[SearchResult], expected_heading: str) -> int | None:
    """Return the expected section's 1-based rank, or None when it was absent."""

    for rank, result in enumerate(results, start=1):
        if result.chunk.heading == expected_heading:
            return rank
    return None


def calculate_metrics(question_results: list[QuestionResult]) -> EvaluationMetrics:
    """Calculate Top-1, Top-3 and Mean Reciprocal Rank."""

    if not question_results:
        raise ValueError("At least one evaluation result is required")

    total = len(question_results)
    top_1 = sum(result.rank == 1 for result in question_results) / total
    top_3 = sum(result.rank is not None and result.rank <= 3 for result in question_results) / total
    mrr = sum(1 / result.rank if result.rank else 0 for result in question_results) / total
    return EvaluationMetrics(top_1, top_3, mrr)


def evaluate_retriever(
    questions: list[EvaluationQuestion],
    search_function: Callable[[str, int], list[SearchResult]],
    limit: int = 3,
) -> tuple[list[QuestionResult], EvaluationMetrics]:
    """Run labelled questions through one retriever and score its rankings."""

    results = []
    for question in questions:
        retrieved = search_function(question.question, limit)
        results.append(
            QuestionResult(
                question=question,
                rank=find_expected_rank(retrieved, question.expected_heading),
                retrieved_headings=[item.chunk.heading for item in retrieved],
            )
        )
    return results, calculate_metrics(results)

