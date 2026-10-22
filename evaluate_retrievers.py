"""Evaluate lexical and semantic retrieval on the same labelled questions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation import EvaluationQuestion, evaluate_retriever
from src.hybrid_retriever import HybridRetriever
from src.retriever import load_chunks, search
from src.semantic_retriever import load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"
QUESTIONS_FILE = PROJECT_DIRECTORY / "data" / "evaluation_questions.json"
RESULTS_FILE = PROJECT_DIRECTORY / "evaluation_results.csv"


def load_questions() -> list[EvaluationQuestion]:
    records = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return [EvaluationQuestion(**record) for record in records]


def print_metrics(name: str, metrics) -> None:
    print(f"\n{name}")
    print(f"  Top-1 accuracy:       {metrics.top_1_accuracy:.1%}")
    print(f"  Top-3 accuracy:       {metrics.top_3_accuracy:.1%}")
    print(f"  Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.3f}")


def save_results(
    questions, lexical_results, semantic_results,
    brand_neutral_results, hybrid_results,
) -> None:
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "style",
                "expected_heading",
                "lexical_rank",
                "semantic_rank",
                "brand_neutral_semantic_rank",
                "hybrid_rank",
                "lexical_top_3",
                "semantic_top_3",
                "brand_neutral_semantic_top_3",
                "hybrid_top_3",
            ],
        )
        writer.writeheader()
        for question, lexical, semantic, brand_neutral, hybrid in zip(
            questions, lexical_results, semantic_results,
            brand_neutral_results, hybrid_results
        ):
            writer.writerow(
                {
                    "question": question.question,
                    "style": question.style,
                    "expected_heading": question.expected_heading,
                    "lexical_rank": lexical.rank or "not found",
                    "semantic_rank": semantic.rank or "not found",
                    "brand_neutral_semantic_rank": brand_neutral.rank or "not found",
                    "hybrid_rank": hybrid.rank or "not found",
                    "lexical_top_3": " | ".join(lexical.retrieved_headings),
                    "semantic_top_3": " | ".join(semantic.retrieved_headings),
                    "brand_neutral_semantic_top_3": " | ".join(
                        brand_neutral.retrieved_headings
                    ),
                    "hybrid_top_3": " | ".join(hybrid.retrieved_headings),
                }
            )


def main() -> None:
    questions = load_questions()
    chunks = load_chunks(DOCUMENTS_DIRECTORY)

    lexical_results, lexical_metrics = evaluate_retriever(
        questions,
        lambda question, limit: search(question, chunks, limit),
    )

    print("Loading the local embedding model...")
    hybrid_retriever = HybridRetriever(chunks, load_embedding_model())
    semantic_results, semantic_metrics = evaluate_retriever(
        questions,
        lambda question, limit: hybrid_retriever.semantic.search(question, limit),
    )
    brand_neutral_results, brand_neutral_metrics = evaluate_retriever(
        questions,
        lambda question, limit: hybrid_retriever.semantic.search(
            question, limit, remove_product_name=True
        ),
    )
    hybrid_results, hybrid_metrics = evaluate_retriever(
        questions,
        lambda question, limit: hybrid_retriever.search(question, limit),
    )

    print(f"Evaluated {len(questions)} labelled questions.")
    print_metrics("Lexical retrieval (Milestone 1)", lexical_metrics)
    print_metrics("Semantic retrieval (Milestone 2)", semantic_metrics)
    print_metrics("Brand-neutral semantic retrieval (candidate)", brand_neutral_metrics)
    print_metrics("Hybrid retrieval (candidate)", hybrid_metrics)
    save_results(
        questions, lexical_results, semantic_results,
        brand_neutral_results, hybrid_results,
    )
    print(f"\nDetailed results saved to: {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()

