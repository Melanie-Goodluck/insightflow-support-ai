"""Inspect one RAG decision without running the full evaluation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.generator import GroundedAnswerGenerator, load_generation_pipeline
from src.hybrid_retriever import HybridRetriever
from src.retriever import load_chunks
from src.semantic_retriever import load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"
DEFAULT_QUESTION = "How can I invite a colleague to InsightFlow?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one InsightFlow RAG decision")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    chunks = load_chunks(DOCUMENTS_DIRECTORY)
    print("Loading the local embedding model...")
    hybrid_retriever = HybridRetriever(chunks, load_embedding_model())
    print("Loading the local answer-generation model...")
    generator = GroundedAnswerGenerator(load_generation_pipeline())

    semantic_results = hybrid_retriever.semantic.search(args.question, limit=3)
    brand_neutral_results = hybrid_retriever.semantic.search(
        args.question, limit=3, remove_product_name=True
    )
    hybrid_results = hybrid_retriever.search(args.question, limit=3)
    retrieved = brand_neutral_results
    response = generator.generate(args.question, retrieved)

    print(f"\nQuestion: {args.question}")
    print("\nSemantic-only candidate evidence:")
    for number, result in enumerate(semantic_results, start=1):
        print(
            f"{number}. {result.chunk.heading} ({result.chunk.source}) "
            f"score={result.score:.3f}"
        )
    print("\nBrand-neutral semantic candidate evidence:")
    for number, result in enumerate(brand_neutral_results, start=1):
        print(
            f"{number}. {result.chunk.heading} ({result.chunk.source}) "
            f"score={result.score:.3f}"
        )
    print("\nHybrid candidate evidence (rejected by overall benchmark):")
    for number, result in enumerate(hybrid_results, start=1):
        print(
            f"{number}. {result.chunk.heading} ({result.chunk.source}) "
            f"score={result.score:.3f}"
        )
    print(f"\nRaw model answer: {response.raw_answer!r}")
    print(
        "Grounding coverage: "
        + ("not calculated" if response.grounding_score is None else f"{response.grounding_score:.3f}")
    )
    print(f"Decision reason: {response.decision_reason}")
    print(f"Refused: {response.refused}")
    print(f"\nFinal answer: {response.answer}")


if __name__ == "__main__":
    main()
