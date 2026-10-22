"""Command-line application for the first InsightFlow retrieval milestone."""

from pathlib import Path

from src.retriever import load_chunks, search


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"


def main() -> None:
    chunks = load_chunks(DOCUMENTS_DIRECTORY)
    print("InsightFlow Support AI - Milestone 1")
    print("Ask a product-support question, or type 'quit' to finish.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break

        results = search(question, chunks)
        if not results:
            print("\nI could not find relevant information in the documentation.\n")
            continue

        print("\nMost relevant evidence:")
        for position, result in enumerate(results, start=1):
            print(f"\n{position}. {result.chunk.heading}")
            print(f"   Source: {result.chunk.source}")
            print(f"   Similarity: {result.score:.3f}")
            print(f"   {result.chunk.text}")
        print()


if __name__ == "__main__":
    main()

