"""Compare Milestone 1 lexical retrieval with Milestone 2 semantic retrieval."""

from pathlib import Path

from src.retriever import load_chunks, search
from src.semantic_retriever import SemanticRetriever, load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"


def print_results(label: str, results: list) -> None:
    print(f"\n{label}")
    for position, result in enumerate(results, start=1):
        print(
            f"{position}. {result.chunk.heading} "
            f"({result.chunk.source}, score={result.score:.3f})"
        )


def main() -> None:
    chunks = load_chunks(DOCUMENTS_DIRECTORY)
    print("Loading the free local embedding model...")
    semantic_retriever = SemanticRetriever(chunks, load_embedding_model())
    print("Model ready. Type 'quit' to finish.")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break

        print_results("Lexical results (Milestone 1)", search(question, chunks))
        print_results(
            "Semantic results (Milestone 2)", semantic_retriever.search(question)
        )


if __name__ == "__main__":
    main()

