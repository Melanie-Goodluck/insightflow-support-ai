"""Interactive local RAG assistant: retrieve, augment, then generate."""

from pathlib import Path

from src.generator import GroundedAnswerGenerator, load_generation_pipeline
from src.retriever import load_chunks
from src.semantic_retriever import SemanticRetriever, load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"


def main() -> None:
    chunks = load_chunks(DOCUMENTS_DIRECTORY)

    print("Loading the local embedding model...")
    retriever = SemanticRetriever(chunks, load_embedding_model())

    print("Loading the local answer-generation model...")
    generator = GroundedAnswerGenerator(load_generation_pipeline())
    print("InsightFlow RAG assistant is ready. Type 'quit' to finish.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        retrieved = retriever.search(
            question, limit=3, remove_product_name=True
        )
        response = generator.generate(question, retrieved)

        print(f"\nAnswer:\n{response.answer}")
        if response.sources:
            print("\nRetrieved evidence:")
            for source in response.sources:
                print(f"- {source}")
        print()


if __name__ == "__main__":
    main()
