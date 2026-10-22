"""Run the labelled evaluation for the complete local RAG application."""

from pathlib import Path

from src.generator import GroundedAnswerGenerator, load_generation_pipeline
from src.rag_evaluation import (
    calculate_rag_metrics, load_rag_evaluation_items, run_rag_evaluation,
    write_rag_results_csv,
)
from src.retriever import load_chunks
from src.semantic_retriever import SemanticRetriever, load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"
QUESTIONS_PATH = PROJECT_DIRECTORY / "data" / "rag_evaluation_questions.json"
RESULTS_PATH = PROJECT_DIRECTORY / "rag_evaluation_results.csv"


def main() -> None:
    print("Loading 18 labelled RAG evaluation questions...")
    items = load_rag_evaluation_items(QUESTIONS_PATH)
    print("Loading the local embedding model...")
    chunks = load_chunks(DOCUMENTS_DIRECTORY)
    retriever = SemanticRetriever(chunks, load_embedding_model())
    print("Loading the local answer-generation model...")
    generator = GroundedAnswerGenerator(load_generation_pipeline())

    def answer_question(question: str):
        retrieved = retriever.search(
            question, limit=3, remove_product_name=True
        )
        response = generator.generate(question, retrieved)
        top_score = retrieved[0].score if retrieved else None
        return response, top_score

    print("Running the complete RAG evaluation. CPU generation may take time...")
    results = run_rag_evaluation(items, answer_question)
    for number, result in enumerate(results, start=1):
        status = "REFUSED" if result.refused else "ANSWERED"
        print(
            f"[{number:02d}/{len(results)}] {result.item.question_id} "
            f"{status} in {result.latency_seconds:.2f}s"
        )

    metrics = calculate_rag_metrics(results)
    write_rag_results_csv(results, RESULTS_PATH)
    print("\nAutomatic RAG evaluation results")
    print("--------------------------------")
    print(f"Questions:                       {metrics.total_questions}")
    print(f"Routing accuracy:                {metrics.routing_accuracy:.1%}")
    print(f"Answerable-question answer rate: {metrics.answerable_answer_rate:.1%}")
    print(f"Unanswerable refusal accuracy:   {metrics.unanswerable_refusal_accuracy:.1%}")
    print(f"False-refusal rate:              {metrics.false_refusal_rate:.1%}")
    print(f"Unsupported-answer rate:         {metrics.unsupported_answer_rate:.1%}")
    print(f"Expected-source retrieval:       {metrics.expected_source_retrieval_rate:.1%}")
    print(f"Mean response time:              {metrics.mean_latency_seconds:.2f}s")
    print(f"Median response time:            {metrics.median_latency_seconds:.2f}s")
    print(f"\nDetailed results saved to: {RESULTS_PATH.name}")
    print(
        "Human review is still required. Complete the correctness, "
        "faithfulness, helpfulness, and review-notes columns in the CSV."
    )


if __name__ == "__main__":
    main()
