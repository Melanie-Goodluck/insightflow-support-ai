import csv
import tempfile
import unittest
from pathlib import Path

from src.generator import GeneratedAnswer
from src.rag_evaluation import (
    RAGEvaluationItem, RAGEvaluationResult, calculate_rag_metrics,
    run_rag_evaluation, write_rag_results_csv,
)


def item(question_id: str, answerable: bool) -> RAGEvaluationItem:
    return RAGEvaluationItem(
        question_id=question_id,
        question=f"Question {question_id}",
        answerable=answerable,
        expected_heading="Supported formats" if answerable else None,
        reference_answer="Reference answer",
    )


def result(
    question_id, answerable, refused, sources=None, latency=1.0,
    candidate_evidence=None,
):
    return RAGEvaluationResult(
        item=item(question_id, answerable), answer="answer", refused=refused,
        sources=sources or [], top_similarity_score=0.5,
        latency_seconds=latency,
        candidate_evidence=candidate_evidence or sources or [],
    )


class RAGEvaluationTests(unittest.TestCase):
    def test_empty_results_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_rag_metrics([])

    def test_metrics_distinguish_false_refusals_and_unsupported_answers(self):
        results = [
            result("A1", True, False), result("A2", True, True),
            result("U1", False, True), result("U2", False, False),
        ]
        metrics = calculate_rag_metrics(results)
        self.assertEqual(metrics.routing_accuracy, 0.5)
        self.assertEqual(metrics.answerable_answer_rate, 0.5)
        self.assertEqual(metrics.unanswerable_refusal_accuracy, 0.5)
        self.assertEqual(metrics.false_refusal_rate, 0.5)
        self.assertEqual(metrics.unsupported_answer_rate, 0.5)

    def test_expected_source_is_detected_from_application_sources(self):
        evaluation_result = result(
            "A1", True, False, ["Supported formats (file_uploads.md)"]
        )
        self.assertTrue(evaluation_result.expected_source_retrieved)

    def test_expected_source_is_detected_even_when_answer_is_refused(self):
        evaluation_result = result(
            "A1", True, True, [],
            candidate_evidence=["Supported formats (file_uploads.md)"],
        )

        self.assertTrue(evaluation_result.expected_source_retrieved)

    def test_runner_records_response_and_top_score(self):
        def fake_runner(question: str):
            return GeneratedAnswer(
                "Supported answer", ["Source"], False,
                raw_answer="Supported answer", decision_reason="answered",
                grounding_score=1.0, retrieved_evidence=["Source"],
            ), 0.42

        results = run_rag_evaluation([item("A1", True)], fake_runner)
        self.assertEqual(results[0].answer, "Supported answer")
        self.assertEqual(results[0].top_similarity_score, 0.42)
        self.assertEqual(results[0].decision_reason, "answered")
        self.assertEqual(results[0].candidate_evidence, ["Source"])
        self.assertGreaterEqual(results[0].latency_seconds, 0.0)

    def test_csv_leaves_human_review_columns_blank(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "results.csv"
            write_rag_results_csv([result("A1", True, False)], destination)
            with destination.open(encoding="utf-8", newline="") as input_file:
                row = next(csv.DictReader(input_file))
        self.assertEqual(row["correctness_0_to_2"], "")
        self.assertEqual(row["faithfulness_0_to_2"], "")
        self.assertEqual(row["helpfulness_0_to_2"], "")

    def test_csv_records_diagnostic_fields(self):
        diagnostic_result = RAGEvaluationResult(
            item=item("A1", True), answer="fallback", refused=True,
            sources=[], top_similarity_score=0.5, latency_seconds=1.0,
            raw_answer="Invitation instructions.",
            decision_reason="low_grounding_coverage", grounding_score=0.25,
            candidate_evidence=["Supported formats (file_uploads.md)"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "results.csv"
            write_rag_results_csv([diagnostic_result], destination)
            with destination.open(encoding="utf-8", newline="") as input_file:
                row = next(csv.DictReader(input_file))

        self.assertEqual(row["decision_reason"], "low_grounding_coverage")
        self.assertEqual(row["raw_model_answer"], "Invitation instructions.")
        self.assertEqual(row["grounding_score"], "0.250")


if __name__ == "__main__":
    unittest.main()
