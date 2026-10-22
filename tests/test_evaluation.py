import unittest

from src.evaluation import (
    EvaluationMetrics,
    EvaluationQuestion,
    QuestionResult,
    calculate_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_the_expected_ranks(self) -> None:
        question = EvaluationQuestion("Example", "Expected", "direct")
        results = [
            QuestionResult(question, 1, ["Expected"]),
            QuestionResult(question, 2, ["Other", "Expected"]),
            QuestionResult(question, None, ["Other"]),
        ]

        metrics = calculate_metrics(results)

        self.assertAlmostEqual(metrics.top_1_accuracy, 1 / 3)
        self.assertAlmostEqual(metrics.top_3_accuracy, 2 / 3)
        self.assertAlmostEqual(metrics.mean_reciprocal_rank, 0.5)

    def test_empty_evaluation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_metrics([])


if __name__ == "__main__":
    unittest.main()

