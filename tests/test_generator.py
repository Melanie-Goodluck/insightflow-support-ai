import unittest

from src.generator import (
    FALLBACK_ANSWER,
    GroundedAnswerGenerator,
    build_grounded_prompt,
    grounding_coverage,
)
from src.retriever import Chunk, SearchResult


class FakeGenerationPipeline:
    def __init__(self) -> None:
        self.last_prompt = ""

    def __call__(self, text, **kwargs):
        self.last_prompt = text
        return [{"generated_text": "Check the file size and column headings."}]


class FixedOutputPipeline:
    def __init__(self, output: str) -> None:
        self.output = output

    def __call__(self, text, **kwargs):
        return [{"generated_text": self.output}]


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = SearchResult(
            Chunk(
                "file_uploads.md",
                "Troubleshooting rejected uploads",
                "Check the file size and column headings.",
            ),
            0.75,
        )

    def test_prompt_contains_question_and_retrieved_evidence(self) -> None:
        prompt = build_grounded_prompt("Why did my upload fail?", [self.result])
        self.assertIn("Why did my upload fail?", prompt)
        self.assertIn("Check the file size and column headings.", prompt)
        self.assertIn("Troubleshooting rejected uploads", prompt)

    def test_generator_returns_answer_and_deterministic_source(self) -> None:
        fake_pipeline = FakeGenerationPipeline()
        generator = GroundedAnswerGenerator(fake_pipeline)

        response = generator.generate("Why did my upload fail?", [self.result])

        self.assertEqual(response.answer, "Check the file size and column headings.")
        self.assertEqual(
            response.sources,
            ["Troubleshooting rejected uploads (file_uploads.md)"],
        )
        self.assertFalse(response.refused)
        self.assertEqual(response.decision_reason, "answered")
        self.assertEqual(response.raw_answer, response.answer)

    def test_low_relevance_result_is_refused_without_generation(self) -> None:
        fake_pipeline = FakeGenerationPipeline()
        generator = GroundedAnswerGenerator(fake_pipeline, minimum_score=0.20)
        low_result = SearchResult(self.result.chunk, 0.10)

        response = generator.generate("What is your refund policy?", [low_result])

        self.assertEqual(response.answer, FALLBACK_ANSWER)
        self.assertTrue(response.refused)
        self.assertEqual(fake_pipeline.last_prompt, "")
        self.assertEqual(response.decision_reason, "low_similarity")
        self.assertEqual(
            response.retrieved_evidence,
            ["Troubleshooting rejected uploads (file_uploads.md)"],
        )

    def test_unsupported_generated_answer_is_refused(self) -> None:
        generator = GroundedAnswerGenerator(FixedOutputPipeline("Refund Policy."))

        response = generator.generate("What is the refund policy?", [self.result])

        self.assertEqual(response.answer, FALLBACK_ANSWER)
        self.assertTrue(response.refused)
        self.assertEqual(response.sources, [])
        self.assertEqual(response.raw_answer, "Refund Policy.")
        self.assertEqual(response.decision_reason, "low_grounding_coverage")
        self.assertEqual(response.grounding_score, 0.0)

    def test_heading_only_answer_falls_back_to_document_text(self) -> None:
        generator = GroundedAnswerGenerator(
            FixedOutputPipeline("Troubleshooting rejected uploads.")
        )

        response = generator.generate("Why did my upload fail?", [self.result])

        self.assertEqual(response.answer, self.result.chunk.text)
        self.assertEqual(
            response.sources,
            ["Troubleshooting rejected uploads (file_uploads.md)"],
        )
        self.assertFalse(response.refused)
        self.assertEqual(response.decision_reason, "low_information_fallback")

    def test_low_information_answer_needs_strong_retrieval_for_fallback(self) -> None:
        generator = GroundedAnswerGenerator(FixedOutputPipeline("Forgot Password."))
        weak_result = SearchResult(
            Chunk(
                "accounts_and_access.md",
                "Sign-in problems",
                "A user can select Forgot Password to request a reset link.",
            ),
            0.276,
        )

        response = generator.generate(
            "How do I permanently delete my account?", [weak_result]
        )

        self.assertEqual(response.answer, FALLBACK_ANSWER)
        self.assertTrue(response.refused)
        self.assertEqual(
            response.decision_reason, "weak_low_information_fallback"
        )
        self.assertEqual(response.raw_answer, "Forgot Password.")

    def test_short_factual_answer_is_not_replaced_by_document_text(self) -> None:
        generator = GroundedAnswerGenerator(
            FixedOutputPipeline("CSV, XLSX, and JSON.")
        )
        formats_result = SearchResult(
            Chunk(
                "file_uploads.md",
                "Supported formats",
                "InsightFlow accepts CSV, XLSX, and JSON files.",
            ),
            0.55,
        )

        response = generator.generate(
            "Which file formats are supported?", [formats_result]
        )

        self.assertEqual(response.answer, "CSV, XLSX, and JSON.")
        self.assertFalse(response.refused)
        self.assertEqual(response.decision_reason, "answered")

    def test_grounding_coverage_detects_supported_words(self) -> None:
        coverage = grounding_coverage(
            "Check the file size and column headings.", [self.result]
        )
        self.assertEqual(coverage, 1.0)

    def test_empty_generation_has_a_distinct_refusal_reason(self) -> None:
        generator = GroundedAnswerGenerator(FixedOutputPipeline(""))

        response = generator.generate("Why did my upload fail?", [self.result])

        self.assertTrue(response.refused)
        self.assertEqual(response.decision_reason, "empty_generation")
        self.assertEqual(response.raw_answer, "")


if __name__ == "__main__":
    unittest.main()
