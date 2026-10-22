import unittest

from src.generator import GeneratedAnswer
from src.rag_service import RAGService
from src.retriever import Chunk, SearchResult


class FakeRetriever:
    def __init__(self):
        self.call = None

    def search(self, question, limit=3, remove_product_name=False):
        self.call = (question, limit, remove_product_name)
        return [SearchResult(Chunk("docs.md", "Help", "Use Settings."), 0.8)]


class FakeGenerator:
    def __init__(self):
        self.call = None

    def generate(self, question, results):
        self.call = (question, results)
        return GeneratedAnswer(
            answer="Use Settings.",
            sources=["Help (docs.md)"],
            refused=False,
            raw_answer="Use Settings.",
            decision_reason="answered",
            grounding_score=1.0,
            retrieved_evidence=["Help (docs.md)"],
        )


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.retriever = FakeRetriever()
        self.generator = FakeGenerator()
        self.service = RAGService(self.retriever, self.generator)

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.service.ask("   ")

    def test_service_uses_brand_neutral_retrieval(self):
        self.service.ask("  How do I use InsightFlow?  ")

        self.assertEqual(
            self.retriever.call,
            ("How do I use InsightFlow?", 3, True),
        )

    def test_service_returns_ui_ready_diagnostics(self):
        response = self.service.ask("How do I use Settings?")

        self.assertEqual(response.answer, "Use Settings.")
        self.assertEqual(response.sources, ["Help (docs.md)"])
        self.assertFalse(response.refused)
        self.assertEqual(response.decision_reason, "answered")
        self.assertEqual(response.grounding_score, 1.0)
        self.assertGreaterEqual(response.latency_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
