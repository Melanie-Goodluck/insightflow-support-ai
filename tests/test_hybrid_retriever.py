import unittest

from src.hybrid_retriever import HybridRetriever, lexical_query
from src.retriever import Chunk


class MisleadingSemanticModel:
    vectors = {
        "Supported formats. InsightFlow accepts CSV files.": [1.0, 0.0],
        "Inviting team members. Administrators can invite a team member.": [0.6, 0.8],
        "How can I invite a colleague to InsightFlow?": [1.0, 0.0],
    }

    def encode(self, sentences, *, normalize_embeddings):
        if isinstance(sentences, str):
            return self.vectors[sentences]
        return [self.vectors[sentence] for sentence in sentences]


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("uploads.md", "Supported formats", "InsightFlow accepts CSV files."),
            Chunk(
                "accounts.md",
                "Inviting team members",
                "Administrators can invite a team member.",
            ),
        ]
        self.retriever = HybridRetriever(self.chunks, MisleadingSemanticModel())

    def test_product_name_is_removed_from_lexical_query(self):
        self.assertEqual(
            lexical_query("How can I invite someone to InsightFlow?"),
            "How can I invite someone to ?",
        )

    def test_lexical_evidence_rescues_a_semantic_miss(self):
        results = self.retriever.search(
            "How can I invite a colleague to InsightFlow?", limit=2
        )

        self.assertEqual(results[0].chunk.heading, "Inviting team members")
        self.assertAlmostEqual(results[0].score, 0.6)

    def test_empty_question_returns_no_results(self):
        self.assertEqual(self.retriever.search("   "), [])


if __name__ == "__main__":
    unittest.main()
