import unittest

from src.retriever import Chunk
from src.semantic_retriever import SemanticRetriever, semantic_query


class FakeEmbeddingModel:
    """Tiny predictable model used to test our logic without a download."""

    vectors = {
        "Upload problems. Files can be rejected during import.": [1.0, 0.0],
        "Dashboard refresh. Reports may contain outdated information.": [0.0, 1.0],
        "My spreadsheet will not import": [0.9, 0.1],
        "My spreadsheet will not import?": [0.9, 0.1],
    }

    def encode(self, sentences, *, normalize_embeddings):
        if isinstance(sentences, str):
            return self.vectors[sentences]
        return [self.vectors[sentence] for sentence in sentences]


class SemanticRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        chunks = [
            Chunk("uploads.md", "Upload problems", "Files can be rejected during import."),
            Chunk(
                "dashboards.md",
                "Dashboard refresh",
                "Reports may contain outdated information.",
            ),
        ]
        self.retriever = SemanticRetriever(chunks, FakeEmbeddingModel())

    def test_semantically_similar_chunk_is_first(self) -> None:
        results = self.retriever.search("My spreadsheet will not import")
        self.assertEqual(results[0].chunk.heading, "Upload problems")

    def test_empty_question_returns_no_results(self) -> None:
        self.assertEqual(self.retriever.search("   "), [])

    def test_product_name_can_be_removed_without_changing_original_text(self):
        question = "How can I invite a colleague to InsightFlow?"

        prepared = semantic_query(question, remove_product_name=True)

        self.assertEqual(prepared, "How can I invite a colleague?")
        self.assertEqual(question, "How can I invite a colleague to InsightFlow?")

    def test_search_can_use_a_brand_neutral_semantic_query(self):
        results = self.retriever.search(
            "My spreadsheet will not import to InsightFlow?",
            remove_product_name=True,
        )

        self.assertEqual(results[0].chunk.heading, "Upload problems")


if __name__ == "__main__":
    unittest.main()

