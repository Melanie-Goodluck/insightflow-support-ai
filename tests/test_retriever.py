import unittest
from pathlib import Path

from src.retriever import load_chunks, search


PROJECT_DIRECTORY = Path(__file__).parents[1]


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_chunks(PROJECT_DIRECTORY / "data" / "product_docs")

    def assert_top_heading(self, question: str, expected_heading: str) -> None:
        results = search(question, self.chunks)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.heading, expected_heading)

    def test_csv_rejection(self) -> None:
        self.assert_top_heading("Why was my CSV upload rejected?", "Troubleshooting rejected uploads")

    def test_postgresql(self) -> None:
        self.assert_top_heading("How do I connect PostgreSQL?", "PostgreSQL setup")

    def test_stale_dashboard(self) -> None:
        self.assert_top_heading("Why does my dashboard show old data?", "Stale dashboard data")

    def test_error_code(self) -> None:
        self.assert_top_heading("What does error E104 mean?", "Troubleshooting rejected uploads")


if __name__ == "__main__":
    unittest.main()

