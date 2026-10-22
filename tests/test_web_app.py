"""Regression tests for the FastAPI boundary."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class WebAppTests(unittest.TestCase):
    def test_ask_accepts_a_json_question(self):
        from fastapi.testclient import TestClient

        from web_app import create_app

        service = SimpleNamespace(
            ask=lambda question: SimpleNamespace(
                answer=f"Received: {question}",
                sources=["test.md"],
                refused=False,
                decision_reason="answered",
                grounding_score=1.0,
                retrieved_evidence=["Test evidence"],
                latency_seconds=0.01,
            )
        )

        with TestClient(create_app(service)) as client:
            response = client.post(
                "/api/ask", json={"question": "How do I connect PostgreSQL?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"], "Received: How do I connect PostgreSQL?"
        )


if __name__ == "__main__":
    unittest.main()
