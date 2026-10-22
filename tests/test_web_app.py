"""Regression tests for the FastAPI boundary."""

from __future__ import annotations

import importlib.util
import json
import unittest
from types import SimpleNamespace


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class WebAppTests(unittest.TestCase):
    @staticmethod
    def build_service():
        return SimpleNamespace(
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

    def test_ask_accepts_a_json_question(self):
        from fastapi.testclient import TestClient

        from web_app import create_app

        with TestClient(create_app(self.build_service())) as client:
            response = client.post(
                "/api/ask", json={"question": "How do I connect PostgreSQL?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"], "Received: How do I connect PostgreSQL?"
        )

    def test_request_has_id_and_privacy_safe_structured_log(self):
        from fastapi.testclient import TestClient

        from web_app import create_app

        private_question = "This private question must not appear in logs"
        with self.assertLogs("insightflow.requests", level="INFO") as captured:
            with TestClient(create_app(self.build_service())) as client:
                response = client.post(
                    "/api/ask", json={"question": private_question}
                )

        request_id = response.headers.get("X-Request-ID")
        self.assertIsNotNone(request_id)
        self.assertEqual(len(request_id), 36)

        log_record = json.loads(captured.output[-1].split(":", 2)[-1])
        self.assertEqual(log_record["event"], "http_request")
        self.assertEqual(log_record["request_id"], request_id)
        self.assertEqual(log_record["method"], "POST")
        self.assertEqual(log_record["path"], "/api/ask")
        self.assertEqual(log_record["status_code"], 200)
        self.assertGreaterEqual(log_record["duration_ms"], 0)
        self.assertNotIn(private_question, captured.output[-1])

    def test_liveness_and_readiness_endpoints(self):
        from fastapi.testclient import TestClient

        from web_app import create_app

        with TestClient(create_app(self.build_service())) as client:
            live_response = client.get("/api/health/live")
            ready_response = client.get("/api/health/ready")
            original_response = client.get("/api/health")

        self.assertEqual(live_response.json(), {"status": "alive"})
        expected_ready = {"status": "ready", "service_loaded": True}
        self.assertEqual(ready_response.json(), expected_ready)
        self.assertEqual(original_response.json(), expected_ready)
        self.assertIsNotNone(ready_response.headers.get("X-Request-ID"))


if __name__ == "__main__":
    unittest.main()
