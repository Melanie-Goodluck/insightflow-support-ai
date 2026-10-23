"""Regression tests for the FastAPI boundary."""

from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import unittest
from types import SimpleNamespace


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class WebAppTests(unittest.TestCase):
    @staticmethod
    def build_response(question):
        return SimpleNamespace(
            answer=f"Received: {question}",
            sources=["test.md"],
            refused=False,
            decision_reason="answered",
            grounding_score=1.0,
            retrieved_evidence=["Test evidence"],
            latency_seconds=0.01,
        )

    @staticmethod
    def build_service():
        return SimpleNamespace(ask=WebAppTests.build_response)

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

    def test_request_logger_has_its_own_output_handler(self):
        from web_app import request_logger

        self.assertTrue(request_logger.handlers)
        self.assertFalse(request_logger.propagate)

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

    def test_busy_model_rejects_extra_request_but_health_stays_responsive(self):
        from fastapi.testclient import TestClient

        from web_app import create_app

        inference_started = Event()
        release_inference = Event()

        def slow_ask(question):
            inference_started.set()
            release_inference.wait(timeout=2)
            return self.build_response(question)

        service = SimpleNamespace(ask=slow_ask)
        app = create_app(
            service,
            max_concurrent_inference=1,
            queue_timeout_seconds=0.05,
        )

        with TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first_request = executor.submit(
                    client.post, "/api/ask", json={"question": "First question"}
                )
                try:
                    self.assertTrue(inference_started.wait(timeout=1))
                    health_response = client.get("/api/health/ready")
                    busy_response = client.post(
                        "/api/ask", json={"question": "Second question"}
                    )
                finally:
                    release_inference.set()

                first_response = first_request.result(timeout=2)

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json()["status"], "ready")
        self.assertEqual(busy_response.status_code, 503)
        self.assertEqual(busy_response.headers["Retry-After"], "1")
        self.assertEqual(
            busy_response.json()["detail"],
            "The AI service is busy. Please try again shortly.",
        )
        self.assertIsNotNone(busy_response.headers.get("X-Request-ID"))
        self.assertEqual(first_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
