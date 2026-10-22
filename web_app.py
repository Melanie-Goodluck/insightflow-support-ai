"""FastAPI web application for the InsightFlow support assistant."""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from src.generator import GroundedAnswerGenerator, load_generation_pipeline
from src.rag_service import RAGService
from src.retriever import load_chunks
from src.semantic_retriever import SemanticRetriever, load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"
STATIC_DIRECTORY = PROJECT_DIRECTORY / "static"
REQUEST_ID_HEADER = "X-Request-ID"
request_logger = logging.getLogger("insightflow.requests")
request_logger.setLevel(logging.INFO)


def build_service() -> RAGService:
    """Load both local models once and assemble the production service."""

    print("Loading InsightFlow documentation and local embedding model...")
    chunks = load_chunks(DOCUMENTS_DIRECTORY)
    retriever = SemanticRetriever(chunks, load_embedding_model())
    print("Loading the local answer-generation model...")
    generator = GroundedAnswerGenerator(load_generation_pipeline())
    return RAGService(retriever, generator)


def create_app(service: RAGService | None = None) -> Any:
    """Create the API; an injected service keeps endpoint tests lightweight."""

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError(
            "FastAPI is not installed. Run: python -m pip install -r requirements.txt"
        ) from error

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.rag_service = build_service()
        else:
            app.state.rag_service = service
        yield

    app = FastAPI(
        title="InsightFlow Support AI",
        version="7.0.0",
        description="A locally run, evaluated RAG support assistant.",
        lifespan=lifespan,
    )

    class AskRequest(BaseModel):
        question: str = Field(min_length=1, max_length=500)

    class AskResponse(BaseModel):
        answer: str
        sources: list[str]
        refused: bool
        decision_reason: str
        grounding_score: float | None
        retrieved_evidence: list[str]
        latency_seconds: float

    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        """Trace each request without storing questions or other user content."""

        request_id = str(uuid4())
        started_at = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            request_logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                    },
                    separators=(",", ":"),
                )
            )

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/api/health/live")
    async def liveness():
        return {"status": "alive"}

    @app.get("/api/health/ready")
    async def readiness(request: Request):
        service_loaded = hasattr(request.app.state, "rag_service")
        return {
            "status": "ready" if service_loaded else "not_ready",
            "service_loaded": service_loaded,
        }

    @app.get("/api/health", include_in_schema=False)
    async def health(request: Request):
        """Keep the original health URL working for existing users."""

        return await readiness(request)

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be empty")

        response = request.app.state.rag_service.ask(question)
        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            refused=response.refused,
            decision_reason=response.decision_reason,
            grounding_score=response.grounding_score,
            retrieved_evidence=response.retrieved_evidence,
            latency_seconds=round(response.latency_seconds, 3),
        )

    return app


app = create_app()
