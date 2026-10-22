"""FastAPI web application for the InsightFlow support assistant."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from src.generator import GroundedAnswerGenerator, load_generation_pipeline
from src.rag_service import RAGService
from src.retriever import load_chunks
from src.semantic_retriever import SemanticRetriever, load_embedding_model


PROJECT_DIRECTORY = Path(__file__).parent
DOCUMENTS_DIRECTORY = PROJECT_DIRECTORY / "data" / "product_docs"
STATIC_DIRECTORY = PROJECT_DIRECTORY / "static"


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
        version="6.0.0",
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

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/api/health")
    async def health(request: Request):
        return {
            "status": "ready",
            "service_loaded": hasattr(request.app.state, "rag_service"),
        }

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
