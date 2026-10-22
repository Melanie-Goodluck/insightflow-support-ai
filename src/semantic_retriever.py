"""Semantic retrieval using a free, locally run sentence embedding model."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Protocol

from src.retriever import Chunk, SearchResult


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PRODUCT_NAME_PATTERN = re.compile(r"\binsightflow\b", re.IGNORECASE)


def semantic_query(question: str, remove_product_name: bool = False) -> str:
    """Prepare a question for embedding without changing the user's text."""

    if not remove_product_name:
        return question
    prepared = " ".join(PRODUCT_NAME_PATTERN.sub(" ", question).split())
    prepared = re.sub(r"\s+(?:to|in|on|from|with|for)\s*([?.!,])", r"\1", prepared)
    return prepared


class EmbeddingModel(Protocol):
    """The small part of SentenceTransformer that our retriever needs."""

    def encode(
        self, sentences: str | list[str], *, normalize_embeddings: bool
    ) -> Sequence[float] | Sequence[Sequence[float]]:
        """Convert one or more texts into numerical vectors."""


def load_embedding_model() -> EmbeddingModel:
    """Load the local model, downloading it on first use if necessary."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "The sentence-transformers package is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    return SentenceTransformer(MODEL_NAME)


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity for vectors already normalised to length 1."""

    return float(sum(a * b for a, b in zip(left, right)))


class SemanticRetriever:
    """Create document embeddings once, then reuse them for every question."""

    def __init__(self, chunks: list[Chunk], model: EmbeddingModel) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required")

        self.chunks = chunks
        searchable_texts = [f"{chunk.heading}. {chunk.text}" for chunk in chunks]
        self.chunk_embeddings = model.encode(
            searchable_texts, normalize_embeddings=True
        )
        self.model = model

    def search(
        self,
        question: str,
        limit: int = 3,
        remove_product_name: bool = False,
    ) -> list[SearchResult]:
        """Return chunks whose meanings are closest to the question's meaning."""

        if not question.strip():
            return []

        prepared_question = semantic_query(question, remove_product_name)
        question_embedding = self.model.encode(
            prepared_question, normalize_embeddings=True
        )
        results = [
            SearchResult(chunk, _dot_product(question_embedding, chunk_embedding))
            for chunk, chunk_embedding in zip(self.chunks, self.chunk_embeddings)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]

