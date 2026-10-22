"""Hybrid retrieval using semantic meaning and transparent lexical evidence."""

from __future__ import annotations

import re

from src.retriever import Chunk, SearchResult, search
from src.semantic_retriever import EmbeddingModel, SemanticRetriever


PRODUCT_NAME_PATTERN = re.compile(r"\binsightflow\b", re.IGNORECASE)


def lexical_query(question: str) -> str:
    """Remove the repeated product name from the lexical side of retrieval."""

    return " ".join(PRODUCT_NAME_PATTERN.sub(" ", question).split())


class HybridRetriever:
    """Fuse semantic and lexical rankings using reciprocal rank fusion."""

    def __init__(
        self,
        chunks: list[Chunk],
        model: EmbeddingModel,
        rrf_k: int = 10,
        lexical_weight: float = 1.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if lexical_weight < 0:
            raise ValueError("lexical_weight cannot be negative")
        self.chunks = chunks
        self.semantic = SemanticRetriever(chunks, model)
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight

    @staticmethod
    def _key(result: SearchResult) -> tuple[str, str]:
        return result.chunk.source, result.chunk.heading

    def search(self, question: str, limit: int = 3) -> list[SearchResult]:
        if not question.strip():
            return []

        semantic_results = self.semantic.search(question, limit=len(self.chunks))
        lexical_results = search(
            lexical_query(question), self.chunks, limit=len(self.chunks)
        )
        semantic_ranks = {
            self._key(result): rank
            for rank, result in enumerate(semantic_results, start=1)
        }
        lexical_ranks = {
            self._key(result): rank
            for rank, result in enumerate(lexical_results, start=1)
        }

        def fused_score(result: SearchResult) -> float:
            key = self._key(result)
            semantic_part = 1 / (self.rrf_k + semantic_ranks[key])
            lexical_rank = lexical_ranks.get(key)
            lexical_part = (
                0.0
                if lexical_rank is None
                else self.lexical_weight / (self.rrf_k + lexical_rank)
            )
            return semantic_part + lexical_part

        reranked = sorted(
            semantic_results,
            key=lambda result: (fused_score(result), result.score),
            reverse=True,
        )
        # Keep the original semantic cosine score so the existing relevance
        # threshold retains the same meaning after reranking.
        return reranked[:limit]
