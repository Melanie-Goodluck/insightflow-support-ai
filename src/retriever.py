"""Small, transparent document retriever for Milestone 1."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True)
class Chunk:
    """One searchable section from a product document."""

    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk together with its similarity score."""

    chunk: Chunk
    score: float


def tokenise(text: str) -> list[str]:
    """Convert text into lowercase searchable words."""

    return WORD_PATTERN.findall(text.lower())


def load_chunks(documents_directory: Path) -> list[Chunk]:
    """Read Markdown files and split them at level-two headings."""

    chunks: list[Chunk] = []
    for path in sorted(documents_directory.glob("*.md")):
        title = path.stem.replace("_", " ").title()
        heading = title
        body: list[str] = []

        def save_section() -> None:
            text = " ".join(line.strip() for line in body if line.strip())
            if text:
                chunks.append(Chunk(path.name, heading, text))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                heading = title
            elif line.startswith("## "):
                save_section()
                body.clear()
                heading = line[3:].strip()
            else:
                body.append(line)
        save_section()

    if not chunks:
        raise ValueError(f"No Markdown documents found in {documents_directory}")
    return chunks


def _idf(chunks: list[Chunk]) -> dict[str, float]:
    """Give rare, informative words more importance than common words."""

    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(tokenise(chunk.text + " " + chunk.heading)))

    total = len(chunks)
    return {
        word: math.log((total + 1) / (frequency + 1)) + 1
        for word, frequency in document_frequency.items()
    }


def _vector(text: str, weights: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenise(text))
    return {word: count * weights.get(word, 1.0) for word, count in counts.items()}


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Measure how closely two weighted word vectors point in the same direction."""

    shared_words = left.keys() & right.keys()
    dot_product = sum(left[word] * right[word] for word in shared_words)
    left_length = math.sqrt(sum(value**2 for value in left.values()))
    right_length = math.sqrt(sum(value**2 for value in right.values()))
    if not left_length or not right_length:
        return 0.0
    return dot_product / (left_length * right_length)


def search(question: str, chunks: list[Chunk], limit: int = 3) -> list[SearchResult]:
    """Return the chunks most similar to the user's question."""

    if not question.strip():
        return []

    weights = _idf(chunks)
    question_vector = _vector(question, weights)
    results = []
    for chunk in chunks:
        searchable_text = f"{chunk.heading} {chunk.text}"
        score = _cosine_similarity(question_vector, _vector(searchable_text, weights))
        results.append(SearchResult(chunk, score))

    results.sort(key=lambda result: result.score, reverse=True)
    return [result for result in results[:limit] if result.score > 0]

