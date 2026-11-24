"""Schémas Pydantic de l'API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _valider_question(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("La question ne peut pas être vide.")
    return value


class DocumentInfo(BaseModel):
    """Métadonnées d'un document indexé."""

    id: str = ""
    title: str = ""
    source: str = ""
    kind: str = ""
    markdown_path: str | None = None
    characters: int | None = None
    chunks: int | None = None


class ContextOut(BaseModel):
    """Chunk retrouvé et document dont il provient."""

    chunk: str
    score: float
    document: DocumentInfo


# --- Flux 1 : prompt + réponse fusionnés -----------------------------------
class PromptRequest(BaseModel):
    """Entrée du flux prompt/réponse : la question à poser."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Quel concert de jazz à Saint-Denis ?"],
    )
    top_k: int = Field(default=4, ge=1, le=20)
    score_min: float = Field(default=0.70, ge=-1.0, le=1.0)

    @field_validator("question")
    @classmethod
    def _question_non_vide(cls, value: str) -> str:
        return _valider_question(value)  # type: ignore[return-value]


class PromptResponse(BaseModel):
    """Sortie JSON : la réponse est renvoyée immédiatement avec les sources."""

    question: str
    answer: str
    contexts: list[ContextOut]
    model: str
    elapsed_ms: float


# --- Flux 2 : ingestion -----------------------------------------------------
class SourceResultat(BaseModel):
    source: str
    title: str | None = None
    kind: str
    status: Literal["ok", "skipped", "error"]
    characters: int = 0
    chunks: int = 0
    markdown_path: str | None = None
    error: str | None = None


class IndexStats(BaseModel):
    exists: bool
    documents: int
    chunks: int
    dimension: int
    updated_at: str | None = None


class IngestionOut(BaseModel):
    replace: bool
    sources: list[SourceResultat]
    index: IndexStats


# --- Flux 3 : métriques -----------------------------------------------------
class EvalCase(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    expected_snippets: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_non_vide(cls, value: str) -> str:
        return _valider_question(value)  # type: ignore[return-value]


class MetricsRequest(BaseModel):
    cases: list[EvalCase] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    score_min: float = Field(default=0.70, ge=-1.0, le=1.0)
    include_generation: bool = False


class MetricsOut(BaseModel):
    n_cases: int
    index: IndexStats
    aggregate: dict[str, Any]
    results: list[dict[str, Any]]
