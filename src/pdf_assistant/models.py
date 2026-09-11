from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    filename: str
    page: int
    heading: str
    text: str
    char_count: int
    content_kind: str = "text"
    metadata_json: str = "{}"
    image_path: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class RetrievedMemory:
    id: str
    kind: str
    source_id: int
    session_id: str
    folder_ids: list[str]
    title: str
    content: str
    created_at: str
    score: float


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    page_count: int
    chunks: list[DocumentChunk]
    warnings: list[str]
