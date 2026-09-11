from __future__ import annotations

import math
import re
import threading
import uuid
from collections import defaultdict
from collections.abc import Sequence

import jieba
import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .database import Database
from .embedding import EmbeddingModel
from .models import DocumentChunk, RetrievedChunk, RetrievedMemory

TOKEN_RE = re.compile(r"[A-Za-z]+(?:_[A-Za-z0-9]+)?|\d+(?:\.\d+)?|[^\W\d_]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    words = [token.strip().lower() for token in jieba.lcut(text) if token.strip()]
    symbols = TOKEN_RE.findall(text.lower())
    return words + symbols


class Retriever:
    collection_name = "pdf_chunks"
    memory_collection_name = "learning_memories"

    def __init__(
        self,
        *,
        database: Database,
        vector_path: str,
        embedder: EmbeddingModel,
        enable_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.database = database
        self.embedder = embedder
        self.enable_reranker = enable_reranker
        self.reranker_model = reranker_model
        self.client = QdrantClient(path=vector_path)
        self._reranker = None
        self._reranker_lock = threading.Lock()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def memory_collection_exists(self) -> bool:
        return self.client.collection_exists(self.memory_collection_name)

    def _ensure_memory_collection(self) -> None:
        if self.memory_collection_exists():
            return
        self.client.create_collection(
            collection_name=self.memory_collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = self.embedder.encode([chunk.text for chunk in batch])
            points = [
                models.PointStruct(
                    id=chunk.id,
                    vector=vector.tolist(),
                    payload={
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page": chunk.page,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "char_count": chunk.char_count,
                        "content_kind": chunk.content_kind,
                        "metadata_json": chunk.metadata_json,
                        "image_path": chunk.image_path,
                    },
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.collection_name, points=points, wait=True)

    @staticmethod
    def _memory_point_id(memory_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pdf-assistant-memory:{memory_id}"))

    def index_memories(self, records: Sequence[dict]) -> list[str]:
        if not records:
            return []
        self._ensure_memory_collection()
        indexed_ids: list[str] = []
        for start in range(0, len(records), 64):
            batch = records[start : start + 64]
            vectors = self.embedder.encode([str(item["content"]) for item in batch])
            points = [
                models.PointStruct(
                    id=self._memory_point_id(str(item["id"])),
                    vector=vector.tolist(),
                    payload={
                        "memory_id": str(item["id"]),
                        "kind": str(item["kind"]),
                        "source_id": int(item["source_id"]),
                        "session_id": str(item["session_id"]),
                        "folder_ids": list(item.get("folder_ids") or []),
                        "title": str(item["title"]),
                        "content": str(item["content"]),
                        "created_at": str(item["created_at"]),
                    },
                )
                for item, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.memory_collection_name, points=points, wait=True)
            indexed_ids.extend(str(item["id"]) for item in batch)
        return indexed_ids

    def delete_session_memories(self, session_id: str) -> None:
        if not session_id or not self.memory_collection_exists():
            return
        self.client.delete(
            collection_name=self.memory_collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search_memories(
        self,
        question: str,
        *,
        session_id: str,
        include_all_sessions: bool,
        folder_id: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedMemory]:
        if not self.memory_collection_exists():
            return []
        conditions: list[models.Condition] = []
        if not include_all_sessions:
            conditions.append(
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            )
        if folder_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="folder_ids",
                    match=models.MatchValue(value=folder_id),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        vector = self.embedder.encode([question])[0].tolist()
        response = self.client.query_points(
            collection_name=self.memory_collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.25,
            with_payload=True,
        )
        results: list[RetrievedMemory] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedMemory(
                    id=str(payload.get("memory_id", "")),
                    kind=str(payload.get("kind", "")),
                    source_id=int(payload.get("source_id", 0)),
                    session_id=str(payload.get("session_id", "")),
                    folder_ids=[str(value) for value in payload.get("folder_ids", [])],
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                    created_at=str(payload.get("created_at", "")),
                    score=float(point.score),
                )
            )
        return results

    def _dense_search(
        self, question: str, document_ids: Sequence[str], limit: int
    ) -> list[tuple[DocumentChunk, float]]:
        if not self.client.collection_exists(self.collection_name):
            return []
        vector = self.embedder.encode([question])[0].tolist()
        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=list(document_ids)),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        results: list[tuple[DocumentChunk, float]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                (
                    DocumentChunk(
                        id=str(point.id),
                        document_id=str(payload.get("document_id", "")),
                        filename=str(payload.get("filename", "")),
                        page=int(payload.get("page", 0)),
                        heading=str(payload.get("heading", "")),
                        text=str(payload.get("text", "")),
                        char_count=int(payload.get("char_count", 0)),
                        content_kind=str(payload.get("content_kind", "text")),
                        metadata_json=str(payload.get("metadata_json", "{}")),
                        image_path=str(payload.get("image_path", "")),
                    ),
                    float(point.score),
                )
            )
        return results

    def _lexical_search(
        self, question: str, document_ids: Sequence[str], limit: int
    ) -> list[tuple[DocumentChunk, float]]:
        chunks = self.database.list_chunks(document_ids or None)
        if not chunks:
            return []
        corpus = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(corpus)
        scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=float)
        if not np.any(scores > 0):
            return []
        indices = np.argsort(scores)[::-1][:limit]
        maximum = float(scores[indices[0]]) if len(indices) else 1.0
        return [
            (chunks[int(index)], float(scores[int(index)] / maximum))
            for index in indices
            if scores[int(index)] > 0
        ]

    def _get_reranker(self):
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder

                    self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def search(
        self,
        question: str,
        document_ids: Sequence[str],
        *,
        candidate_limit: int = 18,
        final_limit: int = 6,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        dense = self._dense_search(question, document_ids, candidate_limit)
        lexical = self._lexical_search(question, document_ids, candidate_limit)

        rrf: dict[str, float] = defaultdict(float)
        dense_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}
        constant = 60.0
        for rank, (chunk, score) in enumerate(dense, start=1):
            chunks[chunk.id] = chunk
            dense_scores[chunk.id] = score
            rrf[chunk.id] += 1.0 / (constant + rank)
        for rank, (chunk, score) in enumerate(lexical, start=1):
            chunks[chunk.id] = chunk
            lexical_scores[chunk.id] = score
            rrf[chunk.id] += 1.0 / (constant + rank)

        ranked_ids = sorted(rrf, key=rrf.get, reverse=True)[:candidate_limit]
        use_reranker = self.enable_reranker if rerank is None else rerank and self.enable_reranker
        if use_reranker and ranked_ids:
            pairs = [(question, chunks[chunk_id].text) for chunk_id in ranked_ids]
            rerank_scores = self._get_reranker().predict(pairs)
            reranked = sorted(
                zip(ranked_ids, rerank_scores, strict=True),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            ranked_ids = [chunk_id for chunk_id, _ in reranked]

        scale = max((rrf[value] for value in ranked_ids), default=1.0)
        if math.isclose(scale, 0.0):
            scale = 1.0
        return [
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=rrf[chunk_id] / scale,
                dense_score=dense_scores.get(chunk_id, 0.0),
                lexical_score=lexical_scores.get(chunk_id, 0.0),
            )
            for chunk_id in ranked_ids[:final_limit]
        ]
