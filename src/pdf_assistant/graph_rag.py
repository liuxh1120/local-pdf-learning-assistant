from __future__ import annotations

import hashlib
import html
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import jieba.analyse

from .models import DocumentChunk
from .retrieval import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .database import Database


TERM_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_-]{2,}(?:\s+[A-Za-z]{3,})?")
SPACE_RE = re.compile(r"\s+")
STOP_TERMS = {
    "一个",
    "一种",
    "这个",
    "这些",
    "可以",
    "进行",
    "通过",
    "如果",
    "其中",
    "因此",
    "以及",
    "对于",
    "使用",
    "需要",
    "例如",
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
}


@dataclass(frozen=True)
class GraphEvidence:
    chunk: DocumentChunk
    score: float
    entities: tuple[str, ...]


class LocalGraphRAG:
    """Local entity co-occurrence graph retrieval for the PDF knowledge base."""

    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _normalize(term: str) -> str:
        return SPACE_RE.sub(" ", term.strip().lower()).strip("-_.,，。:：;；()（）[]【】")

    @classmethod
    def _entity_kind(cls, term: str) -> str:
        if term.startswith("\\"):
            return "formula"
        if term.isascii():
            return "english"
        return "concept"

    @classmethod
    def _extract_terms(cls, text: str, limit: int = 14) -> list[tuple[str, float]]:
        weighted: dict[str, tuple[str, float]] = {}
        for term, weight in jieba.analyse.extract_tags(text, topK=limit, withWeight=True):
            normalized = cls._normalize(term)
            if len(normalized) < 2 or len(normalized) > 48 or normalized in STOP_TERMS:
                continue
            weighted[normalized] = (term.strip(), float(weight))
        for match in TERM_RE.finditer(text):
            term = match.group(0).strip()
            normalized = cls._normalize(term)
            if len(normalized) < 3 or normalized in STOP_TERMS:
                continue
            previous = weighted.get(normalized)
            weight = max(0.65, previous[1] if previous else 0.0)
            weighted[normalized] = (term, weight)
        ordered = sorted(weighted.values(), key=lambda item: item[1], reverse=True)
        return ordered[:limit]

    @staticmethod
    def _entity_id(normalized: str) -> str:
        digest = hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"entity:{digest[:24]}"

    def build(self) -> str:
        documents = self.database.list_documents()
        chunks = self.database.list_chunks()
        if not chunks:
            return "知识库中还没有 PDF，暂时不能构建 GraphRAG。"

        entities: dict[str, dict] = {}
        entity_documents: dict[str, set[str]] = defaultdict(set)
        mentions: list[dict] = []
        relation_weights: dict[tuple[str, str], float] = defaultdict(float)
        relation_counts: dict[tuple[str, str], int] = defaultdict(int)

        for chunk in chunks:
            terms = self._extract_terms(chunk.text)
            chunk_entities: list[tuple[str, float]] = []
            for term, weight in terms:
                normalized = self._normalize(term)
                entity_id = self._entity_id(normalized)
                record = entities.setdefault(
                    entity_id,
                    {
                        "id": entity_id,
                        "name": term,
                        "normalized": normalized,
                        "kind": self._entity_kind(term),
                        "frequency": 0.0,
                        "document_count": 0,
                    },
                )
                record["frequency"] += weight
                entity_documents[entity_id].add(chunk.document_id)
                mentions.append(
                    {
                        "entity_id": entity_id,
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "weight": weight,
                    }
                )
                chunk_entities.append((entity_id, weight))

            for (left_id, left_weight), (right_id, right_weight) in combinations(
                chunk_entities[:8], 2
            ):
                source_id, target_id = sorted((left_id, right_id))
                key = (source_id, target_id)
                relation_weights[key] += (left_weight + right_weight) / 2
                relation_counts[key] += 1

        for entity_id, document_ids in entity_documents.items():
            entities[entity_id]["document_count"] = len(document_ids)
        relations = [
            {
                "source_id": source_id,
                "target_id": target_id,
                "weight": relation_weights[(source_id, target_id)],
                "evidence_count": relation_counts[(source_id, target_id)],
            }
            for source_id, target_id in relation_weights
        ]
        signature_material = "|".join(f"{item['id']}:{item['chunk_count']}" for item in documents)
        signature = hashlib.sha256(signature_material.encode("utf-8")).hexdigest()
        self.database.replace_graph_index(
            entities=list(entities.values()),
            mentions=mentions,
            relations=relations,
            source_signature=signature,
            document_count=len(documents),
            chunk_count=len(chunks),
        )
        return (
            f"GraphRAG 已构建：{len(entities)} 个实体、{len(relations)} 条关系、"
            f"{len(mentions)} 条片段引用，覆盖 {len(documents)} 份 PDF。"
        )

    def status(self) -> dict:
        status = self.database.graph_status()
        stats = self.database.stats()
        status["stale"] = bool(
            status["ready"]
            and (
                int(status["document_count"]) != int(stats["documents"])
                or int(status["chunk_count"]) != int(stats["chunks"])
            )
        )
        return status

    def status_markdown(self) -> str:
        status = self.status()
        if not status["ready"]:
            return "**状态：尚未构建**\n\n上传 PDF 后点击“构建 / 重建 GraphRAG”。"
        stale_notice = "\n\n> 检测到 PDF 已变化，请重新构建图谱。" if status["stale"] else ""
        return (
            "**状态：本地 GraphRAG 已就绪**\n\n"
            f"- 实体：{status['entity_count']}\n"
            f"- 关系：{status['relation_count']}\n"
            f"- 片段引用：{status['mention_count']}\n"
            f"- 覆盖：{status['document_count']} 份 PDF、{status['chunk_count']} 个片段\n"
            f"- 最近构建：{status['built_at']}"
            f"{stale_notice}"
        )

    def search(
        self,
        query: str,
        document_ids: Sequence[str] | None = None,
        limit: int = 6,
    ) -> list[GraphEvidence]:
        query = (query or "").strip()
        if not query or not self.status()["ready"]:
            return []
        entities = self.database.list_graph_entities(limit=5000)
        if not entities:
            return []
        query_terms = {self._normalize(term) for term, _ in self._extract_terms(query, limit=10)}
        query_terms.update(self._normalize(token) for token in tokenize(query) if len(token) >= 2)
        query_normalized = self._normalize(query)

        seed_scores: dict[str, float] = {}
        for entity in entities:
            normalized = str(entity["normalized"])
            score = 0.0
            if normalized and normalized in query_normalized:
                score += 2.0
            if query_normalized and query_normalized in normalized:
                score += 1.2
            score += sum(0.7 for term in query_terms if term and term in normalized)
            if score:
                frequency_bonus = min(0.5, math.log1p(float(entity["frequency"])) / 10)
                seed_scores[str(entity["id"])] = score + frequency_bonus
        if not seed_scores:
            for entity in entities[:3]:
                seed_scores[str(entity["id"])] = 0.25

        seed_ids = [
            item[0] for item in sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        ]
        relations = self.database.graph_neighbors(seed_ids, limit=120)
        max_edge = max((float(item["weight"]) for item in relations), default=1.0)
        entity_scores = dict(seed_scores)
        entity_names = {str(item["id"]): str(item["name"]) for item in entities}
        for relation in relations:
            source_id = str(relation["source_id"])
            target_id = str(relation["target_id"])
            edge_score = float(relation["weight"]) / max_edge
            if source_id in seed_scores:
                entity_scores[target_id] = max(
                    entity_scores.get(target_id, 0.0), seed_scores[source_id] * edge_score * 0.65
                )
            if target_id in seed_scores:
                entity_scores[source_id] = max(
                    entity_scores.get(source_id, 0.0), seed_scores[target_id] * edge_score * 0.65
                )

        selected_ids = [
            entity_id
            for entity_id, _ in sorted(
                entity_scores.items(), key=lambda item: item[1], reverse=True
            )[:16]
        ]
        mentions = self.database.graph_mentions(selected_ids, document_ids, limit=180)
        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_entities: dict[str, set[str]] = defaultdict(set)
        for mention in mentions:
            entity_id = str(mention["entity_id"])
            chunk_id = str(mention["chunk_id"])
            score = entity_scores.get(entity_id, 0.0) * float(mention["weight"])
            chunk_scores[chunk_id] += score
            chunk_entities[chunk_id].add(entity_names.get(entity_id, str(mention["entity_name"])))
        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        if not ranked:
            return []
        maximum = ranked[0][1] or 1.0
        chunks = self.database.get_chunks_by_ids([chunk_id for chunk_id, _ in ranked])
        scores = {chunk_id: score for chunk_id, score in ranked}
        return [
            GraphEvidence(
                chunk=chunk,
                score=min(1.0, scores[chunk.id] / maximum),
                entities=tuple(sorted(chunk_entities[chunk.id])[:8]),
            )
            for chunk in chunks
        ]

    def search_markdown(self, query: str, document_ids: Sequence[str] | None = None) -> str:
        if not (query or "").strip():
            return "请输入要进行图谱检索的问题。"
        if not self.status()["ready"]:
            return "GraphRAG 尚未构建，请先点击“构建 / 重建 GraphRAG”。"
        results = self.search(query, document_ids, limit=8)
        if not results:
            return "图谱中没有找到相关实体和 PDF 片段。"
        sections = [f"## GraphRAG：{query}"]
        for index, result in enumerate(results, start=1):
            entities = " → ".join(result.entities) or "相关概念"
            excerpt = result.chunk.text[:650] + ("…" if len(result.chunk.text) > 650 else "")
            sections.append(
                f"### G{index} · 《{result.chunk.filename}》第 {result.chunk.page} 页\n\n"
                f"图谱路径：{entities}\n\n相关度：**{result.score:.3f}**\n\n{excerpt}"
            )
        return "\n\n---\n\n".join(sections)

    def graph_html(self) -> str:
        entities = self.database.list_graph_entities(limit=28)
        if not entities:
            return '<div class="empty-graph">构建 GraphRAG 后可在这里查看 PDF 概念关系图。</div>'
        ids = [str(item["id"]) for item in entities]
        id_set = set(ids)
        relations = [
            item
            for item in self.database.graph_neighbors(ids, limit=100)
            if item["source_id"] in id_set and item["target_id"] in id_set
        ]
        center = (450.0, 300.0)
        positions = {
            entity_id: (
                center[0] + 230 * math.cos(-math.pi / 2 + 2 * math.pi * index / len(ids)),
                center[1] + 230 * math.sin(-math.pi / 2 + 2 * math.pi * index / len(ids)),
            )
            for index, entity_id in enumerate(ids)
        }
        maximum = max(float(item["frequency"]) for item in entities) or 1.0
        svg = [
            '<div class="memory-graph-wrap graph-rag-visual">',
            (
                '<div class="memory-graph-legend"><span class="l3">PDF 实体</span>'
                '<span class="l2">共现关系</span></div>'
            ),
            '<svg viewBox="0 0 900 600" role="img" aria-label="PDF GraphRAG 概念图">',
        ]
        for relation in relations:
            source = positions[str(relation["source_id"])]
            target = positions[str(relation["target_id"])]
            svg.append(
                f'<line x1="{source[0]:.1f}" y1="{source[1]:.1f}" '
                f'x2="{target[0]:.1f}" y2="{target[1]:.1f}" class="memory-edge"/>'
            )
        for item in entities:
            x, y = positions[str(item["id"])]
            radius = 8 + 12 * math.sqrt(float(item["frequency"]) / maximum)
            name = str(item["name"])
            svg.append(
                f'<g class="memory-node"><title>{html.escape(name)}</title>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#ef762f"/>'
                f'<text x="{x:.1f}" y="{y + radius + 14:.1f}" text-anchor="middle">'
                f"{html.escape(name[:12])}</text></g>"
            )
        svg.extend(["</svg>", "</div>"])
        return "".join(svg)
