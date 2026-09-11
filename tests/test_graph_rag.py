from pathlib import Path

from pdf_assistant.database import Database
from pdf_assistant.graph_rag import LocalGraphRAG
from pdf_assistant.models import DocumentChunk


def test_local_graph_rag_builds_and_retrieves_pdf_evidence(tmp_path: Path):
    database = Database(tmp_path / "graph.sqlite3")
    chunks = [
        DocumentChunk(
            id="graph-chunk-1",
            document_id="graph-doc",
            filename="数理统计.pdf",
            page=12,
            heading="最大似然估计",
            text="最大似然估计的渐近方差由 Fisher 信息矩阵的逆给出。",
            char_count=28,
        ),
        DocumentChunk(
            id="graph-chunk-2",
            document_id="graph-doc",
            filename="数理统计.pdf",
            page=13,
            heading="渐近正态性",
            text="在正则条件下，最大似然估计量具有渐近正态性。",
            char_count=24,
        ),
    ]
    database.add_document(
        document_id="graph-doc",
        filename="数理统计.pdf",
        title="数理统计",
        stored_path="/tmp/statistics.pdf",
        sha256="graph-hash",
        page_count=20,
        chunks=chunks,
    )
    graph = LocalGraphRAG(database)

    message = graph.build()
    status = graph.status()
    results = graph.search("最大似然估计和 Fisher 信息有什么关系？", ["graph-doc"])

    assert "GraphRAG 已构建" in message
    assert status["ready"] is True
    assert status["entity_count"] > 0
    assert status["relation_count"] > 0
    assert results
    assert results[0].chunk.filename == "数理统计.pdf"
    assert "Fisher" in results[0].chunk.text
    assert "PDF GraphRAG 概念图" in graph.graph_html()
