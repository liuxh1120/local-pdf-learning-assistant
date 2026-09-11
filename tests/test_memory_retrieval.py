from pathlib import Path

import numpy as np

from pdf_assistant.database import Database
from pdf_assistant.retrieval import Retriever


class FakeEmbedder:
    dimension = 3

    def encode(self, texts):
        vectors = []
        for text in texts:
            if "中心极限" in text:
                vector = [1.0, 0.0, 0.0]
            elif "回归" in text:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def test_memory_vector_search_respects_session_and_folder(tmp_path: Path):
    database = Database(tmp_path / "memory-search.sqlite3")
    retriever = Retriever(
        database=database,
        vector_path=str(tmp_path / "vectors"),
        embedder=FakeEmbedder(),
    )
    records = [
        {
            "id": "note:1",
            "kind": "note",
            "source_id": 1,
            "session_id": "session-a",
            "folder_ids": ["statistics"],
            "title": "中心极限定理",
            "content": "中心极限定理的学习笔记",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "qa:2",
            "kind": "qa",
            "source_id": 2,
            "session_id": "session-b",
            "folder_ids": ["regression"],
            "title": "线性回归",
            "content": "线性回归的历史问答",
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    ]
    assert retriever.index_memories(records) == ["note:1", "qa:2"]

    current = retriever.search_memories(
        "中心极限定理",
        session_id="session-a",
        include_all_sessions=False,
        folder_id="statistics",
    )
    assert [item.id for item in current] == ["note:1"]

    cross_session = retriever.search_memories(
        "线性回归",
        session_id="session-a",
        include_all_sessions=True,
        folder_id="regression",
    )
    assert [item.id for item in cross_session] == ["qa:2"]
    retriever.client.close()
