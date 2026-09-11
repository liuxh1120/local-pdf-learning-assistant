from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import DocumentChunk


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    folder_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_folders (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    content_kind TEXT NOT NULL DEFAULT 'text',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    image_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '新对话',
                    updated_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    rewritten_question TEXT NOT NULL DEFAULT '',
                    course_key TEXT NOT NULL DEFAULT '__general__',
                    course_name TEXT NOT NULL DEFAULT '通用学习',
                    document_ids TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    source TEXT NOT NULL,
                    folder_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    folder_ids TEXT NOT NULL DEFAULT '[]',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    indexed INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memories_indexed ON learning_memories(indexed);
                CREATE INDEX IF NOT EXISTS idx_memories_session ON learning_memories(session_id);

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    summarized_through_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    courses TEXT NOT NULL DEFAULT '',
                    weak_points TEXT NOT NULL DEFAULT '',
                    explanation_depth TEXT NOT NULL DEFAULT '标准',
                    math_style TEXT NOT NULL DEFAULT '公式推导优先',
                    preferences TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS course_profiles (
                    course_key TEXT PRIMARY KEY,
                    course_name TEXT NOT NULL,
                    courses TEXT NOT NULL DEFAULT '',
                    weak_points TEXT NOT NULL DEFAULT '',
                    explanation_depth TEXT NOT NULL DEFAULT '标准',
                    math_style TEXT NOT NULL DEFAULT '公式推导优先',
                    preferences TEXT NOT NULL DEFAULT '',
                    evidence_summary TEXT NOT NULL DEFAULT '',
                    question_count INTEGER NOT NULL DEFAULT 0,
                    last_history_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    frequency REAL NOT NULL DEFAULT 0,
                    document_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_graph_entities_normalized
                    ON graph_entities(normalized);

                CREATE TABLE IF NOT EXISTS graph_mentions (
                    entity_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    weight REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(entity_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_graph_mentions_chunk ON graph_mentions(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_graph_mentions_document
                    ON graph_mentions(document_id);

                CREATE TABLE IF NOT EXISTS graph_relations (
                    source_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
                    weight REAL NOT NULL DEFAULT 0,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source_id, target_id)
                );

                CREATE TABLE IF NOT EXISTS graph_index_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    built_at TEXT NOT NULL DEFAULT '',
                    document_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    source_signature TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self._ensure_column(db, "sessions", "title", "TEXT NOT NULL DEFAULT '新对话'")
            self._ensure_column(db, "sessions", "updated_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "qa_history", "model", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "qa_history", "rewritten_question", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                db, "qa_history", "course_key", "TEXT NOT NULL DEFAULT '__general__'"
            )
            self._ensure_column(db, "qa_history", "course_name", "TEXT NOT NULL DEFAULT '通用学习'")
            self._ensure_column(db, "documents", "folder_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "notes", "folder_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "chunks", "content_kind", "TEXT NOT NULL DEFAULT 'text'")
            self._ensure_column(db, "chunks", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(db, "chunks", "image_path", "TEXT NOT NULL DEFAULT ''")
            db.execute(
                "UPDATE sessions SET updated_at = started_at WHERE updated_at = '' OR updated_at IS NULL"
            )
            db.execute(
                """
                INSERT OR IGNORE INTO learning_profile(id, updated_at)
                VALUES (1, ?)
                """,
                (utc_now(),),
            )
            legacy = db.execute("SELECT * FROM learning_profile WHERE id = 1").fetchone()
            db.execute(
                """
                INSERT OR IGNORE INTO course_profiles(
                    course_key, course_name, courses, weak_points, explanation_depth,
                    math_style, preferences, updated_at
                ) VALUES ('__general__', '通用学习', ?, ?, ?, ?, ?, ?)
                """,
                (
                    legacy["courses"],
                    legacy["weak_points"],
                    legacy["explanation_depth"],
                    legacy["math_style"],
                    legacy["preferences"],
                    legacy["updated_at"] or utc_now(),
                ),
            )
            self._backfill_memories(db)

    @staticmethod
    def _ensure_column(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _backfill_memories(db: sqlite3.Connection) -> None:
        documents = {
            row["id"]: row["folder_id"]
            for row in db.execute("SELECT id, folder_id FROM documents").fetchall()
        }
        for row in db.execute("SELECT * FROM qa_history ORDER BY id").fetchall():
            try:
                document_ids = json.loads(row["document_ids"] or "[]")
            except json.JSONDecodeError:
                document_ids = []
            folder_ids = sorted(
                {documents[doc_id] for doc_id in document_ids if doc_id in documents}
            )
            db.execute(
                """
                INSERT OR IGNORE INTO learning_memories(
                    id, kind, source_id, session_id, folder_ids, title, content,
                    created_at, indexed
                ) VALUES (?, 'qa', ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    f"qa:{row['id']}",
                    row["id"],
                    row["session_id"],
                    json.dumps(folder_ids, ensure_ascii=False),
                    row["question"],
                    f"问题：{row['question']}\n回答：{row['answer']}",
                    row["created_at"],
                ),
            )
        for row in db.execute("SELECT * FROM notes ORDER BY id").fetchall():
            folder_ids = [row["folder_id"]] if row["folder_id"] or row["folder_id"] == "" else []
            db.execute(
                """
                INSERT OR IGNORE INTO learning_memories(
                    id, kind, source_id, session_id, folder_ids, title, content,
                    created_at, indexed
                ) VALUES (?, 'note', ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    f"note:{row['id']}",
                    row["id"],
                    row["session_id"],
                    json.dumps(folder_ids, ensure_ascii=False),
                    row["concept"],
                    f"知识点：{row['concept']}\n笔记：{row['content']}\n来源：{row['source']}",
                    row["created_at"],
                ),
            )

    def start_session(self, session_id: str, title: str = "新对话") -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO sessions(id, started_at, title, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, now, title, now),
            )

    def list_sessions(self, current_session_id: str, limit: int = 40) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT s.id, s.started_at, s.title, s.updated_at, COUNT(q.id) AS question_count
                FROM sessions AS s
                LEFT JOIN qa_history AS q ON q.session_id = s.id
                GROUP BY s.id
                HAVING COUNT(q.id) > 0 OR s.id = ?
                ORDER BY s.updated_at DESC, s.started_at DESC
                LIMIT ?
                """,
                (current_session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_session_title_from_question(self, session_id: str, question: str) -> None:
        title = " ".join(question.split()).strip()
        if len(title) > 24:
            title = title[:24].rstrip() + "…"
        if not title:
            return
        with self.connect() as db:
            db.execute(
                """
                UPDATE sessions
                SET title = CASE WHEN title = '' OR title = '新对话' THEN ? ELSE title END,
                    updated_at = ?
                WHERE id = ?
                """,
                (title, utc_now(), session_id),
            )

    def rename_session(self, session_id: str, title: str) -> bool:
        normalized = " ".join(title.split()).strip()
        if not normalized:
            return False
        if len(normalized) > 60:
            normalized = normalized[:60].rstrip()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, utc_now(), session_id),
            )
        return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not exists:
                return False
            for table in (
                "session_summaries",
                "learning_memories",
                "events",
                "notes",
                "qa_history",
            ):
                db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    def get_document_by_hash(self, sha256: str):
        with self.connect() as db:
            return db.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()

    def create_folder(self, name: str) -> str:
        folder_id = f"folder_{uuid.uuid4().hex}"
        with self.connect() as db:
            db.execute(
                "INSERT INTO document_folders(id, name, created_at) VALUES (?, ?, ?)",
                (folder_id, name, utc_now()),
            )
        return folder_id

    def list_folders(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM document_folders ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    def folder_exists(self, folder_id: str) -> bool:
        if not folder_id:
            return True
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM document_folders WHERE id = ?", (folder_id,)).fetchone()
        return row is not None

    def set_documents_folder(self, document_ids: Sequence[str], folder_id: str) -> int:
        if not document_ids:
            return 0
        placeholders = ",".join("?" for _ in document_ids)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE documents SET folder_id = ? WHERE id IN ({placeholders})",
                (folder_id, *document_ids),
            )
        return cursor.rowcount

    def add_document(
        self,
        *,
        document_id: str,
        filename: str,
        title: str,
        stored_path: str,
        sha256: str,
        page_count: int,
        chunks: Sequence[DocumentChunk],
        folder_id: str = "",
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO documents(
                    id, filename, title, stored_path, sha256, page_count, chunk_count,
                    folder_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    title,
                    stored_path,
                    sha256,
                    page_count,
                    len(chunks),
                    folder_id,
                    utc_now(),
                ),
            )
            db.executemany(
                """
                INSERT INTO chunks(
                    id, document_id, filename, page, heading, text, char_count,
                    content_kind, metadata_json, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.filename,
                        chunk.page,
                        chunk.heading,
                        chunk.text,
                        chunk.char_count,
                        chunk.content_kind,
                        chunk.metadata_json,
                        chunk.image_path,
                    )
                    for chunk in chunks
                ],
            )

    def list_documents(self, folder_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            query = """
                SELECT d.*, COALESCE(f.name, '') AS folder_name
                FROM documents AS d
                LEFT JOIN document_folders AS f ON f.id = d.folder_id
            """
            parameters: tuple[str, ...] = ()
            if folder_id is not None:
                query += " WHERE d.folder_id = ?"
                parameters = (folder_id,)
            query += """
                ORDER BY COALESCE(f.name, '') COLLATE NOCASE, d.created_at DESC
            """
            rows = db.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, document_ids: Sequence[str] | None = None) -> list[DocumentChunk]:
        with self.connect() as db:
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                rows = db.execute(
                    f"SELECT * FROM chunks WHERE document_id IN ({placeholders}) ORDER BY rowid",
                    tuple(document_ids),
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [
            DocumentChunk(
                id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                page=row["page"],
                heading=row["heading"],
                text=row["text"],
                char_count=row["char_count"],
                content_kind=row["content_kind"],
                metadata_json=row["metadata_json"],
                image_path=row["image_path"],
            )
            for row in rows
        ]

    def add_qa(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        mode: str,
        model: str,
        document_ids: Sequence[str],
        sources: Sequence[dict],
        rewritten_question: str = "",
        course_key: str = "__general__",
        course_name: str = "通用学习",
    ) -> int:
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT INTO qa_history(
                    session_id, question, answer, mode, model, rewritten_question,
                    course_key, course_name, document_ids, sources, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    mode,
                    model,
                    rewritten_question,
                    course_key,
                    course_name,
                    json.dumps(list(document_ids), ensure_ascii=False),
                    json.dumps(list(sources), ensure_ascii=False),
                    utc_now(),
                ),
            )
            history_id = int(cursor.lastrowid)
            folder_ids: list[str] = []
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                folder_ids = [
                    str(row[0])
                    for row in db.execute(
                        f"SELECT DISTINCT folder_id FROM documents WHERE id IN ({placeholders})",
                        tuple(document_ids),
                    ).fetchall()
                ]
            created_at = utc_now()
            db.execute(
                """
                INSERT INTO learning_memories(
                    id, kind, source_id, session_id, folder_ids, title, content,
                    created_at, indexed
                ) VALUES (?, 'qa', ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    f"qa:{history_id}",
                    history_id,
                    session_id,
                    json.dumps(sorted(set(folder_ids)), ensure_ascii=False),
                    question,
                    f"问题：{question}\n回答：{answer}",
                    created_at,
                ),
            )
            db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (utc_now(), session_id),
            )
        return history_id

    def add_note(
        self,
        session_id: str,
        content: str,
        concept: str,
        source: str,
        folder_id: str = "",
    ) -> int:
        with self.connect() as db:
            cursor = db.execute(
                """
                INSERT INTO notes(session_id, content, concept, source, folder_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, content, concept, source, folder_id, utc_now()),
            )
            note_id = int(cursor.lastrowid)
            created_at = utc_now()
            db.execute(
                """
                INSERT INTO learning_memories(
                    id, kind, source_id, session_id, folder_ids, title, content,
                    created_at, indexed
                ) VALUES (?, 'note', ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    f"note:{note_id}",
                    note_id,
                    session_id,
                    json.dumps([folder_id], ensure_ascii=False),
                    concept,
                    f"知识点：{concept}\n笔记：{content}\n来源：{source}",
                    created_at,
                ),
            )
        return note_id

    def list_notes(self, limit: int = 100, folder_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            if folder_id is None:
                rows = db.execute(
                    "SELECT * FROM notes ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM notes WHERE folder_id = ? ORDER BY created_at DESC LIMIT ?",
                    (folder_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_history(
        self,
        limit: int = 50,
        session_id: str | None = None,
        *,
        chronological: bool = False,
        course_key: str | None = None,
    ) -> list[dict]:
        direction = "ASC" if chronological else "DESC"
        with self.connect() as db:
            if session_id and course_key is not None:
                rows = db.execute(
                    f"SELECT * FROM qa_history WHERE session_id = ? AND course_key = ? "
                    f"ORDER BY id {direction} LIMIT ?",
                    (session_id, course_key, limit),
                ).fetchall()
            elif session_id:
                rows = db.execute(
                    f"SELECT * FROM qa_history WHERE session_id = ? ORDER BY id {direction} LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            elif course_key is not None:
                rows = db.execute(
                    f"SELECT * FROM qa_history WHERE course_key = ? ORDER BY id {direction} LIMIT ?",
                    (course_key, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    f"SELECT * FROM qa_history ORDER BY id {direction} LIMIT ?", (limit,)
                ).fetchall()
        return [dict(row) for row in rows]

    def list_unindexed_memories(self, limit: int = 500) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM learning_memories WHERE indexed = 0 ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
        records = [dict(row) for row in rows]
        for record in records:
            try:
                record["folder_ids"] = json.loads(record["folder_ids"] or "[]")
            except json.JSONDecodeError:
                record["folder_ids"] = []
        return records

    def mark_memories_indexed(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self.connect() as db:
            db.execute(
                f"UPDATE learning_memories SET indexed = 1 WHERE id IN ({placeholders})",
                tuple(memory_ids),
            )

    def reset_memory_index(self) -> None:
        with self.connect() as db:
            db.execute("UPDATE learning_memories SET indexed = 0")

    def get_session_summary(self, session_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return (
            dict(row)
            if row
            else {
                "session_id": session_id,
                "summary": "",
                "summarized_through_id": 0,
                "updated_at": "",
            }
        )

    def list_session_summaries(self, limit: int = 50) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT ss.*, COALESCE(s.title, '对话摘要') AS session_title
                FROM session_summaries AS ss
                LEFT JOIN sessions AS s ON s.id = ss.session_id
                ORDER BY ss.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_session_summary(
        self, session_id: str, summary: str, summarized_through_id: int
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO session_summaries(
                    session_id, summary, summarized_through_id, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    summarized_through_id = excluded.summarized_through_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary, summarized_through_id, utc_now()),
            )

    def get_learning_profile(
        self,
        course_key: str = "__general__",
        course_name: str = "通用学习",
    ) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM course_profiles WHERE course_key = ?", (course_key,)
            ).fetchone()
            if row is None:
                db.execute(
                    """
                    INSERT INTO course_profiles(course_key, course_name, courses, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        course_key,
                        course_name,
                        course_name if course_key != "__general__" else "",
                        utc_now(),
                    ),
                )
                row = db.execute(
                    "SELECT * FROM course_profiles WHERE course_key = ?", (course_key,)
                ).fetchone()
        return dict(row)

    def list_learning_profiles(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM course_profiles
                ORDER BY CASE WHEN course_key = '__general__' THEN 0 ELSE 1 END,
                         course_name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def save_learning_profile(
        self,
        *,
        courses: str,
        weak_points: str,
        explanation_depth: str,
        math_style: str,
        preferences: str,
        course_key: str = "__general__",
        course_name: str = "通用学习",
        evidence_summary: str = "",
        question_count: int | None = None,
        last_history_id: int | None = None,
    ) -> None:
        with self.connect() as db:
            current = db.execute(
                "SELECT question_count, last_history_id FROM course_profiles WHERE course_key = ?",
                (course_key,),
            ).fetchone()
            resolved_question_count = (
                int(current["question_count"])
                if question_count is None and current
                else question_count or 0
            )
            resolved_last_history_id = (
                int(current["last_history_id"])
                if last_history_id is None and current
                else last_history_id or 0
            )
            db.execute(
                """
                INSERT INTO course_profiles(
                    course_key, course_name, courses, weak_points, explanation_depth,
                    math_style, preferences, evidence_summary, question_count,
                    last_history_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(course_key) DO UPDATE SET
                    course_name = excluded.course_name,
                    courses = excluded.courses,
                    weak_points = excluded.weak_points,
                    explanation_depth = excluded.explanation_depth,
                    math_style = excluded.math_style,
                    preferences = excluded.preferences,
                    evidence_summary = CASE
                        WHEN excluded.evidence_summary = '' THEN course_profiles.evidence_summary
                        ELSE excluded.evidence_summary END,
                    question_count = excluded.question_count,
                    last_history_id = excluded.last_history_id,
                    updated_at = excluded.updated_at
                """,
                (
                    course_key,
                    course_name,
                    courses,
                    weak_points,
                    explanation_depth,
                    math_style,
                    preferences,
                    evidence_summary,
                    resolved_question_count,
                    resolved_last_history_id,
                    utc_now(),
                ),
            )
            if course_key == "__general__":
                db.execute(
                    """
                    UPDATE learning_profile
                    SET courses = ?, weak_points = ?, explanation_depth = ?,
                        math_style = ?, preferences = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (
                        courses,
                        weak_points,
                        explanation_depth,
                        math_style,
                        preferences,
                        utc_now(),
                    ),
                )

    def add_event(self, session_id: str, event_type: str, payload: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (session_id, event_type, json.dumps(payload, ensure_ascii=False), utc_now()),
            )

    def list_events(self, limit: int = 100, event_type: str | None = None) -> list[dict]:
        with self.connect() as db:
            if event_type:
                rows = db.execute(
                    "SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        records = [dict(row) for row in rows]
        for record in records:
            try:
                record["payload_data"] = json.loads(record["payload"] or "{}")
            except json.JSONDecodeError:
                record["payload_data"] = {}
        return records

    def replace_graph_index(
        self,
        *,
        entities: Sequence[dict],
        mentions: Sequence[dict],
        relations: Sequence[dict],
        source_signature: str,
        document_count: int,
        chunk_count: int,
    ) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM graph_relations")
            db.execute("DELETE FROM graph_mentions")
            db.execute("DELETE FROM graph_entities")
            db.executemany(
                """
                INSERT INTO graph_entities(
                    id, name, normalized, kind, frequency, document_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["id"],
                        item["name"],
                        item["normalized"],
                        item["kind"],
                        float(item["frequency"]),
                        int(item["document_count"]),
                    )
                    for item in entities
                ],
            )
            db.executemany(
                """
                INSERT INTO graph_mentions(entity_id, chunk_id, document_id, weight)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        item["entity_id"],
                        item["chunk_id"],
                        item["document_id"],
                        float(item["weight"]),
                    )
                    for item in mentions
                ],
            )
            db.executemany(
                """
                INSERT INTO graph_relations(
                    source_id, target_id, weight, evidence_count
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        item["source_id"],
                        item["target_id"],
                        float(item["weight"]),
                        int(item["evidence_count"]),
                    )
                    for item in relations
                ],
            )
            db.execute(
                """
                INSERT INTO graph_index_state(
                    id, built_at, document_count, chunk_count, source_signature
                ) VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    built_at = excluded.built_at,
                    document_count = excluded.document_count,
                    chunk_count = excluded.chunk_count,
                    source_signature = excluded.source_signature
                """,
                (utc_now(), document_count, chunk_count, source_signature),
            )

    def graph_status(self) -> dict:
        with self.connect() as db:
            state = db.execute("SELECT * FROM graph_index_state WHERE id = 1").fetchone()
            entity_count = int(db.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0])
            relation_count = int(db.execute("SELECT COUNT(*) FROM graph_relations").fetchone()[0])
            mention_count = int(db.execute("SELECT COUNT(*) FROM graph_mentions").fetchone()[0])
        result = (
            dict(state)
            if state
            else {
                "built_at": "",
                "document_count": 0,
                "chunk_count": 0,
                "source_signature": "",
            }
        )
        result.update(
            {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "mention_count": mention_count,
                "ready": bool(state and entity_count),
            }
        )
        return result

    def list_graph_entities(self, limit: int = 5000) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM graph_entities
                ORDER BY frequency DESC, document_count DESC, name COLLATE NOCASE
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_neighbors(self, entity_ids: Sequence[str], limit: int = 100) -> list[dict]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT r.*, source.name AS source_name, target.name AS target_name
                FROM graph_relations AS r
                JOIN graph_entities AS source ON source.id = r.source_id
                JOIN graph_entities AS target ON target.id = r.target_id
                WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                ORDER BY r.weight DESC, r.evidence_count DESC
                LIMIT ?
                """,
                (*entity_ids, *entity_ids, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_mentions(
        self,
        entity_ids: Sequence[str],
        document_ids: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        if not entity_ids:
            return []
        entity_placeholders = ",".join("?" for _ in entity_ids)
        parameters: list[str | int] = [*entity_ids]
        document_filter = ""
        if document_ids:
            document_placeholders = ",".join("?" for _ in document_ids)
            document_filter = f" AND m.document_id IN ({document_placeholders})"
            parameters.extend(document_ids)
        parameters.append(limit)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT m.*, e.name AS entity_name, e.frequency AS entity_frequency
                FROM graph_mentions AS m
                JOIN graph_entities AS e ON e.id = m.entity_id
                WHERE m.entity_id IN ({entity_placeholders}){document_filter}
                ORDER BY m.weight DESC, e.frequency DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
        by_id = {
            row["id"]: DocumentChunk(
                id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                page=row["page"],
                heading=row["heading"],
                text=row["text"],
                char_count=row["char_count"],
                content_kind=row["content_kind"],
                metadata_json=row["metadata_json"],
                image_path=row["image_path"],
            )
            for row in rows
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def list_memories(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT lm.*, q.course_key AS qa_course_key, q.course_name AS qa_course_name,
                       n.folder_id AS note_folder_id
                FROM learning_memories AS lm
                LEFT JOIN qa_history AS q
                    ON lm.kind = 'qa' AND lm.source_id = q.id
                LEFT JOIN notes AS n
                    ON lm.kind = 'note' AND lm.source_id = n.id
                ORDER BY lm.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = [dict(row) for row in rows]
        for record in records:
            try:
                record["folder_ids"] = json.loads(record["folder_ids"] or "[]")
            except json.JSONDecodeError:
                record["folder_ids"] = []
        return records

    def memory_overview(self) -> dict[str, int]:
        with self.connect() as db:
            l1 = int(db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            l2 = int(db.execute("SELECT COUNT(*) FROM learning_memories").fetchone()[0])
            profiles = int(
                db.execute(
                    """
                    SELECT COUNT(*) FROM course_profiles
                    WHERE question_count > 0 OR weak_points != '' OR preferences != ''
                    """
                ).fetchone()[0]
            )
            summaries = int(db.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0])
        return {"l1": l1, "l2": l2, "l3": profiles + summaries}

    def stats(self) -> dict:
        with self.connect() as db:
            documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            pages = db.execute("SELECT COALESCE(SUM(page_count), 0) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            questions = db.execute("SELECT COUNT(*) FROM qa_history").fetchone()[0]
            notes = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            sessions = db.execute("SELECT COUNT(DISTINCT session_id) FROM qa_history").fetchone()[0]
            folders = db.execute("SELECT COUNT(*) FROM document_folders").fetchone()[0]
            memories = db.execute("SELECT COUNT(*) FROM learning_memories").fetchone()[0]
            summaries = db.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
        return {
            "documents": documents,
            "pages": pages,
            "chunks": chunks,
            "questions": questions,
            "notes": notes,
            "sessions": sessions,
            "folders": folders,
            "memories": memories,
            "summaries": summaries,
        }
