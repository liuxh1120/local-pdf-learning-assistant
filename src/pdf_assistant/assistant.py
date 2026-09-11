from __future__ import annotations

import html
import json
import math
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .database import Database
from .embedding import EmbeddingModel
from .graph_rag import LocalGraphRAG
from .llm import DeepSeekLLM, LLMConfigurationError
from .mcp_client import DEFAULT_TOOL_GROUPS, MCPToolClient
from .pdf_parser import PDFParser, file_sha256
from .retrieval import Retriever
from .web_search import SearXNGSearch, WebSearchError

MODE_PDF = "仅 PDF"
MODE_MIXED = "PDF + 互联网"
MODE_WEB = "仅互联网搜索"
ALL_FOLDERS = "__all_folders__"
MEMORY_CURRENT = "current"
MEMORY_ALL = "all"
GENERAL_COURSE = "__general__"
UNCATEGORIZED_COURSE = "folder:__uncategorized__"

MODEL_FLASH = "deepseek-v4-flash"
MODEL_PRO = "deepseek-v4-pro"
MODEL_CHOICES = [
    ("DeepSeek V4 Flash - 日常问答，速度更快", MODEL_FLASH),
    ("DeepSeek V4 Pro - 复杂证明与深入推理", MODEL_PRO),
]
VALID_MODELS = {MODEL_FLASH, MODEL_PRO}


def allowed_tool_groups_for_mode(mode: str, groups: Sequence[str]) -> list[str]:
    selected = list(dict.fromkeys(groups))
    if mode == MODE_PDF:
        return []
    if mode == MODE_WEB:
        return [group for group in selected if group != "files"]
    return selected


class LearningAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = Database(settings.database_path)
        self.session_id = self._new_session_id()
        self.database.start_session(self.session_id)
        self.parser = PDFParser(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_mb=settings.max_pdf_mb,
            max_pages=settings.max_pdf_pages,
            page_assets_dir=settings.page_assets_dir,
            enable_page_ocr=settings.enable_page_ocr,
        )
        self.embedder = EmbeddingModel(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
        self.retriever = Retriever(
            database=self.database,
            vector_path=str(settings.vector_dir),
            embedder=self.embedder,
            enable_reranker=settings.enable_reranker,
            reranker_model=settings.reranker_model,
        )
        self.graph_rag = LocalGraphRAG(self.database)
        self.llm = DeepSeekLLM(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
        )
        self.web_search = SearXNGSearch(settings.searxng_url)
        self.mcp_client = MCPToolClient(settings)
        self.last_context_status = "等待提问。"

    @staticmethod
    def _new_session_id() -> str:
        return f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def new_conversation(self) -> str:
        self.session_id = self._new_session_id()
        self.database.start_session(self.session_id)
        return self.session_id

    def switch_conversation(self, session_id: str) -> list[dict[str, str]]:
        known_ids = {item["id"] for item in self.database.list_sessions(self.session_id, limit=200)}
        if session_id not in known_ids:
            return []
        self.session_id = session_id
        return self.conversation_messages(session_id)

    def conversation_messages(self, session_id: str | None = None) -> list[dict[str, str]]:
        history = self.database.list_history(
            limit=100,
            session_id=session_id or self.session_id,
            chronological=True,
        )
        messages: list[dict[str, str]] = []
        for item in history:
            messages.extend(
                [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            )
        return messages

    def conversation_choices(self, current_session_id: str | None = None) -> list[tuple[str, str]]:
        return [
            (item["title"] or "新对话", item["id"])
            for item in self.database.list_sessions(current_session_id or self.session_id)
        ]

    def rename_conversation(self, session_id: str, title: str) -> str:
        if not session_id:
            return "请先选择一段对话。"
        if not title or not title.strip():
            return "对话标题不能为空。"
        if not self.database.rename_session(session_id, title):
            return "未找到该对话，可能已被删除。"
        return "已更新对话标题。"

    def delete_conversation(self, session_id: str) -> str:
        if not session_id:
            return "请先选择一段对话。"
        self.retriever.delete_session_memories(session_id)
        if not self.database.delete_session(session_id):
            return "未找到该对话，可能已被删除。"
        if self.session_id == session_id:
            self.new_conversation()
        return "已删除对话及其问答、摘要和会话记忆。"

    def selected_model(self, model: str | None) -> str:
        if model in VALID_MODELS:
            return str(model)
        if self.settings.deepseek_model in VALID_MODELS:
            return self.settings.deepseek_model
        return MODEL_FLASH

    def ingest(self, uploaded_paths: Sequence[str | Path], folder_id: str = "") -> dict:
        messages: list[str] = []
        added: list[str] = []
        warnings: list[str] = []
        folder_id = folder_id if self.database.folder_exists(folder_id) else ""
        for raw_path in uploaded_paths:
            source = Path(raw_path)
            original_name = source.name
            stored_path: Path | None = None
            try:
                self.parser.validate(source)
                sha256 = file_sha256(source)
                duplicate = self.database.get_document_by_hash(sha256)
                if duplicate:
                    if folder_id and duplicate["folder_id"] != folder_id:
                        self.database.set_documents_folder([duplicate["id"]], folder_id)
                        messages.append(f"《{original_name}》已存在，已移入所选文件夹。")
                    else:
                        messages.append(f"《{original_name}》已存在，跳过重复导入。")
                    continue

                document_id = str(uuid.uuid4())
                stored_path = self.parser.copy_into_library(
                    source, self.settings.uploads_dir, document_id
                )
                parsed = self.parser.parse(stored_path, document_id, original_name)
                self.retriever.index(parsed.chunks)
                self.database.add_document(
                    document_id=document_id,
                    filename=original_name,
                    title=parsed.title,
                    stored_path=str(stored_path),
                    sha256=sha256,
                    page_count=parsed.page_count,
                    chunks=parsed.chunks,
                    folder_id=folder_id,
                )
                self.database.add_event(
                    self.session_id,
                    "document_loaded",
                    {
                        "document_id": document_id,
                        "filename": original_name,
                        "pages": parsed.page_count,
                        "chunks": len(parsed.chunks),
                    },
                )
                added.append(document_id)
                messages.append(
                    f"《{original_name}》：{parsed.page_count} 页，建立 {len(parsed.chunks)} 个检索片段。"
                )
                warnings.extend(f"《{original_name}》：{warning}" for warning in parsed.warnings)
            except Exception as exc:  # noqa: BLE001 - one bad upload must not abort the batch
                if stored_path and stored_path.exists():
                    stored_path.unlink(missing_ok=True)
                messages.append(f"《{original_name}》导入失败：{exc}")

        return {"messages": messages, "added_ids": added, "warnings": warnings}

    def create_folder(self, name: str) -> tuple[str, str]:
        name = " ".join((name or "").split()).strip()
        if not name:
            return "文件夹名称不能为空。", ""
        if name == "未分类":
            return "“未分类”是系统保留名称，请换一个名称。", ""
        if len(name) > 50:
            return "文件夹名称不能超过 50 个字符。", ""
        try:
            folder_id = self.database.create_folder(name)
        except sqlite3.IntegrityError:
            return f"文件夹“{name}”已存在。", ""
        return f"已创建文件夹“{name}”。", folder_id

    def move_documents(self, document_ids: Sequence[str], folder_id: str) -> str:
        if not document_ids:
            return "请先选择要移动的 PDF。"
        if not self.database.folder_exists(folder_id):
            return "目标文件夹不存在，请刷新后重试。"
        count = self.database.set_documents_folder(list(document_ids), folder_id)
        target = "未分类"
        if folder_id:
            target = next(
                (item["name"] for item in self.database.list_folders() if item["id"] == folder_id),
                "所选文件夹",
            )
        return f"已将 {count} 份 PDF 移入“{target}”。"

    def _resolve_course_context(
        self,
        folder_filter: str | None,
        selected_ids: Sequence[str],
        documents: Sequence[dict],
    ) -> tuple[str, str, str | None]:
        folders = {item["id"]: item["name"] for item in self.database.list_folders()}
        if folder_filter is not None:
            if folder_filter == "":
                return UNCATEGORIZED_COURSE, "未分类资料", ""
            return f"folder:{folder_filter}", folders.get(folder_filter, "所选课程"), folder_filter
        selected_folders = {
            str(item.get("folder_id", "")) for item in documents if item["id"] in selected_ids
        }
        if len(selected_folders) == 1:
            folder_id = selected_folders.pop()
            if folder_id:
                return f"folder:{folder_id}", folders.get(folder_id, "所选课程"), folder_id
            return UNCATEGORIZED_COURSE, "未分类资料", ""
        return GENERAL_COURSE, "通用学习", None

    def _ensure_memory_index(self) -> int:
        if not self.retriever.memory_collection_exists():
            self.database.reset_memory_index()
        total = 0
        while records := self.database.list_unindexed_memories(limit=256):
            indexed_ids = self.retriever.index_memories(records)
            self.database.mark_memories_indexed(indexed_ids)
            total += len(indexed_ids)
            if len(records) < 256:
                break
        return total

    def _maybe_update_summary(self, model: str, *, force: bool = False) -> str:
        history = self.database.list_history(
            limit=500,
            session_id=self.session_id,
            chronological=True,
        )
        if len(history) <= 4:
            return "当前对话较短，尚无需压缩。"
        summary_record = self.database.get_session_summary(self.session_id)
        candidates = [
            item
            for item in history[:-4]
            if int(item["id"]) > int(summary_record["summarized_through_id"])
        ]
        if not force and len(candidates) < 6:
            return "尚未达到自动压缩阈值。"
        if not candidates:
            return "对话摘要已是最新。"
        batch = candidates[:10]
        summary = self.llm.summarize_conversation(
            existing_summary=str(summary_record["summary"]),
            history=batch,
            model=model,
        )
        if not summary:
            return "未能生成对话摘要。"
        self.database.save_session_summary(
            self.session_id,
            summary,
            int(batch[-1]["id"]),
        )
        return f"已压缩 {len(batch)} 轮较早对话。"

    def ask(
        self,
        question: str,
        document_ids: Sequence[str] | None,
        mode: str,
        advanced: bool = False,
        model: str | None = None,
        memory_scope: str = MEMORY_CURRENT,
        folder_context: str = ALL_FOLDERS,
        enable_agent_tools: bool = True,
        tool_groups: Sequence[str] | None = None,
    ) -> str:
        question = (question or "").strip()
        if not question:
            return "请输入问题。"

        folder_filter = None if folder_context == ALL_FOLDERS else folder_context
        documents = self.database.list_documents(folder_filter)
        allowed_ids = {doc["id"] for doc in documents}
        selected_ids = list(document_ids or [])
        selected_ids = [document_id for document_id in selected_ids if document_id in allowed_ids]
        if not selected_ids and mode != MODE_WEB:
            selected_ids = [doc["id"] for doc in documents]
        if mode == MODE_PDF and not selected_ids:
            return "请先在“知识中心”上传至少一份 PDF。"

        course_key, course_name, course_folder_id = self._resolve_course_context(
            folder_filter, selected_ids, documents
        )

        selected_model = self.selected_model(model)
        recent_history = self.database.list_history(limit=4, session_id=self.session_id)
        summary_record = self.database.get_session_summary(self.session_id)
        context_warnings: list[str] = []
        try:
            rewritten_question = self.llm.rewrite_question(
                question,
                recent_history=recent_history,
                conversation_summary=str(summary_record["summary"]),
                model=selected_model,
            )
        except Exception:  # noqa: BLE001 - query rewriting must have a safe fallback
            rewritten_question = question
            context_warnings.append("问题改写失败，已使用原问检索。")

        pdf_chunks = []
        graph_results = []
        web_results = []
        memories = []
        if mode in {MODE_PDF, MODE_MIXED} and selected_ids:
            pdf_chunks = self.retriever.search(
                rewritten_question,
                selected_ids,
                final_limit=7 if advanced else 5,
                candidate_limit=24 if advanced else 16,
                rerank=advanced,
            )
            graph_results = self.graph_rag.search(
                rewritten_question,
                selected_ids,
                limit=4 if advanced else 2,
            )
        if mode in {MODE_MIXED, MODE_WEB}:
            try:
                web_results = self.web_search.search(rewritten_question, limit=5)
            except WebSearchError as exc:
                if mode == MODE_WEB and not enable_agent_tools:
                    return str(exc)
                web_warning = f"\n\n> 联网搜索未启用：{exc}"
            else:
                web_warning = ""
        else:
            web_warning = ""

        try:
            indexed_count = self._ensure_memory_index()
            memories = self.retriever.search_memories(
                rewritten_question,
                session_id=self.session_id,
                include_all_sessions=memory_scope == MEMORY_ALL,
                folder_id=folder_filter,
                limit=7 if advanced else 5,
            )
        except Exception:  # noqa: BLE001 - memory retrieval must not block PDF Q&A
            indexed_count = 0
            memories = []
            context_warnings.append("长期记忆检索暂时不可用。")

        requested_tool_groups = list(
            dict.fromkeys(DEFAULT_TOOL_GROUPS if tool_groups is None else tool_groups)
        )
        selected_tool_groups = allowed_tool_groups_for_mode(mode, requested_tool_groups)
        agent_active = bool(
            enable_agent_tools and self.settings.enable_mcp_tools and selected_tool_groups
        )

        if (
            not pdf_chunks
            and not graph_results
            and not web_results
            and not memories
            and not agent_active
        ):
            return "没有找到足够相关的资料。请换一种问法，或上传包含该内容的 PDF。" + web_warning

        self.mcp_client.last_trace = []
        try:
            answer = self.llm.answer(
                question=rewritten_question,
                original_question=question,
                pdf_chunks=pdf_chunks,
                graph_evidence=graph_results,
                web_results=web_results,
                memories=memories,
                recent_history=recent_history,
                conversation_summary=str(summary_record["summary"]),
                learning_profile=self.database.get_learning_profile(course_key, course_name),
                model=selected_model,
                tool_client=self.mcp_client,
                enable_agent_tools=agent_active,
                tool_groups=selected_tool_groups,
            )
        except LLMConfigurationError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - present a safe UI error
            return str(exc)

        source_records = [
            {
                "type": "pdf",
                "filename": result.chunk.filename,
                "page": result.chunk.page,
                "heading": result.chunk.heading,
                "content_kind": result.chunk.content_kind,
            }
            for result in pdf_chunks
        ] + [{"type": "web", "title": result.title, "url": result.url} for result in web_results]
        source_records.extend(
            {
                "type": "graph",
                "filename": result.chunk.filename,
                "page": result.chunk.page,
                "entities": list(result.entities),
                "score": result.score,
            }
            for result in graph_results
        )
        source_records.extend(
            {
                "type": memory.kind,
                "memory_id": memory.id,
                "title": memory.title,
                "score": memory.score,
            }
            for memory in memories
        )
        source_records.extend(
            {
                "type": "tool",
                "tool": item.get("tool", ""),
                "ok": bool(item.get("ok")),
                "summary": str(item.get("summary", ""))[:500],
            }
            for item in self.mcp_client.last_trace
        )
        history_id = self.database.add_qa(
            session_id=self.session_id,
            question=question,
            answer=answer,
            mode=mode,
            model=selected_model,
            rewritten_question=rewritten_question,
            document_ids=selected_ids,
            sources=source_records,
            course_key=course_key,
            course_name=course_name,
        )
        self.database.add_event(
            self.session_id,
            "question_answered",
            {
                "history_id": history_id,
                "question": question,
                "course_key": course_key,
                "course_name": course_name,
                "pdf_sources": len(pdf_chunks),
                "graph_sources": len(graph_results),
                "web_sources": len(web_results),
                "memory_sources": len(memories),
                "tools": [item.get("tool", "") for item in self.mcp_client.last_trace],
            },
        )
        self.database.set_session_title_from_question(self.session_id, question)
        try:
            newly_indexed = self._ensure_memory_index()
        except Exception:  # noqa: BLE001 - answer is already safely persisted
            newly_indexed = 0
        try:
            summary_status = self._maybe_update_summary(selected_model)
        except Exception:  # noqa: BLE001 - compression is best effort
            summary_status = "本次未更新对话摘要。"
        try:
            profile_status = self._update_course_profile(
                course_key=course_key,
                course_name=course_name,
                folder_id=course_folder_id,
                history_id=history_id,
                model=selected_model,
            )
        except Exception:  # noqa: BLE001 - profiling must never discard an answer
            profile_status = "本轮画像自动更新失败，将在下次继续尝试。"

        folder_name = "全部文件夹"
        if folder_filter is not None:
            folder_name = next(
                (
                    folder["name"]
                    for folder in self.database.list_folders()
                    if folder["id"] == folder_filter
                ),
                "未分类" if folder_filter == "" else "所选文件夹",
            )
        self.last_context_status = (
            "### 本次上下文\n\n"
            f"- 原问：{question}\n"
            f"- 检索问题：{rewritten_question}\n"
            f"- 文件夹：{folder_name}\n"
            f"- 学习画像：{course_name}；{profile_status}\n"
            f"- 记忆范围：{'全部历史对话' if memory_scope == MEMORY_ALL else '仅当前对话'}\n"
            f"- 证据：PDF {len(pdf_chunks)} 条、GraphRAG {len(graph_results)} 条、"
            f"网页 {len(web_results)} 条、记忆/笔记 {len(memories)} 条\n"
            f"- Agent 工具：{'已启用' if agent_active else '未启用'}；"
            f"调用 {len(self.mcp_client.last_trace)} 次\n"
            f"- 记忆索引：本轮补充 {indexed_count + newly_indexed} 条\n"
            f"- 摘要：{summary_status}"
        )
        if self.mcp_client.last_trace:
            tool_names = ", ".join(
                str(item.get("tool", "工具")) for item in self.mcp_client.last_trace
            )
            self.last_context_status += f"\n- 工具轨迹：{tool_names}"
        if context_warnings:
            self.last_context_status += "\n- 提醒：" + "；".join(context_warnings)
        return answer + web_warning

    def add_note(
        self,
        content: str,
        concept: str = "",
        source: str = "",
        folder_id: str = "",
    ) -> str:
        content = (content or "").strip()
        if not content:
            return "笔记内容不能为空。"
        note_id = self.database.add_note(
            self.session_id,
            content,
            (concept or "未分类").strip(),
            (source or "手动笔记").strip(),
            folder_id if self.database.folder_exists(folder_id) else "",
        )
        self.database.add_event(
            self.session_id,
            "note_saved",
            {
                "note_id": note_id,
                "concept": (concept or "未分类").strip(),
                "folder_id": folder_id if self.database.folder_exists(folder_id) else "",
            },
        )
        try:
            self._ensure_memory_index()
        except Exception:  # noqa: BLE001 - note remains safely stored in SQLite
            return "笔记已保存；向量索引将在下次问答时自动补建。"
        return "笔记已保存，并已加入长期记忆检索。"

    def notes_markdown(self) -> str:
        notes = self.database.list_notes()
        if not notes:
            return "暂无笔记。"
        sections = ["## 学习笔记"]
        folder_names = {folder["id"]: folder["name"] for folder in self.database.list_folders()}
        for note in notes:
            folder_name = folder_names.get(note["folder_id"], "未分类")
            sections.append(
                f"### {note['concept']}\n\n{note['content']}\n\n"
                f"来源：{note['source']} · 文件夹：{folder_name} · {note['created_at']}"
            )
        return "\n\n---\n\n".join(sections)

    def history_markdown(self) -> str:
        history = self.database.list_history(limit=30, session_id=self.session_id)
        if not history:
            return "暂无问答历史。"
        sections = ["## 最近问答"]
        sections.extend(
            (
                f"### 问：{item['question']}\n\n{item['answer']}\n\n"
                f"模式：{item['mode']} · {item['created_at']}"
            )
            for item in history
        )
        return "\n\n---\n\n".join(sections)

    def learning_review(self, model: str | None = None) -> str:
        history = self.database.list_history(limit=20, session_id=self.session_id)
        notes = self.database.list_notes(limit=20)
        if not history and not notes:
            return "还没有学习记录。完成几次问答或添加笔记后再生成回顾。"
        material_parts = [
            f"问题：{item['question']}\n回答：{item['answer']}" for item in reversed(history)
        ]
        material_parts.extend(
            f"笔记（{note['concept']}）：{note['content']}" for note in reversed(notes)
        )
        try:
            return self.llm.summarize_learning(
                "\n\n".join(material_parts), model=self.selected_model(model)
            )
        except Exception as exc:  # noqa: BLE001 - present a safe UI error
            return str(exc)

    def compress_current_conversation(self, model: str | None = None) -> str:
        try:
            return self._maybe_update_summary(self.selected_model(model), force=True)
        except Exception as exc:  # noqa: BLE001 - present a safe UI error
            return f"对话压缩失败：{exc}"

    def session_summary_markdown(self) -> str:
        record = self.database.get_session_summary(self.session_id)
        if not record["summary"]:
            return "当前对话暂无压缩摘要。对话变长后会自动生成。"
        return (
            "## 当前对话压缩摘要\n\n"
            f"{record['summary']}\n\n"
            f"已摘要至问答记录 #{record['summarized_through_id']} · "
            f"更新于 {record['updated_at']}"
        )

    def profile_scope_choices(self) -> list[tuple[str, str]]:
        choices = [("通用画像（跨课程）", GENERAL_COURSE)]
        choices.extend(
            (f"课程：{folder['name']}", f"folder:{folder['id']}")
            for folder in self.database.list_folders()
        )
        choices.append(("课程：未分类资料", UNCATEGORIZED_COURSE))
        return choices

    def _profile_scope_details(self, course_key: str) -> tuple[str, str | None]:
        if course_key == UNCATEGORIZED_COURSE:
            return "未分类资料", ""
        if course_key.startswith("folder:"):
            folder_id = course_key.removeprefix("folder:")
            folder_name = next(
                (item["name"] for item in self.database.list_folders() if item["id"] == folder_id),
                "所选课程",
            )
            return folder_name, folder_id
        return "通用学习", None

    def learning_profile_values(
        self, course_key: str = GENERAL_COURSE
    ) -> tuple[str, str, str, str, str]:
        course_name, _ = self._profile_scope_details(course_key)
        profile = self.database.get_learning_profile(course_key, course_name)
        return (
            profile["courses"],
            profile["weak_points"],
            profile["explanation_depth"],
            profile["math_style"],
            profile["preferences"],
        )

    def load_learning_profile(self, course_key: str) -> tuple[str, str, str, str, str, str]:
        course_name, _ = self._profile_scope_details(course_key)
        profile = self.database.get_learning_profile(course_key, course_name)
        evidence = profile.get("evidence_summary") or "暂无自动推断证据。"
        status = (
            f"当前为“{course_name}”独立画像；已参考 {profile.get('question_count', 0)} 条问答。\n\n"
            f"**推断依据：** {evidence}"
        )
        return (*self.learning_profile_values(course_key), status)

    def save_learning_profile(
        self,
        courses: str,
        weak_points: str,
        explanation_depth: str,
        math_style: str,
        preferences: str,
        course_key: str = GENERAL_COURSE,
    ) -> str:
        depth = explanation_depth if explanation_depth in {"简洁", "标准", "深入"} else "标准"
        course_name, _ = self._profile_scope_details(course_key)
        self.database.save_learning_profile(
            courses=(courses or "").strip(),
            weak_points=(weak_points or "").strip(),
            explanation_depth=depth,
            math_style=(math_style or "公式推导优先").strip(),
            preferences=(preferences or "").strip(),
            course_key=course_key,
            course_name=course_name,
        )
        return f"“{course_name}”学习画像已保存，后续该课程的回答会自动参考。"

    def _update_course_profile(
        self,
        *,
        course_key: str,
        course_name: str,
        folder_id: str | None,
        history_id: int,
        model: str,
    ) -> str:
        current = self.database.get_learning_profile(course_key, course_name)
        if history_id <= int(current.get("last_history_id", 0)):
            return "画像已是最新"
        history = self.database.list_history(limit=24, chronological=True, course_key=course_key)
        notes = self.database.list_notes(limit=24, folder_id=folder_id)
        profile = self.llm.derive_learning_profile(
            current_profile=current,
            history=history,
            notes=list(reversed(notes)),
            course_name=course_name,
            model=model,
        )
        profile["courses"] = profile["courses"] or course_name
        self.database.save_learning_profile(
            **profile,
            course_key=course_key,
            course_name=course_name,
            question_count=len(history),
            last_history_id=history_id,
        )
        self.database.add_event(
            self.session_id,
            "profile_updated",
            {
                "course_key": course_key,
                "course_name": course_name,
                "history_id": history_id,
                "evidence_summary": profile.get("evidence_summary", ""),
            },
        )
        return "已根据本轮问题自动更新"

    def auto_update_learning_profile(
        self, course_key: str = GENERAL_COURSE, model: str | None = None
    ) -> tuple:
        course_name, folder_id = self._profile_scope_details(course_key)
        history = self.database.list_history(limit=40, course_key=course_key)
        notes = self.database.list_notes(limit=40, folder_id=folder_id)
        if not history and not notes:
            return (*self.learning_profile_values(course_key), "暂无足够的该课程学习记录。")
        try:
            current = self.database.get_learning_profile(course_key, course_name)
            profile = self.llm.derive_learning_profile(
                current_profile=current,
                history=list(reversed(history)),
                notes=list(reversed(notes)),
                course_name=course_name,
                model=self.selected_model(model),
            )
            profile["courses"] = profile["courses"] or course_name
            self.database.save_learning_profile(
                **profile,
                course_key=course_key,
                course_name=course_name,
                question_count=len(history),
                last_history_id=max((int(item["id"]) for item in history), default=0),
            )
        except Exception as exc:  # noqa: BLE001 - present a safe UI error
            return (*self.learning_profile_values(course_key), f"自动分析失败：{exc}")
        values = self.learning_profile_values(course_key)
        evidence = profile.get("evidence_summary") or "未形成明确薄弱点证据。"
        return (*values, f"已更新“{course_name}”画像。推断依据：{evidence}")

    def documents_markdown(self) -> str:
        documents = self.database.list_documents()
        if not documents:
            return "知识库为空。上传 PDF 后会显示在这里。"
        lines = ["## 本地知识库", ""]
        current_folder = None
        for doc in documents:
            folder_name = doc["folder_name"] or "未分类"
            if folder_name != current_folder:
                lines.extend([f"### 📁 {folder_name}", ""])
                current_folder = folder_name
            lines.append(
                f"- **{doc['filename']}**：{doc['page_count']} 页，"
                f"{doc['chunk_count']} 个片段，导入于 {doc['created_at']}"
            )
        return "\n".join(lines)

    def document_choices(self, folder_context: str = ALL_FOLDERS) -> list[tuple[str, str]]:
        folder_filter = None if folder_context == ALL_FOLDERS else folder_context
        return [
            (
                f"[{doc['folder_name'] or '未分类'}] {doc['filename']}（{doc['page_count']}页）",
                doc["id"],
            )
            for doc in self.database.list_documents(folder_filter)
        ]

    def folder_choices(self) -> list[tuple[str, str]]:
        return [("未分类", "")] + [
            (f"📁 {folder['name']}", folder["id"]) for folder in self.database.list_folders()
        ]

    def folder_context_choices(self) -> list[tuple[str, str]]:
        return [("全部文件夹", ALL_FOLDERS), ("未分类", "")] + [
            (f"📁 {folder['name']}", folder["id"]) for folder in self.database.list_folders()
        ]

    @staticmethod
    def _dashboard_cards(cards: Sequence[tuple[str, str, int, str, str]]) -> str:
        rendered = []
        for icon, title, count, unit, detail in cards:
            rendered.append(
                '<article class="hub-card">'
                f'<div class="hub-icon">{html.escape(icon)}</div>'
                f'<div class="hub-card-title">{html.escape(title)}</div>'
                f'<div class="hub-stat">{count}<span>{html.escape(unit)}</span></div>'
                f"<p>{html.escape(detail)}</p>"
                "</article>"
            )
        return '<div class="hub-grid">' + "".join(rendered) + "</div>"

    def learning_space_html(self) -> str:
        stats = self.database.stats()
        try:
            tool_count = len(self.mcp_client.list_tool_names())
        except Exception:  # noqa: BLE001 - dashboard remains useful during MCP outage
            tool_count = 0
        profile_count = sum(
            1
            for item in self.database.list_learning_profiles()
            if int(item.get("question_count", 0)) > 0
            or item.get("weak_points")
            or item.get("preferences")
        )
        return self._dashboard_cards(
            [
                ("↻", "聊天历史", stats["sessions"], "段对话", "回顾并继续此前的学习对话。"),
                ("✎", "学习笔记", stats["notes"], "条笔记", "整理来自 PDF、问答和研究的笔记。"),
                ("◎", "课程画像", profile_count, "门课程", "查看薄弱点、偏好和自动推断依据。"),
                ("⌘", "MCP 服务", tool_count, "个工具", "集中查看计算、搜索、Zotero 与 Anki。"),
                ("✦", "内置技能", 6, "项能力", "查看助手按需使用的本地学习技能。"),
                ("◫", "知识资料", stats["documents"], "份 PDF", "按课程文件夹管理本地学习资料。"),
                ("◇", "长期记忆", stats["memories"], "条事实", "语义检索相关历史与笔记。"),
            ]
        )

    def knowledge_center_html(self) -> str:
        stats = self.database.stats()
        graph_status = self.graph_rag.status()
        vector_ready = self.retriever.client.collection_exists(self.retriever.collection_name)
        reranker_status = "已启用" if self.settings.enable_reranker else "可选"
        cards = [
            (
                "◉",
                "本地混合检索",
                stats["chunks"],
                "个片段",
                f"BGE-M3 向量 + BM25 融合；{'索引就绪' if vector_ready else '等待首份 PDF'}。",
            ),
            (
                "⇅",
                "精排模型",
                1 if self.settings.enable_reranker else 0,
                "个模型",
                f"BGE 重排序：{reranker_status}。",
            ),
            (
                "▦",
                "结构化 PDF",
                stats["pages"],
                "页资料",
                "正文、公式、表格、页面图像与 OCR 分层索引。",
            ),
            (
                "⌁",
                "本地 GraphRAG",
                graph_status["entity_count"],
                "个实体",
                "实体共现图 + 邻居扩展检索；已接入 PDF 问答。"
                if graph_status["ready"]
                else "可从现有 PDF 构建本地图谱检索索引。",
            ),
        ]
        return self._dashboard_cards(cards)

    def build_graph_rag(self) -> tuple[str, str]:
        try:
            build_status = self.graph_rag.build()
        except Exception as exc:  # noqa: BLE001 - surface graph build failure in UI
            build_status = f"GraphRAG 构建失败：{exc}"
        return build_status + "\n\n" + self.graph_rag.status_markdown(), self.graph_rag.graph_html()

    def graph_rag_search(self, query: str, folder_context: str = ALL_FOLDERS) -> str:
        folder_filter = None if folder_context == ALL_FOLDERS else folder_context
        documents = self.database.list_documents(folder_filter)
        document_ids = [item["id"] for item in documents]
        return self.graph_rag.search_markdown(query, document_ids)

    def knowledge_search(self, query: str, folder_context: str = ALL_FOLDERS) -> str:
        query = (query or "").strip()
        if not query:
            return "请输入要在知识中心检索的概念、公式或问题。"
        folder_filter = None if folder_context == ALL_FOLDERS else folder_context
        documents = self.database.list_documents(folder_filter)
        if not documents:
            return "当前知识中心没有可检索的 PDF。"
        try:
            results = self.retriever.search(
                query,
                [item["id"] for item in documents],
                candidate_limit=20,
                final_limit=8,
                rerank=self.settings.enable_reranker,
            )
        except Exception as exc:  # noqa: BLE001 - return an actionable search error
            return f"知识中心检索失败：{exc}"
        if not results:
            return "没有找到足够相关的知识片段。"
        kind_names = {
            "text": "正文",
            "formula": "公式",
            "table": "表格",
            "page_image": "页面图像/OCR",
        }
        sections = [f"## “{query}”的检索结果", ""]
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            excerpt = chunk.text[:700] + ("…" if len(chunk.text) > 700 else "")
            sections.append(
                f"### K{index} · 《{chunk.filename}》第 {chunk.page} 页 · "
                f"{kind_names.get(chunk.content_kind, chunk.content_kind)}\n\n"
                f"相关度：**{result.score:.3f}** · 向量 {result.dense_score:.3f} · "
                f"BM25 {result.lexical_score:.3f}\n\n{excerpt}"
            )
        return "\n\n---\n\n".join(sections)

    def memory_overview_html(self) -> str:
        counts = self.database.memory_overview()
        return self._dashboard_cards(
            [
                (
                    "▱",
                    "L1 · 学习事件",
                    counts["l1"],
                    "条追踪",
                    "提问、笔记、资料导入和画像更新的原始事件。",
                ),
                (
                    "⌘",
                    "L2 · 可检索事实",
                    counts["l2"],
                    "条事实",
                    "按问答与笔记整理并建立向量索引的学习事实。",
                ),
                (
                    "◎",
                    "L3 · 跨会话综合",
                    counts["l3"],
                    "条综合",
                    "课程画像、偏好、薄弱点和长对话摘要。",
                ),
            ]
        )

    def memory_l1_markdown(self) -> str:
        events = self.database.list_events(limit=80)
        if not events:
            return "L1 暂无事件。完成提问、保存笔记或上传 PDF 后会自动记录。"
        labels = {
            "document_loaded": "资料导入",
            "question_answered": "完成问答",
            "note_saved": "保存笔记",
            "profile_updated": "画像更新",
        }
        lines = ["## L1 · 仅追加学习事件", ""]
        for item in events:
            payload = item["payload_data"]
            summary = payload.get("question") or payload.get("concept") or payload.get("filename")
            summary = str(summary or payload.get("course_name") or "工作区发生变化")
            lines.append(
                f"- **#{item['id']} {labels.get(item['event_type'], item['event_type'])}** "
                f"· {item['created_at']} · `{item['session_id']}`\n  {summary[:220]}"
            )
        return "\n".join(lines)

    def memory_l2_markdown(self) -> str:
        memories = self.database.list_memories(limit=80)
        if not memories:
            return "L2 暂无事实。历史问答和笔记会在保存后自动进入这一层。"
        lines = ["## L2 · 可检索学习事实", ""]
        for item in memories:
            kind = "问答事实" if item["kind"] == "qa" else "笔记事实"
            lines.append(
                f"### {kind} · {item['title']}\n\n"
                f"{item['content'][:700]}\n\n"
                f"来源：L1 `{item['kind']}:{item['source_id']}` · "
                f"会话 `{item['session_id']}` · {item['created_at']}"
            )
        return "\n\n---\n\n".join(lines)

    def memory_l3_markdown(self) -> str:
        profiles = [
            item
            for item in self.database.list_learning_profiles()
            if int(item.get("question_count", 0)) > 0
            or item.get("weak_points")
            or item.get("preferences")
        ]
        summaries = self.database.list_session_summaries(limit=30)
        if not profiles and not summaries:
            return "L3 暂无综合结论。随着课程问答增多，画像和对话摘要会自动形成。"
        sections = ["## L3 · 跨会话综合"]
        sections.extend(
            (
                f"### 课程画像 · {item['course_name']}\n\n"
                f"- 薄弱点：{item['weak_points'] or '暂无明确证据'}\n"
                f"- 解释深度：{item['explanation_depth']}\n"
                f"- 数学风格：{item['math_style']}\n"
                f"- 偏好：{item['preferences'] or '未设置'}\n"
                f"- L2 依据：{item['evidence_summary'] or '等待更多问答'}"
            )
            for item in profiles
        )
        sections.extend(
            (
                f"### 对话综合 · {item['session_title']}\n\n{item['summary']}\n\n"
                f"覆盖至 L2 问答记录 #{item['summarized_through_id']} · {item['updated_at']}"
            )
            for item in summaries
        )
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _ring_positions(
        count: int, radius: float, center: tuple[float, float]
    ) -> list[tuple[float, float]]:
        if count <= 0:
            return []
        return [
            (
                center[0] + radius * math.cos(-math.pi / 2 + 2 * math.pi * index / count),
                center[1] + radius * math.sin(-math.pi / 2 + 2 * math.pi * index / count),
            )
            for index in range(count)
        ]

    def memory_graph_html(self) -> str:
        events = self.database.list_events(limit=28)
        memories = self.database.list_memories(limit=22)
        profiles = [
            item
            for item in self.database.list_learning_profiles()
            if int(item.get("question_count", 0)) > 0
            or item.get("weak_points")
            or item.get("preferences")
        ][:6]
        summaries = self.database.list_session_summaries(limit=4)
        if not events and not memories and not profiles and not summaries:
            return '<div class="empty-graph">完成几次问答或添加笔记后，这里会生成可追溯的三层记忆图谱。</div>'

        nodes: list[dict] = [
            {
                "id": f"l3:profile:{item['course_key']}",
                "layer": "L3",
                "label": item["course_name"],
                "detail": item.get("evidence_summary") or "课程画像",
            }
            for item in profiles
        ]
        nodes.extend(
            {
                "id": f"l3:summary:{item['session_id']}",
                "layer": "L3",
                "label": item["session_title"],
                "detail": item["summary"][:240],
            }
            for item in summaries
        )
        nodes.extend(
            {
                "id": item["id"],
                "layer": "L2",
                "label": item["title"],
                "detail": item["content"][:240],
            }
            for item in memories
        )
        for item in events:
            payload = item["payload_data"]
            nodes.append(
                {
                    "id": f"l1:{item['id']}",
                    "layer": "L1",
                    "label": str(
                        payload.get("question")
                        or payload.get("concept")
                        or payload.get("filename")
                        or item["event_type"]
                    ),
                    "detail": f"{item['event_type']} · {item['created_at']}",
                    "payload": payload,
                }
            )

        positions: dict[str, tuple[float, float]] = {}
        center = (450.0, 340.0)
        layer_nodes = {
            layer: [item for item in nodes if item["layer"] == layer]
            for layer in ("L3", "L2", "L1")
        }
        for layer, radius in (("L3", 75.0), ("L2", 195.0), ("L1", 305.0)):
            for item, position in zip(
                layer_nodes[layer],
                self._ring_positions(len(layer_nodes[layer]), radius, center),
                strict=True,
            ):
                positions[item["id"]] = position

        node_ids = {item["id"] for item in nodes}
        edges: set[tuple[str, str]] = set()
        profile_ids = {item["course_key"]: f"l3:profile:{item['course_key']}" for item in profiles}
        summary_ids = {item["session_id"]: f"l3:summary:{item['session_id']}" for item in summaries}
        for item in memories:
            if item["kind"] == "qa":
                target = profile_ids.get(item.get("qa_course_key"))
            else:
                folder_id = item.get("note_folder_id") or "__uncategorized__"
                target = profile_ids.get(f"folder:{folder_id}")
            if target and target in node_ids:
                edges.add((item["id"], target))
            summary_target = summary_ids.get(item["session_id"])
            if summary_target and summary_target in node_ids:
                edges.add((item["id"], summary_target))
        for item in events:
            payload = item["payload_data"]
            source = f"l1:{item['id']}"
            if item["event_type"] == "question_answered":
                target = f"qa:{payload.get('history_id')}"
            elif item["event_type"] == "note_saved":
                target = f"note:{payload.get('note_id')}"
            elif item["event_type"] == "profile_updated":
                target = f"l3:profile:{payload.get('course_key')}"
            else:
                target = ""
            if target in node_ids:
                edges.add((source, target))

        colors = {"L1": "#7f858e", "L2": "#d49a66", "L3": "#f36f21"}
        radii = {"L1": 7, "L2": 11, "L3": 16}
        svg = [
            '<div class="memory-graph-wrap">',
            '<div class="memory-graph-legend"><span class="l3">L3 综合</span><span class="l2">L2 事实</span><span class="l1">L1 事件</span></div>',
            '<svg viewBox="0 0 900 680" role="img" aria-label="三层记忆图谱">',
            '<circle cx="450" cy="340" r="75" class="memory-ring l3-ring"/>',
            '<circle cx="450" cy="340" r="195" class="memory-ring l2-ring"/>',
            '<circle cx="450" cy="340" r="305" class="memory-ring l1-ring"/>',
        ]
        for source, target in edges:
            if source not in positions or target not in positions:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="memory-edge"/>'
            )
        for item in nodes:
            x, y = positions[item["id"]]
            layer = item["layer"]
            label = str(item["label"]).strip() or layer
            short = label[:12] + ("…" if len(label) > 12 else "")
            tooltip = html.escape(f"{layer} · {label}\n{item['detail']}")
            svg.append(
                f'<g class="memory-node {layer.lower()}"><title>{tooltip}</title>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radii[layer]}" fill="{colors[layer]}"/>'
                f'<text x="{x:.1f}" y="{y + radii[layer] + 15:.1f}" text-anchor="middle">{html.escape(short)}</text></g>'
            )
        svg.extend(["</svg>", "</div>"])
        return "".join(svg)

    def stats_markdown(self) -> str:
        stats = self.database.stats()
        key_status = "已配置" if self.llm.configured else "未配置"
        return (
            "## 学习统计\n\n"
            f"- PDF 文档：**{stats['documents']}**\n"
            f"- 文档文件夹：**{stats['folders']}**\n"
            f"- 总页数：**{stats['pages']}**\n"
            f"- 检索片段：**{stats['chunks']}**\n"
            f"- 提问次数：**{stats['questions']}**\n"
            f"- 学习笔记：**{stats['notes']}**\n"
            f"- 学习会话：**{stats['sessions']}**\n"
            f"- 长期记忆：**{stats['memories']}**\n"
            f"- 对话压缩摘要：**{stats['summaries']}**\n"
            f"- DeepSeek：**{key_status}**（{self.settings.deepseek_model}）"
        )

    def web_status_markdown(self) -> str:
        available, detail = self.web_search.status()
        if available:
            return f"🟢 **联网搜索已就绪**  \n{detail}"
        return (
            "🔴 **联网搜索未启动**  \n"
            f"{detail}  \n可双击 `start.command` 一键启动，或运行 "
            "`docker compose -f docker-compose.search.yml up -d`。"
        )

    def generate_report(self) -> str:
        stats = self.database.stats()
        documents = self.database.list_documents()
        history = self.database.list_history(limit=50)
        notes = self.database.list_notes(limit=50)
        profile = self.database.get_learning_profile()
        now = datetime.now(UTC)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        report_path = self.settings.reports_dir / f"learning_report_{timestamp}.md"
        lines = [
            "# PDF 学习报告",
            "",
            f"生成时间：{now.isoformat(timespec='seconds')}",
            "",
            "## 统计",
            "",
            f"- 文档：{stats['documents']}",
            f"- 页数：{stats['pages']}",
            f"- 提问：{stats['questions']}",
            f"- 笔记：{stats['notes']}",
            "",
            "## 文档",
            "",
        ]
        lines.extend(
            f"- [{doc['folder_name'] or '未分类'}] {doc['filename']}："
            f"{doc['page_count']} 页，{doc['chunk_count']} 个片段"
            for doc in documents
        )
        lines.extend(
            [
                "",
                "## 学习画像",
                "",
                f"- 课程：{profile['courses'] or '未设置'}",
                f"- 薄弱点：{profile['weak_points'] or '未设置'}",
                f"- 解释深度：{profile['explanation_depth']}",
                f"- 数学风格：{profile['math_style']}",
                f"- 其他偏好：{profile['preferences'] or '无'}",
            ]
        )
        lines.extend(["", "## 笔记", ""])
        lines.extend(
            f"### {note['concept']}\n\n{note['content']}\n\n来源：{note['source']}"
            for note in reversed(notes)
        )
        lines.extend(["", "## 最近问答", ""])
        lines.extend(f"### {item['question']}\n\n{item['answer']}" for item in reversed(history))
        report_path.write_text("\n\n".join(lines), encoding="utf-8")

        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "stats": stats,
                    "profile": profile,
                    "documents": documents,
                    "notes": notes,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(report_path)
