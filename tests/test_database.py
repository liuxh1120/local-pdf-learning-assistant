from pathlib import Path

from pdf_assistant.database import Database
from pdf_assistant.models import DocumentChunk


def test_database_round_trip(tmp_path: Path):
    db = Database(tmp_path / "test.sqlite3")
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="统计.pdf",
        page=3,
        heading="第一章",
        text="样本均值",
        char_count=4,
    )
    db.add_document(
        document_id="doc-1",
        filename="统计.pdf",
        title="统计",
        stored_path="/tmp/test.pdf",
        sha256="abc",
        page_count=10,
        chunks=[chunk],
    )
    assert db.get_document_by_hash("abc")["id"] == "doc-1"
    assert db.list_chunks(["doc-1"])[0] == chunk
    assert db.stats()["documents"] == 1


def test_conversation_history_and_model_are_persisted(tmp_path: Path):
    db = Database(tmp_path / "history.sqlite3")
    db.start_session("session-1")
    db.add_qa(
        session_id="session-1",
        question="什么是中心极限定理？",
        answer="这是一个回答。",
        mode="仅 PDF",
        model="deepseek-v4-pro",
        document_ids=[],
        sources=[],
    )
    db.set_session_title_from_question("session-1", "什么是中心极限定理？")

    sessions = db.list_sessions("session-1")
    assert sessions[0]["title"] == "什么是中心极限定理？"
    assert sessions[0]["question_count"] == 1
    history = db.list_history(session_id="session-1", chronological=True)
    assert history[0]["model"] == "deepseek-v4-pro"


def test_conversation_can_be_renamed_and_deleted(tmp_path: Path):
    db = Database(tmp_path / "conversation-actions.sqlite3")
    db.start_session("session-actions")
    db.add_qa(
        session_id="session-actions",
        question="什么是充分统计量？",
        answer="这是一个回答。",
        mode="仅 PDF",
        model="deepseek-v4-flash",
        document_ids=[],
        sources=[],
    )
    assert db.rename_session("session-actions", "充分统计量复习")
    assert db.list_sessions("session-actions")[0]["title"] == "充分统计量复习"

    assert db.delete_session("session-actions")
    assert db.list_history(session_id="session-actions") == []
    assert db.list_sessions("missing-session") == []


def test_document_folders_are_persisted_and_documents_can_be_moved(tmp_path: Path):
    db = Database(tmp_path / "folders.sqlite3")
    folder_id = db.create_folder("数理统计")
    chunk = DocumentChunk(
        id="chunk-folder",
        document_id="doc-folder",
        filename="概率论.pdf",
        page=1,
        heading="",
        text="随机变量",
        char_count=4,
    )
    db.add_document(
        document_id="doc-folder",
        filename="概率论.pdf",
        title="概率论",
        stored_path="/tmp/probability.pdf",
        sha256="folder-hash",
        page_count=8,
        chunks=[chunk],
        folder_id=folder_id,
    )

    assert db.list_folders()[0]["name"] == "数理统计"
    assert db.list_documents()[0]["folder_name"] == "数理统计"
    assert db.stats()["folders"] == 1
    assert db.set_documents_folder(["doc-folder"], "") == 1
    assert db.list_documents()[0]["folder_name"] == ""


def test_long_term_memory_summary_and_profile_are_persisted(tmp_path: Path):
    db = Database(tmp_path / "memory.sqlite3")
    db.start_session("session-memory")
    history_id = db.add_qa(
        session_id="session-memory",
        question="它需要什么条件？",
        rewritten_question="中心极限定理需要什么条件？",
        answer="需要独立性等条件。",
        mode="仅 PDF",
        model="deepseek-v4-pro",
        document_ids=[],
        sources=[],
    )
    note_id = db.add_note(
        "session-memory",
        "大样本下样本均值近似正态。",
        "中心极限定理",
        "手动笔记",
        "",
    )

    memories = db.list_unindexed_memories()
    assert {item["id"] for item in memories} == {f"qa:{history_id}", f"note:{note_id}"}
    assert db.list_history(session_id="session-memory")[0]["rewritten_question"].startswith(
        "中心极限定理"
    )

    db.mark_memories_indexed([item["id"] for item in memories])
    assert db.list_unindexed_memories() == []
    db.save_session_summary("session-memory", "已学习中心极限定理。", history_id)
    assert db.get_session_summary("session-memory")["summarized_through_id"] == history_id

    db.save_learning_profile(
        courses="数理统计",
        weak_points="极限分布",
        explanation_depth="深入",
        math_style="定义证明优先",
        preferences="使用 LaTeX",
    )
    assert db.get_learning_profile()["courses"] == "数理统计"
    assert db.stats()["memories"] == 2
    assert db.stats()["summaries"] == 1


def test_course_profiles_and_history_are_isolated(tmp_path: Path):
    db = Database(tmp_path / "courses.sqlite3")
    db.start_session("course-session")
    db.add_qa(
        session_id="course-session",
        question="为什么 MLE 会渐近正态？",
        answer="需要正则条件。",
        mode="仅 PDF",
        model="deepseek-v4-pro",
        document_ids=[],
        sources=[],
        course_key="folder:statistics",
        course_name="数理统计",
    )
    db.save_learning_profile(
        course_key="folder:statistics",
        course_name="数理统计",
        courses="数理统计",
        weak_points="渐近理论",
        explanation_depth="深入",
        math_style="定义证明优先",
        preferences="先列正则条件",
        evidence_summary="多次追问渐近正态性的条件。",
        question_count=1,
        last_history_id=1,
    )

    statistics = db.get_learning_profile("folder:statistics", "数理统计")
    timeseries = db.get_learning_profile("folder:timeseries", "时间序列")
    assert statistics["weak_points"] == "渐近理论"
    assert timeseries["weak_points"] == ""
    assert len(db.list_history(course_key="folder:statistics")) == 1
    assert db.list_history(course_key="folder:timeseries") == []


def test_three_layer_memory_overview_is_traceable(tmp_path: Path):
    db = Database(tmp_path / "three-layer-memory.sqlite3")
    db.start_session("session-layers", "渐近理论")
    history_id = db.add_qa(
        session_id="session-layers",
        question="渐近正态需要哪些正则条件？",
        answer="需要可识别性、可微性和信息矩阵非退化等条件。",
        mode="仅 PDF",
        model="deepseek-v4-pro",
        document_ids=[],
        sources=[],
        course_key="folder:statistics",
        course_name="数理统计",
    )
    db.add_event(
        "session-layers",
        "question_answered",
        {"history_id": history_id, "question": "渐近正态需要哪些正则条件？"},
    )
    db.save_learning_profile(
        course_key="folder:statistics",
        course_name="数理统计",
        courses="数理统计",
        weak_points="渐近理论",
        explanation_depth="深入",
        math_style="定义证明优先",
        preferences="",
        evidence_summary="反复追问正则条件。",
        question_count=1,
        last_history_id=history_id,
    )
    db.save_session_summary("session-layers", "正在学习渐近正态。", history_id)

    assert db.list_events()[0]["payload_data"]["history_id"] == history_id
    assert db.list_memories()[0]["qa_course_key"] == "folder:statistics"
    assert db.list_session_summaries()[0]["session_title"] == "渐近理论"
    assert db.memory_overview() == {"l1": 1, "l2": 1, "l3": 2}
