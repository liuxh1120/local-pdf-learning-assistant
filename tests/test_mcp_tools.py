from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pdf_assistant import mcp_server
from pdf_assistant.assistant import (
    MODE_MIXED,
    MODE_PDF,
    MODE_WEB,
    allowed_tool_groups_for_mode,
)
from pdf_assistant.config import Settings
from pdf_assistant.mcp_client import AgentToolRegistry, MCPToolClient


def test_sympy_verifies_identity():
    result = json.loads(
        mcp_server.math_sympy(
            "(x+1)^2-(x^2+2*x+1)",
            operation="simplify",
        )
    )
    assert result["result"] == "0"


def test_python_validator_blocks_system_imports():
    with pytest.raises(ValueError, match="不允许导入"):
        mcp_server._validate_python("import os\nos.system('whoami')")


def test_filesystem_and_sqlite_are_restricted(tmp_path, monkeypatch):
    learning_root = tmp_path / "learning"
    learning_root.mkdir()
    (learning_root / "note.txt").write_text("中心极限定理", encoding="utf-8")
    database_path = tmp_path / "assistant.sqlite3"
    with sqlite3.connect(database_path) as db:
        db.execute("CREATE TABLE notes(id INTEGER, content TEXT)")
        db.execute("INSERT INTO notes VALUES(1, 'MLE')")

    test_settings = replace(
        Settings.load(),
        data_dir=tmp_path,
        learning_files_root=learning_root,
        database_path=database_path,
    )
    monkeypatch.setattr(mcp_server, "settings", test_settings)

    assert "中心极限定理" in mcp_server.files_read("note.txt")
    with pytest.raises(ValueError, match="超出"):
        mcp_server.files_read("../outside.txt")
    result = json.loads(mcp_server.sqlite_query("SELECT * FROM notes"))
    assert result["rows"] == [{"id": 1, "content": "MLE"}]
    with pytest.raises(ValueError, match="只允许"):
        mcp_server.sqlite_query("DELETE FROM notes")


class _FakeMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, **_kwargs):
        return {"role": "assistant", "content": self.content, "tool_calls": []}


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            function = SimpleNamespace(
                name="math_sympy",
                arguments=json.dumps(
                    {"expression": "(x+1)^2-(x^2+2*x+1)", "operation": "simplify"}
                ),
            )
            tool_call = SimpleNamespace(id="call_1", function=function)
            message = _FakeMessage(tool_calls=[tool_call])
        else:
            message = _FakeMessage(content="恒等式成立 [T1]")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_deepseek_agent_loop_calls_mcp_math_tool():
    client = MCPToolClient(Settings.load())
    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    answer = client.run_agent(
        openai_client=fake_openai,
        model="test-model",
        messages=[{"role": "user", "content": "验证恒等式"}],
        groups=["math"],
        user_question="请验证恒等式",
    )
    assert answer == "恒等式成立 [T1]"
    assert client.last_trace[0]["tool"] == "math_sympy"
    assert client.last_trace[0]["ok"] is True


def test_anki_write_requires_explicit_request():
    assert MCPToolClient._anki_write_authorized("介绍一下 Anki") is False
    assert MCPToolClient._anki_write_authorized("请生成两张 Anki 卡片") is True


def test_pdf_mode_disables_every_agent_tool():
    requested = ["math", "academic", "web", "files", "sqlite"]
    assert allowed_tool_groups_for_mode(MODE_PDF, requested) == []
    assert allowed_tool_groups_for_mode(MODE_WEB, requested) == [
        "math",
        "academic",
        "web",
        "sqlite",
    ]
    assert allowed_tool_groups_for_mode(MODE_MIXED, requested) == requested


def test_agent_tool_registry_filters_groups_and_marks_writes():
    tools = [
        SimpleNamespace(
            name="math_sympy", description="math", inputSchema={"type": "object"}
        ),
        SimpleNamespace(
            name="anki_create_card", description="write", inputSchema={"type": "object"}
        ),
    ]
    registry = AgentToolRegistry.from_mcp(tools)
    assert [item.name for item in registry.select({"math"})] == ["math_sympy"]
    assert registry.get("anki_create_card").requires_explicit_write_request is True
    assert registry.group_counts() == {"math": 1, "anki": 1}
