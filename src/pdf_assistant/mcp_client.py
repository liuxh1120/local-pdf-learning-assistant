from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Settings

TOOL_GROUP_CHOICES = [
    ("Python / SymPy 计算", "math"),
    ("学术搜索", "academic"),
    ("多轮网页搜索", "web"),
    ("受限学习文件", "files"),
    ("只读学习数据库", "sqlite"),
    ("Zotero 论文库", "zotero"),
    ("Anki 卡片", "anki"),
]
DEFAULT_TOOL_GROUPS = ["math", "academic", "web", "files", "sqlite", "zotero", "anki"]
ANKI_WRITE_TOOLS = {"anki_create_card", "anki_update_note", "anki_set_due"}


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    group: str
    description: str
    input_schema: dict[str, Any]
    requires_explicit_write_request: bool = False

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class AgentToolRegistry:
    """Runtime registry shared by discovery, policy filtering and the agent loop."""

    def __init__(self, registrations: Sequence[ToolRegistration]):
        self._tools = {item.name: item for item in registrations}

    @classmethod
    def from_mcp(cls, tools: Sequence[Any]) -> AgentToolRegistry:
        return cls(
            [
                ToolRegistration(
                    name=tool.name,
                    group=tool.name.split("_", 1)[0],
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    requires_explicit_write_request=tool.name in ANKI_WRITE_TOOLS,
                )
                for tool in tools
            ]
        )

    def select(self, groups: set[str]) -> list[ToolRegistration]:
        return [item for item in self._tools.values() if item.group in groups]

    def get(self, name: str) -> ToolRegistration | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._tools.values():
            counts[item.group] = counts.get(item.group, 0) + 1
        return counts


class MCPToolClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_trace: list[dict[str, Any]] = []
        self.last_registry: AgentToolRegistry | None = None

    def _server_parameters(self) -> StdioServerParameters:
        env = dict(os.environ)
        env["PDF_ASSISTANT_ROOT"] = str(self.settings.project_root)
        src_path = str(self.settings.project_root / "src")
        env["PYTHONPATH"] = (
            src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        )
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "pdf_assistant.mcp_server"],
            cwd=self.settings.project_root,
            env=env,
        )

    @staticmethod
    def _tool_group(name: str) -> str:
        return name.split("_", 1)[0]

    @staticmethod
    def _tool_result_text(result: Any) -> str:
        parts: list[str] = []
        for item in getattr(result, "content", []):
            if hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "model_dump_json"):
                parts.append(item.model_dump_json())
            else:
                parts.append(str(item))
        text = "\n".join(parts).strip()
        if getattr(result, "isError", False):
            return "工具执行失败：" + text
        return text or "工具执行完成，但没有返回文本。"

    @staticmethod
    def _anki_write_authorized(question: str) -> bool:
        lowered = question.lower()
        object_named = "anki" in lowered or "卡片" in question or "闪卡" in question
        write_named = any(
            word in question
            for word in ("创建", "生成", "制作", "添加", "更新", "修改", "安排", "设置复习")
        )
        return object_named and write_named

    async def _list_tools(self, groups: set[str]) -> list[Any]:
        async with (
            stdio_client(self._server_parameters()) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            response = await session.list_tools()
            return [tool for tool in response.tools if self._tool_group(tool.name) in groups]

    def list_tool_names(self, groups: Sequence[str] | None = None) -> list[str]:
        selected = set(groups or DEFAULT_TOOL_GROUPS)
        return asyncio.run(self._list_tool_names(selected))

    async def _list_tool_names(self, groups: set[str]) -> list[str]:
        tools = await self._list_tools(groups)
        self.last_registry = AgentToolRegistry.from_mcp(tools)
        return self.last_registry.names()

    def run_agent(
        self,
        *,
        openai_client: Any,
        model: str,
        messages: list[dict[str, Any]],
        groups: Sequence[str],
        user_question: str,
        max_tokens: int = 3000,
    ) -> str:
        self.last_trace = []
        if not self.settings.enable_mcp_tools or not groups:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        try:
            return asyncio.run(
                self._run_agent_async(
                    openai_client=openai_client,
                    model=model,
                    messages=messages,
                    groups=set(groups),
                    user_question=user_question,
                    max_tokens=max_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - answer should survive an MCP outage
            self.last_trace.append(
                {
                    "tool": "MCP",
                    "ok": False,
                    "summary": f"MCP 不可用，已回退为普通回答：{type(exc).__name__}: {exc}",
                }
            )
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

    async def _run_agent_async(
        self,
        *,
        openai_client: Any,
        model: str,
        messages: list[dict[str, Any]],
        groups: set[str],
        user_question: str,
        max_tokens: int,
    ) -> str:
        async with (
            stdio_client(self._server_parameters()) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            registry = AgentToolRegistry.from_mcp(listed.tools)
            self.last_registry = registry
            available = registry.select(groups)
            openai_tools = [tool.as_openai_tool() for tool in available]
            if not openai_tools:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()

            working_messages = list(messages)
            seen_calls: dict[str, int] = {}
            for round_number in range(1, self.settings.mcp_max_tool_rounds + 1):
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=working_messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                )
                message = response.choices[0].message
                tool_calls = list(message.tool_calls or [])
                if not tool_calls:
                    return (message.content or "").strip()
                working_messages.append(message.model_dump(exclude_none=True))

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    signature = (
                        f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
                    )
                    seen_calls[signature] = seen_calls.get(signature, 0) + 1
                    registration = registry.get(name)
                    if registration is None:
                        result_text = f"已拒绝未知工具：{name}。"
                        ok = False
                    elif seen_calls[signature] > 2:
                        result_text = "已停止重复的相同工具调用，请根据已有结果完成回答。"
                        ok = False
                    elif (
                        registration.requires_explicit_write_request
                        and not self._anki_write_authorized(user_question)
                    ):
                        result_text = "已拒绝写入 Anki：用户没有明确要求创建、更新或安排卡片。"
                        ok = False
                    else:
                        try:
                            result = await session.call_tool(name, arguments=arguments)
                            result_text = self._tool_result_text(result)
                            ok = not bool(getattr(result, "isError", False))
                        except Exception as exc:  # noqa: BLE001 - let the model recover
                            result_text = f"工具调用失败：{type(exc).__name__}: {exc}"
                            ok = False
                    trace_id = len(self.last_trace) + 1
                    self.last_trace.append(
                        {
                            "id": trace_id,
                            "round": round_number,
                            "tool": name,
                            "arguments": arguments,
                            "ok": ok,
                            "summary": result_text[:500],
                        }
                    )
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"[T{trace_id}] {name}\n{result_text[:16000]}",
                        }
                    )

            final_response = openai_client.chat.completions.create(
                model=model,
                messages=working_messages,
                tools=openai_tools,
                tool_choice="none",
                max_tokens=max_tokens,
            )
            return (final_response.choices[0].message.content or "").strip()

    def status_markdown(self) -> str:
        lines = ["### MCP 工具状态"]
        if not self.settings.enable_mcp_tools:
            return "\n\n".join(lines + ["🔴 MCP 工具已在 `.env` 中关闭。"])
        try:
            names = self.list_tool_names()
        except Exception as exc:  # noqa: BLE001 - diagnostic output
            return "\n\n".join(lines + [f"🔴 MCP 服务启动失败：`{type(exc).__name__}: {exc}`"])
        lines.append(f"🟢 本地 MCP 服务正常，共注册 **{len(names)}** 个工具。")
        if self.last_registry:
            group_text = "、".join(
                f"{group} {count} 个"
                for group, count in sorted(self.last_registry.group_counts().items())
            )
            lines.append(
                f"🟢 Agent 工具注册表已加载：{group_text}；最多循环 {self.settings.mcp_max_tool_rounds} 轮。"
            )
        lines.append(
            f"🟢 Python/SymPy、学术搜索、受限文件、只读 SQLite 已注册。\n\n"
            f"文件访问根目录：`{self.settings.learning_files_root}`"
        )

        try:
            response = httpx.get(
                f"{self.settings.zotero_local_url}/users/0/items",
                params={"limit": 1, "format": "json"},
                timeout=2,
            )
            response.raise_for_status()
            lines.append("🟢 Zotero 本地 API 已连接。")
        except (httpx.HTTPError, RuntimeError, ValueError):
            if self.settings.zotero_user_id and self.settings.zotero_api_key:
                lines.append("🟡 Zotero 已配置云端凭据，将在调用时验证。")
            else:
                lines.append("🟡 Zotero 未连接：需要打开 Zotero 并启用本地 API。")

        try:
            response = httpx.post(
                self.settings.anki_connect_url,
                json={"action": "version", "version": 6},
                timeout=2,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            lines.append("🟢 AnkiConnect 已连接。")
        except (httpx.HTTPError, RuntimeError, ValueError):
            lines.append("🟡 Anki 未连接：需要打开 Anki 并安装 AnkiConnect。")
        return "\n\n".join(lines)

    def trace_markdown(self) -> str:
        if not self.last_trace:
            return "本次回答未调用 MCP 工具。"
        lines = ["### MCP 工具调用"]
        for item in self.last_trace:
            icon = "✅" if item.get("ok") else "⚠️"
            lines.append(
                f"{icon} **{item.get('tool', '工具')}**：{str(item.get('summary', ''))[:240]}"
            )
        return "\n\n".join(lines)
