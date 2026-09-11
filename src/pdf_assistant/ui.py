from __future__ import annotations

import gradio as gr

from .assistant import (
    ALL_FOLDERS,
    MEMORY_ALL,
    MEMORY_CURRENT,
    MODE_MIXED,
    MODE_PDF,
    MODE_WEB,
    MODEL_CHOICES,
    LearningAssistant,
)
from .config import Settings
from .mcp_client import DEFAULT_TOOL_GROUPS, TOOL_GROUP_CHOICES

APP_CSS = """
:root {
  --study-bg: #171615;
  --study-panel: #211f1d;
  --study-panel-2: #292622;
  --study-border: #3b3732;
  --study-text: #f2eee8;
  --study-muted: #a79f96;
  --study-accent: #ef762f;
}
body, .gradio-container { background: var(--study-bg) !important; color: var(--study-text); }
.gradio-container { max-width: none !important; padding: 0 !important; }
#app-sidebar {
  background: #24211f !important;
  border-right: 1px solid var(--study-border) !important;
  padding: 10px 10px 12px !important;
}
.study-brand { padding: 4px 8px 10px; }
.study-brand strong { font-family: Georgia, serif; font-size: 20px; letter-spacing: -.4px; }
.study-brand span { color: var(--study-accent); }
.sidebar-nav { margin: 0 0 4px !important; padding: 0 !important; background: transparent !important; border: 0 !important; }
.sidebar-nav > .styler, .sidebar-nav .form, .form:has(> .sidebar-nav) { background: transparent !important; border: 0 !important; }
.sidebar-nav .wrap { gap: 2px !important; }
.sidebar-nav label {
  min-height: 36px !important; border: 0 !important; border-radius: 9px !important; padding: 7px 10px !important;
  background: transparent !important; color: #d7d1ca !important;
}
.sidebar-nav label:has(input:checked) { background: #34302c !important; color: white !important; }
.sidebar-nav input { display: none !important; }
#home-new-chat { min-height: 36px !important; margin: 0 0 2px !important; padding: 7px 10px !important; justify-content: flex-start !important; background: transparent !important; border: 0 !important; box-shadow: none !important; font-weight: 400 !important; color: #d7d1ca !important; }
#home-new-chat:hover { background: #34302c !important; }
.sidebar-caption { color: var(--study-muted); font-size: 11px; margin: 12px 8px 5px; letter-spacing: .04em; }
#recent-conversations { position: relative !important; min-height: 42px; background: transparent !important; border: 0 !important; box-shadow: none !important; }
#recent-conversations > .styler, #recent-conversations .form, .form:has(> .history-list) { background: transparent !important; border: 0 !important; }
.history-list { max-height: 310px; overflow-y: auto; margin: 0 !important; padding: 0 2px 0 0 !important; background: transparent !important; border: 0 !important; }
.history-list label { width: 100% !important; min-height: 34px !important; padding: 6px 66px 6px 9px !important; border-radius: 8px !important; border: 0 !important; background: transparent !important; }
.history-list label:has(input:checked) { background: transparent !important; color: white !important; }
.history-list label:hover { background: rgba(255,255,255,.035) !important; }
.history-list input[type="radio"] { display: none !important; }
#conversation-rename, #conversation-delete { position: absolute !important; z-index: 5; top: 3px; min-width: 27px !important; width: 27px !important; height: 27px !important; padding: 0 !important; border: 0 !important; border-radius: 6px !important; background: transparent !important; box-shadow: none !important; color: #aaa39b !important; font-size: 15px !important; }
#conversation-rename { right: 33px !important; }
#conversation-delete { right: 4px !important; }
#conversation-rename:hover { background: rgba(255,255,255,.06) !important; color: white !important; }
#conversation-delete:hover { background: rgba(255,90,82,.10) !important; color: #ff746d !important; }
#conversation-delete-panel { margin: 4px 2px 9px !important; padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
#conversation-delete-panel > #conversation-delete-panel { padding: 8px !important; border: 1px solid var(--study-border) !important; border-radius: 10px !important; background: #1d1b19 !important; }
#conversation-delete-panel .form, #conversation-delete-panel > .styler { background: transparent !important; border: 0 !important; }
.inline-conversation-title { width: 100%; height: 28px; min-width: 0; margin: -3px 0; padding: 3px 7px; border: 1px solid #558ee8; border-radius: 6px; outline: 2px solid rgba(85,142,232,.22); background: #1c1b1a; color: white; font: inherit; }
.history-list label:has(.inline-conversation-title) { padding-right: 66px !important; }
#conversation-rename-value, #conversation-rename-save, #conversation-action-status { display: none !important; }
.conversation-action-row { gap: 6px !important; }
.conversation-action-row button { min-height: 31px !important; padding: 4px 7px !important; }
.danger-action { color: #ff7b72 !important; border-color: rgba(255,123,114,.45) !important; }
#sidebar-bottom { position: fixed !important; left: 10px !important; width: 220px !important; bottom: 12px !important; z-index: 4; margin: 0 !important; padding: 8px 0 2px !important; border: 0 !important; border-top: 1px solid var(--study-border) !important; border-radius: 0 !important; background: #24211f !important; box-shadow: none !important; }
#sidebar-bottom > #sidebar-bottom { position: static !important; margin: 0 !important; padding: 0 !important; border: 0 !important; background: transparent !important; }
#sidebar-bottom .form, #sidebar-bottom > .styler { background: transparent !important; border: 0 !important; }
.page-shell {
  width: 100% !important; max-width: none; margin: 0; padding: 0;
  background: transparent !important; border: 0 !important; box-shadow: none !important;
}
.column > .page-shell { max-width: 1320px; margin: 0 auto; padding: 18px 38px 60px; }
.page-shell > .styler { width: 100% !important; background: transparent !important; }
.page-shell .gr-group, .page-shell .group, .page-shell .form,
.detail-shell, .detail-shell > .styler, .detail-shell .form,
.card-row, .card-row > .styler { background: transparent !important; }
.page-header { margin-bottom: 28px; }
.page-kicker { color: var(--study-accent); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }
.page-header h1 { margin: 6px 0 5px; font-family: Georgia, 'Songti SC', serif; font-size: 36px; font-weight: 500; }
.page-header p { color: var(--study-muted); font-size: 14px; }
.home-hero { text-align: center; padding: 4px 0 2px; }
.home-hero h1 { font-family: Georgia, 'Songti SC', serif; font-size: clamp(34px, 4vw, 55px); font-weight: 400; margin: 0; }
.home-hero p { color: var(--study-muted); margin-top: 12px; }
#study-chatbot { border: 0 !important; background: transparent !important; }
#study-chatbot .bubble-wrap { max-width: 900px; margin-inline: auto; }
#study-composer { width: 100% !important; max-width: 980px; margin: 0 auto; border: 0 !important; padding: 0 !important; background: transparent !important; box-shadow: none !important; }
#study-composer > #study-composer { border: 1px solid var(--study-border) !important; border-radius: 24px !important; padding: 14px 16px !important; background: var(--study-panel) !important; }
#study-composer > .styler, #study-composer .form, #study-question, #study-question > .styler,
#study-question .form, .form:has(> #study-question) { background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; }
#study-question textarea { min-height: 112px !important; background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 4px 2px !important; }
#study-send, #study-clear { display: none !important; }
.composer-footer { align-items: center !important; gap: 7px !important; }
.composer-controls { gap: 3px !important; align-items: center !important; flex-wrap: wrap !important; }
.composer-icon { min-width: 36px !important; width: 36px !important; height: 36px !important; flex: 0 0 36px !important; padding: 0 !important; border: 0 !important; border-radius: 8px !important; background: transparent !important; box-shadow: none !important; font-size: 19px !important; color: #d8d1c9 !important; }
.composer-icon:hover { background: rgba(255,255,255,.055) !important; color: var(--study-accent) !important; }
.clean-upload, .clean-upload > .styler, .clean-upload .form,
.clean-upload .wrap, .clean-upload .upload-container,
.clean-upload [class*="upload"], .clean-upload [class*="drop"] {
  border: 0 !important; outline: 0 !important; box-shadow: none !important;
}
#composer-panel { margin: 7px 0 10px !important; border: 0 !important; padding: 0 !important; background: transparent !important; box-shadow: none !important; overflow: visible !important; }
#composer-panel > #composer-panel { border: 1px solid var(--study-border) !important; border-radius: 15px !important; padding: 13px !important; background: #1d1b19 !important; overflow: visible !important; }
#composer-panel .form, #composer-panel > .styler { background: transparent !important; }
.composer-panel-close { max-width: 92px !important; margin-left: auto !important; background: transparent !important; border-color: var(--study-border) !important; }
.card-row { gap: 22px !important; margin-bottom: 22px !important; align-items: stretch !important; }
.nav-card { min-height: 190px !important; padding: 25px !important; border: 1px solid var(--study-border) !important; border-radius: 18px !important; background: var(--study-panel) !important; box-shadow: none !important; white-space: pre-line !important; text-align: left !important; justify-content: flex-start !important; align-items: flex-start !important; line-height: 1.85 !important; font-size: 15px !important; color: var(--study-text) !important; }
.nav-card:hover { border-color: rgba(239,118,47,.62) !important; transform: translateY(-2px); }
.detail-back { max-width: 150px !important; margin-bottom: 20px !important; background: transparent !important; border-color: var(--study-border) !important; }
.detail-shell { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.detail-title { margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid var(--study-border); }
.detail-title h2 { font-family: Georgia, 'Songti SC', serif; font-weight: 500; font-size: 30px; margin: 0 0 6px; }
.detail-title p { color: var(--study-muted); margin: 0; }
.card-detail { padding: 8px 10px 14px !important; }
.compact-refresh { max-width: 180px !important; }
.hub-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; margin: 18px 0 28px; }
.hub-card { min-height: 155px; border: 1px solid var(--study-border); border-radius: 18px; background: var(--study-panel); padding: 20px; }
.hub-card:hover { border-color: rgba(239,118,47,.58); transform: translateY(-1px); }
.hub-icon { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 11px; background: rgba(239,118,47,.12); color: var(--study-accent); font-size: 18px; }
.hub-card-title { margin-top: 16px; font-weight: 650; font-size: 16px; }
.hub-stat { margin-top: 5px; font-size: 29px; font-weight: 700; }
.hub-stat span { margin-left: 7px; font-size: 12px; font-weight: 400; color: var(--study-muted); }
.hub-card p { color: var(--study-muted); font-size: 12px; line-height: 1.6; margin: 10px 0 0; }
.memory-graph-wrap { position: relative; height: min(70vh, 720px); min-height: 560px; overflow: hidden; border: 1px solid var(--study-border); border-radius: 20px; background: radial-gradient(circle at center, #2b211b 0, var(--study-bg) 60%); }
.memory-graph-wrap svg { width: 100%; height: 100%; }
.memory-ring { fill: none; stroke-width: 1; }
.l3-ring { stroke: rgba(239,118,47,.38); } .l2-ring { stroke: rgba(212,154,102,.25); } .l1-ring { stroke: rgba(180,180,180,.16); }
.memory-edge { stroke: rgba(215,195,176,.18); stroke-width: 1; }
.memory-node { cursor: help; } .memory-node circle { transition: filter .15s, transform .15s; transform-origin: center; }
.memory-node:hover circle { filter: drop-shadow(0 0 7px rgba(239,118,47,.7)); }
.memory-node text { fill: #bcb4ab; font-size: 9px; pointer-events: none; }
.memory-graph-legend { position: absolute; z-index: 2; top: 16px; left: 18px; display: flex; gap: 9px; }
.memory-graph-legend span { border: 1px solid var(--study-border); border-radius: 999px; padding: 4px 9px; font-size: 10px; background: rgba(23,22,21,.78); }
.memory-graph-legend .l3 { color: #f36f21; } .memory-graph-legend .l2 { color: #d49a66; } .memory-graph-legend .l1 { color: #a7abb2; }
.empty-graph { padding: 80px 24px; text-align: center; color: var(--study-muted); border: 1px dashed var(--study-border); border-radius: 18px; }
.status-card { border-radius: 12px; }
.section-card { border: 0 !important; background: transparent !important; padding: 0 !important; box-shadow: none !important; }
.section-card > .section-card { border: 1px solid var(--study-border) !important; border-radius: 18px !important; background: var(--study-panel) !important; padding: 18px !important; }
.section-card .styler, .section-card .prose, .section-card .html-container, .section-card .form, .settings-card, .settings-card .styler, .settings-card .prose { background: transparent !important; }
footer { visibility: hidden; }
@media (max-width: 800px) {
  .page-shell { padding: 22px 16px 48px; }
  .hub-grid { grid-template-columns: 1fr; }
  .page-header h1 { font-size: 30px; }
  .card-row { gap: 14px !important; margin-bottom: 14px !important; }
}
"""

KEYBOARD_JS = r"""
if (!window.__pdfAssistantKeyboardInstalled) {
  window.__pdfAssistantKeyboardInstalled = true;
  document.addEventListener("keydown", (event) => {
    const textarea = event.target;
    if (!(textarea instanceof HTMLTextAreaElement)) return;
    if (textarea.getAttribute("placeholder") !==
        "例如：请解释最大似然估计的定义、适用条件，并给出文中的例子。") return;
    if (event.key !== "Enter" || event.isComposing) return;
    if (event.shiftKey) {
      event.stopImmediatePropagation();
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    Array.from(document.querySelectorAll("button"))
      .find((button) => button.id === "study-send" || button.closest("#study-send"))?.click();
  }, true);

  const composerTooltips = {
    "composer-add-pdf": "添加 PDF",
    "composer-model": "DeepSeek 模型",
    "composer-folder": "课程 / 文件夹上下文",
    "composer-mode": "回答模式",
    "composer-memory": "长期记忆范围",
    "composer-pdf": "本轮参考 PDF",
    "composer-tools": "Agent / MCP 工具",
    "composer-trace": "本轮上下文与工具轨迹"
  };
  const applyComposerTooltips = () => {
    Object.entries(composerTooltips).forEach(([id, label]) => {
      document.querySelectorAll(`[id="${id}"] button, button[id="${id}"]`).forEach((button) => {
        button.title = label;
        button.setAttribute("aria-label", label);
      });
    });
  };
  new MutationObserver(applyComposerTooltips).observe(document.body, {childList: true, subtree: true});
  applyComposerTooltips();

  const positionConversationActions = (preferredLabel = null) => {
    const container = document.querySelector('[id="recent-conversations"]');
    if (!container) return;
    const label = preferredLabel || container.querySelector('label:has(input:checked)') || container.querySelector('label');
    if (!label) return;
    const actions = [
      ["conversation-rename", "重命名对话"],
      ["conversation-delete", "删除对话"]
    ];
    actions.forEach(([id, title]) => {
      container.querySelectorAll(`[id="${id}"]`).forEach((control) => {
        control.style.top = `${label.offsetTop + 3}px`;
        const button = control.matches('button') ? control : control.querySelector('button');
        if (button) {
          const effectiveTitle = id === "conversation-rename" && container.querySelector(".inline-conversation-title")
            ? "保存对话标题"
            : title;
          button.title = effectiveTitle;
          button.setAttribute("aria-label", effectiveTitle);
        }
      });
    });
  };
  const wireConversationActions = () => {
    const container = document.querySelector('[id="recent-conversations"]');
    if (!container || container.dataset.actionsWired) return;
    container.dataset.actionsWired = "true";
    container.addEventListener("pointerover", (event) => {
      const label = event.target.closest("label");
      if (label) positionConversationActions(label);
    });
    container.addEventListener("pointerleave", () => positionConversationActions());
    container.addEventListener("change", () => setTimeout(() => positionConversationActions(), 0));
  };
  const refreshConversationActions = () => {
    wireConversationActions();
    positionConversationActions();
  };

  const renameActionButton = () => {
    const root = document.querySelector('[id="recent-conversations"]');
    const control = root?.querySelector('[id="conversation-rename"]');
    return control?.matches("button") ? control : control?.querySelector("button");
  };
  const finishInlineRename = (save) => {
    const input = document.querySelector(".inline-conversation-title");
    if (!input) return;
    const label = input.closest("label");
    const titleSpan = label?.querySelector("span");
    const newTitle = input.value.trim();
    if (save && !newTitle) {
      input.focus();
      return;
    }
    if (save) {
      if (titleSpan) titleSpan.textContent = newTitle;
      const bridge = document.querySelector('[id="conversation-rename-value"] textarea');
      const saveButton = document.querySelector('[id="conversation-rename-save"] button');
      if (bridge && saveButton) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
        setter?.call(bridge, newTitle);
        bridge.dispatchEvent(new Event("input", {bubbles: true}));
        bridge.dispatchEvent(new Event("change", {bubbles: true}));
        saveButton.click();
      }
    }
    input.remove();
    if (titleSpan) titleSpan.style.display = "";
    const button = renameActionButton();
    if (button) {
      button.textContent = "✎";
      button.title = "重命名对话";
      button.setAttribute("aria-label", "重命名对话");
    }
  };
  window.__studyMindCancelInlineRename = () => finishInlineRename(false);
  window.__studyMindToggleInlineRename = () => {
    if (document.querySelector(".inline-conversation-title")) {
      finishInlineRename(true);
      return;
    }
    const container = document.querySelector('[id="recent-conversations"]');
    const label = container?.querySelector('label:has(input:checked)') || container?.querySelector("label");
    const titleSpan = label?.querySelector("span");
    if (!label || !titleSpan) return;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "inline-conversation-title";
    input.value = titleSpan.textContent.trim();
    input.setAttribute("aria-label", "对话标题");
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        finishInlineRename(true);
      } else if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finishInlineRename(false);
      }
    });
    titleSpan.style.display = "none";
    label.append(input);
    const button = renameActionButton();
    if (button) {
      button.textContent = "✓";
      button.title = "保存对话标题";
      button.setAttribute("aria-label", "保存对话标题");
    }
    requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
  };
  new MutationObserver(refreshConversationActions).observe(document.body, {childList: true, subtree: true});
  refreshConversationActions();
}
"""


def _normalize_files(files) -> list[str]:
    if not files:
        return []
    normalized = []
    for item in files if isinstance(files, list) else [files]:
        if isinstance(item, str):
            normalized.append(item)
        elif hasattr(item, "name"):
            normalized.append(item.name)
        else:
            normalized.append(str(item))
    return normalized


def build_ui() -> tuple[gr.Blocks, Settings]:
    settings = Settings.load()
    assistant = LearningAssistant(settings)

    def refresh_documents(selected=None, organizer_selected=None, folder_context=ALL_FOLDERS):
        choices = assistant.document_choices(folder_context)
        organizer_choices = assistant.document_choices()
        valid_ids = {value for _, value in choices}
        organizer_ids = {value for _, value in organizer_choices}
        selected_values = [value for value in (selected or []) if value in valid_ids]
        organizer_values = [value for value in (organizer_selected or []) if value in organizer_ids]
        return (
            gr.Dropdown(choices=choices, value=selected_values),
            gr.Dropdown(choices=organizer_choices, value=organizer_values),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
        )

    def upload_documents(files, selected, folder_id, folder_context):
        paths = _normalize_files(files)
        sidebar_choices = assistant.document_choices(folder_context)
        if not paths:
            return (
                "请选择至少一个 PDF。",
                gr.Dropdown(choices=sidebar_choices, value=selected or []),
                gr.Dropdown(choices=assistant.document_choices(), value=[]),
                assistant.documents_markdown(),
                assistant.documents_markdown(),
                assistant.documents_markdown(),
                None,
            )
        result = assistant.ingest(paths, folder_id or "")
        choices = assistant.document_choices(folder_context)
        current = list(selected or [])
        current.extend(result["added_ids"])
        current = list(dict.fromkeys(current))
        status = "\n\n".join(result["messages"])
        if result["warnings"]:
            status += "\n\n**解析提醒**\n\n" + "\n\n".join(result["warnings"])
        return (
            status,
            gr.Dropdown(choices=choices, value=current),
            gr.Dropdown(choices=choices, value=[]),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
            None,
        )

    def create_document_folder(name):
        status, folder_id = assistant.create_folder(name)
        choices = assistant.folder_choices()
        selected = folder_id or ""
        return (
            status,
            "" if folder_id else name,
            gr.Dropdown(choices=choices, value=selected),
            gr.Dropdown(choices=choices, value=selected),
            gr.Dropdown(choices=assistant.folder_context_choices(), value=selected),
            gr.Dropdown(choices=choices, value=selected),
            gr.Dropdown(
                choices=assistant.profile_scope_choices(),
                value=f"folder:{folder_id}" if folder_id else "__general__",
            ),
            gr.Dropdown(choices=assistant.folder_context_choices(), value=selected),
            gr.Dropdown(choices=assistant.folder_context_choices(), value=selected),
            gr.Dropdown(choices=choices, value=selected),
        )

    def move_library_documents(document_ids, folder_id, selected_docs, folder_context):
        status = assistant.move_documents(document_ids or [], folder_id or "")
        choices = assistant.document_choices(folder_context)
        organizer_choices = assistant.document_choices()
        valid_ids = {value for _, value in choices}
        selected_values = [value for value in (selected_docs or []) if value in valid_ids]
        return (
            status,
            gr.Dropdown(choices=choices, value=selected_values),
            gr.Dropdown(choices=organizer_choices, value=[]),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
            assistant.documents_markdown(),
        )

    def select_folder_context(folder_context, selected_docs):
        choices = assistant.document_choices(folder_context)
        valid_ids = {value for _, value in choices}
        values = [value for value in (selected_docs or []) if value in valid_ids]
        return gr.Dropdown(choices=choices, value=values)

    def sync_tools_with_answer_mode(answer_mode):
        if answer_mode == MODE_PDF:
            return gr.CheckboxGroup(value=[], interactive=False)
        return gr.CheckboxGroup(value=DEFAULT_TOOL_GROUPS, interactive=True)

    def conversation_component(selected_session_id=None):
        return gr.Radio(
            choices=assistant.conversation_choices(selected_session_id),
            value=selected_session_id or assistant.session_id,
        )

    def new_conversation():
        assistant.new_conversation()
        assistant.last_context_status = "等待提问。"
        return (
            [],
            conversation_component(),
            "",
            assistant.last_context_status,
            assistant.session_summary_markdown(),
        )

    def switch_conversation(session_id):
        if not session_id:
            return [], assistant.session_summary_markdown()
        messages = assistant.switch_conversation(session_id)
        return messages, assistant.session_summary_markdown()

    def open_conversation(session_id):
        messages, summary = switch_conversation(session_id)
        return (
            messages,
            summary,
            *show_page("首页"),
            gr.Radio(value=None),
            gr.Radio(value=None),
            gr.Group(visible=False),
        )

    def rename_conversation_ui(session_id, title):
        status = assistant.rename_conversation(session_id, title)
        return status, conversation_component(session_id), gr.Textbox(value=title.strip())

    def delete_conversation_ui(session_id):
        status = assistant.delete_conversation(session_id)
        assistant.last_context_status = "等待提问。"
        return (
            [],
            conversation_component(),
            "",
            assistant.last_context_status,
            assistant.session_summary_markdown(),
            status,
            gr.Group(visible=False),
            *show_page("首页"),
            gr.Radio(value=None),
            gr.Radio(value=None),
        )

    def ask_question(
        question,
        selected_docs,
        mode,
        advanced,
        model,
        memory_scope,
        folder_context,
        enable_agent_tools,
        tool_groups,
        chat_history,
    ):
        question = (question or "").strip()
        history = list(chat_history or [])
        if not question:
            return history, "", conversation_component(), assistant.last_context_status
        history.append({"role": "user", "content": question})
        try:
            answer = assistant.ask(
                question,
                selected_docs or [],
                mode,
                bool(advanced),
                model,
                memory_scope,
                folder_context,
                bool(enable_agent_tools),
                tool_groups or [],
            )
        except Exception as exc:  # noqa: BLE001 - callback must return an actionable message
            answer = f"处理失败：{exc}"
        history.append({"role": "assistant", "content": answer})
        return history, "", conversation_component(), assistant.last_context_status

    def save_note(content, concept, source, folder_id):
        status = assistant.add_note(content, concept, source, folder_id)
        return status, "", assistant.notes_markdown(), assistant.memory_l2_markdown()

    def save_profile(course_key, courses, weak_points, depth, math_style, preferences):
        return assistant.save_learning_profile(
            courses,
            weak_points,
            depth,
            math_style,
            preferences,
            course_key,
        )

    def compress_conversation(model):
        status = assistant.compress_current_conversation(model)
        return status, assistant.session_summary_markdown()

    def section_visibility(selected, names):
        return tuple(gr.Group(visible=selected == name) for name in names)

    def show_page(page_name):
        return section_visibility(page_name, ("首页", "学习空间", "记忆", "知识中心", "设置"))

    def show_navigation_page(page_name):
        if not page_name:
            return (*([gr.skip()] * 5), gr.skip())
        return (*show_page(page_name), gr.Radio(value=None))

    def show_new_home():
        return (*show_page("首页"), gr.Radio(value=None), gr.Radio(value=None))

    def show_composer_panel(panel_name=""):
        panels = ("upload", "model", "folder", "mode", "memory", "pdf", "tools", "trace")
        return (
            gr.Group(visible=bool(panel_name)),
            *section_visibility(panel_name, panels),
        )

    def show_learning_section(section="overview"):
        sections = (
            "overview",
            "history",
            "notes",
            "profile",
            "review",
            "mcp",
            "skills",
            "documents",
            "memory",
        )
        return section_visibility(section, sections)

    def show_memory_section(section="overview"):
        sections = ("overview", "graph", "l1", "l2", "l3")
        return section_visibility(section, sections)

    def show_knowledge_section(section="overview"):
        sections = ("overview", "hybrid", "reranker", "structured", "graph", "library", "search")
        return section_visibility(section, sections)

    def refresh_memory_views():
        return (
            assistant.memory_graph_html(),
            assistant.memory_l1_markdown(),
            assistant.memory_l2_markdown(),
            assistant.memory_l3_markdown(),
        )

    def build_graph_rag_ui():
        status, graph = assistant.build_graph_rag()
        return status, graph

    profile_values = assistant.learning_profile_values()
    space_stats = assistant.database.stats()
    profile_count = sum(
        1
        for item in assistant.database.list_learning_profiles()
        if int(item.get("question_count", 0)) > 0
        or item.get("weak_points")
        or item.get("preferences")
    )
    try:
        tool_count = len(assistant.mcp_client.list_tool_names())
    except Exception:  # noqa: BLE001 - the page can still load while MCP is unavailable
        tool_count = 0
    graph_status_data = assistant.graph_rag.status()
    memory_counts = assistant.database.memory_overview()

    with gr.Blocks(
        title="StudyMind · PDF 学习助手",
        fill_width=True,
        fill_height=True,
    ) as demo:
        gr.HTML(
            "<span aria-hidden='true'></span>",
            js_on_load=KEYBOARD_JS,
            css_template="height: 0; min-height: 0; overflow: hidden;",
            apply_default_css=False,
        )

        with gr.Sidebar(open=True, width=250, elem_id="app-sidebar"):
            gr.HTML('<div class="study-brand"><strong>Study<span>Mind</span></strong></div>')
            home_button = gr.Button("⌂  首页", elem_id="home-new-chat")
            main_navigation = gr.Radio(
                choices=[
                    ("✎  学习空间", "学习空间"),
                    ("⌕  知识中心", "知识中心"),
                ],
                value=None,
                show_label=False,
                elem_classes="sidebar-nav",
            )
            gr.HTML('<div class="sidebar-caption">最近对话</div>')
            with gr.Group(elem_id="recent-conversations"):
                conversation_selector = gr.Radio(
                    choices=assistant.conversation_choices(),
                    value=assistant.session_id,
                    show_label=False,
                    elem_classes="history-list",
                )
                conversation_rename_icon = gr.Button("✎", elem_id="conversation-rename")
                conversation_delete_icon = gr.Button("⌫", elem_id="conversation-delete")
            rename_title = gr.Textbox(
                show_label=False,
                max_lines=1,
                elem_id="conversation-rename-value",
            )
            rename_conversation_button = gr.Button(
                "保存重命名",
                elem_id="conversation-rename-save",
            )
            conversation_action_status = gr.Markdown(elem_id="conversation-action-status")
            with gr.Group(
                visible=False, elem_id="conversation-delete-panel"
            ) as delete_confirmation:
                gr.Markdown("**确认删除这段对话？** 此操作会同时删除问答、摘要和该会话记忆。")
                with gr.Row(elem_classes="conversation-action-row"):
                    confirm_delete_button = gr.Button("确认删除", elem_classes="danger-action")
                    cancel_delete_button = gr.Button("取消")
            with gr.Group(elem_id="sidebar-bottom"):
                bottom_navigation = gr.Radio(
                    choices=[
                        ("◉  记忆", "记忆"),
                        ("⚙  设置", "设置"),
                    ],
                    value=None,
                    show_label=False,
                    elem_classes="sidebar-nav",
                )

        with gr.Group(visible=True, elem_classes="page-shell") as home_page:
            gr.HTML(
                '<section class="home-hero"><div class="page-kicker">PDF · MEMORY · AGENT</div>'
                "<h1>今天想学些什么？</h1><p>上传资料、选择课程，然后向你的本地学习助手提问。</p></section>"
            )
            chatbot = gr.Chatbot(value=[], height=150, show_label=False, elem_id="study-chatbot")
            with gr.Group(elem_id="study-composer"):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="例如：请解释最大似然估计的定义、适用条件，并给出文中的例子。",
                    lines=3,
                    max_lines=10,
                    elem_id="study-question",
                )
                with gr.Group(visible=False, elem_id="composer-panel") as composer_panel:
                    with gr.Group(visible=False) as composer_upload_panel:
                        gr.Markdown("**添加 PDF**  文件解析后会直接加入当前知识库。")
                        quick_upload_folder = gr.Dropdown(
                            choices=assistant.folder_choices(), value="", label="上传到课程文件夹"
                        )
                        quick_uploader = gr.File(
                            label="选择 PDF（支持多选）",
                            file_types=[".pdf"],
                            file_count="multiple",
                            type="filepath",
                            elem_classes="clean-upload",
                        )
                        quick_upload_button = gr.Button("解析并加入知识库", variant="primary")
                        quick_upload_status = gr.Markdown()
                    with gr.Group(visible=False) as composer_model_panel:
                        model_selector = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=assistant.selected_model(settings.deepseek_model),
                            label="DeepSeek 模型",
                        )
                    with gr.Group(visible=False) as composer_folder_panel:
                        folder_context = gr.Dropdown(
                            choices=assistant.folder_context_choices(),
                            value=ALL_FOLDERS,
                            label="课程 / 文件夹上下文",
                        )
                    with gr.Group(visible=False) as composer_mode_panel:
                        mode = gr.Radio(
                            [
                                ("仅 PDF", MODE_PDF),
                                ("PDF + 互联网", MODE_MIXED),
                                ("仅互联网", MODE_WEB),
                            ],
                            value=MODE_MIXED,
                            label="回答模式",
                        )
                    with gr.Group(visible=False) as composer_memory_panel:
                        memory_scope = gr.Radio(
                            [("仅当前对话", MEMORY_CURRENT), ("参考所有历史", MEMORY_ALL)],
                            value=MEMORY_CURRENT,
                            label="长期记忆范围",
                        )
                    with gr.Group(visible=False) as composer_pdf_panel:
                        document_selector = gr.Dropdown(
                            choices=assistant.document_choices(),
                            value=[],
                            multiselect=True,
                            label="本轮参考 PDF",
                            info="不选择时检索当前课程文件夹中的全部 PDF",
                        )
                        refresh_button = gr.Button("刷新 PDF 列表")
                    with gr.Group(visible=False) as composer_tools_panel:
                        enable_agent_tools = gr.Checkbox(
                            value=settings.enable_mcp_tools,
                            label="允许 DeepSeek 按需调用 Agent / MCP 工具",
                        )
                        tool_groups = gr.CheckboxGroup(
                            choices=TOOL_GROUP_CHOICES,
                            value=DEFAULT_TOOL_GROUPS,
                            label="可用工具组",
                            info="仅 PDF 模式会自动清空并禁用全部工具",
                        )
                        advanced = gr.Checkbox(value=False, label="扩大检索范围 / 精排")
                    with gr.Group(visible=False) as composer_trace_panel:
                        context_status = gr.Markdown(assistant.last_context_status)
                    composer_panel_close = gr.Button("收起", elem_classes="composer-panel-close")
                with gr.Row(elem_classes="composer-footer"):
                    with gr.Row(scale=8, elem_classes="composer-controls"):
                        composer_upload_button = gr.Button(
                            "＋", elem_id="composer-add-pdf", elem_classes="composer-icon"
                        )
                        composer_model_button = gr.Button(
                            "◈", elem_id="composer-model", elem_classes="composer-icon"
                        )
                        composer_folder_button = gr.Button(
                            "◫", elem_id="composer-folder", elem_classes="composer-icon"
                        )
                        composer_mode_button = gr.Button(
                            "◎", elem_id="composer-mode", elem_classes="composer-icon"
                        )
                        composer_memory_button = gr.Button(
                            "◇", elem_id="composer-memory", elem_classes="composer-icon"
                        )
                        composer_pdf_button = gr.Button(
                            "▤", elem_id="composer-pdf", elem_classes="composer-icon"
                        )
                        composer_tools_button = gr.Button(
                            "⌘", elem_id="composer-tools", elem_classes="composer-icon"
                        )
                        composer_trace_button = gr.Button(
                            "↻", elem_id="composer-trace", elem_classes="composer-icon"
                        )
                    clear_button = gr.Button("×", elem_id="study-clear")
                    ask_button = gr.Button("↑", variant="primary", elem_id="study-send")

        with gr.Group(visible=False, elem_classes="page-shell") as learning_page:
            with gr.Group(visible=True, elem_classes="detail-shell") as learning_overview:
                gr.HTML(
                    '<header class="page-header"><div class="page-kicker">WORKSPACE</div>'
                    "<h1>学习空间</h1><p>历史、笔记、课程画像、回顾和工具集中在这里。</p></header>"
                )
                with gr.Row(elem_classes="card-row"):
                    learning_history_card = gr.Button(
                        f"↻  聊天历史    ↗\n{space_stats['sessions']} 段对话\n回顾并继续此前的学习对话。",
                        elem_classes="nav-card",
                    )
                    learning_notes_card = gr.Button(
                        f"✎  学习笔记    ↗\n{space_stats['notes']} 条笔记\n整理来自 PDF、问答和研究的笔记。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    learning_profile_card = gr.Button(
                        f"◎  课程画像    ↗\n{profile_count} 门课程\n查看薄弱点、偏好和自动推断证据。",
                        elem_classes="nav-card",
                    )
                    learning_review_card = gr.Button(
                        "◌  回顾与报告    ↗\n阶段总结\n生成学习回顾、统计与可下载报告。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    learning_mcp_card = gr.Button(
                        f"⌘  MCP 服务    ↗\n{tool_count} 个工具\n集中查看计算、搜索、Zotero 与 Anki。",
                        elem_classes="nav-card",
                    )
                    learning_skills_card = gr.Button(
                        "✦  内置技能    ↗\n6 项能力\n查看助手按需使用的本地学习技能。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    learning_documents_card = gr.Button(
                        f"▦  知识资料    ↗\n{space_stats['documents']} 份 PDF\n按课程文件夹管理本地学习资料。",
                        elem_classes="nav-card",
                    )
                    learning_memory_card = gr.Button(
                        f"◇  长期记忆    ↗\n{space_stats['memories']} 条事实\n语义检索相关历史与笔记。",
                        elem_classes="nav-card",
                    )

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_history_detail:
                learning_history_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>聊天历史</h2><p>浏览、压缩并继续当前学习对话。</p></div>'
                )
                history_button = gr.Button("查看当前对话记录", variant="primary")
                history_view = gr.Markdown()
                gr.Markdown("### 长对话摘要")
                compress_button = gr.Button("现在压缩当前对话")
                compress_status = gr.Markdown()
                summary_view = gr.Markdown(assistant.session_summary_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_notes_detail:
                learning_notes_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>学习笔记</h2><p>记录知识点，并作为 [N] 证据参与回答。</p></div>'
                )
                with gr.Row():
                    note_concept = gr.Textbox(label="知识点", placeholder="例如：中心极限定理")
                    note_source = gr.Textbox(label="来源", placeholder="例如：《概率论》第 86 页")
                    note_folder = gr.Dropdown(
                        choices=assistant.folder_choices(), value="", label="课程文件夹"
                    )
                note_content = gr.Textbox(label="笔记内容", lines=7)
                note_button = gr.Button("保存笔记", variant="primary")
                note_status = gr.Markdown()
                notes_view = gr.Markdown(assistant.notes_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_profile_detail:
                learning_profile_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>课程画像</h2><p>不同课程分别累积薄弱点与解释偏好。</p></div>'
                )
                gr.Markdown("每个文档文件夹对应一份独立画像，成功问答后会自动更新薄弱点。")
                profile_scope = gr.Dropdown(
                    choices=assistant.profile_scope_choices(),
                    value="__general__",
                    label="查看 / 编辑哪门课程",
                )
                profile_courses = gr.Textbox(
                    value=profile_values[0], label="正在学习的课程 / 主题", lines=2
                )
                profile_weak_points = gr.Textbox(
                    value=profile_values[1], label="常见薄弱点", lines=3
                )
                with gr.Row():
                    profile_depth = gr.Dropdown(
                        ["简洁", "标准", "深入"], value=profile_values[2], label="解释深度"
                    )
                    profile_math_style = gr.Dropdown(
                        ["直观优先", "定义证明优先", "公式推导优先", "例题优先"],
                        value=profile_values[3],
                        label="数学表达风格",
                    )
                profile_preferences = gr.Textbox(value=profile_values[4], label="其他偏好", lines=3)
                with gr.Row():
                    save_profile_button = gr.Button("保存画像", variant="primary")
                    auto_profile_button = gr.Button("根据该课程历史重新分析")
                profile_status = gr.Markdown(assistant.load_learning_profile("__general__")[-1])

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_review_detail:
                learning_review_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>回顾与报告</h2><p>从问答、笔记和学习统计生成阶段总结。</p></div>'
                )
                review_button = gr.Button("生成学习回顾", variant="primary")
                review_view = gr.Markdown("完成问答或添加笔记后，可以在这里生成回顾。")
                stats_button = gr.Button("刷新统计")
                stats_view = gr.Markdown(assistant.stats_markdown())
                report_button = gr.Button("生成 Markdown / JSON 报告", variant="primary")
                report_file = gr.File(label="下载报告")

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_mcp_detail:
                learning_mcp_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>MCP 服务</h2><p>检查模型可按需使用的计算、检索与学习工具。</p></div>'
                )
                gr.Markdown(
                    "Python/SymPy、学术搜索、网页调查、受限文件、SQLite、Zotero 和 Anki "
                    "均由 Agent 工具注册表管理。"
                )
                tool_status_button = gr.Button("检测全部 MCP 工具", variant="primary")
                tool_status = gr.Markdown(assistant.mcp_client.status_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_skills_detail:
                learning_skills_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>内置技能</h2><p>助手根据问题按需使用的本地学习能力。</p></div>'
                )
                gr.Markdown(
                    "- **结构化 PDF 理解**：正文、公式、表格、页面图像与 OCR\n"
                    "- **上下文问题改写**：把追问还原成可独立检索的问题\n"
                    "- **混合知识检索**：BGE-M3 向量与 BM25 融合\n"
                    "- **长期记忆检索**：按问题提取相关历史问答和笔记\n"
                    "- **课程画像分析**：按课程推断薄弱点与解释偏好\n"
                    "- **学习回顾与报告**：生成阶段回顾和本地报告"
                )

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_documents_detail:
                learning_documents_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>知识资料</h2><p>查看已入库 PDF 及其课程分类。</p></div>'
                )
                learning_documents_refresh = gr.Button("刷新资料列表")
                learning_documents_view = gr.Markdown(assistant.documents_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as learning_memory_detail:
                learning_memory_back = gr.Button("←  学习空间", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>长期记忆</h2><p>查看对当前学习有帮助的 L2 可检索事实。</p></div>'
                )
                learning_memory_button = gr.Button("刷新长期记忆")
                learning_memory_view = gr.Markdown(assistant.memory_l2_markdown())

        with gr.Group(visible=False, elem_classes="page-shell") as memory_page:
            with gr.Group(visible=True, elem_classes="detail-shell") as memory_page_overview:
                gr.HTML(
                    '<header class="page-header"><div class="page-kicker">AUDITABLE MEMORY</div>'
                    "<h1>记忆</h1><p>L1 原始事件、L2 可检索事实、L3 跨会话综合，逐层可追溯。</p></header>"
                )
                memory_refresh = gr.Button("刷新三层记忆", elem_classes="compact-refresh")
                with gr.Row(elem_classes="card-row"):
                    memory_graph_card = gr.Button(
                        f"◌  记忆图谱    ↗\n{memory_counts['l1'] + memory_counts['l2'] + memory_counts['l3']} 个记忆节点\n在一张关系图中查看 L1、L2 与 L3 的追溯关系。",
                        elem_classes="nav-card",
                    )
                    memory_l1_card = gr.Button(
                        f"▱  L1 · 学习事件    ↗\n{memory_counts['l1']} 条追踪\n提问、笔记、资料导入和画像更新的原始事件。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    memory_l2_card = gr.Button(
                        f"⌘  L2 · 可检索事实    ↗\n{memory_counts['l2']} 条事实\n按问答与笔记整理并建立向量索引的学习事实。",
                        elem_classes="nav-card",
                    )
                    memory_l3_card = gr.Button(
                        f"◎  L3 · 跨会话综合    ↗\n{memory_counts['l3']} 条综合\n课程画像、偏好、薄弱点和长对话摘要。",
                        elem_classes="nav-card",
                    )

            with gr.Group(visible=False, elem_classes="detail-shell") as memory_graph_detail:
                memory_graph_back = gr.Button("←  记忆", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>记忆图谱</h2><p>中心为 L3 综合，中圈为 L2 事实，外圈为 L1 原始事件。</p></div>'
                )
                memory_graph = gr.HTML(assistant.memory_graph_html())

            with gr.Group(visible=False, elem_classes="detail-shell") as memory_l1_detail:
                memory_l1_back = gr.Button("←  记忆", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>L1 · 学习事件</h2><p>仅追加的问答、笔记、资料导入与画像更新记录。</p></div>'
                )
                memory_l1 = gr.Markdown(assistant.memory_l1_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as memory_l2_detail:
                memory_l2_back = gr.Button("←  记忆", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>L2 · 可检索事实</h2><p>已整理并建立向量索引的问答与笔记事实。</p></div>'
                )
                memory_l2 = gr.Markdown(assistant.memory_l2_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as memory_l3_detail:
                memory_l3_back = gr.Button("←  记忆", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>L3 · 跨会话综合</h2><p>按课程沉淀的学习画像、薄弱点和对话摘要。</p></div>'
                )
                memory_l3 = gr.Markdown(assistant.memory_l3_markdown())

        with gr.Group(visible=False, elem_classes="page-shell") as knowledge_page:
            with gr.Group(visible=True, elem_classes="detail-shell") as knowledge_overview:
                gr.HTML(
                    '<header class="page-header"><div class="page-kicker">LOCAL RAG</div>'
                    "<h1>知识中心</h1><p>管理 PDF 知识库、课程文件夹和本地混合检索引擎。</p></header>"
                )
                with gr.Row(elem_classes="card-row"):
                    knowledge_hybrid_card = gr.Button(
                        f"◉  本地混合检索    ↗\n{space_stats['chunks']} 个片段\nBGE-M3 向量与 BM25 关键词融合检索。",
                        elem_classes="nav-card",
                    )
                    reranker_state = "已启用" if settings.enable_reranker else "按需启用"
                    knowledge_reranker_card = gr.Button(
                        f"⇅  精排模型    ↗\n{reranker_state}\n复杂问题可对召回证据二次排序。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    knowledge_structured_card = gr.Button(
                        f"▦  结构化 PDF    ↗\n{space_stats['pages']} 页\n分别理解正文、公式、表格和页面图像。",
                        elem_classes="nav-card",
                    )
                    knowledge_graph_card = gr.Button(
                        f"⌁  本地 GraphRAG    ↗\n{graph_status_data['entity_count']} 个实体\n概念关系图与可追溯页码检索。",
                        elem_classes="nav-card",
                    )
                with gr.Row(elem_classes="card-row"):
                    knowledge_library_card = gr.Button(
                        f"▤  知识库管理    ↗\n{space_stats['documents']} 份 PDF\n上传、建文件夹并整理课程资料。",
                        elem_classes="nav-card",
                    )
                    knowledge_search_card = gr.Button(
                        "⌕  检索引擎测试    ↗\n查看证据\n输入问题检查本地召回的页码与内容。",
                        elem_classes="nav-card",
                    )

            with gr.Group(visible=False, elem_classes="detail-shell") as knowledge_hybrid_detail:
                knowledge_hybrid_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>本地混合检索</h2><p>语义向量与关键词双路召回。</p></div>'
                )
                gr.Markdown(
                    "### BGE-M3 + BM25\n\n"
                    "向量语义检索和关键词检索会融合排序；问答时自动启用。\n\n"
                    f"当前索引片段：**{space_stats['chunks']}**"
                )

            with gr.Group(visible=False, elem_classes="detail-shell") as knowledge_reranker_detail:
                knowledge_reranker_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>精排模型</h2><p>对初次召回的证据进行二次相关度排序。</p></div>'
                )
                gr.Markdown(
                    f"当前模型：`{settings.reranker_model}`\n\n"
                    "在首页开启“扩大检索范围 / 精排”后使用；复杂问题更准确，但速度较慢。"
                )

            with gr.Group(
                visible=False, elem_classes="detail-shell"
            ) as knowledge_structured_detail:
                knowledge_structured_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>结构化 PDF</h2><p>正文、公式、表格和页面图像的多路解析。</p></div>'
                )
                gr.Markdown(
                    "分别索引正文、公式、表格和页面图像/OCR；检索结果会保留 PDF 名称、"
                    "页码和证据类型。"
                )
                structured_documents_view = gr.Markdown(assistant.documents_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as knowledge_graph_detail:
                knowledge_graph_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>本地 GraphRAG</h2><p>构建概念共现图并查询可追溯关系证据。</p></div>'
                )
                gr.Markdown(
                    "从 PDF 片段抽取概念、英文术语和公式实体，建立共现关系图；"
                    "查询时进行实体匹配、邻居扩展并返回可追溯页码。"
                )
                graph_build_button = gr.Button("构建 / 重建 GraphRAG", variant="primary")
                graph_build_status = gr.Markdown(assistant.graph_rag.status_markdown())
                graph_visual = gr.HTML(assistant.graph_rag.graph_html())
                graph_folder = gr.Dropdown(
                    choices=assistant.folder_context_choices(),
                    value=ALL_FOLDERS,
                    label="GraphRAG 查询范围",
                )
                graph_query = gr.Textbox(
                    label="图谱问题", placeholder="例如：最大似然估计与 Fisher 信息有什么关系？"
                )
                graph_search_button = gr.Button("查询 GraphRAG")
                graph_results = gr.Markdown()

            with gr.Group(visible=False, elem_classes="detail-shell") as knowledge_library_detail:
                knowledge_library_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>知识库管理</h2><p>按课程文件夹上传、整理并查看 PDF。</p></div>'
                )
                with gr.Group(elem_classes="section-card"):
                    with gr.Row():
                        folder_name = gr.Textbox(
                            label="新课程 / 文件夹名称",
                            placeholder="例如：数理统计、回归分析、论文",
                        )
                        create_folder_button = gr.Button("创建文件夹", variant="secondary")
                    folder_status = gr.Markdown()
                upload_folder = gr.Dropdown(
                    choices=assistant.folder_choices(), value="", label="本次上传到"
                )
                uploader = gr.File(
                    label="上传 PDF（支持多选）",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                    elem_classes="clean-upload",
                )
                upload_button = gr.Button("解析并加入知识中心", variant="primary")
                upload_status = gr.Markdown("等待上传。", elem_classes="status-card")
                with gr.Accordion("整理已入库 PDF", open=False):
                    library_document_selector = gr.Dropdown(
                        choices=assistant.document_choices(),
                        value=[],
                        multiselect=True,
                        label="选择要移动的 PDF",
                    )
                    move_target_folder = gr.Dropdown(
                        choices=assistant.folder_choices(), value="", label="目标文件夹"
                    )
                    move_documents_button = gr.Button("移动到文件夹")
                    move_status = gr.Markdown()
                documents_view = gr.Markdown(assistant.documents_markdown())

            with gr.Group(visible=False, elem_classes="detail-shell") as knowledge_search_detail:
                knowledge_search_back = gr.Button("←  知识中心", elem_classes="detail-back")
                gr.HTML(
                    '<div class="detail-title"><h2>检索引擎测试</h2><p>手动验证知识中心召回的内容与页码。</p></div>'
                )
                knowledge_folder = gr.Dropdown(
                    choices=assistant.folder_context_choices(),
                    value=ALL_FOLDERS,
                    label="检索范围",
                )
                knowledge_query = gr.Textbox(
                    label="检索词 / 问题", placeholder="例如：充分统计量的定义与例子"
                )
                knowledge_search_button = gr.Button("检索知识中心", variant="primary")
                knowledge_results = gr.Markdown()

        config_status = (
            "✅ DeepSeek 已配置"
            if assistant.llm.configured
            else "⚠️ DeepSeek Key 未配置；上传和索引可用，回答暂不可用"
        )
        with gr.Group(visible=False, elem_classes="page-shell") as settings_page:
            gr.HTML(
                '<header class="page-header"><div class="page-kicker">CONTROL PANEL</div>'
                "<h1>设置</h1><p>检查模型、联网搜索、本地服务和数据保存位置。</p></header>"
            )
            with gr.Row():
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown(
                        f"### 模型\n\n{config_status}\n\n当前：`{settings.deepseek_model}`",
                        elem_classes="settings-card",
                    )
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown(
                        f"### 本地数据\n\n`{settings.data_dir}`\n\n"
                        f"嵌入模型：`{settings.embedding_model}`",
                        elem_classes="settings-card",
                    )
            search_status = gr.Markdown(assistant.web_status_markdown())
            search_status_button = gr.Button("检测联网搜索服务")
            gr.Markdown(
                "### 隐私\n\nPDF、向量索引、三层记忆、画像和历史保存在本机。"
                "DeepSeek 只接收问题和检索出的相关证据，不会上传整本 PDF。"
            )

        composer_panel_outputs = [
            composer_panel,
            composer_upload_panel,
            composer_model_panel,
            composer_folder_panel,
            composer_mode_panel,
            composer_memory_panel,
            composer_pdf_panel,
            composer_tools_panel,
            composer_trace_panel,
        ]
        for button, panel_name in (
            (composer_upload_button, "upload"),
            (composer_model_button, "model"),
            (composer_folder_button, "folder"),
            (composer_mode_button, "mode"),
            (composer_memory_button, "memory"),
            (composer_pdf_button, "pdf"),
            (composer_tools_button, "tools"),
            (composer_trace_button, "trace"),
        ):
            button.click(
                lambda name=panel_name: show_composer_panel(name),
                outputs=composer_panel_outputs,
            )
        composer_panel_close.click(
            lambda: show_composer_panel(""),
            outputs=composer_panel_outputs,
        )

        learning_section_outputs = [
            learning_overview,
            learning_history_detail,
            learning_notes_detail,
            learning_profile_detail,
            learning_review_detail,
            learning_mcp_detail,
            learning_skills_detail,
            learning_documents_detail,
            learning_memory_detail,
        ]
        for button, section in (
            (learning_history_card, "history"),
            (learning_notes_card, "notes"),
            (learning_profile_card, "profile"),
            (learning_review_card, "review"),
            (learning_mcp_card, "mcp"),
            (learning_skills_card, "skills"),
            (learning_documents_card, "documents"),
            (learning_memory_card, "memory"),
        ):
            button.click(
                lambda name=section: show_learning_section(name),
                outputs=learning_section_outputs,
            )
        for button in (
            learning_history_back,
            learning_notes_back,
            learning_profile_back,
            learning_review_back,
            learning_mcp_back,
            learning_skills_back,
            learning_documents_back,
            learning_memory_back,
        ):
            button.click(
                lambda: show_learning_section("overview"),
                outputs=learning_section_outputs,
            )

        memory_section_outputs = [
            memory_page_overview,
            memory_graph_detail,
            memory_l1_detail,
            memory_l2_detail,
            memory_l3_detail,
        ]
        for button, section in (
            (memory_graph_card, "graph"),
            (memory_l1_card, "l1"),
            (memory_l2_card, "l2"),
            (memory_l3_card, "l3"),
        ):
            button.click(
                lambda name=section: show_memory_section(name),
                outputs=memory_section_outputs,
            )
        for button in (memory_graph_back, memory_l1_back, memory_l2_back, memory_l3_back):
            button.click(
                lambda: show_memory_section("overview"),
                outputs=memory_section_outputs,
            )

        knowledge_section_outputs = [
            knowledge_overview,
            knowledge_hybrid_detail,
            knowledge_reranker_detail,
            knowledge_structured_detail,
            knowledge_graph_detail,
            knowledge_library_detail,
            knowledge_search_detail,
        ]
        for button, section in (
            (knowledge_hybrid_card, "hybrid"),
            (knowledge_reranker_card, "reranker"),
            (knowledge_structured_card, "structured"),
            (knowledge_graph_card, "graph"),
            (knowledge_library_card, "library"),
            (knowledge_search_card, "search"),
        ):
            button.click(
                lambda name=section: show_knowledge_section(name),
                outputs=knowledge_section_outputs,
            )
        for button in (
            knowledge_hybrid_back,
            knowledge_reranker_back,
            knowledge_structured_back,
            knowledge_graph_back,
            knowledge_library_back,
            knowledge_search_back,
        ):
            button.click(
                lambda: show_knowledge_section("overview"),
                outputs=knowledge_section_outputs,
            )

        ask_button.click(
            ask_question,
            inputs=[
                question,
                document_selector,
                mode,
                advanced,
                model_selector,
                memory_scope,
                folder_context,
                enable_agent_tools,
                tool_groups,
                chatbot,
            ],
            outputs=[chatbot, question, conversation_selector, context_status],
        )
        main_navigation.change(
            show_navigation_page,
            inputs=main_navigation,
            outputs=[
                home_page,
                learning_page,
                memory_page,
                knowledge_page,
                settings_page,
                bottom_navigation,
            ],
        )
        bottom_navigation.change(
            show_navigation_page,
            inputs=bottom_navigation,
            outputs=[
                home_page,
                learning_page,
                memory_page,
                knowledge_page,
                settings_page,
                main_navigation,
            ],
        )
        question.submit(
            ask_question,
            inputs=[
                question,
                document_selector,
                mode,
                advanced,
                model_selector,
                memory_scope,
                folder_context,
                enable_agent_tools,
                tool_groups,
                chatbot,
            ],
            outputs=[chatbot, question, conversation_selector, context_status],
        )
        home_button.click(
            new_conversation,
            outputs=[chatbot, conversation_selector, question, context_status, summary_view],
        )
        home_button.click(
            show_new_home,
            outputs=[
                home_page,
                learning_page,
                memory_page,
                knowledge_page,
                settings_page,
                main_navigation,
                bottom_navigation,
            ],
        )
        conversation_selector.change(
            open_conversation,
            inputs=conversation_selector,
            outputs=[
                chatbot,
                summary_view,
                home_page,
                learning_page,
                memory_page,
                knowledge_page,
                settings_page,
                main_navigation,
                bottom_navigation,
                delete_confirmation,
            ],
        )
        conversation_rename_icon.click(
            lambda: gr.Group(visible=False),
            outputs=delete_confirmation,
            js="() => window.__studyMindToggleInlineRename?.()",
        )
        conversation_delete_icon.click(
            lambda: gr.Group(visible=True),
            outputs=delete_confirmation,
            js="() => window.__studyMindCancelInlineRename?.()",
        )
        rename_conversation_button.click(
            rename_conversation_ui,
            inputs=[conversation_selector, rename_title],
            outputs=[conversation_action_status, conversation_selector, rename_title],
        )
        cancel_delete_button.click(
            lambda: gr.Group(visible=False),
            outputs=delete_confirmation,
        )
        confirm_delete_button.click(
            delete_conversation_ui,
            inputs=conversation_selector,
            outputs=[
                chatbot,
                conversation_selector,
                question,
                context_status,
                summary_view,
                conversation_action_status,
                delete_confirmation,
                home_page,
                learning_page,
                memory_page,
                knowledge_page,
                settings_page,
                main_navigation,
                bottom_navigation,
            ],
        )
        clear_button.click(list, outputs=chatbot)
        upload_button.click(
            upload_documents,
            inputs=[uploader, document_selector, upload_folder, folder_context],
            outputs=[
                upload_status,
                document_selector,
                library_document_selector,
                documents_view,
                learning_documents_view,
                structured_documents_view,
                uploader,
            ],
        )
        quick_upload_button.click(
            upload_documents,
            inputs=[quick_uploader, document_selector, quick_upload_folder, folder_context],
            outputs=[
                quick_upload_status,
                document_selector,
                library_document_selector,
                documents_view,
                learning_documents_view,
                structured_documents_view,
                quick_uploader,
            ],
        )
        refresh_button.click(
            refresh_documents,
            inputs=[document_selector, library_document_selector, folder_context],
            outputs=[
                document_selector,
                library_document_selector,
                documents_view,
                learning_documents_view,
                structured_documents_view,
            ],
        )
        folder_context.change(
            select_folder_context,
            inputs=[folder_context, document_selector],
            outputs=document_selector,
        )
        mode.change(
            sync_tools_with_answer_mode,
            inputs=mode,
            outputs=tool_groups,
        )
        create_folder_button.click(
            create_document_folder,
            inputs=folder_name,
            outputs=[
                folder_status,
                folder_name,
                upload_folder,
                move_target_folder,
                folder_context,
                note_folder,
                profile_scope,
                knowledge_folder,
                graph_folder,
                quick_upload_folder,
            ],
        )
        folder_name.submit(
            create_document_folder,
            inputs=folder_name,
            outputs=[
                folder_status,
                folder_name,
                upload_folder,
                move_target_folder,
                folder_context,
                note_folder,
                profile_scope,
                knowledge_folder,
                graph_folder,
                quick_upload_folder,
            ],
        )
        move_documents_button.click(
            move_library_documents,
            inputs=[
                library_document_selector,
                move_target_folder,
                document_selector,
                folder_context,
            ],
            outputs=[
                move_status,
                document_selector,
                library_document_selector,
                documents_view,
                learning_documents_view,
                structured_documents_view,
            ],
        )
        note_button.click(
            save_note,
            inputs=[note_content, note_concept, note_source, note_folder],
            outputs=[note_status, note_content, notes_view, learning_memory_view],
        )
        save_profile_button.click(
            save_profile,
            inputs=[
                profile_scope,
                profile_courses,
                profile_weak_points,
                profile_depth,
                profile_math_style,
                profile_preferences,
            ],
            outputs=profile_status,
        )
        auto_profile_button.click(
            assistant.auto_update_learning_profile,
            inputs=[profile_scope, model_selector],
            outputs=[
                profile_courses,
                profile_weak_points,
                profile_depth,
                profile_math_style,
                profile_preferences,
                profile_status,
            ],
        )
        profile_scope.change(
            assistant.load_learning_profile,
            inputs=profile_scope,
            outputs=[
                profile_courses,
                profile_weak_points,
                profile_depth,
                profile_math_style,
                profile_preferences,
                profile_status,
            ],
        )
        review_button.click(assistant.learning_review, inputs=model_selector, outputs=review_view)
        history_button.click(assistant.history_markdown, outputs=history_view)
        compress_button.click(
            compress_conversation,
            inputs=model_selector,
            outputs=[compress_status, summary_view],
        )
        stats_button.click(assistant.stats_markdown, outputs=stats_view)
        report_button.click(assistant.generate_report, outputs=report_file)
        search_status_button.click(assistant.web_status_markdown, outputs=search_status)
        tool_status_button.click(assistant.mcp_client.status_markdown, outputs=tool_status)
        learning_documents_refresh.click(
            assistant.documents_markdown,
            outputs=[learning_documents_view],
        )
        learning_memory_button.click(assistant.memory_l2_markdown, outputs=learning_memory_view)
        memory_refresh.click(
            refresh_memory_views,
            outputs=[memory_graph, memory_l1, memory_l2, memory_l3],
        )
        knowledge_search_button.click(
            assistant.knowledge_search,
            inputs=[knowledge_query, knowledge_folder],
            outputs=knowledge_results,
        )
        knowledge_query.submit(
            assistant.knowledge_search,
            inputs=[knowledge_query, knowledge_folder],
            outputs=knowledge_results,
        )
        graph_build_button.click(
            build_graph_rag_ui,
            outputs=[graph_build_status, graph_visual],
        )
        graph_search_button.click(
            assistant.graph_rag_search,
            inputs=[graph_query, graph_folder],
            outputs=graph_results,
        )
        graph_query.submit(
            assistant.graph_rag_search,
            inputs=[graph_query, graph_folder],
            outputs=graph_results,
        )

    return demo, settings
