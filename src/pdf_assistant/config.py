from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _detect_project_root() -> Path:
    explicit = os.getenv("PDF_ASSISTANT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "app.py").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _detect_project_root()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    uploads_dir: Path
    page_assets_dir: Path
    vector_dir: Path
    reports_dir: Path
    database_path: Path

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str

    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    enable_reranker: bool
    reranker_model: str

    chunk_size: int
    chunk_overlap: int
    max_pdf_mb: int
    max_pdf_pages: int
    enable_page_ocr: bool

    searxng_url: str
    enable_mcp_tools: bool
    mcp_max_tool_rounds: int
    learning_files_root: Path
    zotero_local_url: str
    zotero_user_id: str
    zotero_api_key: str
    anki_connect_url: str
    academic_email: str
    app_host: str
    app_port: int

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        raw_data_dir = Path(os.getenv("DATA_DIR", "./data"))
        data_dir = raw_data_dir if raw_data_dir.is_absolute() else PROJECT_ROOT / raw_data_dir
        data_dir = data_dir.resolve()
        raw_learning_root = os.getenv("LEARNING_FILES_ROOT", "").strip()
        learning_files_root = (
            Path(raw_learning_root).expanduser().resolve()
            if raw_learning_root
            else (data_dir / "uploads").resolve()
        )

        settings = cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            uploads_dir=data_dir / "uploads",
            page_assets_dir=data_dir / "page_assets",
            vector_dir=data_dir / "qdrant",
            reports_dir=data_dir / "reports",
            database_path=data_dir / "assistant.sqlite3",
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "auto").strip().lower(),
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))),
            enable_reranker=_as_bool(os.getenv("ENABLE_RERANKER"), False),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
            chunk_size=max(300, int(os.getenv("CHUNK_SIZE", "850"))),
            chunk_overlap=max(0, int(os.getenv("CHUNK_OVERLAP", "100"))),
            max_pdf_mb=max(1, int(os.getenv("MAX_PDF_MB", "200"))),
            max_pdf_pages=max(1, int(os.getenv("MAX_PDF_PAGES", "3000"))),
            enable_page_ocr=_as_bool(os.getenv("ENABLE_PAGE_OCR"), True),
            searxng_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/"),
            enable_mcp_tools=_as_bool(os.getenv("ENABLE_MCP_TOOLS"), True),
            mcp_max_tool_rounds=max(1, min(8, int(os.getenv("MCP_MAX_TOOL_ROUNDS", "4")))),
            learning_files_root=learning_files_root,
            zotero_local_url=os.getenv("ZOTERO_LOCAL_URL", "http://127.0.0.1:23119/api").rstrip(
                "/"
            ),
            zotero_user_id=os.getenv("ZOTERO_USER_ID", "").strip(),
            zotero_api_key=os.getenv("ZOTERO_API_KEY", "").strip(),
            anki_connect_url=os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765").rstrip("/"),
            academic_email=os.getenv("ACADEMIC_EMAIL", "").strip(),
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "7860")),
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.page_assets_dir,
            self.vector_dir,
            self.reports_dir,
            self.learning_files_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
