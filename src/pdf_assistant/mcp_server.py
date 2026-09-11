from __future__ import annotations

import ast
import fnmatch
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import fitz
import httpx
import sympy as sp
from mcp.server.fastmcp import FastMCP

from .config import Settings

settings = Settings.load()
mcp = FastMCP(
    "PDF 学习助手工具",
    instructions=(
        "面向数学统计学习的安全工具集。文件系统限制在学习目录，SQLite 只读；"
        "只有用户明确要求制作、修改或安排 Anki 卡片时才可调用 anki 写入工具。"
    ),
    log_level="ERROR",
)

HTTP_HEADERS = {"User-Agent": "LocalPDFLearningAssistant/0.1"}
ALLOWED_FILE_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".py",
    ".ipynb",
}
ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "random",
    "itertools",
    "numpy",
    "pandas",
    "scipy",
    "sympy",
    "statsmodels",
    "sklearn",
}
FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "getattr",
    "help",
    "input",
    "locals",
    "object",
    "open",
    "quit",
    "exit",
    "setattr",
    "delattr",
    "dir",
    "type",
    "vars",
    "__import__",
}
FORBIDDEN_ATTRIBUTES = {
    "dump",
    "dumps",
    "load",
    "loads",
    "loadmat",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "memmap",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_hdf",
    "read_html",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_table",
    "urlopen",
    "save",
    "savefig",
    "savetxt",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _clean_html(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def _safe_learning_path(relative_path: str = "") -> Path:
    root = settings.learning_files_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("路径超出允许的学习资料目录。")
    return candidate


def _http_get(url: str, *, params: dict | None = None, headers: dict | None = None):
    merged_headers = {**HTTP_HEADERS, **(headers or {})}
    with httpx.Client(timeout=15, follow_redirects=True, headers=merged_headers) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response


def _anki_request(action: str, params: dict | None = None) -> Any:
    payload = {"action": action, "version": 6, "params": params or {}}
    try:
        response = httpx.post(settings.anki_connect_url, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError("无法连接 AnkiConnect。请打开 Anki 并确认已安装 AnkiConnect。") from exc
    if data.get("error"):
        raise RuntimeError(f"AnkiConnect 错误：{data['error']}")
    return data.get("result")


@mcp.tool()
def math_sympy(
    expression: str,
    operation: str = "simplify",
    variable: str = "x",
    point: str = "0",
) -> str:
    """用 SymPy 进行公式化简、因式分解、展开、求解、微分、积分或极限验证。

    operation 可选 simplify、factor、expand、solve、diff、integrate、limit、latex。
    limit 操作使用 point 指定趋近点。表达式使用 ** 表示乘方。
    """
    expression = expression.strip().replace("^", "**")
    if not expression or len(expression) > 1200:
        raise ValueError("表达式为空或过长。")
    if not re.fullmatch(r"[A-Za-z0-9\s+\-*/().,=<>&|!\[\]]+", expression):
        raise ValueError("表达式包含不允许的字符。")
    symbol_names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", expression))
    allowed_functions = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "gamma": sp.gamma,
        "factorial": sp.factorial,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
    }
    local_dict: dict[str, Any] = dict(allowed_functions)
    for name in symbol_names - set(allowed_functions):
        local_dict[name] = sp.Symbol(name, real=True)
    expr = sp.sympify(expression, locals=local_dict)
    var = local_dict.get(variable) or sp.Symbol(variable, real=True)
    operations = {
        "simplify": lambda: sp.simplify(expr),
        "factor": lambda: sp.factor(expr),
        "expand": lambda: sp.expand(expr),
        "solve": lambda: sp.solve(expr, var),
        "diff": lambda: sp.diff(expr, var),
        "integrate": lambda: sp.integrate(expr, var),
        "limit": lambda: sp.limit(expr, var, sp.sympify(point, locals=local_dict)),
        "latex": lambda: sp.latex(expr),
    }
    if operation not in operations:
        raise ValueError(f"不支持的 operation：{operation}")
    result = operations[operation]()
    return _json(
        {
            "operation": operation,
            "input": expression,
            "result": str(result),
            "latex": sp.latex(result) if operation != "latex" else str(result),
        }
    )


def _validate_python(code: str) -> ast.Module:
    if not code.strip() or len(code) > 6000:
        raise ValueError("Python 代码为空或超过 6000 字符。")
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Global, ast.Nonlocal)
        ):
            raise TypeError(f"不允许使用 {type(node).__name__}。")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"不允许导入 {alias.name}。")
        if isinstance(node, ast.ImportFrom) and (
            not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS
        ):
            raise ValueError(f"不允许导入 {node.module}。")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"不允许使用 {node.id}。")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ValueError("字符串中不允许出现双下划线属性名。")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES
        ):
            raise ValueError(f"不允许访问属性 {node.attr}。")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[tree.body[-1].value],
                keywords=[],
            )
        )
        ast.fix_missing_locations(tree)
    return tree


def _resource_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        memory_limit = 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except (ImportError, OSError, ValueError):
        return


@mcp.tool()
def math_python_statistics(code: str) -> str:
    """在受限 Python 子进程中执行数学统计代码并返回标准输出。

    可用 math、statistics、NumPy、pandas、SciPy、SymPy、statsmodels、sklearn。
    禁止文件写入、网络、系统命令和动态代码执行；最长运行 10 秒。
    """
    tree = _validate_python(code)
    compiled_code = ast.unparse(tree)
    workspace = settings.data_dir / "tool_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", compiled_code],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            preexec_fn=_resource_limits if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Python 计算超过 10 秒，已终止。") from exc
    output = completed.stdout[-12000:].strip()
    error = completed.stderr[-4000:].strip()
    return _json(
        {
            "exit_code": completed.returncode,
            "stdout": output,
            "stderr": error,
        }
    )


@mcp.tool()
def files_list(subdirectory: str = "", pattern: str = "*", limit: int = 100) -> str:
    """列出受限学习资料目录中的文件，支持文件名通配符。不会访问目录之外的路径。"""
    directory = _safe_learning_path(subdirectory)
    if not directory.exists() or not directory.is_dir():
        raise ValueError("目录不存在。")
    limit = max(1, min(300, int(limit)))
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_FILE_SUFFIXES:
            continue
        relative = path.relative_to(settings.learning_files_root)
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        rows.append({"path": str(relative), "size": path.stat().st_size})
        if len(rows) >= limit:
            break
    return _json({"root": str(settings.learning_files_root), "files": rows})


@mcp.tool()
def files_read(relative_path: str, max_chars: int = 12000) -> str:
    """读取受限学习资料目录中的 PDF 或文本型文件，返回截断后的内容。"""
    path = _safe_learning_path(relative_path)
    if not path.is_file() or path.suffix.lower() not in ALLOWED_FILE_SUFFIXES:
        raise ValueError("文件不存在或类型不受支持。")
    max_chars = max(1000, min(30000, int(max_chars)))
    if path.suffix.lower() == ".pdf":
        parts: list[str] = []
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                parts.append(f"[第 {page_number} 页]\n{page.get_text('text')}")
                if sum(map(len, parts)) >= max_chars:
                    break
        content = "\n\n".join(parts)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")
    return content[:max_chars]


@mcp.tool()
def sqlite_schema() -> str:
    """返回学习助手 SQLite 数据库中可查询的表和字段。数据库始终以只读方式打开。"""
    uri = f"file:{settings.database_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        schema = {}
        for table in tables:
            schema[table] = [
                {"name": row[1], "type": row[2]}
                for row in db.execute(f'PRAGMA table_info("{table}")')
            ]
    return _json(schema)


@mcp.tool()
def sqlite_query(sql: str, limit: int = 50) -> str:
    """只读查询学习记录、笔记、文档和统计。仅允许单条 SELECT 或 WITH 查询。"""
    normalized = " ".join(sql.strip().split())
    if not normalized or ";" in normalized.rstrip(";"):
        raise ValueError("只允许一条 SQL 查询。")
    if not re.match(r"^(SELECT|WITH)\b", normalized, re.IGNORECASE):
        raise ValueError("只允许 SELECT 或 WITH 查询。")
    if re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|VACUUM|REPLACE|CREATE|TRIGGER)\b",
        normalized,
        re.IGNORECASE,
    ):
        raise ValueError("查询包含写入或管理语句，已拒绝。")
    limit = max(1, min(200, int(limit)))
    uri = f"file:{settings.database_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only = ON")
        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1
            return int(steps > 5000)

        db.set_progress_handler(progress, 1000)
        cursor = db.execute(normalized)
        rows = [dict(row) for row in cursor.fetchmany(limit)]
        columns = [item[0] for item in cursor.description or []]
    return _json({"columns": columns, "rows": rows, "returned": len(rows)})


@mcp.tool()
def web_search(query: str, limit: int = 5) -> str:
    """通过本机 SearXNG 搜索互联网，适合需要最新网页资料或多轮调查的问题。"""
    limit = max(1, min(10, int(limit)))
    response = _http_get(
        f"{settings.searxng_url}/search",
        params={"q": query, "format": "json", "language": "zh-CN", "safesearch": 1},
    )
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": _clean_html(item.get("content", "")),
            "engine": item.get("engine", ""),
        }
        for item in response.json().get("results", [])[:limit]
    ]
    return _json(results)


def _arxiv_search(query: str, limit: int) -> list[dict]:
    response = _http_get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f'all:"{query}"', "start": 0, "max_results": limit},
    )
    root = ET.fromstring(response.content)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    rows = [
        {
            "source": "arXiv",
            "title": " ".join((entry.findtext("atom:title", "", namespace)).split()),
            "authors": [
                node.findtext("atom:name", "", namespace)
                for node in entry.findall("atom:author", namespace)
            ],
            "published": entry.findtext("atom:published", "", namespace),
            "url": entry.findtext("atom:id", "", namespace),
            "abstract": " ".join((entry.findtext("atom:summary", "", namespace)).split())[:1500],
        }
        for entry in root.findall("atom:entry", namespace)
    ]
    return rows


def _openalex_search(query: str, limit: int) -> list[dict]:
    params: dict[str, Any] = {"search": query, "per-page": limit}
    if settings.academic_email:
        params["mailto"] = settings.academic_email
    data = _http_get("https://api.openalex.org/works", params=params).json()
    rows = [
        {
            "source": "OpenAlex",
            "title": item.get("display_name", ""),
            "authors": [
                x.get("author", {}).get("display_name", "") for x in item.get("authorships", [])[:8]
            ],
            "year": item.get("publication_year"),
            "doi": item.get("doi"),
            "url": item.get("primary_location", {}).get("landing_page_url") or item.get("id"),
            "cited_by_count": item.get("cited_by_count", 0),
        }
        for item in data.get("results", [])
    ]
    return rows


def _crossref_search(query: str, limit: int) -> list[dict]:
    headers = {}
    if settings.academic_email:
        headers["User-Agent"] = f"LocalPDFLearningAssistant/0.1 (mailto:{settings.academic_email})"
    data = _http_get(
        "https://api.crossref.org/works",
        params={"query": query, "rows": limit},
        headers=headers,
    ).json()
    rows = []
    for item in data.get("message", {}).get("items", []):
        authors = [
            " ".join([author.get("given", ""), author.get("family", "")]).strip()
            for author in item.get("author", [])[:8]
        ]
        rows.append(
            {
                "source": "Crossref",
                "title": (item.get("title") or [""])[0],
                "authors": authors,
                "year": ((item.get("published") or {}).get("date-parts") or [[None]])[0][0],
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "publisher": item.get("publisher", ""),
            }
        )
    return rows


@mcp.tool()
def academic_search(
    query: str,
    sources: str = "arxiv,openalex,crossref",
    limit_per_source: int = 3,
) -> str:
    """检索 arXiv、OpenAlex 和 Crossref 的论文元数据、DOI、作者和引用量。

    sources 使用逗号分隔；例如 arxiv,openalex,crossref。
    """
    limit = max(1, min(8, int(limit_per_source)))
    requested = {item.strip().lower() for item in sources.split(",") if item.strip()}
    searchers = {
        "arxiv": _arxiv_search,
        "openalex": _openalex_search,
        "crossref": _crossref_search,
    }
    results: list[dict] = []
    errors: dict[str, str] = {}
    for name, searcher in searchers.items():
        if name not in requested:
            continue
        try:
            results.extend(searcher(query, limit))
        except Exception as exc:  # noqa: BLE001 - partial academic results are useful
            errors[name] = f"{type(exc).__name__}: {exc}"
    return _json({"query": query, "results": results, "errors": errors})


def _zotero_endpoint() -> tuple[str, dict[str, str]]:
    if settings.zotero_user_id and settings.zotero_api_key:
        return (
            f"https://api.zotero.org/users/{settings.zotero_user_id}",
            {"Zotero-API-Key": settings.zotero_api_key, "Zotero-API-Version": "3"},
        )
    return f"{settings.zotero_local_url}/users/0", {"Zotero-API-Version": "3"}


@mcp.tool()
def zotero_search(query: str, limit: int = 8, include_notes: bool = True) -> str:
    """只读搜索 Zotero 论文库的题名、作者、摘要、标签、笔记和引用信息。"""
    base_url, headers = _zotero_endpoint()
    limit = max(1, min(25, int(limit)))
    params = {
        "q": query,
        "qmode": "everything" if include_notes else "titleCreatorYear",
        "limit": limit,
        "format": "json",
        "include": "data,bib",
    }
    try:
        items = _http_get(f"{base_url}/items", params=params, headers=headers).json()
    except Exception as exc:
        raise RuntimeError(
            "无法读取 Zotero。请打开 Zotero 并启用本地 API，或在 .env 配置云端用户 ID 和 API Key。"
        ) from exc
    results = []
    for item in items[:limit]:
        data = item.get("data", {})
        creators = [
            (
                creator.get("name")
                or " ".join([creator.get("firstName", ""), creator.get("lastName", "")]).strip()
            )
            for creator in data.get("creators", [])
        ]
        results.append(
            {
                "key": item.get("key") or data.get("key"),
                "type": data.get("itemType"),
                "title": data.get("title") or data.get("note", "")[:160],
                "authors": creators,
                "date": data.get("date", ""),
                "doi": data.get("DOI", ""),
                "url": data.get("url", ""),
                "abstract": data.get("abstractNote", "")[:1500],
                "note": _clean_html(data.get("note", ""))[:1500],
                "tags": [tag.get("tag", "") for tag in data.get("tags", [])],
                "bibliography": _clean_html(item.get("bib", "")),
            }
        )
    return _json(results)


@mcp.tool()
def zotero_collections(limit: int = 100) -> str:
    """只读列出 Zotero 文件夹（collections）及其层级信息。"""
    base_url, headers = _zotero_endpoint()
    try:
        items = _http_get(
            f"{base_url}/collections",
            params={"limit": max(1, min(200, int(limit))), "format": "json"},
            headers=headers,
        ).json()
    except Exception as exc:
        raise RuntimeError("无法读取 Zotero 文件夹。请检查 Zotero 本地 API 或云端配置。") from exc
    return _json(
        [
            {
                "key": item.get("key"),
                "name": item.get("data", {}).get("name", ""),
                "parent": item.get("data", {}).get("parentCollection", ""),
            }
            for item in items
        ]
    )


@mcp.tool()
def anki_status() -> str:
    """检测 AnkiConnect 是否可用并返回 Anki 版本。"""
    return _json({"available": True, "anki_version": _anki_request("version")})


@mcp.tool()
def anki_find_cards(query: str = "", limit: int = 50) -> str:
    """只读搜索 Anki 卡片。query 使用 Anki 搜索语法。"""
    card_ids = _anki_request("findCards", {"query": query})[: max(1, min(200, int(limit)))]
    cards = _anki_request("cardsInfo", {"cards": card_ids}) if card_ids else []
    return _json(cards)


@mcp.tool()
def anki_create_card(
    deck: str,
    front: str,
    back: str,
    tags: list[str] | None = None,
) -> str:
    """在用户明确要求制作 Anki 卡片时，向指定牌组创建一张 Basic 卡片。"""
    if not front.strip() or not back.strip():
        raise ValueError("卡片正面和背面不能为空。")
    result = _anki_request(
        "addNote",
        {
            "note": {
                "deckName": deck,
                "modelName": "Basic",
                "fields": {"Front": front, "Back": back},
                "tags": tags or ["PDF学习助手"],
                "options": {"allowDuplicate": False},
            }
        },
    )
    return _json({"created_note_id": result, "deck": deck})


@mcp.tool()
def anki_update_note(note_id: int, front: str, back: str) -> str:
    """仅在用户明确要求时更新已有 Anki 笔记的 Front 和 Back 字段。"""
    _anki_request(
        "updateNoteFields",
        {"note": {"id": int(note_id), "fields": {"Front": front, "Back": back}}},
    )
    return _json({"updated_note_id": int(note_id)})


@mcp.tool()
def anki_set_due(card_ids: list[int], days: str) -> str:
    """仅在用户明确要求安排复习时设置卡片到期日，例如 days='3' 或 '1-7'。"""
    if not card_ids:
        raise ValueError("card_ids 不能为空。")
    result = _anki_request(
        "setDueDate",
        {"cards": [int(value) for value in card_ids], "days": days},
    )
    return _json({"updated": result, "card_ids": card_ids, "days": days})


if __name__ == "__main__":
    mcp.run(transport="stdio")
