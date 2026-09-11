from __future__ import annotations

from urllib.parse import urlparse

import httpx

from .models import WebResult


class WebSearchError(RuntimeError):
    pass


class SearXNGSearch:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def status(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "未配置 SEARXNG_URL。"
        try:
            with httpx.Client(timeout=3, follow_redirects=True) as client:
                response = client.get(
                    f"{self.base_url}/search",
                    params={"q": "SearXNG status", "format": "json"},
                )
                response.raise_for_status()
                response.json()
        except Exception:  # noqa: BLE001 - status check must stay non-fatal
            return (
                False,
                f"无法连接 `{self.base_url}`。请确认 Docker Desktop 和 SearXNG 均已启动。",
            )
        return True, f"服务地址：`{self.base_url}`"

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        if not self.base_url:
            raise WebSearchError("未配置 SEARXNG_URL")
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "zh-CN",
                        "safesearch": 1,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise WebSearchError(
                "无法连接本地 SearXNG。请先启动联网搜索服务，或改用“仅 PDF”模式。"
            ) from exc

        results: list[WebResult] = []
        for item in payload.get("results", []):
            url = str(item.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                continue
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            title = str(item.get("title") or parsed.netloc).strip()
            if url and (snippet or title):
                results.append(WebResult(title=title, url=url, snippet=snippet[:1800]))
            if len(results) >= limit:
                break
        return results
