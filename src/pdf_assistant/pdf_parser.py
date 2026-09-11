from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import fitz

from .models import DocumentChunk, ParsedDocument

HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+[章节]|[0-9]+(?:\.[0-9]+){0,3}\s+|"
    r"chapter\s+\d+|section\s+\d+)",
    re.IGNORECASE,
)
FORMULA_SYMBOL_RE = re.compile(r"[=≈≠≤≥∑∏∫√∞∂∇±×÷→↦∈∉⊂⊆∪∩^_{}]|\\(?:frac|sum|int|sqrt)")
CAPTION_RE = re.compile(
    r"^(?:图\s*[0-9一二三四五六七八九十.-]+|figure\s*\d+|fig\.\s*\d+|"
    r"表\s*[0-9一二三四五六七八九十.-]+|table\s*\d+)",
    re.IGNORECASE,
)


class PDFValidationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_heading(text: str, fallback: str = "") -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate) <= 80 and (HEADING_RE.match(candidate) or len(candidate) <= 30):
            return candidate
        break
    return fallback


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text without cutting paragraphs when possible."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 1:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + chunk_size)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(start + 1, end - overlap)
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current.strip())
        tail = current[-overlap:].strip() if overlap else ""
        current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def extract_formula_lines(text: str) -> list[str]:
    """Find formula-like lines while keeping their surrounding mathematical notation intact."""
    formulas: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not 3 <= len(line) <= 500 or not FORMULA_SYMBOL_RE.search(line):
            continue
        math_chars = sum(char.isdigit() or char in "=+-*/^_()[]{}<>∑∏∫√∞∂∇±×÷" for char in line)
        if math_chars >= 2 or re.search(r"[A-Za-z]\s*[=≈≤≥]\s*", line):
            formulas.append(line)
    return list(dict.fromkeys(formulas))


def table_to_markdown(rows: list[list[object]]) -> str:
    normalized = [
        [clean_text(str(cell or "")).replace("|", "\\|") for cell in row]
        for row in rows
        if any(str(cell or "").strip() for cell in row)
    ]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_captions(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if CAPTION_RE.match(line.strip())][
        :12
    ]


class PDFParser:
    def __init__(
        self,
        *,
        chunk_size: int,
        overlap: int,
        max_mb: int,
        max_pages: int,
        page_assets_dir: Path | None = None,
        enable_page_ocr: bool = True,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_bytes = max_mb * 1024 * 1024
        self.max_pages = max_pages
        self.page_assets_dir = page_assets_dir
        self.enable_page_ocr = enable_page_ocr

    @staticmethod
    def _chunk(
        *,
        document_id: str,
        filename: str,
        page: int,
        heading: str,
        text: str,
        content_kind: str = "text",
        metadata: dict | None = None,
        image_path: str = "",
    ) -> DocumentChunk:
        return DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            filename=filename,
            page=page,
            heading=heading,
            text=text,
            char_count=len(text),
            content_kind=content_kind,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
            image_path=image_path,
        )

    def _render_page(self, page, document_id: str, page_number: int) -> Path | None:
        if self.page_assets_dir is None:
            return None
        target_dir = self.page_assets_dir / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"page_{page_number:04d}.png"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(target)
        return target

    @staticmethod
    def _ocr_image(path: Path) -> str:
        executable = shutil.which("tesseract")
        if not executable:
            return ""
        try:
            languages = subprocess.run(
                [executable, "--list-langs"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            language = "chi_sim+eng" if "chi_sim" in languages else "eng"
            result = subprocess.run(
                [executable, str(path), "stdout", "-l", language, "--psm", "6"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return clean_text(result.stdout)

    def validate(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise PDFValidationError("文件不存在或不是普通文件")
        if path.suffix.lower() != ".pdf":
            raise PDFValidationError("只支持 PDF 文件")
        if path.stat().st_size > self.max_bytes:
            raise PDFValidationError(
                f"PDF 超过大小限制（当前 {path.stat().st_size / 1024 / 1024:.1f} MB）"
            )
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise PDFValidationError("文件扩展名是 PDF，但内容不是有效 PDF")

    def copy_into_library(self, source: Path, uploads_dir: Path, document_id: str) -> Path:
        target = uploads_dir / f"{document_id}.pdf"
        shutil.copy2(source, target)
        return target

    def parse(self, path: Path, document_id: str, original_name: str) -> ParsedDocument:
        self.validate(path)
        warnings: list[str] = []
        try:
            document = fitz.open(path)
        except Exception as exc:
            raise PDFValidationError(f"无法打开 PDF：{exc}") from exc

        try:
            if document.needs_pass:
                raise PDFValidationError("PDF 已加密，请先移除密码后再上传")
            if document.page_count > self.max_pages:
                raise PDFValidationError(
                    f"PDF 共 {document.page_count} 页，超过 {self.max_pages} 页限制"
                )

            metadata = document.metadata or {}
            title = clean_text(metadata.get("title") or "") or Path(original_name).stem
            chunks: list[DocumentChunk] = []
            empty_pages: list[int] = []
            formula_count = 0
            table_count = 0
            visual_page_count = 0
            current_heading = ""

            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_text = clean_text(page.get_text("text", sort=True))
                page_number = page_index + 1
                page_heading = infer_heading(page_text, current_heading)
                if page_heading:
                    current_heading = page_heading
                chunks.extend(
                    (
                        self._chunk(
                            document_id=document_id,
                            filename=original_name,
                            page=page_number,
                            heading=current_heading,
                            text=part,
                        )
                    )
                    for part in split_text(page_text, self.chunk_size, self.overlap)
                )

                formulas = extract_formula_lines(page_text)
                if formulas:
                    formula_text = "[公式]\n" + "\n".join(formulas)
                    chunks.append(
                        self._chunk(
                            document_id=document_id,
                            filename=original_name,
                            page=page_number,
                            heading=current_heading,
                            text=formula_text,
                            content_kind="formula",
                            metadata={"formula_count": len(formulas)},
                        )
                    )
                    formula_count += len(formulas)

                try:
                    found_tables = page.find_tables()
                    tables = list(getattr(found_tables, "tables", []) or [])
                except Exception:  # noqa: BLE001 - malformed page tables should not abort ingest
                    tables = []
                for table_index, table in enumerate(tables, start=1):
                    try:
                        markdown = table_to_markdown(table.extract())
                    except Exception:  # noqa: BLE001 - keep parsing the rest of the document
                        markdown = ""
                    if not markdown:
                        continue
                    table_text = f"[表格 {table_index}]\n{markdown}"
                    chunks.append(
                        self._chunk(
                            document_id=document_id,
                            filename=original_name,
                            page=page_number,
                            heading=current_heading,
                            text=table_text,
                            content_kind="table",
                            metadata={"table_index": table_index},
                        )
                    )
                    table_count += 1

                image_count = len(page.get_images(full=True))
                drawing_count = len(page.get_drawings())
                captions = extract_captions(page_text)
                should_render = image_count > 0 or drawing_count >= 8 or len(page_text) < 20
                image_path = (
                    self._render_page(page, document_id, page_number) if should_render else None
                )
                if image_path:
                    visual_page_count += 1
                    ocr_text = ""
                    if self.enable_page_ocr and len(page_text) < 20:
                        ocr_text = self._ocr_image(image_path)
                    visual_parts = [
                        "[页面图像]",
                        f"页面包含 {image_count} 张嵌入图像、{drawing_count} 个矢量图形。",
                    ]
                    if captions:
                        visual_parts.append("图题/图注：" + "；".join(captions))
                    if ocr_text:
                        visual_parts.append("页面 OCR：\n" + ocr_text[:5000])
                    chunks.append(
                        self._chunk(
                            document_id=document_id,
                            filename=original_name,
                            page=page_number,
                            heading=current_heading,
                            text="\n".join(visual_parts),
                            content_kind="page_image",
                            metadata={
                                "image_count": image_count,
                                "drawing_count": drawing_count,
                                "captions": captions,
                                "ocr": bool(ocr_text),
                            },
                            image_path=str(image_path),
                        )
                    )

                if len(page_text) < 20 and not any(
                    chunk.page == page_number and chunk.content_kind != "page_image"
                    for chunk in chunks
                ):
                    empty_pages.append(page_number)

            if empty_pages:
                preview = "、".join(map(str, empty_pages[:12]))
                suffix = "等" if len(empty_pages) > 12 else ""
                warnings.append(
                    f"第 {preview} 页{suffix}几乎没有可提取文字，可能是扫描页或图片页。"
                )
            if not chunks:
                raise PDFValidationError(
                    "没有提取到可用文字。该 PDF 可能是扫描版，需要启用 OCR/MinerU。"
                )
            warnings.append(
                f"结构化解析：识别 {formula_count} 条公式、{table_count} 个表格、"
                f"{visual_page_count} 个含图像或矢量图的页面。"
            )
            return ParsedDocument(
                title=title,
                page_count=document.page_count,
                chunks=chunks,
                warnings=warnings,
            )
        finally:
            document.close()
