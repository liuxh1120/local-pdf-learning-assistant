from pathlib import Path

import fitz

from pdf_assistant.pdf_parser import (
    PDFParser,
    clean_text,
    extract_formula_lines,
    infer_heading,
    split_text,
    table_to_markdown,
)


def test_clean_text_removes_noise():
    assert clean_text("A\x00  B\n\n\nC") == "A B\n\nC"


def test_split_text_respects_limit_for_long_paragraph():
    text = "甲" * 1200
    chunks = split_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(0 < len(chunk) <= 500 for chunk in chunks)
    assert chunks[0][-50:] == chunks[1][:50]


def test_split_text_preserves_short_document():
    assert split_text("定义\n\n这是内容", chunk_size=500, overlap=50) == ["定义\n\n这是内容"]


def test_infer_heading():
    assert infer_heading("3.2 最大似然估计\n正文") == "3.2 最大似然估计"


def test_formula_and_table_structure_helpers():
    assert extract_formula_lines("均值\nE(X) = sum_i x_i p_i\n结论") == [
        "E(X) = sum_i x_i p_i"
    ]
    markdown = table_to_markdown([["参数", "估计值"], ["mu", "1.5"]])
    assert "| 参数 | 估计值 |" in markdown
    assert "| mu | 1.5 |" in markdown


def test_parser_creates_formula_and_page_image_evidence(tmp_path: Path):
    pdf_path = tmp_path / "visual.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "2.1 Estimation\nE(X) = sum_i x_i p_i\nFigure 1 sampling distribution")
    for offset in range(8):
        page.draw_line((72, 120 + offset * 5), (240, 120 + offset * 5))
    document.save(pdf_path)
    document.close()

    parser = PDFParser(
        chunk_size=500,
        overlap=50,
        max_mb=10,
        max_pages=10,
        page_assets_dir=tmp_path / "assets",
        enable_page_ocr=False,
    )
    parsed = parser.parse(pdf_path, "doc-visual", "visual.pdf")
    kinds = {chunk.content_kind for chunk in parsed.chunks}
    assert {"text", "formula", "page_image"}.issubset(kinds)
    image_chunk = next(chunk for chunk in parsed.chunks if chunk.content_kind == "page_image")
    assert Path(image_chunk.image_path).exists()
