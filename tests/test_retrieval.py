from pdf_assistant.retrieval import tokenize


def test_tokenize_chinese_and_formula_symbols():
    tokens = tokenize("最大似然估计 MLE 与 x_1 + x_2")
    assert "最大似然" in tokens or "最大" in tokens
    assert "mle" in tokens
    assert "x_1" in tokens
