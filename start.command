#!/bin/zsh
set -e
cd "${0:A:h}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "未找到 .venv。请先按照 README.md 完成安装。"
  read "?按回车键退出..."
  exit 1
fi

echo "正在准备联网搜索服务..."
if ! PDF_ASSISTANT_NO_PAUSE=1 ./start-search.command; then
  echo "联网搜索启动失败，问答助手未启动。"
  read "?按回车键退出..."
  exit 1
fi

if [[ -d "/Applications/Zotero.app" ]]; then
  echo "正在启动 Zotero 本地论文库..."
  open -gj -a Zotero
fi

if [[ -d "/Applications/Anki.app" ]]; then
  echo "正在启动 Anki / AnkiConnect..."
  open -gj -a Anki
fi

if curl -fsS --max-time 2 "http://127.0.0.1:7860/" >/dev/null 2>&1; then
  echo "PDF 学习问答助手已在运行，正在打开页面..."
  open "http://127.0.0.1:7860/"
  exit 0
fi

echo "正在启动 PDF 学习问答助手..."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python app.py
