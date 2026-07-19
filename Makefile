# local-llm-server の導入を簡単にするためのラッパー。
# 導入は `make install` の一度だけ。以降の運用はすべて `gw` サブコマンド（gw help で一覧）。
.PHONY: help install dev

help:
	@echo "make install    gw コマンドを PATH に導入（uv tool install --editable .。以後どこでも gw）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "運用は gw で行う: gw start / gw status / gw ps / gw stop（一覧は gw help）"

install:
	uv tool install --editable .

dev:
	uv sync
