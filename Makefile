# local-llm-server の導入を簡単にするためのラッパー。
# 使い方: `make install`（`gw` コマンドを導入）/ `make start`（デーモンを裏で常駐起動）。
# 状態確認は `gw status` / `gw ps`、停止は `gw stop`（詳細は docs/operation.md）。
.PHONY: help install dev start status stop

help:
	@echo "make install    gw コマンドを PATH に導入（uv tool install --editable .。以後どこでも gw）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "make start      ゲートウェイを裏で常駐起動（gw start と同じ）"
	@echo "make status     稼働状態を表示（gw status と同じ）"
	@echo "make stop       ゲートウェイを停止（gw stop と同じ）"

install:
	uv tool install --editable .

dev:
	uv sync

start:
	gw start

status:
	gw status

stop:
	gw stop
