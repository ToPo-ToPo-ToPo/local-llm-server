# local-llm-server の導入を簡単にするためのラッパー。
# 使い方: `make install`（依存導入）/ `make start`（デーモンを裏で常駐起動）。
# 状態確認は `gw status` / `gw ps`、停止は `gw stop`（詳細は docs/operation.md）。
.PHONY: help install start status stop

help:
	@echo "make install    依存を導入（mlx は Apple Silicon でのみ自動で入る）"
	@echo "make start      ゲートウェイを裏で常駐起動（uv run gw start と同じ）"
	@echo "make status     稼働状態を表示（uv run gw status と同じ）"
	@echo "make stop       ゲートウェイを停止（uv run gw stop と同じ）"

install:
	uv sync

start:
	uv run gw start

status:
	uv run gw status

stop:
	uv run gw stop
