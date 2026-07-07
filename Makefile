# local-llm-server の導入を簡単にするためのラッパー。
# 使い方: `make install`（依存導入）/ `make run`（TUI ダッシュボード起動）。
# 停止・再起動は TUI 内の単キー（q 終了・s 停止・r 再起動）で行う。
.PHONY: help install run

help:
	@echo "make install    依存を導入（mlx は Apple Silicon でのみ自動で入る）"
	@echo "make run        TUI ダッシュボードを起動（uv run gw と同じ）"

install:
	uv sync

run:
	uv run gw
