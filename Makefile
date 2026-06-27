# local-llm-server の導入/停止を簡単にするためのラッパー。
# 使い方: `make install`（依存導入）/ `make stop`（ゲートウェイ停止）。
.PHONY: help install stop

help:
	@echo "make install    依存を導入（mlx は Apple Silicon でのみ自動で入る）"
	@echo "make stop       常駐ゲートウェイを停止"

install:
	uv sync

stop:
	-uv run local-llm-server --stop
