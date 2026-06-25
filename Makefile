# local-llm-server の導入/停止を簡単にするためのラッパー。
# 使い方: `make install`（依存導入）/ `make stop`（ゲートウェイ停止）。
.PHONY: help install stop

help:
	@echo "make install    Apple Silicon 向け推論バックエンド(mlx)を導入"
	@echo "make stop       常駐ゲートウェイを停止"

install:
	uv sync --extra mlx

stop:
	-uv run local-llm-server --stop
