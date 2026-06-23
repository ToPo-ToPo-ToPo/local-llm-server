# local-llm-server の導入/アンインストールを簡単にするためのラッパー。
# 使い方: `make install` / `make uninstall`（macOS / Linux。Windows は README 参照）。
.PHONY: help install app uninstall

help:
	@echo "make install    依存(mlx+gui)を入れてクリック起動アプリを作成"
	@echo "make app        クリック起動アプリだけ作成/更新"
	@echo "make uninstall  アプリを削除しゲートウェイを停止（ログも削除）"

install:
	uv sync --extra mlx --extra gui
	uv run local-llm-server-gui --install-app

app:
	uv run local-llm-server-gui --install-app

uninstall:
	-uv run local-llm-server-gui --uninstall-app --purge
	@echo "コードと依存も消すには、このフォルダを削除してください:"
	@echo "  $(CURDIR)"
