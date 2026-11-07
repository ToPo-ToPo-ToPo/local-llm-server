# local-llm-server の導入を簡単にするためのラッパー。
# 導入は標準の uv sync。運用は gw サブコマンド（gw enable 後はどこからでも）で行う。
.PHONY: help install dev check mtp-check uninstall

help:
	@echo "make install    プロジェクト .venv を導入／更新し、自動起動と ~/.local/bin/gw を登録"
	@echo "make uninstall  自動起動と ~/.local/bin/gw を解除し、設定・ログ・自動DLした llama.cpp を削除（モデル重みと .venv は保持）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "make check      PR 前の検査を手元で回す（lint・型・セキュリティ・依存監査・テストと網羅率）"
	@echo "make mtp-check  mlx-vlm 更新後に MTP の採択率・速度が劣化していないか実測する（Apple Silicon）"
	@echo "運用: gw status / gw ps / gw stop（どこからでも。uv run --frozen gw ... でも同じ）"

# Standard uv project environment (.venv is a real directory).
UV ?= uv
install:
	$(UV) sync --frozen
	$(UV) run --frozen gw enable

# 設定・キャッシュの置き場所（cli.py / provisioner.py と同じ XDG 規則で解決する）。
CONFIG_DIR := $(if $(XDG_CONFIG_HOME),$(XDG_CONFIG_HOME),$(HOME)/.config)/local-llm-server
CACHE_DIR := $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)/local-llm-server

# 常駐と自動起動を解除し、アプリ設定・ログを削除する。
# gw が未導入でも失敗させない（`-` で続行）。モデル本体（~/.cache/huggingface）は
# 他ツールと共用なので触らない——不要なら自分で消す。
# ./.local-llm-server は旧バージョンが cwd に作っていたログの残骸（現行は作らない）。
#
# ~/.local/bin/gw は gw enable が置いたリンク。このチェックアウトの .venv/bin/gw を指すときだけ外す。
LAUNCHER := $(HOME)/.local/bin/gw
uninstall:
	-if [ -x .venv/bin/gw ]; then .venv/bin/gw disable; .venv/bin/gw stop; fi
	-if [ -L "$(LAUNCHER)" ] && [ "$$(readlink "$(LAUNCHER)")" = "$(CURDIR)/.venv/bin/gw" ]; then rm "$(LAUNCHER)"; fi
	rm -rf "$(CONFIG_DIR)" "$(CACHE_DIR)" .local-llm-server
	@echo "モデル重み ~/.cache/huggingface とプロジェクト .venv は未削除"

dev:
	$(UV) sync

# PR 前の検査。GitHub の CI は使わない（課金制限でジョブが起動せず、失敗の信号として
# 機能していなかった）ので、以前 CI の quality ジョブで回していたものをここで回す。
# テストは網羅率の計測を兼ねて 1 回だけ走らせる（下限は pyproject の fail_under）。
check:
	$(UV) sync --dev --frozen
	$(UV) run ruff check --select E9,F63,F7,F82 local_llm_server tests
	$(UV) run mypy --ignore-missing-imports --follow-imports=skip --check-untyped-defs local_llm_server
	$(UV) run bandit -r local_llm_server -q -lll
	$(UV) export --frozen --no-dev --no-hashes -o "$${TMPDIR:-/tmp}/local-llm-server-requirements.txt"
	$(UV) run pip-audit -r "$${TMPDIR:-/tmp}/local-llm-server-requirements.txt" --no-deps --disable-pip
	$(UV) run coverage run -m pytest -q
	$(UV) run coverage report

# mlx-vlm 更新後の MTP 実測（docs/mtp.md）。取得済みの対応モデルを本体単独／MTP ありで生成して比べる。
# 引数は ARGS で渡す（例: make mtp-check ARGS="ToPo-ToPo/Qwen3.6-27B-mlx-4bit --save-baseline"）。
mtp-check:
	$(UV) run --frozen python scripts/check_mtp.py $(ARGS)
