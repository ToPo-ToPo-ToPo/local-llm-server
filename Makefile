# local-llm-server の導入を簡単にするためのラッパー。
# 導入は `make install` の一度だけ。以降の運用はすべて `gw` サブコマンド（gw help で一覧）。
.PHONY: help install dev uninstall

help:
	@echo "make install    gw コマンドを PATH に導入／更新し、自動起動を登録（以後どこでも gw）"
	@echo "make uninstall  自動起動を解除し、gw と設定・ログ・自動DLした llama.cpp を削除（モデル重みは対象外）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "運用は gw で行う: gw status / gw ps / gw stop（一覧は gw help。常駐は OS が世話する）"

# editable 固定（クローンのソースを直接指す＝自動更新が効く）＋ --reinstall で
# 「初回導入」も「依存が変わったときの入れ直し」も同じ 1 コマンドで済ませる。
# 導入後に自動起動（launchd / systemd user unit）を登録する——Ollama と同じく
# 「サーバーを意識しない」がデフォルト。戻したい場合は `gw disable`（手動 gw start 運用）。
# gw はこのシェルの PATH にまだ無いことがある（uv tool update-shell 前）ので、
# uv tool の既定 bin（~/.local/bin）を直接叩く。未対応 OS（Windows）では案内だけ出して続行。
install:
	uv tool install --editable . --reinstall
	@GW="$$HOME/.local/bin/gw"; command -v gw >/dev/null 2>&1 && GW="$$(command -v gw)"; \
	"$$GW" enable || echo "自動起動は未登録です（後から gw enable で登録できます）"

# 設定・キャッシュの置き場所（cli.py / provisioner.py と同じ XDG 規則で解決する）。
CONFIG_DIR := $(if $(XDG_CONFIG_HOME),$(XDG_CONFIG_HOME),$(HOME)/.config)/local-llm-server
CACHE_DIR := $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)/local-llm-server

# `make install` の逆。常駐を止め、gw を除去し、クローン外に残るものまで消す。
# gw が未導入でも失敗させない（`-` で続行）。モデル本体（~/.cache/huggingface）は
# 他ツールと共用なので触らない——不要なら自分で消す。
# ./.local-llm-server は旧バージョンが cwd に作っていたログの残骸（現行は作らない）。
#
# make は /bin/sh（非ログインシェル）で走るため ~/.local/bin が PATH に無いことがある。
# install と同じく uv tool の既定 bin を直接叩き、gw 自体がもう無い場合でも
# launchd 登録の解除とプロセス停止は素の launchctl / pkill で必ずやり切る
# （gw が消えた後にアイコンとデーモンだけ残る事故を防ぐ）。
uninstall:
	-GW="$$HOME/.local/bin/gw"; command -v gw >/dev/null 2>&1 && GW="$$(command -v gw)"; \
	if [ -x "$$GW" ]; then "$$GW" disable && "$$GW" stop; else \
	  launchctl bootout "gui/$$(id -u)/com.local-llm-server.gw" 2>/dev/null; \
	  rm -f "$$HOME/Library/LaunchAgents/com.local-llm-server.gw.plist"; \
	  pkill -f "python[3]* -m local_llm_server" 2>/dev/null; \
	fi; true
	-uv tool uninstall local-llm-server
	rm -rf "$(CONFIG_DIR)" "$(CACHE_DIR)" .local-llm-server
	@echo "削除: $(CONFIG_DIR)（設定）"
	@echo "削除: $(CACHE_DIR)（ログと自動DLした llama.cpp）"
	@echo "モデル重み ~/.cache/huggingface は未削除（共用のため手動で）"

dev:
	uv sync
