# local-llm-server の導入を簡単にするためのラッパー。
# 導入は `make install` の一度だけ。以降の運用はすべて `gw` サブコマンド（gw help で一覧）。
.PHONY: help install dev uninstall

help:
	@echo "make install    gw コマンドを PATH に導入／更新（以後どこでも gw。再実行で入れ直し）"
	@echo "make uninstall  gw コマンドと設定・ログ・自動DLした llama.cpp を削除（モデル重みは対象外）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "運用は gw で行う: gw start / gw status / gw ps / gw stop（一覧は gw help）"

# editable 固定（クローンのソースを直接指す＝自動更新が効く）＋ --reinstall で
# 「初回導入」も「依存が変わったときの入れ直し」も同じ 1 コマンドで済ませる。
install:
	uv tool install --editable . --reinstall

# 設定・キャッシュの置き場所（cli.py / provisioner.py と同じ XDG 規則で解決する）。
CONFIG_DIR := $(if $(XDG_CONFIG_HOME),$(XDG_CONFIG_HOME),$(HOME)/.config)/local-llm-server
CACHE_DIR := $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)/local-llm-server

# `make install` の逆。常駐を止め、gw を除去し、クローン外に残るものまで消す。
# gw が未導入でも失敗させない（`-` で続行）。モデル本体（~/.cache/huggingface）は
# 他ツールと共用なので触らない——不要なら自分で消す。
# ./.local-llm-server は旧バージョンが cwd に作っていたログの残骸（現行は作らない）。
uninstall:
	-command -v gw >/dev/null 2>&1 && gw stop
	-uv tool uninstall local-llm-server
	rm -rf "$(CONFIG_DIR)" "$(CACHE_DIR)" .local-llm-server
	@echo "削除: $(CONFIG_DIR)（設定）"
	@echo "削除: $(CACHE_DIR)（ログと自動DLした llama.cpp）"
	@echo "モデル重み ~/.cache/huggingface は未削除（共用のため手動で）"

dev:
	uv sync
