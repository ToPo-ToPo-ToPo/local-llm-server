# local-llm-server の導入を簡単にするためのラッパー。
# 導入は `make install` の一度だけ。以降の運用はすべて `gw` サブコマンド（gw help で一覧）。
.PHONY: help install dev

help:
	@echo "make install    gw コマンドを PATH に導入／更新（以後どこでも gw。再実行で入れ直し）"
	@echo "make dev        開発用 venv を用意（uv sync。uv run pytest でテスト）"
	@echo "運用は gw で行う: gw start / gw status / gw ps / gw stop（一覧は gw help）"

# editable 固定（クローンのソースを直接指す＝自動更新が効く）＋ --reinstall で
# 「初回導入」も「依存が変わったときの入れ直し」も同じ 1 コマンドで済ませる。
install:
	uv tool install --editable . --reinstall

dev:
	uv sync
