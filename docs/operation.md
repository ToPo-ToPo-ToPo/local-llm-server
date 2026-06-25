# 起動・運用・アンインストール

ゲートウェイは `./gateway.toml` のあるディレクトリで起動する。**ターミナル**（CLI/TUI）から運用する。

## ターミナル

```bash
uv run local-llm-server            # TUI ダッシュボード（既定。状態を毎秒自動更新）
uv run local-llm-server --headless # TUI なしフォアグラウンド実行（Ctrl-C で停止）
uv run local-llm-server --start    # バックグラウンド常駐起動（端末を離す。Ollama 流）
uv run local-llm-server --status   # 応答可否・PID・提供モデル・ログパス
uv run local-llm-server --stop     # 停止（配下のモデルサーバーも止める）
uv run local-llm-server --restart  # 停止→再起動（gateway.toml 変更の反映に）
```

- 引数なしで起動すると **TUI ダッシュボード**が開く。ゲートウェイを裏で常駐させ、`GET /admin/status`
  を毎秒ポーリングして状態を全画面表示する。
  - 上部: ゲートウェイ（応答状態・port・起動経過・累計リクエスト）と各モデルの表
    （loaded/idle/unloaded・内部ポート・処理中数・アイドル自動解放までの残り・累計）。
  - 操作: 単キー `s`停止 / `r`再起動 / `g`起動 / `l`ログ / `q`終了、`:` で打ち込みコマンド。
  - `q` で終了してもゲートウェイは常駐し続ける（停止は `s` か `--stop`）。
- **非対話端末**（パイプ / CI / 裏起動）では自動でフォアグラウンド実行になる。
- いずれも CWD の `./gateway.toml` を読む。ログは `./.local-llm-server/gateway-<port>.log`。
- TUI は [textual](https://textual.textualize.io/) 製（依存に含まれる）。`--headless` では読み込まれない。

## アンインストール

自動起動やシステム改変はしていないので、**ゲートウェイ停止＋リポジトリ削除**で跡形なく消える。

```bash
uv run local-llm-server --stop   # 常駐ゲートウェイと配下のモデルサーバーを停止
```

あとはこのリポジトリのフォルダを削除すれば、コードと依存（`.venv`）、ログ（`./.local-llm-server`）も
まとめて消えて完了。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の共有キャッシュ
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため**アンインストールでは一切
> 削除しない**（場所を案内するだけ）。不要になったら自分で中のモデルフォルダを消す。
