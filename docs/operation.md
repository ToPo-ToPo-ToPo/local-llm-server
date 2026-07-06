# 起動・運用・アンインストール

ゲートウェイは `./gateway.toml` のあるディレクトリで起動する。**ターミナル**（CLI/TUI）から運用する。

## 起動（推奨: TUI ダッシュボード）

**基本は引数なしで起動し、TUI ダッシュボードで運用する。** 状態（使えるモデル・ロード中・処理中数など）
が一目で分かり、停止/再起動も画面内の単キーでできる。

```bash
uv run local-llm-server            # ← 推奨。TUI ダッシュボード（状態を毎秒自動更新）
```

その他の起動形態（用途に応じて）:

```bash
uv run local-llm-server --start    # バックグラウンド常駐起動（端末を離す。Ollama 流）
uv run local-llm-server --headless # TUI なしフォアグラウンド実行（Ctrl-C で停止。CI / スクリプト向け）
uv run local-llm-server --status   # 応答可否・PID・提供モデル・ログパス
uv run local-llm-server --stop     # 停止（配下のモデルサーバーも止める。無関係なプロセスは巻き添えにしない）
uv run local-llm-server --restart  # 停止→再起動（gateway.toml 変更の反映に）
uv run local-llm-server --check-mtp [model]  # MTP に必要なドラフターと取得状況を表示（DL はしない。
                                             # model 省略で対応モデル一覧 → docs/mtp.md）
```

- 引数なしで起動すると **TUI ダッシュボード**が開く。ゲートウェイを裏で常駐させ、`GET /admin/status`
  を毎秒ポーリングして状態を全画面表示する。
  - 上部: ゲートウェイ（応答状態・port・起動経過・累計リクエスト）と各モデルの表
    （loaded/idle/unloaded・処理中数・在席エージェント数・アイドル自動解放までの残り・累計）。
    表には**ロード中のモデルだけでなく、HF キャッシュにある DL 済みモデルも未ロード候補**として並ぶ
    （まだ使っていないモデルもここから選べる）。
  - 操作: 単キー `s`停止 / `r`再起動 / `g`起動 / `m`max_resident 変更 / `l`ログ / `q`終了。
    打ち込みコマンド（`stop` / `restart` / `start` / `max <n>` / `mtp [model]` / `log` / `quit`）は
    上部の入力欄をクリックして入力する（`m` キーで `max ` がプリフィルされる）。
  - `mtp [model]` は MTP に必要なドラフターと取得状況を画面内に表示する（CLI の `--check-mtp` と
    同内容。ダウンロードはしない）。表示中のドラフターのパスや `hf download` コマンドは
    **クリックでコピー**でき、そのまま端末に貼ってダウンロードできる（→ docs/mtp.md）。
  - `q` で終了すると**ゲートウェイ・デーモンも停止する**（次回起動時に最新コードが反映される）。
    ダッシュボードだけ閉じて常駐は残したい場合は `Ctrl+C` などで TUI を抜けず、`--start` 常駐運用にする。
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
