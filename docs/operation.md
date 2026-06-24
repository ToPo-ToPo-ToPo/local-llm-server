# 起動・運用・アンインストール

ゲートウェイは `./gateway.toml` のあるディレクトリで起動する。**ターミナル**（CLI/TUI）と、任意で
**トレイ GUI アプリ**の 2 通りがあり、どちらも同じ運用基盤を共有する。

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

## トレイ GUI アプリ（任意）

ターミナルを使わず、デスクトップで状態をひと目で見たい場合の代替。メニューバー（macOS）/通知領域
（Windows）/トレイ（Linux）に常駐し、起動・停止・監視をクリックで行える。まず一度だけクリック起動
アプリを作る。**`make` を使えば 1 コマンド**（macOS / Linux）:

```bash
make install        # 依存(mlx+gui)を入れて → クリック起動アプリを作成
```

`make` を使わない場合は同等の 2 手順:

```bash
uv add "local-llm-server[gui]"              # pystray / pillow（各 OS のバックエンドも）
uv run local-llm-server-gui --install-app   # gateway.toml のあるディレクトリで1度だけ
```

作られるもの:
- **macOS** … `~/Applications/Local LLM Gateway.app`（専用アイコン付きの普通のアプリ。Dock 表示・
  Cmd+Tab 対応）。Finder で Dock にドラッグすれば常設。Dock に出さずメニューバーだけにしたいときは
  `--install-app --menubar-only`。
- **Linux** … `~/.local/share/applications/*.desktop`／**Windows** … デスクトップに `.cmd`。

あとはアプリをダブルクリックするだけ。ゲートウェイがバックグラウンドで常駐し、アイコンが出る:

- アイコンの**色**で状態（🟢 応答可 / 🟡 起動中 / ⚫ 停止）、**数字**でロード済みモデル数。
- メニューから 起動 / 停止 / 再起動 / ログを開く / 更新。あわせてゲートウェイの PID・起動経過・
  累計リクエスト数、各モデルの状態（バックエンド:内部ポート・loaded/idle・処理中数・アイドル解放
  までの残り・累計）、運用方針（max_resident・idle_timeout）を表示。
- アプリ（トレイ）を閉じてもゲートウェイは常駐し続ける。止めるときはメニューの **停止**。
- 同一バンドル ID なので、起動中に再度クリックしても二重起動しない。

> アプリは内部的に `local-llm-server`（`./gateway.toml` を読むフォアグラウンド実行）を端末から切り
> 離して起動・常駐させる。状態は `GET /admin/status` から取得。Linux はトレイ表示にシステムトレイが
> 要る（GNOME は AppIndicator 拡張など）。アプリは作成時の**リポジトリのパスと Python（venv）を固定**
> するので、クリック時の場所に依存しない。

## アンインストール

自動起動やシステム改変はしていないので、**コマンド1つ＋リポジトリ削除**で跡形なく消える。

```bash
make uninstall      # アプリ削除＋ゲートウェイ停止＋データ掃除（macOS / Linux）
```

`make` を使わない場合（Windows 含む全 OS）:

```bash
uv run local-llm-server-gui --uninstall-app --purge
#  --purge を外すとログ等は残す。Windows はデスクトップの .cmd を削除。
```

`--purge` が消すもの:
- ランチャ（macOS `.app`／Linux `.desktop`／Windows `.cmd`）と、リポジトリ内のログ（`./.local-llm-server`）。
- 念のため macOS がアプリ毎に作りうる場所も掃除（`~/Library/Application Support/local-llm-server`、
  bundle id 配下の Saved State / Preferences / Caches 等）。通常ここには何も作らないが取りこぼし防止。

最後にこのリポジトリのフォルダを削除すれば、コードと依存（`.venv`）も消えて完了。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の共有キャッシュ
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため**アンインストールでは一切
> 削除しない**（場所を案内するだけ）。不要になったら自分で中のモデルフォルダを消す。
