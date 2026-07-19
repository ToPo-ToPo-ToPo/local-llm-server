# 起動・運用・アンインストール

ゲートウェイは `./gateway.toml` のあるディレクトリで、`gw` の **CLI サブコマンド**で運用する。
デーモン本体は端末を持たず裏で常駐し、`gw` は「そのデーモンを起動・停止・監視する薄い CLI」。
マシンに 1 ゲートウェイだけ（`GatewayLock`）なので、`gw` を何度打ってもデーモンは 0 個か 1 個。

## 起動と状態確認

```bash
uv run gw start      # デーモンを裏で常駐起動（既に起動していれば何もしない）
uv run gw status     # 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
uv run gw ps         # ロード中モデルの状態（処理中数・在席・アイドル残り）
```

引数なしの `uv run gw` は `start` してから状態を表示する（従来の手触り）。
`uv tool install local-llm-server` で入れた場合は `uv run` も要らず、どこでも `gw start` で動く。

## サブコマンド一覧

| コマンド | 動作 |
|---|---|
| `gw start` | デーモンを裏で常駐起動（既に居れば何もしない） |
| `gw stop` | このパッケージ由来のゲートウェイ／モデルサーバーを停止 |
| `gw restart` | stop → start |
| `gw status` | 稼働/停止を 1 行で表示（停止中は終了コード 1） |
| `gw ps` | ロード中モデルの状態を表示 |
| `gw list` | 使えるモデル一覧（カタログ＋HF キャッシュ。停止中でも表示可） |
| `gw log [-f] [-n N]` | ゲートウェイログの末尾を表示（`-f` で追従） |
| `gw max <n\|off>` | `max_resident` を無停止で変更（超過はアイドルから LRU 退避） |
| `gw mtp [model]` | MTP に必要なドラフターと取得状況を表示（ダウンロードはしない → docs/mtp.md） |
| `gw update` | PyPI 新版があれば `git pull` で追従し、稼働中なら再起動 |
| `gw help` | サブコマンド一覧を表示（`gw -h` と同じ。`gateway.toml` 不要） |

- **どこからでも状態確認・停止できる**: `start`/`restart` は「何を配信するか」を知るため CWD の
  `./gateway.toml` が要るが、`status`/`stop`/`ps`/`list`/`log`/`max`/`update` は
  **`gateway.toml` の無いディレクトリからでも動く**。起動時にデーモンが接続先（host/port/PID）を
  固定パスのランタイム記録（temp ディレクトリの `local-llm-server-gateway.json`）に残すので、
  マシンに 1 つのデーモンをそこから特定して叩く（単一起動＝`GatewayLock` の裏返し）。記録は正常
  停止で消え、クラッシュで残っても PID 生存チェックで stale を掴まない。

  > **注意**: 「どこからでも」は `gw` コマンド自体が PATH にある前提。`uv run gw` は**プロジェクト
  > 内でしか動かない**（`uv run` は CWD のプロジェクトから `gw` を探すため、ホーム等では
  > `Failed to spawn: gw` になる）。プロジェクト外から打ちたいなら、次のいずれかにする:
  > - `uv tool install --from /path/to/local-llm-server local-llm-server` → 以後どこでも素の `gw`
  > - もしくは `uv run --project /path/to/local-llm-server gw status` のように `--project` を付ける
  > - もしくは alias を張る（例: `alias gw='uv run --project /path/to/local-llm-server gw'`）
- `start`/`restart` は CWD の `./gateway.toml` を読む。ログは `./.local-llm-server/gateway-<port>.log`。
- `gw list` の一覧には**ロード中だけでなく、HF キャッシュにある DL 済みモデルも未ロード候補**として並ぶ。
- **自動更新**は稼働中デーモンが裏で行う。PyPI 新版を検知し、作業ツリーがクリーン（git クローン運用）
  かつ処理中/在席が 0 の瞬間に `git pull` で追従して自分を新コードで再起動する
  （`gateway.toml` の `auto_update = false` で無効化。手動で今すぐなら `gw update`）。

> **裏で動くゲートウェイ本体**は `python -m local_llm_server`（ヘッドレスワーカー）として
> `gw start` が別プロセスで常駐させる。通常は直接触らないが、CLI を介さず素のフォアグラウンド
> 実行がしたいときはこれを直接起動できる（`Ctrl-C` で停止）。

## アンインストール

自動起動やシステム改変はしていないので、**`gw stop` ＋リポジトリ削除**で跡形なく消える。
`gw stop` は常駐ゲートウェイと配下のモデルサーバーをまとめて停止する（同パッケージ由来の
プロセスだけを止め、同じポートをたまたま使う無関係なプロセスは巻き添えにしない）。

`gw` を使わず常駐だけ止めたいときは、ポートを掴んでいるプロセスを直接止める
（ログの `./.local-llm-server/gateway-<port>.log` と `lsof -i :<port>` で PID を確認）。
あとはこのリポジトリのフォルダを削除すれば、コードと依存（`.venv`）、ログ（`./.local-llm-server`）も
まとめて消えて完了。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の共有キャッシュ
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため**アンインストールでは一切
> 削除しない**（場所を案内するだけ）。不要になったら自分で中のモデルフォルダを消す。
