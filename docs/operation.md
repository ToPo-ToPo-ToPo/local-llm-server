# 起動・運用・アンインストール

運用は `gw` の **CLI サブコマンド**で行う。デーモン本体は端末を持たず裏で常駐し、`gw` は
「そのデーモンを起動・停止・監視する薄い CLI」。マシンに 1 ゲートウェイだけ（`GatewayLock`）なので、
`gw` を何度打ってもデーモンは 0 個か 1 個。

## インストール（一度だけ）

```bash
git clone https://github.com/ToPo-ToPo-ToPo/local-llm-server
cd local-llm-server
uv tool install --editable .     # `gw` を PATH に導入（Ollama 流。以後どこでも `gw`）
```

`--editable` なのでソースはこのクローンを指す（`gw update` / 自動更新の `git pull` がそのまま効く）。
`~/.local/bin` が PATH に無いと言われたら `uv tool update-shell` を一度実行する。

## 起動と状態確認（どのディレクトリからでも）

```bash
gw start      # デーモンを裏で常駐起動（既に起動していれば何もしない）
gw status     # 稼働/停止・PID・URL・起動経過・累計リクエストを 1 行表示
gw ps         # ロード中モデルの状態（処理中数・在席・アイドル残り）
```

設定ファイルは **`~/.config/local-llm-server/gateway.toml` の 1 箇所だけ**。無ければ初回の
`gw start` が自動生成する（クローンの例 gateway.toml を複製。以降の編集・参照はこの 1 ファイルのみ）。
引数なしの `gw` はコマンド一覧を表示する（起動の入口は `gw start` の 1 つだけ）。

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
| `gw help` | サブコマンド一覧を表示（設定ファイル不要） |

- **どこからでも動く**: どのディレクトリから打っても、設定は `~/.config` の 1 ファイル、デーモンは
  マシンに 1 つ（`GatewayLock`）なので、常に同じものに届く。`status`/`stop`/`ps`/`list`/`log`/
  `max`/`update` は**稼働中デーモンのランタイム記録**（temp の `local-llm-server-gateway.json`：
  host/port/PID）を最優先で辿るため、実際に動いているデーモンを見失わない。記録は正常停止で消え、
  クラッシュで残っても PID 生存チェックで stale を掴まない。
- ログは `~/.config/local-llm-server/.local-llm-server/gateway-<port>.log`（設定ディレクトリ基準）。
  `gw log` で末尾を表示できる。
- `gw list` の一覧には**ロード中だけでなく、HF キャッシュにある DL 済みモデルも未ロード候補**として並ぶ。
- **自動更新**は稼働中デーモンが裏で行う。PyPI 新版を検知し、作業ツリーがクリーン（git クローン運用）
  かつ処理中/在席が 0 の瞬間に `git pull` で追従して自分を新コードで再起動する
  （`gateway.toml` の `auto_update = false` で無効化。手動で今すぐなら `gw update`）。
  設定はクローンの外（`~/.config`）にあるので、編集してもクリーン判定を妨げない。

## アンインストール

自動起動やシステム改変はしていないので、次の 3 つで跡形なく消える。

```bash
gw stop                              # 常駐ゲートウェイと配下のモデルサーバーを停止
uv tool uninstall local-llm-server   # gw コマンドを除去
rm -rf ~/.config/local-llm-server    # 設定とログを削除（クローンのフォルダも不要なら削除）
```

`gw stop` は同パッケージ由来のプロセスだけを止め、同じポートをたまたま使う無関係なプロセスは
巻き添えにしない。

> **モデル本体（重み）には触れない**。mlx / mlx-vlm は HuggingFace の共有キャッシュ
> `~/.cache/huggingface` にモデルをダウンロードするが、他ツールと共用のため**アンインストールでは一切
> 削除しない**（場所を案内するだけ）。不要になったら自分で中のモデルフォルダを消す。
