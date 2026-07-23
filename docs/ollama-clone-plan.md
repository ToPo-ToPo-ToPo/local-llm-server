# Ollama クローン化 開発計画

local-llm-server に Ollama の良さを取り入れるための計画。

**スコープ確定（2026-07-24）: 目標は「Ollama の運用体験」であり「Ollama のエコシステム
互換」ではない。** Ollama 用クライアント（Open WebUI 等）を使う予定は無く、接続はすべて
OpenAI 互換 API（`/v1/*`）で足りているため、**Phase 1（`/api/*` ネイティブ API）は
作らない**。運用体験（意識しない常駐・メニューバーアイコン・更新 UX）は Phase 0 で完成済み。

残るゴールは 1 つ: **Ollama の CLI / 運用体験を `gw` で再現する**こと
（サービス化 ✅ / アイコン ✅ / 更新 ✅ / `pull` / `rm` / 短縮モデル名）。

## 現状の棚卸し（すでに Ollama 相当ができていること）

| Ollama | local-llm-server | 状態 |
|---|---|---|
| `ollama serve`（常駐デーモン） | `gw start`（daemon.py） | ✅ |
| `ollama ps` | `gw ps` | ✅ |
| `ollama list` | `gw list`（カタログ + HF キャッシュ） | ✅ |
| モデルの遅延ロード / 自動アンロード | 初回リクエストでロード、LRU 退避、`idle_timeout` | ✅（Ollama より高機能） |
| OpenAI 互換 `/v1/chat/completions` | あり（Ollama も v1 互換を持つ） | ✅ |
| マルチバックエンド | mlx / mlx-vlm / llama.cpp / vLLM / SGLang / whisper | ✅（Ollama は llama.cpp のみ） |

## ギャップと採否

| Ollama | 現状 | 採否 |
|---|---|---|
| ネイティブ API `/api/chat` `/api/generate` ほか | なし | **見送り**（Ollama 用クライアントを使わないため。2026-07-24 決定） |
| `ollama run`（ターミナル REPL） | なし | **見送り**（対話 UI は agent-corporation が担当） |
| `ollama pull` / `rm`（進捗付き DL・削除） | ロード時に暗黙 DL のみ | Phase 2 |
| 短いモデル名 + タグ（`qwen3:8b`） | HF リポジトリ ID 直指定 | Phase 3 |
| Modelfile（`ollama create`） | なし | **保留**（system prompt 等はエージェント側が管理しており需要が薄い） |
| `/v1/embeddings`（OpenAI 互換） | なし | Phase 5（RAG 用途が出てきたら） |

**クローンしないもの**: registry.ollama.ai 相当の独自レジストリと blob ストアは作らない。
モデル配布は Hugging Face、格納は HF キャッシュ（`~/.cache/huggingface`）を正とする。
これは本プロジェクトの強み（HF の全モデルが使える）であり、Ollama の弱点を引き継ぐ理由がない。

---

## Phase 0: サービス化と堅牢化（サーバーを意識させない）✅ 実装済み（2026-07-23）

実装: 0a = `server.py` のワーカー台帳（`workers_state_path` / `reap_orphan_workers`、
`run_gateway` 起動時に照合）、0b = `tether.py`（パイプ EOF 検知の監視ラッパー）、
0c = `service.py` + `gw serve/enable/disable`（`make install` が enable を実行）、
0d = `tray.py`（メニューバーアイコン。macOS・AppKit 直接実装 — rumps は非推奨 API
依存で最新 macOS では表示されないため不使用。トレイ専用パイプの EOF で
「アイコンの存在＝デーモンの生存」。Ollama と同じ静的アイコンで常駐中の定期処理なし、
状態はメニューを開いた瞬間に取得。`tray = false` で無効化）、
0e = 更新 UX（新版検知でアイコンが「gw ⬆」に変わり、メニューの「今すぐ更新して再起動」
→ POST /admin/update で一度閉じて更新して再起動。検知は update watcher が常時行い
（auto_update=false でも表示だけはする）、通知はトレイ専用パイプへのプッシュ）。
テスト: `tests/test_reconcile.py` / `test_tether.py` / `test_service.py` / `test_tray.py`。

Ollama が「サーバーを意識せず使える」のは、常駐を OS に委任しているから。同じく
**デフォルトで自動起動**にする（2026-07-23 決定）。ただし前提となる堅牢化を先に入れる —
懸念は「自動復活で孤児プロセスのゴミが残る」「コーディングエージェントがサーバーを
乱立させ状況を追えなくなる」の 2 つであり、対策を積んでから委任する。

### 0a. 起動時の孤児掃除（startup reconciliation）

`gw stop` 側にある孤児対策（既知ポート走査 + `pid_looks_like_ours` + `stop_pid`、
cli.py `_collect_gateway_pids`）を**デーモン起動時にも実行**し、前回の残骸
（親を失ったモデルサーバー等）を回収してから立ち上がる。crash-only 設計:
起動処理 = 復旧処理。これにより自動復活は「ゴミが増える契機」ではなく
「ゴミが自動で掃除される契機」になる。手動運用のままでも価値がある。

### 0b. 子プロセスの親への繋留

モデルサーバー（llama-server / mlx サブプロセス）がデーモンの死を検知して自ら
終了する仕組み（process group、または親とのパイプ切断検知）。`gw stop` 漏れや
デーモンのクラッシュで子だけが残る事態を構造的に防ぐ。

### 0c. OS サービス登録（デフォルト ON・opt-out 可能）

- `make install` / インストール時に launchd agent（macOS）/ systemd user unit（Linux）
  を登録。ログイン時自動起動・異常終了時再起動（スロットリング付き:
  launchd `ThrottleInterval` / systemd `StartLimitBurst`）
- **`gw disable` で opt-out**（登録解除して手動運用に戻る）、`gw enable` で再登録
- Windows はサービス管理が別系統（Task Scheduler）なので後続対応。それまでは
  従来どおり手動 `gw start`
- シングルトン保証は既存機構で足りる: `gw start` は冪等（既に居れば何もしない）、
  runtime 記録で「唯一のデーモン」をどこからでも発見できる。真実は常に `gw status`
- **エージェント側のルールは不変**: agent-corporation の corp やエージェントは
  ライフサイクルに触らない（ユーザー専管 = ユーザーが OS へ委任することを選ぶ）。
  :8799 が常に応えることで、エージェントが自前サーバーを立てる動機そのものを消す

常駐が許される前提は既に揃っている: 遅延ロード + idle アンロードにより、
モデル未使用時の常駐コストはほぼゼロ（Ollama と同じ理屈）。

**受け入れ基準**: 再ログイン後に `gw status` が稼働を示す。`kill -9` でデーモンを
落としても子が残らず、次回起動時に残骸ゼロ。`gw disable` で完全に手動運用へ戻る。

## Phase 1: Ollama ネイティブ API（`/api/*`）❌ 見送り（2026-07-24 決定）

**作らない。** 価値は「Open WebUI 等の Ollama 用クライアントが繋がる」ことだけだが、
それらを使う予定が無く、接続はすべて OpenAI 互換（`/v1/*`）+ local-llm-client で
足りている。誰も呼ばない互換層は保守コストにしかならない。以下の設計メモは、将来
必要になった場合（Ollama 前提のツールを導入したくなった場合）のために残す。

daemon.py の HTTP ハンドラに `/api/*` ルートを追加し、内部的には既存の
`/v1/chat/completions` 転送パスへ変換する薄いアダプタとして実装する。

### 1a. 読み取り系（変換のみ・リスクなし）

- `GET /api/version` — バージョン文字列を返す
- `GET /api/tags` — `gw list` 相当（カタログ + HF キャッシュ）を Ollama の形式で返す
- `GET /api/ps` — ロード中モデル（`manager.status()` がほぼそのまま使える）
- `POST /api/show` — モデル詳細（config.json / GGUF メタデータから合成）

### 1b. 生成系

- `POST /api/chat` — messages 形式。**ストリーミングは NDJSON**（SSE ではない）なので
  変換層が必要。`images`（base64 配列）→ OpenAI の `image_url` へ変換
- `POST /api/generate` — プロンプト直指定。内部で chat 形式に包むか completions へ転送
- `options`（`temperature` / `num_ctx` / `num_predict` など）→ OpenAI パラメータへの対応表
- `keep_alive` パラメータ → 既存の `idle_timeout` / 在席セッション機構にマップ
  （`keep_alive: 0` = 即アンロード。**在席ベース即時アンロードが既にあるので流用できる**）

### 1c. モデル管理系

- `POST /api/pull` — `huggingface_hub.snapshot_download` を実行し、進捗を NDJSON で
  ストリーム返却（Ollama の `status` / `total` / `completed` 形式に合わせる）
- `DELETE /api/delete` — HF キャッシュから該当モデルを削除（ロード中なら先にアンロード）
- `POST /api/copy` / `POST /api/create` — Phase 4 のエイリアス/Modelfile 基盤の上に実装
  （Phase 1 時点では 501 を返してよい）

**受け入れ基準**: Open WebUI がモデル一覧を表示し、ストリーミングチャットできること。

## Phase 2: CLI 体験（`gw pull` / `rm` / `show`）✅ 実装済み（2026-07-24）

- `gw run`（ターミナル REPL）は**見送り** — 対話 UI は agent-corporation が担当
- `gw pull <model>` — 進捗バー付きの事前ダウンロード（`huggingface_hub.snapshot_download`
  を直接使う。Phase 1 を見送ったので `/api/pull` 経由にはしない）。
  「事前 DL 必須」ポリシー（ensure_cached / resolve_gguf）の正規の取得口。
  GGUF repo は**本体 1 種 + mmproj だけ**を選んで落とす（全量子化を巻き込まない。
  複数あればセレクタを案内）。取得後はサーバーと同じ基準で検証してから完了を名乗る
- `gw rm <model>` — HF キャッシュから削除（確認プロンプト付き・`-y` で省略。
  ロード中は拒否。自動では決して走らない）
- `gw show <model>` — バックエンド・量子化・コンテキスト長・サイズ・画像入力・MTP・
  ロード状態を表示（未取得なら gw pull を案内）

実装は cli.py（`cmd_pull` / `cmd_rm` / `cmd_show` + 純粋関数 `plan_gguf_pull` /
`build_show_rows`）。テストは `tests/test_model_files.py`。

cli.py は現在 stdlib argparse のみ。REPL と進捗バーも**依存追加なし**（`input()` +
ANSI エスケープ）で実装し、軽量インストールという既存方針を守る。

## Phase 3: 短縮モデル名とタグ解決

`gw run qwen3:8b` のように短い名前で呼べるようにする。

- エイリアス表（`aliases.toml` を同梱 + ユーザー定義は `~/.config/local-llm-server/` 側）:
  `qwen3:8b` → Apple Silicon なら `mlx-community/Qwen3-8B-4bit`、
  他 OS なら GGUF リポジトリ、のように**プラットフォームに応じて解決**
- タグ規約: `name:size[-quant]`（例 `qwen3:8b-q4`）。タグ省略時は既定タグ
- 解決は入口 1 箇所（daemon のモデル名解決）で行い、`/v1/*`・CLI すべてで有効
- `gw list` は短縮名を主として表示し、実体の HF ID を併記

## Phase 4: Modelfile 相当（`gw create`）⏸ 保留

system prompt・生成パラメータはエージェント側（agent-corporation の各 agent.toml）が
管理しており、ゲートウェイ側で持つ需要が薄い。必要になったら以下の設計で:

- `FROM`（HF ID または短縮名）/ `SYSTEM` / `PARAMETER` / `TEMPLATE` を解釈
- `gw create mymodel -f Modelfile` で `~/.config/local-llm-server/models/mymodel.toml`
  として保存（重みはコピーしない。参照 + 上書き設定のみ）
- 保存したモデル名でリクエストが来たら、system prompt / パラメータを注入して土台モデルへ転送
- `/api/create` / `/api/copy` をここで実装完了にする

## Phase 5: Embeddings（RAG 用途が出てきたら）

- バックエンド対応: llama-server は `--embeddings` フラグ、mlx 系は埋め込み対応モデルの
  ロードパスを追加
- `POST /v1/embeddings`（OpenAI 互換のみ。`/api/embed` は Phase 1 見送りに伴い作らない）
- agent-corporation 側で RAG（ベクトル検索）が必要になったときに着手する

---

## 順序と依存関係

```
Phase 0a → 0b → 0c → 0d → 0e ✅ 完了（堅牢化が先、サービス化・アイコン・更新はその上）
Phase 2（gw pull / rm / show）─→ Phase 3（短縮モデル名）
Phase 5（Embeddings）は独立・需要が出たら
```

- **Phase 0 は完了**。「サーバーを意識しない」体験（自動起動・自動復活・アイコン・更新）が本体だった
- 残りは CLI の利便性（Phase 2）と名前の短縮（Phase 3）。いずれも小粒（各 1〜2 日規模）で、
  急ぐ理由は無い——必要を感じたときにやる
- Phase 3 は設計を先に固める（名前解決の入口を 1 箇所にすること）

## 設計上の決めごと

- モデルの正規 ID は今後も HF リポジトリ ID。短縮名はあくまでエイリアス
- 接続 API は OpenAI 互換（`/v1/*`）の 1 系統だけ。互換層を増やさない
  （Ollama ネイティブ API が将来必要になったら Phase 1 の設計メモから再開する）
