# Linux / Windows 対応 ✕ llama.cpp 自動導入 — 開発方針と計画

対象ブランチ: `feat/crossplatform-llamacpp`（作業は全フェーズこの系列のブランチで行い、
フェーズ単位で main へマージ・リリースする）

## ゴール

1. **OS 自動判定で導入が完結する**: Linux / Windows / macOS のどれで `uv run gw` しても、
   llama.cpp（`llama-server`）が**自動でダウンロード（必要ならビルド）され、そのまま動く**。
2. **計算効率を最大限に**: GPU/CPU を自動検出し、`-ngl`（GPU オフロード）・`--parallel`
   （継続バッチング）・スレッド数・flash-attention 等の既定を環境に合わせて自動決定する。
3. **画像・動画入力**: 画像は全 OS で動作保証、動画はゲートウェイ側のフレーム抽出で
   全バックエンド共通に対応する。

## 現状の棚卸し（2026-07-10 時点 v0.28.0）

すでにあるもの（この開発で壊さない資産）:

- **OS 判定の骨格**: `default_backend()` が Apple Silicon → mlx-vlm / それ以外 → llama-cpp を
  既に自動選択。mlx 系依存は pyproject の環境マーカーで darwin/arm64 のみインストール。
- **Windows 対応の下地**: 単一起動ロック（fcntl.flock / msvcrt.locking の両対応）、
  プロセス管理（setsid / DETACHED_PROCESS・taskkill）、ffmpeg.exe 解決。
- **llama-cpp バックエンド本体**: `llama-server` の起動・`--parallel`・mmproj 自動検出
  （画像入力）・MTP（embedded / 別ヘッド）・メモリガード。
- **ゲートウェイ機能**: 動的ロード・LRU・在席即時解放・ワーカー健全性チェック・
  ホットリロード・単一起動・起動元表示。

欠けているもの（この開発で作るもの）:

| # | ギャップ | 内容 |
|---|---|---|
| G1 | バイナリ導入が手動 | `llama-server` は PATH 前提。brew/winget/Releases の手動導入が必要 |
| G2 | 効率チューニングが手動 | `-ngl` を渡しておらず GPU オフロード任せ。parallel/threads/fa も未自動化 |
| G3 | 動画入力が無い | どのバックエンドも video を受けられない |
| G4 | Linux/Windows の検証手段が無い | CI ゼロ。開発機は macOS のみ |
| G5 | STT が Apple 専用 | mlx-whisper 依存（本計画ではスコープ外、将来 whisper.cpp で対応） |

## 全体方針

### 方針A: バイナリ導入は「プリビルト優先・ソースビルドは opt-in」

**質問「自動でコンパイルまでされる仕様は可能か？」への答え: 可能。ただし既定にはしない。**

- llama.cpp 公式（ggml-org）は **GitHub Releases に OS ✕ アクセラレータ別のプリビルト
  バイナリをビルド番号ごとに公開**しており、**同一バージョンなら配布バイナリとソースビルドで
  挙動は同一**（推論結果・精度・モデル互換性に差なし。差が出るのはビルド構成のみ）。
- 一方ソースビルドは cmake + C++ ツールチェーン（Windows は VS Build Tools、CUDA 版は
  CUDA Toolkit）が必要で、「導入が簡単」という要件と真っ向から衝突する。
- よって既定は **プリビルト自動ダウンロード**。ソースビルドは
  `llama_cpp = "build"` の明示 opt-in で提供し、`-march=native` 最適化や配布版に無い
  構成が欲しい上級者だけが使う。ビルド失敗時はプリビルトへ自動フォールバックする。

```toml
# gateway.toml（新設。すべて省略可 = 全自動）
[llama_cpp]
provision = "auto"    # auto: 管理バイナリを自動導入（既定）/ system: PATH の llama-server を使う
                      # / build: ソースから自動ビルド（cmake・ツールチェーン必要。失敗時 auto へ）
# accel = "auto"      # auto（GPU なら vulkan / mac は metal / 無ければ cpu）/ cuda / vulkan / metal / cpu
# pin = "b9946"       # ビルド番号の固定（省略時は最新を取得し、以後は導入済みを使い続ける）
```

- 導入先は HF キャッシュと同じ発想の**管理ディレクトリ**
  （`~/.cache/local-llm-server/llama.cpp/<build>/`。Windows は `%LOCALAPPDATA%`）。
  PATH は汚さず、ゲートウェイが絶対パスで起動する。PATH に既存の `llama-server` が
  あれば `provision = "system"` で従来どおり尊重。
- **アクセラレータ検出（実装で修正）**: 当初 CUDA 優先を想定したが、実 Releases 調査で
  **Linux に CUDA プリビルトが無く**、**Windows CUDA は別途 cudart DLL が要る**と判明。
  「導入が簡単・確実に動く」を優先し、**auto の GPU 経路は Vulkan に一本化**した
  （NVIDIA/AMD/Intel 共通・単一アセット・追加ランタイム不要）。判定順は macOS→Metal（内蔵）、
  Linux/Windows は GPU 検出（`nvidia-smi` or `vulkaninfo`）→ Vulkan、無ければ CPU。
  **CUDA は Windows 限定の明示 opt-in**（`accel = "cuda"`）。ROCm も同様に Vulkan 経路で代替。
- Releases のアセット命名は上流都合で変わり得るため、命名パターンはコード内テーブル＋
  gateway.toml で上書き可能にしておく（壊れたら設定で即応、後追いでコード修正）。

### 方針B: 計算効率は「検出 → 安全な最大値を既定に、明示指定が常に優先」

| 項目 | 自動決定ロジック | 上書き手段 |
|---|---|---|
| GPU オフロード | GPU 検出時 `-ngl 999`（全層）。VRAM 不足は llama.cpp 側の再配置に任せ、失敗ログを TUI に露出 | `[[models]] extra_args` |
| 並列スロット | `--parallel`: GPU なら 4、CPU のみなら 物理コア数/4（最低1）を既定に。ゲートウェイの複製インスタンス機構とは役割分担（1プロセス内=continuous batching、プロセス複製=負荷分散）を docs に明記 | gateway.toml `parallel` |
| スレッド | `--threads` 物理コア数（P コアのみ相当）。Windows の E コア混在は psutil で物理コア取得 | extra_args |
| flash-attn | GPU 検出時 `--flash-attn` を既定付与（未対応バックエンドでは llama.cpp が無視/エラー → 起動失敗検知でリトライ時に外す） | extra_args |
| KV 量子化 | 既定では付けない（品質影響があるため）。docs で `-ctk q8_0` 等を案内のみ | extra_args |

- 実装は「自動フラグは**ユーザーの extra_args に同項目が無いときだけ**付ける」原則で統一
  （mmproj 自動検出と同じ流儀。既存ユーザーの設定を壊さない）。
- 検証用に `gw` コマンド入力欄へ `bench [model]` を追加（短文生成の tok/s を測って表示）。
  チューニングの効果が数字で見えるようにする。

### 方針C: 動画・画像は「ゲートウェイ層で正規化」

- **画像**: llama.cpp は mmproj 自動検出で既に対応済み。Linux/Windows での動作検証のみ
  （Phase 4 の CI/実機で確認）。
- **動画**: `llama-server` は動画を直接受けられない。**ゲートウェイが動画を受けて
  フレーム画像列に変換して転送**する方式にする:
  - `/v1/chat/completions` の content に `video_url`（data URI / ローカルパス）が来たら、
    ffmpeg で **N フレームを等間隔サンプリング**（既定 8 フレーム・長辺 768px に縮小、
    gateway.toml で調整可）→ `image_url` パーツ列に展開して上流へ渡す。
  - この方式は **mlx-vlm でもそのまま効く**（バックエンド非依存。ゲートウェイ層の変換だから）。
  - ffmpeg は STT で実績のある `imageio-ffmpeg` を**全 OS の依存に昇格**して同梱
    （現在は darwin/arm64 のみ）。brew/apt 不要の方針を維持。
  - client 側（local-llm-client）にも `respond(..., videos=[...])` ヘルパを追加する
    （別リポジトリ・別リリース。結線テストに動画ケースを追加して契約を守る）。
- **音声（読み込み）**: llama.cpp の mtmd は一部モデルで音声入力に対応しつつあるが、
  モデル・機能とも流動的なためスコープ外（STT は既存の whisper バックエンドが担当）。

### 方針D: 検証は「CI マトリクス + 実機スモーク」の二段構え

- 開発機が macOS のみなので、**GitHub Actions の 3OS マトリクス（ubuntu / windows /
  macos）を最初に整備する**（Phase 0）。テストは元々フェイク中心・mlx 依存は環境マーカーで
  自動除外のため、大半はそのまま 3OS で走るはず。走らない箇所（mlx 前提のテスト）に
  skip マーカーを付けるところから始める。
- プロビジョナ（実ダウンロード）は CI で実バイナリ取得＋ `llama-server --version` 実行までを
  スモークテストする（巨大モデルのロードはしない）。
- リリース前の実機確認は、ユーザーの Linux / Windows 実機があればそこで Tier2 相当
  （実モデルでの生成・画像・動画）を行う。無ければ CI スモークまでで段階リリースする。

## フェーズ計画

| Phase | 内容 | 主な成果物 | リリース |
|---|---|---|---|
| **0** ✅ | CI 3OS マトリクス整備 | `.github/workflows/test.yml`、Windows ロック修正・SIGTERM テスト skip、3OS 全緑 | なし（インフラ） |
| **1** ✅ | llama.cpp プロビジョナ | `provisioner.py`（OS/arch/アクセラレータ検出・Releases 取得・管理dir・展開・検証）、`[llama_cpp]` 設定、起動時配線、build/accel の TUI・/admin/status 表示 | 0.29.0 |
| **2** ✅ | 効率自動チューニング | accel に応じた `-ngl 999`（GPU）/`--threads`（CPU）自動付与、TUI `bench` コマンド（tok/s） | 0.30.0 |
| **3** ✅ | ソースビルド opt-in | `provision = "build"`（cmake/git 検出、CUDA/Vulkan/HIP フラグ、失敗時 auto フォールバック） | 0.31.0 |
| **4** ✅ | 動画入力 | `video.py`（video_url → ffmpeg フレーム抽出 → image_url 展開・バックエンド非依存）、`video_frames`/`video_max_edge` 設定、imageio-ffmpeg 全OS化 | 0.32.0 |
| **5** ✅ | 導入導線・総仕上げ | README（自動導入・動画を反映）、docs/llama-cpp.md に自動導入節、gateway.toml 例 | 0.33.0 |

すべて `feat/crossplatform-llamacpp` ブランチ上で実装・3OS CI 緑まで完了（main 統合は最後にまとめて）。
実装で得た主な学び: (1) アクセラレータは Vulkan 一本化（実アセット調査）、(2) macOS は llama-cpp 使用時
のみ導入、(3) CI が Windows の実バグ（強制ロックでの PID 読み取り）を検出。

**残タスク（別リポジトリ / 実機）**:
- client（local-llm-client）の `respond(..., videos=[...])` ヘルパと結線テスト拡充（別リポ・0.7.0）。
  現状もサーバー側は raw `video_url` を受けられるので、クライアントから直接送れば動く。
- 実 GPU（Linux/Windows 実機）での自動導入・生成・動画の Tier2 スモーク（CI では実 GPU 検証不可）。

## 各フェーズの詳細

### Phase 0: CI（前提インフラ）

- GitHub Actions: `{ubuntu-latest, windows-latest, macos-latest} × Python 3.12`、
  `uv sync --dev && uv run pytest -q`。
- 予想される作業: パス区切り・改行・`/tmp` 直書き等の Windows 非互換をテストから排除、
  mlx 前提テストに `skipif(not darwin/arm64)`。
- 完了条件: 3OS でテスト全緑がバッジで見える。

### Phase 1: プロビジョナ（本丸）

新モジュール `local_llm_server/provisioner.py`:

1. `detect_platform()` → `(os, arch, accel)`（例 `("linux", "x64", "vulkan")`）
2. `resolve_asset(build, platform)` → Releases のアセット名（命名テーブル + 上書き設定）
3. `ensure_llama_server()` → 管理ディレクトリに導入済みならそのパス、無ければ
   ダウンロード → 展開 → `llama-server --version` で自己検証 → パスを返す。
   失敗時は明確なエラー（URL・手動導入の docs へ誘導）。
4. `build_command()` の llama-cpp 分岐が `"llama-server"`（PATH 前提）の代わりに
   `ensure_llama_server()` の絶対パスを使う（`provision = "system"` なら従来どおり）。
- ダウンロードは初回リクエスト時ではなく**ゲートウェイ起動時**に行う（初回推論の
  レイテンシに混ぜない。起動ログ / TUI に進捗を出す）。
- 更新は自動更新と同じ思想で「TUI の update 導線」に統合（`pin` 指定時は固定）。
- **実装時の判断（macOS の無駄DL回避）**: 起動時プロビジョニングは「llama-cpp が使われる
  構成」でだけ走らせる（llama-cpp モデルが事前登録済み or OS 既定が llama-cpp＝非 Apple）。
  Apple Silicon（既定 mlx-vlm）で llama-cpp モデル未登録の場合は自動導入しない。macOS で
  GGUF を動的要求したいときは `[[models]]` 登録か `provision = "system"` を使う（macOS の
  初回 GGUF 遅延自動導入は将来の改善候補）。

### Phase 2: 効率チューニング

- `detect_compute()`（GPU 種別・VRAM・物理コア）を provisioner の検出と共通化。
- 自動フラグ付与は build_command 内で「ユーザー未指定のときだけ」原則。
- `bench` コマンド: ロード済みモデルに 128 tok 生成を投げ、tok/s を TUI に表示。

### Phase 3: ソースビルド

- `git clone --depth 1` → `cmake -B build -DGGML_CUDA=ON` 等 → 成果物を管理ディレクトリへ。
- ツールチェーン検出（cmake / cl.exe / gcc / nvcc）に失敗したら**その場で auto（プリビルト）へ
  フォールバックし、何が足りなかったかをログに残す**（ビルド失敗でゲートウェイが立たない、を
  絶対に起こさない）。

### Phase 4: 動画入力

- daemon 層に `_expand_video_parts(body)`: `video_url` パーツを検出 → ffmpeg で
  フレーム抽出（`-vf fps=…,scale=…`）→ base64 `image_url` パーツ列に置換。
- 設定: `video_frames = 8` / `video_max_edge = 768`（gateway.toml・ホットリロード対象）。
- 大きい動画対策: `_MAX_BODY_BYTES`（既存 100MB）を据え置き、超過は 413 のまま。
- client: `build_user_content(text, images=…, videos=…)` 拡張 + 結線テスト（別リポ）。

### Phase 5: 導入導線

- README トップに OS 別 3 行クイックスタート（`uv tool install local-llm-server` →
  `gw` → モデル ID を投げるだけ）。
- docs/llama-cpp.md の手動導入節は「provision = system 向けの補足」に格下げ。

## リスクと対応

| リスク | 対応 |
|---|---|
| Releases のアセット命名変更で自動DLが壊れる | 命名テーブルを設定で上書き可能に。失敗時は明確なエラー + system フォールバック案内 |
| Windows 実機での未知の非互換 | Phase 0 の CI を先行させ、Windows runner で全テスト + プロビジョナのスモークを常時実行 |
| GPU 検出の誤判定（古いドライバ等） | `accel` 明示指定で常に上書き可能。検出結果は起動ログ / admin/status に出して見える化 |
| 自動フラグがモデル/ビルドによっては起動失敗を招く | 起動失敗を健全性チェックで検知したら自動フラグを外して 1 回だけ再起動を試みる |
| 動画で リクエストが巨大化 | フレーム数・解像度の既定を控えめに。413 の既存ガードを維持 |
| ソースビルドの環境差は無限にある | 既定にしない。失敗はフォールバックで吸収し、サポート表明はプリビルト構成のみ |

## スコープ外（明示）

- STT の Linux/Windows 対応（whisper.cpp 統合）— 別計画
- ROCm ネイティブビルドの公式サポート — Vulkan 経路で代替
- llama.cpp の音声入力（mtmd audio）— モデル側の成熟待ち
- Intel Mac 最適化 — llama-cpp CPU/Metal プリビルトで動作はする（性能チューニング対象外）
