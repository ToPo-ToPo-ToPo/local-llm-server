# gateway.toml リファレンス

サーバーはカレントディレクトリの `./gateway.toml` を 1 つの設定として読む。ここで運用方針（公開ポート・
同時常駐数・自動アンロード等）を決める。モデルは列挙しなくてよい（クライアントが指定した `model` を
動的ロードする）—— ID だけでは決まらない個別の上書きが要るモデルだけ `[[models]]` に書く。
リポジトリ直下にそのまま使える例を同梱している（[gateway.toml](../gateway.toml)）。

## 全フィールド

```toml
host = "127.0.0.1"          # 公開ホスト（省略時 127.0.0.1）
port = 8799                 # 公開ポート（省略時 8799）。クライアントの base_url はここ
max_resident = 1            # 同時常駐モデル数のハード上限。超えたら LRU 退避（省略時 無制限）
load_timeout = 300          # 全枠処理中のとき空くのを待つ最大秒数（超過で 503。省略時 300）
idle_timeout = 1200         # この秒数使われないモデルを自動アンロード（省略時 1200=20分。0 で無効）
session_ttl = 90            # 在席エージェントのハートビート猶予秒数。途絶でそのエージェントを無人扱い（省略時 90。0 で無効）
internal_base_port = 9001   # 内部モデルサーバーの割当開始ポート（9001, 9002, … と連番）
default_model = "..."       # model 省略リクエスト時のモデル（任意）
draft_model = "off"         # MTP（speculative decoding）の既定。省略時は mlx-vlm を "auto"（対応表から自動）。"off" で無効
dynamic = true              # 未登録モデルを ID 推論で動的ロード（省略時 true。false で事前登録のみ）
disable_thinking = false    # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先
auto_update = true          # TUI が PyPI 新版を検知して git pull で自動追従（省略時 true。→ 下記「自動更新」）

# [[models]] は任意（dynamic = true なら省略可）。mlx-vlm の MTP と画像入力は動的ロードでも自動で
# 効くので、それ目的の事前登録は不要。parallel・llama.cpp の MTP・llama-server への個別フラグ等、
# ID だけでは決まらない上書きが要るモデルだけ事前登録する。
[[models]]
model = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"  # HuggingFace の repo-id
backend = "llama-cpp"                          # mlx / mlx-vlm / llama-cpp
parallel = 4                                   # 動的ロードでは付かない個別オプションの例

[[models]]
model = "ToPo-ToPo/Qwen3.6-27B-mlx-4bit"
backend = "mlx-vlm"
# draft_model = "off"   # MTP は自動で効く。このモデルだけ無効化したいときだけ書く
```

| キー | 既定 | 説明 |
|---|---|---|
| `host` | `127.0.0.1` | bind ホスト。既定はローカルのみ。別PCから繋ぐなら `"0.0.0.0"`（全IF公開）か特定の IPv4 アドレス。IPv6（`"::"`）は非対応（→ [別PCから接続する](#別pcから接続するネットワーク公開)） |
| `port` | `8799` | 公開ポート（クライアントの `base_url` はここ） |
| `api_key` | なし | ネットワーク公開時の認証キー。設定するとクライアントは `Authorization: Bearer <key>` を要求される（省略/空で認証なし） |
| `max_resident` | 無制限 | 同時に常駐させるモデル数のハード上限。超過は LRU で退避 |
| `max_memory_fraction` | なし | 常駐モデルの推定占有量の合計を総RAMのこの割合（`0<x≤1`）に制限。超えるロードは退避→不足なら 503。→ [llama-cpp.md](llama-cpp.md#メモリガードmax_memory_fraction) |
| `parallel` | なし | 動的ロード時の並列スロット既定（**llama-cpp のみ**。mlx 系は無視）。各 `[[models]]` で上書き。→ [llama-cpp.md](llama-cpp.md#並列スロットparallel) |
| `load_timeout` | `300` | 全枠が処理中のとき空きを待つ最大秒数（超過で 503） |
| `start_timeout` | `120` | モデルサーバー1つの起動完了（ready）を待つ最大秒数。巨大モデルで足りなければ延ばす |
| `request_timeout` | `600` | 上流モデルサーバーとの通信が無応答のとき打ち切る秒数（`0` で無制限）。トークンが流れている限り切れない。ハングした／沈黙したサーバーが枠を塞ぎ続ける事故の保険 |
| `idle_timeout` | `1200` | この秒数使われないモデルを自動アンロード（`0` で無効） |
| `session_ttl` | `90` | 在席エージェントのハートビート猶予秒数。途絶で無人扱い（`0` で無効）。→ [在席ベースの即時アンロード](#在席ベースの即時アンロード) |
| `internal_base_port` | `9001` | 内部モデルサーバーの割当開始ポート |
| `default_model` | なし | `model` 省略リクエスト時に使うモデル |
| `vision_model` | なし | **画像を含むリクエストの振り分け先モデル**。設定すると、画像入りリクエストだけをこのモデル（画像が確実に動く gemma-4 系など）へ流す。テキストは元モデルのまま。画像が壊れている vision モデル（Qwen3.6-27B 等）の回避に使う。→ [mtp.md](mtp.md#画像入力vision) |
| `draft_model` | mlx-vlm は `auto` | 動的ロード時の MTP 既定。省略時は mlx-vlm が対応表から自動選択、`"off"` で無効。各 `[[models]]` で上書き。→ [mtp.md](mtp.md) |
| `dynamic` | `true` | 未登録モデルを ID 推論で動的ロードする。`false` で事前登録のみ（旧挙動） |
| `disable_thinking` | `false` | 動的ロード時の既定。事前登録モデルは各 `[[models]]` の値が優先 |
| `video_frames` | `8` | **動画入力**で 1 本から等間隔に抜くフレーム数。`video_url` をこの枚数の画像に展開して渡す |
| `video_max_edge` | `768` | 動画フレームの縮小サイズ（長辺 px）。大きいほど精細だがトークン増 |
| `[llama_cpp]` | 全自動 | `llama-server` の自動導入テーブル。`provision`（auto/system/build）・`accel`（auto/cuda/vulkan/metal/cpu）・`pin`（ビルド番号）。→ [llama-cpp.md](llama-cpp.md#自動導入llama_cpp) |

`[[models]]` は 1 モデル 1 エントリ。`model`（HuggingFace ID）と `backend`（`mlx` / `mlx-vlm` /
`llama-cpp` / `whisper`）が必須。各エントリで `draft_model` を上書きできる。`dynamic = true` なら
`[[models]]` は省略可（全て動的ロード）。`whisper` は音声→テキスト（STT）バックエンド
（→ [音声認識（STT / whisper）](#音声認識stt--whisper)）。

## 振る舞い

- **遅延起動**: 各モデルは**初回リクエスト時に起動**し、2 回目以降は常駐して即応答する。
- **動的ロード（`dynamic = true`）**: `[[models]]` に無いモデルもリクエストされた時点で起動・管理する。
  バックエンドは ID から推論（`whisper`/`parakeet`→whisper、`gguf`→llama-cpp、`mlx`→mlx-vlm、
  他→OS 既定。→ [docs/llama-cpp.md](llama-cpp.md)）。
  ロードされると一覧（`/v1/models`・ダッシュボード）に現れ、アンロードされると消える。すでにロード済みの
  モデルが再指定されたら**相乗り**（共有）する。**画像入力（mmproj 自動検出）と mlx-vlm の MTP（対応表に
  在る本体は `draft_model="auto"` を自動適用）は動的ロードでも効く**。一方 `parallel` や llama.cpp の MTP
  など ID だけでは決まらない上書きは付かないので、それが要るモデルだけ `[[models]]` に事前登録する。
  llama-cpp の repo-id は事前に取得済みである必要があり（未取得は 400）、mlx は HF から自動DLされる。
- **同一モデルの並列インスタンス（負荷ベース）**: 同じモデルに複数エージェントが集中し、既存インスタンスが
  すべて満杯になると、`max_resident` とメモリの範囲で**同一モデルの複製インスタンスを増やして並列化**する。
  → [同一モデルを並列化する（複数インスタンス）](#同一モデルを並列化する複数インスタンス)
- **モデル発見（ダウンロード済みの一覧）**: **TUI ダッシュボード**が、ロード中のモデルに加えて
  **HF キャッシュにある DL 済みのチャットモデル**を未ロード候補として並べる（`ollama list` 風に
  「いま手元で動かせるモデル」が一目で分かる）。判定はヒューリスティック（GGUF 本体、
  `*ForCausalLM`/`*ForConditionalGeneration` の mlx/重み repo、または whisper 系の STT repo）で、
  埋め込み・分類などの非チャット・非STT モデルは除外する。`/v1/models`（API）は標準どおり
  「登録済み＋ロード中」だけを返す。
- **LRU 退避**: 常駐数が `max_resident` を超えると、最も使われていないモデルから停止する。
  全枠が処理中なら空くまで待つ（OOM 回避。`load_timeout` で打ち切り→ 503）。
- **`max_resident` の実行中変更**: 再起動せず稼働中に上限を変えられる（→ [max_resident をライブで変える](#max_resident-をライブで変える)）。
  **処理中（busy）のモデルは止めない**ので、生成を中断せずに同時常駐数を増減できる。
- **メモリ上限**: `max_memory_fraction` を設定すると、常駐モデルの推定占有量の合計が総RAMの指定割合を
  超えるロードを拒否する（アイドル退避→不足なら 503）。→ [llama-cpp.md](llama-cpp.md#メモリガードmax_memory_fraction)
- **アイドル自動解放**: `idle_timeout` 秒使われないモデルをアンロードしてメモリを返す。
- **在席ベースの即時解放**: エージェントが利用終了を通知すると、そのモデルを使う在席が 0 になった
  瞬間（＝他に同じモデルへ接続しているエージェントが居ない）に、処理中でなければ `idle_timeout` を
  待たず即アンロードする。→ [在席ベースの即時アンロード](#在席ベースの即時アンロード)
- **1 公開ポートで集約**: 例 `http://127.0.0.1:8799/v1`。クライアントは公開ポートに繋ぎ、
  リクエストの `model` で振り分けられる（クライアントはサーバーを起動しない）。
- **ワーカー健全性チェック**: 掃除スレッドが定期的（~15秒）に各内部ワーカーの生存を確認し、
  クラッシュ（`kill -9` 等）で落ちたインスタンスを登録から外して枠を戻す。死んだワーカーへ
  リクエストを流し続けて 502 を返す事態を防ぎ、次のリクエストで新規ロードし直せるようにする。
  各ワーカーの PID は `GET /admin/status` の `models[].pids` で確認できる。
- **孤児ワーカーの回収**: 前回のクラッシュ / `kill -9` で内部ポートに取り残されたモデルサーバー
  （このパッケージ由来と判定できるものだけ）を、新しいワーカーを起動する直前に停止して回収する。
  ポート衝突による起動失敗（502）と、孤児が GPU メモリを掴んだままになる無駄を防ぐ。**無関係な
  別プロセス・ゲートウェイ自身には一切手を出さない**。
- **単一起動（1 マシン 1 ゲートウェイ）**: 起動時に OS レベルの排他ロック（`flock`）を取り、
  既にゲートウェイが動いていれば **2 個目を立てずに明示エラーで終了する**（終了コード 3、保持者の
  PID をログに出す）。ロックは **cwd 非依存の固定パス**（temp ディレクトリ）なので、別ディレクトリや
  別ポートから起動しても束ねられる（開発ツール等が裏で勝手に起動しても乱立しない）。ロックは
  プロセス生存中だけ握り、クラッシュ・`kill` を含む終了で OS が自動解放するため stale にならない。
- **画像入りリクエストの振り分け（`vision_model`）**: `vision_model` を設定すると、**画像を含む
  リクエストだけ**を（元の `model` に関わらず）そのモデルへ流す。テキストは元モデルのまま。
  現行 `mlx_vlm` では一部の vision モデル（Qwen3.6-27B 等の qwen3_5 系）が画像入力で壊れて
  いるため、画像だけを「画像が確実に動くモデル」（gemma-4 系など）へ逃がすのに使う。→ [mtp.md](mtp.md#画像入力vision)
- **設定のホットリロード**: `gateway.toml` を**保存した瞬間**にポリシー設定を無停止で反映する
  （プロセスは動かしたまま。~1 秒以内）。反映されるのは `vision_model`・`default_model`・
  `max_resident`・`request_timeout`・`idle_timeout`・`session_ttl`・`load_timeout`・`api_key`
  と動的ロードの既定（`draft_model`・`parallel`・`disable_thinking`・`max_memory_fraction`・
  `dynamic`・`start_timeout`）。動的ロード既定は**次回ロードから**有効。一方 `host`・`port`・
  `internal_base_port`・`[[models]]` はソケット bind 済み等で稼働中に変えられないため、変更を
  検知しても**適用せず「要再起動」をログ警告**する（サーバーは旧値のまま動き続ける）。編集途中の
  不正な TOML は無視して現行設定を維持する。反映内容は標準エラー（TUI のログ画面）に出る。
  → [ホットリロードの反映範囲](#ホットリロードの反映範囲)

MTP（speculative decoding）による高速化は [mtp.md](mtp.md) を参照。

## 音声認識（STT / whisper）

`backend = "whisper"` で mlx-whisper を **OpenAI 互換の STT サーバ**として束ねる。チャット/画像の
モデルとまったく同じく、初回リクエストで遅延起動し、LRU 退避・idle アンロード・在席即時解放・
`max_resident` のメモリ会計がそのまま効く。**狙いはエージェント側から mlx 依存を剥がすこと** ——
各エージェントは mlx-whisper を持たず、ゲートウェイの 1 ポートに音声を POST するだけでよい
（mlx-whisper のバージョンはこのサーバ 1 箇所で管理する）。

- **公開エンドポイント**（`model` は他と同じくリクエストで指定。動的ロードなら事前登録不要）:
  - `POST /v1/audio/transcriptions` … 文字起こし
  - `POST /v1/audio/translations` … 英訳
  いずれも OpenAI 仕様どおり **`multipart/form-data`**（`file` に音声、`model` にモデル ID）。
  `language` / `prompt` / `temperature` / `response_format`（`json`・`text`・`verbose_json`・
  `srt`・`vtt`）に対応する。
- **モデル ID**: whisper 系の mlx repo（例 `mlx-community/whisper-large-v3-turbo`、
  `mlx-community/whisper-large-v3-mlx`、`kaiinui/kotoba-whisper-v2.0-mlx`）。ID に `whisper` /
  `parakeet` を含めば動的ロードで自動的に whisper バックエンドへ振り分けられる。
- **要件**: 音声デコードに **ffmpeg CLI**（PATH 上）が要る（`brew install ffmpeg`）。本体重みは他の
  mlx 同様に事前 DL 必須（`hf download <repo>`。ゲートウェイは `HF_HUB_OFFLINE=1` で起動するため、
  未取得だとロード時にエラー）。

```bash
# 例: 文字起こし（クライアントは公開ポートに音声を投げるだけ。mlx 依存は不要）
curl http://127.0.0.1:8799/v1/audio/transcriptions \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "language=ja" \
  -F "file=@input.wav"
# → {"text": "..."}
```

OpenAI SDK からもそのまま使える（`client.audio.transcriptions.create(model=..., file=...)`）。
`base_url` を公開ポートに向けるだけで、振り分け・遅延起動・アンロードはゲートウェイが行う。

## 自動更新（PyPI 新版に git で追従）

このリポジトリは PyPI に公開しつつ、実運用は **GitHub から clone して `uv run gw`** で動かす。
そのままだと新版が出るたび手で `git pull` が要る。`auto_update = true`（既定）なら **TUI が
起動時と実行中（30分毎）に PyPI の最新版を確認し、新しければソースを追従して新コードで再起動する** ——
複数 PC を「開いておくだけで最新に揃う」状態にできる。

- **適用条件（安全側）**: **git クローン & upstream 追跡ブランチ & 作業ツリーがクリーン**のときだけ
  `git pull --ff-only`（＋`uv sync`）を実行する。開発中で未コミット変更がある PC では**適用せず**、
  ステータス行に「`⬆ x.y.z 利用可（ローカル変更あり・保留）`」を出すだけ（`u` キーで手動適用）。
  → その PC の編集中コードを勝手に上書きしない。
- **中断しない（drain 方式）**: 更新の取得（`git pull` + `uv sync`）は**稼働中のゲートウェイに
  触れずに**先に済ませる（この間も通常どおりリクエストを受ける）。再起動は、ゲートウェイ自身が
  **「処理中 0・在席エージェント 0」の確認と新規受付の停止を同一ロックで原子的に行う drain**
  （`POST /admin/drain`）が通ったときだけ実行する。確認と再起動の間に生成が滑り込んで
  強制終了される余地が無い。処理中/在席があれば何も止めずに保留し、空いた瞬間に再起動する
  （ステータス行に「⬆ 更新適用済み・処理中/在席が空き次第 再起動」）。drain 中〜再起動直後の
  数秒間に届いた新規リクエストは 503 になるが、クライアント（openai SDK / local-llm-client）は
  自動リトライするので新プロセスへ繋ぎ直される。drain は 120 秒で自動失効する
  （再起動側が死んでも受付不能のまま固まらない）。
- **git 運用でないとき**: `.git` が無い（PyPI から `uv tool install` した等）場合は何もしない。
- **無効化**: `auto_update = false`。
- **手動**: TUI の `u` キー（またはコマンド欄 `update`）でいつでも今すぐ確認・適用できる。

トリガーは PyPI の公開版。公開と同時に `main` へ push 済みなので、`git pull` で同じコードが得られる。

## 同一モデルを並列化する（複数インスタンス）

同じモデルに複数のエージェントが接続したとき、ゲートウェイは**負荷に応じて同一モデルのインスタンス
（プロセス）を複数起動**し、並列に捌く。挙動は次のとおり。

- **振り分け**: リクエストは、そのモデルの ready なインスタンスのうち**最も空いているもの**へ流す。
- **複製の起動（負荷ベース）**: 「最も空いているインスタンスすら満杯」＝リクエストが競合しているときだけ、
  **バックグラウンドで複製インスタンスを 1 つ増やす**。起動を待っている間も現在のリクエストは既存
  インスタンスへ流すので、**待たされない**（複製は将来の負荷に備えたウォームアップ）。
- **1 インスタンスの容量**: llama-cpp は 1 プロセス内の `parallel` スロット数（重み共有でメモリ効率が良い）。
  mlx / mlx-vlm は逐次のため 1。つまり llama-cpp は**まず parallel スロットを使い切ってから**複製し、
  mlx は 2 本目の同時リクエストで複製を検討する。
- **上限**: 複製を含めた**起動インスタンスの総数が `max_resident` を超えない**（メモリ上限
  `max_memory_fraction` も尊重）。枠が足りないときは、他モデルの**アイドル**インスタンスを LRU 退避して
  空ける。空けられない（残りが全て処理中）ときは複製せず、既存インスタンスで捌く（**処理中は止めない**）。
- **縮小**: 各インスタンスは独立に idle_timeout / LRU / 在席解放の対象になり、負荷が引けば通常どおり
  アンロードされる。

> **メモリに注意**: 複製インスタンスは**モデルの重みを複製**する（例: 27B・4bit ≒ 16GB を 2 本で ≒ 32GB）。
> 並列化したい本数は搭載メモリと相談し、`max_resident`（総インスタンス数）や `max_memory_fraction` で上限を張る。
> **どちらも設定していない場合、複製は行われない**（重みのコピーが際限なく増えて OOM する事故を防ぐため）。
> 単一プロセス内で並列化できる **llama-cpp は `parallel=N` の方がメモリ効率が良い**（複製は主に mlx 系で効く）。

各モデルの起動インスタンス数は `GET /admin/status` の `models[].instances`、TUI では STATE 列の「×N」で見える
（`max_resident` を 1 のままにすると複製は起きない＝従来どおり 1 モデル 1 プロセス共有）。

## max_resident をライブで変える

同時常駐モデル数の上限（`max_resident`）は、ゲートウェイを**再起動せずに実行中へ反映**できる。
複数モデルを並行常駐させたいとき（`max_resident` を上げる）や、逆に絞りたいときに、`gateway.toml`
を書き換えて再起動する必要はない。

- **TUI から**: コマンド欄に `max <n>` を入力する（例 `max 2`、無制限は `max off`）。`m` キーで
  コマンド欄に `max ` を入れた状態でフォーカスするので、数値だけ打って Enter でもよい。変更後の値は
  ダッシュボード下部の `max_resident` 表示に即反映される。
- **API から**: `POST /admin/config` に `{"max_resident": N}`（`N` は 1 以上、`null`/`0`/`"off"` で
  無制限）。

**処理中（busy）のモデルは止めない。** 上限を下げたとき、超過分は**アイドルなモデルからのみ** LRU で
非同期に退避する。全て処理中なら 1 つも止めず、生成が終わって枠が空いた時点（次の release/acquire）
または `idle_timeout` で片付ける。上限を上げたときは、枠待ちで止まっていたロードを起こすだけ。

この変更は**実行中のみ**有効で、`gateway.toml` には書き戻さない。ゲートウェイを再起動すると
`gateway.toml` の値に戻る（恒久的に変えたいときはファイルの `max_resident` を編集する）。

| 変更 | エンドポイント | ボディ | 効果 |
|---|---|---|---|
| 上限変更 | `POST /admin/config` | `{"max_resident": 2}` | 常駐上限を 2 に。busy は止めず超過アイドルを非同期退避 |
| 無制限化 | `POST /admin/config` | `{"max_resident": null}` | 上限撤廃（メモリが許す限り常駐）|

（`POST /admin/config` / TUI の `max` は**実行中だけの一時変更**。恒久的に変えるなら次節のとおり
`gateway.toml` を編集する —— 保存すれば同じく無停止で反映され、そちらが永続値になる。）

## ホットリロードの反映範囲

サーバーを**起動しっぱなしのまま**、`gateway.toml` を編集・保存するだけで設定を反映する。ゲートウェイは
ファイルの更新時刻を ~1 秒周期で監視し、保存を検知したら読み直して**無停止で適用**する（反映内容は
標準エラー＝ TUI のログ画面に出る）。運用中にモデルを落とさず方針だけ変えられる。

| 種別 | 対象 | 反映 |
|---|---|---|
| **即時反映（ポリシー）** | `vision_model`, `default_model`, `max_resident`, `request_timeout`, `idle_timeout`, `session_ttl`, `load_timeout`, `api_key` | 保存した瞬間に有効 |
| **次回ロードから（動的既定）** | `draft_model`, `parallel`, `disable_thinking`, `max_memory_fraction`, `dynamic`, `start_timeout` | 既にロード済みのモデルは次にロードし直すまで旧設定のまま |
| **要再起動（構造）** | `host`, `port`, `internal_base_port`, `[[models]]` | 稼働中は変えられない（ソケット bind 済み・内部ポート割当は起動時固定）。変更を検知しても**適用せず「要再起動」をログ警告**し、旧値のまま動き続ける |

- `max_resident` の即時反映は `POST /admin/config` と同じ挙動（**busy は止めず、超過アイドルのみ非同期
  LRU 退避**。→ [max_resident をライブで変える](#max_resident-をライブで変える)）。
- **編集途中の壊れた TOML は無視**して現行設定を維持する（保存の瞬間に構文エラーがあってもサーバーは
  落ちない）。直後に有効な内容を保存すれば、また反映される。
- 構造設定（`host`/`port` 等）を本当に変えたいときだけ、TUI の `r`（再起動）またはプロセス再起動を行う。

## 別PCから接続する（ネットワーク公開）

既定ではゲートウェイは `127.0.0.1`（ローカルのみ）に bind するため、別PCからは繋がらない。LAN 上の
他のマシンから使うには次のようにする。

1. **ネットワークに bind する** — `gateway.toml` で `host = "0.0.0.0"`（全インターフェース）にする。
   起動ログ／TUI に、リモートのクライアントが指す `reachable from LAN: http://<このPCのIP>:8799/v1` が
   表示される（TUI では下部の `LAN …` 行）。
2. **API キーを設定する（推奨）** — `api_key = "<長めのランダム文字列>"` を設定する。クライアントは
   リクエストに `Authorization: Bearer <key>` を付ける必要があり、無い/不一致なら **401**。未設定なら認証
   なし（＝LAN 上の誰でも叩ける）。
3. **クライアント側** — 各クライアントの `base_url` を `http://<ゲートウェイPCのIP>:8799/v1` にし、API キーを
   設定していれば `Authorization: Bearer <key>` を送るようにする（OpenAI 互換クライアントの `api_key`
   相当）。
4. **ファイアウォール** — 受信接続（該当ポート）を許可する。macOS なら「システム設定 → ネットワーク →
   ファイアウォール」で許可。

**セキュリティの要点**:

- **内部のモデルサーバーは常に `127.0.0.1` のまま**で、外部に晒されない。公開されるのは公開ポート
  （ゲートウェイ）だけ。
- **`/admin/status` と `/admin/config`（状態・設定変更・複製制御）は同一マシン限定**（ループバック、
  または特定 IP に bind した場合はその IP からの自己接続も可）。リモートからは **403**。
  `max_resident` の変更や状態監視はゲートウェイPC本体（TUI/CLI）からだけ行える。
- **chat（`/v1/*`）と在席セッション（`/admin/sessions/*`）は API キーで保護**（設定時）。在席セッションは
  クライアントプロトコルの一部なのでリモートからも使えるが、キーが要る。
- API キーの比較はタイミング安全（`hmac.compare_digest`）。キーはログに出さない。

> 公開（`host` が非ループバック）かつ `api_key` 未設定のときは、起動時に**警告**を出す。信頼できる閉じた
> LAN 以外では必ず `api_key` を設定すること。

## 在席ベースの即時アンロード

`idle_timeout`（既定20分）は「最後のリクエストから一定時間」でアンロードする保険だが、エージェントが
明示的に「使い終わった」と通知すれば**待たずに即メモリ解放**できる。エージェントが利用開始/終了を
ゲートウェイに登録し、あるモデルの在席エージェントが 0 になった瞬間（＝そのモデルを使う人が誰も
居ない）に、処理中（`inflight>0`）でなければ即アンロードする。

これは GPU/RAM が逼迫する `max_resident = 1` 運用で特に効く（あるエージェントが終わった瞬間に枠が
空き、次のモデルへの切り替えが速くなる）。在席はメモリを**ピン留めしない** — 枠が足りなければ従来
どおり LRU 退避が優先される（OOM 回避）。あくまで「使う人が居なくなったら早く片付ける」仕組み。

### プロトコル（管理エンドポイント）

チャット転送（`/v1/...`）とは別系統。公開ポートに対して次を叩く。

| 操作 | リクエスト | ボディ | 補足 |
|---|---|---|---|
| 利用開始 | `POST /admin/sessions/register` | `{"agent_id", "model"}` | 在席を宣言。モデルは従来どおり初回リクエストで遅延ロード |
| 生存通知 | `POST /admin/sessions/heartbeat` | `{"agent_id"}` | `session_ttl` 内に定期送信。未知の `agent_id` は 404（要再 register） |
| 利用終了 | `POST /admin/sessions/release` | `{"agent_id"}` | `DELETE /admin/sessions` でも可。最後の在席なら即アンロード |

- **正常終了**は `release` で即解放（最速）。
- **異常終了**（`release` を呼べずに落ちた）はハートビートが `session_ttl` 秒途絶した時点で掃除
  スレッドが無人扱いし、同じく即アンロードする。ハートビート間隔は `session_ttl` より十分短くする
  （例: TTL 90s に対し 30s ごと）。`session_ttl = 0` で無効化（`release` のみで運用）。
- 各モデルの在席数は `GET /admin/status` の `models[].sessions`、および TUI の `AGENTS` 列で見える。

### エージェント側の実装（任意・推奨）

通知は**任意**で、登録しなければ従来どおり `idle_timeout` でのみ解放される。即時解放したいエージェント
だけ、起動時に `register`＋ハートビート、終了時に `release` を仕込めばよい。base_url は従来どおり公開
ポートのまま、追加でこの数本を叩くだけ（チャットの送り方は変えない）。

標準ライブラリだけで完結する最小実装の例:

```python
import atexit, json, signal, threading, urllib.request

class GatewaySession:
    """ゲートウェイに在席を登録し、終了時に即アンロードさせるヘルパー。

        with GatewaySession(agent_id="agent-7", model="org/Model:Q4"):
            ...  # base_url=http://127.0.0.1:8799/v1 でいつものチャット
        # ブロックを抜けた瞬間、他に同モデル利用者が居なければメモリが即解放される
    """
    def __init__(self, *, base="http://127.0.0.1:8799", agent_id, model, heartbeat=30):
        self.base, self.agent_id, self.model, self.hb = base, agent_id, model, heartbeat
        self._stop = threading.Event()

    def _call(self, path, payload):
        req = urllib.request.Request(
            self.base + path, json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass  # ゲートウェイ未起動でもエージェント本体は止めない

    def __enter__(self):
        self._call("/admin/sessions/register", {"agent_id": self.agent_id, "model": self.model})
        threading.Thread(target=self._beat, daemon=True).start()
        atexit.register(self.release)                       # プロセス終了時の保険
        signal.signal(signal.SIGTERM, lambda *_: self.release())  # kill されたら解放
        return self

    def _beat(self):
        while not self._stop.wait(self.hb):
            self._call("/admin/sessions/heartbeat", {"agent_id": self.agent_id})

    def release(self):
        if not self._stop.is_set():
            self._stop.set()
            self._call("/admin/sessions/release", {"agent_id": self.agent_id})

    def __exit__(self, *exc):
        self.release()
```

`with` ブロックを抜ける／プロセスが終わる／`SIGTERM` で殺される、のいずれでも `release` が呼ばれる。
万一それも取りこぼしても、ハートビート途絶で `session_ttl` 後に回収される（二重の安全網）。

> `agent_id` はエージェントごとに一意な文字列にする（PID やUUID等）。同一 `agent_id` で別 `model` を
> `register` し直すと、旧モデルから自動的に外れる（乗り換え。旧モデルが無人になれば解放される）。
