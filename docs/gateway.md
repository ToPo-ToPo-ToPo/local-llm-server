# gateway.toml リファレンス

サーバーは `~/.config/local-llm-server/gateway.toml`（初回の `gw start` が自動生成）を
唯一の設定として読む。ここで運用方針（公開ポート・
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
internal_base_port = 9001   # 内部モデルサーバーの割当開始ポート（9001, 9002, … と連番）
default_model = "..."       # model 省略リクエスト時のモデル（任意）
draft_model = "off"         # MTP（speculative decoding）の既定。省略時は mlx-vlm を "auto"（対応表から自動）。"off" で無効
dynamic = true              # 未登録モデルを ID 推論で動的ロード（省略時 true。false で事前登録のみ）
disable_thinking = false    # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先
auto_update = true          # 常駐デーモンが新しいリリースタグを検知して自動追従（省略時 true。→ 下記「自動更新」）
                            # false でも新版チェックは行い、メニューバーの更新マークで知らせる（適用だけ止まる）
tray = true                 # 稼働中メニューバーに「gw」アイコンを表示（macOS のみ。省略時 true。false で非表示）

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
| `internal_base_port` | `9001` | 内部モデルサーバーの割当開始ポート |
| `default_model` | なし | `model` 省略リクエスト時に使うモデル |
| `draft_model` | mlx-vlm は `auto` | 動的ロード時の MTP 既定。省略時は mlx-vlm が対応表から自動選択、`"off"` で無効。各 `[[models]]` で上書き。→ [mtp.md](mtp.md) |
| `dynamic` | `true` | 未登録モデルを ID 推論で動的ロードする。`false` で事前登録のみ（旧挙動） |
| `disable_thinking` | `false` | 動的ロード時の既定。事前登録モデルは各 `[[models]]` の値が優先 |
| `video_frames` | `8` | **動画入力**で 1 本から等間隔に抜くフレーム数。`video_url` をこの枚数の画像に展開して渡す |
| `video_max_edge` | `768` | 動画フレームの縮小サイズ（長辺 px）。大きいほど精細だがトークン増 |
| `image_max_edge` | `1024` | **静止画の長辺上限 px**（`0` で無効）。上流へ渡す前に縮小し、vision トークンの膨張を防ぐ。→ [画像入力の縮小](#画像入力の縮小image_max_edge) |
| `[llama_cpp]` | 全自動 | `llama-server` の自動導入テーブル。`accel`（auto/cuda/vulkan/metal/cpu）・`pin`（ビルド番号）。導入方法の選択肢は無い（常に自動導入）。→ [llama-cpp.md](llama-cpp.md#自動導入llama_cpp) |

`[[models]]` は 1 モデル 1 エントリ。`model`（HuggingFace ID）と `backend`（`mlx` / `mlx-vlm` /
`llama-cpp` / `whisper`）が必須。各エントリで `draft_model` を上書きできる。`dynamic = true` なら
`[[models]]` は省略可（全て動的ロード）。`whisper` は音声→テキスト（STT）バックエンド
（→ [音声認識（STT / whisper）](#音声認識stt--whisper)）。

## 振る舞い

- **遅延起動**: 各モデルは**初回リクエスト時に起動**し、2 回目以降は常駐して即応答する。
- **動的ロード（`dynamic = true`）**: `[[models]]` に無いモデルもリクエストされた時点で起動・管理する。
  バックエンドは ID から推論（`whisper`/`parakeet`→whisper、`gguf`→llama-cpp、`mlx`→mlx-vlm、
  他→OS 既定。→ [docs/llama-cpp.md](llama-cpp.md)）。
  ロードされると一覧（`/v1/models`・`gw list`）に現れ、アンロードされると消える。すでにロード済みの
  モデルが再指定されたら**相乗り**（共有）する。**画像入力（mmproj 自動検出）と mlx-vlm の MTP（対応表に
  在る本体は `draft_model="auto"` を自動適用）は動的ロードでも効く**。一方 `parallel` や llama.cpp の MTP
  など ID だけでは決まらない上書きは付かないので、それが要るモデルだけ `[[models]]` に事前登録する。
  llama-cpp の repo-id は事前に取得済みである必要があり（未取得は 400）、mlx は HF から自動DLされる。
- **同一モデルの並列インスタンス（負荷ベース）**: 同じモデルに複数エージェントが集中し、既存インスタンスが
  すべて満杯になると、`max_resident` とメモリの範囲で**同一モデルの複製インスタンスを増やして並列化**する。
  → [同一モデルを並列化する（複数インスタンス）](#同一モデルを並列化する複数インスタンス)
- **モデル発見（ダウンロード済みの一覧）**: **`gw list`** が、ロード中のモデルに加えて
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
- **画像入力の縮小（`image_max_edge`）**: 長辺がこの px を超える画像は、上流へ渡す前に縮小する
  （既定 1024px、`0` で無効。拡大はしない）。解像度に比例して vision トークンが増えるモデル
  （Qwen3.6 等の qwen3_5 系）に巨大画像を渡すと、1 枚が数千トークンになり Dense モデルでは
  prefill だけで数十秒かかる。→ [画像入力の縮小](#画像入力の縮小image_max_edge)
- **設定のホットリロード**: `gateway.toml` を**保存した瞬間**にポリシー設定を無停止で反映する
  （プロセスは動かしたまま。~1 秒以内）。反映されるのは `default_model`・`image_max_edge`・
  `max_resident`・`request_timeout`・`idle_timeout`・`load_timeout`・`api_key`
  と動的ロードの既定（`draft_model`・`parallel`・`disable_thinking`・`max_memory_fraction`・
  `dynamic`・`start_timeout`）。動的ロード既定は**次回ロードから**有効。一方 `host`・`port`・
  `internal_base_port`・`[[models]]` はソケット bind 済み等で稼働中に変えられないため、変更を
  検知しても**適用せず「要再起動」をログ警告**する（サーバーは旧値のまま動き続ける）。編集途中の
  不正な TOML は無視して現行設定を維持する。反映内容は標準エラー（`gw log` で見えるログ）に出る。
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

## 画像入力の縮小（`image_max_edge`）

長辺がこの px を超える画像は、**上流へ渡す前にゲートウェイが縮小する**（既定 1024px、`0` で無効。
拡大はしない）。バックエンド非依存で、mlx-vlm / llama.cpp どちらでも効く。

理由は vision トークンの量がモデルによって根本的に違うから。同じ 1400px の画像で実測:

| モデル | 画像のトークン数 |
|---|---|
| gemma-4-31b | 281（**解像度によらず固定**） |
| Qwen3.6-27B（qwen3_5 系） | 1,960（**解像度に比例**） |

後者に大きい画像を渡すと、1 枚が数千トークンになる。さらに Dense モデルの prefill は MoE より
桁で遅いので、画像 1 枚で数十秒待つことになる。長辺を縮めるとトークン数はおおむね比例して減り、
応答時間もそれに追随する（縮小の効果は自分の環境で実測して決めること）。

- 対象は **data URL**（`data:image/...;base64,...`）と、トップレベル `images=[...]` の base64。
- **リモート URL（http/https）は対象外** —— 上流が自分で取得するため。ゲートウェイが代理取得すると
  SSRF やプライバシーの別問題が出るので、あえて触らない。
- 壊れた画像・未対応形式・Pillow 未導入は**素通し**（縮小が効かないだけで、リクエストは通る）。
- 動画フレームの展開（`video_max_edge`）の**後**に走るので、抽出済みフレームは既に小さく無変更。
- ホットリロード対応（保存した瞬間に反映）。

既定の 1024px は、上の表で応答が十数秒に収まる水準を選んだもの。細かい文字の OCR 等で解像度が
足りなければ上げる（1568 くらいまでは実用範囲）。逆にもっと速くしたいなら 768 まで下げられる。

## 自動更新（リリースタグに git で追従）

このリポジトリは **GitHub から clone して `make install`（editable 導入）** で動かす。
そのままだと新版が出るたび手で `git pull` が要る。`auto_update = true`（既定）なら **常駐デーモンが
実行中（起動 1 分後に初回、以降 1 時間毎）にリモートのリリースタグを確認し、新しければそのタグへ
追従して新コードで再起動する** —— 複数 PC を「起動しておくだけで最新に揃う」状態にできる。

リリース手順（配る側）:

```bash
git tag v0.38.7
git push origin v0.38.7
```

- **トリガーはリリースタグ**（`最新タグ > いま動いているソースの版`）。照会は
  `git ls-remote --tags origin`。タグの形式は `vX.Y.Z`（先頭 v は任意）で、それ以外のタグは
  無視する。現行版はクローンの `pyproject.toml`（ソース）から読む —— editable インストールで
  固定されるメタデータ版ではなく追従で上がる版を見るので、追従後は `available` が偽に戻り
  **再起動ループにならない**。
- **適用条件（安全側）**: **既定ブランチ（`main`）& git クローン & upstream 追跡 & 作業ツリーが
  クリーン**のときだけ、**タグへの fast-forward**（`git fetch --tags` + `git merge --ff-only <タグ>`
  ＋`uv sync`）を実行する。ブランチ先端ではなくタグに合わせるので、タグ後に main へ積まれた
  未リリースのコミットは配られない。
  - **機能ブランチ（`main` 以外）では自動更新しない**（`reason=not-on-default-branch`）。開発中の
    ブランチを勝手に触らないため。機能ブランチで pull してもリリース（main）は入らず版も上がらない
    ので、放置による再起動ループも起きない。
  - 未コミット変更がある PC でも**適用せず**待機する（`dirty`）—— 編集中コードを勝手に上書きしない。
  - どちらも今すぐ確認だけなら `gw update`（同じ判定で、適用可否と理由を表示する）。
- **中断しない・落とさない（quiesce + Listen ソケット引き継ぎ）**: 更新の取得（`git pull` +
  `uv sync`）は**稼働中のゲートウェイに触れずに**先に済ませる（この間も通常どおりリクエストを
  受ける）。再起動は、accept を止めて**処理中の接続が 0 になったことを確認できた**
  （quiesce 成功）ときだけ実行する。受信中・生成中の接続が残っていれば accept を再開して保留し
  （取得済みのまま 30 秒毎に再試行）、空いた瞬間に再起動する。
  **再起動の窓に届いた新規リクエストは失敗しない**: accept を止めても Listen ソケットは
  開いたままなので、新規接続は拒否されずカーネルの accept キューで待ち、`execv` 後の
  新イメージが**ソケットごと引き継いで**順に処理する（クライアントから見ると応答が
  十数秒遅れるだけ。503 や接続エラーは発生しない）。在席セッションは再起動を妨げない
  （在席は「解放を早める」だけの存在。置き去りが残っても更新は進む）。
- **再起動の仕組み**: quiesce 成功後、デーモンは配下のモデルサーバーを停止し、単一起動ロックを
  解放してから **自分自身を `python -m local_llm_server` で `execv`**（新コードに置換）。
  公開ポートの Listen ソケットは閉じずに fd を引き継ぐ（環境変数 `GW_LISTEN_FD`）。
  PID は変わらず、ログ（`gateway-<port>.log`）も継続する。モデルは遅延ロードなので、
  再起動後の最初のリクエストだけ再ロード（＋プロンプトキャッシュ再構築）のぶん遅くなる。
- **git 運用でないとき**: `.git` が無い場合は何もしない。
- **無効化**: `auto_update = false`。
- **手動**: `gw update` でいつでも今すぐ確認・適用できる（稼働中なら適用後に再起動）。

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

> **メモリに注意**: 複製インスタンスは**モデルの重みを複製**する（例: 27B・4bit を 2 本なら重みも 2 倍）。
> 並列化したい本数は搭載メモリと相談し、`max_resident`（総インスタンス数）や `max_memory_fraction` で上限を張る。
> **どちらも設定していない場合、複製は行われない**（重みのコピーが際限なく増えて OOM する事故を防ぐため）。
> 単一プロセス内で並列化できる **llama-cpp は `parallel=N` の方がメモリ効率が良い**（複製は主に mlx 系で効く）。

各モデルの起動インスタンス数は `GET /admin/status` の `models[].instances`、`gw ps` の INSTANCES 列で見える
（`max_resident` を 1 のままにすると複製は起きない＝従来どおり 1 モデル 1 プロセス共有）。

## max_resident をライブで変える

同時常駐モデル数の上限（`max_resident`）は、ゲートウェイを**再起動せずに実行中へ反映**できる。
複数モデルを並行常駐させたいとき（`max_resident` を上げる）や、逆に絞りたいときに、`gateway.toml`
を書き換えて再起動する必要はない。

- **CLI から**: `gw max <n>` を実行する（例 `gw max 2`、無制限は `gw max off`）。変更後の値は
  `gw status` の `loaded k/N` 表示に即反映される。
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

（`POST /admin/config` / `gw max` は**実行中だけの一時変更**。恒久的に変えるなら次節のとおり
`gateway.toml` を編集する —— 保存すれば同じく無停止で反映され、そちらが永続値になる。）

## ホットリロードの反映範囲

サーバーを**起動しっぱなしのまま**、`gateway.toml` を編集・保存するだけで設定を反映する。ゲートウェイは
ファイルの更新時刻を ~1 秒周期で監視し、保存を検知したら読み直して**無停止で適用**する（反映内容は
標準エラー＝ `gw log` で見えるログに出る）。運用中にモデルを落とさず方針だけ変えられる。

| 種別 | 対象 | 反映 |
|---|---|---|
| **即時反映（ポリシー）** | `default_model`, `image_max_edge`, `max_resident`, `request_timeout`, `idle_timeout`, `load_timeout`, `api_key` | 保存した瞬間に有効 |
| **次回ロードから（動的既定）** | トップレベルの `draft_model`, `parallel`, `disable_thinking`, `max_memory_fraction`, `dynamic`, `start_timeout` | 既にロード済みのモデルは次にロードし直すまで旧設定のまま |
| **要再起動（構造）** | `host`, `port`, `internal_base_port`, `[[models]]` | 稼働中は変えられない（ソケット bind 済み・内部ポート割当は起動時固定）。変更を検知しても**適用せず「要再起動」をログ警告**し、旧値のまま動き続ける |

- `max_resident` の即時反映は `POST /admin/config` と同じ挙動（**busy は止めず、超過アイドルのみ非同期
  LRU 退避**。→ [max_resident をライブで変える](#max_resident-をライブで変える)）。
- **編集途中の壊れた TOML は無視**して現行設定を維持する（保存の瞬間に構文エラーがあってもサーバーは
  落ちない）。直後に有効な内容を保存すれば、また反映される。
- 構造設定（`host`/`port` 等）を本当に変えたいときだけ、`gw restart`（またはプロセス再起動）を行う。

> ⚠ **`draft_model` / `parallel` / `disable_thinking` は名前がかぶる2つの別設定がある**。
> トップレベル（動的ロードの既定値）は上表のとおり「次回ロードから」反映されるが、
> **`[[models]]` の各エントリ内**に書いたものは配列 `[[models]]` の一部として扱われるため、
> **エントリの追加・削除はもちろん、既存エントリ内の値を1つ変えるだけでも「要再起動」側**
> になる（保存しても無視され、旧設定のまま動き続ける）。新しいモデルを事前登録して個別設定
> （`draft_model = "off"` や `disable_thinking = true` 等）を効かせたいときは、保存後に
> **必ずゲートウェイを再起動**すること。再起動を忘れると、そのモデルは登録した個別設定なしの
> **動的ロード扱い**でサイレントに動いてしまい、意図した挙動と食い違う（気づきにくい事故の元）。

## 別PCから接続する（ネットワーク公開）

既定ではゲートウェイは `127.0.0.1`（ローカルのみ）に bind するため、別PCからは繋がらない。LAN 上の
他のマシンから使うには次のようにする。

1. **ネットワークに bind する** — `gateway.toml` で `host = "0.0.0.0"`（全インターフェース）にする。
   起動ログ（`gw log`）に、リモートのクライアントが指す
   `reachable from LAN: http://<このPCのIP>:8799/v1` が表示される。
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
  `max_resident` の変更や状態監視はゲートウェイPC本体（`gw` CLI）からだけ行える。
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
| 利用終了 | `POST /admin/sessions/release` | `{"agent_id"}` | `DELETE /admin/sessions` でも可。最後の在席なら猶予後にアンロード |
| （互換）生存通知 | `POST /admin/sessions/heartbeat` | `{"agent_id"}` | 受けても何も更新しない（生存推定はしない）。既知 200 / 未知 404。404 を受けた旧クライアントは再 register する（＝再起動や在席掃除の後に登録が自己修復する） |

- **正常終了**は `release` で解放。ただし**即時ではなく 60 秒の猶予**を置く（コード定数
  `_RELEASE_LINGER_S`。設定にはしない）。タスクごとに子プロセスを起動する構成では release の
  直後に次のタスクが `register` するため、即時解放するとアンロード→再ロードを毎回繰り返す
  （巨大なモデルではロードのたびに無視できない待ちが発生する）。猶予中に `register` されれば解放は取り消される。
- **異常終了**（`release` を呼べずに落ちた）の置き去りは、そのモデルが `idle_timeout` で解放される
  ときに一緒に掃除される。**ハートビートによる生存推定はしない** — 在席したまま無応答でも
  モデルは落とされない（旧実装はこれで、生成中のクライアントの足元から巨大なモデルを
  外す事故を起こした）。
- したがって**在席は解放を早めるだけで、遅らせる力を持たない**。モデルの保持時間を延ばしたい
  ときは `idle_timeout` を調整する（在席の有無に関わらず効く唯一の時間軸）。
- 各モデルの在席数は `GET /admin/status` の `models[].sessions`、および `gw ps` の SESSIONS 列で見える。

### エージェント側の実装（任意・推奨）

通知は**任意**で、登録しなければ従来どおり `idle_timeout` でのみ解放される。早期に解放したい
エージェントだけ、起動時に `register`、終了時に `release` を仕込めばよい（ハートビートは不要）。base_url は従来どおり公開
ポートのまま、追加でこの数本を叩くだけ（チャットの送り方は変えない）。

標準ライブラリだけで完結する最小実装の例:

```python
import atexit, json, signal, urllib.request

class GatewaySession:
    """ゲートウェイに在席を登録し、終了時に早期アンロードさせるヘルパー。

        with GatewaySession(agent_id="agent-7", model="org/Model:Q4"):
            ...  # base_url=http://127.0.0.1:8799/v1 でいつものチャット
        # ブロックを抜けて猶予（60秒）の間に誰も来なければメモリが解放される
    """
    def __init__(self, *, base="http://127.0.0.1:8799", agent_id, model):
        self.base, self.agent_id, self.model = base, agent_id, model

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
        atexit.register(self.release)                       # プロセス終了時の保険
        signal.signal(signal.SIGTERM, lambda *_: self.release())  # kill されたら解放
        return self

    def release(self):
        if not getattr(self, "_released", False):
            self._released = True
            self._call("/admin/sessions/release", {"agent_id": self.agent_id})

    def __exit__(self, *exc):
        self.release()
```

`with` ブロックを抜ける／プロセスが終わる／`SIGTERM` で殺される、のいずれでも `release` が呼ばれる。
万一それも取りこぼしても、モデルが `idle_timeout` で解放される際に置き去りの在席ごと回収される（二重の安全網）。

> `agent_id` はエージェントごとに一意な文字列にする（PID やUUID等）。同一 `agent_id` で別 `model` を
> `register` し直すと、旧モデルから自動的に外れる（乗り換え。旧モデルが無人になれば解放される）。
