# gateway.toml リファレンス

サーバーはカレントディレクトリの `./gateway.toml` を 1 つの設定として読む。これがモデルカタログ
（どのモデルを・どのバックエンドで提供するか）と運用方針（同時常駐数・自動アンロード等）を決める。
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
draft_model = "auto"        # MTP（speculative decoding）の既定。各 [[models]] で上書き／"off" で無効
dynamic = true              # 未登録モデルを ID 推論で動的ロード（省略時 true。false で事前登録のみ）
disable_thinking = false    # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先

# [[models]] は任意（dynamic = true なら省略可）。個別オプション（MTP/parallel/mmproj 上書き等）が
# 要るモデルだけ事前登録し、それ以外は動的ロードに任せる、という使い分けができる。
[[models]]
model = "mlx-community/Qwen3.6-27B-4bit"   # HuggingFace のモデル ID
backend = "mlx-vlm"                        # mlx / mlx-vlm / llama-cpp
# draft_model 省略 → 上の既定 "auto" を継承

[[models]]
model = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
backend = "mlx-vlm"
# draft_model = "off"   # このモデルだけ MTP を無効化
```

| キー | 既定 | 説明 |
|---|---|---|
| `host` | `127.0.0.1` | 公開ホスト |
| `port` | `8799` | 公開ポート（クライアントの `base_url` はここ） |
| `max_resident` | 無制限 | 同時に常駐させるモデル数のハード上限。超過は LRU で退避 |
| `load_timeout` | `300` | 全枠が処理中のとき空きを待つ最大秒数（超過で 503） |
| `idle_timeout` | `1200` | この秒数使われないモデルを自動アンロード（`0` で無効） |
| `session_ttl` | `90` | 在席エージェントのハートビート猶予秒数。途絶で無人扱い（`0` で無効）。→ [在席ベースの即時アンロード](#在席ベースの即時アンロード) |
| `internal_base_port` | `9001` | 内部モデルサーバーの割当開始ポート |
| `default_model` | なし | `model` 省略リクエスト時に使うモデル |
| `draft_model` | なし | MTP の全体既定（`"auto"` で自動選択／`"off"` で無効）。→ [mtp.md](mtp.md) |
| `dynamic` | `true` | 未登録モデルを ID 推論で動的ロードする。`false` で事前登録のみ（旧挙動） |
| `disable_thinking` | `false` | 動的ロード時の既定。事前登録モデルは各 `[[models]]` の値が優先 |

`[[models]]` は 1 モデル 1 エントリ。`model`（HuggingFace ID）と `backend`（`mlx` / `mlx-vlm` /
`llama-cpp`）が必須。各エントリで `draft_model` を上書きできる。`dynamic = true` なら `[[models]]` は
省略可（全て動的ロード）。

## 振る舞い

- **遅延起動**: 各モデルは**初回リクエスト時に起動**し、2 回目以降は常駐して即応答する。
- **動的ロード（`dynamic = true`）**: `[[models]]` に無いモデルもリクエストされた時点で起動・管理する。
  バックエンドは ID から推論（`gguf`→llama-cpp、`mlx`→mlx-vlm、他→OS 既定。→ [docs/llama-cpp.md](llama-cpp.md)）。
  ロードされると一覧（`/v1/models`・ダッシュボード）に現れ、アンロードされると消える。すでにロード済みの
  モデルが再指定されたら**相乗り**（共有）する。個別オプション（MTP/parallel/mmproj 上書き）は付かない
  ので、必要なモデルだけ `[[models]]` に事前登録する。llama-cpp の repo-id は事前に取得済みである必要が
  あり（未取得は 400）、mlx は HF から自動DLされる。
- **LRU 退避**: 常駐数が `max_resident` を超えると、最も使われていないモデルから停止する。
  全枠が処理中なら空くまで待つ（OOM 回避。`load_timeout` で打ち切り→ 503）。
- **アイドル自動解放**: `idle_timeout` 秒使われないモデルをアンロードしてメモリを返す。
- **在席ベースの即時解放**: エージェントが利用終了を通知すると、そのモデルを使う在席が 0 になった
  瞬間（＝他に同じモデルへ接続しているエージェントが居ない）に、処理中でなければ `idle_timeout` を
  待たず即アンロードする。→ [在席ベースの即時アンロード](#在席ベースの即時アンロード)
- **1 公開ポートで集約**: 例 `http://127.0.0.1:8799/v1`。クライアントは公開ポートに繋ぎ、
  リクエストの `model` で振り分けられる（クライアントはサーバーを起動しない）。

MTP（speculative decoding）による高速化は [mtp.md](mtp.md) を参照。

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
