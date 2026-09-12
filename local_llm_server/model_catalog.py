"""Model metadata, Hugging Face cache discovery, and MTP pairing.

This module intentionally contains no process lifecycle code.  ``server`` re-exports
its public names to preserve the existing API.
"""

from __future__ import annotations

import glob
import json
import os
import time

from .backend_core import infer_backend


# 思考チャネルの開始/終了マーカー（mlx-vlm へ env で渡す）。mlx-vlm は env で渡された 1 対を
# 最優先で試し、続けて内蔵既定（<|channel>thought / <think> / <|START_THINKING|>）を試す。
# 既定は gemma-4-A4B 系の形式。内蔵既定のどれとも違う形式のモデルだけここに書く。
# 判定はモデル ID の部分一致（小文字化）——ローカルパス登録でも効かせるため。
_THINKING_MARKERS = (
    # Inkling（Thinking Machines）: 思考は
    #   <|content_thinking|>…<|end_message|><|message_model|><|content_text|>本文<|end_message|>
    # の形で出る。この形式は mlx-vlm の内蔵既定に無く、既定のままだと思考が丸ごと
    # content に漏れる（実測確認済み）。
    # 終端に <|end_message|> 単体を使ってはいけない: 本文の終端でもあるため、思考 OFF
    # （reasoning_effort="none"）のときに**本文全体が思考と誤判定**され content が空になる。
    # 思考ブロックの直後にだけ現れる 2 トークン列を終端にすると両方で正しく割れる。
    ("inkling", ("<|content_thinking|>", "<|end_message|><|message_model|>")),
)
_DEFAULT_THINKING_MARKERS = ("<|channel>thought", "<channel|>")


def thinking_markers(model: str) -> tuple[str, str]:
    """モデル ID から思考チャネルのマーカー対を引く（未収載は gemma-4 形式の既定）。"""
    lowered = model.lower()
    for needle, markers in _THINKING_MARKERS:
        if needle in lowered:
            return markers
    return _DEFAULT_THINKING_MARKERS


# 本体（target）→ 対応する MTP ドラフター（assistant）の内蔵対応表。
# mlx-community のペアで、いずれも実機で検証済み。draft_model = "auto" のときに
# 本体名から対応ドラフターを引く（明示指定すればここを介さない）。未収載のモデルを
# auto にした場合はエラーで明示指定を促す（MTP 自体は非収載でも明示すれば使える）。
# Gemma 4 が中心だが、Qwen3.6 も MTP 方式で動作確認済み（mlx_vlm --draft-kind mtp）。
MTP_DRAFTERS = {
    "mlx-community/gemma-4-E4B-it-qat-4bit": "mlx-community/gemma-4-E4B-it-qat-assistant-bf16",
    "mlx-community/gemma-4-12B-it-qat-4bit": "mlx-community/gemma-4-12B-it-qat-assistant-4bit",
    "mlx-community/gemma-4-26B-A4B-it-qat-4bit": "mlx-community/gemma-4-26B-A4B-it-qat-assistant-nvfp4",
    "mlx-community/gemma-4-31B-it-qat-4bit": "mlx-community/gemma-4-31B-it-qat-assistant-bf16",
    # 非QAT 8bit（26B-A4B）。ドラフターは非QAT の assistant-bf16。
    "mlx-community/gemma-4-26b-a4b-it-8bit": "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
    # Qwen3.6-27B（既定モデル）の MTP ドラフター。
    "mlx-community/Qwen3.6-27B-4bit": "mlx-community/Qwen3.6-27B-MTP-4bit",
    # 自作 ToPo-ToPo 版の Qwen3.6-27B（既定運用）。ドラフターは Qwen3.8-27B と同じ手順で、
    # 公式 bf16 チェックポイント内蔵の mtp.* を切り出した自作 MTP ヘッド
    # （量子化後のリポからは mtp が落ちるので必ず公式 bf16 から切り出す）。量子化違いは
    # 同一ドラフターで共用できる。mlx-community/Qwen3.6-27B-MTP-4bit も使えるが、
    # 実測では bf16 の方が採択率が高い（コード生成で 95.2%）。
    "ToPo-ToPo/Qwen3.6-27B-mlx-4bit": "ToPo-ToPo/Qwen3.6-27B-MTP-bf16",
    "ToPo-ToPo/Qwen3.6-27B-mlx-8bit": "ToPo-ToPo/Qwen3.6-27B-MTP-bf16",
    "ToPo-ToPo/Qwen3.6-27B-mlx-bf16": "ToPo-ToPo/Qwen3.6-27B-MTP-bf16",
    # Qwen3.8-27B（自作 ToPo-ToPo 版）。ドラフターは公式 bf16 チェックポイント内蔵の mtp.* を
    # 切り出した自作 MTP ヘッド（量子化後のリポからは mtp が落ちるので必ず公式 bf16 から切り出す）。
    # 量子化違いは同一ドラフターで共用できる（greedy では bf16 と 4bit で採択が一致する）。
    "ToPo-ToPo/Qwen3.8-27B-mlx-4bit": "ToPo-ToPo/Qwen3.8-27B-MTP-bf16",
    "ToPo-ToPo/Qwen3.8-27B-mlx-8bit": "ToPo-ToPo/Qwen3.8-27B-MTP-bf16",
    "ToPo-ToPo/Qwen3.8-27B-mlx-bf16": "ToPo-ToPo/Qwen3.8-27B-MTP-bf16",
    # Qwen3.8-Flash-Next（qwen4_exp。自作 ToPo-ToPo 版）。ドラフターは公式 bf16 内蔵の mtp.* を
    # 切り出した自作 MTP ヘッド（block_size=2 を config に焼き込み済み。実測でこれが最速:
    # 25.95 → 35.99 tok/s の 1.39 倍・採択率 94.1%）。量子化違いは同一ドラフターで共用。
    # 実行には qwen4_exp_mtp を持つ mlx-vlm >= 0.7.0 が必要（pyproject のロックで担保）。
    "ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit": "ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16",
    "ToPo-ToPo/Qwen3.8-Flash-Next-mlx-8bit": "ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16",
    "ToPo-ToPo/Qwen3.8-Flash-Next-mlx-bf16": "ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16",
    # 自作 ToPo-ToPo 版 gemma 4。各 model card が推奨する Google 公式 MTP ドラフター
    # google/gemma-4-<size>-it-assistant を使う（mlx-vlm で変換不要・サイズ固有で量子化に依らず共通。
    # mlx-vlm >= 0.6.3 が必要）。
    "ToPo-ToPo/gemma-4-31b-it-mlx-4bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-mlx-8bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-mlx-bf16": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-31b-it-qat-mlx-4bit": "google/gemma-4-31B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-4bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-8bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-mlx-bf16": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-26B-A4B-it-qat-mlx-4bit": "google/gemma-4-26B-A4B-it-assistant",
    "ToPo-ToPo/gemma-4-E4B-it-qat-mlx-4bit": "google/gemma-4-E4B-it-assistant",
    "ToPo-ToPo/gemma-4-E2B-it-qat-mlx-4bit": "google/gemma-4-E2B-it-assistant",
}


# 対応表（MTP_DRAFTERS）には載せないが、ドラフターであることが分かっている repo。
# 対応表に載せると draft_model="auto" が引いてしまうので載せられない——けれど発見一覧には
# 出したくない、というものをここに書く（例: 上流バグで実用不能なため gateway.toml では
# off にしている DeepSeek-V4-Flash の MTP ヘッド）。
_EXTRA_DRAFTER_REPOS = frozenset(
    {
        "ToPo-ToPo/DeepSeek-V4-Flash-MTP-bf16",
    }
)

# MTP ドラフター（speculative decoding 用の補助モデル）の repo-id 集合。これ自体は
# 単体のチャットモデルとして使うものではないので、発見一覧（discover_cached_models）には
# 「使えるモデル」として出さない。`org/repo:selector` 形式のドラフターは repo 部分で判定する。
_DRAFTER_REPOS = (
    frozenset([v.split(":", 1)[0] for v in MTP_DRAFTERS.values()])
    | _EXTRA_DRAFTER_REPOS
)


def resolve_drafter(model: str, draft_model: str | None) -> str | None:
    """draft_model を解決する。

    - None / 空 … ドラフター無し（speculative decodingを使わない）。
    - "auto"   … 本体名 model から対応する MTP ドラフター（Gemma 4 / Qwen3.6）を
                 内蔵表で引く。未収載なら ValueError（HF id を明示するよう促す）。
    - それ以外 … その値（ドラフターの HF id / パス）をそのまま使う。
    """
    if not draft_model:
        return None
    if draft_model != "auto":
        return draft_model
    drafter = MTP_DRAFTERS.get(model)
    if drafter is None:
        known = ", ".join(sorted(MTP_DRAFTERS))
        raise ValueError(
            f'draft_model="auto" に対応するドラフターが見つかりません（model={model!r}）。'
            f" 自動対応している本体: {known}。"
            " 他のモデルでは draft_model にドラフターの HF id を明示してください。"
        )
    return drafter


def _hf_hub_cache() -> str:
    """HuggingFace Hub のキャッシュ（models--org--name/snapshots/...）ルートを返す。"""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def resolve_gguf(model: str, *, cache_root: str | None = None) -> str:
    """llama.cpp の `model`（HF repo-id）を DL 済みキャッシュの実 GGUF パスに解決する。

    `model` は必ず **HF repo-id（`org/repo[:セレクタ]`）**で指定する（実ファイルパスは非対応）。
    HF キャッシュ（`hf download` 済み）から該当 GGUF を探して返す。`-hf` の自動DLには依存しない
    （トークン不要・401 回避）。次の場合はいずれも ValueError（取得方法を案内）:

    - repo-id 形式でない（実パス等）／キャッシュに無い／該当 GGUF が無い。
    - `org/repo` に GGUF が複数あって 1 つに定まらない（`:セレクタ` で絞る）。

    `org/repo:selector` はファイル名の一部（量子化名や `F16-MTP` 等）。セレクタ無しのときは mmproj と
    MTP ヘッドを除いた「本体」GGUF を選ぶ。
    """
    spec = model.strip()
    repo, _sep, selector = spec.partition(":")
    if looks_like_local_path(repo) or repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"model は HF repo-id（org/repo[:量子化名]）で指定してください（実パス非対応）: {model!r}"
        )
    org, name = repo.split("/", 1)
    cache_dir = os.path.join(
        cache_root or _hf_hub_cache(), f"models--{org}--{name}", "snapshots"
    )
    if not os.path.isdir(cache_dir):
        raise ValueError(
            f"'{repo}' がローカルキャッシュにありません。先に取得してください: "
            f"hf download {repo} <ファイル名.gguf>"
        )
    ggufs: list[str] = []
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.lower().endswith(".gguf"):
                ggufs.append(os.path.join(root, f))
    if selector:
        matched = [g for g in ggufs if selector.lower() in os.path.basename(g).lower()]
    else:
        # 本体＝mmproj でも MTP ヘッドでもないもの
        matched = [
            g
            for g in ggufs
            if "mmproj" not in os.path.basename(g).lower()
            and "mtp" not in os.path.basename(g).lower()
        ]
    # 複数スナップショットが同じ blob を指すことがあるので実体で重複排除する。ただし返すのは
    # スナップショット側のパス（実ファイル名が残り、隣の mmproj を検出できる）。
    by_blob: dict[str, str] = {}
    for g in sorted(matched):
        by_blob.setdefault(os.path.realpath(g), g)
    pool = sorted(by_blob.values())
    if not pool:
        hint = f"（セレクタ '{selector}' に一致なし）" if selector else ""
        raise ValueError(
            f"'{model}' に該当する GGUF がキャッシュにありません{hint}。"
            f"hf download {repo} <ファイル名.gguf> で取得してください。"
        )
    if len(pool) > 1:
        names = sorted(os.path.basename(g) for g in pool)
        raise ValueError(
            f"'{model}' に複数の GGUF が該当します {names}。"
            f"'{repo}:<量子化名など>' でファイルを 1 つに絞ってください。"
        )
    return pool[0]


def _snapshot_weights_complete(snap: str) -> bool:
    """スナップショットの重みが実体（シンボリックリンク先）まで揃っているか。

    `model.safetensors.index.json` があるときは **weight_map が要求する全シャード**を
    確認する。1 つでも欠けていれば未完了とみなす（歯抜けのまま「重みが 1 つはある」で
    通すと、ロードして初めて落ちる）。index を持たないリポジトリ（単一 safetensors や
    whisper 系の *.npz）は、重みが 1 つ以上あることをもって完了とする。
    """
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as fh:
                shards = set(json.load(fh).get("weight_map", {}).values())
        except Exception:  # noqa: BLE001 壊れた index は「index 無し」として扱う
            shards = set()
        if shards:
            return all(
                os.path.exists(os.path.realpath(os.path.join(snap, s))) for s in shards
            )
    return any(
        os.path.exists(os.path.realpath(f))
        for pattern in ("*.safetensors", "*.npz")
        for f in glob.glob(os.path.join(snap, pattern))
    )


def _blocking_incomplete(blobs_dir: str) -> list[str]:
    """「取得途中」と判断すべき `*.incomplete` だけを返す。

    hf は `<sha>.<乱数>.incomplete` に書いてから `<sha>` へ確定させるが、**中断して
    再試行が成功しても前回の .incomplete が消えずに残ることがある**。残骸の有無だけで
    判定すると、完全に取得できているモデルが永久に「キャッシュにありません」になる
    （実際に起きた）。よって **対応する確定 blob が無いものだけ**を取得途中とみなす。
    """
    out = []
    for f in glob.glob(os.path.join(blobs_dir, "*.incomplete")):
        sha = os.path.basename(f).split(".", 1)[0]
        if not os.path.exists(os.path.join(blobs_dir, sha)):
            out.append(f)
    return out


def ensure_cached(
    repo: str, *, what: str = "モデル", cache_root: str | None = None
) -> str:
    """mlx 系（mlx / mlx-vlm）の HF repo-id がローカルキャッシュに**完全に**存在するか検証する。

    本サーバーは自動ダウンロードを行わない（事前に `hf download` 済みであることを要求する）。
    起動前にここで存在を確認し、無ければ取得方法を案内して ValueError を送出する
    （llama-cpp の resolve_gguf と同じ「事前 DL 必須」ポリシー）。返り値は確認したスナップショット
    ディレクトリ（実ファイルパス指定時はそのパス）。

    次のいずれも「未取得」とみなしてエラーにする:
      - スナップショットが存在しない。
      - ダウンロードが途中（確定 blob の無い *.incomplete が blobs/ に残っている）。
        確定 blob が既にある .incomplete は**再試行が成功した後の残骸**なので無視する。
      - 重み（*.safetensors / *.npz）の実体がキャッシュに揃っていない。index がある
        場合は全シャードを要求する。
    """
    spec = repo.strip()
    # 実ファイル/ディレクトリパス指定（repo-id ではない）はそのパスの存在のみ確認する。
    if looks_like_local_path(spec):
        path = os.path.expanduser(spec)
        if not os.path.exists(path):
            raise ValueError(f"{what}のパスが見つかりません: {repo!r}")
        return path
    if spec.count("/") != 1 or not all(spec.split("/")):
        raise ValueError(f"{what}は HF repo-id（org/repo）で指定してください: {repo!r}")
    org, name = spec.split("/", 1)
    base = os.path.join(cache_root or _hf_hub_cache(), f"models--{org}--{name}")
    snap_root = os.path.join(base, "snapshots")
    # ダウンロードが途中なら「未取得」と同じ扱い（DL 停滞の主症状）。ただし確定 blob が
    # 既にある .incomplete は再試行成功後の残骸なので数えない（_blocking_incomplete）。
    incomplete = _blocking_incomplete(os.path.join(base, "blobs"))
    snaps = (
        sorted(glob.glob(os.path.join(snap_root, "*")))
        if os.path.isdir(snap_root)
        else []
    )
    if not snaps or incomplete:
        raise ValueError(
            f"{what} '{repo}' がローカルキャッシュにありません（自動ダウンロードは無効）。"
            f" 先に取得してください: hf download {spec}"
        )
    # 重みの実体（シンボリックリンク先まで）が揃っているスナップショットを選ぶ。
    # whisper 系の mlx リポジトリは *.npz で重みを持つものがあるため両方を許容する。
    complete = [s for s in snaps if _snapshot_weights_complete(s)]
    if not complete:
        raise ValueError(
            f"{what} '{repo}' の重み（*.safetensors / *.npz）がキャッシュに揃っていません。"
            f" 取得し直してください: hf download {spec}"
        )
    return complete[0]


def mtp_status(
    model: str, drafter: str | None = None, *, cache_root: str | None = None
) -> str | None:
    """model の MTP（Multi-Token Prediction による高速化）の利用可否を返す。

    使うドラフターが決まるかと、それがローカルに在るかで判定する:

    - "ready"     … ドラフターが手元にある。そのまま MTP が効く。
    - "available" … ドラフターは決まるが未取得。`hf download <drafter>` で有効化できる。
    - None        … MTP なし（明示指定も対応表の項目も無い）。

    `drafter` は gateway.toml で **明示指定された（＝解決済みの）** draft_model。指定があれば
    対応表より優先する——対応表は `draft_model="auto"` 用の内蔵ペア表でしかないので、これを
    見ないと「gateway.toml で明示指定してあるのに一覧では MTP 非対応に見える」ことになる。
    無効化（off/none/""）は呼び出し側で解決済み＝None で渡ってくる前提。

    一覧表示（discover_cached_models / merge_status / TUI）から呼ぶ。ドラフターの有無確認に
    ensure_cached を使う（自動 DL はしない方針と一貫）。
    """
    drafter = drafter or MTP_DRAFTERS.get(model)
    if not drafter:
        return None
    if looks_like_local_path(drafter):
        # ローカルパス指定のドラフター（HF キャッシュではない実ディレクトリ）は重みを直接見る。
        return (
            "ready"
            if _dir_weight_bytes(os.path.expanduser(drafter.strip()))
            else "available"
        )
    try:
        ensure_cached(drafter, what="ドラフター", cache_root=cache_root)
        return "ready"
    except ValueError:
        return "available"


_DISCOVER_CACHE: dict = {"t": -1e9, "v": []}

# チャット/生成に使わない（埋め込み・STT・分類・エンコーダ）モデルタイプ。発見一覧から除く。
_NON_CHAT_MODEL_TYPES = frozenset(
    {
        "bert",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "deberta",
        "deberta-v2",
        "mpnet",
        "camembert",
        "electra",
        "albert",
        "nomic_bert",
        "whisper",
        "wav2vec2",
        "clip",
        "siglip",
        "t5",
        "mt5",
    }
)


def _is_generative_repo(snap_root: str) -> bool:
    """スナップショット内の config.json を見て、生成（チャット）系モデルかを判定する。

    埋め込み（e5/MiniLM 等）・STT（whisper）・分類器など非チャットのモデルを発見一覧から
    除くためのフィルタ。config.json が読めなければ True（取りこぼしを避ける＝控えめに除外）。
    """
    cfg_path = None
    for sroot, _d, files in os.walk(snap_root):
        if "config.json" in files:
            cfg_path = os.path.join(sroot, "config.json")
            break
    if not cfg_path:
        return True
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return True
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type in _NON_CHAT_MODEL_TYPES:
        return False
    archs = cfg.get("architectures") or []
    if not archs:
        return True  # アーキ不明なら除外しない
    return any(a.endswith(("ForCausalLM", "ForConditionalGeneration")) for a in archs)


def discover_cached_models(
    ttl: float = 10.0, *, cache_root: str | None = None
) -> list[dict]:
    """HF キャッシュにある**実行可能なチャットモデル**を列挙する（発見用）。

    LM Studio / Ollama のように「いま手元で動かせる候補」をクライアントに見せるための一覧。
    ロード済みかどうかに関わらず、ダウンロード済みモデルを `{"id", "backend", "mtp"}` のリストで
    返す（`mtp` は "ready" / "available" / None＝mtp_status）。MTP ドラフター自体は単体で使う
    モデルではないので一覧からは除外する（_DRAFTER_REPOS）。判定はヒューリスティック:

    - GGUF を含む repo → llama-cpp。本体（mmproj / MTP ヘッドを除く）が 1 つなら `org/repo`、
      複数あれば `org/repo:<ファイル名>` を量子化ごとに列挙（そのままロードできる形）。
    - `config.json` ＋ 重み（`.safetensors` / `.npz`）を持ち、生成系アーキ（`*ForCausalLM` /
      `*ForConditionalGeneration`）の repo → mlx 系（mlx-vlm で動的ロード）。埋め込み・STT・
      分類などの非チャットモデルは除外する（`_is_generative_repo`）。

    `ttl` 秒は結果をキャッシュする（`/admin/status` の毎秒ポーリングで毎回走査しないため）。
    """
    now = time.monotonic()
    if now - _DISCOVER_CACHE["t"] < ttl:
        return list(_DISCOVER_CACHE["v"])
    root = cache_root or _hf_hub_cache()
    out: list[dict] = []
    seen: set[str] = set()
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("models--") or entry.count("--") < 2:
                continue
            _, org, name = entry.split("--", 2)
            repo = f"{org}/{name}"
            # MTP ドラフターは「使えるモデル」ではないので一覧に出さない。
            if repo in _DRAFTER_REPOS:
                continue
            snap_root = os.path.join(root, entry, "snapshots")
            if not os.path.isdir(snap_root):
                continue
            files = [f for _r, _d, fs in os.walk(snap_root) for f in fs]
            ggufs = [f for f in files if f.lower().endswith(".gguf")]
            if ggufs:
                bodies = [
                    f
                    for f in ggufs
                    if "mmproj" not in f.lower() and "mtp" not in f.lower()
                ]
                if not bodies:
                    continue  # mmproj / MTP ヘッドだけの repo は本体ではない
                if len(bodies) == 1:
                    cands = [repo]
                else:
                    cands = [f"{repo}:{os.path.splitext(f)[0]}" for f in sorted(bodies)]
                backend = "llama-cpp"
            elif (
                "config.json" in files
                and any(f.endswith((".safetensors", ".npz")) for f in files)
                # 生成系（チャット）または STT（whisper）を対象にする。埋め込み・分類器などの
                # 非チャット・非STT モデルは除外する（_is_generative_repo）。
                and (_is_generative_repo(snap_root) or infer_backend(repo) == "whisper")
            ):
                cands = [repo]
                backend = infer_backend(
                    repo
                )  # whisper → STT、mlx → mlx-vlm、他は OS 既定
            else:
                continue
            # MTP（高速化）の利用可否を本体ごとに付与する（ドラフターが揃っていれば "ready"）。
            mtp = mtp_status(repo, cache_root=root)
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    out.append({"id": c, "backend": backend, "mtp": mtp})
    _DISCOVER_CACHE["t"] = now
    _DISCOVER_CACHE["v"] = out
    return list(out)


def looks_like_local_path(spec: str) -> bool:
    """model / draft_model の指定が HF repo-id ではなくローカルパスか。

    POSIX の絶対・相対・チルダに加えて **Windows のドライブレターと逆スラッシュ**も見る
    （`C:\\models\\x` / `C:/models/x` / `\\\\server\\share`）。ここを POSIX 限定にしていたため、
    Windows ではローカル変換物の登録がすべて repo-id 扱いになり、メモリ見積もりが
    None（＝ガード無効）に落ちていた。
    """
    spec = spec.strip()
    if spec.startswith(("/", "./", "../", "~", "\\")):
        return True
    # C:\... / C:/...（ドライブレター）
    return len(spec) >= 3 and spec[1] == ":" and spec[2] in ("\\", "/")


def _dir_weight_bytes(directory: str) -> int:
    """ディレクトリ直下の重みファイル（*.safetensors / *.npz）の合計バイト数。

    ローカル変換物（HF キャッシュではない実ディレクトリ）の占有見積もりに使う。
    tokenizer.json 等の小物は数えない（下限寄りの見積もりという方針は repo-id 側と同じ）。
    """
    total = 0
    if not os.path.isdir(directory):
        return 0
    for pattern in ("*.safetensors", "*.npz"):
        for path in glob.glob(os.path.join(directory, pattern)):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total
