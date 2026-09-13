"""Parsing and validation for ``gateway.toml``.

Runtime orchestration remains in :mod:`local_llm_server.daemon`; keeping parsing
here makes configuration policy independently testable and easier to extend.
"""

from __future__ import annotations

import ipaddress
import math
import sys
import tomllib
from dataclasses import dataclass, field

from .backend_core import (
    BACKEND_SPECS,
    ServerConfig,
    backend_spec,
)
from .constants import BACKENDS
from .model_catalog import resolve_drafter

_DRAFT_OFF = ("", "off", "none")


def _strict_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _strict_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _port(value, name: str) -> int:
    result = _strict_int(value, name)
    if not (1 <= result <= 65535):
        raise ValueError(f"{name} must be between 1 and 65535")
    return result


@dataclass
class GatewayConfig:
    host: str
    port: int
    max_resident: int | None
    default_model: str | None
    models: list[ServerConfig] = field(default_factory=list)
    idle_timeout: float | None = (
        1200.0  # 秒。これだけ使われないモデルを自動アンロード（既定 1200=20分。None/0 で無効）
    )
    load_timeout: float = (
        300.0  # 秒。全枠処理中のとき、空くのを待つ最大時間（超過で 503）
    )
    start_timeout: float = 120.0  # 秒。モデルサーバー1つの起動完了（ready）を待つ最大時間（巨大モデルは延ばす）
    request_timeout: float | None = (
        600.0  # 秒。上流との通信が無応答のとき打ち切る（0 で無制限）。ハングした／
    )
    # 沈黙した上流が inflight を握ったまま枠を塞ぎ続けるのを防ぐ保険。トークンが
    # 流れている限り切れないので、長時間ストリーミング生成は妨げない（既定 600=10分）
    dynamic: bool = (
        True  # 未登録モデルを ID 推論で動的ロードする（false で事前登録のみ）
    )
    disable_thinking: bool = (
        False  # 動的ロード時の既定（思考抑制）。事前登録は各 [[models]] が優先
    )
    stream_tool_calls: bool = False  # ツール呼び出しの生成中トークンを流す（mlx-vlm）。各 [[models]] で上書き可。→ ServerConfig.stream_tool_calls
    draft_model: str | None = (
        None  # 動的ロード時の MTP 既定。None で mlx-vlm は "auto"（対応表から自動）。"off" で無効
    )
    parallel: int | None = (
        None  # 動的ロード時の並列スロット既定（llama-cpp のみ。他は無視）
    )
    max_memory_fraction: float | None = (
        None  # 常駐モデルの推定占有量の合計を総RAMのこの割合に制限（None で無効）
    )
    internal_base_port: int = (
        9001  # 内部サーバーの割当開始ポート（動的モデルもこの続きから割り当て）
    )
    api_key: str | None = (
        None  # ネットワーク公開時の API キー（None/空 で認証なし）。chat と在席セッションに要求
    )
    allow_unauthenticated_remote: bool = (
        False  # 明示的に危険なLAN無認証公開を許す場合のみtrue
    )
    tray: bool = True  # 稼働中メニューバーにアイコンを出す（macOS のみ。false で非表示 → tray.py）
    # CLI-only provenance. It is not a TOML setting and is excluded from equality/repr.
    _config_dir: str | None = field(default=None, repr=False, compare=False)
    # --- llama.cpp（llama-server）バイナリの自動導入。[llama_cpp] テーブルで設定 ---
    # 導入方法は選ばせない（管理dirの導入済みを再利用→無ければプリビルト自動DL の一本道）。
    llama_accel: str = "auto"  # auto=検出（GPU なら vulkan、mac は metal、無ければ cpu）/ cuda / vulkan / metal / cpu
    llama_build: str | None = (
        None  # ビルド番号の固定（例 "b9946"）。省略で最新を取得し導入済みを使い続ける
    )
    # vLLM / SGLang（Linux/NVIDIA・Windows は WSL2）も一本道: 現在の環境に有ればそれを、
    # 無ければ隔離 venv へ自動導入（backend='vllm'/'sglang' のモデルを使ったときだけ動く）。
    # --- 動画入力: ゲートウェイが video_url をフレーム画像列へ展開して上流へ渡す ---
    video_frames: int = 8  # 1 本の動画から等間隔で抜くフレーム数
    video_max_edge: int = 768  # 各フレームの縮小サイズ（長辺ピクセル）
    image_max_edge: int = 1024  # 静止画の長辺上限。解像度上限の無い VLM の vision トークン爆発を防ぐ（0 で無効）
    max_request_workers: int = (
        32  # accept 後に同時処理するHTTP接続数（Slowloris/メモリ枯渇対策）
    )
    max_media_workers: int = 2  # 同時に動画をDL/ffmpeg展開する数
    # --- 繰り返しループ抑制: mlx 系バックエンド（mlx / mlx-vlm）宛の chat リクエストに
    #     repetition_penalty を既定注入する（mlx-lm/mlx-vlm 拡張パラメータ）。低温・量子化の
    #     ローカル LLM が「同じ内容を繰り返して終わらない」degeneration の緩和。llama-cpp は
    #     パラメータ名が異なる（repeat_penalty）ので対象外＝mlx 系だけに付ける（ユーザー方針）。
    #     クライアントが自分で repetition_penalty を指定していれば尊重する（上書きしない）。---
    repetition_penalty: float | None = (
        1.1  # 既定 1.1（llama.cpp 既定と同値の穏当な値）。
    )
    # 0 / false / "off" で無効化（注入しない）＝設定しない選択
    repetition_context_size: int | None = (
        None  # 併せて注入する参照窓（mlx 既定 20。研究推奨 64）。
    )
    # None なら注入しない（repetition_penalty だけ付ける）
    # 構造化リクエスト（`tools`＝native ツールコール / `response_format`＝構造化出力）には
    # repetition_penalty を注入しないオプション。既定 false（＝従来どおり全 chat に注入）。
    # true にすると、必須の繰り返し記号（JSON 構文・フィールド名）を減点しうる structured 生成を
    # 保護できる（実測では 1.1 で実害は無いが、保険として明示的に切れるようにする）。
    repetition_penalty_skip_structured: bool = False


_TOP_LEVEL_CONFIG_KEYS = {
    "host",
    "port",
    "max_resident",
    "default_model",
    "models",
    "idle_timeout",
    "load_timeout",
    "start_timeout",
    "request_timeout",
    "dynamic",
    "disable_thinking",
    "stream_tool_calls",
    "draft_model",
    "parallel",
    "max_memory_fraction",
    "internal_base_port",
    "api_key",
    "allow_unauthenticated_remote",
    "auto_update",
    "tray",
    "llama_cpp",
    "video_frames",
    "video_max_edge",
    "image_max_edge",
    "max_request_workers",
    "max_media_workers",
    "repetition_penalty",
    "repetition_context_size",
    "repetition_penalty_skip_structured",
    "session_ttl",  # 旧設定互換: 個別に警告して無視する
}
_MODEL_CONFIG_KEYS = {
    "model",
    "backend",
    "parallel",
    "disable_thinking",
    "stream_tool_calls",
    "draft_model",
    "extra_args",
}
_LLAMA_CPP_CONFIG_KEYS = {"accel", "pin"}


def _reject_unknown_keys(data: dict, allowed: set[str], scope: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} setting(s): {', '.join(unknown)}")


def _resolve_model_draft(
    entry: dict, default_draft, backend: str, model: str
) -> str | None:
    """1 モデルの MTP ドラフターを解決する（個別指定 > ゲートウェイ既定）。

    - 個別の `draft_model` があればそれを、無ければゲートウェイ既定を継承する。
    - `""` / `"off"` / `"none"` で無効化（継承既定の打ち消しに使える）。
    - mlx-vlm のみ MTP が効くので、その場合だけ `resolve_drafter` で解決する。`"auto"` が
      本体名から引けないときは**静かに MTP 無しにする**（動的ロードの `_dynamic_draft` と
      同じ扱い）。`"auto"` は「対応表に在れば使う」であって MTP を使うという宣言ではなく、
      そもそも MTP が存在しないモデルのほうが多数派なので、警告するとただのノイズになる。
      ここで例外にするのはもっと悪く、トップレベルの `draft_model = "auto"` を継承した
      未収載モデルが 1 つ在るだけで**ゲートウェイ全体の設定読み込みが失敗する**——
      MTP が効かないだけで済む話が、全モデルを起動不能にしてしまう。
    - 警告を出すのは「使うと宣言しているのに使えない」ときだけ（ドラフターの HF id が
      決まっているのに未取得＝build_command 側、または MTP 非対応バックエンドへの明示指定）。
    - 他バックエンドでは無視するが、**個別に明示**されていた場合だけ「無視される」旨を警告する。
    """
    has_own = "draft_model" in entry
    raw = entry.get("draft_model", default_draft)
    if raw is not None and not isinstance(raw, str):
        raise ValueError(f"draft_model must be a string (model {model})")
    if isinstance(raw, str) and raw.strip().lower() in _DRAFT_OFF:
        raw = None
    if not raw:
        return None
    style = backend_spec(backend).draft_style
    if style == "mtp":
        try:
            return resolve_drafter(model, raw)
        except ValueError:
            return None  # "auto" が対応表に無い → MTP なしで普通に登録する
    if style == "gguf":
        # speculative decoding のドラフト GGUF のパス/HF id を直接指定する方式（-md）。
        # MTP ヘッドのファイル名は build_command 側で検出して --spec-type draft-mtp を付ける。
        # "auto" の自動解決表は無いので、明示パス以外は無効扱い。
        if raw == "auto":
            return None
        return raw
    if has_own:
        mtp_capable = ", ".join(
            name for name, spec in BACKEND_SPECS.items() if spec.draft_style == "mtp"
        )
        print(
            f"Warning: draft_model (MTP) is ignored for backend '{backend}' "
            f"(MTP needs {mtp_capable}); model {model}",
            file=sys.stderr,
        )
    return None


def _parse_repetition_settings(data: dict):
    """繰り返しループ抑制の既定注入（mlx 系のみ）の設定を読む。

    既定 1.1。0 / false / "off" / "none" で無効化（＝注入しない＝「設定しない」選択）。
    < 1.0 は繰り返しを助長するので拒否する（1.0 は中立＝無効相当だが受け付ける）。
    戻り値: (repetition_penalty, repetition_context_size, skip_structured)。
    """
    rp_raw = data.get("repetition_penalty", 1.1)
    if (
        rp_raw is None
        or rp_raw is False
        or (
            isinstance(rp_raw, str)
            and rp_raw.strip().lower() in ("off", "none", "false", "")
        )
    ):
        repetition_penalty = None
    else:
        repetition_penalty = _finite_float(rp_raw, "repetition_penalty")
        if repetition_penalty == 0.0:
            repetition_penalty = None  # 0 も無効化として扱う
        elif repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be >= 1.0 (1.0 = neutral; < 1.0 encourages "
                'repetition). Use 0 / false / "off" to disable injection.'
            )
    # 併せて注入する参照窓（省略時は付けない＝上流の既定 20 に任せる）。
    rcs_raw = data.get("repetition_context_size")
    repetition_context_size = (
        None if rcs_raw is None else _strict_int(rcs_raw, "repetition_context_size")
    )
    if repetition_context_size is not None and repetition_context_size < 1:
        raise ValueError("repetition_context_size must be 1 or greater")
    # 構造化リクエスト（tools / response_format）を注入対象から外すか（既定 false）。
    skip_structured = _strict_bool(
        data.get("repetition_penalty_skip_structured", False),
        "repetition_penalty_skip_structured",
    )
    return repetition_penalty, repetition_context_size, skip_structured


def _parse_llama_cpp_table(data: dict):
    """[llama_cpp] テーブル: llama-server バイナリの自動導入設定（すべて省略可＝全自動）。

    導入方法の選択肢（旧 provision）は無い——一本道なので accel / pin だけ。
    戻り値: (llama_accel, llama_build)。
    """
    llama = data.get("llama_cpp", {})
    if not isinstance(llama, dict):
        raise ValueError("[llama_cpp] must be a table")
    _reject_unknown_keys(llama, _LLAMA_CPP_CONFIG_KEYS, "llama_cpp")
    llama_accel = llama.get("accel", "auto")
    if not isinstance(llama_accel, str):
        raise ValueError("llama_cpp.accel must be a string")
    if llama_accel not in ("auto", "cuda", "vulkan", "metal", "cpu"):
        raise ValueError("llama_cpp.accel must be auto / cuda / vulkan / metal / cpu")
    llama_build = llama.get("pin")
    if llama_build is not None:
        if not isinstance(llama_build, str):
            raise ValueError("llama_cpp.pin must be a string")
        llama_build = llama_build.strip() or None
    return llama_accel, llama_build


def _parse_media_settings(data: dict):
    """画像・動画入力とリクエスト並列数の設定を検証する。"""
    # 動画入力のフレーム展開設定。省略で 8 フレーム / 長辺 768px。
    video_frames = _strict_int(data.get("video_frames", 8), "video_frames")
    if not (1 <= video_frames <= 32):
        raise ValueError("video_frames must be between 1 and 32")
    video_max_edge = _strict_int(data.get("video_max_edge", 768), "video_max_edge")
    if not (64 <= video_max_edge <= 2048):
        raise ValueError("video_max_edge must be between 64 and 2048")
    # 静止画の長辺上限（0 で無効）。極端に小さい値は事故なので 64px を下限にする。
    image_max_edge = _strict_int(data.get("image_max_edge", 1024), "image_max_edge")
    if image_max_edge != 0 and not (64 <= image_max_edge <= 4096):
        raise ValueError("image_max_edge must be 0 (disabled) or between 64 and 4096")
    max_request_workers = _strict_int(
        data.get("max_request_workers", 32), "max_request_workers"
    )
    if not (1 <= max_request_workers <= 256):
        raise ValueError("max_request_workers must be between 1 and 256")
    max_media_workers = _strict_int(
        data.get("max_media_workers", 2), "max_media_workers"
    )
    if not (1 <= max_media_workers <= min(max_request_workers, 16)):
        raise ValueError(
            "max_media_workers must be between 1 and min(max_request_workers, 16)"
        )
    return (
        video_frames,
        video_max_edge,
        image_max_edge,
        max_request_workers,
        max_media_workers,
    )


def _parse_model_entries(
    data: dict,
    *,
    dynamic: bool,
    internal_base: int,
    public_port: int,
    default_draft,
    default_backend: str,
    default_stream_tool_calls: bool = False,
):
    """[[models]] 配列を検証して ServerConfig 群に組み立てる。

    戻り値: (configs, seen)。seen は登録済み model id の集合（default_model の検証に使う）。
    """
    entries = data.get("models", [])
    if not isinstance(entries, list):
        raise ValueError("[[models]] must be an array")
    if not entries and not dynamic:
        raise ValueError(
            "gateway config needs a non-empty [[models]] array (or set dynamic = true)"
        )

    configs: list[ServerConfig] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("model"):
            raise ValueError("each [[models]] entry needs a 'model'")
        _reject_unknown_keys(entry, _MODEL_CONFIG_KEYS, f"models[{i}]")
        if not isinstance(entry["model"], str):
            raise ValueError("each [[models]].model must be a string")
        model = entry["model"].strip()
        if not model:
            raise ValueError("each [[models]] entry needs a non-empty 'model'")
        if model in seen:
            raise ValueError(f"duplicate model in gateway config: {model}")
        seen.add(model)
        backend = entry.get("backend", default_backend)
        if not isinstance(backend, str):
            raise ValueError(f"backend must be a string (model {model})")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS} (model {model})")
        internal_port = internal_base + i
        if internal_port > 65535:
            raise ValueError(
                "internal model ports exceed 65535; lower internal_base_port"
            )
        if internal_port == public_port:
            raise ValueError(
                f"internal port {internal_port} collides with the public port {public_port}; "
                "raise internal_base_port"
            )
        parallel = entry.get("parallel")
        if parallel is not None:
            parallel = _strict_int(parallel, f"parallel (model {model})")
            if parallel < 1:
                raise ValueError(f"parallel must be 1 or greater (model {model})")
        extra_args = entry.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(v, str) for v in extra_args
        ):
            raise ValueError(f"extra_args must be an array of strings (model {model})")
        draft = _resolve_model_draft(entry, default_draft, backend, model)
        configs.append(
            ServerConfig(
                backend=backend,
                model=model,
                host="127.0.0.1",
                port=internal_port,
                parallel=parallel,
                disable_thinking=_strict_bool(
                    entry.get("disable_thinking", False),
                    f"disable_thinking (model {model})",
                ),
                stream_tool_calls=_strict_bool(
                    entry.get("stream_tool_calls", default_stream_tool_calls),
                    f"stream_tool_calls (model {model})",
                ),
                draft_model=draft,
                extra_args=list(extra_args),
            )
        )
    return configs, seen


def load_gateway_config(path: str, *, default_backend: str) -> GatewayConfig:
    """ゲートウェイ設定 TOML を読み込んで検証する。

    形式（例）:
        host = "127.0.0.1"          # 公開ホスト（省略時 127.0.0.1）
        port = 8799                 # 公開ポート（省略時 8799）
        max_resident = 2            # 同時常駐モデル数の上限（ハード。省略時 無制限）
        load_timeout = 300          # 全枠処理中のとき空くのを待つ最大秒数（超過で 503。省略時 300）
        idle_timeout = 1200         # この秒数使われないモデルを自動アンロード（省略時 1200=20分。0 で無効）
        internal_base_port = 9001   # 内部サーバーの割当開始ポート（省略時 9001）
        default_model = "..."       # model 省略リクエスト時のモデル（省略可）
        draft_model = "auto"        # 全モデルの MTP ドラフター既定（mlx-vlm のみ有効。省略可）

        [[models]]
        model = "ToPo-ToPo/Qwen3.6-27B-mlx-4bit"
        backend = "mlx-vlm"
        # draft_model 省略 → 上の既定 "auto" を継承（Qwen3.6 の MTP）

        [[models]]
        model = "mlx-community/gemma-4-31b-it-4bit"
        backend = "mlx"
        draft_model = "off"         # このモデルだけ MTP を無効化（既定の打ち消し）
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    _reject_unknown_keys(data, _TOP_LEVEL_CONFIG_KEYS, "top-level")

    host = data.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    host = host.strip()
    # ゲートウェイは AF_INET（IPv4）で bind する。"::" / "*" は bind 時に分かりにくい
    # OSError で落ちるので、設定読み込みの時点で明確に断る。
    if host in ("::", "*"):
        raise ValueError(
            f'host = "{host}" is not supported; use "0.0.0.0" to listen on all '
            "IPv4 interfaces (or a specific IPv4 address)"
        )
    port = _port(data.get("port", 8799), "port")
    internal_base = _port(data.get("internal_base_port", 9001), "internal_base_port")
    # ネットワーク公開時の API キー（省略/空 で認証なし）。chat（/v1/*）と在席セッション
    api_key = data.get("api_key")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise ValueError("api_key must be a string")
        api_key = api_key.strip() or None
    allow_unauthenticated_remote = _strict_bool(
        data.get("allow_unauthenticated_remote", False),
        "allow_unauthenticated_remote",
    )
    try:
        is_loopback_bind = ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        is_loopback_bind = host.lower() == "localhost"
    if not is_loopback_bind and api_key is None and not allow_unauthenticated_remote:
        raise ValueError(
            "api_key is required when host is not loopback "
            "(set allow_unauthenticated_remote = true only for a trusted isolated LAN)"
        )
    max_resident = data.get("max_resident")
    if max_resident is not None:
        max_resident = _strict_int(max_resident, "max_resident")
        if max_resident < 1:
            raise ValueError("max_resident must be 1 or greater")
    default_model = data.get("default_model")
    if default_model is not None:
        if not isinstance(default_model, str) or not default_model.strip():
            raise ValueError("default_model must be a non-empty string")
        default_model = default_model.strip()
    # 一定時間使われないモデルを自動アンロードする秒数（idle TTL）。省略時 1200（=20分）、0 で無効。
    idle_timeout = data.get("idle_timeout", 1200)
    if idle_timeout is not None:
        idle_timeout = _finite_float(idle_timeout, "idle_timeout")
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be 0 or greater (0 disables)")
        if idle_timeout == 0:
            idle_timeout = None
    # 全枠が処理中のとき、空くのを待つ最大秒数（超過で 503）。
    load_timeout = _finite_float(data.get("load_timeout", 300.0), "load_timeout")
    if load_timeout < 1:
        raise ValueError("load_timeout must be 1 or greater")
    # モデルサーバー1つの起動完了（ready）を待つ最大秒数。巨大モデル・コールドディスクでは
    # 120 秒を超えることがあるので設定可能にする。
    start_timeout = _finite_float(data.get("start_timeout", 120.0), "start_timeout")
    if start_timeout < 1:
        raise ValueError("start_timeout must be 1 or greater")
    # 上流モデルサーバーとの通信タイムアウト（ソケット単位の無応答秒数）。省略時 600（=10分）、0 で無制限。
    # ハングした／沈黙したサーバーが inflight を握り続けて枠を塞ぐ事故の保険（トークンが流れている
    # 限り切れないので、ストリーミングの長時間生成は妨げない）。正当な長時間生成を切らないよう高め。
    request_timeout = data.get("request_timeout", 600.0)
    if request_timeout is not None:
        request_timeout = _finite_float(request_timeout, "request_timeout")
        if request_timeout < 0:
            raise ValueError("request_timeout must be 0 or greater (0 disables)")
        if request_timeout == 0:
            request_timeout = None
    # session_ttl は廃止（ハートビートによる生存推定をやめたため）。古い設定ファイルを
    # そのまま読めるよう、キーが在っても**エラーにせず無視**して警告だけ出す。
    if "session_ttl" in data:
        print(
            "gateway.toml: session_ttl は廃止されました（ハートビートによる生存推定を"
            "やめたため無視します）。モデルの保持時間は idle_timeout で調整してください。",
            file=sys.stderr,
        )
    # 0.38.15 で更新適用を手動操作だけに限定した。旧設定は移行前／読み取り専用設定でも
    # 起動を妨げないよう型だけ検証して受け入れるが、値にかかわらず適用には使わない。
    if "auto_update" in data:
        _strict_bool(data["auto_update"], "auto_update")
    tray = _strict_bool(data.get("tray", True), "tray")
    # 未登録モデルを ID 推論で動的ロードするか（既定 true）。false なら事前登録のみ（旧挙動）。
    dynamic = _strict_bool(data.get("dynamic", True), "dynamic")
    # 動的ロード時の既定 disable_thinking（事前登録の [[models]] は各自の値が優先）。
    dyn_disable_thinking = _strict_bool(
        data.get("disable_thinking", False), "disable_thinking"
    )
    # ツール呼び出しの生成中トークンを流す（mlx-vlm）。既定 off。[[models]] は各自の値が優先
    stream_tool_calls = _strict_bool(
        data.get("stream_tool_calls", False), "stream_tool_calls"
    )
    # ゲートウェイ全体の MTP ドラフター既定。各 [[models]] が draft_model を持たなければ
    # これを継承する（"auto" で本体名から自動選択）。個別に "" / "off" / "none" で無効化。
    default_draft = data.get("draft_model")
    if default_draft is not None and not isinstance(default_draft, str):
        raise ValueError("draft_model must be a string")
    # 動的ロード時の並列スロット既定（llama-cpp のみ。他バックエンドは逐次処理なので無視）。
    default_parallel = data.get("parallel")
    if default_parallel is not None:
        default_parallel = _strict_int(default_parallel, "parallel")
        if default_parallel < 1:
            raise ValueError("parallel must be 1 or greater")
    # メモリガード（→ docs/llama-cpp.md）。常駐モデルの推定占有量の合計を総RAMのこの割合に
    # 制限する。0 < x <= 1。省略で無効。
    max_memory_fraction = data.get("max_memory_fraction")
    if max_memory_fraction is not None:
        max_memory_fraction = _finite_float(max_memory_fraction, "max_memory_fraction")
        if not (0.0 < max_memory_fraction <= 1.0):
            raise ValueError("max_memory_fraction must be in (0, 1]")

    repetition_penalty, repetition_context_size, repetition_penalty_skip_structured = (
        _parse_repetition_settings(data)
    )

    llama_accel, llama_build = _parse_llama_cpp_table(data)

    (
        video_frames,
        video_max_edge,
        image_max_edge,
        max_request_workers,
        max_media_workers,
    ) = _parse_media_settings(data)

    configs, seen = _parse_model_entries(
        data,
        dynamic=dynamic,
        internal_base=internal_base,
        public_port=port,
        default_draft=default_draft,
        default_backend=default_backend,
        default_stream_tool_calls=stream_tool_calls,
    )

    # dynamic 無効のときだけ default_model が事前登録に在ることを要求する
    # （dynamic 有効なら未登録でも動的ロードされる）。
    if default_model is not None and not dynamic and default_model not in seen:
        raise ValueError(f"default_model '{default_model}' is not listed in [[models]]")

    return GatewayConfig(
        host,
        port,
        max_resident,
        default_model,
        configs,
        idle_timeout,
        load_timeout,
        start_timeout=start_timeout,
        request_timeout=request_timeout,
        dynamic=dynamic,
        disable_thinking=dyn_disable_thinking,
        stream_tool_calls=stream_tool_calls,
        draft_model=default_draft,
        parallel=default_parallel,
        max_memory_fraction=max_memory_fraction,
        internal_base_port=internal_base,
        api_key=api_key,
        allow_unauthenticated_remote=allow_unauthenticated_remote,
        tray=tray,
        llama_accel=llama_accel,
        llama_build=llama_build,
        video_frames=video_frames,
        video_max_edge=video_max_edge,
        image_max_edge=image_max_edge,
        max_request_workers=max_request_workers,
        max_media_workers=max_media_workers,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        repetition_penalty_skip_structured=repetition_penalty_skip_structured,
    )
