"""接続オーケストレーション: 相乗り判定・モデル整合チェック・自動起動。

複数のエージェントが各自実装していた「既存サーバーがあれば相乗りし、無ければ
自動起動して終了時に止める」ロジックを 1 箇所に集約する。local-automata の
`_prepare_server` を UI（argparse / print）非依存でライブラリ化したもの。

    from local_llm_server import ensure_server

    handle = ensure_server(model="ToPo-ToPo/Qwen3.6-27B-mlx-4bit", draft_model="auto")
    try:
        ...  # handle.base_url の OpenAI 互換 API を使う
    finally:
        handle.stop()   # 自分で起動した場合のみ停止（相乗りなら何もしない）

`register_atexit=True`（既定）なら自動起動分はプロセス終了時にも停止される。
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from typing import Callable

from .constants import DEFAULT_MODEL
from .server import (
    MTP_DRAFTERS,
    LocalServer,
    ServerConfig,
    default_backend,
    is_ready,
    list_models,
    models_match,
    parallel_supported,
    parse_host_port,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


class ServerNotRunningError(RuntimeError):
    """auto_start=False で、接続先にサーバーが居なかった。"""


@dataclass
class ServerHandle:
    """ensure_server の結果。相乗りか自動起動かを保持する。

    - rode_along=True … 既存サーバーに相乗りした。server は None（stop しても無害）。
    - rode_along=False … 自分で起動した。server に LocalServer が入り、stop() で止まる。
    """

    base_url: str
    server: LocalServer | None
    rode_along: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def started(self) -> bool:
        """このハンドルがサーバーを自動起動したか（＝stop で止める対象を持つか）。"""
        return self.server is not None

    def stop(self) -> None:
        """自動起動したサーバーを停止する。相乗り時は何もしない。冪等。"""
        if self.server is not None:
            self.server.stop()
            self.server = None

    def __enter__(self) -> "ServerHandle":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


def check_model_served(
    base_url: str, model: str | None, *, timeout: float = 5.0
) -> list[str]:
    """相乗り先が設定モデルを提供しているか確認し、警告メッセージ群を返す。

    取り違え（既存サーバーのモデルが意図と違う）を早期に気づくための警告。
    - 単一モデルサーバーでロード済みが食い違う → そのサーバーのモデルが使われる旨。
    - 多モデル（router 等）でカタログに無い → リクエストが失敗しうる旨。
    一覧が取れない/モデル未指定なら警告なし（空リスト）。
    """
    if not model:
        return []
    models = list_models(base_url, timeout)
    if not models or any(models_match(m, model) for m in models):
        return []
    if len(models) == 1:
        return [
            f"the running server has loaded '{models[0]}', but '{model}' was requested. "
            "The existing server's model will be used; stop it and restart to use the "
            "configured model."
        ]
    return [
        f"the server at {base_url} does not offer '{model}' ({len(models)} models "
        "available). The request may fail; point base_url at a server that serves it."
    ]


def ensure_server(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = DEFAULT_MODEL,
    backend: str | None = None,
    auto_start: bool = True,
    parallel: int | None = None,
    disable_thinking: bool = False,
    draft_model: str | None = None,
    start_timeout: float = 120.0,
    register_atexit: bool = True,
    log: Callable[[str], None] | None = None,
) -> ServerHandle:
    """接続先サーバーを用意する（相乗り or 自動起動）。

    手順:
      1. is_ready(base_url) が真 … 既存サーバーに相乗り（モデル整合を確認し警告）。
      2. auto_start=False で居ない … ServerNotRunningError。
      3. auto_start=True で居ない … LocalServer を起動し、応答可能になるまで待つ。

    backend を省略すると OS から自動判定（mac arm64→mlx-vlm、他→llama-cpp）。
    draft_model="auto" は本体名から MTP ドラフターを引く（未対応モデルは MTP 無しで
    起動。明示の HF id はそのまま使う）。MTP は mlx-vlm バックエンドでのみ有効。
    """
    emit = log or (lambda _m: None)

    # 1. 既存サーバーに相乗り
    if is_ready(base_url):
        emit(f"Connected: {base_url}")
        warnings = check_model_served(base_url, model)
        for w in warnings:
            emit("Warning: " + w)
        return ServerHandle(base_url, server=None, rode_along=True, warnings=warnings)

    # 2. 自動起動しない設定
    if not auto_start:
        be = backend or default_backend()
        hint = f"--backend {be}" + (f" --model {model}" if model else "")
        raise ServerNotRunningError(
            f"No server found at {base_url}. Start one with "
            f"`local-llm-server {hint}`, or call ensure_server(auto_start=True)."
        )

    # 3. 自動起動
    if not model:
        raise ValueError("model is required to auto-start a server.")
    be = backend or default_backend()
    if parallel is not None and not parallel_supported(be):
        emit(f"Warning: {be} does not support parallel slots; --parallel is ignored.")
        parallel = None

    # draft_model="auto" は未対応モデルだとエラーになるため、ここで穏当に無効化する
    # （明示 id はそのまま LocalServer 側で解決される）。
    draft = draft_model
    if draft == "auto" and model not in MTP_DRAFTERS:
        draft = None

    host, port = parse_host_port(base_url)
    server = LocalServer(
        ServerConfig(
            backend=be,
            model=model,
            host=host,
            port=port,
            parallel=parallel,
            disable_thinking=disable_thinking,
            draft_model=draft,
        )
    )
    emit(f"No server detected -> auto-starting {be} (model: {model})...")
    server.start()
    emit(f"  Server log: {server.log_path}")
    try:
        server.wait_until_ready(timeout=start_timeout)
    except (RuntimeError, TimeoutError):
        server.stop()
        raise
    if register_atexit:
        atexit.register(server.stop)
    return ServerHandle(base_url, server=server, rode_along=False, warnings=[])
