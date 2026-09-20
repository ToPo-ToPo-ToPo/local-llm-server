"""Dependency-free backend metadata and server configuration types.

This module is the foundation of the gateway dependency graph.  It deliberately
does not import command builders, process management, or gateway orchestration.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from .constants import BACKENDS


def default_backend() -> str:
    """Return the platform-appropriate default inference backend."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx-vlm"
    return "llama-cpp"


DEFAULT_BACKEND = default_backend()


_STT_HINTS = ("whisper", "parakeet")


def infer_backend(model: str) -> str:
    """Infer a backend for a model that is not explicitly registered."""
    low = model.lower()
    if any(hint in low for hint in _STT_HINTS):
        return "whisper"
    if "gguf" in low:
        return "llama-cpp"
    if "mlx" in low:
        return "mlx-vlm"
    return default_backend()


@dataclass(frozen=True)
class PromptCacheConfig:
    """プロンプトキャッシュ（mlx-vlm の APC）の設定。gateway.toml の `[prompt_cache]` テーブル。

    人が触るのはこの設定ファイルだけで、モデルサーバーへは起動時の環境変数（APC_*）として内部で渡す
    （環境変数はモデルサーバー側の受け口であって、利用者向けの設定口ではない）。既定値はここが正本。

    - enabled: キャッシュを使う
    - entries: 保持するスナップショットの数（exact モード。ハイブリッド注意機構のモデルはプレフィックス
      丸ごとのスナップショット。1 手番に複数回呼ぶエージェント向けに上流既定 2 より多い）
    - guard_tokens: スナップショットをプロンプト末尾から何トークン手前で切るか（毎回変わる末尾の長さぶん）
    - memory_max_gb: メモリ上のキャッシュの上限（None で mlx-vlm の見積り＝空き RAM から判断）
    - disk: ディスク層を使う（スナップショットをディスクにも書き、再起動後も前方一致を復元できる。
      書き込みはプリフィル中に走る）
    - disk_max_gb: ディスク層の上限（None で mlx-vlm 既定 20）

    命中の有無と量はモデルサーバーの INFO ログ ``Prefill completed: … cached_tokens=N`` で分かる（専用の
    デバッグ設定は持たない。mlx-vlm 内部の詳細が要るときだけ、その環境変数 APC_DEBUG を直接使う）。
    """

    enabled: bool = True
    entries: int = 8
    guard_tokens: int = 1024
    memory_max_gb: float | None = None
    disk: bool = True
    disk_max_gb: float | None = None

    def env(self) -> dict[str, str]:
        """モデルサーバー（mlx-vlm）へ渡す環境変数。"""
        out = {
            "APC_ENABLED": "1" if self.enabled else "0",
            "APC_EXACT_CACHE_ENTRIES": str(int(self.entries)),
            "APC_EXACT_PREFIX_GUARD_TOKENS": str(int(self.guard_tokens)),
            "APC_DISK_ENABLED": "1" if self.disk else "0",
        }
        if self.memory_max_gb is not None:
            out["APC_MEMORY_MAX_GB"] = f"{float(self.memory_max_gb):g}"
        if self.disk_max_gb is not None:
            out["APC_DISK_MAX_GB"] = f"{float(self.disk_max_gb):g}"
        return out


@dataclass
class ServerConfig:
    """Configuration for one local model-server process."""

    backend: str
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    parallel: int | None = None
    disable_thinking: bool = False
    draft_model: str | None = None
    stream_tool_calls: bool = False
    extra_args: list[str] = field(default_factory=list)
    prompt_cache: PromptCacheConfig = field(default_factory=PromptCacheConfig)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass(frozen=True)
class BackendSpec:
    """Stable backend capabilities, independent from command construction."""

    name: str
    # Kept on the public descriptor for compatibility.  Foundation registries
    # leave it unset; server.py attaches concrete command builders at composition.
    build: Callable[[ServerConfig], list[str]] | None = None
    draft_style: str | None = None
    parallel: bool = False
    gguf: bool = False
    provisioner: str | None = None


BACKEND_SPECS: dict[str, BackendSpec] = {
    spec.name: spec
    for spec in (
        BackendSpec("mlx"),
        BackendSpec("mlx-vlm", draft_style="mtp"),
        BackendSpec(
            "llama-cpp",
            draft_style="gguf",
            parallel=True,
            gguf=True,
            provisioner="llama",
        ),
        BackendSpec("whisper"),
        BackendSpec("vllm", provisioner="vllm"),
        BackendSpec("sglang", provisioner="sglang"),
    )
}


def backend_spec(name: str) -> BackendSpec:
    """Look up backend capabilities, rejecting unknown names."""
    spec = BACKEND_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown backend: {name!r} (choose from {BACKENDS})")
    return spec


def parallel_supported(backend: str) -> bool:
    """Whether the backend accepts an explicit parallel-slot count."""
    spec = BACKEND_SPECS.get(backend)
    return spec is not None and spec.parallel
