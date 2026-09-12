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
