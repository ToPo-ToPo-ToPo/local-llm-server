"""Resolved backend executables for the current gateway process."""

from __future__ import annotations

import sys


_PROVISIONED: dict[str, dict] = {}


def set_provisioned(kind: str, info: dict | None) -> None:
    if info is None:
        _PROVISIONED.pop(kind, None)
    else:
        _PROVISIONED[kind] = dict(info)


def provisioned(kind: str) -> dict | None:
    return _PROVISIONED.get(kind)


def set_llama_server_binary(
    path: str | None, *, build: str | None = None, accel: str | None = None
) -> None:
    set_provisioned(
        "llama",
        None if path is None else {"binary": path, "build": build, "accel": accel},
    )


def llama_server_binary() -> str:
    return ((provisioned("llama") or {}).get("binary")) or "llama-server"


def llama_provision_info() -> dict | None:
    return provisioned("llama")


def set_vllm_python(path: str | None) -> None:
    set_provisioned("vllm", None if path is None else {"python": path})


def vllm_python() -> str:
    return ((provisioned("vllm") or {}).get("python")) or sys.executable


def vllm_provision_info() -> dict | None:
    return provisioned("vllm")


def set_sglang_python(path: str | None) -> None:
    set_provisioned("sglang", None if path is None else {"python": path})


def sglang_python() -> str:
    return ((provisioned("sglang") or {}).get("python")) or sys.executable


def sglang_provision_info() -> dict | None:
    return provisioned("sglang")
