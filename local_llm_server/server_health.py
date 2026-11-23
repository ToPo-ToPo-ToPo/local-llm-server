"""OpenAI-compatible server health and model-list queries."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


def is_ready(base_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)
    except (urllib.error.URLError, OSError):
        return False


def preload_in_progress(base_url: str, timeout: float = 1.0) -> bool:
    """mlx-vlm（0.8 以降）が起動直後のモデル読み込み中かどうか（ルートの /health で見る）。

    0.8 の mlx-vlm は /v1/models に 200 を返したあとも読み込みを続け、その間の推論
    （POST）を 503 で断る。/v1/models だけで準備完了とすると、ゲートウェイがロード直後の
    最初の依頼をそのまま 503 で返してしまうので、起動待ちではこれも見る。
    /health が無い・読めないサーバー（llama-server など）は「読み込み中ではない」とみなす。
    """
    root = base_url.removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("preload_in_progress") is True


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [
        item["id"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    ]


def running_model(base_url: str, timeout: float = 5.0) -> str | None:
    models = list_models(base_url, timeout)
    return models[0] if models else None


def model_available(
    base_url: str, model: str | None, timeout: float = 5.0
) -> bool | None:
    if not model:
        return None
    models = list_models(base_url, timeout)
    if not models:
        return None
    return any(models_match(candidate, model) for candidate in models)


def models_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    if a == b:
        return True
    basename = lambda value: value.rstrip("/").split("/")[-1].lower()  # noqa: E731
    return basename(a) == basename(b)


def parse_host_port(base_url: str, default_port: int = 8080) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(base_url)
    return parsed.hostname or "127.0.0.1", parsed.port or default_port
