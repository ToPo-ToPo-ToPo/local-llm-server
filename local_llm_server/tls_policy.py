"""TLS requirements shared by launch, downloads and isolated environments.

CPython's October 2026 security releases reject compromised eMudhra SPKIs.
CA bundles and programs with independent TLS implementations need separate care.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys

PYTHON_FLOORS = {(3, 11): 18, (3, 12): 16, (3, 13): 18, (3, 14): 10, (3, 15): 2}
CERTIFI_MIN = (2026, 10, 19)
PIP_MIN = (26, 2, 2)
UV_VERSION = "0.13.6"
# Independent of the certificate policy: known-vulnerable installed dependencies.
# Linux CUDA backends retain their upstream setuptools cap; do not force a
# conflicting version into those environments.
DEPENDENCY_FLOORS = {
    "filelock": (4, 2, 1),
    "pillow": (13, 0, 0),
    "setuptools": (84, 1, 0),
    "urllib3": (2, 8, 0),
}


def dependency_constraints() -> tuple[str, ...]:
    return tuple(
        name + ">=" + ".".join(map(str, floor))
        + ("; sys_platform == 'darwin'" if name == "setuptools" else "")
        for name, floor in DEPENDENCY_FLOORS.items()
    )



def python_is_secure(version: tuple[int, ...]) -> bool:
    major, minor, patch = version[:3]
    floor = PYTHON_FLOORS.get((major, minor))
    return patch >= floor if floor is not None else (major, minor) > (3, 15)


def require_secure_python() -> None:
    if not python_is_secure((sys.version_info.major, sys.version_info.minor, sys.version_info.micro)):
        raise RuntimeError(
            "eMudhra対策済みのPythonが必要です: 3.11.18 / 3.12.16 / "
            "3.13.18 / 3.14.10 / 3.15.2以上（同じ系列内）。"
            "リポジトリでは scripts/uv を使って環境を作成してください。"
        )


def ca_file() -> str:
    import certifi

    version = tuple(int(part) for part in importlib.metadata.version("certifi").split("."))
    if version < CERTIFI_MIN:
        raise RuntimeError("certifi>=2026.10.19が必要です。環境を同期してください。")
    return certifi.where()


def git_environment() -> dict[str, str]:
    """Configure this Git child only; never edit Git or OS settings."""
    bundle = ca_file()
    env = dict(os.environ)
    env.pop("GIT_SSL_NO_VERIFY", None)
    env["GIT_SSL_CAINFO"] = bundle
    # Avoid merging a stale OpenSSL hashed directory into the selected bundle.
    env["GIT_SSL_CAPATH"] = str(Path(bundle).parent)
    return env


def disable_xet() -> None:
    """Use the Python HTTP transport until a patched hf-xet wheel is verified."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        setattr(constants, "HF_HUB_DISABLE_XET", True)


def installed_constraints() -> str:
    """Preserve installed versions when updating a uv tool environment."""
    pins = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.lower().replace("_", "-") != "local-llm-server":
            pins.append(f"{name}=={dist.version}")
    return "\n".join(sorted(set(pins), key=str.lower)) + "\n"


def project_uv(root: Path) -> Path:
    return root / ".tools" / "uv" / UV_VERSION / ("uv.exe" if os.name == "nt" else "uv")


# Standalone environments do not contain local_llm_server. Check them in their
# own interpreter instead of borrowing the parent's metadata or CA bundle.
RUNTIME_CHECK = (
    "import sys,importlib.metadata as m;"
    f"floors={PYTHON_FLOORS!r};"
    "v=sys.version_info;f=floors.get(v[:2]);"
    "\nif not (v.micro>=f if f is not None else v[:2]>(3,15)): raise RuntimeError('unpatched Python')\n"
    "cv=tuple(map(int,m.version('certifi').split('.')));"
    f"\nif cv<{CERTIFI_MIN!r}: raise RuntimeError('unpatched certifi')\n"
    "import re\n"
    f"for name, floor in {DEPENDENCY_FLOORS!r}.items():\n"
    " if name == 'setuptools' and sys.platform != 'darwin': continue\n"
    " try: version = m.version(name)\n"
    " except m.PackageNotFoundError: continue\n"
    " match = re.fullmatch(r'(\\d+)\\.(\\d+)(?:\\.(\\d+))?(?:\\.post\\d+)?', version)\n"
    " if not match: raise RuntimeError('unsupported dependency version: ' + name + '==' + version)\n"
    " if tuple(int(p or 0) for p in match.groups()) < floor: raise RuntimeError('unpatched dependency: ' + name + '==' + version)\n"
)


def runtime_compatible(py: str, run, *, require_pip: bool = False) -> bool:
    import subprocess

    script = RUNTIME_CHECK
    if require_pip:
        script += (
            "pv=tuple(map(int,m.version('pip').split('.')));"
            f"\nif pv<{PIP_MIN!r}: raise RuntimeError('unpatched pip')\n"
        )
    try:
        return run([py, "-c", script], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
