"""Choose only reviewed security updates; usable by the standalone installer.

Keep this module dependency-free: bootstrapping must not import the application
or require packages from the environment that is about to be replaced.
"""
from __future__ import annotations

import importlib.metadata
import json
import re
import sys
import urllib.parse

SECURITY_RELEASES = {
    "certifi": ((2026, 10, 19), "2026.10.23"),
    "pip": ((26, 2, 2), "26.2.2"),
    "filelock": ((4, 2, 1), "4.2.4"),
    "pillow": ((13, 0, 0), "13.0.0"),
    "setuptools": ((84, 1, 0), "84.2.1"),
    "urllib3": ((2, 8, 0), "2.8.0"),
    "msgpack": ((1, 2, 3), "1.2.3"),
}
_RELEASE = re.compile(
    r"(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?P<pre>(?:a|b|rc)\d+)?(?P<post>\.post\d+)?(?P<dev>\.dev\d+)?"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?", re.IGNORECASE,
)


def canonical_name(name: str) -> str:
    name = re.sub(r"[-_.]+", "-", name).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError(f"Invalid distribution name: {name!r}")
    return name


def installed_versions() -> dict[str, str]:
    """Read installed metadata without downloading or importing backends."""
    versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = canonical_name(dist.metadata["Name"])
        if name == "local-llm-server":
            continue
        # A version alone cannot reproduce a local/VCS dependency's source.
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        origin = urllib.parse.urlsplit(direct.get("url", ""))
        # Bootstrap pip/CA wheels can be staged locally before a safe network
        # environment exists. They are reviewed tooling, not another editable
        # application; retain their version or use the reviewed security update.
        seed_bootstrap = (name in ("pip", "certifi") and origin.scheme == "file"
                    and urllib.parse.unquote(origin.path).rsplit("/", 1)[-1]
                    == f"{name}-{dist.version}-py3-none-any.whl"
                    and "archive_info" in direct
                    and "dir_info" not in direct and "vcs_info" not in direct)
        if direct and not seed_bootstrap:
            raise ValueError(f"{name} uses a direct URL; preserve its source explicitly before migration")
        if name in versions and versions[name] != dist.version:
            raise ValueError(f"Conflicting installed versions: {name}")
        versions[name] = dist.version
    return versions


def below_floor(version: str, floor: tuple[int, ...]) -> bool:
    """Compare normalized distribution versions without a bootstrap dependency."""
    match = _RELEASE.fullmatch(version)
    if match is None:
        raise ValueError(f"Unsupported security dependency version: {version!r}")
    if int(match.group("epoch") or 0):
        return False
    release = tuple(map(int, match.group("release").split(".")))
    width = max(len(release), len(floor))
    release += (0,) * (width - len(release))
    minimum = floor + (0,) * (width - len(floor))
    if release != minimum:
        return release < minimum
    return bool(match.group("pre") or (match.group("dev") and not match.group("post")))


def plan_migration(
    versions: dict[str, str], *, platform: str | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, str | None]]]:
    """Preserve all pins except reviewed security updates and bootstrap tools."""
    platform = sys.platform if platform is None else platform
    pins: dict[str, str] = {}
    for original, version in versions.items():
        name = canonical_name(original)
        if name == "local-llm-server":
            continue
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.!+_-]*", version):
            raise ValueError(f"Invalid installed version for {name}: {version!r}")
        if name in pins and pins[name] != version:
            raise ValueError(f"Conflicting installed versions: {name}")
        pins[name] = version
    changes: dict[str, dict[str, str | None]] = {}
    for name, (minimum, target) in SECURITY_RELEASES.items():
        if name == "setuptools" and platform != "darwin":
            continue  # Preserve upstream CUDA caps; do not silently alter them.
        before = pins.get(name)
        if before is None and name not in ("certifi", "pip"):
            continue  # Do not introduce optional/development packages.
        if before is None or below_floor(before, minimum):
            pins[name] = target
            changes[name] = {"before": before, "after": target}
    return pins, changes


def format_constraints(versions: dict[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))
