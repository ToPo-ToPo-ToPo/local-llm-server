"""Stage a versioned uv tool environment without deleting its predecessor.

Run with scripts/uv run --no-project --python 3.13.18 scripts/install_runtime.py.
Use --activate to switch this app's launcher and register its user service.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def launcher_python(launcher: Path) -> Path | None:
    """Find the venv interpreter even when a long shebang uses a shell shim."""
    if not launcher.exists():
        return None
    target = launcher.resolve()
    sibling = target.parent / "python"
    if sibling.is_file():
        return sibling
    first = target.read_text().splitlines()[0]
    if first.startswith("#!"):
        interpreter = Path(first[2:])
        if interpreter.name.startswith("python") and interpreter.is_file():
            return interpreter
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--previous-python", type=Path)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 13, 18):
        parser.error("run with scripts/uv run --no-project --python 3.13.18")
    if os.name == "nt":
        parser.error("This service installer is for macOS/Linux; see docs/operation.md for Windows.")
    launcher = Path.home() / ".local" / "bin" / "gw"
    if args.uninstall:
        if not launcher.is_symlink():
            print("No managed gw symlink; no environment was deleted.")
            return
        target = launcher.resolve()
        prefix = target.parent.parent
        if target.name != "gw" or prefix.name != "local-llm-server":
            raise RuntimeError(f"Refusing to unlink another launcher: {launcher}")
        if prefix.is_relative_to(ROOT / ".runtime") and prefix.name == "local-llm-server":
            env = {**os.environ, "UV_TOOL_DIR": str(prefix.parent),
                   "UV_TOOL_BIN_DIR": str(prefix.parents[2] / "bin")}
            run([str(ROOT / "scripts/uv"), "tool", "uninstall", "local-llm-server"], env=env)
        # Historical environments are kept for explicit cleanup / rollback.
        launcher.unlink()
        return
    previous = args.previous_python
    if previous is None:
        previous = launcher_python(launcher)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = ROOT / ".runtime" / "installs" / stamp
    destination.mkdir(parents=True)
    constraints = destination / "constraints.txt"
    if previous is not None:
        # Do not flatten a running environment to the potentially older dev lock.
        code = (
            "import importlib.metadata as m,json;"
            "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
        )
        versions = json.loads(run([str(previous), "-B", "-c", code], capture_output=True).stdout)
        versions = {n:v for n,v in versions.items() if n.lower().replace('_','-') != 'local-llm-server'}
        versions["certifi"] = "2026.10.23"
        versions["pip"] = "26.2.2"
        constraints.write_text("".join(f"{n}=={v}\n" for n,v in sorted(versions.items())), encoding="utf-8")
    else:
        exported = run([str(ROOT/"scripts/uv"), "export", "--frozen", "--no-dev",
                        "--no-emit-project", "--no-hashes"], cwd=ROOT, capture_output=True)
        constraints.write_text(exported.stdout, encoding="utf-8")
    env = {**os.environ, "UV_TOOL_DIR": str(destination/"uv/tools"),
           "UV_TOOL_BIN_DIR": str(destination/"bin")}
    run([str(ROOT/"scripts/uv"), "tool", "install", "--editable", str(ROOT),
         "--python", sys.executable, "--constraint", str(constraints),
         "--with", "pip==26.2.2"], cwd=ROOT, env=env)
    prefix = destination / "uv/tools/local-llm-server"
    py = prefix / "bin/python"
    gw = prefix / "bin/gw"
    verify = (
        "from pathlib import Path;import sys;"
        "from local_llm_server import tls_policy as t,update;"
        "t.require_secure_python();t.ca_file();"
        "p=Path(sys.prefix);"
        "(p/'.gw-runtime-constraints.txt').write_text(t.installed_constraints());"
        "(p/update._DEPS_FINGERPRINT_NAME).write_text(update._deps_fingerprint(Path(sys.argv[1]))+'\\n')"
    )
    run([str(py), "-B", "-c", verify, str(ROOT)])
    run([str(ROOT/"scripts/uv"), "pip", "check", "--python", str(py)], cwd=ROOT)
    record = {"prefix": str(prefix), "python": str(py), "gw": str(gw),
              "previous_python": str(previous) if previous else None,
              "previous_launcher": os.readlink(launcher) if launcher.is_symlink() else None,
              "activated": False}
    if args.activate:
        launcher.parent.mkdir(parents=True, exist_ok=True)
        if launcher.exists() and not launcher.is_symlink():
            raise RuntimeError(f"Refusing to replace a non-symlink launcher: {launcher}")
        pending = launcher.with_name(".gw-new-"+stamp)
        pending.symlink_to(gw)
        pending.replace(launcher)
        record["activated"] = True
        run([str(launcher), "enable"])
    (destination/"install.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
