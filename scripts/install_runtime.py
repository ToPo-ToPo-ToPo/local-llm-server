"""Build and verify a new runtime while preserving its predecessor.

Run with scripts/uv run --no-project --python 3.13.18 scripts/install_runtime.py.
Use --dry-run to inspect migration; --activate keeps the active and previous runtimes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
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
    lines = target.read_text().splitlines()
    if lines and lines[0].startswith("#!"):
        interpreter = Path(lines[0][2:])
        if interpreter.name.startswith("python") and interpreter.is_file():
            return interpreter
    return None


def read_versions(python: Path) -> dict[str, str]:
    # Isolate sys.path so cwd/PYTHONPATH cannot substitute the new project's
    # metadata for the previous environment. No backend or network is used.
    code = (
        "import runpy,json,sys;p=runpy.run_path(sys.argv[1]);"
        "print(json.dumps(p['installed_versions']()))"
    )
    try:
        result = run([str(python), "-I", "-B", "-c", code,
                      str(ROOT/"local_llm_server/migration_policy.py")], capture_output=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("旧環境の読み取りに失敗しました: " + (error.stderr or str(error))[-1500:]) from error
    return json.loads(result.stdout)


def verify_activation(python: Path) -> dict:
    """Check HTTP health, process ownership, environment, and loaded version."""
    code = """
import json,sys,time
from pathlib import Path
from importlib.metadata import version
from local_llm_server.gateway_runtime import read_gateway_runtime,gateway_admin_status
end=time.monotonic()+25
while time.monotonic()<end:
    record=read_gateway_runtime()
    if record and record.get('cmdline') and Path(record['cmdline'][0]).absolute()==Path(sys.executable).absolute():
        status=gateway_admin_status(record['host'],record['port'],timeout=1)
        if status and status.get('pid')==record['pid'] and status.get('update',{}).get('running')==version('local-llm-server'):
            print(json.dumps({'pid':record['pid'],'python':sys.executable,'version':version('local-llm-server')}))
            break
    time.sleep(.25)
else:
    raise RuntimeError('New runtime did not become healthy; older environments retained')
"""
    result = run([str(python), "-I", "-B", "-c", code], capture_output=True, timeout=35)
    return json.loads(result.stdout)


def prune_old_installations(python: Path, prefix: Path, previous: str | None,
                            launcher: Path) -> dict:
    code = (
        "import json,sys;from pathlib import Path;"
        "from local_llm_server.runtime_retention import prune_installations;"
        "print(json.dumps(prune_installations(Path(sys.argv[1]),Path(sys.argv[2]),"
        "sys.argv[3] or None,Path(sys.argv[4]))))"
    )
    result = run([str(python), "-I", "-B", "-c", code, str(ROOT), str(prefix),
                  previous or "", str(launcher)], capture_output=True, timeout=60)
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--activate", action="store_true")
    mode.add_argument("--uninstall", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--previous-python", type=Path)
    args = parser.parse_args(argv)
    if sys.version_info[:3] != (3, 13, 18):
        parser.error("run with scripts/uv run --no-project --python 3.13.18")
    if os.name == "nt":
        parser.error("This service installer is for macOS/Linux; see docs/operation.md for Windows.")
    launcher = Path.home()/".local/bin/gw"
    if args.dry_run:
        _install(args, launcher)
        return
    retention = runpy.run_path(str(ROOT/"local_llm_server/runtime_retention.py"))
    with retention["installation_lock"](ROOT):
        _install(args, launcher)


def _install(args, launcher: Path) -> None:
    if args.uninstall:
        if not launcher.is_symlink():
            print("No managed gw symlink; no environment was deleted.")
            return
        target = launcher.resolve()
        prefix = target.parent.parent
        if target.name != "gw" or prefix.name != "local-llm-server":
            raise RuntimeError(f"Refusing to unlink another launcher: {launcher}")
        if prefix.is_relative_to(ROOT/".runtime"):
            env = {**os.environ, "UV_TOOL_DIR": str(prefix.parent),
                   "UV_TOOL_BIN_DIR": str(prefix.parents[2]/"bin")}
            run([str(ROOT/"scripts/uv"), "tool", "uninstall", "local-llm-server"], env=env)
        launcher.unlink()
        return

    policy = runpy.run_path(str(ROOT/"local_llm_server/migration_policy.py"))
    previous = args.previous_python or launcher_python(launcher)
    versions = read_versions(previous) if previous is not None else {}
    pins, changes = policy["plan_migration"](versions)
    plan = {"previous_python": str(previous) if previous else None,
            "security_changes": changes, "preserved_packages": len(pins)-len(changes)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.activate and launcher.exists() and not launcher.is_symlink():
        raise RuntimeError(f"Refusing to replace a non-symlink launcher: {launcher}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = ROOT/".runtime/installs"/stamp
    destination.mkdir(parents=True)
    constraints = destination/"constraints.txt"
    prefix = destination/"uv/tools/local-llm-server"
    python, gw = prefix/"bin/python", prefix/"bin/gw"
    record = {**plan, "prefix": str(prefix), "python": str(python), "gw": str(gw),
              "previous_launcher": os.readlink(launcher) if launcher.is_symlink() else None,
              "activated": False, "status": "planned"}
    def save():
        (destination/"install.json").write_text(json.dumps(record, indent=2)+"\n")
    save()
    try:
        if previous is not None:
            constraints.write_text(policy["format_constraints"](pins), encoding="utf-8")
        else:
            exported = run([str(ROOT/"scripts/uv"), "export", "--frozen", "--no-dev",
                            "--no-emit-project", "--no-hashes"], cwd=ROOT, capture_output=True)
            constraints.write_text(exported.stdout, encoding="utf-8")
        env = {**os.environ, "UV_TOOL_DIR": str(destination/"uv/tools"),
               "UV_TOOL_BIN_DIR": str(destination/"bin")}
        command = [str(ROOT/"scripts/uv"), "tool", "install", "--editable", str(ROOT),
                   "--python", sys.executable, "--constraint", str(constraints),
                   "--with", "pip=="+pins["pip"]]
        if previous is not None:
            # Constraints alone do not retain packages installed through extras
            # or --with. Request every previous pin in this new environment.
            command += ["--with-requirements", str(constraints)]
        record["status"] = "installing"
        save()
        run(command, cwd=ROOT, env=env)
        actual = read_versions(python)
        expected = pins if previous is not None else {"pip": pins["pip"]}
        mismatches = {name: {"expected": version, "actual": actual.get(name)}
                      for name, version in expected.items() if actual.get(name) != version}
        if mismatches:
            raise RuntimeError("Migration changed pinned packages: "+json.dumps(mismatches))
        verify = (
            "from pathlib import Path;import sys,subprocess;"
            "from local_llm_server import tls_policy as t,update;"
            "t.require_secure_python();t.ca_file();"
            "ok=t.runtime_compatible(sys.executable,subprocess.run,require_pip=True);"
            "\nif not ok: raise RuntimeError('Runtime security checks failed')\n"
            "p=Path(sys.prefix);"
            "(p/'.gw-runtime-constraints.txt').write_text(t.installed_constraints());"
            "(p/update._DEPS_FINGERPRINT_NAME).write_text(update._deps_fingerprint(Path(sys.argv[1]))+chr(10))"
        )
        run([str(python), "-I", "-B", "-c", verify, str(ROOT)])
        run([str(ROOT/"scripts/uv"), "pip", "check", "--python", str(python)], cwd=ROOT)
        record["status"] = "verified"
        save()
        if args.activate:
            launcher.parent.mkdir(parents=True, exist_ok=True)
            current = os.readlink(launcher) if launcher.is_symlink() else None
            if current != record["previous_launcher"] or (launcher.exists() and not launcher.is_symlink()):
                raise RuntimeError("Launcher changed while preparing the environment")
            pending = launcher.with_name(".gw-new-"+stamp)
            pending.symlink_to(gw)
            pending.replace(launcher)
            record["activated"] = True
            record["status"] = "activated"
            save()
            try:
                run([str(launcher), "enable"])
            except Exception:
                if launcher.is_symlink() and os.readlink(launcher) == str(gw):
                    if record["previous_launcher"] is None:
                        launcher.unlink()
                    else:
                        rollback = launcher.with_name(".gw-rollback-"+stamp)
                        rollback.symlink_to(record["previous_launcher"])
                        rollback.replace(launcher)
                    record["activated"] = False
                raise
            record["health"] = verify_activation(python)
            save()
            try:
                record["cleanup"] = prune_old_installations(
                    python, prefix, record["previous_launcher"], launcher)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                # An uncertain cleanup must not undo a healthy activation.
                record["cleanup"] = {"removed": [], "error": str(error)[-1500:]}
                print("Old-runtime cleanup skipped: " + str(error)[-1500:], file=sys.stderr)
            save()
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)[:1000]
        save()
        raise
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"環境の作成・移行に失敗しました: {error}", file=sys.stderr)
        raise SystemExit(1)
