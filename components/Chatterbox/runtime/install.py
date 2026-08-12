#!/usr/bin/env python3
"""CLI for the user-local Chatterbox CPU runtime lane."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from .runtime import ProvisioningError, build_paths, install_runtime, preflight, verify_lock
except ImportError:  # Direct execution: python components/Chatterbox/runtime/install.py
    from runtime import ProvisioningError, build_paths, install_runtime, preflight, verify_lock


def _runtime_dir() -> Path:
    return Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision the isolated Linux x86_64 Python 3.11 Chatterbox CPU runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "verify-lock", "install"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--python", dest="python_path", help="explicit Python 3.11 interpreter")
        sub.add_argument("--runtime-dir", type=Path, default=_runtime_dir())
        sub.add_argument("--repo-root", type=Path)
        sub.add_argument("--worker", type=Path, help="worker script for the optional --self-test hook")
    return parser


def _interpreter(value: str | None) -> Path:
    if not value:
        raise ProvisioningError("--python is required; pass an explicit Python 3.11 interpreter")
    resolved = shutil.which(value) or value
    return Path(resolved).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = build_paths(runtime_dir=args.runtime_dir, repo_root=args.repo_root)
        if args.command == "verify-lock":
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            result = verify_lock(paths.lock_path, manifest)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        interpreter = _interpreter(args.python_path)
        if args.command == "preflight":
            result = preflight(paths, interpreter)
        else:
            result = install_runtime(paths, interpreter, worker_script=args.worker)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok", True) else 2
    except (OSError, ProvisioningError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
