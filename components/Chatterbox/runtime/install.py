#!/usr/bin/env python3
"""CLI for the user-local Chatterbox CPU runtime lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

try:
    from .runtime import (
        ProvisioningError,
        acquire_model,
        build_paths,
        install_runtime,
        model_preflight,
        preflight,
        product_status,
        verify_lock,
    )
    from .contract_data import PKUSEG_DATA_FILENAME, PKUSEG_DATA_SHA256, SUPPORTED_MODEL_PROFILES
    from .measurement import (
        DISPOSABLE_MARKER,
        DISPOSABLE_MARKER_CONTENT,
        MeasurementSession,
        REQUIRED_PATH_NAMES,
        VerifiedInput,
    )
    from .measurement_harness import (
        CURRENT_LOCK_REQUIREMENT_COUNT,
        CURRENT_LOCK_SHA256,
        REQUIRED_ENVIRONMENT_PATHS,
        run_disposable_measurement,
        validate_complete_wheelhouse,
    )
except ImportError:  # Direct execution: python components/Chatterbox/runtime/install.py
    from runtime import (
        ProvisioningError,
        acquire_model,
        build_paths,
        install_runtime,
        model_preflight,
        preflight,
        product_status,
        verify_lock,
    )
    from contract_data import PKUSEG_DATA_FILENAME, PKUSEG_DATA_SHA256, SUPPORTED_MODEL_PROFILES  # type: ignore[no-redef]
    from measurement import (  # type: ignore[no-redef]
        DISPOSABLE_MARKER,
        DISPOSABLE_MARKER_CONTENT,
        MeasurementSession,
        REQUIRED_PATH_NAMES,
        VerifiedInput,
    )
    from measurement_harness import (  # type: ignore[no-redef]
        CURRENT_LOCK_REQUIREMENT_COUNT,
        CURRENT_LOCK_SHA256,
        REQUIRED_ENVIRONMENT_PATHS,
        run_disposable_measurement,
        validate_complete_wheelhouse,
    )


def _runtime_dir() -> Path:
    return Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision the isolated Linux x86_64 Python 3.11 Chatterbox CPU runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight",
        "verify-lock",
        "install",
        "model-preflight",
        "acquire-model",
        "status",
        "measure",
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--python", dest="python_path", help="explicit Python 3.11 interpreter")
        sub.add_argument("--runtime-dir", type=Path, default=_runtime_dir())
        sub.add_argument(
            "--model",
            choices=SUPPORTED_MODEL_PROFILES,
            default="v2",
            help="multilingual checkpoint profile (default: v2)",
        )
        sub.add_argument("--repo-root", type=Path)
        sub.add_argument("--worker", type=Path, help="worker script for the mandatory self-test (defaults to the repository worker)")
        if command == "measure":
            sub.add_argument("--measurement-root", type=Path, required=True)
            sub.add_argument("--wheelhouse", type=Path, required=True)
            sub.add_argument("--model-root", type=Path)
            sub.add_argument(
                "--pkuseg-data",
                type=Path,
                help="verified spacy_ontonotes.zip input for model-baseline measurement",
            )
            sub.add_argument("--run-id", default="runtime-and-model-1")
            sub.add_argument("--scenario", choices=("runtime-baseline", "model-baseline"), default="model-baseline")
            sub.add_argument("--environment-id", default="local-disposable")
            sub.add_argument("--filesystem-layout", choices=("same-filesystem", "split-filesystem"), default="same-filesystem")
            sub.add_argument("--network-isolation-capability")
            sub.add_argument("--sample-interval", type=float, default=0.05)
    return parser


def _manifest_path(runtime_dir: Path, model: str) -> Path:
    filename = "runtime-manifest.json" if model == "v2" else f"runtime-manifest-{model}.json"
    return (runtime_dir / filename).resolve()


def _interpreter(value: str | None) -> Path:
    if not value:
        raise ProvisioningError("--python is required; pass an explicit Python 3.11 interpreter")
    resolved = shutil.which(value) or value
    return Path(resolved).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _link_or_copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ProvisioningError(f"measurement input is not a regular file: {source}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ProvisioningError(f"measurement destination is not a regular file: {destination}")
        if _sha256(destination) != _sha256(source):
            raise ProvisioningError(f"measurement destination already differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _measurement_paths(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    surface = root / "surface"
    environment = root / "environment"
    surface_paths = {name: surface / name for name in REQUIRED_PATH_NAMES}
    environment_paths = {name: environment / name for name in REQUIRED_ENVIRONMENT_PATHS}
    for path in (*surface_paths.values(), *environment_paths.values()):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return surface_paths, environment_paths


def _stage_measurement_inputs(
    *,
    paths,
    root: Path,
    wheelhouse_source: Path,
    model_source: Path,
    surface_paths: dict[str, Path],
    pkuseg_data_source: Path | None = None,
) -> tuple[MeasurementSession, object]:
    if wheelhouse_source.is_symlink() or not wheelhouse_source.is_dir():
        raise ProvisioningError("measurement wheelhouse must be a real directory")
    wheelhouse = surface_paths["cache"] / "wheelhouse"
    wheelhouse.mkdir(mode=0o700, parents=True, exist_ok=True)
    for artifact in sorted(wheelhouse_source.iterdir(), key=lambda item: item.name):
        if artifact.is_symlink() or not artifact.is_file():
            raise ProvisioningError(f"measurement wheelhouse contains a non-file: {artifact.name}")
        _link_or_copy(artifact, wheelhouse / artifact.name)

    lock_destination = surface_paths["cache"] / paths.lock_path.name
    _link_or_copy(paths.lock_path, lock_destination)
    lock_input = VerifiedInput(
        lock_destination.name,
        lock_destination,
        lock_destination.stat().st_size,
        _sha256(lock_destination),
    )
    package_inputs = [
        VerifiedInput(
            artifact.name,
            artifact,
            artifact.stat().st_size,
            _sha256(artifact),
        )
        for artifact in sorted(wheelhouse.iterdir(), key=lambda item: item.name)
    ]
    if len(package_inputs) != CURRENT_LOCK_REQUIREMENT_COUNT:
        raise ProvisioningError(
            f"measurement wheelhouse contains {len(package_inputs)} artifacts; "
            f"expected {CURRENT_LOCK_REQUIREMENT_COUNT}"
        )

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    model_records = manifest.get("sources", {}).get("model", {}).get("files", [])
    if not isinstance(model_records, list):
        raise ProvisioningError("manifest model file records are missing")
    model_by_name = {str(item.get("path")): item for item in model_records if isinstance(item, dict)}
    model_paths = tuple(model_by_name)
    if not model_paths or len(model_by_name) != len(model_records):
        raise ProvisioningError("manifest model files must be a nonempty unique declared set")
    model_inputs: list[VerifiedInput] = []
    for relative in model_paths:
        record = model_by_name[relative]
        source_entry = model_source / relative
        if source_entry.is_symlink():
            raise ProvisioningError(f"verified model input is a symlink: {relative}")
        source = source_entry.resolve()
        if not _regular_file(source):
            raise ProvisioningError(f"verified model input is missing: {source}")
        if source.stat().st_size != int(record["size_bytes"]) or _sha256(source).lower() != str(record["sha256"]).lower():
            raise ProvisioningError(f"verified model input identity differs: {relative}")
        destination = surface_paths["model"] / relative
        _link_or_copy(source, destination)
        model_inputs.append(
            VerifiedInput(
                relative,
                destination,
                destination.stat().st_size,
                _sha256(destination),
            )
        )

    worker_data_inputs: list[VerifiedInput] = []
    if pkuseg_data_source is not None:
        if pkuseg_data_source.is_symlink() or not _regular_file(pkuseg_data_source):
            raise ProvisioningError("verified worker data input is not a regular file")
        if _sha256(pkuseg_data_source).lower() != PKUSEG_DATA_SHA256:
            raise ProvisioningError(
                f"verified worker data input checksum differs: {PKUSEG_DATA_FILENAME}"
            )
        worker_data_dir = surface_paths["cache"] / "worker-data"
        worker_data_destination = worker_data_dir / PKUSEG_DATA_FILENAME
        _link_or_copy(pkuseg_data_source, worker_data_destination)
        worker_data_inputs.append(
            VerifiedInput(
                PKUSEG_DATA_FILENAME,
                worker_data_destination,
                worker_data_destination.stat().st_size,
                _sha256(worker_data_destination),
            )
        )

    session = MeasurementSession(
        disposable_root=root,
        paths=surface_paths,
        wheelhouse=wheelhouse,
        lock_input=lock_input,
        package_inputs=package_inputs,
        expected_package_count=CURRENT_LOCK_REQUIREMENT_COUNT,
        model_inputs=model_inputs,
        model_profile=str(
            manifest.get("sources", {}).get("model", {}).get(
                "profile",
                manifest.get("sources", {}).get("model", {}).get("variant", "v2"),
            )
        ),
        model_file_paths=model_paths,
        worker_data_inputs=worker_data_inputs,
    )
    sizes = {item.sha256: item.size_bytes for item in package_inputs}
    evidence = validate_complete_wheelhouse(
        root=root,
        wheelhouse=wheelhouse,
        lock_path=lock_destination,
        expected_lock_sha256=CURRENT_LOCK_SHA256,
        expected_artifact_sizes=sizes,
        expected_requirement_count=CURRENT_LOCK_REQUIREMENT_COUNT,
    )
    return session, evidence


def _run_measurement(args, paths, interpreter: Path) -> dict:
    requested_root = args.measurement_root.expanduser()
    if not requested_root.is_absolute() or ".." in requested_root.parts:
        raise ProvisioningError("measurement root must be absolute without traversal")
    if not requested_root.is_dir() or requested_root.is_symlink():
        raise ProvisioningError("measurement root must be an existing real directory")
    root = requested_root.resolve(strict=True)
    if requested_root != root:
        raise ProvisioningError("measurement root must use its canonical path")
    if hasattr(os, "geteuid") and root.stat().st_uid != os.geteuid():
        raise ProvisioningError("measurement root is not owned by the invoking user")
    marker = root / DISPOSABLE_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ProvisioningError("measurement root marker is missing")
    if marker.read_text(encoding="utf-8") != DISPOSABLE_MARKER_CONTENT:
        raise ProvisioningError("measurement root marker is invalid")
    if args.scenario == "model-baseline" and args.pkuseg_data is None:
        raise ProvisioningError(
            "model-baseline measurement requires the verified spacy_ontonotes.zip input"
        )
    surface_paths, environment_paths = _measurement_paths(root)
    runtime_environment = {
        "HOME": str(environment_paths["home"]),
        "XDG_DATA_HOME": str(environment_paths["xdg_data"]),
        "XDG_STATE_HOME": str(environment_paths["xdg_state"]),
        # Keep all installer-owned compatibility paths below observed XDG
        # roots.  The harness must account for every mutable path, including
        # worker-created caches, not only the receipt directories.
        "E2A_ROOT": str(environment_paths["xdg_data"] / "e2a"),
        "E2A_MODELS_DIR": str(environment_paths["xdg_data"] / "models"),
        "E2A_RUN_DIR": str(environment_paths["xdg_runtime"] / "e2a-run"),
    }
    measurement_paths = build_paths(
        runtime_dir=args.runtime_dir,
        repo_root=args.repo_root,
        environment=runtime_environment,
        manifest_path=paths.manifest_path,
    )
    session, evidence = _stage_measurement_inputs(
        paths=measurement_paths,
        root=root,
        wheelhouse_source=args.wheelhouse.expanduser().resolve(),
        model_source=(args.model_root or (measurement_paths.repo_root / "models/tts/chatterbox")).expanduser().resolve(),
        surface_paths=surface_paths,
        pkuseg_data_source=(
            args.pkuseg_data.expanduser().resolve()
            if args.pkuseg_data is not None
            else None
        ),
    )
    isolation = None
    if args.network_isolation_capability:
        isolation = {
            "status": "enforced",
            "enforced": True,
            "capability": args.network_isolation_capability,
            "explanation": "caller-provided VM or sandbox network isolation",
        }
    target = {
        "os": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "python": "3.11",
        "backend": "cpu",
    }
    return run_disposable_measurement(
        session=session,
        wheelhouse_evidence=evidence,
        environment_paths=environment_paths,
        runtime_paths=measurement_paths,
        interpreter=interpreter,
        run_id=args.run_id,
        scenario=args.scenario,
        target=target,
        environment_id=args.environment_id,
        filesystem_layout=args.filesystem_layout,
        worker_script=args.worker,
        network_isolation=isolation,
        sample_interval_seconds=args.sample_interval,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = build_paths(
            runtime_dir=args.runtime_dir,
            repo_root=args.repo_root,
            manifest_path=_manifest_path(args.runtime_dir, args.model),
        )
        if args.command == "verify-lock":
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            result = verify_lock(paths.lock_path, manifest)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        if args.command == "status":
            result = product_status(paths)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        if args.command == "model-preflight":
            result = model_preflight(paths)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        if args.command == "acquire-model":
            result = acquire_model(paths, worker_script=args.worker)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        interpreter = _interpreter(args.python_path)
        if args.command == "measure":
            result = _run_measurement(args, paths, interpreter)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["execution"]["status"] == "completed" else 2
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
