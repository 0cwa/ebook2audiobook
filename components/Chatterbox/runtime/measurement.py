"""Observation-only helpers for disposable Chatterbox capacity measurements.

This module does not provision, publish receipts, mutate the runtime manifest,
or report readiness.  A future measurement harness may call it around the
existing provisioning primitives after placing every input and output under a
root that has been explicitly marked disposable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contract_data import canonical_model_file_paths, normalize_model_variant
except ImportError:  # Direct execution support for the runtime CLI.
    from contract_data import canonical_model_file_paths, normalize_model_variant  # type: ignore[no-redef]


MEASUREMENT_SCHEMA = "ebook2audiobook.chatterbox-capacity-observations.v1"
DISPOSABLE_MARKER = ".chatterbox-measurement-root"
DISPOSABLE_MARKER_CONTENT = "observation-only-v1\n"
REQUIRED_PATH_NAMES = (
    "runtime",
    "cache",
    "model",
    "state",
    "run",
    "temporary",
    "activation",
)
CANONICAL_MODEL_FILE_PATHS = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
PHASE_EVENTS = frozenset(
    {
        "baseline",
        "candidate_reserved",
        "artifact_started",
        "artifact_completed",
        "verification_complete",
        "receipt_publishing",
        "receipt_published",
        "rollback_started",
        "cleanup_complete",
        "activation_started",
        "activation_complete",
        "retry_started",
        "concurrency_started",
        "concurrency_complete",
        "fault_injected",
    }
)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class MeasurementConfigurationError(ValueError):
    """The requested observation surface is not safely disposable."""


class InjectedMeasurementFault(RuntimeError):
    """A declared measurement fault was reached."""


@dataclass(frozen=True)
class VerifiedInput:
    """One local input whose bytes have an explicit expected identity."""

    name: str
    path: Path
    size_bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_traversal(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise MeasurementConfigurationError(f"{label} must be absolute")
    if ".." in path.parts:
        raise MeasurementConfigurationError(f"{label} must not contain traversal")


def _reject_symlink_components(path: Path, root: Path, label: str) -> None:
    current = root
    if stat.S_ISLNK(root.lstat().st_mode):
        raise MeasurementConfigurationError("disposable root must not be a symlink")
    for part in path.relative_to(root).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise MeasurementConfigurationError(f"{label} contains a symlink component")


def _guard_path(path: Path, root: Path, label: str, *, must_exist: bool = False) -> Path:
    candidate = Path(path)
    _reject_traversal(candidate, label)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MeasurementConfigurationError(f"{label} escapes the disposable root") from exc
    _reject_symlink_components(candidate, root, label)
    if must_exist and not candidate.exists():
        raise MeasurementConfigurationError(f"{label} does not exist")
    return candidate


def validate_disposable_paths(root: Path, paths: Mapping[str, Path]) -> dict[str, Path]:
    """Validate a complete, non-aliased path map without creating anything."""

    requested_root = Path(root)
    _reject_traversal(requested_root, "disposable root")
    if not requested_root.is_dir() or requested_root.is_symlink():
        raise MeasurementConfigurationError("disposable root must be an existing real directory")
    resolved_root = requested_root.resolve(strict=True)
    if requested_root != resolved_root:
        raise MeasurementConfigurationError("disposable root must use its canonical path")
    marker = resolved_root / DISPOSABLE_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise MeasurementConfigurationError("disposable root marker is missing")
    try:
        marker_content = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MeasurementConfigurationError("disposable root marker is unreadable") from exc
    if marker_content != DISPOSABLE_MARKER_CONTENT:
        raise MeasurementConfigurationError("disposable root marker is invalid")

    if set(paths) != set(REQUIRED_PATH_NAMES):
        missing = sorted(set(REQUIRED_PATH_NAMES) - set(paths))
        extra = sorted(set(paths) - set(REQUIRED_PATH_NAMES))
        raise MeasurementConfigurationError(f"disposable path map mismatch: missing={missing}, extra={extra}")
    guarded = {
        name: _guard_path(Path(paths[name]), resolved_root, f"{name} path")
        for name in REQUIRED_PATH_NAMES
    }
    resolved_values = [path.resolve(strict=False) for path in guarded.values()]
    if len(set(resolved_values)) != len(resolved_values):
        raise MeasurementConfigurationError("disposable paths must not alias one another")
    return guarded


def _verify_inputs(
    inputs: Sequence[VerifiedInput],
    *,
    root: Path,
    parent: Path,
    label: str,
    expected_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if not inputs:
        raise MeasurementConfigurationError(f"{label} inputs must not be empty")
    if expected_names is not None and {item.name for item in inputs} != set(expected_names):
        raise MeasurementConfigurationError(f"{label} inputs do not match the canonical file set")
    if len({item.name for item in inputs}) != len(inputs):
        raise MeasurementConfigurationError(f"{label} input names must be unique")
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_files: set[tuple[int, int]] = set()
    for item in inputs:
        path = _guard_path(Path(item.path), root, f"{label} input {item.name}", must_exist=True)
        try:
            path.relative_to(parent)
        except ValueError as exc:
            raise MeasurementConfigurationError(f"{label} input {item.name} is outside its declared namespace") from exc
        if path in seen_paths or not path.is_file() or path.is_symlink():
            raise MeasurementConfigurationError(f"{label} input {item.name} is not a unique regular file")
        metadata = path.stat()
        file_identity = (metadata.st_dev, metadata.st_ino)
        if file_identity in seen_files:
            raise MeasurementConfigurationError(f"{label} input {item.name} aliases another input")
        if not isinstance(item.size_bytes, int) or item.size_bytes < 0:
            raise MeasurementConfigurationError(f"{label} input {item.name} has an invalid size")
        if not isinstance(item.sha256, str) or not _HEX64.fullmatch(item.sha256):
            raise MeasurementConfigurationError(f"{label} input {item.name} has an invalid SHA-256")
        actual_size = path.stat().st_size
        allocated_size = metadata.st_blocks * 512
        if actual_size > 0 and allocated_size < actual_size:
            raise MeasurementConfigurationError(
                f"{label} input {item.name} is sparse or under-allocated"
            )
        actual_sha256 = _sha256_file(path)
        if actual_size != item.size_bytes or actual_sha256.lower() != item.sha256.lower():
            raise MeasurementConfigurationError(f"{label} input {item.name} failed verification")
        seen_paths.add(path)
        seen_files.add(file_identity)
        records.append(
            {
                "name": item.name,
                "path": str(path),
                "logical_bytes": actual_size,
                "allocated_bytes": allocated_size,
                "sha256": actual_sha256,
            }
        )
    return records


def offline_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an explicit no-index/offline environment without proxy inheritance."""

    allowed = {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    environment = {key: value for key, value in (base or {}).items() if key in allowed}
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "NO_PROXY": "*",
        }
    )
    return environment


def _path_usage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "entries": 0, "logical_bytes": 0, "allocated_bytes": 0}
    logical = 0
    allocated = 0
    entries = 0
    seen: set[tuple[int, int]] = set()
    pending = [path]
    while pending:
        current = pending.pop()
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise MeasurementConfigurationError(f"observation path became a symlink: {current}")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        entries += 1
        logical += metadata.st_size
        allocated += metadata.st_blocks * 512
        if stat.S_ISDIR(metadata.st_mode):
            with os.scandir(current) as iterator:
                pending.extend(Path(entry.path) for entry in iterator)
    return {
        "path": str(path),
        "exists": True,
        "entries": entries,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def _filesystem_observation(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    metadata = probe.stat()
    filesystem = os.statvfs(probe)
    fragment_size = filesystem.f_frsize or filesystem.f_bsize
    return {
        "filesystem_id": f"device:{metadata.st_dev}",
        "probe_path": str(probe),
        "block_size": filesystem.f_bsize,
        "fragment_size": fragment_size,
        "free_bytes": filesystem.f_bavail * fragment_size,
        "total_bytes": filesystem.f_blocks * fragment_size,
    }


def _validate_observation_detail(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = {str(key).lower() for key in value} & {"ok", "status", "ready", "readiness"}
        if forbidden:
            raise MeasurementConfigurationError(
                f"observation detail contains readiness-like fields: {sorted(forbidden)}"
            )
        for nested in value.values():
            _validate_observation_detail(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_observation_detail(nested)


class MeasurementSession:
    """Validated, read-only observation context for one disposable run."""

    def __init__(
        self,
        *,
        disposable_root: Path,
        paths: Mapping[str, Path],
        wheelhouse: Path,
        lock_input: VerifiedInput,
        package_inputs: Sequence[VerifiedInput],
        expected_package_count: int,
        model_inputs: Sequence[VerifiedInput],
        model_variant: str = "v2",
        worker_data_inputs: Sequence[VerifiedInput] = (),
        fault_plan: Mapping[tuple[str, int, str], str] | None = None,
    ) -> None:
        self.root = Path(disposable_root).resolve(strict=True)
        self.paths = validate_disposable_paths(Path(disposable_root), paths)
        self.wheelhouse = _guard_path(Path(wheelhouse), self.root, "wheelhouse", must_exist=True)
        if not self.wheelhouse.is_dir() or self.wheelhouse.is_symlink():
            raise MeasurementConfigurationError("wheelhouse must be a real directory")
        try:
            self.wheelhouse.relative_to(self.paths["cache"])
        except ValueError as exc:
            raise MeasurementConfigurationError("wheelhouse must be under the disposable cache path") from exc
        self._lock_input = lock_input
        self._package_inputs = tuple(package_inputs)
        self._model_inputs = tuple(model_inputs)
        normalized_variant = normalize_model_variant(model_variant)
        if normalized_variant is None:
            raise MeasurementConfigurationError(
                f"unsupported Chatterbox model variant for measurement: {model_variant!r}"
            )
        self.model_variant = normalized_variant
        self.expected_model_file_paths = canonical_model_file_paths(normalized_variant)
        self._worker_data_inputs = tuple(worker_data_inputs)
        self.expected_package_count = expected_package_count
        lock_records = _verify_inputs([lock_input], root=self.root, parent=self.paths["cache"], label="lock")
        if expected_package_count < 1 or len(package_inputs) != expected_package_count:
            raise MeasurementConfigurationError("package input count does not match the explicit expected count")
        package_records = _verify_inputs(
            package_inputs, root=self.root, parent=self.wheelhouse, label="package"
        )
        model_records = _verify_inputs(
            model_inputs,
            root=self.root,
            parent=self.paths["model"],
            label="model",
            expected_names=self.expected_model_file_paths,
        )
        worker_data_records: list[dict[str, Any]] = []
        if self._worker_data_inputs:
            worker_data_parent = self.paths["cache"] / "worker-data"
            worker_data_records = _verify_inputs(
                self._worker_data_inputs,
                root=self.root,
                parent=worker_data_parent,
                label="worker data",
            )
        self._inputs = {
            "lock": lock_records[0],
            "packages": package_records,
            "models": model_records,
            "worker_data": worker_data_records,
        }
        self._fault_plan = dict(fault_plan or {})
        unknown_fault_events = {key[0] for key in self._fault_plan} - PHASE_EVENTS
        if unknown_fault_events:
            raise MeasurementConfigurationError(f"fault plan has unknown events: {sorted(unknown_fault_events)}")
        self._events: list[dict[str, Any]] = []

    def revalidate_inputs(self, *, required_package_count: int | None = None) -> dict[str, Any]:
        """Re-read every declared input before an attempted execution boundary."""

        if required_package_count is not None and self.expected_package_count != required_package_count:
            raise MeasurementConfigurationError(
                f"execution requires exactly {required_package_count} locked package inputs"
            )
        if len(self._package_inputs) != self.expected_package_count:
            raise MeasurementConfigurationError("package input count changed after session creation")
        lock_records = _verify_inputs(
            [self._lock_input], root=self.root, parent=self.paths["cache"], label="lock"
        )
        package_records = _verify_inputs(
            self._package_inputs,
            root=self.root,
            parent=self.wheelhouse,
            label="package",
        )
        model_records = _verify_inputs(
            self._model_inputs,
            root=self.root,
            parent=self.paths["model"],
            label="model",
            expected_names=self.expected_model_file_paths,
        )
        worker_data_records: list[dict[str, Any]] = []
        if self._worker_data_inputs:
            worker_data_records = _verify_inputs(
                self._worker_data_inputs,
                root=self.root,
                parent=self.paths["cache"] / "worker-data",
                label="worker data",
            )
        self._inputs = {
            "lock": lock_records[0],
            "packages": package_records,
            "models": model_records,
            "worker_data": worker_data_records,
        }
        return json.loads(json.dumps(self._inputs))

    @property
    def offline_pip_arguments(self) -> list[str]:
        return [
            "--isolated",
            "--no-index",
            "--find-links",
            str(self.wheelhouse),
            "--require-hashes",
            "-r",
            self._inputs["lock"]["path"],
        ]

    def record_event(
        self,
        name: str,
        *,
        attempt: int = 1,
        participant: str = "main",
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one named observation checkpoint and apply a declared fault."""

        if name not in PHASE_EVENTS:
            raise MeasurementConfigurationError(f"unknown phase event: {name}")
        if attempt < 1 or not participant:
            raise MeasurementConfigurationError("attempt and participant must identify an observation lane")
        details = dict(detail or {})
        _validate_observation_detail(details)
        json.dumps(details)
        path_observations = {key: _path_usage(path) for key, path in self.paths.items()}
        filesystems: dict[str, dict[str, Any]] = {}
        for path in self.paths.values():
            observation = _filesystem_observation(path)
            filesystems.setdefault(observation["filesystem_id"], observation)
        event = {
            "name": name,
            "timestamp_ns": time.time_ns(),
            "attempt": attempt,
            "participant": participant,
            "detail": details,
            "paths": path_observations,
            "filesystems": filesystems,
        }
        self._events.append(event)
        fault = self._fault_plan.get((name, attempt, participant))
        if fault is not None:
            fault_event = {
                **event,
                "name": "fault_injected",
                "detail": {"at_event": name, "label": str(fault)},
            }
            self._events.append(fault_event)
            raise InjectedMeasurementFault(str(fault))
        return json.loads(json.dumps(event))

    def record_retry(self, *, attempt: int, reason: str, participant: str = "main") -> dict[str, Any]:
        return self.record_event("retry_started", attempt=attempt, participant=participant, detail={"reason": reason})

    def record_concurrency(self, participant: str, *, started: bool, attempt: int = 1) -> dict[str, Any]:
        event = "concurrency_started" if started else "concurrency_complete"
        return self.record_event(event, attempt=attempt, participant=participant)

    def observations(self) -> dict[str, Any]:
        """Return evidence only; this schema intentionally has no readiness fields."""

        report = {
            "schema": MEASUREMENT_SCHEMA,
            "mode": "observation_only",
            "disposable_root": str(self.root),
            "path_map": {name: str(path) for name, path in self.paths.items()},
            "offline": {
                "public_network_authorized": False,
                "pip_arguments": self.offline_pip_arguments,
                "environment": offline_environment(),
            },
            "verified_inputs": self._inputs,
            "events": self._events,
        }
        return json.loads(json.dumps(report))
