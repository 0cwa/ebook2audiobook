"""Bounded execution harness for Chatterbox capacity measurement.

The report-only helpers remain deliberately inert.  The disposable execution
entry point below is separate and requires a marked root plus verified local
inputs; it may publish receipts only inside that root and never changes the
production manifest or host readiness state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import threading
import time
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .measurement import (
    DISPOSABLE_MARKER,
    DISPOSABLE_MARKER_CONTENT,
    MeasurementConfigurationError,
    MeasurementSession,
    offline_environment,
)
from .contract_data import PKUSEG_DATA_FILENAME, PKUSEG_DATA_SHA256


HARNESS_SCHEMA = "ebook2audiobook.chatterbox-capacity-harness.v1"
AGGREGATE_SCHEMA = "ebook2audiobook.chatterbox-capacity-aggregate.v1"
CURRENT_LOCK_REQUIREMENT_COUNT = 107
CURRENT_LOCK_SHA256 = "b615b428258c69771b52da7310d954040aa5340efbb5b7f988e598347c56206d"
INVENTORY_DIGEST_CHUNK_BYTES = 1024 * 1024
REQUIRED_ENVIRONMENT_PATHS = (
    "home",
    "xdg_cache",
    "xdg_data",
    "xdg_state",
    "xdg_runtime",
    "tmpdir",
    "python_cache",
    "pip_cache",
)
REQUIRED_OBSERVATION_PATHS = tuple(
    [f"surface.{name}" for name in (
        "runtime", "cache", "model", "state", "run", "temporary", "activation"
    )]
    + [f"environment.{name}" for name in REQUIRED_ENVIRONMENT_PATHS]
)
CANONICAL_SCENARIOS = (
    "runtime-baseline",
    "model-baseline",
)
CANONICAL_FILESYSTEM_LAYOUTS = (
    "same-filesystem",
    "split-filesystem",
)
_RUNTIME_MUTABLE_PATH_FIELDS = (
    "data_home",
    "state_home",
    "e2a_root",
    "models_dir",
    "run_dir",
    "model_namespace",
    "legacy_model_cache",
    "model_objects_dir",
    "run_namespace",
    "state_dir",
    "env_base",
    "runtime_objects_dir",
    "legacy_environment",
    "environment",
    "runtime_receipt_path",
    "model_receipt_path",
    "activation_receipt_path",
    "install_lock_path",
    "model_install_lock_path",
    "result_path",
)
REQUIRED_TARGET_METADATA = frozenset({"os", "architecture", "python", "backend"})
REQUIRED_FILESYSTEM_METADATA = frozenset(
    {
        "filesystem_id",
        "device",
        "block_size",
        "fragment_size",
        "total_bytes",
        "source",
        "mount_point",
        "filesystem_type",
        "mount_options",
        "super_options",
        "quota_indicators",
        "cow_indicators",
        "compression_indicators",
        "reflink_support",
        "overlay",
    }
)
TRANSACTION_KINDS = frozenset(
    {
        "runtime",
        "model",
        "activation",
        "rollback",
        "retry",
        "runtime_concurrency",
        "model_concurrency",
    }
)
_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)\s+--hash=sha256:([0-9a-fA-F]{64})$"
)
_NORMALIZE_PROJECT = re.compile(r"[-_.]+")
@dataclass(frozen=True)
class LockRequirement:
    project: str
    version: str
    sha256: str


@dataclass(frozen=True)
class WheelhouseEvidence:
    lock_path: Path
    lock_sha256: str
    requirements: tuple[LockRequirement, ...]
    artifacts: tuple[Mapping[str, Any], ...]

    @property
    def pip_arguments(self) -> list[str]:
        wheelhouse = Path(str(self.artifacts[0]["path"])).parent
        return [
            "--isolated",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--cache-dir",
            "{PIP_CACHE_DIR}",
            "--require-hashes",
            "-r",
            str(self.lock_path),
        ]


@dataclass(frozen=True)
class TransactionInvocation:
    """Explicit input supplied to a caller-owned transaction hook."""

    kind: str
    attempt: int
    participant: str
    disposable_root: Path
    paths: Mapping[str, Path]
    environment: Mapping[str, str]
    pip_arguments: Sequence[str]
    checkpoint: Callable[[str, Mapping[str, Any] | None], Mapping[str, Any]]


TransactionHook = Callable[[TransactionInvocation], Mapping[str, Any] | None]


@dataclass(frozen=True)
class TransactionHooks:
    """Opt-in interfaces; absent hooks are fail-closed and never inferred."""

    runtime: TransactionHook | None = None
    model: TransactionHook | None = None
    activation: TransactionHook | None = None
    rollback: TransactionHook | None = None
    retry: TransactionHook | None = None
    runtime_concurrency: TransactionHook | None = None
    model_concurrency: TransactionHook | None = None

    def get(self, kind: str) -> TransactionHook:
        if kind not in TRANSACTION_KINDS:
            raise MeasurementConfigurationError(f"unknown transaction kind: {kind}")
        hook = getattr(self, kind)
        if hook is None:
            raise MeasurementConfigurationError(f"transaction hook is not configured: {kind}")
        return hook


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_project(value: str) -> str:
    return _NORMALIZE_PROJECT.sub("-", value).lower()


def _validate_marked_root(root: Path) -> Path:
    requested = Path(root)
    if not requested.is_absolute() or ".." in requested.parts:
        raise MeasurementConfigurationError("disposable root must be absolute without traversal")
    if not requested.is_dir() or requested.is_symlink():
        raise MeasurementConfigurationError("disposable root must be an existing real directory")
    resolved = requested.resolve(strict=True)
    if requested != resolved:
        raise MeasurementConfigurationError("disposable root must use its canonical path")
    marker = resolved / DISPOSABLE_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise MeasurementConfigurationError("disposable root marker is missing")
    try:
        content = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MeasurementConfigurationError("disposable root marker is unreadable") from exc
    if content != DISPOSABLE_MARKER_CONTENT:
        raise MeasurementConfigurationError("disposable root marker is invalid")
    return resolved


def _guard_under_root(
    path: Path,
    root: Path,
    label: str,
    *,
    must_exist: bool = False,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise MeasurementConfigurationError(f"{label} must be absolute without traversal")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MeasurementConfigurationError(f"{label} escapes the disposable root") from exc
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise MeasurementConfigurationError(f"{label} contains a symlink component")
    if must_exist and not candidate.exists():
        raise MeasurementConfigurationError(f"{label} does not exist")
    return candidate


def _validated_environment_paths(root: Path, paths: Mapping[str, Path]) -> dict[str, Path]:
    resolved_root = _validate_marked_root(Path(root))
    if set(paths) != set(REQUIRED_ENVIRONMENT_PATHS):
        missing = sorted(set(REQUIRED_ENVIRONMENT_PATHS) - set(paths))
        extra = sorted(set(paths) - set(REQUIRED_ENVIRONMENT_PATHS))
        raise MeasurementConfigurationError(
            f"environment path map mismatch: missing={missing}, extra={extra}"
        )
    guarded = {
        name: _guard_under_root(Path(paths[name]), resolved_root, f"{name} path")
        for name in REQUIRED_ENVIRONMENT_PATHS
    }
    resolved_values = [path.resolve(strict=False) for path in guarded.values()]
    if len(set(resolved_values)) != len(resolved_values):
        raise MeasurementConfigurationError("environment paths must not alias one another")
    return guarded


def _validate_runtime_observation_surface(
    runtime_paths: Any,
    *,
    root: Path,
    observed_paths: Mapping[str, Path],
) -> None:
    """Require every mutable runtime path to be inside an observed owned path."""

    observed_roots = tuple(Path(path).resolve(strict=False) for path in observed_paths.values())
    for field in _RUNTIME_MUTABLE_PATH_FIELDS:
        value = getattr(runtime_paths, field, None)
        if value is None:
            continue
        candidate = _guard_under_root(Path(value), root, f"runtime {field}")
        if not any(
            candidate == observed or observed in candidate.parents
            for observed in observed_roots
        ):
            raise MeasurementConfigurationError(
                f"runtime {field} is outside the observed disposable paths"
            )


def build_disposable_environment(
    root: Path,
    paths: Mapping[str, Path],
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create a disposable environment map; this alone is not network isolation."""

    guarded = _validated_environment_paths(root, paths)
    for path in guarded.values():
        path.mkdir(parents=True, exist_ok=True)
        _guard_under_root(path, _validate_marked_root(Path(root)), "created environment path", must_exist=True)

    environment = offline_environment(base)
    environment.update(
        {
            "HOME": str(guarded["home"]),
            "XDG_CACHE_HOME": str(guarded["xdg_cache"]),
            "XDG_DATA_HOME": str(guarded["xdg_data"]),
            "XDG_STATE_HOME": str(guarded["xdg_state"]),
            "XDG_RUNTIME_DIR": str(guarded["xdg_runtime"]),
            "TMPDIR": str(guarded["tmpdir"]),
            "TMP": str(guarded["tmpdir"]),
            "TEMP": str(guarded["tmpdir"]),
            "PYTHONPYCACHEPREFIX": str(guarded["python_cache"]),
            "PYTHONNOUSERSITE": "1",
            "PIP_CACHE_DIR": str(guarded["pip_cache"]),
            "PIP_CONFIG_FILE": os.devnull,
        }
    )
    return environment


def parse_hashed_lock(lock_path: Path) -> tuple[LockRequirement, ...]:
    """Parse the strict current lock shape without honoring index directives."""

    try:
        lines = Path(lock_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise MeasurementConfigurationError("lock is unreadable") from exc
    if "--require-hashes" not in {line.strip() for line in lines}:
        raise MeasurementConfigurationError("lock must enable --require-hashes")
    requirements: list[LockRequirement] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            raise MeasurementConfigurationError(f"unsupported lock requirement: {line}")
        project, version, digest = match.groups()
        normalized = _normal_project(project)
        if normalized in seen:
            raise MeasurementConfigurationError(f"duplicate locked project: {project}")
        seen.add(normalized)
        requirements.append(LockRequirement(normalized, version, digest.lower()))
    if not requirements:
        raise MeasurementConfigurationError("lock contains no requirements")
    return tuple(requirements)


def _artifact_identity(filename: str, requirement: LockRequirement) -> bool:
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        return (
            len(parts) >= 5
            and _normal_project(parts[0]) == requirement.project
            and parts[1].replace("_", "-") == requirement.version.replace("_", "-")
        )
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".zip"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            expected_suffix = f"-{requirement.version}"
            return stem.endswith(expected_suffix) and _normal_project(
                stem[: -len(expected_suffix)]
            ) == requirement.project
    return False


def _verify_non_sparse_regular(path: Path, label: str) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise MeasurementConfigurationError(f"{label} is not a regular file")
    metadata = path.stat()
    allocated = metadata.st_blocks * 512
    if metadata.st_size > 0 and allocated < metadata.st_size:
        raise MeasurementConfigurationError(f"{label} is sparse or under-allocated")
    return metadata


def validate_complete_wheelhouse(
    *,
    root: Path,
    wheelhouse: Path,
    lock_path: Path,
    expected_lock_sha256: str,
    expected_artifact_sizes: Mapping[str, int],
    expected_requirement_count: int = 107,
) -> WheelhouseEvidence:
    """Prove one local archive for every locked requirement and no foreign files."""

    resolved_root = _validate_marked_root(Path(root))
    guarded_wheelhouse = _guard_under_root(
        Path(wheelhouse), resolved_root, "wheelhouse", must_exist=True
    )
    guarded_lock = _guard_under_root(Path(lock_path), resolved_root, "lock", must_exist=True)
    if not guarded_wheelhouse.is_dir() or guarded_wheelhouse.is_symlink():
        raise MeasurementConfigurationError("wheelhouse must be a real directory")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_lock_sha256):
        raise MeasurementConfigurationError("expected lock SHA-256 is invalid")
    actual_lock_sha256 = _sha256_file(guarded_lock)
    if actual_lock_sha256 != expected_lock_sha256.lower():
        raise MeasurementConfigurationError("lock identity does not match the expected SHA-256")
    requirements = parse_hashed_lock(guarded_lock)
    if len(requirements) != expected_requirement_count:
        raise MeasurementConfigurationError(
            f"lock requirement count is {len(requirements)}, expected {expected_requirement_count}"
        )
    locked_hashes = {requirement.sha256 for requirement in requirements}
    normalized_sizes = {str(digest).lower(): size for digest, size in expected_artifact_sizes.items()}
    if set(normalized_sizes) != locked_hashes:
        missing = sorted(locked_hashes - set(normalized_sizes))
        foreign = sorted(set(normalized_sizes) - locked_hashes)
        raise MeasurementConfigurationError(
            f"artifact size map does not match the lock: missing={missing}, foreign={foreign}"
        )
    if any(not isinstance(size, int) or size <= 0 for size in normalized_sizes.values()):
        raise MeasurementConfigurationError("artifact size map contains an invalid size")

    entries = sorted(guarded_wheelhouse.iterdir(), key=lambda item: item.name)
    if len(entries) != len(requirements):
        raise MeasurementConfigurationError(
            f"wheelhouse artifact count is {len(entries)}, expected {len(requirements)}"
        )
    by_hash: dict[str, LockRequirement] = {}
    for requirement in requirements:
        if requirement.sha256 in by_hash:
            raise MeasurementConfigurationError("lock contains an ambiguous artifact hash")
        by_hash[requirement.sha256] = requirement

    records: list[Mapping[str, Any]] = []
    matched: set[str] = set()
    for entry in entries:
        guarded = _guard_under_root(entry, resolved_root, f"wheelhouse artifact {entry.name}", must_exist=True)
        metadata = _verify_non_sparse_regular(guarded, f"wheelhouse artifact {entry.name}")
        digest = _sha256_file(guarded)
        requirement = by_hash.get(digest)
        if requirement is None:
            raise MeasurementConfigurationError(f"foreign wheelhouse artifact: {entry.name}")
        if requirement.project in matched:
            raise MeasurementConfigurationError(f"duplicate wheelhouse project: {requirement.project}")
        if not _artifact_identity(entry.name, requirement):
            raise MeasurementConfigurationError(
                f"wheelhouse filename does not match locked project/version: {entry.name}"
            )
        if metadata.st_size != normalized_sizes[digest]:
            raise MeasurementConfigurationError(
                f"wheelhouse artifact size does not match explicit evidence: {entry.name}"
            )
        if entry.suffix in {".whl", ".zip"} and not zipfile.is_zipfile(entry):
            raise MeasurementConfigurationError(f"wheelhouse artifact is not a valid ZIP archive: {entry.name}")
        matched.add(requirement.project)
        records.append(
            {
                "project": requirement.project,
                "version": requirement.version,
                "path": str(entry),
                "logical_bytes": metadata.st_size,
                "allocated_bytes": metadata.st_blocks * 512,
                "sha256": digest,
            }
        )
    if matched != {requirement.project for requirement in requirements}:
        missing = sorted({requirement.project for requirement in requirements} - matched)
        raise MeasurementConfigurationError(f"wheelhouse is incomplete: {missing}")
    return WheelhouseEvidence(
        guarded_lock,
        actual_lock_sha256,
        requirements,
        tuple(records),
    )


def materialize_pip_arguments(evidence: WheelhouseEvidence, pip_cache: Path, root: Path) -> list[str]:
    guarded_cache = _guard_under_root(
        Path(pip_cache), _validate_marked_root(Path(root)), "pip cache", must_exist=True
    )
    return [str(guarded_cache) if item == "{PIP_CACHE_DIR}" else item for item in evidence.pip_arguments]


def _revalidate_execution_inputs(
    session: MeasurementSession,
    evidence: WheelhouseEvidence,
) -> WheelhouseEvidence:
    """Bind an attempted execution to the current lock and the session's exact inputs."""

    session_inputs = session.revalidate_inputs(
        required_package_count=CURRENT_LOCK_REQUIREMENT_COUNT
    )
    if evidence.lock_sha256 != CURRENT_LOCK_SHA256:
        raise MeasurementConfigurationError("execution lock does not match the current 107-entry lock")
    expected_sizes = {
        str(record["sha256"]): int(record["logical_bytes"])
        for record in evidence.artifacts
    }
    refreshed = validate_complete_wheelhouse(
        root=session.root,
        wheelhouse=session.wheelhouse,
        lock_path=evidence.lock_path,
        expected_lock_sha256=CURRENT_LOCK_SHA256,
        expected_artifact_sizes=expected_sizes,
        expected_requirement_count=CURRENT_LOCK_REQUIREMENT_COUNT,
    )
    if Path(session_inputs["lock"]["path"]) != refreshed.lock_path:
        raise MeasurementConfigurationError("session lock and validated wheelhouse lock differ")
    session_packages = {
        (str(record["path"]), str(record["sha256"]), int(record["logical_bytes"]))
        for record in session_inputs["packages"]
    }
    wheelhouse_packages = {
        (str(record["path"]), str(record["sha256"]), int(record["logical_bytes"]))
        for record in refreshed.artifacts
    }
    if session_packages != wheelhouse_packages:
        raise MeasurementConfigurationError(
            "session package inputs do not match the complete validated wheelhouse"
        )
    return refreshed


def _path_usage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "entries": 0, "logical_bytes": 0, "allocated_bytes": 0}
    pending = [path]
    seen: set[tuple[int, int]] = set()
    entries = logical = allocated = 0
    while pending:
        current = pending.pop()
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # Installation can replace temporary bytecode/cache entries while
            # the sampler is walking the owned tree.  The next observation
            # will account for the replacement; a vanished entry is not a
            # measurement failure.
            continue
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        entries += 1
        logical += metadata.st_size
        allocated += metadata.st_blocks * 512
        if stat.S_ISDIR(metadata.st_mode):
            try:
                with os.scandir(current) as iterator:
                    pending.extend(Path(item.path) for item in iterator)
            except (FileNotFoundError, NotADirectoryError):
                # A concurrently replaced directory may disappear or become
                # a file after lstat; retain the entries already observed.
                continue
    return {
        "path": str(path),
        "entries": entries,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def _stage_pkuseg_data(
    *,
    source: Path,
    destination: Path,
    root: Path,
) -> dict[str, Any]:
    """Copy and safely extract the verified tokenizer archive below root."""

    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise MeasurementConfigurationError("verified tokenizer archive is not a regular file")
    if _sha256_file(source).lower() != PKUSEG_DATA_SHA256:
        raise MeasurementConfigurationError("verified tokenizer archive checksum does not match")
    destination = _guard_under_root(destination, root, "PKUSEG_HOME")
    if destination.exists() or destination.is_symlink():
        raise MeasurementConfigurationError("PKUSEG_HOME already exists")
    destination.mkdir(mode=0o700, parents=True)
    archive_destination = _guard_under_root(
        destination / PKUSEG_DATA_FILENAME,
        root,
        "tokenizer archive destination",
    )
    shutil.copyfile(source, archive_destination)
    with archive_destination.open("rb") as stream:
        os.fsync(stream.fileno())

    model_directory = _guard_under_root(
        destination / "spacy_ontonotes",
        root,
        "tokenizer model directory",
    )
    model_directory.mkdir(mode=0o700)
    extracted: list[str] = []
    try:
        with zipfile.ZipFile(archive_destination) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if (
                    not member.filename
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise MeasurementConfigurationError(
                        f"tokenizer archive contains an unsafe member: {member.filename!r}"
                    )
                target = _guard_under_root(
                    model_directory.joinpath(*relative.parts),
                    root,
                    "tokenizer extracted file",
                )
                if member.is_dir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with archive.open(member) as source_stream, target.open("xb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
                extracted.append(relative.as_posix())
    except zipfile.BadZipFile as exc:
        raise MeasurementConfigurationError("verified tokenizer archive is not a valid zip file") from exc
    return {
        "archive": {
            "path": str(archive_destination),
            "sha256": PKUSEG_DATA_SHA256,
            "size_bytes": archive_destination.stat().st_size,
        },
        "pkuseg_home": str(destination),
        "extracted": sorted(extracted),
    }


def _unescape_mount(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _mount_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": None,
        "mount_point": None,
        "filesystem_type": None,
        "mount_options": [],
        "super_options": [],
        "metadata_source": "unavailable",
    }
    mountinfo = Path("/proc/self/mountinfo")
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return result
    best: tuple[int, dict[str, Any]] | None = None
    resolved = path.resolve(strict=False)
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) < 6 or len(right) < 3:
            continue
        mount_point = Path(_unescape_mount(left[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate = {
            "source": _unescape_mount(right[1]),
            "mount_point": str(mount_point),
            "filesystem_type": right[0],
            "mount_options": sorted(set(left[5].split(","))),
            "super_options": sorted(set(right[2].split(","))),
            "metadata_source": str(mountinfo),
        }
        length = len(mount_point.parts)
        if best is None or length > best[0]:
            best = (length, candidate)
    return best[1] if best is not None else result


def _filesystem_probe(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    metadata = probe.stat()
    filesystem = os.statvfs(probe)
    fragment = filesystem.f_frsize or filesystem.f_bsize
    mount = _mount_metadata(probe)
    options = set(mount["mount_options"]) | set(mount["super_options"])
    return {
        "filesystem_id": f"device:{metadata.st_dev}",
        "probe_path": str(probe),
        "device": metadata.st_dev,
        "block_size": filesystem.f_bsize,
        "fragment_size": fragment,
        "free_bytes": filesystem.f_bavail * fragment,
        "total_bytes": filesystem.f_blocks * fragment,
        "source": mount["source"],
        "mount_point": mount["mount_point"],
        "filesystem_type": mount["filesystem_type"],
        "mount_options": mount["mount_options"],
        "super_options": mount["super_options"],
        "metadata_source": mount["metadata_source"],
        "quota_indicators": sorted(option for option in options if "quota" in option),
        "cow_indicators": sorted(option for option in options if option in {"cow", "nodatacow"}),
        "compression_indicators": sorted(option for option in options if "compress" in option),
        "reflink_support": "unverified",
        "overlay": mount["filesystem_type"] == "overlay",
    }


def capture_snapshot(root: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    resolved_root = _validate_marked_root(Path(root))
    guarded = {
        name: _guard_under_root(Path(path), resolved_root, f"observed {name} path")
        for name, path in paths.items()
    }
    path_records = {name: _path_usage(path) for name, path in guarded.items()}
    filesystems: dict[str, dict[str, Any]] = {}
    path_filesystems: dict[str, str] = {}
    for path in guarded.values():
        probe = _filesystem_probe(path)
        filesystems.setdefault(probe["filesystem_id"], probe)
    for name, path in guarded.items():
        path_filesystems[name] = _filesystem_probe(path)["filesystem_id"]
    return {
        "timestamp_ns": time.time_ns(),
        "paths": path_records,
        "path_filesystems": path_filesystems,
        "filesystems": filesystems,
        "owned_logical_bytes": sum(record["logical_bytes"] for record in path_records.values()),
        "owned_allocated_bytes": sum(record["allocated_bytes"] for record in path_records.values()),
    }


def _combined_observation_paths(
    session: MeasurementSession,
    environment_paths: Mapping[str, Path],
) -> dict[str, Path]:
    guarded_environment = _validated_environment_paths(session.root, environment_paths)
    combined = {
        **{f"surface.{name}": path for name, path in session.paths.items()},
        **{f"environment.{name}": path for name, path in guarded_environment.items()},
    }
    if set(combined) != set(REQUIRED_OBSERVATION_PATHS):
        raise MeasurementConfigurationError("combined observation path contract is incomplete")
    resolved = {name: path.resolve(strict=False) for name, path in combined.items()}
    for first_name, first_path in resolved.items():
        for second_name, second_path in resolved.items():
            if first_name >= second_name:
                continue
            if first_path == second_path:
                raise MeasurementConfigurationError(
                    f"observation paths alias: {first_name}, {second_name}"
                )
            try:
                first_path.relative_to(second_path)
            except ValueError:
                pass
            else:
                raise MeasurementConfigurationError(
                    f"observation paths overlap: {first_name}, {second_name}"
                )
            try:
                second_path.relative_to(first_path)
            except ValueError:
                pass
            else:
                raise MeasurementConfigurationError(
                    f"observation paths overlap: {first_name}, {second_name}"
                )
    return combined


def _inventory_fingerprint(path: Path, metadata: os.stat_result) -> tuple[str, str] | tuple[None, None]:
    """Return a memory-bounded, complete content digest for baseline comparison."""

    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        return "sha256-symlink-target-v1", hashlib.sha256(
            os.fsencode(target)
        ).hexdigest()
    if not stat.S_ISREG(metadata.st_mode):
        return None, None

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != identity:
            raise MeasurementConfigurationError(
                f"inventory entry changed while fingerprinting: {path}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(
                lambda: stream.read(INVENTORY_DIGEST_CHUNK_BYTES), b""
            ):
                digest.update(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != identity:
            raise MeasurementConfigurationError(
                f"inventory entry changed while fingerprinting: {path}"
            )
        return "sha256-full-v1", digest.hexdigest()
    finally:
        os.close(descriptor)


def capture_root_inventory(root: Path) -> dict[str, dict[str, Any]]:
    resolved_root = _validate_marked_root(root)
    records: dict[str, dict[str, Any]] = {}
    pending = [resolved_root]
    while pending:
        current = pending.pop()
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if current != resolved_root:
            relative = str(current.relative_to(resolved_root))
            fingerprint_method, fingerprint = _inventory_fingerprint(current, metadata)
            records[relative] = {
                "mode": stat.S_IFMT(metadata.st_mode),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "size": metadata.st_size,
                "fingerprint_method": fingerprint_method,
                "fingerprint": fingerprint,
            }
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            with os.scandir(current) as iterator:
                pending.extend(Path(entry.path) for entry in iterator)
    return records


def _is_owned_entry(candidate: Path, owned_roots: Sequence[Path], root: Path) -> bool:
    for owned in owned_roots:
        try:
            candidate.relative_to(owned)
            return True
        except ValueError:
            pass
    return False


def _is_expected_owned_directory_change(
    *,
    root: Path,
    relative: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    owned_roots: Sequence[Path],
) -> bool:
    """Recognize disposable directory identity changes without hiding file edits.

    Overlay and copy-on-write filesystems may replace a directory inode when a
    child is written.  The measurement surface is disposable, so that
    identity change is not residue.  Regular-file identity or content changes
    remain failures and are reported as changed baseline entries.
    """

    if before.get("mode") != stat.S_IFDIR or after.get("mode") != stat.S_IFDIR:
        return False
    candidate = root / relative
    if candidate == root:
        return False
    return any(
        candidate == owned
        or owned in candidate.parents
        or candidate in owned.parents
        for owned in owned_roots
    )


_OBSERVED_CLEANUP_STATES = frozenset({"complete", "observed"})
_UNOBSERVED_CLEANUP_STATES = frozenset(
    {"not_observed", "not_observed_execution_refused", "not_observed_unavailable"}
)


def _optional_cleanup_observation_is_clean(
    cleanup: Mapping[str, Any],
    *,
    observation_field: str,
    residue_field: str,
) -> bool:
    state = cleanup.get(observation_field)
    residue = cleanup.get(residue_field)
    if state in _OBSERVED_CLEANUP_STATES:
        return isinstance(residue, list) and not residue
    if state in _UNOBSERVED_CLEANUP_STATES:
        return residue is None
    return False


def _cleanup_contents_are_complete(cleanup: Mapping[str, Any]) -> bool:
    """Check the evidence required for a disposable measurement cleanup.

    Process enumeration and deleted-open-file inspection are useful when the
    host can provide them, but they are not prerequisites for an otherwise
    complete owned-filesystem cleanup.  If either observation is performed,
    any reported residue remains a hard failure.
    """

    if (
        cleanup.get("filesystem_complete") is not True
        or cleanup.get("filesystem_cleanup_status") != "complete"
        or cleanup.get("completion_scope")
        not in {"filesystem_and_worker", "filesystem_and_process"}
        or cleanup.get("worker_observation") not in _OBSERVED_CLEANUP_STATES
    ):
        return False
    for field in (
        "filesystem_residue",
        "unowned_residue",
        "missing_baseline_entries",
        "changed_baseline_entries",
        "errors",
    ):
        value = cleanup.get(field)
        if not isinstance(value, list) or value:
            return False
    if not _optional_cleanup_observation_is_clean(
        cleanup,
        observation_field="process_observation",
        residue_field="process_residue",
    ):
        return False
    if not _optional_cleanup_observation_is_clean(
        cleanup,
        observation_field="deleted_open_file_observation",
        residue_field="deleted_open_files",
    ):
        return False
    worker_residue = cleanup.get("worker_residue")
    return worker_residue is None or worker_residue == []


def _cleanup_evidence_is_complete(cleanup: Mapping[str, Any]) -> bool:
    return cleanup.get("complete") is True and _cleanup_contents_are_complete(cleanup)


def _descriptor_cleanup_capability_error() -> str | None:
    missing: list[str] = []
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value == 0:
            missing.append(name)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    for operation, name in (
        (os.open, "os.open(dir_fd)"),
        (os.stat, "os.stat(dir_fd)"),
        (os.unlink, "os.unlink(dir_fd)"),
        (os.rmdir, "os.rmdir(dir_fd)"),
    ):
        if operation not in supports_dir_fd:
            missing.append(name)
    if os.stat not in getattr(os, "supports_follow_symlinks", ()):
        missing.append("os.stat(follow_symlinks=False)")
    if missing:
        return "descriptor-anchored no-follow cleanup unavailable: " + ", ".join(missing)
    return None


def _cleanup_directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_marked_cleanup_root(root: Path) -> int:
    expected = root.lstat()
    descriptor = os.open(root, _cleanup_directory_open_flags())
    try:
        opened = os.fstat(descriptor)
        expected_identity = (expected.st_dev, expected.st_ino, expected.st_mode)
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_mode)
        if expected_identity != opened_identity or not stat.S_ISDIR(opened.st_mode):
            raise MeasurementConfigurationError(
                "disposable root changed before descriptor anchoring"
            )

        marker_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        marker_descriptor = os.open(
            DISPOSABLE_MARKER,
            marker_flags,
            dir_fd=descriptor,
        )
        try:
            marker_metadata = os.fstat(marker_descriptor)
            if not stat.S_ISREG(marker_metadata.st_mode):
                raise MeasurementConfigurationError(
                    "disposable root marker is not a regular file"
                )
            expected_content = DISPOSABLE_MARKER_CONTENT.encode("utf-8")
            marker_chunks: list[bytes] = []
            remaining = len(expected_content) + 1
            while remaining:
                chunk = os.read(marker_descriptor, remaining)
                if not chunk:
                    break
                marker_chunks.append(chunk)
                remaining -= len(chunk)
            marker_content = b"".join(marker_chunks)
            if marker_content != expected_content:
                raise MeasurementConfigurationError(
                    "disposable root marker is invalid through anchored descriptor"
                )
        finally:
            os.close(marker_descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _remove_inventory_entry(
    root_descriptor: int,
    relative: str,
    expected: Mapping[str, Any],
) -> None:
    relative_path = Path(relative)
    parts = relative_path.parts
    if (
        relative_path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise MeasurementConfigurationError("cleanup inventory path is invalid")

    parent_descriptor = root_descriptor
    opened_descriptors: list[int] = []
    try:
        for part in parts[:-1]:
            parent_descriptor = os.open(
                part,
                _cleanup_directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            opened_descriptors.append(parent_descriptor)

        final_name = parts[-1]
        metadata = os.stat(
            final_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        observed_identity = (
            stat.S_IFMT(metadata.st_mode),
            metadata.st_dev,
            metadata.st_ino,
        )
        expected_identity = (
            expected.get("mode"),
            expected.get("device"),
            expected.get("inode"),
        )
        if observed_identity != expected_identity:
            raise MeasurementConfigurationError(
                "cleanup entry changed after inventory"
            )

        if stat.S_ISDIR(metadata.st_mode):
            os.rmdir(final_name, dir_fd=parent_descriptor)
        elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            os.unlink(final_name, dir_fd=parent_descriptor)
        elif scenario == "model-baseline":
            raise MeasurementConfigurationError(
                "special filesystem entry requires review"
            )
    finally:
        for descriptor in reversed(opened_descriptors):
            os.close(descriptor)


def _cleanup_report(
    *,
    root: Path,
    owned_roots: Sequence[Path],
    baseline_inventory: Mapping[str, Mapping[str, Any]],
    removed: Sequence[str],
    errors: Sequence[Mapping[str, str]],
    descriptor_capability: str,
    refused: bool,
) -> dict[str, Any]:
    after = capture_root_inventory(root)
    filesystem_residue = sorted(set(after) - set(baseline_inventory))
    missing_baseline = sorted(set(baseline_inventory) - set(after))
    changed_baseline: list[str] = []
    expected_owned_directory_changes: list[str] = []
    for relative in sorted(set(baseline_inventory) & set(after)):
        before_record = dict(baseline_inventory[relative])
        after_record = dict(after[relative])
        comparable_fields = (
            "mode",
            "device",
            "inode",
            "fingerprint_method",
            "fingerprint",
        )
        if any(
            before_record.get(field) != after_record.get(field)
            for field in comparable_fields
        ):
            if _is_expected_owned_directory_change(
                root=root,
                relative=relative,
                before=before_record,
                after=after_record,
                owned_roots=owned_roots,
            ):
                expected_owned_directory_changes.append(relative)
            else:
                changed_baseline.append(relative)
        elif (
            before_record.get("mode") != stat.S_IFDIR
            and before_record.get("size") != after_record.get("size")
        ):
            changed_baseline.append(relative)
    filesystem_complete = (
        not filesystem_residue
        and not missing_baseline
        and not changed_baseline
        and not errors
    )
    if refused:
        filesystem_cleanup_status = "refused"
    elif filesystem_complete:
        filesystem_cleanup_status = "complete"
    else:
        filesystem_cleanup_status = "incomplete"
    unowned_residue = sorted(
        relative
        for relative in filesystem_residue
        if not _is_owned_entry(root / relative, owned_roots, root)
    )
    return {
        "complete": False,
        "completion_scope": "filesystem_only",
        "filesystem_complete": filesystem_complete,
        "filesystem_cleanup_status": filesystem_cleanup_status,
        "deletion_strategy": "descriptor_anchored_no_follow",
        "descriptor_cleanup_capability": descriptor_capability,
        "removed": sorted(removed),
        "errors": [dict(error) for error in errors],
        "filesystem_residue": filesystem_residue,
        "unowned_residue": unowned_residue,
        "missing_baseline_entries": missing_baseline,
        "changed_baseline_entries": changed_baseline,
        "expected_owned_directory_changes": expected_owned_directory_changes,
        "process_residue": None,
        "worker_residue": None,
        "deleted_open_files": None,
        "process_observation": "not_observed_execution_refused",
        "worker_observation": "not_observed_execution_refused",
        "deleted_open_file_observation": "not_observed_execution_refused",
    }


def cleanup_owned_artifacts(
    *,
    root: Path,
    owned_paths: Mapping[str, Path],
    baseline_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Remove owned additions through root-anchored, no-follow descriptors only."""

    resolved_root = _validate_marked_root(root)
    guarded_roots = [
        _guard_under_root(Path(path), resolved_root, f"cleanup {name} path")
        for name, path in owned_paths.items()
    ]
    capability_error = _descriptor_cleanup_capability_error()
    if capability_error is not None:
        return _cleanup_report(
            root=resolved_root,
            owned_roots=guarded_roots,
            baseline_inventory=baseline_inventory,
            removed=[],
            errors=[{"path": ".", "error": capability_error}],
            descriptor_capability="unavailable",
            refused=True,
        )

    removed: list[str] = []
    errors: list[dict[str, str]] = []
    try:
        root_descriptor = _open_marked_cleanup_root(resolved_root)
    except (OSError, MeasurementConfigurationError) as exc:
        return _cleanup_report(
            root=resolved_root,
            owned_roots=guarded_roots,
            baseline_inventory=baseline_inventory,
            removed=removed,
            errors=[{"path": ".", "error": str(exc)}],
            descriptor_capability="available",
            refused=True,
        )

    try:
        current = capture_root_inventory(resolved_root)
        new_entries = sorted(
            set(current) - set(baseline_inventory),
            key=lambda value: (len(Path(value).parts), value),
            reverse=True,
        )
        for relative in new_entries:
            candidate = resolved_root / relative
            if not _is_owned_entry(candidate, guarded_roots, resolved_root):
                continue
            try:
                _remove_inventory_entry(
                    root_descriptor,
                    relative,
                    current[relative],
                )
                removed.append(relative)
            except FileNotFoundError:
                continue
            except (OSError, MeasurementConfigurationError) as exc:
                errors.append({"path": relative, "error": str(exc)})
    finally:
        os.close(root_descriptor)

    return _cleanup_report(
        root=resolved_root,
        owned_roots=guarded_roots,
        baseline_inventory=baseline_inventory,
        removed=removed,
        errors=errors,
        descriptor_capability="available",
        refused=False,
    )


def _high_water(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(snapshots) < 2:
        raise MeasurementConfigurationError("a phase must contain at least two snapshots")
    stages = [snapshot.get("stage") for snapshot in snapshots]
    if any(not isinstance(stage, str) or not stage for stage in stages):
        raise MeasurementConfigurationError("every snapshot requires a stage identity")
    baseline = snapshots[0]
    path_names = set(baseline["paths"])
    if any(set(snapshot["paths"]) != path_names for snapshot in snapshots):
        raise MeasurementConfigurationError("snapshot path coverage changed during the phase")
    path_maxima: dict[str, dict[str, Any]] = {}
    for name in sorted(path_names):
        logical_values = [int(snapshot["paths"][name]["logical_bytes"]) for snapshot in snapshots]
        allocated_values = [int(snapshot["paths"][name]["allocated_bytes"]) for snapshot in snapshots]
        logical_maximum = max(logical_values)
        allocated_maximum = max(allocated_values)
        path_maxima[name] = {
            "logical_bytes": logical_maximum,
            "allocated_bytes": allocated_maximum,
            "observation_count": len(snapshots),
            "logical_value_classification": (
                "observed_zero" if logical_maximum == 0 else "observed_nonzero"
            ),
            "allocated_value_classification": (
                "observed_zero" if allocated_maximum == 0 else "observed_nonzero"
            ),
            "logical_maximum_stages": [
                stages[index]
                for index, value in enumerate(logical_values)
                if value == logical_maximum
            ],
            "allocated_maximum_stages": [
                stages[index]
                for index, value in enumerate(allocated_values)
                if value == allocated_maximum
            ],
        }
    filesystem_ids = set(baseline["filesystems"])
    if any(set(snapshot["filesystems"]) != filesystem_ids for snapshot in snapshots):
        raise MeasurementConfigurationError("snapshot filesystem coverage changed during the phase")
    filesystem_maxima: dict[str, Any] = {}
    for filesystem_id in sorted(filesystem_ids):
        baseline_free = int(baseline["filesystems"][filesystem_id]["free_bytes"])
        observations = [snapshot["filesystems"][filesystem_id] for snapshot in snapshots]
        minimum_free = min(item["free_bytes"] for item in observations)
        maximum_drop = max(0, baseline_free - minimum_free)
        filesystem_maxima[filesystem_id] = {
            "baseline_free_bytes": baseline_free,
            "minimum_free_bytes": minimum_free,
            "maximum_free_space_drop_bytes": maximum_drop,
            "observation_count": len(observations),
            "value_classification": (
                "observed_zero" if maximum_drop == 0 else "observed_nonzero"
            ),
            "minimum_free_stages": [
                stages[index]
                for index, item in enumerate(observations)
                if item["free_bytes"] == minimum_free
            ],
            "metadata": observations[-1],
        }
    return {
        "owned_logical_bytes": max(snapshot["owned_logical_bytes"] for snapshot in snapshots),
        "owned_allocated_bytes": max(snapshot["owned_allocated_bytes"] for snapshot in snapshots),
        "paths": path_maxima,
        "filesystems": filesystem_maxima,
    }


def run_process_phase(
    *,
    session: MeasurementSession,
    environment_paths: Mapping[str, Path],
    wheelhouse_evidence: WheelhouseEvidence,
    command: Sequence[str],
    phase: str,
    run_id: str,
    scenario: str,
    target: Mapping[str, str],
    environment_id: str,
    filesystem_layout: str,
    timeout_seconds: float = 30.0,
    sample_interval_seconds: float = 0.02,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Validate a proposed command but refuse it without enforced isolation."""

    if not command or not all(isinstance(item, str) and item for item in command):
        raise MeasurementConfigurationError("command must be a non-empty argument sequence")
    if timeout_seconds <= 0 or sample_interval_seconds <= 0:
        raise MeasurementConfigurationError("timeout and sample interval must be positive")
    if not run_id or not scenario or not environment_id or not filesystem_layout:
        raise MeasurementConfigurationError("run, scenario, environment, and layout identities are required")
    if not target or not all(isinstance(key, str) and isinstance(value, str) and value for key, value in target.items()):
        raise MeasurementConfigurationError("target metadata must be a non-empty text mapping")
    root = _validate_marked_root(session.root)
    refreshed_evidence = _revalidate_execution_inputs(session, wheelhouse_evidence)
    observed_paths = _combined_observation_paths(session, environment_paths)
    baseline_inventory = capture_root_inventory(root)
    first_snapshot = capture_snapshot(root, observed_paths)
    first_snapshot["stage"] = "start"
    environment = build_disposable_environment(root, environment_paths, base=os.environ)
    observed_snapshot = capture_snapshot(root, observed_paths)
    observed_snapshot["stage"] = "network_refusal"
    working_directory = _guard_under_root(
        Path(cwd) if cwd is not None else session.paths["run"],
        root,
        "working directory",
        must_exist=True,
    )
    pip_arguments = materialize_pip_arguments(
        refreshed_evidence, Path(environment["PIP_CACHE_DIR"]), root
    )
    cleanup = cleanup_owned_artifacts(
        root=root,
        owned_paths=observed_paths,
        baseline_inventory=baseline_inventory,
    )
    final_snapshot = capture_snapshot(root, observed_paths)
    final_snapshot["stage"] = "post_cleanup"
    snapshots = [first_snapshot, observed_snapshot, final_snapshot]
    return {
        "schema": HARNESS_SCHEMA,
        "mode": "report_only",
        "run_id": run_id,
        "phase": phase,
        "scenario": scenario,
        "target": dict(target),
        "environment": {
            "environment_id": environment_id,
            "filesystem_layout": filesystem_layout,
            "clean_target": "unverified",
            "configured_paths": {name: str(path) for name, path in environment_paths.items()},
        },
        "command": list(command),
        "working_directory": str(working_directory),
        "execution": {
            "status": "refused",
            "performed": False,
            "reason": "network_unverified",
        },
        "network_isolation": {
            "status": "network_unverified",
            "enforced": False,
            "capability": None,
            "explanation": "The Python standard library does not provide enforceable process network isolation.",
        },
        "offline_inputs": {
            "lock_sha256": refreshed_evidence.lock_sha256,
            "requirement_count": len(refreshed_evidence.requirements),
            "pip_arguments": pip_arguments,
        },
        "path_filesystems": observed_snapshot["path_filesystems"],
        "snapshots": snapshots,
        "high_water": _high_water(snapshots),
        "raw_observations": snapshots,
        "cleanup": cleanup,
        "measurement_evidence_eligible": False,
        "public_network_authorized": False,
    }


def _run_measured_operation(
    *,
    root: Path,
    observed_paths: Mapping[str, Path],
    operation: Callable[[], Mapping[str, Any]],
    phase: str,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    """Run one real operation while sampling owned paths and filesystems."""

    if sample_interval_seconds <= 0:
        raise MeasurementConfigurationError("sample interval must be positive")
    snapshots: list[dict[str, Any]] = []
    sampling_errors: list[dict[str, str]] = []

    def take_snapshot(stage: str) -> None:
        try:
            snapshot = capture_snapshot(root, observed_paths)
            snapshot["stage"] = f"{phase}:{stage}"
            snapshots.append(snapshot)
        except (OSError, MeasurementConfigurationError) as exc:
            sampling_errors.append(
                {"phase": phase, "stage": stage, "error": str(exc)}
            )

    take_snapshot("start")
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(sample_interval_seconds):
            take_snapshot("sample")

    sampler = threading.Thread(target=sample, name=f"chatterbox-measure-{phase}", daemon=True)
    sampler.start()
    operation_result: Mapping[str, Any] | None = None
    operation_error: dict[str, str] | None = None
    try:
        operation_result = operation()
    except Exception as exc:  # The report must preserve the actionable failure.
        operation_error = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
        }
    finally:
        stop.set()
        sampler.join(timeout=max(1.0, sample_interval_seconds * 4))
        take_snapshot("end")

    if len(snapshots) < 2:
        raise MeasurementConfigurationError(
            f"{phase} did not produce two valid filesystem observations"
        )
    errors = list(sampling_errors)
    if operation_error is not None:
        errors.append(operation_error)
    return {
        "status": "completed" if not errors else "failed",
        "performed": operation_result is not None,
        "result": dict(operation_result) if operation_result is not None else None,
        "errors": errors,
        "snapshots": snapshots,
        "high_water": _high_water(snapshots),
    }


def _validate_worker_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    worker_script: Path | None,
) -> dict[str, Any]:
    required = [("runtime", "runtime_self_test", "self_test")]
    if scenario == "model-baseline":
        required.append(("model", "model_self_test", "model_self_test"))
    expected_worker = worker_script.expanduser().resolve() if worker_script is not None else None
    normalized = [dict(item) for item in observations]
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for phase, kind, event_name in required:
        matches = [
            item
            for item in normalized
            if item.get("kind") == kind and item.get("execution_observed") is True
        ]
        successful = None
        for item in matches:
            command = item.get("command")
            events = item.get("stdout_events")
            if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
                continue
            if expected_worker is not None and str(expected_worker) not in command:
                continue
            if item.get("returncode") != 0 or item.get("expected_event") != event_name:
                continue
            if not isinstance(events, list):
                continue
            event = next(
                (
                    payload
                    for payload in events
                    if isinstance(payload, Mapping)
                    and payload.get("event") == event_name
                    and payload.get("ok") is True
                ),
                None,
            )
            if event is not None:
                successful = {
                    "phase": phase,
                    "kind": kind,
                    "command": command,
                    "returncode": item.get("returncode"),
                    "event": dict(event),
                    "stdout_tail": item.get("stdout_tail", ""),
                    "stderr_tail": item.get("stderr_tail", ""),
                }
                break
        if successful is None:
            errors.append(
                f"{phase} worker execution was not observed with a successful {event_name} event"
            )
        else:
            checks.append(successful)
    return {
        "status": "complete" if not errors else "incomplete",
        "ok": not errors,
        "required": [
            {"phase": phase, "kind": kind, "event": event}
            for phase, kind, event in required
        ],
        "checks": checks,
        "observations": normalized,
        "errors": errors,
    }


def run_disposable_measurement(
    *,
    session: MeasurementSession,
    wheelhouse_evidence: WheelhouseEvidence,
    environment_paths: Mapping[str, Path],
    runtime_paths: Any,
    interpreter: Path,
    run_id: str,
    scenario: str = "runtime-baseline",
    target: Mapping[str, str],
    environment_id: str,
    filesystem_layout: str,
    worker_script: Path | None = None,
    network_isolation: Mapping[str, Any] | None = None,
    sample_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    """Execute the real installer and model activation inside one disposable root.

    The caller owns creation of the marked root and staging of the verified
    wheelhouse/model inputs.  This function only creates runtime artifacts
    below that root, records observations, and removes additions below the
    declared owned paths.  It never edits the production manifest.
    """

    if not run_id or not scenario or not environment_id:
        raise MeasurementConfigurationError(
            "run, scenario, and environment identities are required"
        )
    if scenario not in CANONICAL_SCENARIOS:
        raise MeasurementConfigurationError(f"unsupported measurement scenario: {scenario}")
    if filesystem_layout not in CANONICAL_FILESYSTEM_LAYOUTS:
        raise MeasurementConfigurationError(
            f"unsupported filesystem layout: {filesystem_layout}"
        )
    if not REQUIRED_TARGET_METADATA.issubset(target) or any(
        not isinstance(value, str) or not value for value in target.values()
    ):
        raise MeasurementConfigurationError("target metadata is incomplete")
    if network_isolation is None:
        isolation = {
            "status": "offline_inputs",
            "enforced": False,
            "capability": "verified_local_inputs",
            "explanation": "all acquisition inputs are verified local files and the disposable run provides no network-capable acquisition hook",
        }
    else:
        isolation = dict(network_isolation)
        if isolation.get("enforced") is True and not isinstance(
            isolation.get("capability"), str
        ):
            raise MeasurementConfigurationError(
                "enforced network isolation requires a capability identity"
            )
        if not isinstance(isolation.get("status"), str):
            raise MeasurementConfigurationError("network isolation status is required")
        json.dumps(isolation)

    root = _validate_marked_root(session.root)
    guarded_environment = _validated_environment_paths(root, environment_paths)
    observed_paths = _combined_observation_paths(session, guarded_environment)
    refreshed_evidence = _revalidate_execution_inputs(session, wheelhouse_evidence)

    # Import lazily so the report-only harness remains usable without loading
    # the installer's subprocess and receipt machinery at module import time.
    from .runtime import (
        _fsync_directory,
        acquire_model,
        install_runtime,
        model_preflight,
        product_status,
        runtime_status,
        validate_measurement_scope,
    )

    validate_measurement_scope(runtime_paths, root)
    _validate_runtime_observation_surface(
        runtime_paths,
        root=root,
        observed_paths=observed_paths,
    )
    if not runtime_paths.lock_path.is_file() or runtime_paths.lock_path.is_symlink():
        raise MeasurementConfigurationError("runtime lock input is not a regular file")
    if _sha256_file(runtime_paths.lock_path) != refreshed_evidence.lock_sha256:
        raise MeasurementConfigurationError(
            "runtime lock and disposable wheelhouse lock identities differ"
        )

    session_inputs = session.revalidate_inputs(
        required_package_count=CURRENT_LOCK_REQUIREMENT_COUNT
    )
    model_sources = {
        str(record["name"]): Path(str(record["path"]))
        for record in session_inputs["models"]
    }
    if set(model_sources) != set(session.expected_model_file_paths):
        raise MeasurementConfigurationError("disposable model inputs are incomplete")
    worker_data_sources = {
        str(record["name"]): Path(str(record["path"]))
        for record in session_inputs.get("worker_data", [])
    }
    if scenario == "model-baseline" and set(worker_data_sources) != {PKUSEG_DATA_FILENAME}:
        raise MeasurementConfigurationError(
            "model-baseline measurement requires the verified spacy_ontonotes.zip input"
        )

    baseline_inventory = capture_root_inventory(root)
    snapshots: list[dict[str, Any]] = []
    phase_reports: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    runtime_result: Mapping[str, Any] | None = None
    model_result: Mapping[str, Any] | None = None
    disposable_status: Mapping[str, Any] | None = None
    receipts: dict[str, Any] = {}
    pip_arguments: list[str] = []
    worker_observations: list[dict[str, Any]] = []
    worker_observation: dict[str, Any] = {
        "status": "incomplete",
        "ok": False,
        "required": [],
        "checks": [],
        "observations": [],
        "errors": ["worker execution has not been observed"],
    }

    def record_worker_observation(phase: str, observation: Mapping[str, Any]) -> None:
        record = dict(observation)
        record["phase"] = phase
        worker_observations.append(record)

    initial = capture_snapshot(root, observed_paths)
    initial["stage"] = "measurement:start"
    snapshots: list[dict[str, Any]] = [initial]

    try:
        build_disposable_environment(root, guarded_environment, base=os.environ)
        pip_arguments = materialize_pip_arguments(
            refreshed_evidence, guarded_environment["pip_cache"], root
        )
        initialized = capture_snapshot(root, observed_paths)
        initialized["stage"] = "measurement:environment_initialized"
        snapshots.append(initialized)

        runtime_phase = _run_measured_operation(
            root=root,
            observed_paths=observed_paths,
            phase="runtime-install",
            sample_interval_seconds=sample_interval_seconds,
            operation=lambda: install_runtime(
                runtime_paths,
                interpreter,
                worker_script=worker_script,
                measurement_root=root,
                wheelhouse=Path(refreshed_evidence.artifacts[0]["path"]).parent,
                pip_cache=guarded_environment["pip_cache"],
                worker_observer=lambda observation: record_worker_observation("runtime", observation),
            ),
        )
        phase_reports["runtime"] = runtime_phase
        snapshots.extend(runtime_phase["snapshots"])
        errors.extend(runtime_phase["errors"])
        runtime_result = runtime_phase["result"]

        if runtime_phase["status"] == "completed" and scenario == "model-baseline":
            def local_model_download(url: str, destination: Path) -> None:
                filename = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
                source = model_sources.get(filename)
                if source is None:
                    raise MeasurementConfigurationError(
                        f"model URL is not one of the verified disposable inputs: {filename}"
                    )
                if source.is_symlink() or not source.is_file():
                    raise MeasurementConfigurationError(
                        f"verified model input is no longer a regular file: {filename}"
                    )
                destination = _guard_under_root(
                    destination, root, f"model destination {filename}"
                )
                if destination.exists() or destination.is_symlink():
                    raise MeasurementConfigurationError(
                        f"model destination already exists: {destination}"
                    )
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                descriptor = os.open(
                    destination,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    with source.open("rb") as source_stream:
                        with os.fdopen(descriptor, "wb", closefd=False) as target_stream:
                            shutil.copyfileobj(source_stream, target_stream)
                            target_stream.flush()
                            os.fsync(target_stream.fileno())
                finally:
                    os.close(descriptor)
                _fsync_directory(destination.parent)

            def disposable_model_preflight(
                selected_paths: Any, **kwargs: Any
            ) -> Mapping[str, Any]:
                kwargs.pop("measurement_root", None)
                kwargs["measurement_root"] = root
                kwargs["check_network"] = False
                return model_preflight(selected_paths, **kwargs)

            prepared_worker_data: dict[str, Any] | None = None

            def model_operation() -> Mapping[str, Any]:
                nonlocal prepared_worker_data
                prepared_worker_data = _stage_pkuseg_data(
                    source=worker_data_sources[PKUSEG_DATA_FILENAME],
                    destination=runtime_paths.run_namespace / "cache" / "pkuseg",
                    root=root,
                )
                return acquire_model(
                    runtime_paths,
                    worker_script=worker_script,
                    measurement_root=root,
                    preflight_runner=disposable_model_preflight,
                    download_file=local_model_download,
                    worker_observer=lambda observation: record_worker_observation("model", observation),
                )

            model_phase = _run_measured_operation(
                root=root,
                observed_paths=observed_paths,
                phase="model-activation",
                sample_interval_seconds=sample_interval_seconds,
                operation=model_operation,
            )
            if prepared_worker_data is not None:
                model_phase["worker_data"] = prepared_worker_data
            phase_reports["model"] = model_phase
            snapshots.extend(model_phase["snapshots"])
            errors.extend(model_phase["errors"])
            model_result = model_phase["result"]
        else:
            phase_reports["model"] = {
                "status": "skipped",
                "performed": False,
                "result": None,
                "errors": [{"phase": "model-activation", "error": "runtime install failed"}],
                "snapshots": [],
                "high_water": None,
            }

        if scenario == "model-baseline" and model_result is not None and not errors:
            disposable_status = product_status(runtime_paths)
        elif scenario == "runtime-baseline" and runtime_result is not None and not errors:
            disposable_status = runtime_status(runtime_paths)
        receipts = {
            label: {
                "path": str(path),
                "observed": path.is_file() and not path.is_symlink(),
            }
            for label, path in (
                ("runtime", runtime_paths.runtime_receipt_path),
                ("model", runtime_paths.model_receipt_path),
                ("activation", runtime_paths.activation_receipt_path),
            )
        }
        expected_receipt_labels = (
            ("runtime",)
            if scenario == "runtime-baseline"
            else ("runtime", "model", "activation")
        )
        missing_receipts = [
            label for label in expected_receipt_labels if not receipts[label]["observed"]
        ]
        if missing_receipts:
            errors.append(
                {
                    "phase": "receipt-validation",
                    "error": f"disposable receipts were not created: {', '.join(missing_receipts)}",
                }
            )
    except Exception as exc:
        errors.append(
            {
                "phase": "measurement",
                "error_type": type(exc).__name__,
                "error": str(exc) or type(exc).__name__,
            }
        )
    finally:
        worker_observation = _validate_worker_observations(
            worker_observations,
            scenario=scenario,
            worker_script=worker_script,
        )
        errors.extend(
            {"phase": "worker-observation", "error": error}
            for error in worker_observation["errors"]
        )
        try:
            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={
                    **{f"surface.{name}": path for name, path in session.paths.items()},
                    **{f"environment.{name}": path for name, path in guarded_environment.items()},
                },
                baseline_inventory=baseline_inventory,
            )
        except Exception as exc:
            cleanup = {
                "complete": False,
                "completion_scope": "filesystem_and_process",
                "filesystem_complete": False,
                "filesystem_cleanup_status": "failed",
                "removed": [],
                "errors": [{"path": ".", "error": str(exc) or type(exc).__name__}],
                "filesystem_residue": [],
                "unowned_residue": [],
                "missing_baseline_entries": [],
                "changed_baseline_entries": [],
                "process_residue": [],
                "worker_residue": [],
                "deleted_open_files": [],
                "process_observation": "complete",
                "worker_observation": "complete",
                "deleted_open_file_observation": "complete",
            }
        if cleanup.get("filesystem_complete") and worker_observation["ok"]:
            cleanup.update(
                {
                    "completion_scope": "filesystem_and_worker",
                    "worker_observation": "complete",
                }
            )
            cleanup["complete"] = _cleanup_contents_are_complete(cleanup)
        errors.extend(
            {"phase": "cleanup", "error": str(item.get("error", "cleanup failed"))}
            for item in cleanup.get("errors", [])
        )

    final = capture_snapshot(root, observed_paths)
    final["stage"] = "measurement:post_cleanup"
    snapshots.append(final)
    high_water = _high_water(snapshots)
    required_phase = "runtime" if scenario == "runtime-baseline" else "model"
    completed = (
        not errors
        and phase_reports.get(required_phase, {}).get("status") == "completed"
    )
    expected_receipt_labels = (
        ("runtime",)
        if scenario == "runtime-baseline"
        else ("runtime", "model", "activation")
    )
    receipts_complete = bool(receipts) and all(
        receipts.get(label, {}).get("observed") is True
        for label in expected_receipt_labels
    )
    production_manifest_mutated = False
    production_readiness_mutated = False
    evidence_eligible = bool(
        completed
        and receipts_complete
        and worker_observation["ok"]
        and _cleanup_evidence_is_complete(cleanup)
        and production_manifest_mutated is False
        and production_readiness_mutated is False
        and (
            (
                isolation.get("enforced") is True
                and isinstance(isolation.get("capability"), str)
                and bool(isolation.get("capability"))
            )
            or (
                isolation.get("status") == "offline_inputs"
                and isolation.get("enforced") is False
                and isolation.get("capability") == "verified_local_inputs"
            )
        )
    )
    return {
        "schema": HARNESS_SCHEMA,
        "mode": "disposable_execution",
        "run_id": run_id,
        "phase": "runtime" if scenario == "runtime-baseline" else "model",
        "scenario": scenario,
        "target": dict(target),
        "environment": {
            "environment_id": environment_id,
            "filesystem_layout": filesystem_layout,
            "clean_target": True,
            "configured_paths": {
                name: str(path) for name, path in guarded_environment.items()
            },
        },
        "execution": {
            "status": "completed" if completed else "failed",
            "performed": True,
            "errors": errors,
            "phases": phase_reports,
            "disposable_product_status": disposable_status,
            "receipts": receipts,
            "worker_observation": worker_observation,
        },
        "network_isolation": isolation,
        "offline_inputs": {
            "lock_sha256": refreshed_evidence.lock_sha256,
            "requirement_count": len(refreshed_evidence.requirements),
            "pip_arguments": pip_arguments,
            "model_files": session_inputs["models"],
            "worker_data": session_inputs.get("worker_data", []),
        },
        "path_filesystems": initial["path_filesystems"],
        "snapshots": snapshots,
        "high_water": high_water,
        "raw_observations": snapshots,
        "cleanup": cleanup,
        "measurement_evidence_eligible": evidence_eligible,
        "production_manifest_mutated": production_manifest_mutated,
        "production_readiness_mutated": production_readiness_mutated,
        "public_network_authorized": False,
    }


def invoke_transaction_hook(
    *,
    session: MeasurementSession,
    hooks: TransactionHooks,
    kind: str,
    environment_paths: Mapping[str, Path],
    wheelhouse_evidence: WheelhouseEvidence,
    run_id: str,
    scenario: str,
    target: Mapping[str, str],
    environment_id: str,
    filesystem_layout: str,
    attempt: int = 1,
    participant: str = "main",
) -> dict[str, Any]:
    """Validate a proposed hook but refuse invocation until isolation exists."""

    hooks.get(kind)
    report = run_process_phase(
        session=session,
        environment_paths=environment_paths,
        wheelhouse_evidence=wheelhouse_evidence,
        command=("<deferred-transaction-hook>", kind),
        phase=f"transaction-{kind}",
        run_id=run_id,
        scenario=scenario,
        target=target,
        environment_id=environment_id,
        filesystem_layout=filesystem_layout,
    )
    report.update(
        {
            "transaction_kind": kind,
            "attempt": attempt,
            "participant": participant,
            "hook_invoked": False,
        }
    )
    return report


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MeasurementConfigurationError(f"{label} must be a non-negative integer")
    return value


def _explicit_required_values(values: Sequence[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise MeasurementConfigurationError(f"required {label} must be a sequence")
    normalized = {
        item for item in values if isinstance(item, str) and item
    }
    if not normalized or len(normalized) != len(values):
        raise MeasurementConfigurationError(
            f"required {label} must be explicit and unique"
        )
    return normalized


def _comparison_signature(value: Mapping[str, Any], label: str) -> str:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MeasurementConfigurationError(f"{label} is not serializable metadata") from exc


def _filesystem_metadata_signature(
    filesystem_id: str,
    metadata: Mapping[str, Any],
    label: str,
) -> str:
    missing = REQUIRED_FILESYSTEM_METADATA - set(metadata)
    if missing:
        raise MeasurementConfigurationError(
            f"{label} filesystem metadata is incomplete: {sorted(missing)}"
        )
    if metadata.get("filesystem_id") != filesystem_id:
        raise MeasurementConfigurationError(f"{label} filesystem identity differs")
    for field in ("device", "block_size", "fragment_size", "total_bytes"):
        value = _nonnegative_integer(metadata.get(field), f"{label} {field}")
        if field != "device" and value == 0:
            raise MeasurementConfigurationError(f"{label} {field} must be positive")
    for field in ("source", "mount_point", "filesystem_type", "reflink_support"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise MeasurementConfigurationError(f"{label} {field} metadata is incomplete")
    for field in (
        "mount_options",
        "super_options",
        "quota_indicators",
        "cow_indicators",
        "compression_indicators",
    ):
        value = metadata.get(field)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or any(not isinstance(item, str) for item in value)
        ):
            raise MeasurementConfigurationError(f"{label} {field} metadata is incomplete")
    if not isinstance(metadata.get("overlay"), bool):
        raise MeasurementConfigurationError(f"{label} overlay metadata is incomplete")
    comparable = {field: metadata[field] for field in sorted(REQUIRED_FILESYSTEM_METADATA)}
    return _comparison_signature(comparable, label)


def _value_classification(value: int) -> str:
    return "observed_zero" if value == 0 else "observed_nonzero"


def _canonical_phase(scenario: str) -> str | None:
    if scenario == "runtime-baseline":
        return "runtime"
    if scenario == "model-baseline":
        return "model"
    return None


# Optional variance-escalation helper. The shipping measurement contract uses
# one eligible run for directly exercised buckets; callers opt into repetition
# or scenario/layout coverage only when a concrete variance or implementation
# risk justifies it.
def aggregate_repeated_runs(
    reports: Sequence[Mapping[str, Any]],
    *,
    reserve_percent: float,
    reserve_bytes: int | Mapping[str, int] = 0,
    minimum_runs: int = 1,
    required_scenarios: Sequence[str] | None = None,
    required_filesystem_layouts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate eligible evidence and compute per-filesystem capacity.

    Repetition and scenario/layout coverage are caller-selected escalation
    policies.  The default accepts one eligible run and aggregates the evidence
    it actually supplies; it does not impose a synthetic baseline matrix.
    """

    if isinstance(minimum_runs, bool) or not isinstance(minimum_runs, int) or minimum_runs < 1:
        raise MeasurementConfigurationError("minimum runs must be a positive integer")
    if not reports:
        raise MeasurementConfigurationError("aggregation requires reports")
    if (
        isinstance(reserve_percent, bool)
        or not isinstance(reserve_percent, (int, float))
        or not math.isfinite(reserve_percent)
        or reserve_percent < 0
    ):
        raise MeasurementConfigurationError("reserve percentage must be finite and non-negative")
    required_scenario_set = (
        _explicit_required_values(required_scenarios, "scenarios")
        if required_scenarios is not None
        else None
    )
    required_layout_set = (
        _explicit_required_values(required_filesystem_layouts, "filesystem layouts")
        if required_filesystem_layouts is not None
        else None
    )

    run_ids: set[str] = set()
    targets: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in reports:
        if report.get("schema") != HARNESS_SCHEMA or report.get("mode") not in {
            "report_only",
            "disposable_execution",
        }:
            raise MeasurementConfigurationError(
                "aggregation accepts only harness report-only or disposable-execution inputs"
            )
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise MeasurementConfigurationError("run IDs must be present and unique")
        run_ids.add(run_id)
        if report.get("measurement_evidence_eligible") is not True:
            raise MeasurementConfigurationError("report is not eligible measurement evidence")
        isolation = report.get("network_isolation")
        enforced_isolation = (
            isinstance(isolation, Mapping)
            and isolation.get("status") == "enforced"
            and isolation.get("enforced") is True
            and isinstance(isolation.get("capability"), str)
            and bool(isolation["capability"])
        )
        offline_input_provenance = (
            isinstance(isolation, Mapping)
            and isolation.get("status") == "offline_inputs"
            and isolation.get("enforced") is False
            and isolation.get("capability") == "verified_local_inputs"
        )
        if not (enforced_isolation or offline_input_provenance):
            raise MeasurementConfigurationError(
                "report lacks enforced network-isolation or verified offline-input provenance"
            )

        cleanup = report.get("cleanup")
        if not isinstance(cleanup, Mapping):
            raise MeasurementConfigurationError("report lacks complete cleanup evidence")
        if not _cleanup_evidence_is_complete(cleanup):
            for field in (
                "filesystem_residue",
                "unowned_residue",
                "process_residue",
                "worker_residue",
                "deleted_open_files",
                "missing_baseline_entries",
                "changed_baseline_entries",
                "errors",
            ):
                if isinstance(cleanup.get(field), list) and cleanup[field]:
                    raise MeasurementConfigurationError(
                        "report cleanup evidence contains residue"
                    )
            raise MeasurementConfigurationError("report lacks complete cleanup evidence")

        target = report.get("target")
        if not isinstance(target, Mapping) or not REQUIRED_TARGET_METADATA.issubset(target):
            raise MeasurementConfigurationError("target provenance is incomplete")
        if any(not isinstance(value, str) or not value for value in target.values()):
            raise MeasurementConfigurationError("target provenance must contain text values")
        target_signature = _comparison_signature(target, "target provenance")
        targets.setdefault(target_signature, dict(target))

        environment = report.get("environment")
        if not isinstance(environment, Mapping):
            raise MeasurementConfigurationError("report lacks environment provenance")
        environment_id = environment.get("environment_id")
        layout = environment.get("filesystem_layout")
        if (
            not isinstance(environment_id, str)
            or not environment_id
            or not isinstance(layout, str)
            or not layout
        ):
            raise MeasurementConfigurationError(
                "environment and filesystem-layout identities are required"
            )
        if environment.get("clean_target") is not True:
            raise MeasurementConfigurationError("report lacks clean-target provenance")
        configured_paths = environment.get("configured_paths")
        if (
            not isinstance(configured_paths, Mapping)
            or set(configured_paths) != set(REQUIRED_ENVIRONMENT_PATHS)
        ):
            raise MeasurementConfigurationError(
                "configured environment-path provenance is incomplete"
            )
        if any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in configured_paths.values()
        ):
            raise MeasurementConfigurationError("configured environment paths must be absolute")
        environment_signature = _comparison_signature(
            environment, "environment provenance"
        )

        path_filesystems = report.get("path_filesystems")
        if (
            not isinstance(path_filesystems, Mapping)
            or set(path_filesystems) != set(REQUIRED_OBSERVATION_PATHS)
            or any(not isinstance(value, str) or not value for value in path_filesystems.values())
        ):
            raise MeasurementConfigurationError(
                "report path-to-filesystem matrix is incomplete"
            )
        filesystem_ids = set(path_filesystems.values())
        if layout == "same-filesystem" and len(filesystem_ids) != 1:
            raise MeasurementConfigurationError("same-filesystem layout metadata is incomparable")
        if layout == "split-filesystem" and len(filesystem_ids) < 2:
            raise MeasurementConfigurationError("split-filesystem layout metadata is incomparable")

        high_water = report.get("high_water")
        if not isinstance(high_water, Mapping):
            raise MeasurementConfigurationError("harness report is missing high-water evidence")
        high_paths = high_water.get("paths")
        if (
            not isinstance(high_paths, Mapping)
            or set(high_paths) != set(REQUIRED_OBSERVATION_PATHS)
        ):
            raise MeasurementConfigurationError("report lacks per-path high-water evidence")
        high_filesystems = high_water.get("filesystems")
        if (
            not isinstance(high_filesystems, Mapping)
            or set(high_filesystems) != filesystem_ids
        ):
            raise MeasurementConfigurationError("report lacks per-filesystem high-water evidence")
        high_owned_logical = _nonnegative_integer(
            high_water.get("owned_logical_bytes"), "owned logical high-water"
        )
        high_owned_allocated = _nonnegative_integer(
            high_water.get("owned_allocated_bytes"), "owned allocated high-water"
        )

        raw_observations = report.get("raw_observations")
        if (
            not isinstance(raw_observations, Sequence)
            or isinstance(raw_observations, (str, bytes))
            or len(raw_observations) < 2
        ):
            raise MeasurementConfigurationError(
                "report requires at least two raw observations"
            )
        timestamps: set[int] = set()
        stages: list[str] = []
        path_locations: dict[str, str] | None = None
        raw_owned_logical: list[int] = []
        raw_owned_allocated: list[int] = []
        raw_path_logical = {name: [] for name in REQUIRED_OBSERVATION_PATHS}
        raw_path_allocated = {name: [] for name in REQUIRED_OBSERVATION_PATHS}
        raw_filesystem_free = {filesystem_id: [] for filesystem_id in filesystem_ids}
        raw_filesystem_signatures = {
            filesystem_id: set() for filesystem_id in filesystem_ids
        }
        for index, observation in enumerate(raw_observations):
            if not isinstance(observation, Mapping):
                raise MeasurementConfigurationError("raw observations must be structured")
            timestamp = _nonnegative_integer(
                observation.get("timestamp_ns"), "raw observation timestamp"
            )
            if timestamp in timestamps:
                raise MeasurementConfigurationError("raw observation timestamps are ambiguous")
            timestamps.add(timestamp)
            stage = observation.get("stage")
            if not isinstance(stage, str) or not stage or stage in stages:
                raise MeasurementConfigurationError("raw observation stages are ambiguous")
            stages.append(stage)
            paths = observation.get("paths")
            if not isinstance(paths, Mapping) or set(paths) != set(REQUIRED_OBSERVATION_PATHS):
                raise MeasurementConfigurationError("raw observation path coverage is incomplete")
            if dict(observation.get("path_filesystems", {})) != dict(path_filesystems):
                raise MeasurementConfigurationError(
                    "raw observation path/filesystem provenance differs"
                )
            locations: dict[str, str] = {}
            summed_logical = 0
            summed_allocated = 0
            for name, record in paths.items():
                if not isinstance(record, Mapping):
                    raise MeasurementConfigurationError("raw per-path record is incomplete")
                location = record.get("path")
                if not isinstance(location, str) or not Path(location).is_absolute():
                    raise MeasurementConfigurationError("raw per-path location is incomplete")
                entries = _nonnegative_integer(record.get("entries"), f"{name} entries")
                logical = _nonnegative_integer(
                    record.get("logical_bytes"), f"{name} logical bytes"
                )
                allocated = _nonnegative_integer(
                    record.get("allocated_bytes"), f"{name} allocated bytes"
                )
                if entries == 0 and (logical != 0 or allocated != 0):
                    raise MeasurementConfigurationError("empty per-path record has numeric bytes")
                locations[name] = location
                raw_path_logical[name].append(logical)
                raw_path_allocated[name].append(allocated)
                summed_logical += logical
                summed_allocated += allocated
            for environment_name, configured_path in configured_paths.items():
                if locations[f"environment.{environment_name}"] != configured_path:
                    raise MeasurementConfigurationError(
                        "raw environment path metadata is incomparable"
                    )
            if path_locations is None:
                path_locations = locations
            elif locations != path_locations:
                raise MeasurementConfigurationError("raw path metadata is incomparable")
            observed_logical = _nonnegative_integer(
                observation.get("owned_logical_bytes"), "raw owned logical bytes"
            )
            observed_allocated = _nonnegative_integer(
                observation.get("owned_allocated_bytes"), "raw owned allocated bytes"
            )
            if observed_logical != summed_logical or observed_allocated != summed_allocated:
                raise MeasurementConfigurationError("raw owned totals are inconsistent")
            raw_owned_logical.append(observed_logical)
            raw_owned_allocated.append(observed_allocated)
            filesystems = observation.get("filesystems")
            if not isinstance(filesystems, Mapping) or set(filesystems) != filesystem_ids:
                raise MeasurementConfigurationError(
                    "raw observation filesystem coverage is incomplete"
                )
            for filesystem_id, metadata in filesystems.items():
                if not isinstance(metadata, Mapping):
                    raise MeasurementConfigurationError("raw filesystem metadata is incomplete")
                raw_filesystem_free[filesystem_id].append(
                    _nonnegative_integer(
                        metadata.get("free_bytes"),
                        f"raw {filesystem_id} free bytes",
                    )
                )
                raw_filesystem_signatures[filesystem_id].add(
                    _filesystem_metadata_signature(
                        filesystem_id,
                        metadata,
                        f"raw observation {index} {filesystem_id}",
                    )
                )

        if high_owned_logical != max(raw_owned_logical) or high_owned_allocated != max(
            raw_owned_allocated
        ):
            raise MeasurementConfigurationError("owned high-water evidence is inconsistent")
        for name, record in high_paths.items():
            if not isinstance(record, Mapping):
                raise MeasurementConfigurationError("per-path high-water record is incomplete")
            logical = _nonnegative_integer(
                record.get("logical_bytes"), f"{name} high-water logical bytes"
            )
            allocated = _nonnegative_integer(
                record.get("allocated_bytes"), f"{name} high-water allocated bytes"
            )
            if logical != max(raw_path_logical[name]) or allocated != max(
                raw_path_allocated[name]
            ):
                raise MeasurementConfigurationError("per-path high-water evidence is inconsistent")
            if record.get("observation_count") != len(raw_observations):
                raise MeasurementConfigurationError("per-path observation count is incomplete")
            if record.get("logical_value_classification") != _value_classification(logical):
                raise MeasurementConfigurationError("per-path zero/nonzero provenance is ambiguous")
            if record.get("allocated_value_classification") != _value_classification(allocated):
                raise MeasurementConfigurationError("per-path zero/nonzero provenance is ambiguous")
            expected_logical_stages = [
                stages[index]
                for index, value in enumerate(raw_path_logical[name])
                if value == logical
            ]
            expected_allocated_stages = [
                stages[index]
                for index, value in enumerate(raw_path_allocated[name])
                if value == allocated
            ]
            if record.get("logical_maximum_stages") != expected_logical_stages:
                raise MeasurementConfigurationError("per-path logical maximum provenance differs")
            if record.get("allocated_maximum_stages") != expected_allocated_stages:
                raise MeasurementConfigurationError("per-path allocated maximum provenance differs")

        filesystem_signatures: dict[str, str] = {}
        for filesystem_id, record in high_filesystems.items():
            if not isinstance(record, Mapping):
                raise MeasurementConfigurationError("path filesystem maximum is incomplete")
            baseline_free = _nonnegative_integer(
                record.get("baseline_free_bytes"),
                f"{filesystem_id} baseline free bytes",
            )
            minimum_free = _nonnegative_integer(
                record.get("minimum_free_bytes"),
                f"{filesystem_id} minimum free bytes",
            )
            maximum_drop = _nonnegative_integer(
                record.get("maximum_free_space_drop_bytes"),
                f"{filesystem_id} maximum free-space drop",
            )
            observed_free = raw_filesystem_free[filesystem_id]
            if (
                baseline_free != observed_free[0]
                or minimum_free != min(observed_free)
                or maximum_drop != max(0, observed_free[0] - min(observed_free))
            ):
                raise MeasurementConfigurationError("filesystem high-water evidence is inconsistent")
            if record.get("observation_count") != len(raw_observations):
                raise MeasurementConfigurationError("filesystem observation count is incomplete")
            if record.get("value_classification") != _value_classification(maximum_drop):
                raise MeasurementConfigurationError("filesystem zero/nonzero provenance is ambiguous")
            expected_minimum_stages = [
                stages[index]
                for index, value in enumerate(observed_free)
                if value == minimum_free
            ]
            if record.get("minimum_free_stages") != expected_minimum_stages:
                raise MeasurementConfigurationError("filesystem minimum provenance differs")
            metadata = record.get("metadata")
            if not isinstance(metadata, Mapping):
                raise MeasurementConfigurationError("filesystem provenance metadata is incomplete")
            filesystem_signature = _filesystem_metadata_signature(
                filesystem_id, metadata, f"high-water {filesystem_id}"
            )
            if raw_filesystem_signatures[filesystem_id] != {filesystem_signature}:
                raise MeasurementConfigurationError("raw filesystem metadata is incomparable")
            filesystem_signatures[filesystem_id] = filesystem_signature

        phase = report.get("phase")
        scenario = report.get("scenario")
        if (
            not isinstance(phase, str)
            or not phase.strip()
            or not isinstance(scenario, str)
            or not scenario.strip()
        ):
            raise MeasurementConfigurationError("phase and scenario metadata are required")
        canonical_phase = _canonical_phase(scenario)
        if canonical_phase is not None and phase != canonical_phase:
            raise MeasurementConfigurationError(
                f"scenario {scenario} requires phase {canonical_phase}"
            )
        path_signature = _comparison_signature(
            {
                "path_filesystems": dict(path_filesystems),
                "path_locations": path_locations,
            },
            "path provenance",
        )
        filesystem_signature = _comparison_signature(
            filesystem_signatures, "filesystem provenance"
        )
        groups.setdefault((scenario, layout), []).append(
            {
                "report": report,
                "phase": phase,
                "environment_id": environment_id,
                "environment_signature": environment_signature,
                "path_signature": path_signature,
                "filesystem_signature": filesystem_signature,
                "filesystem_signatures": filesystem_signatures,
            }
        )

    if len(targets) != 1:
        raise MeasurementConfigurationError("supplied runs have incomparable targets")
    observed_scenarios = {key[0] for key in groups}
    observed_layouts = {key[1] for key in groups}
    if required_scenario_set is not None and observed_scenarios != required_scenario_set:
        raise MeasurementConfigurationError(
            "scenario coverage mismatch: "
            f"missing={sorted(required_scenario_set - observed_scenarios)}, "
            f"foreign={sorted(observed_scenarios - required_scenario_set)}"
        )
    if required_layout_set is not None and observed_layouts != required_layout_set:
        raise MeasurementConfigurationError(
            "filesystem-layout coverage mismatch: "
            f"missing={sorted(required_layout_set - observed_layouts)}, "
            f"foreign={sorted(observed_layouts - required_layout_set)}"
        )
    for key, grouped in groups.items():
        if len(grouped) < minimum_runs:
            raise MeasurementConfigurationError(
                f"required scenario/layout group {key} has fewer than {minimum_runs} runs"
            )
        for signature_name in (
            "environment_signature",
            "path_signature",
            "filesystem_signature",
        ):
            if len({item[signature_name] for item in grouped}) != 1:
                raise MeasurementConfigurationError(
                    f"required scenario/layout group {key} has incomparable {signature_name.removesuffix('_signature')} metadata"
                )
    for scenario in observed_scenarios:
        phases = {
            item["phase"]
            for (group_scenario, _layout), grouped in groups.items()
            if group_scenario == scenario
            for item in grouped
        }
        if len(phases) != 1:
            raise MeasurementConfigurationError(
                f"scenario {scenario} has incomparable phase metadata"
            )
    for layout in observed_layouts:
        layout_reports = [
            item
            for (_scenario, group_layout), grouped in groups.items()
            if group_layout == layout
            for item in grouped
        ]
        for signature_name in (
            "environment_signature",
            "path_signature",
            "filesystem_signature",
        ):
            if len({item[signature_name] for item in layout_reports}) != 1:
                raise MeasurementConfigurationError(
                    f"filesystem layout {layout} has incomparable {signature_name.removesuffix('_signature')} metadata"
                )
    filesystem_identity_signatures: dict[str, set[str]] = {}
    for grouped in groups.values():
        for item in grouped:
            for filesystem_id, signature in item["filesystem_signatures"].items():
                filesystem_identity_signatures.setdefault(filesystem_id, set()).add(signature)
    if any(len(signatures) != 1 for signatures in filesystem_identity_signatures.values()):
        raise MeasurementConfigurationError(
            "repeated runs have incomparable filesystem identity metadata"
        )

    group_maxima: list[dict[str, Any]] = []
    filesystem_maxima: dict[str, int] = {}
    for (scenario, layout), grouped in sorted(groups.items()):
        grouped_reports = [item["report"] for item in grouped]
        owned_maximum = max(
            report["high_water"]["owned_allocated_bytes"]
            for report in grouped_reports
        )
        per_filesystem: dict[str, int] = {}
        for report in grouped_reports:
            for filesystem_id, record in report["high_water"]["filesystems"].items():
                value = record["maximum_free_space_drop_bytes"]
                per_filesystem[filesystem_id] = max(
                    per_filesystem.get(filesystem_id, 0), value
                )
                filesystem_maxima[filesystem_id] = max(
                    filesystem_maxima.get(filesystem_id, 0), value
                )
        group_maxima.append(
            {
                "phase": grouped[0]["phase"],
                "scenario": scenario,
                "filesystem_layout": layout,
                "environment_id": grouped[0]["environment_id"],
                "run_ids": sorted(report["run_id"] for report in grouped_reports),
                "maximum_owned_allocated_bytes": owned_maximum,
                "per_filesystem_maximum_free_space_drop_bytes": per_filesystem,
            }
        )

    if isinstance(reserve_bytes, bool):
        raise MeasurementConfigurationError("fixed reserve values must be non-negative integers")
    if isinstance(reserve_bytes, int):
        fixed_reserves = {
            filesystem_id: _nonnegative_integer(
                reserve_bytes, f"{filesystem_id} fixed reserve"
            )
            for filesystem_id in filesystem_maxima
        }
    elif isinstance(reserve_bytes, Mapping):
        if set(reserve_bytes) != set(filesystem_maxima):
            raise MeasurementConfigurationError(
                "fixed reserve map must cover every observed filesystem exactly"
            )
        fixed_reserves = {
            filesystem_id: _nonnegative_integer(
                reserve_bytes[filesystem_id], f"{filesystem_id} fixed reserve"
            )
            for filesystem_id in filesystem_maxima
        }
    else:
        raise MeasurementConfigurationError("fixed reserve values must be non-negative integers")

    per_filesystem_capacity: dict[str, dict[str, Any]] = {}
    for filesystem_id, maximum in sorted(filesystem_maxima.items()):
        percentage_reserve = math.ceil(maximum * reserve_percent / 100.0)
        fixed_reserve = fixed_reserves[filesystem_id]
        total_reserve = percentage_reserve + fixed_reserve
        per_filesystem_capacity[filesystem_id] = {
            "maximum_simultaneous_high_water_bytes": maximum,
            "reserve": {
                "percent": reserve_percent,
                "percentage_bytes": percentage_reserve,
                "fixed_bytes": fixed_reserve,
                "total_bytes": total_reserve,
            },
            "proposed_capacity_bytes": maximum + total_reserve,
        }
    retained_raw_observations = [
        {
            "run_id": report["run_id"],
            "observations": copy.deepcopy(report["raw_observations"]),
        }
        for report in sorted(reports, key=lambda item: item["run_id"])
    ]
    return {
        "schema": AGGREGATE_SCHEMA,
        "mode": "report_only",
        "target": next(iter(targets.values())),
        "run_count": len(reports),
        "run_ids": sorted(run_ids),
        "minimum_runs": minimum_runs,
        "required_scenarios": (
            sorted(required_scenario_set) if required_scenario_set is not None else None
        ),
        "required_filesystem_layouts": (
            sorted(required_layout_set) if required_layout_set is not None else None
        ),
        "observed_scenarios": sorted(observed_scenarios),
        "observed_filesystem_layouts": sorted(observed_layouts),
        "selection_rule": "maximum observed high-water per filesystem across supplied eligible runs, plus per-filesystem percentage and fixed reserves",
        "group_maxima": group_maxima,
        "per_filesystem_maximum_free_space_drop_bytes": filesystem_maxima,
        "per_filesystem_capacity": per_filesystem_capacity,
        "raw_observation_evidence": {
            "retention": "embedded",
            "runs": retained_raw_observations,
        },
        "measurement_evidence_candidate": True,
    }
