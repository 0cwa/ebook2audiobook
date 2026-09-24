"""Implementation of the Chatterbox runtime provisioning boundary.

The first supported target is Linux x86_64 with Python 3.11 and CPU Torch.
This module is intentionally a preflight/install seam, not a lifecycle
manager: it does not activate, upgrade, roll back, repair, or uninstall an
environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .contract_data import (
    ACTIVATION_RECEIPT_SCHEMA,
    MANIFEST_VERSION,
    MODEL_ACQUISITION_STORAGE_PHASES,
    MODEL_RECEIPT_SCHEMA,
    model_profile_spec,
    normalize_model_profile,
    RUNTIME_CONTRACT_VERSION,
    RUNTIME_INSTALL_STORAGE_PHASES,
    RUNTIME_RECEIPT_SCHEMA,
    STORAGE_CONTRACT_VERSION,
    STORAGE_MEASUREMENT_EVIDENCE_SCHEMA,
    TARGET_ARCH,
    TARGET_BACKEND,
    TARGET_OS,
    TARGET_PYTHON,
)
from .measurement import DISPOSABLE_MARKER, DISPOSABLE_MARKER_CONTENT

try:
    import fcntl
except ImportError:  # The provisioner is Linux-only, but imports stay portable.
    fcntl = None


RUNTIME_IDENTITY_SCHEMA = "ebook2audiobook.chatterbox-runtime-identity.v1"
MODEL_IDENTITY_SCHEMA = "ebook2audiobook.chatterbox-model-identity.v1"
ACTIVATION_IDENTITY_SCHEMA = "ebook2audiobook.chatterbox-activation-identity.v1"
RUNTIME_OWNER_SCHEMA = "ebook2audiobook.chatterbox-runtime-owner.v1"
MODEL_OWNER_SCHEMA = "ebook2audiobook.chatterbox-model-owner.v1"
# Keep the original module-level compatibility constants available to callers
# that imported the baseline runtime module directly.
RUNTIME_VERSION = "1"
DEFAULT_MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_NETWORK_TIMEOUT = 8.0

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_PINNED = re.compile(r"(?:===|==)\s*[^\s;]+")
_HASH = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
_UNRESOLVED_MARKERS = ("<unresolved", "todo", "replace-me", "master", "@main")
# Quarantine a receipt in-process from the instant it becomes visible until
# every directory durability step succeeds. Persistent publication markers
# carry the same guarantee across process restart.
_AMBIGUOUS_RECEIPT_PUBLICATIONS: set[Path] = set()


class RuntimeErrorBase(RuntimeError):
    """Base error for safe runtime operations."""


class RuntimeExpectedError(RuntimeErrorBase):
    """An expected runtime boundary failure with a stable classification."""


class RuntimeConfigurationError(RuntimeExpectedError):
    """The runtime manifest or path configuration is unsafe or incomplete."""


class RuntimeManifestError(RuntimeConfigurationError):
    """The runtime manifest is missing, malformed, or not a regular file."""


class RuntimeReceiptError(RuntimeConfigurationError):
    """A runtime, model, or activation receipt is unreadable or malformed."""


class RuntimeFilesystemError(RuntimeExpectedError):
    """An expected filesystem inspection failure at the runtime boundary."""


class ProvisioningError(RuntimeExpectedError):
    """Provisioning cannot proceed without violating a runtime gate."""


@dataclass(frozen=True)
class RuntimePaths:
    """All paths owned or consumed by the Chatterbox runtime lane."""

    runtime_dir: Path
    repo_root: Path
    data_home: Path
    state_home: Path
    e2a_root: Path
    models_dir: Path
    run_dir: Path
    # Compatibility worker input. U3 points this at the single receipt-bound
    # verified snapshot; it never points at a Hugging Face cache.
    model_namespace: Path
    verified_model_root: Path | None
    legacy_model_cache: Path
    model_objects_dir: Path
    run_namespace: Path
    state_dir: Path
    env_base: Path
    runtime_objects_dir: Path
    legacy_environment: Path
    environment: Path
    runtime_receipt_path: Path
    model_receipt_path: Path
    activation_receipt_path: Path
    install_lock_path: Path
    model_install_lock_path: Path
    result_path: Path
    manifest_path: Path
    lock_path: Path
    fingerprint: str
    runtime_fingerprint: str
    model_fingerprint: str
    activation_fingerprint: str
    manifest_sha256: str
    lock_sha256: str | None


def _freeze_status_value(value: Any) -> Any:
    """Recursively freeze status data without changing its public values."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_status_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_status_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_status_value(item) for item in value)
    return value


def _thaw_status_value(value: Any) -> Any:
    """Return ordinary containers for the compatibility mapping view."""

    if isinstance(value, Mapping):
        return {key: _thaw_status_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_status_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_status_value(item) for item in value}
    return value


@dataclass(frozen=True, slots=True)
class HostRuntimeStatus(MappingABC[str, Any]):
    """Read-only host contract and receipt-backed runtime selection.

    The record intentionally contains no :class:`RuntimePaths` and no
    runtime-internal namespace.  ``environment`` is the sanitized worker
    environment, and the path fields are only the approved inputs a host
    worker launch needs.  Dimension reports remain additive compatibility
    data; their nested containers are frozen when the record is built.
    """

    ok: bool
    supported: bool
    status: str
    artifact_status: str
    capacity_status: str
    error: str | None
    errors: tuple[str, ...]
    error_kind: str | None
    target: Mapping[str, Any]
    runtime: Mapping[str, Any]
    model: Mapping[str, Any]
    activation: Mapping[str, Any]
    interpreter: Path | None
    environment: Mapping[str, str] | None
    manifest_path: Path | None
    verified_model_root: Path | None
    runtime_root: Path | None
    manifest_root: Path | None
    model_revision: str | None
    ambiguous_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "ambiguous_paths", tuple(str(item) for item in self.ambiguous_paths))
        for name in ("target", "runtime", "model", "activation"):
            object.__setattr__(self, name, _freeze_status_value(getattr(self, name)))
        if self.environment is not None:
            object.__setattr__(self, "environment", _freeze_status_value(self.environment))

    @property
    def sanitized_worker_environment(self) -> Mapping[str, str] | None:
        """Compatibility name for the allowlisted worker environment."""

        return self.environment

    @property
    def model_root(self) -> Path | None:
        """Compatibility name for the receipt-selected verified model root."""

        return self.verified_model_root

    def as_dict(self) -> dict[str, Any]:
        """Return a detached mapping without exposing ``RuntimePaths``."""

        return {
            "ok": self.ok,
            "supported": self.supported,
            "status": self.status,
            "artifact_status": self.artifact_status,
            "capacity_status": self.capacity_status,
            "error": self.error,
            "errors": list(self.errors),
            "error_kind": self.error_kind,
            "target": _thaw_status_value(self.target),
            "runtime": _thaw_status_value(self.runtime),
            "model": _thaw_status_value(self.model),
            "activation": _thaw_status_value(self.activation),
            "interpreter": self.interpreter,
            "environment": _thaw_status_value(self.environment),
            "manifest_path": self.manifest_path,
            "verified_model_root": self.verified_model_root,
            "runtime_root": self.runtime_root,
            "manifest_root": self.manifest_root,
            "model_revision": self.model_revision,
            "ambiguous_paths": list(self.ambiguous_paths),
        }

    # A small mapping view lets a later host adapter migrate incrementally
    # while the record itself remains the ownership boundary.
    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def _default_runtime_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_repo_root(runtime_dir: Path) -> Path:
    # components/Chatterbox/runtime -> repository root
    return runtime_dir.resolve().parents[2]


def _path_from_env(value: str | None, default: Path) -> Path:
    return Path(value).expanduser() if value else default


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _require_relative_child(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    if not _is_within(resolved, parent):
        raise RuntimeConfigurationError(f"{label} must remain under {parent}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeManifestError(f"manifest is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(f"cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeManifestError("runtime manifest must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_artifact(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "filename": source.get("filename"),
        "version": source.get("version"),
        "sha256": source.get("artifact_sha256"),
    }


def _model_profile_id(model: Mapping[str, Any]) -> str | None:
    return normalize_model_profile(model.get("profile", model.get("variant")))


def _model_repositories(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return canonical repository declarations, accepting the legacy single-repo shape."""

    raw = model.get("repositories")
    if isinstance(raw, Mapping) and raw:
        return {
            str(name): dict(value)
            for name, value in raw.items()
            if isinstance(name, str) and name and isinstance(value, Mapping)
        }
    locator = model.get("locator")
    revision = model.get("revision")
    if locator is None and revision is None:
        return {}
    return {"model": {"locator": locator, "revision": revision}}


def _model_file_source(
    item: Mapping[str, Any],
    repositories: Mapping[str, Mapping[str, Any]],
) -> str | None:
    source = item.get("source")
    if isinstance(source, str) and source:
        return source
    if len(repositories) == 1:
        return next(iter(repositories))
    return None


def build_identity_contract(manifest: Mapping[str, Any], lock_sha256: str | None) -> dict[str, Any]:
    """Build separate executable identities from only identity-bearing fields."""

    sources = manifest.get("sources") if isinstance(manifest.get("sources"), Mapping) else {}
    target = manifest.get("target") if isinstance(manifest.get("target"), Mapping) else {}
    product = manifest.get("product") if isinstance(manifest.get("product"), Mapping) else {}
    chatterbox = sources.get("chatterbox_package") if isinstance(sources.get("chatterbox_package"), Mapping) else {}
    perth = sources.get("perth") if isinstance(sources.get("perth"), Mapping) else {}
    model = sources.get("model") if isinstance(sources.get("model"), Mapping) else {}
    model_files = model.get("files") if isinstance(model.get("files"), list) else []

    runtime_payload = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "runtime_contract_version": manifest.get("runtime_contract_version"),
        "target": dict(target),
        "lock_sha256": lock_sha256 or "missing",
        "artifacts": {
            "chatterbox": _selected_artifact(chatterbox),
            "perth": _selected_artifact(perth),
        },
    }
    model_profile = _model_profile_id(model) or "unknown"
    try:
        profile_spec = model_profile_spec(model_profile)
        loader_kind = str(model.get("loader_kind") or profile_spec.loader_kind)
        family = str(model.get("family") or profile_spec.family)
    except ValueError:
        loader_kind = str(model.get("loader_kind") or "unknown")
        family = str(model.get("family") or "unknown")
    repositories = _model_repositories(model)
    model_payload = {
        "schema": MODEL_IDENTITY_SCHEMA,
        "profile": model_profile,
        "loader_kind": loader_kind,
        "family": family,
        "repositories": {
            name: {
                "locator": repository.get("locator"),
                "revision": repository.get("revision"),
            }
            for name, repository in sorted(repositories.items())
        },
        "files": sorted(
            [
                {
                    "path": item.get("path"),
                    "source": _model_file_source(item, repositories),
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in model_files
                if isinstance(item, Mapping)
            ],
            key=lambda item: (str(item.get("path")), str(item.get("source"))),
        ),
    }
    runtime_digest = _canonical_digest(runtime_payload)
    model_digest = _canonical_digest(model_payload)
    runtime_fingerprint = f"py311-linux-x86_64-cpu-{runtime_digest[:16]}"
    model_fingerprint = f"chatterbox-{model_profile}-{model_digest[:16]}"
    activation_payload = {
        "schema": ACTIVATION_IDENTITY_SCHEMA,
        "runtime_fingerprint": runtime_fingerprint,
        "model_fingerprint": model_fingerprint,
        "model_profile": model_profile,
        "loader_kind": loader_kind,
        "product_profile": product.get("profile"),
        "worker_protocol": product.get("worker_protocol"),
    }
    activation_fingerprint = f"chatterbox-{model_profile}-cpu-{_canonical_digest(activation_payload)[:16]}"
    return {
        "runtime": {"fingerprint": runtime_fingerprint, "payload": runtime_payload},
        "model": {"fingerprint": model_fingerprint, "payload": model_payload},
        "activation": {"fingerprint": activation_fingerprint, "payload": activation_payload},
    }


def _fingerprint(manifest: Mapping[str, Any], lock_sha256: str | None) -> str:
    """Compatibility alias for the isolated runtime fingerprint."""

    return build_identity_contract(manifest, lock_sha256)["runtime"]["fingerprint"]


def build_paths(
    *,
    runtime_dir: Path | None = None,
    repo_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
    lock_path: Path | None = None,
) -> RuntimePaths:
    """Resolve deterministic paths without creating or modifying anything."""

    env = dict(os.environ if environment is None else environment)
    runtime = (runtime_dir or _default_runtime_dir()).resolve()
    repository = (repo_root or _default_repo_root(runtime)).resolve()
    manifest = (manifest_path or runtime / "runtime-manifest.json").resolve()
    data_home = _path_from_env(env.get("XDG_DATA_HOME"), Path.home() / ".local/share").resolve()
    state_home = _path_from_env(env.get("XDG_STATE_HOME"), Path.home() / ".local/state").resolve()
    e2a_root = _path_from_env(env.get("E2A_ROOT"), repository).resolve()
    models_dir = _path_from_env(env.get("E2A_MODELS_DIR"), e2a_root / "models").resolve()
    run_dir = _path_from_env(env.get("E2A_RUN_DIR"), e2a_root / "run").resolve()
    manifest_value = _read_json(manifest)

    lock_value = manifest_value.get("lock")
    if not isinstance(lock_value, dict) or not isinstance(lock_value.get("path"), str):
        raise RuntimeConfigurationError("manifest.lock.path is required")
    declared_lock = _require_relative_child(runtime / lock_value["path"], runtime, "lock path")
    if lock_path is not None:
        candidate_lock = lock_path.expanduser().resolve()
        if candidate_lock != declared_lock:
            raise RuntimeConfigurationError("an override lock must match manifest.lock.path")
    actual_lock_sha256 = sha256_file(declared_lock) if declared_lock.is_file() else None
    manifest_sha256 = sha256_file(manifest)
    identities = build_identity_contract(manifest_value, actual_lock_sha256)
    runtime_fingerprint = identities["runtime"]["fingerprint"]
    model_fingerprint = identities["model"]["fingerprint"]
    activation_fingerprint = identities["activation"]["fingerprint"]
    legacy_environment = (data_home / "ebook2audiobook/chatterbox/envs" / runtime_fingerprint).resolve()
    runtime_objects_dir = (
        data_home / "ebook2audiobook/chatterbox/envs/objects" / runtime_fingerprint
    ).resolve()
    # Existing Hugging Face cache content remains untouched and is never
    # adopted as a verified model. Canonical model objects live in the
    # component's user-data namespace and are selected only by a valid receipt.
    legacy_model_cache = (models_dir / "tts/chatterbox").resolve()
    model_objects_dir = (
        data_home / "ebook2audiobook/chatterbox/models/objects" / model_fingerprint
    ).resolve()
    unpublished_model = (model_objects_dir / ".unpublished/snapshot").resolve()
    state_dir = (state_home / "ebook2audiobook/chatterbox").resolve()

    paths = RuntimePaths(
        runtime_dir=runtime,
        repo_root=repository,
        data_home=data_home,
        state_home=state_home,
        e2a_root=e2a_root,
        models_dir=models_dir,
        run_dir=run_dir,
        model_namespace=unpublished_model,
        verified_model_root=None,
        legacy_model_cache=legacy_model_cache,
        model_objects_dir=model_objects_dir,
        run_namespace=(run_dir / "components/chatterbox").resolve(),
        state_dir=state_dir,
        env_base=(data_home / "ebook2audiobook/chatterbox/envs").resolve(),
        runtime_objects_dir=runtime_objects_dir,
        legacy_environment=legacy_environment,
        environment=(runtime_objects_dir / ".unpublished").resolve(),
        runtime_receipt_path=(state_dir / f"runtime-receipt-{runtime_fingerprint}.json").resolve(),
        model_receipt_path=(state_dir / f"model-receipt-{model_fingerprint}.json").resolve(),
        activation_receipt_path=(state_dir / f"activation-receipt-{activation_fingerprint}.json").resolve(),
        install_lock_path=(state_dir / f"runtime-install-{runtime_fingerprint}.lock").resolve(),
        model_install_lock_path=(state_dir / f"model-install-{model_fingerprint}.lock").resolve(),
        result_path=(state_home / f"ebook2audiobook/chatterbox/installation-result-{runtime_fingerprint}.json").resolve(),
        manifest_path=manifest,
        lock_path=declared_lock,
        fingerprint=runtime_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        model_fingerprint=model_fingerprint,
        activation_fingerprint=activation_fingerprint,
        manifest_sha256=manifest_sha256,
        lock_sha256=actual_lock_sha256,
    )
    return paths


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _executable_venv_python(path: Path, expected_source: Path | None = None) -> bool:
    """Accept the symlink Python's venv module normally creates for bin/python."""

    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    if expected_source is None or not path.is_symlink():
        return True
    try:
        return path.resolve(strict=True) == expected_source.resolve(strict=True)
    except OSError:
        return False


def _path_entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists without following symlinks."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An uninspectable ambiguity path must remain fail-closed.
        return True
    return True


def _nearest_existing(path: Path) -> Path | None:
    current = path
    while current != current.parent:
        if current.exists():
            return current
        current = current.parent
    return current if current.exists() else None


def _writable_location(path: Path) -> tuple[bool, str]:
    existing = _nearest_existing(path)
    if existing is None:
        return False, f"no existing parent for {path}"
    if not existing.is_dir():
        return False, f"nearest parent is not a directory: {existing}"
    if not os.access(existing, os.W_OK | os.X_OK):
        return False, f"path is not writable: {existing}"
    return True, str(existing)


def _probe_interpreter(interpreter: Path) -> tuple[dict[str, Any] | None, str | None]:
    code = (
        "import json,sys; print(json.dumps({"
        "'version':[sys.version_info[0],sys.version_info[1],sys.version_info[2]],"
        "'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"interpreter probe failed: {type(exc).__name__}"
    if completed.returncode != 0:
        return None, "interpreter probe returned a non-zero status"
    try:
        value = json.loads(completed.stdout.strip())
    except (json.JSONDecodeError, TypeError):
        return None, "interpreter probe returned invalid metadata"
    return value, None


def _module_hook(interpreter: Path, modules: Sequence[str]) -> tuple[bool, str]:
    encoded = json.dumps(list(modules), separators=(",", ":"))
    code = (
        "import importlib,json,sys; "
        f"[importlib.import_module(name) for name in json.loads({encoded!r})]; "
        "print('ok')"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"import hook failed: {type(exc).__name__}"
    return completed.returncode == 0, "import hook passed" if completed.returncode == 0 else "import hook failed"


def _worker_stdout_events(output: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in output.splitlines():
        if len(line) > 64 * 1024:
            continue
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, MappingABC):
            events.append(dict(payload))
    return events


def _worker_command_observation(
    interpreter: Path,
    args: Sequence[str],
    completed: subprocess.CompletedProcess[str] | None,
    *,
    observation_kind: str,
    error: str | None = None,
) -> dict[str, Any]:
    expected_event = None
    if "--model-self-test" in args:
        expected_event = "model_self_test"
    elif "--self-test" in args:
        expected_event = "self_test"
    stdout = completed.stdout if completed is not None and isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if completed is not None and isinstance(completed.stderr, str) else ""
    return {
        "kind": observation_kind,
        "command": [str(interpreter), *args],
        "execution_observed": completed is not None,
        "returncode": completed.returncode if completed is not None else None,
        "expected_event": expected_event,
        "stdout_events": _worker_stdout_events(stdout),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        **({"error": error} if error is not None else {}),
    }


def _command_hook(
    interpreter: Path,
    args: Sequence[str],
    env: Mapping[str, str],
    *,
    observation_callback: Callable[[Mapping[str, Any]], None] | None = None,
    observation_kind: str = "worker_self_test",
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [str(interpreter), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=600,
            shell=False,
            env=dict(env),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if observation_callback is not None:
            observation_callback(
                _worker_command_observation(
                    interpreter,
                    args,
                    None,
                    observation_kind=observation_kind,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        return False, f"self-test hook failed: {type(exc).__name__}"
    observation = _worker_command_observation(
        interpreter,
        args,
        completed,
        observation_kind=observation_kind,
    )
    if observation_callback is not None:
        observation_callback(observation)
    if completed.returncode == 0:
        expected_event = observation["expected_event"]
        if expected_event is not None and not any(
            event.get("event") == expected_event and event.get("ok") is True
            for event in observation["stdout_events"]
        ):
            return False, f"worker self-test did not emit a successful {expected_event} event"
        return True, "self-test hook passed"
    detail = (completed.stderr or completed.stdout or "").strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    suffix = f": {detail}" if detail else ""
    return False, f"self-test hook failed (exit {completed.returncode}){suffix}"


def _network_reachable(url: str, timeout: float = DEFAULT_NETWORK_TIMEOUT) -> tuple[bool, str]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ebook2audiobook-runtime-preflight/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # A reachable index may reject HEAD while still being available to pip.
        if exc.code in {403, 405, 429}:
            return True, f"HTTP {exc.code} (reachable)"
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, type(exc).__name__


def _identity_errors(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")
    if manifest.get("runtime_contract_version") != RUNTIME_CONTRACT_VERSION:
        errors.append(f"runtime_contract_version must be {RUNTIME_CONTRACT_VERSION}")
    target = manifest.get("target")
    if target != {"os": TARGET_OS, "architecture": TARGET_ARCH, "python": "3.11.x", "backend": TARGET_BACKEND}:
        errors.append("manifest target does not match Linux x86_64 Python 3.11 CPU")
    unresolved = manifest.get("unresolved_identities")
    if not isinstance(unresolved, list):
        errors.append("manifest.unresolved_identities must be a list")
    elif unresolved:
        errors.append(f"unresolved identities: {len(unresolved)}")

    product = manifest.get("product")
    if not isinstance(product, dict):
        errors.append("manifest.product must be an object")
    else:
        if not isinstance(product.get("profile"), str) or not product["profile"]:
            errors.append("manifest.product.profile is required")
        if not isinstance(product.get("worker_protocol"), int) or product["worker_protocol"] < 1:
            errors.append("manifest.product.worker_protocol must be a positive integer")

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return errors + ["manifest.sources must be an object"]
    required = {
        "chatterbox_package": ("filename", "version", "locator", "artifact_sha256"),
        "chatterbox_source": ("locator", "revision", "association_to_artifact"),
        "perth": ("filename", "version", "locator", "commit", "artifact_sha256", "association_to_artifact"),
        "model": ("files",),
    }
    for name, fields in required.items():
        source = sources.get(name)
        if not isinstance(source, dict):
            errors.append(f"missing source identity: {name}")
            continue
        for field in fields:
            value = source.get(field)
            if value in (None, "", [], {}):
                errors.append(f"unresolved source identity: {name}.{field}")
        for field in ("revision", "commit"):
            value = source.get(field)
            if value is not None and (not isinstance(value, str) or not _HEX40.fullmatch(value)):
                errors.append(f"{name}.{field} must be an immutable 40-character revision")
        artifact = source.get("artifact_sha256")
        if artifact is not None and (not isinstance(artifact, str) or not _HEX64.fullmatch(artifact)):
            errors.append(f"invalid SHA-256: {name}.artifact_sha256")
        locator = source.get("locator")
        if locator is not None and (not isinstance(locator, str) or not locator.startswith("https://")):
            errors.append(f"{name}.locator must be HTTPS")
        if name in {"chatterbox_source", "perth"} and source.get("association_to_artifact") != "unverified":
            errors.append(f"{name}.association_to_artifact must be unverified")
        if name == "model" and isinstance(source.get("files"), list):
            profile = _model_profile_id(source)
            if profile is None:
                errors.append("model.profile must identify a supported Chatterbox profile")
                profile_spec = None
            else:
                try:
                    profile_spec = model_profile_spec(profile)
                except ValueError:
                    profile_spec = None
                    errors.append("model.profile must identify a supported Chatterbox profile")
            if profile_spec is not None:
                loader_kind = source.get("loader_kind", profile_spec.loader_kind)
                family = source.get("family", profile_spec.family)
                if loader_kind != profile_spec.loader_kind:
                    errors.append("model.loader_kind does not match the selected profile")
                if family != profile_spec.family:
                    errors.append("model.family does not match the selected profile")

            repositories = _model_repositories(source)
            if not repositories:
                errors.append("model.repositories must contain at least one immutable repository")
            for repository_name, repository in repositories.items():
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", repository_name):
                    errors.append("model repository names must be stable identifiers")
                    continue
                locator = repository.get("locator")
                revision = repository.get("revision")
                if not isinstance(locator, str) or not locator.startswith("https://huggingface.co/"):
                    errors.append(f"model.repositories.{repository_name}.locator must be a Hugging Face HTTPS URL")
                if not isinstance(revision, str) or not _HEX40.fullmatch(revision):
                    errors.append(
                        f"model.repositories.{repository_name}.revision must be an immutable 40-character revision"
                    )

            if not source["files"]:
                errors.append("model.files must contain at least one declared artifact")
            model_paths: list[str] = []
            for item in source["files"]:
                if not isinstance(item, dict):
                    errors.append("model file identity must be an object")
                    continue
                relative = PurePosixPath(str(item.get("path", "")))
                if (
                    not item.get("path")
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or "\\" in str(item.get("path"))
                ):
                    errors.append("model file identity path must be a safe relative path")
                source_name = _model_file_source(item, repositories)
                if source_name is None or source_name not in repositories:
                    errors.append("model file identity must name a declared source repository")
                if not isinstance(item.get("size_bytes"), int) or isinstance(item.get("size_bytes"), bool) or item["size_bytes"] < 0:
                    errors.append("model file identity must contain a non-negative size")
                if not _HEX64.fullmatch(str(item.get("sha256", ""))):
                    errors.append("model file identity must contain SHA-256")
                model_paths.append(relative.as_posix())
            if len(model_paths) != len(set(model_paths)):
                errors.append("model.files paths must be unique")
    return errors


def validate_manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate versioned identity declarations without inspecting artifacts."""

    errors = _identity_errors(manifest)
    return {"ok": not errors, "errors": errors}


def receipt_identity_contracts(identities: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return identity-link fields for future receipts.

    These contracts do not assert that an environment or model exists, has
    been verified, or is ready. U2/U3 own artifact validation and readiness.
    """

    runtime_fingerprint = identities.get("runtime", {}).get("fingerprint")
    model_fingerprint = identities.get("model", {}).get("fingerprint")
    activation_fingerprint = identities.get("activation", {}).get("fingerprint")
    return {
        "runtime": {
            "schema": RUNTIME_RECEIPT_SCHEMA,
            "runtime_fingerprint": runtime_fingerprint,
        },
        "model": {
            "schema": MODEL_RECEIPT_SCHEMA,
            "model_fingerprint": model_fingerprint,
        },
        "activation": {
            "schema": ACTIVATION_RECEIPT_SCHEMA,
            "activation_fingerprint": activation_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "model_fingerprint": model_fingerprint,
        },
    }


def validate_receipt_identity_links(
    identities: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
    model_receipt: Mapping[str, Any] | None,
    activation_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate receipt identity links only, not artifacts or readiness."""

    expected = receipt_identity_contracts(identities)
    actual = {
        "runtime": runtime_receipt,
        "model": model_receipt,
        "activation": activation_receipt,
    }
    errors: list[str] = []
    for identity_name in ("runtime", "model", "activation"):
        fingerprint = identities.get(identity_name, {}).get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            errors.append(f"{identity_name} identity fingerprint is missing")
    for receipt_name, contract in expected.items():
        receipt = actual[receipt_name]
        if not isinstance(receipt, Mapping):
            errors.append(f"{receipt_name} receipt is missing")
            continue
        for field, expected_value in contract.items():
            if receipt.get(field) != expected_value:
                errors.append(f"{receipt_name} receipt {field} does not match")
    return {"ok": not errors, "identity_links_valid": not errors, "errors": errors}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_file(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    """Durably write one private JSON file without following a final symlink."""

    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_receipt(path: Path) -> dict[str, Any]:
    if not _regular_file(path):
        raise RuntimeReceiptError(f"receipt is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeReceiptError(f"cannot read receipt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeReceiptError(f"receipt must be a JSON object: {path}")
    return value


def _owner_marker_path(candidate: Path) -> Path:
    return candidate / ".e2a-chatterbox-runtime-owner.json"


def _runtime_receipt_publication_path(paths: RuntimePaths) -> Path:
    return paths.runtime_receipt_path.with_name(f"{paths.runtime_receipt_path.name}.publishing")


def _published_receipt_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.published")


def _ambiguous_receipt_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.ambiguous")


def _ensure_receipt_ambiguity_marker(receipt_path: Path, *, label: str) -> Path:
    """Durably quarantine a receipt before its publication proof is promoted."""

    ambiguity_path = _ambiguous_receipt_path(receipt_path)
    if _path_entry_exists(ambiguity_path):
        return ambiguity_path
    _write_json_file(
        ambiguity_path,
        {
            "schema": f"ebook2audiobook.chatterbox-{label}-receipt-ambiguity.v1",
            "status": "repair_required",
            "receipt": str(receipt_path),
        },
        exclusive=True,
    )
    return ambiguity_path


def _restore_receipt_quarantine(
    receipt_path: Path,
    publication_path: Path,
    published_path: Path,
    *,
    label: str,
) -> list[str]:
    """Restore a persistent fail-closed marker after uncertain publication.

    The independent ``.ambiguous`` journal is made durable before rollback, so
    even combined rename and hard-link failures remain fail-closed after a
    process restart.  The normal rollback still restores ``.publishing`` to
    preserve the existing publication protocol.
    """

    errors: list[str] = []
    ambiguity_path = _ambiguous_receipt_path(receipt_path)
    if not _path_entry_exists(ambiguity_path):
        try:
            _ensure_receipt_ambiguity_marker(receipt_path, label=label.replace(" receipt", ""))
        except (FileExistsError, OSError) as exc:
            errors.append(f"{label} persistent ambiguity journal failed: {exc}")
    if _path_entry_exists(publication_path):
        return errors
    if not _path_entry_exists(published_path):
        return [f"{label} publication proof disappeared before rollback"]

    try:
        os.replace(published_path, publication_path)
    except OSError as exc:
        errors.append(f"{label} publication rollback rename failed: {exc}")
        try:
            os.link(published_path, publication_path, follow_symlinks=False)
        except FileExistsError:
            pass
        except OSError as link_exc:
            errors.append(f"{label} publication quarantine link failed: {link_exc}")

    if not _path_entry_exists(publication_path):
        _AMBIGUOUS_RECEIPT_PUBLICATIONS.add(receipt_path)
        errors.append(f"{label} publication quarantine could not be restored")
        return errors

    try:
        _fsync_directory(receipt_path.parent)
    except OSError as exc:
        errors.append(f"{label} publication quarantine fsync failed: {exc}")
    return errors


def _receipt_publication_error(
    receipt_path: Path,
    publication_path: Path,
    *,
    label: str,
) -> str | None:
    """Validate the persistent publication state before a receipt is trusted."""

    published_path = _published_receipt_path(receipt_path)
    ambiguity_path = _ambiguous_receipt_path(receipt_path)
    if (
        _path_entry_exists(publication_path)
        or _path_entry_exists(ambiguity_path)
        or receipt_path in _AMBIGUOUS_RECEIPT_PUBLICATIONS
    ):
        return f"{label} receipt publication is incomplete or its durability is ambiguous"
    if not receipt_path.exists():
        if _path_entry_exists(published_path):
            return f"{label} receipt publication proof exists without its receipt"
        return None
    if not _regular_file(published_path):
        return f"{label} receipt has no durable publication proof"
    try:
        proof = _read_receipt(published_path)
    except RuntimeConfigurationError:
        return f"{label} receipt publication proof is invalid"
    expected_schema = f"ebook2audiobook.chatterbox-{label}-receipt-publication.v1"
    if (
        proof.get("schema") != expected_schema
        or proof.get("status") != "published"
        or proof.get("receipt") != str(receipt_path)
        or proof.get("receipt_sha256") != sha256_file(receipt_path)
    ):
        return f"{label} receipt publication proof does not match"
    return None


def _process_start_identity() -> str:
    try:
        fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        return fields[21]
    except (OSError, UnicodeError, IndexError):
        return "unavailable"


def _owner_payload(paths: RuntimePaths, candidate: Path, nonce: str) -> dict[str, Any]:
    return {
        "schema": RUNTIME_OWNER_SCHEMA,
        "runtime_fingerprint": paths.runtime_fingerprint,
        "candidate": str(candidate),
        "nonce": nonce,
        "uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "pid": os.getpid(),
        "process_start_identity": _process_start_identity(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _owned_candidate(paths: RuntimePaths, candidate: Path, nonce: str) -> bool:
    if candidate.is_symlink() or not candidate.is_dir() or not _is_within(candidate, paths.runtime_objects_dir):
        return False
    marker_path = _owner_marker_path(candidate)
    try:
        marker = _read_receipt(marker_path)
    except RuntimeConfigurationError:
        return False
    return (
        marker.get("schema") == RUNTIME_OWNER_SCHEMA
        and marker.get("runtime_fingerprint") == paths.runtime_fingerprint
        and marker.get("candidate") == str(candidate)
        and marker.get("nonce") == nonce
    )


def _cleanup_owned_candidate(paths: RuntimePaths, candidate: Path, nonce: str) -> bool:
    """Remove only the exact candidate proven to belong to this invocation."""

    if not _owned_candidate(paths, candidate, nonce):
        return False
    try:
        shutil.rmtree(candidate)
        _fsync_directory(candidate.parent)
    except OSError:
        return False
    return True


def _rollback_candidate_reservation(candidate: Path) -> list[str]:
    """Roll back only files created before durable ownership was established."""

    errors: list[str] = []
    marker = _owner_marker_path(candidate)
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"cannot remove candidate ownership marker: {exc}")
    try:
        candidate.rmdir()
    except OSError as exc:
        errors.append(f"cannot remove reserved candidate: {exc}")
    if not candidate.exists():
        try:
            _fsync_directory(candidate.parent)
        except OSError as exc:
            errors.append(f"cannot durably record candidate rollback: {exc}")
    return errors


def _runtime_receipt_payload(
    paths: RuntimePaths,
    candidate: Path,
    nonce: str,
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _owner_marker_path(candidate)
    return {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_fingerprint": paths.runtime_fingerprint,
        "manifest_sha256": paths.manifest_sha256,
        "lock_sha256": paths.lock_sha256,
        "artifact": {
            "path": str(candidate),
            "ownership_nonce": nonce,
            "owner_marker_sha256": sha256_file(marker),
        },
        "checks": dict(checks),
    }


def validate_runtime_receipt(paths: RuntimePaths, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persistent runtime receipt and its bound environment artifact."""

    errors: list[str] = []
    expected = receipt_identity_contracts(
        {
            "runtime": {"fingerprint": paths.runtime_fingerprint},
            "model": {"fingerprint": paths.model_fingerprint},
            "activation": {"fingerprint": paths.activation_fingerprint},
        }
    )["runtime"]
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(f"runtime receipt {field} does not match")
    if receipt.get("status") != "ready":
        errors.append("runtime receipt status is not ready")
    # Full-manifest hashes are audit metadata. Runtime readiness is bound to
    # the runtime fingerprint and lock identity so model/policy-only manifest
    # updates do not force an environment rebuild.
    if receipt.get("lock_sha256") != paths.lock_sha256:
        errors.append("runtime receipt lock_sha256 does not match")

    artifact = receipt.get("artifact")
    candidate: Path | None = None
    nonce: str | None = None
    if not isinstance(artifact, Mapping):
        errors.append("runtime receipt artifact binding is missing")
    else:
        raw_path = artifact.get("path")
        nonce_value = artifact.get("ownership_nonce")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append("runtime receipt artifact path is missing")
        else:
            candidate = Path(raw_path)
            if not candidate.is_absolute() or not _is_within(candidate, paths.runtime_objects_dir):
                errors.append("runtime receipt artifact path is outside the runtime object namespace")
        if not isinstance(nonce_value, str) or not nonce_value:
            errors.append("runtime receipt ownership nonce is missing")
        else:
            nonce = nonce_value
        if candidate is not None and nonce is not None:
            if not _owned_candidate(paths, candidate, nonce):
                errors.append("runtime receipt artifact ownership does not match")
            else:
                marker_hash = sha256_file(_owner_marker_path(candidate))
                if artifact.get("owner_marker_sha256") != marker_hash:
                    errors.append("runtime receipt owner marker hash does not match")
                env_python = candidate / "bin/python"
                if not _executable_venv_python(env_python):
                    errors.append("runtime receipt artifact has no executable Python")

    checks = receipt.get("checks")
    required_checks = ("venv_creation", "lock_install", "pip_check", "imports", "worker_self_test", "receipt_validation")
    if not isinstance(checks, Mapping):
        errors.append("runtime receipt checks are missing")
    else:
        for name in required_checks:
            value = checks.get(name)
            if not isinstance(value, Mapping) or value.get("ok") is not True:
                errors.append(f"runtime receipt check did not pass: {name}")
    return {
        "ok": not errors,
        "status": "ready" if not errors else "repair_required",
        "errors": errors,
        "environment": str(candidate) if candidate is not None else None,
    }


def runtime_status(paths: RuntimePaths) -> dict[str, Any]:
    """Report receipt-backed runtime readiness without mutating any state."""

    publication = _runtime_receipt_publication_path(paths)
    publication_error = _receipt_publication_error(
        paths.runtime_receipt_path,
        publication,
        label="runtime",
    )
    if publication_error:
        ambiguous_paths = [
            str(path)
            for path in (
                publication,
                _published_receipt_path(paths.runtime_receipt_path),
                _ambiguous_receipt_path(paths.runtime_receipt_path),
            )
            if _path_entry_exists(path)
        ] or [str(paths.runtime_receipt_path)]
        return {
            "ok": False,
            "status": "repair_required",
            "errors": [publication_error],
            "error_kind": "receipt",
            "ambiguous_paths": ambiguous_paths,
            "environment": None,
            "legacy_result_ignored": paths.result_path.exists(),
        }

    if paths.runtime_receipt_path.exists():
        try:
            receipt = _read_receipt(paths.runtime_receipt_path)
        except RuntimeReceiptError as exc:
            return {
                "ok": False,
                "status": "repair_required",
                "errors": [str(exc)],
                "error_kind": "receipt",
                "environment": None,
            }
        except RuntimeConfigurationError as exc:
            return {
                "ok": False,
                "status": "repair_required",
                "errors": [str(exc)],
                "error_kind": "configuration",
                "environment": None,
            }
        result = validate_runtime_receipt(paths, receipt)
        result["receipt"] = str(paths.runtime_receipt_path)
        return result

    ambiguous: list[str] = []
    if paths.legacy_environment.exists():
        ambiguous.append(str(paths.legacy_environment))
    if paths.runtime_objects_dir.exists():
        try:
            ambiguous.extend(str(item) for item in paths.runtime_objects_dir.iterdir())
        except OSError as exc:
            return {"ok": False, "status": "repair_required", "errors": [f"cannot inspect runtime objects: {exc}"], "environment": None}
    if ambiguous:
        return {
            "ok": False,
            "status": "repair_required",
            "errors": ["runtime artifacts exist without a valid runtime receipt"],
            "ambiguous_paths": sorted(set(ambiguous)),
            "environment": None,
            "legacy_result_ignored": paths.result_path.exists(),
        }
    return {
        "ok": False,
        "status": "missing",
        "errors": [],
        "environment": None,
        "legacy_result_ignored": paths.result_path.exists(),
    }


def _model_owner_marker_path(candidate: Path) -> Path:
    return candidate / ".e2a-chatterbox-model-owner.json"


def _model_receipt_publication_path(paths: RuntimePaths) -> Path:
    return paths.model_receipt_path.with_name(f"{paths.model_receipt_path.name}.publishing")


def _activation_receipt_publication_path(paths: RuntimePaths) -> Path:
    return paths.activation_receipt_path.with_name(f"{paths.activation_receipt_path.name}.publishing")


def _model_owner_payload(paths: RuntimePaths, candidate: Path, nonce: str) -> dict[str, Any]:
    return {
        "schema": MODEL_OWNER_SCHEMA,
        "model_fingerprint": paths.model_fingerprint,
        "candidate": str(candidate),
        "nonce": nonce,
        "uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "pid": os.getpid(),
        "process_start_identity": _process_start_identity(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _owned_model_candidate(paths: RuntimePaths, candidate: Path, nonce: str) -> bool:
    if candidate.is_symlink() or not candidate.is_dir() or candidate.parent != paths.model_objects_dir:
        return False
    marker_path = _model_owner_marker_path(candidate)
    try:
        marker = _read_receipt(marker_path)
    except RuntimeConfigurationError:
        return False
    return (
        marker.get("schema") == MODEL_OWNER_SCHEMA
        and marker.get("model_fingerprint") == paths.model_fingerprint
        and marker.get("candidate") == str(candidate)
        and marker.get("nonce") == nonce
    )


def _cleanup_owned_model_candidate(paths: RuntimePaths, candidate: Path, nonce: str) -> bool:
    if not _owned_model_candidate(paths, candidate, nonce):
        return False
    try:
        shutil.rmtree(candidate)
        _fsync_directory(candidate.parent)
    except OSError:
        return False
    return True


def _model_file_records(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    model = manifest.get("sources", {}).get("model", {})
    if not isinstance(model, Mapping):
        raise RuntimeConfigurationError("manifest sources.model is required")
    profile = _model_profile_id(model)
    if profile is None:
        raise RuntimeConfigurationError("manifest sources.model.profile is unsupported")
    repositories = _model_repositories(model)
    if not repositories:
        raise RuntimeConfigurationError("manifest sources.model.repositories is empty")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeConfigurationError("manifest sources.model.files must contain declared artifacts")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeConfigurationError("manifest model file records must be objects")
        raw_path = item.get("path")
        relative = PurePosixPath(str(raw_path or ""))
        if (
            not raw_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in str(raw_path)
        ):
            raise RuntimeConfigurationError("manifest model file paths must be safe relative paths")
        path = relative.as_posix()
        if path in seen:
            raise RuntimeConfigurationError("manifest model file paths must be unique")
        seen.add(path)
        source = _model_file_source(item, repositories)
        if source is None or source not in repositories:
            raise RuntimeConfigurationError(
                f"manifest model file source is invalid: {path}"
            )
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeConfigurationError(f"manifest model file size is invalid: {path}")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise RuntimeConfigurationError(f"manifest model file SHA-256 is invalid: {path}")
        records.append({
            "path": path,
            "source": source,
            "size_bytes": size,
            "sha256": digest.lower(),
        })
    return tuple(records)


def _verified_snapshot_report(snapshot: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify one immutable snapshot contains exactly the declared regular files."""

    errors: list[str] = []
    if snapshot.is_symlink() or not snapshot.is_dir():
        return {"ok": False, "errors": ["verified model snapshot is not a regular directory"], "files": []}
    expected = {str(item["path"]): item for item in records}
    discovered: set[str] = set()
    stack = [snapshot]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as exc:
            errors.append(f"cannot inspect verified model snapshot: {exc}")
            break
        for entry in entries:
            path = Path(entry.path)
            try:
                relative = path.relative_to(snapshot).as_posix()
            except ValueError:
                errors.append("verified model snapshot contains a path escape")
                continue
            if entry.is_symlink():
                errors.append(f"verified model snapshot contains a symlink: {relative}")
            elif entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                discovered.add(relative)
            else:
                errors.append(f"verified model snapshot contains a special file: {relative}")
    for extra in sorted(discovered - set(expected)):
        errors.append(f"verified model snapshot contains an undeclared file: {extra}")
    verified: list[dict[str, Any]] = []
    for relative, record in sorted(expected.items()):
        path = snapshot / Path(*PurePosixPath(relative).parts)
        if relative not in discovered or path.is_symlink() or not path.is_file():
            errors.append(f"verified model snapshot is missing: {relative}")
            continue
        try:
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            errors.append(f"cannot verify model file {relative}: {exc}")
            continue
        if size != int(record["size_bytes"]):
            errors.append(f"verified model snapshot size does not match: {relative}")
            continue
        if digest.lower() != str(record["sha256"]).lower():
            errors.append(f"verified model snapshot checksum does not match: {relative}")
            continue
        verified.append({"path": relative, "size_bytes": size, "sha256": digest.lower()})
    return {"ok": not errors, "errors": errors, "files": verified}


def validate_model_receipt(paths: RuntimePaths, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persistent model receipt and its exact local snapshot."""

    errors: list[str] = []
    expected = receipt_identity_contracts(
        {
            "runtime": {"fingerprint": paths.runtime_fingerprint},
            "model": {"fingerprint": paths.model_fingerprint},
            "activation": {"fingerprint": paths.activation_fingerprint},
        }
    )["model"]
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(f"model receipt {field} does not match")
    if receipt.get("status") != "ready":
        errors.append("model receipt status is not ready")
    # Full-manifest hashes remain recorded for audit; the model fingerprint
    # and exact file verification are the executable readiness authority.
    artifact = receipt.get("artifact")
    candidate: Path | None = None
    snapshot: Path | None = None
    nonce: str | None = None
    if not isinstance(artifact, Mapping):
        errors.append("model receipt artifact binding is missing")
    else:
        raw_object = artifact.get("object_path")
        raw_snapshot = artifact.get("snapshot_path")
        nonce_value = artifact.get("ownership_nonce")
        if isinstance(raw_object, str) and raw_object:
            candidate = Path(raw_object)
            if not candidate.is_absolute() or candidate.parent != paths.model_objects_dir:
                errors.append("model receipt object is outside the canonical model namespace")
        else:
            errors.append("model receipt object path is missing")
        if isinstance(raw_snapshot, str) and raw_snapshot:
            snapshot = Path(raw_snapshot)
            if candidate is None or snapshot != candidate / "snapshot":
                errors.append("model receipt snapshot does not use the canonical object layout")
        else:
            errors.append("model receipt snapshot path is missing")
        if isinstance(nonce_value, str) and nonce_value:
            nonce = nonce_value
        else:
            errors.append("model receipt ownership nonce is missing")
        if candidate is not None and nonce is not None:
            if not _owned_model_candidate(paths, candidate, nonce):
                errors.append("model receipt artifact ownership does not match")
            else:
                marker_hash = sha256_file(_model_owner_marker_path(candidate))
                if artifact.get("owner_marker_sha256") != marker_hash:
                    errors.append("model receipt owner marker hash does not match")
    manifest_records: tuple[dict[str, Any], ...] = ()
    if snapshot is not None:
        try:
            manifest = _read_json(paths.manifest_path)
            manifest_records = _model_file_records(manifest)
            snapshot_report = _verified_snapshot_report(snapshot, manifest_records)
        except RuntimeConfigurationError as exc:
            snapshot_report = {"ok": False, "errors": [str(exc)], "files": []}
        errors.extend(snapshot_report["errors"])
    else:
        snapshot_report = {"ok": False, "errors": [], "files": []}
    expected_files = sorted(
        (dict(item) for item in manifest_records),
        key=lambda item: item["path"],
    )
    if receipt.get("files") != expected_files:
        errors.append("model receipt file metadata does not match the canonical manifest records")
    checks = receipt.get("checks")
    for name in ("acquisition", "snapshot_validation", "receipt_validation"):
        value = checks.get(name) if isinstance(checks, Mapping) else None
        if not isinstance(value, Mapping) or value.get("ok") is not True:
            errors.append(f"model receipt check did not pass: {name}")
    return {
        "ok": not errors,
        "status": "ready" if not errors else "repair_required",
        "errors": errors,
        "snapshot": str(snapshot) if snapshot is not None else None,
        "files": snapshot_report["files"],
    }


def model_status(paths: RuntimePaths) -> dict[str, Any]:
    publication = _model_receipt_publication_path(paths)
    publication_error = _receipt_publication_error(
        paths.model_receipt_path,
        publication,
        label="model",
    )
    if publication_error:
        ambiguous_paths = [
            str(path)
            for path in (
                publication,
                _published_receipt_path(paths.model_receipt_path),
                _ambiguous_receipt_path(paths.model_receipt_path),
            )
            if _path_entry_exists(path)
        ] or [str(paths.model_receipt_path)]
        return {
            "ok": False,
            "status": "repair_required",
            "errors": [publication_error],
            "error_kind": "receipt",
            "ambiguous_paths": ambiguous_paths,
            "snapshot": None,
        }
    if paths.model_receipt_path.exists():
        try:
            receipt = _read_receipt(paths.model_receipt_path)
        except RuntimeReceiptError as exc:
            return {
                "ok": False,
                "status": "repair_required",
                "errors": [str(exc)],
                "error_kind": "receipt",
                "snapshot": None,
            }
        except RuntimeConfigurationError as exc:
            return {
                "ok": False,
                "status": "repair_required",
                "errors": [str(exc)],
                "error_kind": "configuration",
                "snapshot": None,
            }
        result = validate_model_receipt(paths, receipt)
        result["receipt"] = str(paths.model_receipt_path)
        return result
    ambiguous: list[str] = []
    if paths.model_objects_dir.exists():
        try:
            ambiguous.extend(str(item) for item in paths.model_objects_dir.iterdir())
        except OSError as exc:
            return {"ok": False, "status": "repair_required", "errors": [f"cannot inspect model objects: {exc}"], "snapshot": None}
    if ambiguous:
        return {
            "ok": False,
            "status": "repair_required",
            "errors": ["model artifacts exist without a valid model receipt"],
            "ambiguous_paths": sorted(set(ambiguous)),
            "snapshot": None,
        }
    return {"ok": False, "status": "missing", "errors": [], "snapshot": None}


def validate_activation_receipt(
    paths: RuntimePaths,
    receipt: Mapping[str, Any],
    *,
    runtime_result: Mapping[str, Any] | None = None,
    model_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected = receipt_identity_contracts(
        {
            "runtime": {"fingerprint": paths.runtime_fingerprint},
            "model": {"fingerprint": paths.model_fingerprint},
            "activation": {"fingerprint": paths.activation_fingerprint},
        }
    )["activation"]
    for field, value in expected.items():
        if receipt.get(field) != value:
            errors.append(f"activation receipt {field} does not match")
    if receipt.get("status") != "ready":
        errors.append("activation receipt status is not ready")
    # Activation is linked through runtime/model fingerprints and receipt
    # hashes. The full manifest hash is non-binding audit metadata.
    runtime = runtime_result if runtime_result is not None else runtime_status(paths)
    model = model_result if model_result is not None else model_status(paths)
    if not runtime.get("ok"):
        errors.append("activation receipt runtime is not ready")
    if not model.get("ok"):
        errors.append("activation receipt model is not ready")
    if paths.runtime_receipt_path.is_file():
        if receipt.get("runtime_receipt_sha256") != sha256_file(paths.runtime_receipt_path):
            errors.append("activation receipt runtime receipt hash does not match")
    if paths.model_receipt_path.is_file():
        if receipt.get("model_receipt_sha256") != sha256_file(paths.model_receipt_path):
            errors.append("activation receipt model receipt hash does not match")
    if receipt.get("environment") != runtime.get("environment"):
        errors.append("activation receipt environment does not match")
    if receipt.get("model_snapshot") != model.get("snapshot"):
        errors.append("activation receipt model snapshot does not match")
    check = receipt.get("checks", {}).get("local_model_load") if isinstance(receipt.get("checks"), Mapping) else None
    if not isinstance(check, Mapping) or check.get("ok") is not True:
        errors.append("activation receipt local model load check did not pass")
    return {"ok": not errors, "status": "ready" if not errors else "repair_required", "errors": errors}


def activation_status(
    paths: RuntimePaths,
    *,
    runtime_result: Mapping[str, Any] | None = None,
    model_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    publication = _activation_receipt_publication_path(paths)
    publication_error = _receipt_publication_error(
        paths.activation_receipt_path,
        publication,
        label="activation",
    )
    if publication_error:
        ambiguous_paths = [
            str(path)
            for path in (
                publication,
                _published_receipt_path(paths.activation_receipt_path),
                _ambiguous_receipt_path(paths.activation_receipt_path),
            )
            if _path_entry_exists(path)
        ] or [str(paths.activation_receipt_path)]
        return {
            "ok": False,
            "status": "repair_required",
            "errors": [publication_error],
            "error_kind": "receipt",
            "ambiguous_paths": ambiguous_paths,
        }
    if not paths.activation_receipt_path.exists():
        return {"ok": False, "status": "missing", "errors": []}
    try:
        receipt = _read_receipt(paths.activation_receipt_path)
    except RuntimeReceiptError as exc:
        return {
            "ok": False,
            "status": "repair_required",
            "errors": [str(exc)],
            "error_kind": "receipt",
        }
    except RuntimeConfigurationError as exc:
        return {
            "ok": False,
            "status": "repair_required",
            "errors": [str(exc)],
            "error_kind": "configuration",
        }
    result = validate_activation_receipt(
        paths,
        receipt,
        runtime_result=runtime_result,
        model_result=model_result,
    )
    result["receipt"] = str(paths.activation_receipt_path)
    return result


def _receipt_selected_paths(
    paths: RuntimePaths,
    runtime: Mapping[str, Any],
    model: Mapping[str, Any],
) -> RuntimePaths:
    """Apply already-validated receipt selections without inspecting artifacts."""

    selected = paths
    environment = runtime.get("environment")
    if runtime.get("ok") and isinstance(environment, str):
        selected = replace(selected, environment=Path(environment))
    snapshot = model.get("snapshot")
    if model.get("ok") and isinstance(snapshot, str):
        snapshot_path = Path(snapshot)
        selected = replace(selected, model_namespace=snapshot_path, verified_model_root=snapshot_path)
    return selected


def readiness_snapshot(paths: RuntimePaths) -> dict[str, Any]:
    """Inspect runtime/model once, then validate activation against that snapshot."""

    runtime = runtime_status(paths)
    model_paths = _receipt_selected_paths(paths, runtime, {"ok": False})
    model = model_status(model_paths)
    selected_paths = _receipt_selected_paths(paths, runtime, model)
    activation = activation_status(
        selected_paths,
        runtime_result=runtime,
        model_result=model,
    )
    dimensions = {"runtime": runtime, "model": model, "activation": activation}
    if any(item.get("status") == "repair_required" for item in dimensions.values()):
        status = "repair_required"
    elif runtime.get("status") != "ready":
        status = "runtime_missing"
    elif model.get("status") != "ready":
        status = "model_missing"
    elif activation.get("status") != "ready":
        status = "activation_missing"
    else:
        status = "ready"
    return {"ok": status == "ready", "status": status, **dimensions, "paths": selected_paths}


def product_status(paths: RuntimePaths) -> dict[str, Any]:
    """Report complete product readiness; a runtime receipt alone is insufficient."""

    snapshot = readiness_snapshot(paths)
    snapshot.pop("paths")
    runtime = snapshot["runtime"]
    model = snapshot["model"]
    activation = snapshot["activation"]
    dimensions = {"runtime": runtime, "model": model, "activation": activation}
    return {**snapshot, **dimensions}


def _host_target_status() -> dict[str, Any]:
    """Return the current host target without probing or mutating runtime state."""

    actual_os = platform.system().lower()
    actual_arch = platform.machine().lower()
    supported = actual_os == TARGET_OS and actual_arch in {TARGET_ARCH, "amd64"}
    return {
        "supported": supported,
        "status": "supported" if supported else "unsupported",
        "os": actual_os,
        "architecture": actual_arch,
        "expected": {
            "os": TARGET_OS,
            "architecture": TARGET_ARCH,
            "python": "3.11.x",
            "backend": TARGET_BACKEND,
        },
        "error": None
        if supported
        else "Chatterbox supports Linux x86_64/amd64 CPU only",
    }


def _manifest_model_revision(manifest_path: Path) -> str:
    """Read the immutable model revision from the authoritative manifest."""

    manifest = _read_json(manifest_path)
    sources = manifest.get("sources")
    model = sources.get("model") if isinstance(sources, Mapping) else None
    revision = model.get("revision") if isinstance(model, Mapping) else None
    if not isinstance(revision, str) or not _HEX40.fullmatch(revision):
        raise RuntimeManifestError("runtime manifest must contain an immutable model revision")
    return revision.lower()


def _runtime_error_kind(error: BaseException) -> str | None:
    """Map only established expected boundary failures to stable categories."""

    if isinstance(error, RuntimeReceiptError):
        return "receipt"
    if isinstance(error, RuntimeManifestError):
        return "manifest"
    if isinstance(error, RuntimeConfigurationError):
        return "configuration"
    if isinstance(error, RuntimeFilesystemError) or isinstance(error, OSError):
        return "filesystem"
    return None


def _infer_status_error_kind(status: Mapping[str, Any]) -> str | None:
    explicit = status.get("error_kind")
    if isinstance(explicit, str) and explicit:
        return explicit
    for value in status.get("errors", ()):
        text = str(value).lower()
        if "receipt" in text:
            return "receipt"
        if "manifest" in text:
            return "manifest"
        if any(marker in text for marker in ("filesystem", "inspect", "free-space", "disk")):
            return "filesystem"
    return None


def _host_status_error(label: str, state: str, errors: Sequence[Any] = ()) -> str:
    details = "; ".join(str(item) for item in errors if item)
    if state == "repair_required":
        suffix = f": {details}" if details else ""
        return f"Chatterbox {label} requires repair{suffix}."
    if state == "provisioning":
        return (
            f"Chatterbox {label} provisioning is blocked by storage_budget_unknown; "
            "clean capacity measurements are required before provisioning."
        )
    return f"Chatterbox {label} is missing; provision the approved local {label} before use."


def _host_selection(
    paths: RuntimePaths,
    dimensions: Mapping[str, Mapping[str, Any]],
    selected_paths: RuntimePaths,
) -> dict[str, Any]:
    """Build only the approved worker inputs from a validated readiness view."""

    runtime = dimensions["runtime"]
    model = dimensions["model"]
    runtime_environment = runtime.get("environment")
    interpreter: Path | None = None
    environment: Mapping[str, str] | None = None
    if runtime.get("status") == "ready" and isinstance(runtime_environment, str):
        environment_path = Path(runtime_environment)
        interpreter = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        environment = sanitized_worker_environment(
            replace(selected_paths, environment=environment_path),
        )

    verified_model_root: Path | None = None
    snapshot = model.get("snapshot")
    if model.get("status") == "ready" and isinstance(snapshot, str):
        verified_model_root = Path(snapshot)

    return {
        "interpreter": interpreter,
        "environment": environment,
        "manifest_path": paths.manifest_path,
        "verified_model_root": verified_model_root,
        "runtime_root": paths.runtime_dir,
        "manifest_root": paths.runtime_dir,
        "model_revision": _manifest_model_revision(paths.manifest_path),
    }


def _host_record(
    *,
    target: Mapping[str, Any],
    ok: bool,
    supported: bool,
    status: str,
    artifact_status: str,
    capacity_status: str,
    error: str | None,
    errors: Sequence[Any] = (),
    error_kind: str | None = None,
    runtime: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    activation: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
    ambiguous_paths: Sequence[Any] = (),
) -> HostRuntimeStatus:
    selected = selection or {}
    return HostRuntimeStatus(
        ok=ok,
        supported=supported,
        status=status,
        artifact_status=artifact_status,
        capacity_status=capacity_status,
        error=error,
        errors=tuple(str(item) for item in errors),
        error_kind=error_kind,
        target=target,
        runtime=runtime or {},
        model=model or {},
        activation=activation or {},
        interpreter=selected.get("interpreter"),
        environment=selected.get("environment"),
        manifest_path=selected.get("manifest_path"),
        verified_model_root=selected.get("verified_model_root"),
        runtime_root=selected.get("runtime_root"),
        manifest_root=selected.get("manifest_root"),
        model_revision=selected.get("model_revision"),
        ambiguous_paths=tuple(str(item) for item in ambiguous_paths),
    )


def _host_error_record(
    error: BaseException,
    *,
    target: Mapping[str, Any],
    runtime_dir: Path | None = None,
    manifest_path: Path | None = None,
    paths: RuntimePaths | None = None,
) -> HostRuntimeStatus:
    """Classify an expected error; let unrecognized defects propagate."""

    error_kind = _runtime_error_kind(error)
    if error_kind is None:
        raise error
    runtime_root = paths.runtime_dir if paths is not None else (runtime_dir or _default_runtime_dir()).resolve()
    selected_manifest = paths.manifest_path if paths is not None else (manifest_path or runtime_root / "runtime-manifest.json").resolve()
    selection = {
        "manifest_path": selected_manifest,
        "runtime_root": runtime_root,
        "manifest_root": runtime_root,
    }
    detail = str(error)
    return _host_record(
        target=target,
        ok=False,
        supported=True,
        status="repair_required",
        artifact_status="repair_required",
        capacity_status="not_checked",
        error=f"Chatterbox readiness requires repair: {detail}.",
        errors=(detail,),
        error_kind=error_kind,
        selection=selection,
    )


def _host_status_from_snapshot(
    paths: RuntimePaths,
    snapshot: Mapping[str, Any],
    target: Mapping[str, Any],
) -> HostRuntimeStatus:
    dimensions = {
        name: dict(snapshot.get(name, {}))
        for name in ("runtime", "model", "activation")
    }
    selected_paths = snapshot.get("paths", paths)
    selection = _host_selection(paths, dimensions, selected_paths)

    # Receipt-backed readiness is authoritative.  A ready chain deliberately
    # bypasses capacity inspection, preserving the existing activation-only
    # fast path and avoiding a false provisioning block.
    if snapshot.get("ok") and snapshot.get("status") == "ready":
        return _host_record(
            target=target,
            ok=True,
            supported=True,
            status="ready",
            artifact_status="ready",
            capacity_status="not_required",
            error=None,
            runtime=dimensions["runtime"],
            model=dimensions["model"],
            activation=dimensions["activation"],
            selection=selection,
        )

    for label, dimension in dimensions.items():
        if dimension.get("status") == "repair_required":
            errors = tuple(dimension.get("errors", ()))
            return _host_record(
                target=target,
                ok=False,
                supported=True,
                status="repair_required",
                artifact_status="repair_required",
                capacity_status="not_checked",
                error=_host_status_error(label, "repair_required", errors),
                errors=errors,
                error_kind=_infer_status_error_kind(dimension),
                runtime=dimensions["runtime"],
                model=dimensions["model"],
                activation=dimensions["activation"],
                selection=selection,
                ambiguous_paths=dimension.get("ambiguous_paths", ()),
            )

    incomplete = next(
        (name for name in ("runtime", "model", "activation") if dimensions[name].get("status") != "ready"),
        "activation",
    )
    manifest = _read_json(paths.manifest_path)
    phases = (
        RUNTIME_INSTALL_STORAGE_PHASES
        if incomplete == "runtime"
        else MODEL_ACQUISITION_STORAGE_PHASES
    )
    storage = calculate_storage_plan(
        selected_paths,
        manifest,
        phases=phases,
        verified_model_root=selection["verified_model_root"],
    )
    dimensions[incomplete]["storage"] = storage
    capacity_status = str(storage.get("status", "not_checked"))
    if capacity_status == "storage_budget_unknown":
        status = "provisioning"
        error = _host_status_error(incomplete, status, storage.get("errors", ()))
    elif capacity_status == "insufficient_storage":
        status = "missing"
        error = f"Chatterbox {incomplete} is missing; provisioning has insufficient storage."
    elif capacity_status == "sufficient":
        status = "missing"
        error = _host_status_error(incomplete, status, storage.get("errors", ()))
    else:
        status = "repair_required"
        error = _host_status_error(incomplete, status, storage.get("errors", ()))

    artifact_status = (
        "repair_required"
        if any(dimension.get("status") == "repair_required" for dimension in dimensions.values())
        else "ready"
        if all(dimension.get("status") == "ready" for dimension in dimensions.values())
        else "missing"
    )
    all_errors: list[str] = []
    for dimension in dimensions.values():
        all_errors.extend(str(item) for item in dimension.get("errors", ()))
    all_errors.extend(str(item) for item in storage.get("errors", ()))
    return _host_record(
        target=target,
        ok=False,
        supported=True,
        status=status,
        artifact_status=artifact_status,
        capacity_status=capacity_status,
        error=error,
        errors=tuple(dict.fromkeys(all_errors)),
        error_kind=(
            "configuration"
            if capacity_status == "invalid_storage_contract"
            else "filesystem"
            if capacity_status == "storage_check_failed"
            else None
        ),
        runtime=dimensions["runtime"],
        model=dimensions["model"],
        activation=dimensions["activation"],
        selection=selection,
    )


def host_runtime_status(
    paths: RuntimePaths | None = None,
    *,
    runtime_dir: Path | None = None,
    repo_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
    lock_path: Path | None = None,
) -> HostRuntimeStatus:
    """Return a frozen, host-facing status without exposing ``RuntimePaths``.

    Callers may omit ``paths`` and provide only the same optional resolution
    hints accepted by :func:`build_paths`.  The optional ``paths`` argument is
    retained for internal callers and focused tests; it is never included in
    the returned record.
    """

    target = _host_target_status()
    if not target["supported"]:
        return _host_record(
            target=target,
            ok=False,
            supported=False,
            status="unsupported",
            artifact_status="unsupported",
            capacity_status="not_checked",
            error=target["error"],
        )

    resolved_paths = paths
    try:
        if resolved_paths is None:
            resolved_paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=repo_root,
                environment=environment,
                manifest_path=manifest_path,
                lock_path=lock_path,
            )
        snapshot = readiness_snapshot(resolved_paths)
        return _host_status_from_snapshot(resolved_paths, snapshot, target)
    except RuntimeExpectedError as exc:
        return _host_error_record(
            exc,
            target=target,
            runtime_dir=runtime_dir,
            manifest_path=manifest_path,
            paths=resolved_paths,
        )
    except OSError as exc:
        return _host_error_record(
            exc,
            target=target,
            runtime_dir=runtime_dir,
            manifest_path=manifest_path,
            paths=resolved_paths,
        )


# A short compatibility spelling for callers that already use ``status`` as
# the runtime boundary name.  The descriptive name above is canonical.
host_status = host_runtime_status


def _storage_contract_errors(manifest: Mapping[str, Any]) -> list[str]:
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping):
        return ["manifest.storage must be an object"]
    errors: list[str] = []
    if "min_free_bytes" in storage:
        errors.append("flat storage.min_free_bytes is unsupported; use phase-aware storage buckets")
    if storage.get("contract_version") != STORAGE_CONTRACT_VERSION:
        errors.append(f"manifest.storage.contract_version must be {STORAGE_CONTRACT_VERSION}")
    phases = storage.get("phases")
    if not isinstance(phases, list) or not phases:
        return errors + ["manifest.storage.phases must be a non-empty list"]
    model = manifest.get("sources", {}).get("model", {}) if isinstance(manifest.get("sources"), Mapping) else {}
    model_files = model.get("files", []) if isinstance(model, Mapping) else []
    phase_names: set[str] = set()
    for phase in phases:
        if not isinstance(phase, Mapping) or not isinstance(phase.get("name"), str):
            errors.append("every storage phase must have a name")
            continue
        if phase["name"] in phase_names:
            errors.append(f"duplicate storage phase: {phase['name']}")
        phase_names.add(phase["name"])
        buckets = phase.get("buckets")
        if not isinstance(buckets, list) or not buckets:
            errors.append(f"storage phase {phase['name']} must contain buckets")
            continue
        for bucket in buckets:
            if not isinstance(bucket, Mapping):
                errors.append(f"storage phase {phase['name']} contains an invalid bucket")
                continue
            if not isinstance(bucket.get("name"), str) or not bucket["name"]:
                errors.append(f"storage phase {phase['name']} has an unnamed bucket")
            if bucket.get("destination") not in {"environment", "verified_model_root", "state", "run"}:
                errors.append(f"storage bucket {bucket.get('name')} has an invalid destination")
            source = bucket.get("source")
            required = bucket.get("required_bytes")
            if source == "model_files":
                if required is not None:
                    errors.append(f"storage bucket {bucket.get('name')} must derive model file bytes")
                invalid_model_sizes = any(
                    not isinstance(item, Mapping)
                    or not isinstance(item.get("size_bytes"), int)
                    or item["size_bytes"] < 0
                    for item in model_files
                )
                if not model_files or invalid_model_sizes:
                    errors.append(f"storage bucket {bucket.get('name')} requires valid model file sizes")
            elif required is None:
                if bucket.get("measurement_status") != "required_clean_disposable":
                    errors.append(f"storage bucket {bucket.get('name')} has no reproducible measurement")
            elif not isinstance(required, int) or required < 0:
                errors.append(f"storage bucket {bucket.get('name')} required_bytes must be non-negative")
            status = bucket.get("measurement_status")
            evidence = bucket.get("measurement_evidence")
            if status in {"measured_clean_disposable", "not_applicable"}:
                if not isinstance(evidence, Mapping):
                    errors.append(f"storage bucket {bucket.get('name')} lacks measurement evidence")
                    continue
                if evidence.get("schema") != STORAGE_MEASUREMENT_EVIDENCE_SCHEMA:
                    errors.append(f"storage bucket {bucket.get('name')} has invalid measurement evidence schema")
                if not isinstance(evidence.get("evidence_id"), str) or not evidence["evidence_id"].strip():
                    errors.append(f"storage bucket {bucket.get('name')} has no measurement evidence ID")
                measured = evidence.get("measured_bytes")
                if not isinstance(measured, int) or measured < 0:
                    errors.append(f"storage bucket {bucket.get('name')} has invalid measured bytes")
                if not isinstance(evidence.get("runs"), int) or evidence["runs"] < 1:
                    errors.append(f"storage bucket {bucket.get('name')} has invalid measurement run count")
                if status == "not_applicable" and (
                    not isinstance(evidence.get("basis"), str) or not evidence["basis"].strip()
                ):
                    errors.append(f"storage bucket {bucket.get('name')} lacks a not-applicable basis")
                if isinstance(required, int) and isinstance(measured, int) and required < measured:
                    errors.append(f"storage bucket {bucket.get('name')} budget is below measured bytes")
    return errors


def _has_measured_zero_provenance(bucket: Mapping[str, Any]) -> bool:
    """Accept a zero-byte bucket only when a reproducible observation proves it."""

    evidence = bucket.get("measurement_evidence")
    return (
        bucket.get("measurement_status") in {"measured_clean_disposable", "not_applicable"}
        and isinstance(evidence, Mapping)
        and evidence.get("schema") == STORAGE_MEASUREMENT_EVIDENCE_SCHEMA
        and isinstance(evidence.get("evidence_id"), str)
        and bool(evidence["evidence_id"].strip())
        and evidence.get("measured_bytes") == 0
        and isinstance(evidence.get("runs"), int)
        and evidence["runs"] >= 1
        and (
            bucket.get("measurement_status") != "not_applicable"
            or (isinstance(evidence.get("basis"), str) and bool(evidence["basis"].strip()))
        )
    )


def verified_file_credit(root: Path, files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Credit only regular files whose relative path, size, and SHA-256 match."""

    credited_bytes = 0
    verified: list[str] = []
    rejected: list[dict[str, str]] = []
    for item in files:
        relative = PurePosixPath(str(item.get("path", "")))
        if not item.get("path") or relative.is_absolute() or ".." in relative.parts:
            rejected.append({"path": str(item.get("path", "")), "reason": "unsafe_path"})
            continue
        candidate = root.joinpath(*relative.parts)
        if not _is_within(candidate, root):
            rejected.append({"path": relative.as_posix(), "reason": "unsafe_path"})
            continue
        if not _regular_file(candidate):
            rejected.append({"path": relative.as_posix(), "reason": "missing_or_not_regular"})
            continue
        expected_size = item.get("size_bytes")
        try:
            actual_size = candidate.stat().st_size
        except OSError:
            rejected.append({"path": relative.as_posix(), "reason": "stat_failed"})
            continue
        if not isinstance(expected_size, int) or actual_size != expected_size:
            rejected.append({"path": relative.as_posix(), "reason": "size_mismatch"})
            continue
        try:
            actual_sha256 = sha256_file(candidate)
        except OSError:
            rejected.append({"path": relative.as_posix(), "reason": "hash_failed"})
            continue
        if actual_sha256.lower() != str(item.get("sha256", "")).lower():
            rejected.append({"path": relative.as_posix(), "reason": "sha256_mismatch"})
            continue
        credited_bytes += actual_size
        verified.append(relative.as_posix())
    return {"credited_bytes": credited_bytes, "verified": verified, "rejected": rejected}


def _filesystem_location(path: Path) -> tuple[str, Path]:
    existing = _nearest_existing(path)
    if existing is None:
        raise OSError(f"no existing ancestor for {path}")
    return f"device:{existing.stat().st_dev}", existing


def calculate_storage_plan(
    paths: RuntimePaths,
    manifest: Mapping[str, Any],
    *,
    phases: Sequence[str] | None = None,
    verified_model_root: Path | None = None,
    filesystem_resolver: Callable[[Path], tuple[str, Path]] | None = None,
    free_bytes_provider: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    """Calculate selected phase peaks against receipt-selected artifact paths."""

    contract_errors = _storage_contract_errors(manifest)
    if contract_errors:
        return {"ok": False, "status": "invalid_storage_contract", "errors": contract_errors, "phases": [], "filesystems": {}}

    resolver = filesystem_resolver or _filesystem_location
    free_provider = free_bytes_provider or (lambda path: shutil.disk_usage(path).free)
    destination_paths: dict[str, tuple[Path, Path | None]] = {
        "environment": (paths.environment, paths.environment),
        "verified_model_root": (
            verified_model_root or paths.verified_model_root or paths.model_objects_dir,
            verified_model_root or paths.verified_model_root,
        ),
        "state": (paths.state_dir, paths.state_dir),
        "run": (paths.run_namespace, paths.run_namespace),
    }
    model = manifest.get("sources", {}).get("model", {})
    model_files = model.get("files", []) if isinstance(model, Mapping) else []
    phase_reports: list[dict[str, Any]] = []
    filesystem_paths: dict[str, Path] = {}
    filesystem_phase_bytes: dict[str, list[int]] = {}
    filesystem_unknowns: dict[str, list[str]] = {}
    errors: list[str] = []

    requested_phases = tuple(phases) if phases is not None else tuple(phase["name"] for phase in manifest["storage"]["phases"])
    available_phases = {phase["name"]: phase for phase in manifest["storage"]["phases"]}
    unknown_phases = [phase for phase in requested_phases if phase not in available_phases]
    if unknown_phases:
        return {
            "ok": False,
            "status": "invalid_storage_contract",
            "errors": [f"unknown storage phase: {phase}" for phase in unknown_phases],
            "selected_phases": list(requested_phases),
            "phases": [],
            "filesystems": {},
        }

    for phase_name in requested_phases:
        phase = available_phases[phase_name]
        grouped: dict[str, dict[str, Any]] = {}
        for bucket in phase["buckets"]:
            storage_probe_path, declared_destination = destination_paths[bucket["destination"]]
            try:
                filesystem_id, probe_path = resolver(storage_probe_path)
            except OSError as exc:
                errors.append(f"cannot resolve filesystem for {bucket['name']}: {exc}")
                continue
            filesystem_paths.setdefault(filesystem_id, probe_path)
            group = grouped.setdefault(filesystem_id, {"known_required_bytes": 0, "unknown_buckets": [], "buckets": []})
            source = bucket.get("source")
            credit = {"credited_bytes": 0, "verified": [], "rejected": []}
            if source == "model_files":
                required_bytes: int | None = sum(int(item["size_bytes"]) for item in model_files)
                if bucket.get("credit_verified_files", False) and verified_model_root is not None:
                    credit = verified_file_credit(verified_model_root, model_files)
            else:
                required_bytes = bucket.get("required_bytes")
                if required_bytes == 0 and not _has_measured_zero_provenance(bucket):
                    required_bytes = None
            if required_bytes is None:
                group["unknown_buckets"].append(bucket["name"])
                remaining_bytes = None
            else:
                remaining_bytes = max(0, required_bytes - int(credit["credited_bytes"]))
                group["known_required_bytes"] += remaining_bytes
            group["buckets"].append(
                {
                    "name": bucket["name"],
                    "destination": str(declared_destination) if declared_destination is not None else None,
                    "filesystem_probe_path": str(storage_probe_path),
                    "required_bytes": required_bytes,
                    "credited_bytes": credit["credited_bytes"],
                    "remaining_bytes": remaining_bytes,
                    "verified_files": credit["verified"],
                    "rejected_files": credit["rejected"],
                    "measurement_status": bucket.get("measurement_status", "declared"),
                    "measurement_evidence": bucket.get("measurement_evidence"),
                }
            )
        phase_reports.append({"name": phase["name"], "filesystems": grouped})
        for filesystem_id, group in grouped.items():
            filesystem_phase_bytes.setdefault(filesystem_id, []).append(group["known_required_bytes"])
            filesystem_unknowns.setdefault(filesystem_id, []).extend(group["unknown_buckets"])

    filesystem_reports: dict[str, Any] = {}
    has_unknown = False
    insufficient = False
    for filesystem_id, probe_path in filesystem_paths.items():
        known_required = max(filesystem_phase_bytes.get(filesystem_id, [0]))
        unknown_buckets = sorted(set(filesystem_unknowns.get(filesystem_id, [])))
        has_unknown = has_unknown or bool(unknown_buckets)
        try:
            available = int(free_provider(probe_path))
        except OSError as exc:
            errors.append(f"free-space check failed for {filesystem_id}: {exc}")
            available = None
        required = None if unknown_buckets else known_required
        shortfall = None if required is None or available is None else max(0, required - available)
        insufficient = insufficient or bool(shortfall)
        filesystem_reports[filesystem_id] = {
            "probe_path": str(probe_path),
            "known_required_bytes": known_required,
            "required_bytes": required,
            "available_bytes": available,
            "shortfall_bytes": shortfall,
            "unknown_buckets": unknown_buckets,
        }

    if errors:
        status = "storage_check_failed"
    elif has_unknown:
        status = "storage_budget_unknown"
    elif insufficient:
        status = "insufficient_storage"
    else:
        status = "sufficient"
    return {
        "ok": status == "sufficient",
        "status": status,
        "errors": errors,
        "selected_phases": list(requested_phases),
        "phases": phase_reports,
        "filesystems": filesystem_reports,
    }


def verify_lock(lock_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a hash-checked, immutable pip lock without invoking pip."""

    result: dict[str, Any] = {"path": str(lock_path), "ok": False, "sha256": None, "errors": []}
    errors: list[str] = result["errors"]
    if not _regular_file(lock_path):
        errors.append("lock is missing or is not a regular file")
        return result
    try:
        contents = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        errors.append("lock is not readable UTF-8 text")
        return result
    digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    result["sha256"] = digest
    lock_meta = manifest.get("lock")
    if not isinstance(lock_meta, dict):
        errors.append("manifest.lock is required")
    else:
        expected = lock_meta.get("sha256")
        if not isinstance(expected, str) or not _HEX64.fullmatch(expected):
            errors.append("manifest.lock.sha256 is unresolved")
        elif expected.lower() != digest.lower():
            errors.append("lock SHA-256 does not match manifest")
        if lock_meta.get("status") != "verified":
            errors.append("manifest.lock.status must be verified")

    lines = contents.splitlines()
    if "# e2a-lock-format: 1" not in lines:
        errors.append("lock must declare '# e2a-lock-format: 1'")
    if "--require-hashes" not in lines:
        errors.append("lock must enable --require-hashes")
    requirements = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            if line not in {
                "--require-hashes",
                "--index-url https://pypi.org/simple",
                "--extra-index-url https://download.pytorch.org/whl/cpu",
            }:
                errors.append(f"unsupported or unsafe lock option: {line.split()[0]}")
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in _UNRESOLVED_MARKERS):
            errors.append("lock contains mutable or unresolved identity")
        if " @ " in line or line.startswith(("git+", "http://", "https://")):
            errors.append("lock must not contain direct mutable URL/VCS requirements")
        requirement_part = line.split(";", 1)[0]
        if not _PINNED.search(requirement_part):
            errors.append("every lock requirement must be exactly version-pinned")
        if not _HASH.search(line):
            errors.append("every lock requirement must contain a SHA-256 hash")
        requirements += 1
    if requirements == 0:
        errors.append("lock contains no requirements")
    result["requirements"] = requirements
    result["ok"] = not errors
    return result


def _path_checks(paths: RuntimePaths) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    expected_root = (paths.data_home / "ebook2audiobook/chatterbox").resolve()
    if not _is_within(paths.environment, expected_root):
        errors.append("environment is outside the declared user-local data path")
    if not _is_within(paths.runtime_objects_dir, paths.env_base):
        errors.append("runtime object namespace is outside the environment base")
    if not _is_within(paths.runtime_receipt_path, paths.state_dir):
        errors.append("runtime receipt is outside the component state path")
    if not _is_within(paths.model_objects_dir, expected_root):
        errors.append("model object namespace is outside the declared user-local data path")
    if not _is_within(paths.model_receipt_path, paths.state_dir):
        errors.append("model receipt is outside the component state path")
    if not _is_within(paths.activation_receipt_path, paths.state_dir):
        errors.append("activation receipt is outside the component state path")
    for label, path in {
        "data_home": paths.data_home,
        "state_home": paths.state_home,
        "models_dir": paths.models_dir,
        "run_dir": paths.run_dir,
        "environment_parent": paths.env_base,
        "runtime_objects_parent": paths.runtime_objects_dir,
        "model_objects_parent": paths.model_objects_dir,
        "legacy_model_cache_parent": paths.legacy_model_cache,
        "run_namespace_parent": paths.run_namespace,
        "state_parent": paths.state_dir,
    }.items():
        ok, detail = _writable_location(path)
        checks[label] = {"ok": ok, "nearest_existing": detail}
        if not ok:
            errors.append(f"{label}: {detail}")
    return checks, errors


def validate_measurement_scope(paths: RuntimePaths, root: Path) -> Path:
    """Validate the complete mutable runtime surface inside a marked root.

    This is the only authority that permits unresolved storage budgets.  The
    manifest, source tree, and interpreter may remain outside the root as
    read-only inputs; every path the installer can create or publish must be
    rooted below it and the root must be owned by the invoking user.
    """

    requested = Path(root)
    if not requested.is_absolute() or ".." in requested.parts:
        raise RuntimeConfigurationError(
            "measurement root must be absolute without traversal"
        )
    if not requested.is_dir() or requested.is_symlink():
        raise RuntimeConfigurationError(
            "measurement root must be an existing real directory"
        )
    resolved = requested.resolve(strict=True)
    if requested != resolved:
        raise RuntimeConfigurationError("measurement root must use its canonical path")
    try:
        metadata = resolved.stat()
    except OSError as exc:
        raise RuntimeConfigurationError("measurement root ownership is unreadable") from exc
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise RuntimeConfigurationError("measurement root is not owned by the invoking user")
    marker = resolved / DISPOSABLE_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeConfigurationError("measurement root marker is missing")
    try:
        marker_content = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeConfigurationError("measurement root marker is unreadable") from exc
    if marker_content != DISPOSABLE_MARKER_CONTENT:
        raise RuntimeConfigurationError("measurement root marker is invalid")

    mutable_paths = {
        "data_home": paths.data_home,
        "state_home": paths.state_home,
        "e2a_root": paths.e2a_root,
        "models_dir": paths.models_dir,
        "run_dir": paths.run_dir,
        "model_namespace": paths.model_namespace,
        "legacy_model_cache": paths.legacy_model_cache,
        "model_objects_dir": paths.model_objects_dir,
        "run_namespace": paths.run_namespace,
        "state_dir": paths.state_dir,
        "env_base": paths.env_base,
        "runtime_objects_dir": paths.runtime_objects_dir,
        "legacy_environment": paths.legacy_environment,
        "environment": paths.environment,
        "runtime_receipt_path": paths.runtime_receipt_path,
        "model_receipt_path": paths.model_receipt_path,
        "activation_receipt_path": paths.activation_receipt_path,
        "install_lock_path": paths.install_lock_path,
        "model_install_lock_path": paths.model_install_lock_path,
        "result_path": paths.result_path,
    }
    for label, candidate in mutable_paths.items():
        candidate = Path(candidate)
        if not _is_within(candidate, resolved):
            raise RuntimeConfigurationError(
                f"measurement {label} escapes the disposable root"
            )
        current = resolved
        try:
            relative_parts = candidate.resolve(strict=False).relative_to(resolved).parts
        except ValueError as exc:
            raise RuntimeConfigurationError(
                f"measurement {label} escapes the disposable root"
            ) from exc
        for part in relative_parts:
            current = current / part
            try:
                if current.is_symlink():
                    raise RuntimeConfigurationError(
                        f"measurement {label} contains a symlink component"
                    )
            except OSError as exc:
                raise RuntimeConfigurationError(
                    f"measurement {label} cannot be inspected safely"
                ) from exc
    return resolved


def _validate_measurement_child(root: Path, path: Path, label: str, *, directory: bool = False) -> Path:
    """Validate one additional disposable input without following symlinks."""

    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts or not _is_within(candidate, root):
        raise RuntimeConfigurationError(f"measurement {label} escapes the disposable root")
    relative = candidate.resolve(strict=False).relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise RuntimeConfigurationError(
                    f"measurement {label} contains a symlink component"
                )
        except OSError as exc:
            raise RuntimeConfigurationError(
                f"measurement {label} cannot be inspected safely"
            ) from exc
    if directory and (candidate.is_symlink() or not candidate.is_dir()):
        raise RuntimeConfigurationError(f"measurement {label} must be a directory")
    return candidate


def _interpreter_checks(paths: RuntimePaths, interpreter: Path) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {"path": str(interpreter), "ok": False}
    errors: list[str] = []
    candidate = Path(shutil.which(str(interpreter)) or str(interpreter)).expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        errors.append("explicit interpreter is not executable")
        return checks, errors
    probe, probe_error = _probe_interpreter(candidate)
    if probe_error:
        errors.append(probe_error)
        return checks, errors
    checks["probe"] = {"version": probe.get("version"), "prefix": probe.get("prefix"), "base_prefix": probe.get("base_prefix")}
    if probe.get("version", [None, None])[:2] != list(TARGET_PYTHON):
        errors.append("explicit interpreter is not Python 3.11")
    if probe.get("prefix") != probe.get("base_prefix"):
        errors.append("explicit interpreter is already inside a virtual environment")
    forbidden = [paths.repo_root / "python_env"]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if conda_prefix:
        forbidden.append(Path(conda_prefix).expanduser())
    if virtual_env:
        forbidden.append(Path(virtual_env).expanduser())
    if any(_is_within(candidate, item) for item in forbidden):
        errors.append("explicit interpreter belongs to the host/root environment")
    for module in ("venv", "pip"):
        try:
            completed = subprocess.run(
                [str(candidate), "-m", module, "--help" if module == "venv" else "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=15,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            completed = None
            errors.append(f"{module} probe failed: {type(exc).__name__}")
        if completed is not None and completed.returncode != 0:
            errors.append(f"python -m {module} probe failed")
    checks["ok"] = not errors
    return checks, errors


def preflight(
    paths: RuntimePaths,
    interpreter: Path,
    *,
    storage_phases: Sequence[str] = RUNTIME_INSTALL_STORAGE_PHASES,
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
    check_network: bool = True,
    measurement_root: Path | None = None,
) -> dict[str, Any]:
    """Perform all read-only gates and return a safe JSON-serializable report."""

    checks: dict[str, Any] = {
        "target": {
            "os": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "expected": {"os": TARGET_OS, "architecture": TARGET_ARCH, "python": "3.11.x", "backend": TARGET_BACKEND},
        }
    }
    errors: list[str] = []
    measurement_mode = measurement_root is not None
    measurement_scope_valid = False
    if measurement_mode:
        try:
            measurement_root = validate_measurement_scope(paths, Path(measurement_root))
            measurement_scope_valid = True
        except RuntimeConfigurationError as exc:
            errors.append(str(exc))
    if hasattr(os, "geteuid") and os.geteuid() == 0 and not measurement_scope_valid:
        errors.append("runtime provisioning must not run as root")
    if platform.system().lower() != TARGET_OS:
        errors.append("host OS is not Linux")
    machine = platform.machine().lower()
    if machine not in {TARGET_ARCH, "amd64"}:
        errors.append("host architecture is not x86_64")

    try:
        manifest = _read_json(paths.manifest_path)
        identity_check = validate_manifest_identity(manifest)
        identity_errors = identity_check["errors"]
    except RuntimeConfigurationError as exc:
        manifest = {}
        identity_errors = [str(exc)]
    checks["manifest"] = {"path": str(paths.manifest_path), "sha256": paths.manifest_sha256, "ok": not identity_errors}
    errors.extend(identity_errors)

    interpreter_check, interpreter_errors = _interpreter_checks(paths, interpreter)
    checks["interpreter"] = interpreter_check
    errors.extend(interpreter_errors)

    path_check, path_errors = _path_checks(paths)
    checks["paths"] = path_check
    errors.extend(path_errors)

    lock_check = verify_lock(paths.lock_path, manifest)
    checks["lock"] = lock_check
    errors.extend(lock_check["errors"])

    storage_check = calculate_storage_plan(paths, manifest, phases=storage_phases)
    checks["storage"] = storage_check
    if not storage_check["ok"]:
        errors.extend(storage_check["errors"])
        if storage_check["status"] == "storage_budget_unknown":
            known_shortfall = any(
                isinstance(report, Mapping)
                and isinstance(report.get("available_bytes"), int)
                and isinstance(report.get("known_required_bytes"), int)
                and report["available_bytes"] < report["known_required_bytes"]
                for report in storage_check.get("filesystems", {}).values()
            )
            if measurement_scope_valid and not known_shortfall and not any(
                error for error in errors if "measurement root" in error
            ):
                checks["storage"] = {
                    **storage_check,
                    "measurement_mode": "unresolved_budgets_allowed",
                }
            else:
                errors.append(
                    "storage_budget_unknown: clean disposable measurements are required"
                )
        elif storage_check["status"] == "insufficient_storage":
            errors.append("insufficient free disk space for the declared runtime phases")
        elif storage_check["status"] == "invalid_storage_contract":
            errors.append("runtime storage contract is invalid")

    network_urls = manifest.get("network", {}).get("first_acquisition_urls", []) if isinstance(manifest.get("network"), dict) else []
    if storage_check.get("status") == "storage_budget_unknown" and not measurement_scope_valid:
        checks["network"] = {
            "required": bool(network_urls),
            "attempted": False,
            "status": "skipped",
            "reason": "storage_budget_unknown",
            "results": {},
            "ok": False,
        }
        return {"ok": False, "errors": errors, "checks": checks, "fingerprint": paths.fingerprint}
    if measurement_scope_valid:
        checks["network"] = {
            "required": False,
            "attempted": False,
            "status": "skipped",
            "reason": "disposable_measurement_uses_verified_offline_inputs",
            "results": {},
            "ok": True,
        }
        return {"ok": not errors, "errors": errors, "checks": checks, "fingerprint": paths.fingerprint}
    if not check_network:
        checks["network"] = {
            "required": False,
            "attempted": False,
            "status": "deferred",
            "reason": "serialized_preflight_pending",
            "results": {},
            "ok": True,
        }
        return {"ok": not errors, "errors": errors, "checks": checks, "fingerprint": paths.fingerprint}

    network_results: dict[str, Any] = {}
    checker = network_checker or (lambda url: _network_reachable(url, network_timeout))
    for url in network_urls:
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append("network check contains a non-HTTPS URL")
            continue
        ok, detail = checker(url)
        network_results[url] = {"ok": bool(ok), "detail": detail}
        if not ok:
            errors.append(f"first-acquisition network unavailable: {url}")
    network_ok = not any(not item["ok"] for item in network_results.values())
    checks["network"] = {
        "required": bool(network_urls),
        "attempted": bool(network_urls),
        "status": "passed" if network_ok else "failed",
        "reason": None,
        "results": network_results,
        "ok": network_ok,
    }

    return {"ok": not errors, "errors": errors, "checks": checks, "fingerprint": paths.fingerprint}


def model_preflight(
    paths: RuntimePaths,
    *,
    runtime_result: Mapping[str, Any] | None = None,
    model_result: Mapping[str, Any] | None = None,
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
    check_network: bool = True,
    measurement_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only gates for explicit model acquisition and activation."""

    errors: list[str] = []
    measurement_mode = measurement_root is not None
    measurement_scope_valid = False
    if measurement_mode:
        try:
            measurement_root = validate_measurement_scope(paths, Path(measurement_root))
            measurement_scope_valid = True
        except RuntimeConfigurationError as exc:
            errors.append(str(exc))
    try:
        manifest = _read_json(paths.manifest_path)
        identity = validate_manifest_identity(manifest)
    except RuntimeConfigurationError as exc:
        manifest = {}
        identity = {"ok": False, "errors": [str(exc)]}
    errors.extend(identity["errors"])
    runtime = dict(runtime_result) if runtime_result is not None else runtime_status(paths)
    if not runtime.get("ok"):
        errors.append(f"runtime is not ready: {runtime.get('status')}")
    model_paths = _receipt_selected_paths(paths, runtime, {"ok": False})
    model = (
        dict(model_result)
        if model_result is not None
        else model_status(model_paths)
    )
    selected_paths = _receipt_selected_paths(paths, runtime, model)
    storage = calculate_storage_plan(
        selected_paths,
        manifest,
        phases=MODEL_ACQUISITION_STORAGE_PHASES,
        verified_model_root=selected_paths.verified_model_root,
    )
    if not storage.get("ok"):
        errors.extend(storage.get("errors", []))
        if storage.get("status") == "storage_budget_unknown":
            known_shortfall = any(
                isinstance(report, Mapping)
                and isinstance(report.get("available_bytes"), int)
                and isinstance(report.get("known_required_bytes"), int)
                and report["available_bytes"] < report["known_required_bytes"]
                for report in storage.get("filesystems", {}).values()
            )
            if not measurement_scope_valid or known_shortfall or any(
                error for error in errors if "measurement root" in error
            ):
                errors.append(
                    "storage_budget_unknown: clean model/activation measurements are required"
                )
    network_urls = manifest.get("network", {}).get("model_acquisition_urls", []) if isinstance(manifest.get("network"), Mapping) else []
    network_results: dict[str, Any] = {}
    if storage.get("status") == "storage_budget_unknown" and not measurement_scope_valid:
        network = {
            "required": bool(network_urls),
            "attempted": False,
            "status": "skipped",
            "reason": "storage_budget_unknown",
            "results": network_results,
            "ok": False,
        }
    elif measurement_scope_valid:
        network = {
            "required": False,
            "attempted": False,
            "status": "skipped",
            "reason": "disposable_measurement_uses_verified_offline_inputs",
            "results": network_results,
            "ok": True,
        }
    elif not check_network:
        network = {
            "required": False,
            "attempted": False,
            "status": "deferred",
            "reason": "serialized_preflight_pending",
            "results": network_results,
            "ok": True,
        }
    else:
        checker = network_checker or (lambda url: _network_reachable(url, network_timeout))
        for url in network_urls:
            if not isinstance(url, str) or not url.startswith("https://"):
                errors.append("model acquisition network check contains a non-HTTPS URL")
                continue
            ok, detail = checker(url)
            network_results[url] = {"ok": bool(ok), "detail": detail}
            if not ok:
                errors.append(f"model acquisition network unavailable: {url}")
        network_ok = all(item["ok"] for item in network_results.values())
        network = {
            "required": bool(network_urls),
            "attempted": bool(network_urls),
            "status": "passed" if network_ok else "failed",
            "reason": None,
            "results": network_results,
            "ok": network_ok,
        }
    return {
        "ok": not errors,
        "errors": errors,
        "checks": {
            "manifest": identity,
            "runtime": runtime,
            "storage": storage,
            "network": network,
        },
        "model_fingerprint": paths.model_fingerprint,
        "activation_fingerprint": paths.activation_fingerprint,
    }


def sanitized_worker_environment(paths: RuntimePaths, base_environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an allowlisted, post-acquisition local-only worker environment."""

    source = dict(os.environ if base_environment is None else base_environment)
    allowed = {key: value for key, value in source.items() if key in {"HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}}
    env_bin = paths.environment / "bin"
    worker_cache = paths.run_namespace / "cache"
    allowed["PATH"] = os.pathsep.join((str(env_bin), "/usr/local/bin", "/usr/bin", "/bin"))
    values = {
        "E2A_ROOT": str(paths.e2a_root),
        "E2A_MODELS_DIR": str(paths.models_dir),
        "E2A_RUN_DIR": str(paths.run_dir),
        "E2A_CHATTERBOX_RUN_DIR": str(paths.run_namespace),
        "HF_HOME": str(worker_cache / "hf"),
        "HF_HUB_CACHE": str(worker_cache / "hf/hub"),
        "HF_DATASETS_CACHE": str(worker_cache / "hf/datasets"),
        "HF_XET_CACHE": str(worker_cache / "hf/xet"),
        "HF_ASSETS_CACHE": str(worker_cache / "hf/assets"),
        "PKUSEG_HOME": str(worker_cache / "pkuseg"),
        "TORCH_HOME": str(worker_cache / "torch"),
        "TTS_CACHE": str(worker_cache / "tts"),
        "XDG_CACHE_HOME": str(worker_cache / "xdg"),
        "XDG_CONFIG_HOME": str(paths.state_dir / "config"),
        "TMPDIR": str(paths.run_namespace / "tmp"),
        "GRADIO_TEMP_DIR": str(paths.run_namespace / "gradio"),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "E2A_CHATTERBOX_MODEL_RECEIPT": str(paths.model_receipt_path),
        "E2A_CHATTERBOX_ACTIVATION_RECEIPT": str(paths.activation_receipt_path),
        "E2A_CHATTERBOX_RUNTIME_RECEIPT": str(paths.runtime_receipt_path),
    }
    allowed.update(values)
    return allowed


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep result records limited to non-content installation facts."""

    checks = result.get("checks", {})
    return {
        "schema": "ebook2audiobook.chatterbox-installation-result.v1",
        "status": result.get("status", "passed"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"os": TARGET_OS, "architecture": TARGET_ARCH, "python": "3.11.x", "backend": TARGET_BACKEND},
        "environment": result.get("environment"),
        "fingerprint": result.get("fingerprint"),
        "manifest_sha256": result.get("manifest_sha256"),
        "lock_sha256": result.get("lock_sha256"),
        "checks": {
            key: value if key in {"pip_check", "imports", "worker_self_test"} else {"ok": bool(value.get("ok"))} if isinstance(value, dict) else bool(value)
            for key, value in checks.items()
            if key in {"pip_check", "imports", "worker_self_test"}
        },
    }


def write_result(paths: RuntimePaths, result: Mapping[str, Any]) -> Path:
    """Write one 0600 result record, refusing to overwrite an existing path."""

    if paths.result_path.exists():
        raise ProvisioningError("installation result already exists; refusing to overwrite it")
    paths.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(_safe_result(result), indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix="installation-result.", suffix=".tmp", dir=paths.state_dir)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, paths.result_path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return paths.result_path


def write_runtime_receipt(paths: RuntimePaths, receipt: Mapping[str, Any]) -> Path:
    """Publish the runtime receipt once, failing closed on uncertain durability."""

    publication = _runtime_receipt_publication_path(paths)
    published_proof = _published_receipt_path(paths.runtime_receipt_path)
    ambiguity = _ambiguous_receipt_path(paths.runtime_receipt_path)
    if (
        paths.runtime_receipt_path.exists()
        or _path_entry_exists(publication)
        or _path_entry_exists(published_proof)
        or _path_entry_exists(ambiguity)
    ):
        raise ProvisioningError("runtime receipt already exists; refusing to overwrite it")
    paths.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"runtime-receipt-{paths.runtime_fingerprint}.",
        suffix=".tmp",
        dir=paths.state_dir,
    )
    temporary = Path(temporary_name)
    published = False
    publication_owned = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt_sha256 = sha256_file(temporary)
        try:
            _write_json_file(
                publication,
                {
                    "schema": "ebook2audiobook.chatterbox-runtime-receipt-publication.v1",
                    "status": "published",
                    "runtime_fingerprint": paths.runtime_fingerprint,
                    "receipt": str(paths.runtime_receipt_path),
                    "receipt_sha256": receipt_sha256,
                },
                exclusive=True,
            )
        except FileExistsError as exc:
            raise ProvisioningError("runtime receipt publication is already in progress") from exc
        publication_owned = True
        try:
            os.link(temporary, paths.runtime_receipt_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProvisioningError("runtime receipt already exists; refusing to overwrite it") from exc
        published = True
        _AMBIGUOUS_RECEIPT_PUBLICATIONS.add(paths.runtime_receipt_path)
        temporary.unlink()
        _fsync_directory(paths.state_dir)
        _ensure_receipt_ambiguity_marker(paths.runtime_receipt_path, label="runtime")
        os.replace(publication, published_proof)
        _fsync_directory(paths.state_dir)
        ambiguity.unlink()
        _fsync_directory(paths.state_dir)
        _AMBIGUOUS_RECEIPT_PUBLICATIONS.discard(paths.runtime_receipt_path)
    except Exception as exc:
        if not published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if publication_owned:
                try:
                    publication.unlink(missing_ok=True)
                    _fsync_directory(paths.state_dir)
                except OSError:
                    pass
        elif _path_entry_exists(published_proof) and not _path_entry_exists(publication):
            rollback_errors = _restore_receipt_quarantine(
                paths.runtime_receipt_path,
                publication,
                published_proof,
                label="runtime receipt",
            )
            if rollback_errors and hasattr(exc, "add_note"):
                exc.add_note("; ".join(rollback_errors))
        raise
    return paths.runtime_receipt_path


def _write_bound_receipt(
    receipt_path: Path,
    publication: Path,
    receipt: Mapping[str, Any],
    *,
    label: str,
    fingerprint_field: str,
    fingerprint: str,
) -> Path:
    """Publish one immutable receipt and leave ambiguity visibly fail-closed."""

    published_proof = _published_receipt_path(receipt_path)
    ambiguity = _ambiguous_receipt_path(receipt_path)
    if (
        receipt_path.exists()
        or _path_entry_exists(publication)
        or _path_entry_exists(published_proof)
        or _path_entry_exists(ambiguity)
    ):
        raise ProvisioningError(f"{label} receipt already exists or is being published")
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{label}-receipt-{fingerprint}.", suffix=".tmp", dir=receipt_path.parent
    )
    temporary = Path(temporary_name)
    published = False
    publication_owned = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt_sha256 = sha256_file(temporary)
        try:
            _write_json_file(
                publication,
                {
                    "schema": f"ebook2audiobook.chatterbox-{label}-receipt-publication.v1",
                    "status": "published",
                    fingerprint_field: fingerprint,
                    "receipt": str(receipt_path),
                    "receipt_sha256": receipt_sha256,
                },
                exclusive=True,
            )
        except FileExistsError as exc:
            raise ProvisioningError(f"{label} receipt publication is already in progress") from exc
        publication_owned = True
        try:
            os.link(temporary, receipt_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProvisioningError(f"{label} receipt already exists; refusing to overwrite it") from exc
        published = True
        _AMBIGUOUS_RECEIPT_PUBLICATIONS.add(receipt_path)
        temporary.unlink()
        _fsync_directory(receipt_path.parent)
        _ensure_receipt_ambiguity_marker(receipt_path, label=label)
        os.replace(publication, published_proof)
        _fsync_directory(receipt_path.parent)
        ambiguity.unlink()
        _fsync_directory(receipt_path.parent)
        _AMBIGUOUS_RECEIPT_PUBLICATIONS.discard(receipt_path)
    except Exception as exc:
        if not published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if publication_owned:
                try:
                    publication.unlink(missing_ok=True)
                    _fsync_directory(receipt_path.parent)
                except OSError:
                    pass
        elif _path_entry_exists(published_proof) and not _path_entry_exists(publication):
            rollback_errors = _restore_receipt_quarantine(
                receipt_path,
                publication,
                published_proof,
                label=f"{label} receipt",
            )
            if rollback_errors and hasattr(exc, "add_note"):
                exc.add_note("; ".join(rollback_errors))
        raise
    return receipt_path


def write_model_receipt(paths: RuntimePaths, receipt: Mapping[str, Any]) -> Path:
    return _write_bound_receipt(
        paths.model_receipt_path,
        _model_receipt_publication_path(paths),
        receipt,
        label="model",
        fingerprint_field="model_fingerprint",
        fingerprint=paths.model_fingerprint,
    )


def write_activation_receipt(paths: RuntimePaths, receipt: Mapping[str, Any]) -> Path:
    return _write_bound_receipt(
        paths.activation_receipt_path,
        _activation_receipt_publication_path(paths),
        receipt,
        label="activation",
        fingerprint_field="activation_fingerprint",
        fingerprint=paths.activation_fingerprint,
    )


def _model_file_url(manifest: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    model = manifest.get("sources", {}).get("model", {})
    if not isinstance(model, Mapping):
        raise RuntimeConfigurationError("manifest sources.model is required")
    repositories = _model_repositories(model)
    source = record.get("source")
    repository = repositories.get(str(source)) if isinstance(source, str) else None
    if not isinstance(repository, Mapping):
        raise RuntimeConfigurationError("model file source repository is invalid")
    locator = str(repository.get("locator", "")).rstrip("/")
    revision = str(repository.get("revision", ""))
    if not locator.startswith("https://huggingface.co/") or not _HEX40.fullmatch(revision):
        raise RuntimeConfigurationError("model locator or immutable revision is invalid")
    relative_path = str(record.get("path", ""))
    quoted = "/".join(
        urllib.parse.quote(part, safe="")
        for part in PurePosixPath(relative_path).parts
    )
    return f"{locator}/resolve/{revision}/{quoted}"


def _download_model_file(url: str, destination: Path, timeout: float = 1800.0) -> None:
    """Stream one allow-listed immutable artifact into an owned candidate."""

    if not url.startswith("https://"):
        raise ProvisioningError("model downloads require HTTPS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    request = urllib.request.Request(url, headers={"User-Agent": "ebook2audiobook-chatterbox-model/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)


def _model_receipt_payload(
    paths: RuntimePaths,
    candidate: Path,
    snapshot: Path,
    nonce: str,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": MODEL_RECEIPT_SCHEMA,
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint": paths.model_fingerprint,
        "manifest_sha256": paths.manifest_sha256,
        "artifact": {
            "object_path": str(candidate),
            "snapshot_path": str(snapshot),
            "ownership_nonce": nonce,
            "owner_marker_sha256": sha256_file(_model_owner_marker_path(candidate)),
        },
        "files": [dict(item) for item in files],
        "checks": {
            "acquisition": {"ok": True, "files": len(files)},
            "snapshot_validation": {"ok": True},
            "receipt_validation": {"ok": True},
        },
    }


def _activation_receipt_payload(
    paths: RuntimePaths,
    runtime: Mapping[str, Any],
    model: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "schema": ACTIVATION_RECEIPT_SCHEMA,
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activation_fingerprint": paths.activation_fingerprint,
        "runtime_fingerprint": paths.runtime_fingerprint,
        "model_fingerprint": paths.model_fingerprint,
        "manifest_sha256": paths.manifest_sha256,
        "runtime_receipt_sha256": sha256_file(paths.runtime_receipt_path),
        "model_receipt_sha256": sha256_file(paths.model_receipt_path),
        "environment": runtime.get("environment"),
        "model_snapshot": model.get("snapshot"),
        "checks": {"local_model_load": {"ok": True, "status": detail}},
    }


@contextmanager
def _model_installer_lock(paths: RuntimePaths):
    if fcntl is None:
        raise ProvisioningError("model acquisition locking requires Linux flock support")
    paths.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        paths.model_install_lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def acquire_model(
    paths: RuntimePaths,
    *,
    worker_script: Path | None = None,
    measurement_root: Path | None = None,
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
    preflight_runner: Callable[..., dict[str, Any]] = model_preflight,
    download_file: Callable[[str, Path], None] = _download_model_file,
    activation_hook: Callable[[Path, Sequence[str], Mapping[str, str]], tuple[bool, str]] = _command_hook,
    worker_observer: Callable[[Mapping[str, Any]], None] | None = None,
    model_receipt_writer: Callable[[RuntimePaths, Mapping[str, Any]], Path] = write_model_receipt,
    activation_receipt_writer: Callable[[RuntimePaths, Mapping[str, Any]], Path] = write_activation_receipt,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
) -> dict[str, Any]:
    """Acquire, verify, and activate the selected immutable multilingual snapshot."""

    measurement_mode = measurement_root is not None
    if measurement_mode:
        try:
            measurement_root = validate_measurement_scope(paths, Path(measurement_root))
        except RuntimeConfigurationError as exc:
            raise ProvisioningError(str(exc)) from exc
        if download_file is _download_model_file:
            raise ProvisioningError(
                "disposable measurement requires an offline model copy hook"
            )

    def run_model_preflight(
        selected_paths: RuntimePaths,
        *,
        check_network: bool,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "network_checker": network_checker,
            "check_network": check_network,
        }
        if measurement_mode:
            arguments["measurement_root"] = measurement_root
        return preflight_runner(selected_paths, **arguments)

    runtime = runtime_status(paths)
    if not runtime.get("ok"):
        raise ProvisioningError(json.dumps(runtime, sort_keys=True))
    current_model = model_status(paths)
    current_activation = activation_status(paths)
    if current_model.get("ok") and current_activation.get("ok"):
        return {**product_status(paths), "ok": True, "status": "already_ready"}

    preflight_paths = _receipt_selected_paths(paths, runtime, current_model)
    report = run_model_preflight(preflight_paths, check_network=False)
    if not report.get("ok"):
        raise ProvisioningError(json.dumps(report, sort_keys=True))

    with _model_installer_lock(paths):
        runtime = runtime_status(paths)
        if not runtime.get("ok"):
            raise ProvisioningError(json.dumps(runtime, sort_keys=True))
        current_model = model_status(paths)
        current_activation = activation_status(paths)
        if current_model.get("status") == "repair_required" or current_activation.get("status") == "repair_required":
            raise ProvisioningError(json.dumps(product_status(paths), sort_keys=True))
        if current_model.get("ok") and current_activation.get("ok"):
            return {**product_status(paths), "ok": True, "status": "already_ready"}

        preflight_paths = _receipt_selected_paths(paths, runtime, current_model)
        report = run_model_preflight(
            preflight_paths,
            check_network=not current_model.get("ok"),
        )
        if not report.get("ok"):
            raise ProvisioningError(json.dumps(report, sort_keys=True))

        if not current_model.get("ok"):
            manifest = _read_json(paths.manifest_path)
            records = _model_file_records(manifest)
            paths.model_objects_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            nonce = nonce_factory()
            if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", nonce):
                raise ProvisioningError("model transaction nonce is invalid")
            candidate = paths.model_objects_dir / nonce
            snapshot = candidate / "snapshot"
            reserved = False
            try:
                candidate.mkdir(mode=0o700)
                _fsync_directory(paths.model_objects_dir)
                _write_json_file(
                    _model_owner_marker_path(candidate),
                    _model_owner_payload(paths, candidate, nonce),
                    exclusive=True,
                )
                reserved = True
                snapshot.mkdir(mode=0o700)
                for record in records:
                    relative = PurePosixPath(str(record["path"]))
                    destination = snapshot / Path(*relative.parts)
                    if not _is_within(destination, snapshot):
                        raise ProvisioningError("model destination escapes the owned snapshot")
                    download_file(_model_file_url(manifest, record), destination)
                    if destination.is_symlink() or not destination.is_file():
                        raise ProvisioningError(f"downloaded model artifact is not a regular file: {relative}")
                    if destination.stat().st_size != int(record["size_bytes"]):
                        raise ProvisioningError(f"downloaded model artifact has the wrong size: {relative}")
                    if sha256_file(destination).lower() != str(record["sha256"]).lower():
                        raise ProvisioningError(f"downloaded model artifact has the wrong checksum: {relative}")
                verification = _verified_snapshot_report(snapshot, records)
                if not verification["ok"]:
                    raise ProvisioningError(json.dumps(verification, sort_keys=True))
                receipt = _model_receipt_payload(paths, candidate, snapshot, nonce, verification["files"])
                validation = validate_model_receipt(paths, receipt)
                if not validation["ok"]:
                    raise ProvisioningError(json.dumps(validation, sort_keys=True))
                model_receipt_writer(paths, receipt)
                current_model = model_status(paths)
                if not current_model.get("ok"):
                    raise ProvisioningError(json.dumps(current_model, sort_keys=True))
            except BaseException as exc:
                if reserved and not paths.model_receipt_path.exists():
                    if not _cleanup_owned_model_candidate(paths, candidate, nonce):
                        raise ProvisioningError(
                            "model acquisition failed and owned candidate cleanup is incomplete; repair_required"
                        ) from exc
                elif not reserved and candidate.exists():
                    try:
                        candidate.rmdir()
                        _fsync_directory(paths.model_objects_dir)
                    except OSError:
                        raise ProvisioningError("model candidate reservation rollback failed; repair_required") from exc
                raise

        worker = (worker_script or (paths.repo_root / "components/Chatterbox/worker.py")).expanduser().resolve()
        if not _is_within(worker, paths.repo_root) or not _regular_file(worker):
            raise ProvisioningError("model activation worker is missing or outside the repository")
        current_model = model_status(paths)
        snapshot_value = current_model.get("snapshot")
        if not isinstance(snapshot_value, str):
            raise ProvisioningError("verified model snapshot is unavailable")
        environment = sanitized_worker_environment(replace(paths, model_namespace=Path(snapshot_value), verified_model_root=Path(snapshot_value)))
        args = [
            str(worker),
            "--model-self-test",
            "--model-manifest", str(paths.manifest_path),
            "--approved-model-root", snapshot_value,
            "--approved-manifest-root", str(paths.runtime_dir),
        ]
        if worker_observer is not None and activation_hook is _command_hook:
            activated, detail = _command_hook(
                Path(str(runtime["environment"])) / "bin/python",
                args,
                environment,
                observation_callback=worker_observer,
                observation_kind="model_self_test",
            )
        else:
            activated, detail = activation_hook(
                Path(str(runtime["environment"])) / "bin/python", args, environment
            )
        if not activated:
            raise ProvisioningError(detail)
        activation_receipt = _activation_receipt_payload(paths, runtime, current_model, detail)
        activation_validation = validate_activation_receipt(paths, activation_receipt)
        if not activation_validation["ok"]:
            raise ProvisioningError(json.dumps(activation_validation, sort_keys=True))
        activation_receipt_writer(paths, activation_receipt)
        ready = product_status(paths)
        if not ready["ok"]:
            raise ProvisioningError(json.dumps(ready, sort_keys=True))
        return ready


@contextmanager
def _installer_lock(paths: RuntimePaths):
    if fcntl is None:
        raise ProvisioningError("runtime installation locking requires Linux flock support")
    paths.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        paths.install_lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def install_runtime(
    paths: RuntimePaths,
    interpreter: Path,
    *,
    worker_script: Path | None = None,
    measurement_root: Path | None = None,
    wheelhouse: Path | None = None,
    pip_cache: Path | None = None,
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    import_hook: Callable[[Path, Sequence[str]], tuple[bool, str]] = _module_hook,
    self_test_hook: Callable[[Path, Sequence[str], Mapping[str, str]], tuple[bool, str]] = _command_hook,
    worker_observer: Callable[[Mapping[str, Any]], None] | None = None,
    receipt_writer: Callable[[RuntimePaths, Mapping[str, Any]], Path] = write_runtime_receipt,
    receipt_validator: Callable[[RuntimePaths, Mapping[str, Any]], dict[str, Any]] = validate_runtime_receipt,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
) -> dict[str, Any]:
    """Provision one immutable runtime object and publish its receipt.

    ``measurement_root`` is an explicit, disposable-only escape hatch for
    unresolved storage budgets.  It requires a complete local wheelhouse and
    keeps the ordinary host install path unchanged.
    """

    source = Path(shutil.which(str(interpreter)) or str(interpreter)).expanduser().resolve()
    measurement_mode = measurement_root is not None
    if measurement_mode:
        if wheelhouse is None or pip_cache is None:
            raise ProvisioningError(
                "disposable measurement requires a wheelhouse and pip cache"
            )
        try:
            measurement_root = validate_measurement_scope(paths, Path(measurement_root))
        except RuntimeConfigurationError as exc:
            raise ProvisioningError(str(exc)) from exc
        try:
            wheelhouse = _validate_measurement_child(
                measurement_root, Path(wheelhouse), "wheelhouse", directory=True
            )
            pip_cache = _validate_measurement_child(
                measurement_root, Path(pip_cache), "pip cache"
            )
        except RuntimeConfigurationError as exc:
            raise ProvisioningError(str(exc)) from exc
    elif wheelhouse is not None or pip_cache is not None:
        raise ProvisioningError(
            "wheelhouse and pip cache are only valid with a disposable measurement root"
        )
    existing = runtime_status(paths)
    if existing["status"] == "ready":
        return {**existing, "ok": True, "status": "already_ready"}

    report = preflight(
        paths,
        interpreter,
        network_checker=network_checker,
        check_network=False,
        measurement_root=measurement_root,
    )
    if not report["ok"]:
        raise ProvisioningError(json.dumps(report, sort_keys=True))

    with _installer_lock(paths):
        existing = runtime_status(paths)
        if existing["status"] == "ready":
            return {**existing, "ok": True, "status": "already_ready"}
        if existing["status"] == "repair_required":
            raise ProvisioningError(json.dumps(existing, sort_keys=True))

        report = preflight(
            paths,
            interpreter,
            network_checker=network_checker,
            check_network=not measurement_mode,
            measurement_root=measurement_root,
        )
        if not report["ok"]:
            raise ProvisioningError(json.dumps(report, sort_keys=True))

        paths.runtime_objects_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        nonce = nonce_factory()
        if not isinstance(nonce, str) or not re.fullmatch(r"[0-9A-Za-z._-]{8,128}", nonce):
            raise ProvisioningError("candidate nonce is invalid")
        candidate = paths.runtime_objects_dir / nonce
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ProvisioningError("candidate object already exists; refusing ambiguous path") from exc
        try:
            _fsync_directory(paths.runtime_objects_dir)
            marker = _owner_payload(paths, candidate, nonce)
            _write_json_file(_owner_marker_path(candidate), marker, exclusive=True)
        except BaseException as exc:
            rollback_errors = _rollback_candidate_reservation(candidate)
            if rollback_errors:
                raise ProvisioningError(
                    "candidate reservation failed and bounded rollback is incomplete; "
                    f"repair_required: {'; '.join(rollback_errors)}"
                ) from exc
            raise
        candidate_paths = replace(paths, environment=candidate)
        checks: dict[str, Any] = {}
        published = False
        try:
            try:
                venv = command_runner(
                    [str(source), "-m", "venv", str(candidate)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=300,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ProvisioningError(f"venv creation failed: {type(exc).__name__}") from exc
            if venv.returncode != 0:
                raise ProvisioningError("venv creation failed")
            checks["venv_creation"] = {"ok": True}

            env_python = candidate / "bin/python"
            if not _executable_venv_python(env_python, source):
                raise ProvisioningError("venv creation did not produce an executable Python")
            worker_env = sanitized_worker_environment(candidate_paths)
            pip_options: list[str] = []
            if measurement_mode:
                pip_cache = Path(pip_cache)
                pip_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
                worker_env.update(
                    {
                        "PIP_NO_INDEX": "1",
                        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                        "PIP_CACHE_DIR": str(pip_cache),
                        "PIP_CONFIG_FILE": os.devnull,
                    }
                )
                pip_options.extend(
                    [
                        "--no-index",
                        "--find-links",
                        str(Path(wheelhouse)),
                        "--cache-dir",
                        str(pip_cache),
                    ]
                )
            try:
                installed = command_runner(
                    [
                        str(env_python), "-m", "pip", "--isolated", "install",
                        "--disable-pip-version-check", "--no-input", "--require-virtualenv",
                        "--require-hashes", *pip_options, "-r", str(paths.lock_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=1800,
                    shell=False,
                    env=worker_env,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ProvisioningError(f"pip install failed: {type(exc).__name__}") from exc
            if installed.returncode != 0:
                raise ProvisioningError("pip install failed")
            checks["lock_install"] = {"ok": True}

            try:
                pip_check = command_runner(
                    [str(env_python), "-m", "pip", "--isolated", "check"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=120,
                    shell=False,
                    env=worker_env,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ProvisioningError(f"pip check failed: {type(exc).__name__}") from exc
            if pip_check.returncode != 0:
                raise ProvisioningError("pip check failed")
            checks["pip_check"] = {"ok": True}

            manifest = _read_json(paths.manifest_path)
            modules = manifest.get("validation", {}).get("import_modules", [])
            if not isinstance(modules, list) or not modules:
                raise ProvisioningError("manifest.validation.import_modules is required")
            imports_ok, imports_detail = import_hook(env_python, [str(module) for module in modules])
            if not imports_ok:
                raise ProvisioningError(imports_detail)
            checks["imports"] = {"ok": True, "modules": [str(module) for module in modules]}

            worker = (worker_script or (paths.repo_root / "components/Chatterbox/worker.py")).expanduser().resolve()
            if not _is_within(worker, paths.repo_root) or not _regular_file(worker):
                raise ProvisioningError("worker self-test path is missing or outside the repository")
            worker_args = manifest.get("validation", {}).get("worker_self_test_args", ["--self-test"])
            if not isinstance(worker_args, list) or any(not isinstance(item, str) for item in worker_args):
                raise ProvisioningError("worker self-test arguments are invalid")
            worker_command = [str(worker), *worker_args]
            if worker_observer is not None and self_test_hook is _command_hook:
                worker_ok, worker_detail = _command_hook(
                    env_python,
                    worker_command,
                    worker_env,
                    observation_callback=worker_observer,
                    observation_kind="runtime_self_test",
                )
            else:
                worker_ok, worker_detail = self_test_hook(env_python, worker_command, worker_env)
            if not worker_ok:
                raise ProvisioningError(worker_detail)
            checks["worker_self_test"] = {"ok": True, "status": worker_detail}

            checks["receipt_validation"] = {"ok": True}
            receipt = _runtime_receipt_payload(paths, candidate, nonce, checks)
            receipt_validation = receipt_validator(paths, receipt)
            if not receipt_validation["ok"]:
                raise ProvisioningError(json.dumps(receipt_validation, sort_keys=True))
            receipt_writer(paths, receipt)
            published = True
            final_validation = runtime_status(paths)
            if not final_validation["ok"]:
                raise ProvisioningError(json.dumps(final_validation, sort_keys=True))
            # The legacy result remains diagnostic only and cannot establish readiness.
            return {
                "ok": True,
                "status": "ready",
                "environment": str(candidate),
                "runtime_receipt": str(paths.runtime_receipt_path),
                "legacy_result": str(paths.result_path) if paths.result_path.exists() else None,
                "fingerprint": paths.runtime_fingerprint,
                "manifest_sha256": paths.manifest_sha256,
                "lock_sha256": paths.lock_sha256,
                "checks": checks,
            }
        except BaseException as exc:
            receipt_visible = paths.runtime_receipt_path.exists()
            if not published and not receipt_visible:
                if not _cleanup_owned_candidate(paths, candidate, nonce):
                    raise ProvisioningError(
                        "runtime provisioning failed and owned candidate cleanup is incomplete; repair_required"
                    ) from exc
            raise
