"""Safe, standard-library-only provisioning support for Chatterbox.

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
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RUNTIME_VERSION = "1"
TARGET_OS = "linux"
TARGET_ARCH = "x86_64"
TARGET_PYTHON = (3, 11)
TARGET_BACKEND = "cpu"
DEFAULT_MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_NETWORK_TIMEOUT = 8.0

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_PINNED = re.compile(r"(?:===|==)\s*[^\s;]+")
_HASH = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
_UNRESOLVED_MARKERS = ("<unresolved", "todo", "replace-me", "master", "@main")


class RuntimeErrorBase(RuntimeError):
    """Base error for safe runtime operations."""


class RuntimeConfigurationError(RuntimeErrorBase):
    """The runtime manifest or path configuration is unsafe or incomplete."""


class ProvisioningError(RuntimeErrorBase):
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
    model_namespace: Path
    run_namespace: Path
    state_dir: Path
    env_base: Path
    environment: Path
    result_path: Path
    manifest_path: Path
    lock_path: Path
    fingerprint: str
    manifest_sha256: str
    lock_sha256: str | None


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
        raise RuntimeConfigurationError(f"manifest is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError(f"cannot read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeConfigurationError("runtime manifest must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(manifest: Mapping[str, Any], lock_sha256: str | None) -> str:
    payload = {
        "runtime_version": RUNTIME_VERSION,
        "target": {
            "os": TARGET_OS,
            "architecture": TARGET_ARCH,
            "python": "3.11",
            "backend": TARGET_BACKEND,
        },
        "manifest": manifest,
        "lock_sha256": lock_sha256 or "missing",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"py311-linux-x86_64-cpu-{hashlib.sha256(encoded).hexdigest()[:16]}"


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
    fingerprint = _fingerprint(manifest_value, actual_lock_sha256)

    return RuntimePaths(
        runtime_dir=runtime,
        repo_root=repository,
        data_home=data_home,
        state_home=state_home,
        e2a_root=e2a_root,
        models_dir=models_dir,
        run_dir=run_dir,
        model_namespace=(models_dir / "tts/chatterbox").resolve(),
        run_namespace=(run_dir / "components/chatterbox").resolve(),
        state_dir=(state_home / "ebook2audiobook/chatterbox").resolve(),
        env_base=(data_home / "ebook2audiobook/chatterbox/envs").resolve(),
        environment=(data_home / "ebook2audiobook/chatterbox/envs" / fingerprint).resolve(),
        result_path=(state_home / f"ebook2audiobook/chatterbox/installation-result-{fingerprint}.json").resolve(),
        manifest_path=manifest,
        lock_path=declared_lock,
        fingerprint=fingerprint,
        manifest_sha256=manifest_sha256,
        lock_sha256=actual_lock_sha256,
    )


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


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


def _command_hook(interpreter: Path, args: Sequence[str], env: Mapping[str, str]) -> tuple[bool, str]:
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
        return False, f"self-test hook failed: {type(exc).__name__}"
    return completed.returncode == 0, "self-test hook passed" if completed.returncode == 0 else "self-test hook failed"


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
    if manifest.get("manifest_version") != "1.0.0":
        errors.append("manifest_version must be 1.0.0")
    target = manifest.get("target")
    if target != {"os": TARGET_OS, "architecture": TARGET_ARCH, "python": "3.11.x", "backend": TARGET_BACKEND}:
        errors.append("manifest target does not match Linux x86_64 Python 3.11 CPU")
    unresolved = manifest.get("unresolved_identities")
    if not isinstance(unresolved, list):
        errors.append("manifest.unresolved_identities must be a list")
    elif unresolved:
        errors.append(f"unresolved identities: {len(unresolved)}")

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return errors + ["manifest.sources must be an object"]
    required = {
        "chatterbox_package": ("artifact_sha256",),
        "chatterbox_source": ("revision",),
        "perth": ("commit", "artifact_sha256"),
        "model": ("revision", "files"),
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
        if name == "model" and isinstance(source.get("files"), list):
            if not source["files"]:
                errors.append("model.files must not be empty")
            for item in source["files"]:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not _HEX64.fullmatch(str(item.get("sha256", ""))):
                    errors.append("model file identity must contain path and SHA-256")
    return errors


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
    if paths.environment.exists():
        errors.append("environment path already exists; refusing unexpected existing path")
    if paths.result_path.exists():
        errors.append("installation result already exists; refusing to overwrite it")
    for label, path in {
        "data_home": paths.data_home,
        "state_home": paths.state_home,
        "models_dir": paths.models_dir,
        "run_dir": paths.run_dir,
        "environment_parent": paths.env_base,
        "model_namespace_parent": paths.model_namespace,
        "run_namespace_parent": paths.run_namespace,
        "state_parent": paths.state_dir,
    }.items():
        ok, detail = _writable_location(path)
        checks[label] = {"ok": ok, "nearest_existing": detail}
        if not ok:
            errors.append(f"{label}: {detail}")
    return checks, errors


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
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
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
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        errors.append("runtime provisioning must not run as root")
    if platform.system().lower() != TARGET_OS:
        errors.append("host OS is not Linux")
    machine = platform.machine().lower()
    if machine not in {TARGET_ARCH, "amd64"}:
        errors.append("host architecture is not x86_64")

    try:
        manifest = _read_json(paths.manifest_path)
        identity_errors = _identity_errors(manifest)
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

    minimum = DEFAULT_MIN_FREE_BYTES
    if isinstance(manifest.get("storage"), dict) and isinstance(manifest["storage"].get("min_free_bytes"), int):
        minimum = manifest["storage"]["min_free_bytes"]
    storage_path = _nearest_existing(paths.models_dir) or _nearest_existing(paths.data_home)
    storage_ok = False
    storage_detail: dict[str, Any] = {"path": str(storage_path) if storage_path else None, "minimum_free_bytes": minimum}
    if storage_path is None:
        errors.append("no existing path available for disk-space check")
    else:
        try:
            free_bytes = shutil.disk_usage(storage_path).free
            storage_detail["free_bytes"] = free_bytes
            storage_ok = free_bytes >= minimum
        except OSError:
            errors.append("disk-space check failed")
        if not storage_ok and "free_bytes" in storage_detail:
            errors.append("insufficient free disk space for the declared runtime")
    storage_detail["ok"] = storage_ok
    checks["storage"] = storage_detail

    network_urls = manifest.get("network", {}).get("first_acquisition_urls", []) if isinstance(manifest.get("network"), dict) else []
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
    checks["network"] = {"required": True, "results": network_results, "ok": not any(not item["ok"] for item in network_results.values())}

    return {"ok": not errors, "errors": errors, "checks": checks, "fingerprint": paths.fingerprint}


def sanitized_worker_environment(paths: RuntimePaths, base_environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an allowlisted worker environment with namespaced cache paths."""

    source = dict(os.environ if base_environment is None else base_environment)
    allowed = {key: value for key, value in source.items() if key in {"HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}}
    env_bin = paths.environment / "bin"
    allowed["PATH"] = os.pathsep.join((str(env_bin), "/usr/local/bin", "/usr/bin", "/bin"))
    values = {
        "E2A_ROOT": str(paths.e2a_root),
        "E2A_MODELS_DIR": str(paths.models_dir),
        "E2A_RUN_DIR": str(paths.run_dir),
        "E2A_CHATTERBOX_RUN_DIR": str(paths.run_namespace),
        "HF_HOME": str(paths.model_namespace / "hf"),
        "HF_HUB_CACHE": str(paths.model_namespace / "hf/hub"),
        "HF_DATASETS_CACHE": str(paths.model_namespace / "hf/datasets"),
        "HF_XET_CACHE": str(paths.model_namespace / "hf/xet"),
        "HF_ASSETS_CACHE": str(paths.model_namespace / "hf/assets"),
        "TORCH_HOME": str(paths.model_namespace / "torch"),
        "TTS_CACHE": str(paths.model_namespace / "tts"),
        "XDG_CACHE_HOME": str(paths.model_namespace / "xdg"),
        "XDG_CONFIG_HOME": str(paths.state_dir / "config"),
        "TMPDIR": str(paths.run_namespace / "tmp"),
        "GRADIO_TEMP_DIR": str(paths.run_namespace / "gradio"),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
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


def install_runtime(
    paths: RuntimePaths,
    interpreter: Path,
    *,
    worker_script: Path | None = None,
    network_checker: Callable[[str], tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    """Provision a new environment from the verified lock using argv lists."""

    report = preflight(paths, interpreter, network_checker=network_checker)
    if not report["ok"]:
        raise ProvisioningError(json.dumps(report, sort_keys=True))
    if paths.environment.exists():
        raise ProvisioningError("environment path already exists; refusing to modify it")

    paths.env_base.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.environment.mkdir(mode=0o700)
    source = Path(shutil.which(str(interpreter)) or str(interpreter)).expanduser().resolve()
    try:
        venv = subprocess.run(
            [str(source), "-m", "venv", str(paths.environment)],
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

    env_python = paths.environment / "bin/python"
    worker_env = sanitized_worker_environment(paths)
    pip_install = [
        str(env_python),
        "-m",
        "pip",
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--require-virtualenv",
        "--require-hashes",
        "-r",
        str(paths.lock_path),
    ]
    try:
        installed = subprocess.run(
            pip_install,
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

    try:
        pip_check = subprocess.run(
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

    manifest = _read_json(paths.manifest_path)
    modules = manifest.get("validation", {}).get("import_modules", [])
    if not isinstance(modules, list) or not modules:
        raise ProvisioningError("manifest.validation.import_modules is required")
    imports_ok, imports_detail = _module_hook(env_python, [str(module) for module in modules])
    if not imports_ok:
        raise ProvisioningError(imports_detail)

    worker_status = "not_run"
    if worker_script is not None:
        worker = worker_script.expanduser().resolve()
        if not _is_within(worker, paths.repo_root) or not _regular_file(worker):
            raise ProvisioningError("worker self-test path is missing or outside the repository")
        worker_args = manifest.get("validation", {}).get("worker_self_test_args", ["--self-test"])
        if not isinstance(worker_args, list) or any(not isinstance(item, str) for item in worker_args):
            raise ProvisioningError("worker self-test arguments are invalid")
        worker_ok, worker_detail = _command_hook(env_python, [str(worker), *worker_args], worker_env)
        worker_status = worker_detail
        if not worker_ok:
            raise ProvisioningError(worker_detail)

    result = {
        "status": "passed" if worker_status == "self-test hook passed" else "runtime-passed-worker-self-test-not-run",
        "environment": str(paths.environment),
        "fingerprint": paths.fingerprint,
        "manifest_sha256": paths.manifest_sha256,
        "lock_sha256": paths.lock_sha256,
        "checks": {
            "pip_check": {"ok": True},
            "imports": {"ok": True, "modules": [str(module) for module in modules]},
            "worker_self_test": {"ok": worker_status == "self-test hook passed", "status": worker_status},
        },
    }
    write_result(paths, result)
    return result
