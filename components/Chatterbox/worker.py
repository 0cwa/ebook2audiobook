"""Private JSONL worker for the isolated Chatterbox runtime.

The module deliberately has no Chatterbox, Torch, or audio-library imports at
module import time.  This keeps protocol validation and the worker entry point
usable in a dependency-light test environment.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import threading
import time
import uuid
from typing import Any, Callable, Mapping, Sequence
import wave

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from components.Chatterbox.runtime.contract_data import (
    ACTIVATION_RECEIPT_SCHEMA,
    APPROVED_LANGUAGE_IDS,
    CHANNELS,
    DEVICE,
    MAX_SEGMENTS,
    MAX_SILENCE_SECONDS,
    MAX_TEXT_CHARS,
    MAX_TOTAL_TEXT_CHARS,
    MODEL_FAMILY,
    MODEL_RECEIPT_SCHEMA,
    MODEL_VARIANT,
    model_profile_spec,
    normalize_model_profile,
    normalize_model_variant,
    PKUSEG_DATA_FILENAME,
    PKUSEG_DATA_SHA256,
    PROTOCOL_VERSION,
    RUNTIME_RECEIPT_SCHEMA,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)

_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_local_pkuseg_data() -> None:
    """Require the pinned tokenizer data before importing Chatterbox."""

    raw_home = os.environ.get("PKUSEG_HOME")
    if not raw_home:
        raise RuntimeError("PKUSEG_HOME is required for the local Chatterbox worker")
    home = Path(raw_home).expanduser()
    if not home.is_absolute() or home.is_symlink() or not home.is_dir():
        raise RuntimeError("PKUSEG_HOME must be an existing local directory")
    try:
        if home.resolve(strict=True) != home:
            raise RuntimeError("PKUSEG_HOME must not contain a symlink")
    except OSError as exc:
        raise RuntimeError("PKUSEG_HOME is not readable") from exc

    archive = home / PKUSEG_DATA_FILENAME
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError(
            f"pinned tokenizer data is missing: {PKUSEG_DATA_FILENAME}"
        )
    if _sha256_file(archive).lower() != PKUSEG_DATA_SHA256:
        raise RuntimeError("pinned tokenizer data checksum does not match")
    model_dir = home / "spacy_ontonotes"
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise RuntimeError("pinned tokenizer data is not extracted")
    for name in ("features.msgpack", "weights.npz"):
        entry = model_dir / name
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"pinned tokenizer data is incomplete: {name}")


def _enforce_local_only_environment() -> None:
    """Prevent model/network fallback inside an activated synthesis worker."""

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        os.environ.pop(key, None)
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    })


class WorkerRequestError(ValueError):
    """A request that cannot be safely or meaningfully processed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerCancelled(Exception):
    """Raised when the client asks the worker to cancel a request."""


class ReceiptRepairRequiredError(RuntimeError):
    """A receipt publication ambiguity that requires explicit repair."""


def _runtime_language_ids(capabilities: Any) -> tuple[str, ...]:
    """Validate and normalize the installed package's language capability set."""

    if isinstance(capabilities, Mapping):
        raw_languages = tuple(capabilities.keys())
    elif isinstance(capabilities, (list, tuple, set, frozenset)):
        raw_languages = tuple(capabilities)
    else:
        raise RuntimeError(
            "pinned Chatterbox runtime returned an invalid language capability set"
        )
    languages = tuple(str(language).lower() for language in raw_languages)
    if any(not language for language in languages) or len(set(languages)) != len(languages):
        raise RuntimeError(
            "pinned Chatterbox runtime returned duplicate or invalid language capabilities"
        )
    runtime_ids = frozenset(languages)
    if runtime_ids != APPROVED_LANGUAGE_IDS:
        missing = ", ".join(sorted(APPROVED_LANGUAGE_IDS - runtime_ids)) or "none"
        additional = ", ".join(sorted(runtime_ids - APPROVED_LANGUAGE_IDS)) or "none"
        raise RuntimeError(
            "pinned Chatterbox runtime language capabilities do not match the approved "
            f"set (missing: {missing}; additional: {additional})"
        )
    return tuple(language for language in SUPPORTED_LANGUAGES if language in runtime_ids)


def _load_local_chatterbox_model(snapshot_path: Path, variant: str) -> tuple[Any, bool]:
    """Load one verified multilingual model.

    Chatterbox 0.1.7 predates the public V3 selector.  Prefer the upstream
    selector when present; otherwise reproduce the minimal V3 loading path
    against the pinned 0.1.7 runtime and disable its V2-only alignment
    analyzer.  The fallback remains local-only and uses the same verified
    snapshot boundary as V2.
    """

    import inspect

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    from_local = ChatterboxMultilingualTTS.from_local
    if variant == "v2":
        return from_local(str(snapshot_path), device=DEVICE), False
    if variant != "v3":
        raise RuntimeError(f"unsupported Chatterbox model variant: {variant}")

    try:
        parameters = inspect.signature(from_local).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "t3_model" in parameters:
        return from_local(str(snapshot_path), device=DEVICE, t3_model="v3"), False

    import torch
    from safetensors.torch import load_file as load_safetensors

    import chatterbox.models.t3.t3 as t3_module
    import chatterbox.mtl_tts as mtl_module
    from chatterbox.mtl_tts import Conditionals
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.s3tokenizer import S3_TOKEN_RATE
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder

    class _DisabledAlignmentStreamAnalyzer:
        """Compatibility shim for upstream V3's analyzer-free inference path."""

        def __init__(self, *_args: Any, eos_idx: int = 0, **_kwargs: Any):
            self.eos_idx = eos_idx

        def step(self, logits: Any, next_token: Any = None) -> Any:
            return logits

    # The pinned 0.1.7 T3.inference resolves this module global each time it
    # compiles its backend.  A V3 worker hosts one model for its lifetime, so
    # replacing the analyzer here cannot affect a co-resident V2 model.
    t3_module.AlignmentStreamAnalyzer = _DisabledAlignmentStreamAnalyzer

    map_location = torch.device("cpu") if DEVICE in {"cpu", "mps"} else None
    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(
        torch.load(snapshot_path / "ve.pt", map_location=map_location, weights_only=True)
    )
    voice_encoder.to(DEVICE).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(snapshot_path / "t3_mtl23ls_v3.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(DEVICE).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(
        torch.load(snapshot_path / "s3gen.pt", map_location=map_location, weights_only=True)
    )
    s3gen.to(DEVICE).eval()

    tokenizer = MTLTokenizer(str(snapshot_path / "grapheme_mtl_merged_expanded_v1.json"))
    conditionals = None
    builtin_voice = snapshot_path / "conds.pt"
    if builtin_voice.exists():
        conditionals = Conditionals.load(
            builtin_voice, map_location=map_location
        ).to(DEVICE)

    model = ChatterboxMultilingualTTS(
        t3,
        s3gen,
        voice_encoder,
        tokenizer,
        DEVICE,
        conds=conditionals,
    )

    # Upstream V3 drops the final post-filter speech token before watermarking.
    # Capture the filtered token count without copying the whole generate()
    # implementation so the pinned 0.1.7 runtime keeps its own synthesis path.
    token_state: dict[str, int | None] = {"count": None}
    original_drop_invalid_tokens = mtl_module.drop_invalid_tokens

    def _capture_filtered_tokens(tokens: Any) -> Any:
        filtered = original_drop_invalid_tokens(tokens)
        token_state["count"] = int(filtered.shape[-1])
        return filtered

    mtl_module.drop_invalid_tokens = _capture_filtered_tokens

    class _V3TrimmedWatermarker:
        def __init__(self, inner: Any):
            self.inner = inner

        def apply_watermark(self, wav: Any, sample_rate: int) -> Any:
            count = token_state.get("count")
            if count:
                speech_samples = max(1, count - 1) * (sample_rate // S3_TOKEN_RATE)
                wav = wav[:speech_samples]
            return self.inner.apply_watermark(wav, sample_rate=sample_rate)

    model.watermarker = _V3TrimmedWatermarker(model.watermarker)
    return model, True


def _load_local_turbo_model(snapshot_path: Path, profile: str) -> tuple[Any, bool]:
    """Load Turbo/Nano from the verified snapshot with a narrow local fallback."""

    if profile not in {"turbo", "nano"}:
        raise RuntimeError(f"unsupported Turbo-family Chatterbox profile: {profile}")

    import inspect

    native_class = None
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        native_class = ChatterboxTurboTTS
    except Exception:
        native_class = None

    nano = profile == "nano"
    if native_class is not None:
        from_local = native_class.from_local
        try:
            parameters = inspect.signature(from_local).parameters
        except (TypeError, ValueError):
            parameters = {}

        if not nano:
            kwargs: dict[str, Any] = {"device": DEVICE}
            if "nano" in parameters:
                kwargs["nano"] = False
            return from_local(str(snapshot_path), **kwargs), False

        if "nano" in parameters:
            try:
                from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
            except Exception:
                LLAMA_CONFIGS = {}
            if "GPT2_small" in LLAMA_CONFIGS:
                return from_local(str(snapshot_path), device=DEVICE, nano=True), False

    from components.Chatterbox.turbo_compat import load_turbo_compat

    return load_turbo_compat(snapshot_path, device=DEVICE, nano=nano), True


def _message(message: Mapping[str, Any]) -> None:
    """Write exactly one protocol object to stdout."""

    sys.stdout.write(json.dumps(message, separators=(",", ":"), ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _error_response(request_id: Any, code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "id": request_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


def _safe_string(value: Any, field: str, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise WorkerRequestError("invalid_request", f"{field} must be a non-empty string")
    if "\x00" in value:
        raise WorkerRequestError("invalid_request", f"{field} contains an invalid character")
    return value


def _as_roots(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise WorkerRequestError("invalid_request", "approved roots must be a list")
    roots = []
    for item in value:
        if isinstance(item, os.PathLike):
            item = os.fspath(item)
        root = Path(_safe_string(item, "approved root")).expanduser()
        if not root.is_absolute():
            raise WorkerRequestError("invalid_request", "approved roots must be absolute")
        roots.append(root.resolve(strict=False))
    return tuple(roots)


def normalise_approved_roots(value: Any) -> dict[str, tuple[Path, ...]]:
    """Return category-specific approved roots from a request or config."""

    if isinstance(value, Mapping):
        shared = _as_roots(value.get("all"))
        voice = _as_roots(value.get("voice")) or shared
        output = _as_roots(value.get("output")) or shared
        if not voice and not output:
            raise WorkerRequestError("invalid_request", "approved roots are required")
        return {"voice": voice, "output": output}
    roots = _as_roots(value)
    if not roots:
        raise WorkerRequestError("invalid_request", "approved roots are required")
    return {"voice": roots, "output": roots}


def _within(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _approved_path(value: Any, field: str, roots: Sequence[Path]) -> Path:
    raw = _safe_string(value, field, max_length=16384)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise WorkerRequestError("invalid_request", f"{field} must be absolute")
    resolved = path.resolve(strict=False)
    if not _within(resolved, roots):
        raise WorkerRequestError("invalid_request", f"{field} is outside approved roots")
    return resolved


def _absolute_path(value: Any, field: str) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    raw = _safe_string(value, field, max_length=16384)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{field} must be absolute")
    return path.resolve(strict=False)


def _approved_regular_file(value: Any, field: str, roots: Sequence[Path]) -> Path:
    raw = Path(os.fspath(value)).expanduser()
    if not raw.is_absolute():
        raise RuntimeError(f"{field} must be absolute")
    if raw.is_symlink():
        raise RuntimeError(f"{field} must be an approved regular file")
    path = raw.resolve(strict=False)
    if not _within(path, roots):
        raise RuntimeError(f"{field} is outside approved roots")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{field} must be an approved regular file")
    return path


def _immutable_revision(value: Any, field: str = "model revision") -> str:
    revision = _safe_string(value, field, max_length=40)
    if not _HEX40.fullmatch(revision):
        raise RuntimeError(f"{field} must be an immutable 40-character commit")
    return revision.lower()


def _repository_id(value: Any) -> str:
    locator = _safe_string(value, "model repository", max_length=2048).rstrip("/")
    prefix = "https://huggingface.co/"
    if locator.startswith(prefix):
        locator = locator[len(prefix):]
    if not re.fullmatch(r"[^/\\?#]+/[^/\\?#]+", locator):
        raise RuntimeError("model repository must be a Hugging Face repository ID")
    return locator


def _manifest_file_records(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("runtime manifest model.files must contain declared artifacts")
    records = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeError("runtime manifest model.files entries must be objects")
        raw_path = _safe_string(item.get("path"), "model file path", max_length=1024)
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in raw_path
        ):
            raise RuntimeError("runtime manifest model file paths must be safe relative paths")
        path = relative.as_posix()
        if path in seen:
            raise RuntimeError("runtime manifest model files must be unique")
        seen.add(path)
        digest = _safe_string(item.get("sha256"), f"sha256 for {path}", max_length=64).lower()
        if not _HEX64.fullmatch(digest):
            raise RuntimeError(f"sha256 for {path} must be a 64-character hex digest")
        size = item.get("size_bytes")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise RuntimeError(f"size_bytes for {path} must be a non-negative integer")
        source = item.get("source")
        if source is not None and (not isinstance(source, str) or not source):
            raise RuntimeError(f"source for {path} must be a non-empty repository identifier")
        records.append({"path": path, "source": source, "sha256": digest, "size_bytes": size})
    return tuple(records)


def _read_model_manifest(manifest_path: Path, manifest_roots: Sequence[Path]) -> dict[str, Any]:
    approved_manifest = _approved_regular_file(manifest_path, "model manifest", manifest_roots)
    try:
        value = json.loads(approved_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("model manifest could not be read") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("model manifest must be a JSON object")
    sources = value.get("sources")
    model = sources.get("model") if isinstance(sources, Mapping) else None
    if not isinstance(model, Mapping):
        raise RuntimeError("model manifest sources.model is required")
    profile = normalize_model_profile(model.get("profile", model.get("variant")))
    if profile is None:
        raise RuntimeError("model manifest profile is unsupported")
    spec = model_profile_spec(profile)
    loader_kind = model.get("loader_kind", spec.loader_kind)
    family = model.get("family", spec.family)
    if loader_kind != spec.loader_kind or family != spec.family:
        raise RuntimeError("model manifest profile semantics do not match the selected profile")
    revision_value = model.get("revision")
    locator_value = model.get("locator")
    repositories = model.get("repositories")
    if (revision_value is None or locator_value is None) and isinstance(repositories, Mapping) and len(repositories) == 1:
        repository = next(iter(repositories.values()))
        if isinstance(repository, Mapping):
            revision_value = repository.get("revision")
            locator_value = repository.get("locator")
    revision = _immutable_revision(revision_value)
    records = _manifest_file_records(model.get("files"))
    return {
        "repository": _repository_id(locator_value),
        "revision": revision,
        "profile": profile,
        "variant": profile,
        "loader_kind": loader_kind,
        "family": family,
        "files": records,
        "allow_patterns": [record["path"] for record in records],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_snapshot(
    snapshot: Path,
    records: Sequence[Mapping[str, Any]],
    approved_root: Path | None = None,
) -> Path:
    raw_snapshot = snapshot.expanduser()
    if raw_snapshot.is_symlink():
        raise RuntimeError("verified model snapshot must not be a symlink")
    snapshot = raw_snapshot.resolve(strict=False)
    cache_root = (approved_root or snapshot).resolve(strict=False)
    if not snapshot.is_dir() or not _within(snapshot, (cache_root,)):
        raise RuntimeError("verified model snapshot is not an approved directory")
    expected = {str(record["path"]) for record in records}
    discovered: set[str] = set()
    stack = [snapshot]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            entries = list(iterator)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(snapshot).as_posix()
            if entry.is_symlink():
                raise RuntimeError(f"verified model snapshot contains a symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                discovered.add(relative)
            else:
                raise RuntimeError(f"verified model snapshot contains a special file: {relative}")
    extras = sorted(discovered - expected)
    if extras:
        raise RuntimeError(f"verified model snapshot contains undeclared files: {', '.join(extras)}")
    for record in records:
        relative = PurePosixPath(str(record["path"]))
        raw_candidate = snapshot / Path(*relative.parts)
        candidate = raw_candidate.resolve(strict=False)
        if (
            not _within(candidate, (snapshot,))
            or raw_candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise RuntimeError(f"verified model snapshot is missing {record['path']}")
        expected_size = record.get("size_bytes")
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise RuntimeError(f"verified model snapshot size does not match {record['path']}")
        if _sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"verified model snapshot checksum does not match {record['path']}")
    return snapshot


def _read_private_receipt(path_value: str | None, label: str, schema: str) -> tuple[Path, dict[str, Any]]:
    if not path_value:
        raise RuntimeError(f"{label} receipt path is required")
    raw = Path(path_value).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise RuntimeError(f"{label} receipt must be an absolute regular file")
    path = raw.resolve(strict=False)
    publication = path.with_name(f"{path.name}.publishing")
    published_proof = path.with_name(f"{path.name}.published")
    ambiguity = path.with_name(f"{path.name}.ambiguous")
    if os.path.lexists(os.fspath(publication)) or os.path.lexists(os.fspath(ambiguity)):
        raise ReceiptRepairRequiredError(
            f"{label} receipt publication is incomplete or its durability is ambiguous; "
            "repair_required"
        )
    if not raw.is_file():
        raise RuntimeError(f"{label} receipt must be an absolute regular file")
    if os.path.lexists(os.fspath(published_proof)) and (
        published_proof.is_symlink() or not published_proof.is_file()
    ):
        raise ReceiptRepairRequiredError(
            f"{label} receipt publication proof is ambiguous; repair_required"
        )
    if not published_proof.is_file():
        raise RuntimeError(f"{label} receipt has no durable publication proof")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        proof = json.loads(published_proof.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} receipt or publication proof could not be read") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != schema or payload.get("status") != "ready":
        raise RuntimeError(f"{label} receipt is not ready")
    if (
        not isinstance(proof, Mapping)
        or proof.get("schema") != f"ebook2audiobook.chatterbox-{label}-receipt-publication.v1"
        or proof.get("status") != "published"
        or proof.get("receipt") != str(path)
        or proof.get("receipt_sha256") != _sha256_file(path)
    ):
        raise RuntimeError(f"{label} receipt publication proof does not match")
    return path, dict(payload)


def _validate_activation_chain(model_root: Path) -> None:
    """Require the runtime, model, and activation receipts for normal service."""

    runtime_path, runtime = _read_private_receipt(
        os.environ.get("E2A_CHATTERBOX_RUNTIME_RECEIPT"), "runtime", RUNTIME_RECEIPT_SCHEMA
    )
    model_path, model = _read_private_receipt(
        os.environ.get("E2A_CHATTERBOX_MODEL_RECEIPT"), "model", MODEL_RECEIPT_SCHEMA
    )
    _, activation = _read_private_receipt(
        os.environ.get("E2A_CHATTERBOX_ACTIVATION_RECEIPT"), "activation", ACTIVATION_RECEIPT_SCHEMA
    )
    snapshot_value = model.get("artifact", {}).get("snapshot_path") if isinstance(model.get("artifact"), Mapping) else None
    if not isinstance(snapshot_value, str) or Path(snapshot_value).resolve(strict=False) != model_root.resolve(strict=False):
        raise RuntimeError("model receipt does not bind the approved snapshot")
    if activation.get("model_snapshot") != snapshot_value:
        raise RuntimeError("activation receipt model snapshot does not match")
    if activation.get("runtime_fingerprint") != runtime.get("runtime_fingerprint"):
        raise RuntimeError("activation receipt runtime identity does not match")
    if activation.get("model_fingerprint") != model.get("model_fingerprint"):
        raise RuntimeError("activation receipt model identity does not match")
    if activation.get("runtime_receipt_sha256") != _sha256_file(runtime_path):
        raise RuntimeError("activation receipt runtime hash does not match")
    if activation.get("model_receipt_sha256") != _sha256_file(model_path):
        raise RuntimeError("activation receipt model hash does not match")
    local_load = activation.get("checks", {}).get("local_model_load") if isinstance(activation.get("checks"), Mapping) else None
    if not isinstance(local_load, Mapping) or local_load.get("ok") is not True:
        raise RuntimeError("activation receipt local model load check did not pass")


def _prompt(
    prompt: Any,
    roots: Sequence[Path],
    *,
    minimum_prompt_seconds: float | None = None,
) -> dict[str, Any] | None:
    if prompt is None:
        return None
    if not isinstance(prompt, Mapping):
        raise WorkerRequestError("invalid_request", "voice prompt must be an object")
    path = _approved_path(prompt.get("path"), "voice prompt path", roots)
    if not path.is_file():
        raise WorkerRequestError("voice_missing", "voice prompt file does not exist")
    digest = prompt.get("sha256")
    if digest is not None:
        digest = _safe_string(digest, "voice prompt sha256", max_length=128).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise WorkerRequestError("invalid_request", "voice prompt sha256 is invalid")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise WorkerRequestError("voice_missing", "voice prompt checksum does not match")
    if minimum_prompt_seconds is not None:
        try:
            with wave.open(str(path), "rb") as stream:
                if stream.getnchannels() != CHANNELS or stream.getframerate() != SAMPLE_RATE:
                    raise WorkerRequestError(
                        "invalid_request",
                        "voice prompt must be mono 24 kHz WAV audio",
                    )
                duration = stream.getnframes() / float(stream.getframerate())
        except WorkerRequestError:
            raise
        except (EOFError, OSError, wave.Error) as exc:
            raise WorkerRequestError(
                "invalid_request",
                "voice prompt could not be decoded as WAV audio",
            ) from exc
        if duration <= minimum_prompt_seconds:
            raise WorkerRequestError(
                "invalid_request",
                f"voice prompt must be longer than {minimum_prompt_seconds:g} seconds",
            )
    return {"path": str(path), "sha256": digest}


def validate_request(
    request: Mapping[str, Any],
    *,
    configured_roots: Any = None,
    expected_model: Mapping[str, str] | None = None,
    supported_languages: Sequence[str] = SUPPORTED_LANGUAGES,
    minimum_prompt_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate and normalize a synthesis request without importing runtime dependencies."""

    if not isinstance(request, Mapping):
        raise WorkerRequestError("invalid_request", "request must be an object")
    if request.get("protocol") != PROTOCOL_VERSION:
        raise WorkerRequestError("invalid_request", "unsupported protocol version")
    request_id = _safe_string(request.get("id"), "request id", max_length=256)
    if request.get("op") != "synthesize":
        raise WorkerRequestError("invalid_request", "expected a synthesize request")
    if request.get("device") != DEVICE:
        raise WorkerRequestError("unsupported_device", "Chatterbox worker supports CPU only")

    model = request.get("model")
    if not isinstance(model, Mapping):
        raise WorkerRequestError("invalid_request", "model must be an object")
    profile = normalize_model_profile(model.get("profile", model.get("t3_model")))
    if profile is None:
        raise WorkerRequestError("invalid_request", "unsupported Chatterbox model profile")
    profile_spec = model_profile_spec(profile)
    if model.get("family") != profile_spec.family:
        raise WorkerRequestError("invalid_request", "unsupported Chatterbox model family")
    loader_kind = model.get("loader_kind", profile_spec.loader_kind)
    if loader_kind != profile_spec.loader_kind:
        raise WorkerRequestError("invalid_request", "model loader kind does not match the profile")
    if profile_spec.loader_kind != "multilingual" and model.get("t3_model") is not None:
        raise WorkerRequestError(
            "invalid_request",
            "Turbo/Nano profiles must not be encoded as multilingual t3_model variants",
        )
    if profile_spec.loader_kind == "multilingual":
        legacy_variant = normalize_model_variant(model.get("t3_model", profile))
        if legacy_variant != profile:
            raise WorkerRequestError("invalid_request", "multilingual model variant does not match the profile")
    revision = _safe_string(model.get("revision"), "model revision", max_length=512)
    if expected_model is not None:
        expected_revision = expected_model.get("revision")
        if expected_revision and revision != expected_revision:
            raise WorkerRequestError("invalid_request", "model revision does not match the worker")
        expected_profile = normalize_model_profile(
            expected_model.get("profile", expected_model.get("variant"))
        )
        if expected_profile and profile != expected_profile:
            raise WorkerRequestError("invalid_request", "model profile does not match the worker")
        expected_family = expected_model.get("family")
        if expected_family and profile_spec.family != expected_family:
            raise WorkerRequestError("invalid_request", "model family does not match the worker")

    language = _safe_string(request.get("language"), "language", max_length=16).lower()
    if language not in supported_languages:
        raise WorkerRequestError("unsupported_language", f"language is not supported: {language}")

    request_roots = normalise_approved_roots(request.get("approved_roots"))
    if configured_roots is not None:
        configured = normalise_approved_roots(configured_roots)
        roots = {
            "voice": configured["voice"] or request_roots["voice"],
            "output": configured["output"] or request_roots["output"],
        }
    else:
        roots = request_roots

    output = request.get("output")
    if not isinstance(output, Mapping):
        raise WorkerRequestError("invalid_request", "output must be an object")
    requested_output = _approved_path(output.get("path"), "output path", roots["output"])
    final_output = requested_output
    if final_output.name.endswith(".part"):
        final_output = final_output.with_name(final_output.name[:-5])
    if final_output.suffix.lower() != ".flac":
        raise WorkerRequestError("invalid_request", "output path must be a FLAC file")
    if not _within(final_output, roots["output"]):
        raise WorkerRequestError("invalid_request", "output path is outside approved roots")
    if output.get("sample_rate") != SAMPLE_RATE or output.get("channels") != CHANNELS:
        raise WorkerRequestError("invalid_request", "output must be mono 24 kHz FLAC")

    segments = request.get("segments")
    if not isinstance(segments, list) or not segments or len(segments) > MAX_SEGMENTS:
        raise WorkerRequestError("invalid_request", "segments must be a non-empty bounded list")
    total_text = 0
    normalized_segments = []
    top_level_prompt = request.get("reference_prompt")
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise WorkerRequestError("invalid_request", "each segment must be an object")
        kind = segment.get("kind")
        if kind == "text":
            text = _safe_string(segment.get("text"), "segment text", max_length=MAX_TEXT_CHARS)
            total_text += len(text)
            if total_text > MAX_TOTAL_TEXT_CHARS:
                raise WorkerRequestError("invalid_request", "request text is too long")
            prompt = segment.get("voice_prompt", top_level_prompt)
            normalized_segments.append({
                "kind": "text",
                "text": text,
                "voice_prompt": _prompt(
                    prompt,
                    roots["voice"],
                    minimum_prompt_seconds=minimum_prompt_seconds,
                ),
            })
        elif kind == "silence":
            seconds = segment.get("seconds")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                raise WorkerRequestError("invalid_request", "silence seconds must be numeric")
            if not math.isfinite(float(seconds)) or seconds <= 0 or seconds > MAX_SILENCE_SECONDS:
                raise WorkerRequestError("invalid_request", "silence duration is out of bounds")
            normalized_segments.append({"kind": "silence", "seconds": float(seconds)})
        else:
            raise WorkerRequestError("invalid_request", "segment kind must be text or silence")

    return {
        "id": request_id,
        "language": language,
        "revision": revision,
        "model_profile": profile,
        "loader_kind": profile_spec.loader_kind,
        "segments": normalized_segments,
        "output": final_output,
        "roots": roots,
    }


def validate_audio_file(path: Path, *, torchaudio_module: Any = None) -> dict[str, Any]:
    """Validate a saved file using lazily supplied torchaudio."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise WorkerRequestError("output_invalid", "generated audio is empty")
    if torchaudio_module is None:
        try:
            import torchaudio as torchaudio_module
        except Exception as exc:
            raise WorkerRequestError("output_invalid", "torchaudio is unavailable for output validation") from exc
    try:
        waveform, sample_rate = torchaudio_module.load(str(path))
    except Exception as exc:
        raise WorkerRequestError("output_invalid", "generated FLAC cannot be decoded") from exc
    shape = getattr(waveform, "shape", ())
    if len(shape) != 2 or int(shape[0]) != CHANNELS or int(shape[1]) <= 0:
        raise WorkerRequestError("output_invalid", "generated audio must be nonempty mono audio")
    if int(sample_rate) != SAMPLE_RATE:
        raise WorkerRequestError("output_invalid", "generated audio has the wrong sample rate")
    try:
        finite = bool(waveform.isfinite().all().item())
    except Exception as exc:
        raise WorkerRequestError("output_invalid", "generated audio could not be checked") from exc
    if not finite:
        raise WorkerRequestError("output_invalid", "generated audio contains non-finite samples")
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "samples": int(shape[1]),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class ChatterboxWorker:
    """Single-model, serialized worker implementation."""

    def __init__(
        self,
        *,
        approved_roots: Any = None,
        model_revision: str | None = None,
        model_manifest_path: str | os.PathLike[str] | None = None,
        approved_model_root: str | os.PathLike[str] | None = None,
        approved_manifest_root: str | os.PathLike[str] | None = None,
        model_loader: Callable[[], Any] | None = None,
        require_activation_receipt: bool = False,
    ):
        self.approved_roots = approved_roots
        self.model_revision = model_revision
        self.model_manifest_path = model_manifest_path
        self.approved_model_root = approved_model_root
        self.approved_manifest_root = approved_manifest_root
        self.model_loader = model_loader
        self.require_activation_receipt = require_activation_receipt
        self.model = None
        self.model_profile = MODEL_VARIANT
        self.model_variant = MODEL_VARIANT
        self.loader_kind = "multilingual"
        self.model_family = MODEL_FAMILY
        self.minimum_prompt_seconds: float | None = None
        self._v3_compat_mode = False
        self._turbo_compat_mode = False
        self.supported_languages = SUPPORTED_LANGUAGES
        self.sample_rate = SAMPLE_RATE
        self._audio_backend = None
        self._send_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_id: str | None = None
        self._active_cancel: threading.Event | None = None
        self._active_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def load_model(self) -> None:
        if self.model is not None:
            return
        if self.model_loader is not None:
            self.model = self.model_loader()
        else:
            _enforce_local_only_environment()
            try:
                if self.model_manifest_path is None or self.approved_model_root is None:
                    raise RuntimeError(
                        "a runtime model manifest and verified local model snapshot are required"
                    )
                if self.approved_manifest_root is None:
                    raise RuntimeError("an approved model manifest root is required")
                raw_model_root = Path(os.fspath(self.approved_model_root)).expanduser()
                if not raw_model_root.is_absolute():
                    raise RuntimeError("approved model root must be absolute")
                if raw_model_root.is_symlink():
                    raise RuntimeError("approved model root must not be a symlink")
                model_root = raw_model_root.resolve(strict=False)
                if self.require_activation_receipt:
                    _validate_activation_chain(model_root)
                raw_manifest_root = Path(os.fspath(self.approved_manifest_root)).expanduser()
                if not raw_manifest_root.is_absolute():
                    raise RuntimeError("approved manifest root must be absolute")
                if raw_manifest_root.is_symlink():
                    raise RuntimeError("approved manifest root must not be a symlink")
                manifest_root = raw_manifest_root.resolve(strict=False)
                manifest = _read_model_manifest(
                    _absolute_path(self.model_manifest_path, "model manifest"),
                    (manifest_root,),
                )
                self.model_revision = manifest["revision"]
                self.model_profile = manifest["profile"]
                self.model_variant = manifest["profile"]
                self.loader_kind = manifest["loader_kind"]
                self.model_family = manifest["family"]
                profile_spec = model_profile_spec(self.model_profile)
                self.minimum_prompt_seconds = profile_spec.minimum_prompt_seconds
                try:
                    snapshot_path = _verify_model_snapshot(model_root, manifest["files"], model_root)
                except Exception as exc:
                    raise RuntimeError(f"verified local Chatterbox model is unavailable: {exc}") from exc

                if self.loader_kind == "multilingual":
                    try:
                        with redirect_stdout(sys.stderr):
                            _require_local_pkuseg_data()
                            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    except Exception as exc:
                        raise RuntimeError(
                            "Chatterbox is unavailable; install the pinned worker runtime"
                        ) from exc
                    get_supported_languages = getattr(ChatterboxMultilingualTTS, "get_supported_languages", None)
                    if not callable(get_supported_languages):
                        raise RuntimeError(
                            "pinned Chatterbox runtime does not expose get_supported_languages()"
                        )
                    runtime_languages = _runtime_language_ids(get_supported_languages())
                    try:
                        with redirect_stdout(sys.stderr):
                            self.model, self._v3_compat_mode = _load_local_chatterbox_model(
                                snapshot_path,
                                self.model_profile,
                            )
                    except Exception as exc:
                        raise RuntimeError(f"Chatterbox model load failed: {exc}") from exc
                elif self.loader_kind == "turbo":
                    runtime_languages = profile_spec.supported_languages
                    try:
                        with redirect_stdout(sys.stderr):
                            self.model, self._turbo_compat_mode = _load_local_turbo_model(
                                snapshot_path,
                                self.model_profile,
                            )
                    except Exception as exc:
                        raise RuntimeError(f"Chatterbox {self.model_profile} model load failed: {exc}") from exc
                else:
                    raise RuntimeError(
                        f"unsupported Chatterbox loader kind in this worker: {self.loader_kind}"
                    )
            except ReceiptRepairRequiredError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Chatterbox model load failed: {exc}") from exc
            self.supported_languages = runtime_languages
        self.sample_rate = int(getattr(self.model, "sr", SAMPLE_RATE) or SAMPLE_RATE)
        if self.sample_rate != SAMPLE_RATE:
            raise RuntimeError("Chatterbox returned an unsupported sample rate")

    def _send(self, message: Mapping[str, Any]) -> None:
        with self._send_lock:
            _message(message)

    def _load_audio_backend(self) -> tuple[Any, Any]:
        if self._audio_backend is None:
            try:
                import torch
                import torchaudio
            except Exception as exc:
                raise RuntimeError("Torch and torchaudio are required by the worker") from exc
            self._audio_backend = (torch, torchaudio)
        return self._audio_backend

    def _generation_kwargs(
        self,
        request: Mapping[str, Any],
        segment: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.loader_kind == "turbo":
            kwargs: dict[str, Any] = {}
        else:
            kwargs = {"language_id": request["language"]}
            if self._v3_compat_mode:
                # Match the upstream V3 default introduced with the opt-in
                # checkpoint rather than 0.1.7's V2-era default of 2.0.
                kwargs["repetition_penalty"] = 1.2
        prompt = segment.get("voice_prompt")
        if prompt is not None:
            kwargs["audio_prompt_path"] = prompt["path"]
        return kwargs

    def _generate_file(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        torch, torchaudio = self._load_audio_backend()
        chunks = []
        for segment in request["segments"]:
            if cancel.is_set():
                raise WorkerCancelled()
            if segment["kind"] == "silence":
                samples = int(round(segment["seconds"] * SAMPLE_RATE))
                chunks.append(torch.zeros((CHANNELS, samples), dtype=torch.float32))
                continue
            kwargs = self._generation_kwargs(request, segment)
            try:
                with redirect_stdout(sys.stderr), torch.inference_mode():
                    waveform = self.model.generate(segment["text"], **kwargs)
            except Exception as exc:
                raise RuntimeError(f"Chatterbox generation failed: {exc}") from exc
            if cancel.is_set():
                raise WorkerCancelled()
            waveform = waveform.detach().to("cpu").float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 2 or int(waveform.shape[0]) != CHANNELS or int(waveform.shape[1]) <= 0:
                raise RuntimeError("Chatterbox returned invalid mono audio")
            chunks.append(waveform)

        if not chunks:
            raise RuntimeError("Chatterbox returned no audio")
        waveform = torch.cat(chunks, dim=1)
        final_path = request["output"]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.part.flac"
        )
        try:
            with redirect_stdout(sys.stderr):
                torchaudio.save(str(part_path), waveform, SAMPLE_RATE, format="FLAC")
                metadata = validate_audio_file(part_path, torchaudio_module=torchaudio)
            os.replace(part_path, final_path)
            try:
                with redirect_stdout(sys.stderr):
                    validate_audio_file(final_path, torchaudio_module=torchaudio)
            except WorkerRequestError:
                final_path.unlink(missing_ok=True)
                raise
        except WorkerRequestError:
            part_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(f"audio output failed: {exc}") from exc
        return {"path": str(final_path), **metadata}

    def _handle_synthesis(self, request: Mapping[str, Any], request_id: str, cancel: threading.Event) -> None:
        try:
            normalized = validate_request(
                request,
                configured_roots=self.approved_roots,
                expected_model={
                    "revision": self.model_revision,
                    "profile": self.model_profile,
                    "family": self.model_family,
                } if self.model_revision else None,
                supported_languages=self.supported_languages,
                minimum_prompt_seconds=self.minimum_prompt_seconds,
            )
            result = self._generate_file(normalized, cancel)
            if cancel.is_set():
                Path(result["path"]).unlink(missing_ok=True)
                raise WorkerCancelled()
            self._send({"protocol": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result})
        except WorkerCancelled:
            self._send(_error_response(request_id, "cancelled", "synthesis cancelled", retryable=True))
        except WorkerRequestError as exc:
            self._send(_error_response(request_id, exc.code, exc.message, retryable=exc.code in {"timeout", "cancelled"}))
        except Exception as exc:
            print(f"worker synthesis error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._send(_error_response(request_id, "generation_failed", "Chatterbox synthesis failed", retryable=False))
        finally:
            with self._active_lock:
                self._active_id = None
                self._active_cancel = None
                self._active_thread = None

    def _start_synthesis(self, request: Mapping[str, Any]) -> None:
        request_id = request.get("id")
        if not isinstance(request_id, str):
            self._send(_error_response(request_id, "invalid_request", "request id is required"))
            return
        with self._active_lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                self._send(_error_response(request_id, "not_ready", "worker is busy", retryable=True))
                return
            cancel = threading.Event()
            self._active_id = request_id
            self._active_cancel = cancel
            thread = threading.Thread(
                target=self._handle_synthesis,
                args=(request, request_id, cancel),
                name="chatterbox-synthesis",
                daemon=True,
            )
            self._active_thread = thread
            thread.start()

    def _handle_cancel(self, request: Mapping[str, Any]) -> None:
        target_id = request.get("target_id")
        with self._active_lock:
            if target_id == self._active_id and self._active_cancel is not None:
                self._active_cancel.set()
                accepted = True
            else:
                accepted = False
        self._send({
            "protocol": PROTOCOL_VERSION,
            "id": request.get("id"),
            "ok": True,
            "result": {"target_id": target_id, "cancel_requested": accepted},
        })

    def run(self) -> int:
        try:
            self.load_model()
        except ReceiptRepairRequiredError as exc:
            print(f"worker receipt repair error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._send({
                "protocol": PROTOCOL_VERSION,
                "event": "error",
                "error": {
                    "code": "repair_required",
                    "message": "Chatterbox receipt publication requires repair",
                },
            })
            return 1
        except Exception as exc:
            print(f"worker model load error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._send({
                "protocol": PROTOCOL_VERSION,
                "event": "error",
                "error": {"code": "model_load_failed", "message": "Chatterbox runtime/model load failed"},
            })
            return 1

        self._send({
            "protocol": PROTOCOL_VERSION,
            "event": "ready",
            "device": DEVICE,
            "languages": list(self.supported_languages),
            "sample_rate": SAMPLE_RATE,
        })
        for raw_line in sys.stdin:
            try:
                request = json.loads(raw_line)
                if not isinstance(request, Mapping):
                    raise WorkerRequestError("invalid_request", "message must be an object")
                if request.get("protocol") != PROTOCOL_VERSION:
                    raise WorkerRequestError("invalid_request", "unsupported protocol version")
                operation = request.get("op")
                if operation == "synthesize":
                    self._start_synthesis(request)
                elif operation == "cancel":
                    self._handle_cancel(request)
                elif operation == "ping":
                    self._send({
                        "protocol": PROTOCOL_VERSION,
                        "id": request.get("id"),
                        "ok": True,
                        "result": {
                            "device": DEVICE,
                            "languages": list(self.supported_languages),
                            "sample_rate": SAMPLE_RATE,
                        },
                    })
                elif operation == "shutdown":
                    self._send({"protocol": PROTOCOL_VERSION, "id": request.get("id"), "ok": True, "result": {}})
                    self._stop.set()
                    break
                else:
                    self._send(_error_response(request.get("id"), "invalid_request", "unknown operation"))
            except json.JSONDecodeError:
                self._send(_error_response(None, "invalid_request", "message is not valid JSON"))
            except WorkerRequestError as exc:
                self._send(_error_response(None, exc.code, exc.message))
        active = self._active_thread
        if active is not None and active.is_alive():
            self._active_cancel.set() if self._active_cancel is not None else None
            active.join(timeout=0.25)
        return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ebook2audiobook Chatterbox worker")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the worker entrypoint without loading the model",
    )
    parser.add_argument(
        "--model-self-test",
        action="store_true",
        help="load the verified local model without starting the JSONL service",
    )
    parser.add_argument("--device", default=DEVICE)
    # Kept for compatibility with the current host adapter.  When a runtime
    # manifest is configured, its immutable revision is the sole authority.
    parser.add_argument("--model-revision")
    parser.add_argument("--model-manifest")
    parser.add_argument("--approved-model-root")
    parser.add_argument("--approved-manifest-root")
    parser.add_argument("--approved-root", action="append", default=[])
    parser.add_argument("--approved-voice-root", action="append", default=[])
    parser.add_argument("--approved-output-root", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _message({"protocol": PROTOCOL_VERSION, "event": "self_test", "ok": True})
        return 0
    if args.device != DEVICE:
        _message({
            "protocol": PROTOCOL_VERSION,
            "event": "error",
            "error": {"code": "unsupported_device", "message": "Chatterbox worker supports CPU only"},
        })
        return 2
    if args.model_self_test:
        try:
            instance = ChatterboxWorker(
                model_revision=args.model_revision,
                model_manifest_path=args.model_manifest,
                approved_model_root=args.approved_model_root,
                approved_manifest_root=args.approved_manifest_root,
            )
            instance.load_model()
        except Exception as exc:
            print(f"worker local model self-test error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _message({
                "protocol": PROTOCOL_VERSION,
                "event": "error",
                "error": {"code": "model_load_failed", "message": "verified local model load failed"},
            })
            return 1
        _message({
            "protocol": PROTOCOL_VERSION,
            "event": "model_self_test",
            "ok": True,
            "languages": list(instance.supported_languages),
            "sample_rate": instance.sample_rate,
        })
        return 0
    try:
        shared_roots = _as_roots(args.approved_root)
        voice_roots = _as_roots(args.approved_voice_root) or shared_roots
        output_roots = _as_roots(args.approved_output_root) or shared_roots
        roots = normalise_approved_roots({"voice": voice_roots, "output": output_roots})
    except WorkerRequestError as exc:
        _message({
            "protocol": PROTOCOL_VERSION,
            "event": "error",
            "error": {"code": exc.code, "message": exc.message},
        })
        return 2
    return ChatterboxWorker(
        approved_roots=roots,
        model_revision=args.model_revision,
        model_manifest_path=args.model_manifest,
        approved_model_root=args.approved_model_root,
        approved_manifest_root=args.approved_manifest_root,
        require_activation_receipt=True,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
