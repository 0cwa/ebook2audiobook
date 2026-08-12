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


PROTOCOL_VERSION = 1
SAMPLE_RATE = 24000
CHANNELS = 1
DEVICE = "cpu"
MODEL_FAMILY = "chatterbox-multilingual"
MODEL_VARIANT = "v2"

SUPPORTED_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
)

MAX_SEGMENTS = 32
MAX_TEXT_CHARS = 12000
MAX_TOTAL_TEXT_CHARS = 40000
MAX_SILENCE_SECONDS = 30.0
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class WorkerRequestError(ValueError):
    """A request that cannot be safely or meaningfully processed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerCancelled(Exception):
    """Raised when the client asks the worker to cancel a request."""


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
        raise RuntimeError("runtime manifest model.files must be a non-empty list")
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
        records.append({"path": path, "sha256": digest, "size_bytes": size})
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
    revision = _immutable_revision(model.get("revision"))
    records = _manifest_file_records(model.get("files"))
    return {
        "repository": _repository_id(model.get("locator")),
        "revision": revision,
        "variant": model.get("variant"),
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
    snapshot = snapshot.resolve(strict=False)
    cache_root = (approved_root or snapshot).resolve(strict=False)
    if not snapshot.is_dir():
        raise RuntimeError("downloaded model snapshot is not a directory")
    for record in records:
        relative = PurePosixPath(str(record["path"]))
        raw_candidate = snapshot / Path(*relative.parts)
        candidate = raw_candidate.resolve(strict=False)
        if (
            not _within(candidate, (cache_root,))
            or not candidate.is_file()
        ):
            raise RuntimeError(f"model snapshot is missing {record['path']}")
        expected_size = record.get("size_bytes")
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise RuntimeError(f"model snapshot size does not match {record['path']}")
        if _sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"model snapshot checksum does not match {record['path']}")
    return snapshot


def _prompt(prompt: Any, roots: Sequence[Path]) -> dict[str, Any] | None:
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
    return {"path": str(path), "sha256": digest}


def validate_request(
    request: Mapping[str, Any],
    *,
    configured_roots: Any = None,
    expected_model: Mapping[str, str] | None = None,
    supported_languages: Sequence[str] = SUPPORTED_LANGUAGES,
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
    if model.get("family") != MODEL_FAMILY:
        raise WorkerRequestError("invalid_request", "unsupported Chatterbox model family")
    if model.get("t3_model") != MODEL_VARIANT:
        raise WorkerRequestError("invalid_request", "Chatterbox multilingual V2 is required")
    revision = _safe_string(model.get("revision"), "model revision", max_length=512)
    if expected_model is not None:
        expected_revision = expected_model.get("revision")
        if expected_revision and revision != expected_revision:
            raise WorkerRequestError("invalid_request", "model revision does not match the worker")

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
                "voice_prompt": _prompt(prompt, roots["voice"]),
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
    ):
        self.approved_roots = approved_roots
        self.model_revision = model_revision
        self.model_manifest_path = model_manifest_path
        self.approved_model_root = approved_model_root
        self.approved_manifest_root = approved_manifest_root
        self.model_loader = model_loader
        self.model = None
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
            try:
                if self.model_manifest_path is None or self.approved_model_root is None:
                    raise RuntimeError(
                        "a runtime model manifest and approved model cache root are required"
                    )
                if self.approved_manifest_root is None:
                    raise RuntimeError("an approved model manifest root is required")
                raw_model_root = Path(os.fspath(self.approved_model_root)).expanduser()
                if not raw_model_root.is_absolute():
                    raise RuntimeError("approved model root must be absolute")
                if raw_model_root.is_symlink():
                    raise RuntimeError("approved model root must not be a symlink")
                model_root = raw_model_root.resolve(strict=False)
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
                try:
                    with redirect_stdout(sys.stderr):
                        from huggingface_hub import snapshot_download
                except Exception as exc:
                    raise RuntimeError(
                        "huggingface_hub is unavailable; install the pinned worker runtime"
                    ) from exc
                try:
                    with redirect_stdout(sys.stderr):
                        snapshot = snapshot_download(
                            repo_id=manifest["repository"],
                            revision=manifest["revision"],
                            allow_patterns=manifest["allow_patterns"],
                            cache_dir=str(model_root),
                        )
                    snapshot_path = Path(snapshot).expanduser().resolve(strict=False)
                    if not _within(snapshot_path, (model_root,)):
                        raise RuntimeError("downloaded model snapshot is outside the approved cache root")
                    _verify_model_snapshot(snapshot_path, manifest["files"], model_root)
                except Exception as exc:
                    raise RuntimeError(f"pinned Chatterbox model acquisition failed: {exc}") from exc
                try:
                    with redirect_stdout(sys.stderr):
                        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                except Exception as exc:
                    raise RuntimeError(
                        "Chatterbox is unavailable; install the pinned worker runtime"
                    ) from exc
            except Exception as exc:
                raise RuntimeError(f"Chatterbox model load failed: {exc}") from exc
            try:
                with redirect_stdout(sys.stderr):
                    self.model = ChatterboxMultilingualTTS.from_local(
                        str(snapshot_path),
                        device=DEVICE,
                    )
            except Exception as exc:
                raise RuntimeError(f"Chatterbox model load failed: {exc}") from exc
        self.sample_rate = int(getattr(self.model, "sr", SAMPLE_RATE) or SAMPLE_RATE)
        if self.sample_rate != SAMPLE_RATE:
            raise RuntimeError("Chatterbox returned an unsupported sample rate")

    def _load_audio_backend(self) -> tuple[Any, Any]:
        if self._audio_backend is None:
            try:
                import torch
                import torchaudio
            except Exception as exc:
                raise RuntimeError("Torch and torchaudio are required by the worker") from exc
            self._audio_backend = (torch, torchaudio)
        return self._audio_backend

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
            kwargs = {"language_id": request["language"]}
            prompt = segment.get("voice_prompt")
            if prompt is not None:
                kwargs["audio_prompt_path"] = prompt["path"]
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
                expected_model={"revision": self.model_revision} if self.model_revision else None,
            )
            result = self._generate_file(normalized, cancel)
            if cancel.is_set():
                Path(result["path"]).unlink(missing_ok=True)
                raise WorkerCancelled()
            _message({"protocol": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result})
        except WorkerCancelled:
            _message(_error_response(request_id, "cancelled", "synthesis cancelled", retryable=True))
        except WorkerRequestError as exc:
            _message(_error_response(request_id, exc.code, exc.message, retryable=exc.code in {"timeout", "cancelled"}))
        except Exception as exc:
            print(f"worker synthesis error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _message(_error_response(request_id, "generation_failed", "Chatterbox synthesis failed", retryable=False))
        finally:
            with self._active_lock:
                self._active_id = None
                self._active_cancel = None
                self._active_thread = None

    def _start_synthesis(self, request: Mapping[str, Any]) -> None:
        request_id = request.get("id")
        if not isinstance(request_id, str):
            _message(_error_response(request_id, "invalid_request", "request id is required"))
            return
        with self._active_lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                _message(_error_response(request_id, "not_ready", "worker is busy", retryable=True))
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
        _message({
            "protocol": PROTOCOL_VERSION,
            "id": request.get("id"),
            "ok": True,
            "result": {"target_id": target_id, "cancel_requested": accepted},
        })

    def run(self) -> int:
        try:
            self.load_model()
        except Exception as exc:
            print(f"worker model load error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _message({
                "protocol": PROTOCOL_VERSION,
                "event": "error",
                "error": {"code": "model_load_failed", "message": "Chatterbox runtime/model load failed"},
            })
            return 1

        _message({
            "protocol": PROTOCOL_VERSION,
            "event": "ready",
            "device": DEVICE,
            "languages": list(SUPPORTED_LANGUAGES),
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
                    _message({
                        "protocol": PROTOCOL_VERSION,
                        "id": request.get("id"),
                        "ok": True,
                        "result": {
                            "device": DEVICE,
                            "languages": list(SUPPORTED_LANGUAGES),
                            "sample_rate": SAMPLE_RATE,
                        },
                    })
                elif operation == "shutdown":
                    _message({"protocol": PROTOCOL_VERSION, "id": request.get("id"), "ok": True, "result": {}})
                    self._stop.set()
                    break
                else:
                    _message(_error_response(request.get("id"), "invalid_request", "unknown operation"))
            except json.JSONDecodeError:
                _message(_error_response(None, "invalid_request", "message is not valid JSON"))
            except WorkerRequestError as exc:
                _message(_error_response(None, exc.code, exc.message))
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
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
