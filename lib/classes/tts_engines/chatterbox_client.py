"""Dependency-free client for the isolated Chatterbox worker."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = 1
DEVICE = "cpu"


class ChatterboxClientError(RuntimeError):
    """An actionable worker/client failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.error_code = code
        self.retryable = retryable
        self.message = message


class _WorkerEOF(Exception):
    pass


def _redact(value: str, paths: Sequence[Path] = ()) -> str:
    result = value
    for path in paths:
        result = result.replace(str(path), "<approved-path>")
    result = re.sub(
        r"(?i)(hf_token|token|authorization|password)(\s*[=:]\s*)\S+",
        r"\1\2<redacted>",
        result,
    )
    return result[-1000:]


def _normalise_roots(value: Any) -> dict[str, tuple[Path, ...]]:
    if isinstance(value, Mapping):
        shared = value.get("all", ())
        voice = value.get("voice", shared)
        output = value.get("output", shared)
    else:
        voice = output = value
    if isinstance(voice, (str, os.PathLike)):
        voice = [voice]
    if isinstance(output, (str, os.PathLike)):
        output = [output]
    voice_paths_raw = tuple(Path(path).expanduser() for path in (voice or ()))
    output_paths_raw = tuple(Path(path).expanduser() for path in (output or ()))
    if any(not path.is_absolute() for path in (*voice_paths_raw, *output_paths_raw)):
        raise ValueError("approved roots must be absolute")
    voice_paths = tuple(path.resolve(strict=False) for path in voice_paths_raw)
    output_paths = tuple(path.resolve(strict=False) for path in output_paths_raw)
    if not voice_paths or not output_paths:
        raise ValueError("approved voice and output roots are required")
    return {"voice": voice_paths, "output": output_paths}


def _within(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _config_path(value: Any, field: str) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute")
    return path.resolve(strict=False)


class ChatterboxClient:
    """Own one worker process and at most one in-flight synthesis request."""

    def __init__(
        self,
        interpreter: str | os.PathLike[str],
        worker_path: str | os.PathLike[str],
        *,
        approved_roots: Mapping[str, Any] | Sequence[str | os.PathLike[str]],
        model: Mapping[str, str] | None = None,
        readiness_timeout: float = 120.0,
        request_timeout: float = 600.0,
        cancel_grace: float = 2.0,
        worker_args: Sequence[str] = (),
        extra_env: Mapping[str, str] | None = None,
        model_manifest_path: str | os.PathLike[str] | None = None,
        approved_model_root: str | os.PathLike[str] | None = None,
        approved_manifest_root: str | os.PathLike[str] | None = None,
    ):
        self.interpreter = str(Path(interpreter).expanduser())
        self.worker_path = str(Path(worker_path).expanduser().resolve(strict=False))
        self.approved_roots = _normalise_roots(approved_roots)
        self.model = dict(model or {
            "family": "chatterbox-multilingual",
            "revision": "immutable",
            "t3_model": "v2",
        })
        self.readiness_timeout = float(readiness_timeout)
        self.request_timeout = float(request_timeout)
        self.cancel_grace = float(cancel_grace)
        self.worker_args = tuple(str(arg) for arg in worker_args)
        self.extra_env = {str(key): str(value) for key, value in (extra_env or {}).items()}
        self._use_explicit_environment = extra_env is not None
        self.model_manifest_path = _config_path(model_manifest_path, "model manifest")
        self.approved_model_root = _config_path(approved_model_root, "approved model root")
        self.approved_manifest_root = _config_path(approved_manifest_root, "approved manifest root")
        if (self.model_manifest_path is None) != (self.approved_model_root is None):
            raise ValueError("model manifest and approved model root must be supplied together")
        if self.model_manifest_path is not None and self.approved_manifest_root is None:
            raise ValueError("approved manifest root is required with a model manifest")
        self.process: subprocess.Popen[str] | None = None
        self.capabilities: dict[str, Any] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._reader_threads: list[threading.Thread] = []
        self._io_lock = threading.RLock()
        self._busy = False
        self._current_output: Path | None = None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _environment(self) -> dict[str, str]:
        # An explicitly supplied environment is the complete worker
        # allowlist.  Do not start with the host environment and overlay it:
        # that would leak unrelated tokens and interpreter settings across
        # the runtime boundary.
        env = dict(self.extra_env) if self._use_explicit_environment else dict(os.environ)
        for key in (
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV", "CONDA_EXE", "CONDA_PROMPT_MODIFIER",
        ):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _command(self) -> list[str]:
        command = [self.interpreter, "-u", self.worker_path, "--device", DEVICE]
        for root in sorted(str(path) for path in self.approved_roots["voice"]):
            command.extend(("--approved-voice-root", root))
        for root in sorted(str(path) for path in self.approved_roots["output"]):
            command.extend(("--approved-output-root", root))
        command.extend(self.worker_args)
        # These are explicit argv entries, never shell text.  Append them
        # after legacy worker_args so the manifest/cache configuration cannot
        # be overridden by a stale --model-revision label or duplicate option.
        if self.model_manifest_path is not None:
            command.extend(("--model-manifest", str(self.model_manifest_path)))
            command.extend(("--approved-model-root", str(self.approved_model_root)))
            command.extend(("--approved-manifest-root", str(self.approved_manifest_root)))
        return command

    def _start(self) -> None:
        if self.is_running and self.capabilities is not None:
            return
        self._terminate()
        self._stdout_queue = queue.Queue()
        self._stderr_tail.clear()
        try:
            popen_kwargs: dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "bufsize": 1,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "env": self._environment(),
                "shell": False,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            self.process = subprocess.Popen(self._command(), **popen_kwargs)
        except OSError as exc:
            self.process = None
            raise ChatterboxClientError("worker_unavailable", f"could not start Chatterbox worker: {exc}", retryable=True) from exc
        self.capabilities = None
        self._start_readers()
        deadline = time.monotonic() + self.readiness_timeout
        while time.monotonic() < deadline:
            try:
                message = self._read_message(min(0.1, deadline - time.monotonic()))
            except queue.Empty:
                continue
            except _WorkerEOF as exc:
                detail = self._stderr_detail()
                self._terminate()
                raise ChatterboxClientError(
                    "worker_unavailable",
                    f"Chatterbox worker exited before ready{detail}",
                    retryable=True,
                ) from exc
            except ChatterboxClientError:
                self._terminate()
                raise
            if message.get("protocol") != PROTOCOL_VERSION:
                self._terminate()
                raise ChatterboxClientError("protocol_error", "worker used an unsupported protocol version")
            if message.get("event") == "ready":
                if message.get("device") != DEVICE or message.get("sample_rate") != 24000:
                    self._terminate()
                    raise ChatterboxClientError("protocol_error", "worker reported invalid capabilities")
                self.capabilities = dict(message)
                return
            if message.get("event") == "error":
                error = message.get("error") if isinstance(message.get("error"), Mapping) else {}
                code = str(error.get("code") or "worker_unavailable")
                text = str(error.get("message") or "worker failed during startup")
                self._terminate()
                raise ChatterboxClientError(code, _redact(text), retryable=True)
            self._terminate()
            raise ChatterboxClientError("protocol_error", "worker sent an unexpected startup message")
        self._terminate()
        raise ChatterboxClientError("timeout", "Chatterbox worker readiness timed out", retryable=True)

    def _start_readers(self) -> None:
        assert self.process is not None
        process = self.process
        self._reader_threads = []

        def read_stdout() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    self._stdout_queue.put(line)
            finally:
                self._stdout_queue.put(None)

        def read_stderr() -> None:
            assert process.stderr is not None
            try:
                for line in process.stderr:
                    self._stderr_tail.append(_redact(line.rstrip()))
            except OSError:
                pass

        for target in (read_stdout, read_stderr):
            thread = threading.Thread(target=target, daemon=True)
            self._reader_threads.append(thread)
            thread.start()

    def _read_message(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._stdout_queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            raise
        if line is None:
            raise _WorkerEOF()
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChatterboxClientError("protocol_error", "worker emitted malformed JSON") from exc
        if not isinstance(message, dict):
            raise ChatterboxClientError("protocol_error", "worker emitted a non-object message")
        return message

    def _write(self, message: Mapping[str, Any]) -> None:
        if not self.is_running or self.process is None or self.process.stdin is None:
            raise ChatterboxClientError("worker_unavailable", "Chatterbox worker is not running", retryable=True)
        try:
            self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ChatterboxClientError("worker_crash", "Chatterbox worker pipe closed", retryable=True) from exc

    def _stderr_detail(self) -> str:
        if not self._stderr_tail:
            return ""
        return f": {_redact(self._stderr_tail[-1])}"

    def _final_output(self, request: Mapping[str, Any]) -> Path:
        output = request.get("output")
        if not isinstance(output, Mapping):
            raise ChatterboxClientError("invalid_request", "output must be an object")
        raw = output.get("path")
        if not isinstance(raw, str) or not raw:
            raise ChatterboxClientError("invalid_request", "output path is required")
        path = Path(raw).expanduser().resolve(strict=False)
        if path.name.endswith(".part"):
            path = path.with_name(path.name[:-5])
        if path.suffix.lower() != ".flac" or not _within(path, self.approved_roots["output"]):
            raise ChatterboxClientError("invalid_request", "output path is outside approved FLAC roots")
        return path

    def _validate_client_request(self, request: Mapping[str, Any]) -> Path:
        if request.get("device", DEVICE) != DEVICE:
            raise ChatterboxClientError("unsupported_device", "Chatterbox client supports CPU only")
        language = request.get("language")
        if not isinstance(language, str) or not language:
            raise ChatterboxClientError("invalid_request", "language is required")
        for segment in request.get("segments", ()):
            if not isinstance(segment, Mapping) or segment.get("kind") != "text":
                continue
            prompt = segment.get("voice_prompt") or request.get("reference_prompt")
            if isinstance(prompt, Mapping) and isinstance(prompt.get("path"), str):
                prompt_path = Path(prompt["path"]).expanduser().resolve(strict=False)
                if not _within(prompt_path, self.approved_roots["voice"]):
                    raise ChatterboxClientError("invalid_request", "voice prompt is outside approved roots")
        return self._final_output(request)

    def ping(self) -> dict[str, Any]:
        with self._io_lock:
            self._start()
            request_id = uuid.uuid4().hex
            try:
                self._write({"protocol": PROTOCOL_VERSION, "id": request_id, "op": "ping"})
                return self._wait_for_response(request_id, self.request_timeout)
            except ChatterboxClientError:
                self._terminate()
                raise

    def _wait_for_response(self, request_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChatterboxClientError("timeout", "Chatterbox worker request timed out", retryable=True)
            try:
                message = self._read_message(min(0.1, remaining))
            except queue.Empty:
                continue
            except _WorkerEOF as exc:
                raise ChatterboxClientError("worker_crash", "Chatterbox worker exited unexpectedly", retryable=True) from exc
            if message.get("protocol") != PROTOCOL_VERSION:
                raise ChatterboxClientError("protocol_error", "worker used an unsupported protocol version")
            if message.get("event") is not None:
                continue
            if message.get("id") != request_id:
                continue
            if message.get("ok") is True:
                result = message.get("result")
                if not isinstance(result, Mapping):
                    raise ChatterboxClientError("protocol_error", "worker success response has no result")
                return dict(result)
            error = message.get("error")
            if not isinstance(error, Mapping):
                raise ChatterboxClientError("protocol_error", "worker failure response has no error")
            raise ChatterboxClientError(
                str(error.get("code") or "generation_failed"),
                _redact(str(error.get("message") or "Chatterbox request failed")),
                retryable=bool(error.get("retryable", False)),
            )

    def synthesize(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float | None = None,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        with self._io_lock:
            if self._busy:
                raise ChatterboxClientError("not_ready", "Chatterbox client already has a request in flight", retryable=True)
            final_output = self._validate_client_request(request)
            self._start()
            request_id = uuid.uuid4().hex
            payload = dict(request)
            payload.update({
                "protocol": PROTOCOL_VERSION,
                "id": request_id,
                "op": "synthesize",
                "device": DEVICE,
                "model": dict(payload.get("model") or self.model),
                "approved_roots": {
                    "voice": [str(path) for path in self.approved_roots["voice"]],
                    "output": [str(path) for path in self.approved_roots["output"]],
                },
            })
            self._busy = True
            self._current_output = final_output
            try:
                try:
                    self._write(payload)
                except ChatterboxClientError:
                    self._abort_current()
                    raise
                deadline = time.monotonic() + (self.request_timeout if timeout is None else float(timeout))
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        self._send_cancel(request_id)
                        cancel_deadline = time.monotonic() + self.cancel_grace
                        while time.monotonic() < cancel_deadline:
                            try:
                                message = self._read_message(min(0.05, cancel_deadline - time.monotonic()))
                            except queue.Empty:
                                continue
                            except (_WorkerEOF, ChatterboxClientError):
                                break
                            if message.get("id") == request_id:
                                break
                        self._abort_current()
                        raise ChatterboxClientError("cancelled", "Chatterbox synthesis cancelled", retryable=True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._send_cancel(request_id)
                        self._abort_current()
                        raise ChatterboxClientError("timeout", "Chatterbox synthesis timed out", retryable=True)
                    try:
                        message = self._read_message(min(0.1, remaining))
                    except queue.Empty:
                        continue
                    except _WorkerEOF as exc:
                        self._abort_current()
                        raise ChatterboxClientError("worker_crash", "Chatterbox worker exited unexpectedly", retryable=True) from exc
                    except ChatterboxClientError:
                        self._abort_current()
                        raise
                    if message.get("protocol") != PROTOCOL_VERSION:
                        self._abort_current()
                        raise ChatterboxClientError("protocol_error", "worker used an unsupported protocol version")
                    if message.get("event") is not None or message.get("id") != request_id:
                        continue
                    if message.get("ok") is True:
                        result = message.get("result")
                        if not isinstance(result, Mapping):
                            self._abort_current()
                            raise ChatterboxClientError("protocol_error", "worker success response has no result")
                        self._validate_result(result, final_output)
                        return dict(result)
                    error = message.get("error")
                    if not isinstance(error, Mapping):
                        self._abort_current()
                        raise ChatterboxClientError("protocol_error", "worker failure response has no error")
                    code = str(error.get("code") or "generation_failed")
                    if code in {"timeout", "cancelled"}:
                        self._remove_output_artifacts()
                    raise ChatterboxClientError(
                        code,
                        _redact(str(error.get("message") or "Chatterbox request failed")),
                        retryable=bool(error.get("retryable", False)),
                    )
            except ChatterboxClientError:
                raise
            finally:
                self._current_output = None
                self._busy = False

    def _validate_result(self, result: Mapping[str, Any], expected: Path) -> None:
        result_path = result.get("path")
        if not isinstance(result_path, str) or Path(result_path).resolve(strict=False) != expected:
            self._abort_current()
            raise ChatterboxClientError("output_invalid", "worker returned an unexpected output path")
        if not expected.is_file() or expected.stat().st_size <= 0:
            self._abort_current()
            raise ChatterboxClientError("output_invalid", "worker reported an empty output")
        if result.get("sample_rate") != 24000 or result.get("channels") != 1:
            self._abort_current()
            raise ChatterboxClientError("output_invalid", "worker returned non-mono 24 kHz output")
        reported_hash = result.get("sha256")
        if isinstance(reported_hash, str):
            digest = hashlib.sha256(expected.read_bytes()).hexdigest()
            if digest != reported_hash:
                self._abort_current()
                raise ChatterboxClientError("output_invalid", "worker output checksum does not match")
        if Path(str(expected) + ".part").exists():
            self._abort_current()
            raise ChatterboxClientError("output_invalid", "worker left a partial output")

    def _send_cancel(self, request_id: str) -> None:
        try:
            self._write({
                "protocol": PROTOCOL_VERSION,
                "id": uuid.uuid4().hex,
                "op": "cancel",
                "target_id": request_id,
            })
        except ChatterboxClientError:
            pass

    def _remove_output_artifacts(self) -> None:
        if self._current_output is None:
            return
        output = self._current_output
        artifacts = {
            output,
            Path(str(output) + ".part"),
            *output.parent.glob(f".{output.name}.*.part"),
            *output.parent.glob(f".{output.name}.*.part.flac"),
        }
        for path in artifacts:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _abort_current(self) -> None:
        self._remove_output_artifacts()
        self._terminate()

    def _terminate(self) -> None:
        process = self.process
        self.process = None
        self.capabilities = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    process.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def abort(self) -> None:
        with self._io_lock:
            self._abort_current()
            self._busy = False
            self._current_output = None

    def close(self) -> None:
        with self._io_lock:
            if self._busy:
                self._abort_current()
            elif self.is_running:
                try:
                    request_id = uuid.uuid4().hex
                    self._write({"protocol": PROTOCOL_VERSION, "id": request_id, "op": "shutdown"})
                    self._wait_for_response(request_id, min(2.0, self.cancel_grace + 1.0))
                except (ChatterboxClientError, _WorkerEOF):
                    pass
                finally:
                    self._terminate()
            self._busy = False
            self._current_output = None

    def __enter__(self) -> "ChatterboxClient":
        self._start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
