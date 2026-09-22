"""Host-side adapter for the isolated Chatterbox CPU worker.

The host process owns E2A's sentence/SML/voice conventions. Chatterbox and
its incompatible dependency graph remain behind :class:`ChatterboxClient`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence
import wave

from lib.classes.tts_engines.chatterbox_client import ChatterboxClient, ChatterboxClientError
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.utils import TTSUtils
from lib.classes.tts_registry import TTSRegistry
from lib.conf import devices, run_dir, voices_dir
from lib.conf_chatterbox_languages import chatterbox_language_id
from lib.conf_models import TTS_ENGINES, SML_TAG_PATTERN, chatterbox_target_status


SAMPLE_RATE = 24000
CHANNELS = 1
DEVICE = devices["CPU"]["proc"]
MODEL_FAMILY = "chatterbox-multilingual"
DEFAULT_MODEL_VARIANT = "v2"
DEFAULT_BREAK_SECONDS = 0.4
DEFAULT_PAUSE_SECONDS = 0.8
MAX_SILENCE_SECONDS = 30.0


@dataclass(frozen=True)
class _RuntimeDetails:
    """Resolved host-side inputs for the isolated Chatterbox runtime."""

    interpreter: Path
    environment: dict[str, str]
    manifest_path: Path
    model_root: Path
    manifest_root: Path
    model_revision: str


def _selected_model_variant(session: Any) -> str:
    requested = session.get("fine_tuned") if hasattr(session, "get") else None
    if requested in (None, "", "internal", "v2"):
        return "v2"
    if requested == "v3":
        return "v3"
    return DEFAULT_MODEL_VARIANT


def _manifest_path(repo_root: Path, variant: str) -> Path:
    filename = "runtime-manifest-v3.json" if variant == "v3" else "runtime-manifest.json"
    return repo_root / "components" / "Chatterbox" / "runtime" / filename


def chatterbox_host_status(session: Any, *, repo_root: Path | None = None) -> Mapping[str, Any]:
    """Return the runtime-owned, host-facing readiness record.

    The static target check remains here because the runtime contract
    describes the approved host, while this adapter also needs to reject a
    selected non-CPU device.  Artifact, capacity, receipt, and manifest
    semantics belong to ``host_runtime_status``.  The runtime import stays
    lazy so registry discovery does not load the optional runtime package.
    """

    device = session.get("device", DEVICE) if hasattr(session, "get") else DEVICE
    target = chatterbox_target_status(device)
    if not target["supported"]:
        return {
            "ok": False,
            "supported": False,
            "status": "unsupported",
            "artifact_status": "unsupported",
            "capacity_status": "not_checked",
            "error": target["error"],
            "target": target,
        }

    from components.Chatterbox.runtime.runtime import host_runtime_status

    root = repo_root or Path(__file__).resolve().parents[3]
    variant = _selected_model_variant(session)
    return host_runtime_status(
        repo_root=root,
        manifest_path=_manifest_path(root, variant),
    )


class _SessionCancellation:
    """Adapt E2A's session flag to the client's event-like interface."""

    def __init__(self, session: Any):
        self.session = session

    def is_set(self) -> bool:
        return bool(self.session.get("cancellation_requested", False))


def _within(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _unique_paths(values: Sequence[Any]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser().resolve(strict=False)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def _status_path(status: Mapping[str, Any], field: str) -> Path:
    value = status.get(field)
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"Chatterbox runtime status is missing {field}")
    return Path(value).expanduser().resolve(strict=False)


def _runtime_details(variant: str = DEFAULT_MODEL_VARIANT) -> _RuntimeDetails:
    """Resolve worker inputs from the stable host status contract."""

    status = chatterbox_host_status({"device": DEVICE, "fine_tuned": variant})
    if not status.get("ok"):
        raise ValueError(status.get("error") or "Chatterbox is not ready")

    environment = status.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("Chatterbox runtime status is missing environment")
    model_revision = status.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("Chatterbox runtime status is missing model_revision")

    return _RuntimeDetails(
        interpreter=_status_path(status, "interpreter"),
        environment={str(key): str(value) for key, value in environment.items()},
        manifest_path=_status_path(status, "manifest_path"),
        model_root=_status_path(status, "verified_model_root"),
        manifest_root=_status_path(status, "manifest_root"),
        model_revision=model_revision,
    )


def _audio_file_is_valid(path: Path) -> tuple[bool, str | None]:
    """Validate the worker's file at the host boundary using soundfile."""

    if not path.is_file() or path.stat().st_size <= 0:
        return False, "Chatterbox produced an empty audio file"
    try:
        import numpy as np
        import soundfile as sf

        data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as soundfile_error:
        # The root E2A environment may not carry soundfile.  Reuse the
        # application's existing FFmpeg toolchain rather than adding a root
        # dependency just for this isolated engine's boundary check.
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return False, f"Chatterbox output could not be decoded: {soundfile_error}"
        try:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=channels,sample_rate,nb_frames,duration",
                    "-of", "json",
                    str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30,
            )
            if probe.returncode != 0:
                return False, "Chatterbox output could not be decoded by FFmpeg"
            streams = json.loads(probe.stdout).get("streams", [])
            metadata = streams[0] if streams else {}
            channels = int(metadata.get("channels", 0))
            samplerate = int(metadata.get("sample_rate", 0))
            frames = int(metadata.get("nb_frames", 0) or 0)
            duration = float(metadata.get("duration", 0) or 0)
            if channels != CHANNELS or samplerate != SAMPLE_RATE or (frames <= 0 and duration <= 0):
                return False, "Chatterbox output must be nonempty mono 24 kHz audio"
            return True, None
        except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            return False, f"Chatterbox output could not be decoded: {exc}"
    shape = getattr(data, "shape", ())
    # soundfile returns frames x channels, unlike torchaudio's channels x
    # frames representation used inside the worker.
    if len(shape) != 2 or int(shape[1]) != CHANNELS or int(shape[0]) <= 0:
        return False, "Chatterbox output must be nonempty mono audio"
    if int(samplerate) != SAMPLE_RATE:
        return False, "Chatterbox output must be 24 kHz audio"
    if not bool(np.isfinite(data).all()):
        return False, "Chatterbox output contains non-finite samples"
    return True, None


class Chatterbox(TTSUtils, TTSRegistry, name="chatterbox"):
    """E2A engine seam backed by one persistent isolated CPU worker."""

    def __init__(self, session: Any):
        self.session = session
        self.tts_engine = TTS_ENGINES["CHATTERBOX"]
        self.models = load_engine_presets(self.tts_engine)
        requested_model = self.session.get("fine_tuned") or "v2"
        if requested_model == "internal":
            requested_model = "v2"
        if requested_model not in self.models:
            raise ValueError(f"Invalid Chatterbox model {requested_model!r}")
        self.fine_tuned = requested_model
        self.model_variant = str(self.models[requested_model].get("variant", requested_model))
        readiness = chatterbox_host_status(
            {**dict(session), "fine_tuned": self.model_variant}
            if isinstance(session, Mapping)
            else session
        )
        if not readiness.get("ok"):
            raise ValueError(readiness.get("error") or "Chatterbox is not ready")
        self.tts_key = self.session.get("model_cache") or f"chatterbox-{self.model_variant}"
        self.tts_zs_key = None
        self.device = self.session.get("device", DEVICE)
        if self.device != DEVICE:
            raise ValueError("Chatterbox first slice supports CPU only")

        language = (
            self.session.get("translate")
            if self.session.get("translate_enabled") and self.session.get("translate")
            else self.session.get("language")
        )
        self.language = language
        self.language_id = chatterbox_language_id(language)
        if self.language_id is None:
            raise ValueError(f"Language {language!r} is not supported by Chatterbox first slice")

        self.params = {"samplerate": SAMPLE_RATE, "current_voice": None}
        self._client: ChatterboxClient | None = None
        self._client_output_roots: tuple[Path, ...] = ()
        self._runtime: _RuntimeDetails | None = None
        self._closed = False

        self.voice_roots = _unique_paths(
            (
                voices_dir,
                self.session.get("voice_dir"),
                self.session.get("custom_model_dir"),
            )
        )
        if not self.voice_roots:
            raise ValueError("No approved voice root is available for Chatterbox")

    def _set_voice(self, voice: str | None) -> tuple[str | None, str | None]:
        """Accept only existing normalized WAV prompts under E2A voice roots."""

        selected = voice if voice is not None else self.session.get("voice")
        if selected is None:
            self.params["current_voice"] = None
            return None, None
        if not isinstance(selected, (str, os.PathLike)):
            return None, "Chatterbox voice prompt must be a path or None"
        path = Path(selected).expanduser().resolve(strict=False)
        if path.suffix.lower() != ".wav":
            return None, "Chatterbox voice prompt must be a normalized WAV file"
        if not _within(path, self.voice_roots):
            return None, "Chatterbox voice prompt is outside approved voice roots"
        if not path.is_file() or path.stat().st_size <= 0:
            return None, f"Chatterbox voice prompt does not exist or is empty: {path}"
        try:
            with wave.open(str(path), "rb") as stream:
                if stream.getnchannels() != CHANNELS or stream.getframerate() != SAMPLE_RATE or stream.getnframes() <= 0:
                    return None, "Chatterbox voice prompt must be nonempty mono 24 kHz WAV audio"
        except (EOFError, OSError, wave.Error) as exc:
            return None, f"Chatterbox voice prompt could not be decoded: {exc}"
        self.params["current_voice"] = str(path)
        return str(path), None

    def _voice_prompt(self, voice: str | None) -> dict[str, str] | None:
        if voice is None:
            return None
        path = Path(voice)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _runtime_contract(self) -> _RuntimeDetails:
        if self._runtime is None:
            self._runtime = _runtime_details(self.model_variant)
        return self._runtime

    @staticmethod
    def _tag_value(part: str) -> tuple[str, bool, str | None] | None:
        match = SML_TAG_PATTERN.fullmatch(part)
        if not match:
            return None
        return match.group("tag"), bool(match.group("close")), match.group("value")

    def _silence_seconds(self, tag: str, value: str | None) -> float:
        if tag == "break":
            seconds = DEFAULT_BREAK_SECONDS
        elif value:
            try:
                seconds = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Chatterbox pause duration must be numeric") from exc
        else:
            seconds = DEFAULT_PAUSE_SECONDS
        if not math.isfinite(seconds) or seconds <= 0 or seconds > MAX_SILENCE_SECONDS:
            raise ValueError("Chatterbox pause duration is outside the supported range")
        return seconds

    def build_request(
        self,
        sentence_file: str | os.PathLike[str],
        sentence: str,
        block_voice: str | None = None,
    ) -> dict[str, Any]:
        """Translate E2A SML into worker text/silence segments."""

        base_voice, error = self._set_voice(block_voice)
        if error:
            raise ValueError(error)
        current_voice = base_voice
        segments: list[dict[str, Any]] = []
        for raw_part in self._split_sentence_on_sml(sentence):
            part = raw_part.strip()
            if not part:
                continue
            tag = self._tag_value(part)
            if tag is not None:
                name, closing, value = tag
                if name in {"break", "pause"}:
                    if closing:
                        raise ValueError(f"Closing [{name}] tag is not supported")
                    segments.append({"kind": "silence", "seconds": self._silence_seconds(name, value)})
                elif name == "voice":
                    if closing:
                        current_voice = base_voice
                        self.params["current_voice"] = current_voice
                    else:
                        if not value:
                            raise ValueError("Chatterbox voice tag requires a WAV path")
                        current_voice, error = self._set_voice(value)
                        if error:
                            raise ValueError(error)
                continue
            if not any(char.isalnum() for char in part):
                continue
            segment: dict[str, Any] = {"kind": "text", "text": part}
            prompt = self._voice_prompt(current_voice)
            if prompt is not None:
                segment["voice_prompt"] = prompt
            segments.append(segment)

        if not segments:
            raise ValueError("Chatterbox sentence contains no speakable text or pause")
        runtime = self._runtime_contract()
        return {
            "model": {
                "family": MODEL_FAMILY,
                "revision": runtime.model_revision,
                "t3_model": self.model_variant,
            },
            "language": self.language_id,
            "segments": segments,
            "output": {
                "path": str(Path(sentence_file).expanduser().resolve(strict=False)),
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
            },
            "device": DEVICE,
        }

    def _output_roots(self, sentence_file: Path) -> tuple[Path, ...]:
        return _unique_paths(
            (
                self.session.get("process_dir"),
                self.session.get("sentences_dir"),
                sentence_file.parent,
                Path(run_dir) / "components" / "chatterbox",
            )
        )

    def _get_client(self, sentence_file: Path) -> ChatterboxClient:
        roots = self._output_roots(sentence_file)
        if self._client is not None:
            if _within(sentence_file, self._client_output_roots):
                return self._client
            self._client.close()
            self._client = None
        runtime = self._runtime_contract()
        worker_path = Path(__file__).resolve().parents[3] / "components" / "Chatterbox" / "worker.py"
        self._client_output_roots = roots
        self._client = ChatterboxClient(
            interpreter=runtime.interpreter,
            worker_path=worker_path,
            approved_roots={"voice": self.voice_roots, "output": roots},
            model={
                "family": MODEL_FAMILY,
                "revision": runtime.model_revision,
                "t3_model": self.model_variant,
            },
            extra_env=runtime.environment,
            model_manifest_path=runtime.manifest_path,
            approved_model_root=runtime.model_root,
            approved_manifest_root=runtime.manifest_root,
        )
        return self._client

    def convert(self, sentence_file: str, sentence: str, **kwargs: Any) -> tuple[bool, str | None]:
        try:
            if self._closed:
                return False, "Chatterbox engine is closed"
            output_path = Path(sentence_file).expanduser().resolve(strict=False)
            request = self.build_request(
                output_path,
                sentence,
                kwargs.get("block_voice", self.session.get("voice")),
            )
            client = self._get_client(output_path)
            result = client.synthesize(request, cancel_event=_SessionCancellation(self.session))
            if self.session.get("cancellation_requested", False):
                self.abort()
                return False, "Chatterbox synthesis cancelled"
            reported_path = Path(str(result.get("path", output_path))).resolve(strict=False)
            if reported_path != output_path:
                self.abort()
                return False, "Chatterbox worker returned an unexpected output path"
            valid, error = _audio_file_is_valid(output_path)
            if not valid:
                self.abort()
                return False, error
            return True, None
        except ChatterboxClientError as exc:
            if exc.code in {"timeout", "worker_crash", "cancelled", "output_invalid"}:
                self.abort()
            return False, f"Chatterbox {exc.code}: {exc.message}"
        except Exception as exc:
            self.abort()
            return False, f"Chatterbox convert failed: {exc}"

    def abort(self) -> None:
        if self._client is not None:
            self._client.abort()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
