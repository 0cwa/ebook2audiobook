"""Focused host-adapter tests without importing the Chatterbox package."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import patch

from lib.classes.tts_registry import TTSRegistry


def _load_chatterbox_adapter():
    """Load only this adapter without importing unrelated optional engines."""

    repository = Path(__file__).resolve().parents[3]
    package_name = "lib.classes.tts_engines"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repository / "lib/classes/tts_engines")]
    sys.modules[package_name] = package

    from lib.conf_models import SML_TAG_PATTERN

    utils_name = "lib.classes.tts_engines.common.utils"
    utils = types.ModuleType(utils_name)

    class TTSUtils:
        @staticmethod
        def _split_sentence_on_sml(sentence: str) -> list[str]:
            parts: list[str] = []
            last = 0
            for match in SML_TAG_PATTERN.finditer(sentence):
                start, end = match.span()
                if start > last:
                    parts.append(sentence[last:start])
                parts.append(match.group(0))
                last = end
            if last < len(sentence):
                parts.append(sentence[last:])
            return parts

    utils.TTSUtils = TTSUtils
    sys.modules[utils_name] = utils
    return importlib.import_module("lib.classes.tts_engines.chatterbox")


chatterbox_module = _load_chatterbox_adapter()


class FakeClient:
    instances = []
    write_valid_audio = True
    cancel_immediately = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.abort_calls = 0
        self.close_calls = 0
        self.__class__.instances.append(self)

    def synthesize(self, request, *, cancel_event=None):
        self.requests.append(request)
        if self.cancel_immediately or (cancel_event is not None and cancel_event.is_set()):
            raise chatterbox_module.ChatterboxClientError("cancelled", "fake cancellation", retryable=True)
        output = Path(request["output"]["path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.write_valid_audio:
            output.write_bytes(b"fake-audio")
        else:
            output.write_bytes(b"not-a-flac")
        return {
            "path": str(output),
            "sample_rate": 24000,
            "channels": 1,
        }

    def abort(self):
        self.abort_calls += 1

    def close(self):
        self.close_calls += 1


class ChatterboxAdapterTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances.clear()
        FakeClient.write_valid_audio = True
        FakeClient.cancel_immediately = False

    @staticmethod
    def session(root: Path, language: str = "eng", *, voice: str | None = None, device: str = "cpu"):
        voice_dir = root / "voices"
        process_dir = root / "process"
        (process_dir / "chapters" / "sentences").mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)
        return {
            "device": device,
            "language": language,
            "language_iso1": "en" if language == "eng" else "sv",
            "translate_enabled": False,
            "translate": None,
            "fine_tuned": "internal",
            "model_cache": "chatterbox-internal",
            "voice": voice,
            "voice_dir": str(voice_dir),
            "process_dir": str(process_dir),
            "sentences_dir": str(process_dir / "chapters" / "sentences"),
            "custom_model_dir": None,
            "cancellation_requested": False,
        }

    @staticmethod
    def make_voice(path: Path):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(b"\x00\x00" * 240)

    def test_registration_registers_chatterbox_without_importing_runtime_package(self):
        self.assertIn("chatterbox", TTSRegistry.ENGINES)
        self.assertNotIn("chatterbox", sys.modules)
        from lib.conf_models import default_engine_settings, TTS_ENGINES
        self.assertIn("watermark", default_engine_settings[TTS_ENGINES["CHATTERBOX"]]["notice"])

    def test_language_and_device_rejection_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "not supported"):
                chatterbox_module.Chatterbox(self.session(root, "deu"))
            with self.assertRaisesRegex(ValueError, "CPU only"):
                chatterbox_module.Chatterbox(self.session(root, device="cuda"))

    def test_request_translation_removes_sml_and_preserves_voice_and_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice = root / "voices" / "base.wav"
            inline = root / "voices" / "inline.wav"
            engine = chatterbox_module.Chatterbox(self.session(root, voice=str(voice)))
            self.make_voice(voice)
            self.make_voice(inline)
            request = engine.build_request(
                root / "process" / "sentence.flac",
                f"Before [voice:{inline}] after [/voice] [break] final [pause:1.25]",
            )
            self.assertEqual(request["language"], "en")
            self.assertEqual([part["kind"] for part in request["segments"]], ["text", "text", "silence", "text", "silence"])
            self.assertTrue(all("[" not in part.get("text", "") for part in request["segments"]))
            self.assertEqual(request["segments"][0]["voice_prompt"]["path"], str(voice.resolve()))
            self.assertEqual(request["segments"][1]["voice_prompt"]["path"], str(inline.resolve()))
            self.assertEqual(request["segments"][3]["voice_prompt"]["path"], str(voice.resolve()))
            self.assertEqual(request["segments"][2]["seconds"], 0.4)
            self.assertEqual(request["segments"][4]["seconds"], 1.25)
            self.assertEqual(
                request["segments"][1]["voice_prompt"]["sha256"],
                hashlib.sha256(inline.read_bytes()).hexdigest(),
            )

    def test_voice_path_is_limited_to_normalized_wav_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "voices" / "valid.wav"
            outside = root / "outside.wav"
            engine = chatterbox_module.Chatterbox(self.session(root))
            self.make_voice(valid)
            self.make_voice(outside)
            selected, error = engine._set_voice(str(valid))
            self.assertEqual(selected, str(valid.resolve()))
            self.assertIsNone(error)
            selected, error = engine._set_voice(str(outside))
            self.assertIsNone(selected)
            self.assertIn("outside approved", error)

    def test_success_path_uses_client_and_host_decodes_valid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = chatterbox_module.Chatterbox(self.session(root))
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            with (
                patch.object(chatterbox_module, "ChatterboxClient", FakeClient),
                patch.object(chatterbox_module, "_audio_file_is_valid", return_value=(True, None)),
            ):
                success, error = engine.convert(str(output), "Hello world.")
            self.assertTrue(success)
            self.assertIsNone(error)
            self.assertTrue(output.is_file())
            self.assertEqual(FakeClient.instances[0].requests[0]["segments"][0]["text"], "Hello world.")

    def test_client_uses_pinned_runtime_manifest_and_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            model_root = root / "models" / "tts" / "chatterbox"
            manifest = runtime_root / "runtime-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            revision = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
            details = chatterbox_module._RuntimeDetails(
                interpreter=root / "env" / "bin" / "python",
                environment={"E2A_ROOT": str(root)},
                manifest_path=manifest,
                model_root=model_root,
                manifest_root=runtime_root,
                model_revision=revision,
            )
            engine = chatterbox_module.Chatterbox(self.session(root))
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            with (
                patch.dict("os.environ", {"E2A_CHATTERBOX_MODEL_REVISION": "immutable"}),
                patch.object(chatterbox_module, "_runtime_details", return_value=details),
                patch.object(chatterbox_module, "ChatterboxClient", FakeClient),
                patch.object(chatterbox_module, "_audio_file_is_valid", return_value=(True, None)),
            ):
                success, error = engine.convert(str(output), "Hello world.")
            self.assertTrue(success)
            self.assertIsNone(error)
            client = FakeClient.instances[0]
            self.assertEqual(client.kwargs["interpreter"], details.interpreter)
            self.assertEqual(client.kwargs["extra_env"], details.environment)
            self.assertEqual(client.kwargs["model_manifest_path"], details.manifest_path)
            self.assertEqual(client.kwargs["approved_model_root"], details.model_root)
            self.assertEqual(client.kwargs["approved_manifest_root"], details.manifest_root)
            self.assertEqual(client.kwargs["model"]["revision"], revision)
            self.assertEqual(client.requests[0]["model"]["revision"], revision)
            self.assertNotEqual(client.requests[0]["model"]["revision"], "immutable")

    def test_invalid_output_is_rejected_and_aborts_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = chatterbox_module.Chatterbox(self.session(root))
            FakeClient.write_valid_audio = False
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            with (
                patch.object(chatterbox_module, "ChatterboxClient", FakeClient),
                patch.object(
                    chatterbox_module,
                    "_audio_file_is_valid",
                    return_value=(False, "fake invalid audio"),
                ),
            ):
                success, error = engine.convert(str(output), "Hello world.")
            self.assertFalse(success)
            self.assertIn("fake invalid audio", error)
            self.assertEqual(FakeClient.instances[0].abort_calls, 1)

    def test_host_audio_validation_falls_back_to_ffprobe(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.flac"
            output.write_bytes(b"not-decoded-by-root-soundfile")
            probe = types.SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"channels":1,"sample_rate":"24000","nb_frames":"100","duration":"0.1"}]}',
                stderr="",
            )
            with (
                patch.object(chatterbox_module.shutil, "which", return_value="ffprobe"),
                patch.object(chatterbox_module.subprocess, "run", return_value=probe),
            ):
                valid, error = chatterbox_module._audio_file_is_valid(output)
            self.assertTrue(valid)
            self.assertIsNone(error)

    def test_cancellation_aborts_and_close_closes_persistent_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.session(root)
            engine = chatterbox_module.Chatterbox(session)
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            FakeClient.cancel_immediately = True
            with patch.object(chatterbox_module, "ChatterboxClient", FakeClient):
                success, error = engine.convert(str(output), "Hello world.")
                self.assertFalse(success)
                self.assertIn("cancelled", error)
                self.assertEqual(FakeClient.instances[0].abort_calls, 1)
                engine.close()
            self.assertEqual(FakeClient.instances[0].close_calls, 1)
            self.assertFalse(engine.convert(str(output), "Hello again.")[0])


if __name__ == "__main__":
    unittest.main()
