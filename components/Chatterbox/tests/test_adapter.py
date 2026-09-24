"""Focused host-adapter tests without importing the Chatterbox package."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import patch

from components.Chatterbox.tests._adapter_loader import load_chatterbox_adapter
from lib.classes.tts_registry import TTSRegistry


chatterbox_module = load_chatterbox_adapter()
ORIGINAL_HOST_STATUS = chatterbox_module.chatterbox_host_status
ORIGINAL_RUNTIME_DETAILS = chatterbox_module._runtime_details


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
        readiness = patch.object(
            chatterbox_module,
            "chatterbox_host_status",
            return_value={"ok": True, "supported": True, "status": "ready", "error": None},
        )
        readiness.start()
        self.addCleanup(readiness.stop)
        runtime = patch.object(
            chatterbox_module,
            "_runtime_details",
            return_value=chatterbox_module._RuntimeDetails(
                interpreter=Path("/approved/env/bin/python"),
                environment={},
                manifest_path=Path("/approved/runtime-manifest.json"),
                model_root=Path("/approved/model"),
                manifest_root=Path("/approved"),
                model_profile="v2",
                model_fingerprint="chatterbox-v2-aaaaaaaaaaaaaaaa",
                model_revision="5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
            ),
        )
        runtime.start()
        self.addCleanup(runtime.stop)

    @staticmethod
    def session(
        root: Path,
        language: str = "eng",
        *,
        voice: str | None = None,
        device: str = "cpu",
        model: str = "internal",
    ):
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
            "fine_tuned": model,
            "model_cache": "chatterbox-internal" if model == "internal" else f"chatterbox-{model}",
            "voice": voice,
            "voice_dir": str(voice_dir),
            "process_dir": str(process_dir),
            "sentences_dir": str(process_dir / "chapters" / "sentences"),
            "custom_model_dir": None,
            "cancellation_requested": False,
        }

    @staticmethod
    def make_voice(path: Path, *, seconds: float = 0.01):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(b"\x00\x00" * int(round(24000 * seconds)))

    def test_registration_is_isolated_without_importing_runtime_package(self):
        self.assertIs(
            chatterbox_module._test_registry["chatterbox"],
            chatterbox_module.Chatterbox,
        )
        self.assertIsNot(
            TTSRegistry.ENGINES.get("chatterbox"),
            chatterbox_module.Chatterbox,
        )
        self.assertNotIn("chatterbox", sys.modules)
        from lib.conf_models import default_engine_settings, TTS_ENGINES
        self.assertIn("watermark", default_engine_settings[TTS_ENGINES["CHATTERBOX"]]["notice"])

    def test_preloaded_production_adapter_is_reloaded_for_isolated_import(self):
        probe = """
import importlib
import sys

from lib.classes.tts_registry import TTSRegistry

production = importlib.import_module("lib.classes.tts_engines.chatterbox")
production_registry = TTSRegistry.ENGINES
production_registry_snapshot = dict(production_registry)
production_class = production.Chatterbox
production_presets_loader = production.load_engine_presets

from components.Chatterbox.tests import _adapter_loader

loaded = _adapter_loader.load_chatterbox_adapter()
assert loaded is not production
assert sys.modules["lib.classes.tts_engines.chatterbox"] is production
assert TTSRegistry.ENGINES is production_registry
assert TTSRegistry.ENGINES == production_registry_snapshot
assert TTSRegistry.ENGINES["chatterbox"] is production_class
assert production.load_engine_presets is production_presets_loader
assert loaded._test_registry["chatterbox"] is loaded.Chatterbox
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_language_and_device_rejection_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "not supported"):
                chatterbox_module.Chatterbox(self.session(root, "xx"))
            with self.assertRaisesRegex(ValueError, "CPU only"):
                chatterbox_module.Chatterbox(self.session(root, device="cuda"))

    def test_static_target_contract_is_linux_x86_64_cpu_only(self):
        from lib.conf_models import chatterbox_target_status

        self.assertTrue(chatterbox_target_status("cpu", system="linux", architecture="x86_64")["supported"])
        self.assertTrue(chatterbox_target_status("cpu", system="linux", architecture="amd64")["supported"])
        for device, system, architecture in (
            ("cuda", "linux", "x86_64"),
            ("cpu", "darwin", "x86_64"),
            ("cpu", "linux", "aarch64"),
        ):
            status = chatterbox_target_status(device, system=system, architecture=architecture)
            self.assertFalse(status["supported"])
            self.assertIn("Linux x86_64/amd64 CPU only", status["error"])

    def test_real_host_status_rejects_unsupported_os_and_arch_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for system, architecture in (("Darwin", "x86_64"), ("Linux", "aarch64")):
                with (
                    self.subTest(system=system, architecture=architecture),
                    patch.object(chatterbox_module, "chatterbox_host_status", ORIGINAL_HOST_STATUS),
                    patch("lib.conf_models.platform.system", return_value=system),
                    patch("lib.conf_models.platform.machine", return_value=architecture),
                    patch.object(chatterbox_module, "ChatterboxClient") as client,
                    self.assertRaisesRegex(ValueError, "supports Linux x86_64/amd64 CPU only"),
                ):
                    chatterbox_module.Chatterbox(self.session(root))
                client.assert_not_called()

    def test_host_status_consumes_runtime_record_without_paths_or_capacity_work(self):
        from components.Chatterbox.runtime import runtime as runtime_module

        runtime_status = {
            "ok": True,
            "supported": True,
            "status": "ready",
            "artifact_status": "ready",
            "capacity_status": "not_required",
            "error": None,
            "interpreter": Path("/approved/env/bin/python"),
            "environment": {},
            "manifest_path": Path("/approved/runtime-manifest.json"),
            "verified_model_root": Path("/approved/model"),
            "runtime_root": Path("/approved"),
            "manifest_root": Path("/approved"),
            "model_revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        }
        with (
            patch.object(chatterbox_module, "chatterbox_host_status", ORIGINAL_HOST_STATUS),
            patch.object(chatterbox_module, "chatterbox_target_status", return_value={"supported": True, "status": "supported", "error": None}),
            patch.object(runtime_module, "host_runtime_status", return_value=runtime_status),
            patch.object(runtime_module, "calculate_storage_plan", side_effect=AssertionError("adapter must not inspect capacity")),
        ):
            status = ORIGINAL_HOST_STATUS({"device": "cpu"})

        self.assertTrue(status["ok"])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["artifact_status"], "ready")
        self.assertEqual(status["capacity_status"], "not_required")
        self.assertNotIn("paths", status)

    def test_host_status_preserves_missing_provisioning_repair_and_storage_states(self):
        from components.Chatterbox.runtime import runtime as runtime_module

        cases = (
            ("missing", "sufficient", "runtime is missing"),
            ("provisioning", "storage_budget_unknown", "storage_budget_unknown"),
            ("repair_required", "not_checked", "model requires repair"),
            ("missing", "insufficient_storage", "insufficient storage"),
        )
        for state, capacity_status, detail in cases:
            runtime_status = {
                "ok": False,
                "supported": True,
                "status": state,
                "artifact_status": "missing" if state != "repair_required" else "repair_required",
                "capacity_status": capacity_status,
                "error": f"Chatterbox {detail}.",
                "runtime": {"status": "missing"},
                "model": {"status": "missing"},
                "activation": {"status": "missing"},
            }
            with (
                self.subTest(state=state, capacity_status=capacity_status),
                patch.object(chatterbox_module, "chatterbox_host_status", ORIGINAL_HOST_STATUS),
                patch.object(chatterbox_module, "chatterbox_target_status", return_value={"supported": True, "status": "supported", "error": None}),
                patch.object(runtime_module, "host_runtime_status", return_value=runtime_status),
            ):
                status = ORIGINAL_HOST_STATUS({"device": "cpu"})
            self.assertEqual(status["status"], state)
            self.assertEqual(status["capacity_status"], capacity_status)
            self.assertIn(detail, status["error"])

    def test_unexpected_runtime_defect_is_not_mapped_to_repair_required(self):
        from components.Chatterbox.runtime import runtime as runtime_module

        with (
            patch.object(chatterbox_module, "chatterbox_target_status", return_value={"supported": True, "status": "supported", "error": None}),
            patch.object(runtime_module, "host_runtime_status", side_effect=AssertionError("programming defect")),
            self.assertRaisesRegex(AssertionError, "programming defect"),
        ):
            ORIGINAL_HOST_STATUS({"device": "cpu"})

    def test_direct_manager_rejects_all_nonready_states_before_worker(self):
        from lib.classes.tts_manager import TTSManager

        with tempfile.TemporaryDirectory() as directory:
            for state, error in (
                ("unsupported", "supports Linux x86_64/amd64 CPU only"),
                ("missing", "runtime is missing"),
                ("provisioning", "storage_budget_unknown"),
                ("repair_required", "model requires repair"),
            ):
                status = {
                    "ok": False,
                    "supported": state != "unsupported",
                    "status": state,
                    "error": f"Chatterbox {error}.",
                }
                with (
                    self.subTest(state=state),
                    patch.object(TTSRegistry, "ENGINES", chatterbox_module._test_registry),
                    patch.object(chatterbox_module, "chatterbox_host_status", return_value=status),
                    patch.object(chatterbox_module, "ChatterboxClient") as client,
                    self.assertRaisesRegex(ValueError, error),
                ):
                    TTSManager({**self.session(Path(directory)), "tts_engine": "chatterbox"})
                client.assert_not_called()

    def test_runtime_details_ignore_python_override_and_use_receipt_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(chatterbox_module, "_runtime_details", ORIGINAL_RUNTIME_DETAILS),
                patch.object(
                    chatterbox_module,
                    "chatterbox_host_status",
                    return_value={
                        "ok": True,
                        "status": "ready",
                        "interpreter": root / "approved" / "env" / "bin" / "python",
                        "environment": {"E2A_ROOT": str(root)},
                        "manifest_path": root / "runtime-manifest.json",
                        "verified_model_root": root / "approved" / "model",
                        "manifest_root": root / "approved",
                        "model_revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
                    },
                ),
                patch.dict("os.environ", {"E2A_CHATTERBOX_PYTHON": "/tmp/unapproved-python"}),
            ):
                details = chatterbox_module._runtime_details()
            self.assertEqual(details.interpreter, root / "approved" / "env" / "bin" / "python")
            self.assertNotEqual(str(details.interpreter), "/tmp/unapproved-python")

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

    def test_v3_selection_is_forwarded_to_client_and_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = chatterbox_module.Chatterbox(self.session(root, model="v3"))
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            with (
                patch.object(chatterbox_module, "ChatterboxClient", FakeClient),
                patch.object(chatterbox_module, "_audio_file_is_valid", return_value=(True, None)),
            ):
                success, error = engine.convert(str(output), "Hello from V3.")
            self.assertTrue(success)
            self.assertIsNone(error)
            client = FakeClient.instances[0]
            self.assertEqual(engine.model_variant, "v3")
            self.assertEqual(client.kwargs["model"]["t3_model"], "v3")
            self.assertEqual(client.requests[0]["model"]["t3_model"], "v3")

    def test_turbo_and_nano_are_english_only_and_not_multilingual_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in ("turbo", "nano"):
                with self.subTest(profile=profile):
                    engine = chatterbox_module.Chatterbox(
                        self.session(root, language="eng", model=profile)
                    )
                    request = engine.build_request(
                        root / "process" / f"{profile}.flac",
                        "That was unexpected [chuckle], but it worked.",
                    )
                    self.assertEqual(engine.model_profile, profile)
                    self.assertEqual(engine.loader_kind, "turbo")
                    self.assertEqual(request["language"], "en")
                    self.assertEqual(request["model"]["profile"], profile)
                    self.assertEqual(request["model"]["loader_kind"], "turbo")
                    self.assertEqual(request["model"]["family"], "chatterbox-turbo")
                    self.assertEqual(
                        request["model"]["fingerprint"],
                        "chatterbox-v2-aaaaaaaaaaaaaaaa",
                    )
                    self.assertNotIn("t3_model", request["model"])
                    self.assertEqual(
                        request["segments"][0]["text"],
                        "That was unexpected [chuckle], but it worked.",
                    )
                    engine.close()

                    with self.assertRaisesRegex(ValueError, "does not support language"):
                        chatterbox_module.Chatterbox(
                            self.session(root, language="swe", model=profile)
                        )

    def test_turbo_and_nano_voice_prompt_duration_is_checked_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "voices" / "prompt.wav"
            for profile in ("turbo", "nano"):
                with self.subTest(profile=profile):
                    engine = chatterbox_module.Chatterbox(
                        self.session(root, language="eng", model=profile)
                    )
                    self.make_voice(prompt, seconds=5.0)
                    selected, error = engine._set_voice(str(prompt))
                    self.assertIsNone(selected)
                    self.assertIn("longer than 5 seconds", error)

                    self.make_voice(prompt, seconds=5.01)
                    selected, error = engine._set_voice(str(prompt))
                    self.assertEqual(selected, str(prompt.resolve()))
                    self.assertIsNone(error)
                    engine.close()

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
                model_profile="v2",
                model_fingerprint="chatterbox-v2-aaaaaaaaaaaaaaaa",
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

    def test_client_ignores_arbitrary_worker_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside_worker = root / "outside-worker.py"
            outside_worker.write_text("raise SystemExit('unapproved worker executed')\n", encoding="utf-8")
            engine = chatterbox_module.Chatterbox(self.session(root))
            output = root / "process" / "chapters" / "sentences" / "0.flac"
            approved_worker = (
                Path(chatterbox_module.__file__).resolve().parents[3]
                / "components"
                / "Chatterbox"
                / "worker.py"
            )
            with (
                patch.dict("os.environ", {"E2A_CHATTERBOX_WORKER": str(outside_worker)}),
                patch.object(chatterbox_module, "ChatterboxClient", FakeClient),
                patch.object(chatterbox_module, "_audio_file_is_valid", return_value=(True, None)),
            ):
                success, error = engine.convert(str(output), "Hello world.")
            self.assertTrue(success)
            self.assertIsNone(error)
            self.assertEqual(FakeClient.instances[0].kwargs["worker_path"], approved_worker)
            self.assertNotEqual(FakeClient.instances[0].kwargs["worker_path"], outside_worker)

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
