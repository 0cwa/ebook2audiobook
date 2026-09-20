"""Contract tests for the complete host-side Chatterbox language surface."""

from __future__ import annotations

import importlib.util
import importlib.machinery
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from components.Chatterbox.tests._adapter_loader import load_chatterbox_adapter
from lib.conf_chatterbox_languages import CHATTERBOX_LANGUAGES, chatterbox_language_id
from lib.conf_lang import language_mapping
from lib.conf_models import TTS_ENGINES, default_engine_settings

ROOT = Path(__file__).resolve().parents[3]
CLIENT_PATH = ROOT / "lib" / "classes" / "tts_engines" / "chatterbox_client.py"
CLIENT_SPEC = importlib.util.spec_from_file_location("chatterbox_client_language_tests", CLIENT_PATH)
assert CLIENT_SPEC is not None and CLIENT_SPEC.loader is not None
client_module = importlib.util.module_from_spec(CLIENT_SPEC)
CLIENT_SPEC.loader.exec_module(client_module)
FAKE_WORKER = ROOT / "components" / "Chatterbox" / "tests" / "fake_worker.py"
WORKER_PATH = ROOT / "components" / "Chatterbox" / "worker.py"
WORKER_SPEC = importlib.util.spec_from_file_location("chatterbox_worker_language_tests", WORKER_PATH)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
worker_module = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker_module)


chatterbox_module = load_chatterbox_adapter()


EXPECTED_LANGUAGES = {
    "eng": "en",
    "ara": "ar",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "spa": "es",
    "fin": "fi",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "zlm": "ms",
    "nld": "nl",
    "nob": "no",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "swe": "sv",
    "swh": "sw",
    "tur": "tr",
    "zho": "zh",
}

UNICODE_TEXT = {
    "ara": "مرحبا بالعالم.",
    "ell": "Γειά σου κόσμε.",
    "heb": "שלום עולם.",
    "hin": "नमस्ते दुनिया।",
    "jpn": "こんにちは、世界。",
    "kor": "안녕하세요, 세계.",
    "rus": "Привет, мир.",
    "zho": "你好，世界。",
}

SML_UNICODE_TEXT = {
    "fra": "Élève déjà vu — ¿Qué tal?",
    "ara": "مَرْحَبًا بالعالم.",
    "heb": "שָׁלוֹם עוֹלָם.",
    "zho": "你好，世界。",
    "jpn": "こんにちは、世界。",
    "hin": "नमस्ते दुनिया।",
    "ell": "Γειά σου κόσμε.",
    "rus": "Привет, мир.",
}


class ChatterboxLanguageSupportTests(unittest.TestCase):
    def setUp(self):
        readiness = patch.object(
            chatterbox_module,
            "chatterbox_host_status",
            return_value={"ok": True, "supported": True, "status": "ready", "error": None},
        )
        readiness.start()
        self.addCleanup(readiness.stop)

    def test_combined_discovery_restores_repository_module_state(self):
        """The optional-engine adapter import must not poison later imports."""

        package_name = "lib.classes.tts_engines"
        utils_name = f"{package_name}.common.utils"
        adapter_name = f"{package_name}.chatterbox"
        previous_utils = sys.modules.get(utils_name)
        previous_adapter = sys.modules.get(adapter_name)

        repository_spec = importlib.machinery.PathFinder.find_spec(
            utils_name,
            [str(ROOT / "lib" / "classes" / "tts_engines" / "common")],
        )
        self.assertIsNotNone(repository_spec)
        self.assertEqual(
            Path(repository_spec.origin).resolve(),
            ROOT / "lib" / "classes" / "tts_engines" / "common" / "utils.py",
        )
        self.assertIs(sys.modules.get(utils_name), previous_utils)

        fresh_spec = importlib.util.spec_from_file_location(
            "chatterbox_fresh_repository_import",
            CLIENT_PATH,
        )
        self.assertIsNotNone(fresh_spec)
        self.assertIsNotNone(fresh_spec.loader)
        fresh_module = importlib.util.module_from_spec(fresh_spec)
        fresh_spec.loader.exec_module(fresh_module)
        self.assertEqual(Path(fresh_module.__file__).resolve(), CLIENT_PATH)
        self.assertIs(sys.modules.get(utils_name), previous_utils)
        self.assertIs(sys.modules.get(adapter_name), previous_adapter)

    def test_mapping_is_exact_and_uses_repository_language_keys(self):
        self.assertEqual(CHATTERBOX_LANGUAGES, EXPECTED_LANGUAGES)
        self.assertEqual(len(CHATTERBOX_LANGUAGES), 23)
        self.assertEqual(set(CHATTERBOX_LANGUAGES.values()), set(worker_module.SUPPORTED_LANGUAGES))
        self.assertEqual(set(CHATTERBOX_LANGUAGES.values()), {
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh",
        })
        self.assertTrue(set(CHATTERBOX_LANGUAGES).issubset(language_mapping))

    def test_noncanonical_aliases_are_rejected(self):
        for alias in ("msa", "swa", "nor"):
            self.assertIsNone(chatterbox_language_id(alias))
        self.assertIsNone(chatterbox_language_id("xx"))
        self.assertIsNone(chatterbox_language_id(None))

    def test_configuration_declares_every_language(self):
        chatterbox = TTS_ENGINES["CHATTERBOX"]
        configured = default_engine_settings[chatterbox]["languages"]
        self.assertIs(configured, CHATTERBOX_LANGUAGES)
        for language in EXPECTED_LANGUAGES:
            compatible = [
                engine
                for engine, settings in default_engine_settings.items()
                if language in settings.get("languages", {})
            ]
            self.assertIn(chatterbox, compatible, language)

    def test_runtime_capability_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "do not match the approved set"):
            worker_module._runtime_language_ids({"en": "English"})

    def test_adapter_request_preserves_all_short_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = chatterbox_module._RuntimeDetails(
                interpreter=root / "env" / "bin" / "python",
                environment={},
                manifest_path=root / "runtime-manifest.json",
                model_root=root / "models",
                manifest_root=root,
                model_revision="test",
            )
            with patch.object(chatterbox_module, "_runtime_details", return_value=details):
                for language, language_id in EXPECTED_LANGUAGES.items():
                    engine = chatterbox_module.Chatterbox(
                        self._session(root, language)
                    )
                    request = engine.build_request(
                        root / f"{language}.flac",
                        "Target language text.",
                    )
                    self.assertEqual(request["language"], language_id, language)

    def test_unicode_text_survives_adapter_request_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = chatterbox_module._RuntimeDetails(
                interpreter=root / "env" / "bin" / "python",
                environment={},
                manifest_path=root / "runtime-manifest.json",
                model_root=root / "models",
                manifest_root=root,
                model_revision="test",
            )
            with patch.object(chatterbox_module, "_runtime_details", return_value=details):
                for language, text in UNICODE_TEXT.items():
                    engine = chatterbox_module.Chatterbox(self._session(root, language))
                    request = engine.build_request(root / f"{language}.flac", text)
                    self.assertEqual(request["language"], EXPECTED_LANGUAGES[language])
                    self.assertEqual(request["segments"][0]["text"], text)

    def test_unicode_text_survives_sml_request_construction_without_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = chatterbox_module._RuntimeDetails(
                interpreter=root / "env" / "bin" / "python",
                environment={},
                manifest_path=root / "runtime-manifest.json",
                model_root=root / "models",
                manifest_root=root,
                model_revision="test",
            )
            with patch.object(chatterbox_module, "_runtime_details", return_value=details):
                for language, text in SML_UNICODE_TEXT.items():
                    engine = chatterbox_module.Chatterbox(self._session(root, language))
                    request = engine.build_request(
                        root / f"{language}-sml.flac",
                        f"{text} [break] {text}",
                    )
                    self.assertEqual(request["language"], EXPECTED_LANGUAGES[language])
                    self.assertEqual(
                        [segment["kind"] for segment in request["segments"]],
                        ["text", "silence", "text"],
                    )
                    self.assertEqual(
                        [segment["text"] for segment in request["segments"] if segment["kind"] == "text"],
                        [text, text],
                    )
                    self.assertEqual(request["segments"][1]["seconds"], 0.4)
                    self.assertNotIn("[break]", " ".join(
                        segment.get("text", "") for segment in request["segments"]
                    ))

    def test_fake_worker_readiness_and_forwarding_cover_all_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = client_module.ChatterboxClient(
                interpreter=__import__("sys").executable,
                worker_path=FAKE_WORKER,
                approved_roots={"voice": [root], "output": [root]},
                readiness_timeout=1.0,
                request_timeout=1.0,
            )
            try:
                client.ping()
                self.assertEqual(
                    set(client.capabilities["languages"]),
                    set(EXPECTED_LANGUAGES.values()),
                )
                for language_id in EXPECTED_LANGUAGES.values():
                    result = client.synthesize({
                        "language": language_id,
                        "segments": [{"kind": "text", "text": "test"}],
                        "output": {
                            "path": str(root / f"{language_id}.flac"),
                            "sample_rate": 24000,
                            "channels": 1,
                        },
                    })
                    self.assertEqual(result["language"], language_id)
            finally:
                client.close()

    @staticmethod
    def _session(root: Path, language: str) -> dict[str, object]:
        process_dir = root / "process"
        voice_dir = root / "voices"
        (process_dir / "chapters" / "sentences").mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)
        return {
            "device": "cpu",
            "language": language,
            "language_iso1": "en",
            "translate_enabled": False,
            "translate": None,
            "fine_tuned": "internal",
            "model_cache": "chatterbox-internal",
            "voice": None,
            "voice_dir": str(voice_dir),
            "process_dir": str(process_dir),
            "sentences_dir": str(process_dir / "chapters" / "sentences"),
            "custom_model_dir": None,
            "cancellation_requested": False,
        }


if __name__ == "__main__":
    unittest.main()
