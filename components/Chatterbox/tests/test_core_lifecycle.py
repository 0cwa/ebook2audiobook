"""Dependency-light tests for the host TTS lifecycle cleanup hooks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lib.core as core


class _SessionContext:
    def __init__(self, session: dict):
        self.session = session

    def get_session(self, session_id: str) -> dict:
        return self.session if session_id == self.session["id"] else {}


class _FakeManager:
    def __init__(self, engine):
        self.engine = engine

    def convert_sentence2audio(self, sentence_file: str, sentence: str, **kwargs):
        return self.engine.convert_sentence2audio(sentence_file, sentence, **kwargs)


class _FakeEngine:
    def __init__(self, result=(True, None)):
        self.result = result
        self.close_calls = 0
        self.convert_calls = []
        self.tts_key = "fake-tts"
        self.tts_zs_key = None

    def convert_sentence2audio(self, sentence_file: str, sentence: str, **kwargs):
        self.convert_calls.append((sentence_file, sentence, kwargs))
        return self.result

    def close(self):
        self.close_calls += 1


def _conversion_session(root: Path) -> dict:
    return {
        "id": "lifecycle-session",
        "cancellation_requested": False,
        "language": "eng",
        "translate_enabled": False,
        "translate": None,
        "voice": None,
        "filename_noext": "test-book",
        "ebook": str(root / "test-book.epub"),
        "chapters_dir": str(root / "chapters"),
        "sentences_dir": str(root / "sentences"),
        "is_gui_process": False,
        "blocks_current": {
            "blocks": [{
                "id": "block-1",
                "keep": True,
                "text": "Hello",
                "sentences": ["Hello"],
                "voice": None,
            }],
            "block_resume": 0,
            "sentence_resume": 0,
        },
        "blocks_saved": {"blocks": []},
    }


class CoreLifecycleTests(unittest.TestCase):
    def test_natural_sort_key_uses_numeric_basename_runs(self):
        paths = [
            "/cache/hash-z/volume-10.epub",
            "/cache/hash-a/volume-2.epub",
            "/cache/hash-b/volume-1.epub",
        ]

        self.assertEqual(
            sorted(paths, key=core.natural_sort_key),
            [paths[2], paths[1], paths[0]],
        )

    def test_token_spacing_preserves_contractions_and_sml_boundaries(self):
        self.assertEqual(core.foreign2latin("can't", "eng"), "can't")
        self.assertEqual(
            core.foreign2latin("hello [pause] world", "eng"),
            "hello [pause] world",
        )

    def test_normalize_text_assigns_emoji_removal_result(self):
        self.assertEqual(
            core.normalize_text("Hello 😀 world", "eng", "en", "piper"),
            "Hello world",
        )

    def test_chatterbox_compatibility_is_target_and_device_aware(self):
        chatterbox = core.TTS_ENGINES["CHATTERBOX"]
        with (
            patch("lib.conf_models.platform.system", return_value="Linux"),
            patch("lib.conf_models.platform.machine", return_value="x86_64"),
        ):
            self.assertIn(chatterbox, core.get_compatible_tts_engines("eng", "cpu"))
            self.assertIn(chatterbox, core.get_compatible_tts_engines("eng"))
            self.assertNotIn(chatterbox, core.get_compatible_tts_engines("eng", "cuda"))
        with patch("lib.conf_models.platform.system", return_value="Darwin"):
            self.assertNotIn(chatterbox, core.get_compatible_tts_engines("eng", "cpu"))
        with (
            patch("lib.conf_models.platform.system", return_value="Linux"),
            patch("lib.conf_models.platform.machine", return_value="aarch64"),
        ):
            self.assertNotIn(chatterbox, core.get_compatible_tts_engines("eng", "cpu"))

    def test_unload_tts_manager_closes_engine_and_evicts_its_cache(self):
        engine = _FakeEngine()
        manager = SimpleNamespace(engine=engine)
        loaded_tts = {engine.tts_key: engine, "unrelated": object()}
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

        with (
            patch.object(core, "loaded_tts", loaded_tts),
            patch.object(core.gc, "collect"),
            patch.object(core.sys, "platform", "darwin"),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            core.unload_tts_manager(manager)

        self.assertEqual(engine.close_calls, 1)
        self.assertIsNone(manager.engine)
        self.assertNotIn(engine.tts_key, loaded_tts)
        self.assertIn("unrelated", loaded_tts)

    def test_unload_tts_manager_is_safe_without_engine_close(self):
        manager = SimpleNamespace(engine=SimpleNamespace(tts_key=None, tts_zs_key=None))
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

        with (
            patch.object(core, "loaded_tts", {}),
            patch.object(core.gc, "collect"),
            patch.object(core.sys, "platform", "darwin"),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            core.unload_tts_manager(manager)

        self.assertIsNone(manager.engine)

    def test_successful_conversion_closes_engine(self):
        self._assert_conversion_cleanup((True, None), expected_result=True)

    def test_failed_conversion_closes_engine_through_unload(self):
        self._assert_conversion_cleanup((False, "conversion failed"), expected_result=False)

    def _assert_conversion_cleanup(self, conversion_result, *, expected_result: bool):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _conversion_session(root)
            engine = _FakeEngine(conversion_result)
            manager = _FakeManager(engine)
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

            with (
                patch.object(core, "context", _SessionContext(session)),
                patch.object(core, "TTSManager", return_value=manager),
                patch.object(core, "show_alert"),
                patch.object(core, "save_db_stamp"),
                patch.object(core, "save_json_blocks"),
                patch.object(core, "combine_audio_sentences", return_value=True),
                patch.object(core, "loaded_tts", {engine.tts_key: engine}),
                patch.object(core.gc, "collect"),
                patch.object(core.sys, "platform", "darwin"),
                patch.dict(sys.modules, {"torch": fake_torch}),
            ):
                result = core.convert_chapters2audio(session["id"])

            self.assertIs(result, expected_result)
            self.assertEqual(engine.close_calls, 1)
            if not expected_result:
                self.assertIsNone(manager.engine)


if __name__ == "__main__":
    unittest.main()
