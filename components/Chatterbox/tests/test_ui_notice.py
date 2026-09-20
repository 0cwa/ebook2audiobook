"""Focused tests for the pure TTS rating and notice formatter."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _load_gradio_helpers():
    """Load dependency-light Gradio helpers without the application graph."""

    core = types.ModuleType("lib.core")
    core.gr = types.SimpleNamespace(Blocks=object)
    with patch.dict(sys.modules, {"lib.core": core}):
        gradio_module = importlib.import_module("lib.gradio")
    return gradio_module.format_tts_engine_rating, gradio_module._change_device_tts_controls


format_tts_engine_rating, change_device_tts_controls = _load_gradio_helpers()


class TTSRatingNoticeTests(unittest.TestCase):
    def test_restored_chatterbox_device_change_refreshes_all_dependent_outputs(self):
        session = {
            "id": "restored-session",
            "device": "cpu",
            "tts_engine": "chatterbox",
            "fine_tuned": "internal",
            "custom_model": "/stale/chatterbox-model",
        }
        calls = []

        def update_engine_list():
            calls.append("engine")
            session["tts_engine"] = "piper"
            return {"choices": ["piper"], "value": "piper"}

        def refresh_engine_controls():
            calls.append("dependents")
            self.assertEqual(session["device"], "cuda")
            self.assertEqual(session["tts_engine"], "piper")
            session["custom_model"] = None
            return (
                {"value": "piper rating"},
                {"visible": False},
                {"visible": False},
                {"visible": False},
                {"choices": ["internal"], "value": "internal"},
                {"label": "*Upload Custom Model not available for piper"},
                {"choices": [("None", None)], "value": None},
            )

        outputs = change_device_tts_controls(
            session,
            "cuda",
            update_engine_list=update_engine_list,
            refresh_engine_controls=refresh_engine_controls,
            empty_update=dict,
        )

        self.assertEqual(calls, ["engine", "dependents"])
        self.assertEqual(len(outputs), 8)
        self.assertEqual(outputs[0]["value"], "piper")
        self.assertNotIn("Perth watermark", outputs[1]["value"])
        self.assertFalse(outputs[4]["visible"])
        self.assertEqual(outputs[5]["value"], "internal")
        self.assertIn("not available for piper", outputs[6]["label"])
        self.assertIsNone(outputs[7]["value"])
        self.assertIsNone(session["custom_model"])

    def test_chatterbox_output_contains_perth_watermark_notice(self):
        from lib.conf_models import TTS_ENGINES, default_engine_settings

        settings = default_engine_settings[TTS_ENGINES["CHATTERBOX"]]
        output = format_tts_engine_rating(TTS_ENGINES["CHATTERBOX"], {"chatterbox": settings})

        self.assertIn("Perth watermark", output)
        self.assertIn("Multilingual V2", output)
        self.assertIn("23 languages", output)
        self.assertIn("CPU-only", output)
        self.assertIn("isolated Python 3.11", output)
        self.assertIn("Linux x86_64/amd64", output)
        self.assertIn(settings["notice"], output)

    def test_engine_without_notice_has_no_empty_notice_element(self):
        settings = {
            "plain": {
                "rating": {"VRAM": 1, "CPU": 2, "RAM": 3, "Realism": 4},
            }
        }

        output = format_tts_engine_rating("plain", settings)

        self.assertEqual(output.count("<div"), 1)
        self.assertNotIn("margin-top:4px", output)
        self.assertNotIn("None", output)

    def test_rating_content_is_preserved(self):
        settings = {
            "engine": {
                "rating": {"VRAM": 3, "CPU": 2, "RAM": 9, "Realism": 4},
            }
        }

        output = format_tts_engine_rating("engine", settings)

        self.assertIn("<b>VRAM:</b>", output)
        self.assertIn("3 GB", output)
        self.assertIn("<b>CPU:</b>", output)
        self.assertIn("<b>RAM:</b>", output)
        self.assertIn("9 GB", output)
        self.assertIn("<b>Realism:</b>", output)
        self.assertEqual(output.count("★"), 6)


if __name__ == "__main__":
    unittest.main()
