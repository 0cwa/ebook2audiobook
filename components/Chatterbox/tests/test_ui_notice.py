"""Focused tests for the pure TTS rating and notice formatter."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _load_rating_formatter():
    """Load the formatter without importing the application dependency graph."""

    core = types.ModuleType("lib.core")
    core.gr = types.SimpleNamespace(Blocks=object)
    with patch.dict(sys.modules, {"lib.core": core}):
        gradio_module = importlib.import_module("lib.gradio")
    return gradio_module.format_tts_engine_rating


format_tts_engine_rating = _load_rating_formatter()


class TTSRatingNoticeTests(unittest.TestCase):
    def test_chatterbox_output_contains_perth_watermark_notice(self):
        from lib.conf_models import TTS_ENGINES, default_engine_settings

        settings = default_engine_settings[TTS_ENGINES["CHATTERBOX"]]
        output = format_tts_engine_rating(TTS_ENGINES["CHATTERBOX"], {"chatterbox": settings})

        self.assertIn("Perth watermark", output)
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
