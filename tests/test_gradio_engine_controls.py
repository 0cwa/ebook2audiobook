"""Focused tests for the shared TTS engine state transition helper."""

import unittest
from unittest.mock import patch

import lib.gradio as gradio


class GradioEngineControlTests(unittest.TestCase):
    def setUp(self):
        self.engine_settings = {
            "ENGINE_A": {"voice": "default-a"},
            "ENGINE_B": {"voice": "default-b"},
        }

    def test_translation_same_engine_forces_default_state_reset(self):
        session = {
            "tts_engine": "ENGINE_A",
            "voice": "default-a",
            "fine_tuned": "custom",
        }

        with (
            patch.object(gradio, "default_engine_settings", self.engine_settings),
            patch.object(gradio, "default_fine_tuned", "internal"),
        ):
            changed = gradio._set_session_tts_engine(session, "ENGINE_A", force=True)

        self.assertTrue(changed)
        self.assertEqual(session["tts_engine"], "ENGINE_A")
        self.assertIsNone(session["voice"])
        self.assertEqual(session["fine_tuned"], "internal")

    def test_changed_engine_preserves_custom_voice(self):
        session = {
            "tts_engine": "ENGINE_A",
            "voice": "/voices/custom.wav",
            "fine_tuned": "custom",
        }

        with (
            patch.object(gradio, "default_engine_settings", self.engine_settings),
            patch.object(gradio, "default_fine_tuned", "internal"),
        ):
            changed = gradio._set_session_tts_engine(session, "ENGINE_B")

        self.assertTrue(changed)
        self.assertEqual(session["tts_engine"], "ENGINE_B")
        self.assertEqual(session["voice"], "/voices/custom.wav")
        self.assertEqual(session["fine_tuned"], "internal")


if __name__ == "__main__":
    unittest.main()
