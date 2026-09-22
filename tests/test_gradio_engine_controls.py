"""Focused tests for the shared TTS engine state transition helper."""

import ast
import inspect
import textwrap

import unittest
from unittest.mock import patch

import lib.gradio as gradio


class _GradioUpdateStub:
    @staticmethod
    def update(**kwargs):
        return kwargs


class _SessionContextStub:
    def __init__(self, session):
        self.session = session

    def get_session(self, session_id):
        return self.session


def _extract_build_interface_function(name):
    """Load one nested callback from build_interface without building the UI."""

    build_source = textwrap.dedent(inspect.getsource(gradio.build_interface))
    build_node = ast.parse(build_source).body[0]
    callback_node = next(
        node
        for node in ast.walk(build_node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    namespace = {
        "gr": _GradioUpdateStub,
        "_set_session_tts_engine": gradio._set_session_tts_engine,
        "exception_alert": lambda *_args, **_kwargs: None,
    }
    exec(compile(ast.Module([callback_node], type_ignores=[]), "<callback>", "exec"), namespace)
    return namespace[name]


def _has_translation_engine_callback_chain():
    """Check the source-level Gradio event chain without constructing the UI."""

    build_source = textwrap.dedent(inspect.getsource(gradio.build_interface))
    build_node = ast.parse(build_source).body[0]
    for node in ast.walk(build_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "then" or not isinstance(node.func.value, ast.Call):
            continue
        change_call = node.func.value
        if not (
            isinstance(change_call.func, ast.Attribute)
            and change_call.func.attr == "change"
            and isinstance(change_call.func.value, ast.Name)
            and change_call.func.value.id == "gr_translate"
        ):
            continue
        change_fn = next((kw.value.id for kw in change_call.keywords if kw.arg == "fn" and isinstance(kw.value, ast.Name)), None)
        then_fn = next((kw.value.id for kw in node.keywords if kw.arg == "fn" and isinstance(kw.value, ast.Name)), None)
        if change_fn == "_change_gr_translate" and then_fn == "_change_gr_tts_engine_list":
            return True
    return False


class GradioEngineControlTests(unittest.TestCase):
    def setUp(self):
        self.engine_settings = {
            "ENGINE_A": {"voice": "default-a"},
            "ENGINE_B": {"voice": "default-b"},
        }

    def test_preset_fallback_uses_first_available_model_when_internal_is_absent(self):
        with patch.object(gradio, "default_fine_tuned", "internal"):
            self.assertEqual(
                gradio._select_fine_tuned_preset("internal", ["v2", "v3"]),
                "v2",
            )
            self.assertEqual(
                gradio._select_fine_tuned_preset("v3", ["v2", "v3"]),
                "v3",
            )
            self.assertEqual(
                gradio._select_fine_tuned_preset("custom", ["internal", "v3"]),
                "internal",
            )

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

    def test_translation_change_wires_engine_callback_and_forces_reset(self):
        self.assertTrue(_has_translation_engine_callback_chain())
        session = {
            "id": "session-1",
            "tts_engine": "ENGINE_A",
            "voice": "default-a",
            "fine_tuned": "custom",
            "translate_enabled": True,
        }
        refresh_calls = []

        callback = _extract_build_interface_function("_change_gr_tts_engine_list")
        callback_globals = callback.__globals__
        callback_globals["context"] = _SessionContextStub(session)
        callback_globals["_refresh_gr_tts_engine_controls"] = lambda session_id: refresh_calls.append(session_id) or ("refreshed",) * 7
        with (
            patch.object(gradio, "default_engine_settings", self.engine_settings),
            patch.object(gradio, "default_fine_tuned", "internal"),
        ):
            result = callback("session-1", "ENGINE_A")

        self.assertEqual(result, ("refreshed",) * 7)
        self.assertEqual(refresh_calls, ["session-1"])
        self.assertIsNone(session["voice"])
        self.assertEqual(session["fine_tuned"], "internal")


if __name__ == "__main__":
    unittest.main()
