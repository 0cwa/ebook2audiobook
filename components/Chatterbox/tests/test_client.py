"""Dependency-light client tests using the local fake worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
import time
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
CLIENT_PATH = ROOT / "lib" / "classes" / "tts_engines" / "chatterbox_client.py"
SPEC = importlib.util.spec_from_file_location("chatterbox_client_under_test", CLIENT_PATH)
assert SPEC is not None and SPEC.loader is not None
client_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client_module)

FAKE_WORKER = ROOT / "components" / "Chatterbox" / "tests" / "fake_worker.py"


class ClientTests(unittest.TestCase):
    def make_client(self, root: Path, mode: str = "success", **kwargs):
        return client_module.ChatterboxClient(
            interpreter=__import__("sys").executable,
            worker_path=FAKE_WORKER,
            approved_roots={"voice": [root], "output": [root]},
            worker_args=("--mode", mode),
            readiness_timeout=kwargs.pop("readiness_timeout", 1.0),
            request_timeout=kwargs.pop("request_timeout", 1.0),
            cancel_grace=kwargs.pop("cancel_grace", 0.1),
            **kwargs,
        )

    @staticmethod
    def request(root: Path, name: str = "sentence.flac"):
        return {
            "model": {"family": "chatterbox-multilingual", "revision": "test", "t3_model": "v2"},
            "language": "sv",
            "segments": [{"kind": "text", "text": "Hej"}],
            "output": {"path": str(root / name), "sample_rate": 24000, "channels": 1},
        }

    def test_ready_ping_and_success_framing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(root)
            try:
                capabilities = client.ping()
                self.assertEqual(capabilities["sample_rate"], 24000)
                result = client.synthesize(self.request(root))
                self.assertEqual(result["channels"], 1)
                self.assertTrue((root / "sentence.flac").is_file())
            finally:
                client.close()
            self.assertFalse(client.is_running)

    def test_malformed_response_is_protocol_error_and_cleans_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(root, "malformed")
            try:
                with self.assertRaises(client_module.ChatterboxClientError) as context:
                    client.synthesize(self.request(root))
                self.assertEqual(context.exception.code, "protocol_error")
                self.assertFalse(client.is_running)
            finally:
                client.close()

    def test_timeout_terminates_process_and_removes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(root, "slow", request_timeout=0.15)
            try:
                with self.assertRaises(client_module.ChatterboxClientError) as context:
                    client.synthesize(self.request(root))
                self.assertEqual(context.exception.code, "timeout")
                self.assertFalse(client.is_running)
                self.assertFalse((root / "sentence.flac").exists())
            finally:
                client.close()

    def test_timeout_removes_randomized_worker_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(root, "partial-slow", request_timeout=0.15)
            try:
                with self.assertRaises(client_module.ChatterboxClientError) as context:
                    client.synthesize(self.request(root))
                self.assertEqual(context.exception.code, "timeout")
                self.assertEqual(list(root.glob(".sentence.flac.*.part*")), [])
            finally:
                client.close()

    def test_cancellation_terminates_process_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(root, "cancel-wait", request_timeout=5.0, cancel_grace=0.1)
            cancel = threading.Event()
            result = []

            def run():
                try:
                    client.synthesize(self.request(root), cancel_event=cancel)
                except Exception as exc:
                    result.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            time.sleep(0.15)
            cancel.set()
            thread.join(timeout=3.0)
            try:
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].code, "cancelled")
                self.assertFalse(client.is_running)
                self.assertFalse((root / "sentence.flac").exists())
            finally:
                client.close()

    def test_import_does_not_load_chatterbox(self):
        self.assertNotIn("chatterbox", __import__("sys").modules)

    def test_manifest_and_model_root_are_passed_as_separate_argv_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runtime-manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            model_root = root / "model-cache"
            client = self.make_client(
                root,
                model_manifest_path=manifest,
                approved_model_root=model_root,
                approved_manifest_root=root,
            )
            command = client._command()
            self.assertIn("--model-manifest", command)
            self.assertEqual(command[command.index("--model-manifest") + 1], str(manifest.resolve()))
            self.assertEqual(command[command.index("--approved-model-root") + 1], str(model_root.resolve()))
            self.assertEqual(command[command.index("--approved-manifest-root") + 1], str(root.resolve()))
            self.assertNotIn(" ", command[command.index("--model-manifest") + 1])

    def test_manifest_configuration_requires_all_approved_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runtime-manifest.json"
            with self.assertRaisesRegex(ValueError, "approved manifest root"):
                self.make_client(
                    root,
                    model_manifest_path=manifest,
                    approved_model_root=root / "model-cache",
                )

    def test_explicit_worker_environment_does_not_inherit_host_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = self.make_client(
                root,
                extra_env={"PATH": "/runtime/bin", "HF_HOME": "/runtime/cache"},
            )
            with patch.dict(
                "os.environ",
                {"HF_TOKEN": "must-not-cross", "PYTHONPATH": "/host/site"},
                clear=False,
            ):
                environment = client._environment()
            self.assertEqual(environment["PATH"], "/runtime/bin")
            self.assertEqual(environment["HF_HOME"], "/runtime/cache")
            self.assertNotIn("HF_TOKEN", environment)
            self.assertNotIn("PYTHONPATH", environment)


if __name__ == "__main__":
    unittest.main()
