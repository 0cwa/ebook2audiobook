"""Dependency-light worker validation tests."""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = ROOT / "components" / "Chatterbox" / "worker.py"
SPEC = importlib.util.spec_from_file_location("chatterbox_worker_under_test", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerValidationTests(unittest.TestCase):
    @staticmethod
    def write_manifest(path: Path, *, revision: str, file_path: str, digest: str, size: int) -> None:
        path.write_text(json.dumps({
            "sources": {
                "model": {
                    "locator": "https://huggingface.co/ResembleAI/chatterbox",
                    "variant": "multilingual-v2",
                    "revision": revision,
                    "files": [{"path": file_path, "size_bytes": size, "sha256": digest}],
                },
            },
        }), encoding="utf-8")

    def test_valid_request_is_normalized_without_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice = root / "voice.wav"
            voice.write_bytes(b"voice")
            request = {
                "protocol": 1,
                "id": "request-1",
                "op": "synthesize",
                "model": {"family": "chatterbox-multilingual", "revision": "test", "t3_model": "v2"},
                "device": "cpu",
                "language": "sv",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{"kind": "text", "text": "Hej", "voice_prompt": {"path": str(voice)}}],
                "output": {"path": str(root / "sentence.flac.part"), "sample_rate": 24000, "channels": 1},
            }
            normalized = worker.validate_request(request)
            self.assertEqual(normalized["output"], root / "sentence.flac")
            self.assertEqual(normalized["segments"][0]["voice_prompt"]["path"], str(voice))
        self.assertNotIn("chatterbox", __import__("sys").modules)

    def test_paths_outside_approved_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "protocol": 1,
                "id": "request-2",
                "op": "synthesize",
                "model": {"family": "chatterbox-multilingual", "revision": "test", "t3_model": "v2"},
                "device": "cpu",
                "language": "en",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{"kind": "text", "text": "Hello"}],
                "output": {"path": "/tmp/not-approved.flac", "sample_rate": 24000, "channels": 1},
            }
            with self.assertRaises(worker.WorkerRequestError) as context:
                worker.validate_request(request)
            self.assertEqual(context.exception.code, "invalid_request")

    def test_unsupported_device_and_language_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "protocol": 1,
                "id": "request-3",
                "op": "synthesize",
                "model": {"family": "chatterbox-multilingual", "revision": "test", "t3_model": "v2"},
                "device": "cuda",
                "language": "xx",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{"kind": "text", "text": "Hello"}],
                "output": {"path": str(root / "sentence.flac"), "sample_rate": 24000, "channels": 1},
            }
            with self.assertRaises(worker.WorkerRequestError) as context:
                worker.validate_request(base)
            self.assertEqual(context.exception.code, "unsupported_device")
            base["device"] = "cpu"
            with self.assertRaises(worker.WorkerRequestError) as context:
                worker.validate_request(base)
            self.assertEqual(context.exception.code, "unsupported_language")

    def test_manifest_requires_immutable_revision_and_safe_file_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runtime-manifest.json"
            self.write_manifest(
                manifest,
                revision="main",
                file_path="weights.bin",
                digest="0" * 64,
                size=1,
            )
            with self.assertRaisesRegex(RuntimeError, "40-character commit"):
                worker._read_model_manifest(manifest, (root,))

            self.write_manifest(
                manifest,
                revision="a" * 40,
                file_path="../outside.bin",
                digest="0" * 64,
                size=1,
            )
            with self.assertRaisesRegex(RuntimeError, "safe relative paths"):
                worker._read_model_manifest(manifest, (root,))

    def test_default_loader_downloads_pinned_snapshot_and_uses_from_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "model-cache"
            revision = "a" * 40
            snapshot = cache / "models--ResembleAI--chatterbox" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            model_file = snapshot / "weights.bin"
            model_file.write_bytes(b"verified model")
            digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
            manifest = root / "runtime-manifest.json"
            self.write_manifest(
                manifest,
                revision=revision,
                file_path="weights.bin",
                digest=digest,
                size=model_file.stat().st_size,
            )

            download_calls = []
            from_local_calls = []

            def snapshot_download(**kwargs):
                download_calls.append(kwargs)
                return str(snapshot)

            class FakeModel:
                sr = 24000

            class FakeMultilingual:
                @classmethod
                def from_pretrained(cls, **_kwargs):
                    raise AssertionError("default worker path must not call from_pretrained")

                @classmethod
                def from_local(cls, checkpoint_dir, **kwargs):
                    from_local_calls.append((checkpoint_dir, kwargs))
                    return FakeModel()

            chatterbox_package = types.ModuleType("chatterbox")
            chatterbox_package.__path__ = []
            chatterbox_module = types.ModuleType("chatterbox.mtl_tts")
            chatterbox_module.ChatterboxMultilingualTTS = FakeMultilingual
            hub_module = types.ModuleType("huggingface_hub")
            hub_module.snapshot_download = snapshot_download

            with patch.dict(sys.modules, {
                "chatterbox": chatterbox_package,
                "chatterbox.mtl_tts": chatterbox_module,
                "huggingface_hub": hub_module,
            }):
                instance = worker.ChatterboxWorker(
                    model_manifest_path=manifest,
                    approved_model_root=cache,
                    approved_manifest_root=root,
                )
                instance.load_model()

            self.assertEqual(instance.model_revision, revision)
            self.assertEqual(download_calls[0]["repo_id"], "ResembleAI/chatterbox")
            self.assertEqual(download_calls[0]["revision"], revision)
            self.assertEqual(download_calls[0]["allow_patterns"], ["weights.bin"])
            self.assertEqual(download_calls[0]["cache_dir"], str(cache))
            self.assertEqual(from_local_calls, [(str(snapshot.resolve()), {"device": "cpu"})])

    def test_default_loader_rejects_snapshot_checksum_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "model-cache"
            revision = "b" * 40
            snapshot = cache / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "weights.bin").write_bytes(b"wrong")
            manifest = root / "runtime-manifest.json"
            self.write_manifest(
                manifest,
                revision=revision,
                file_path="weights.bin",
                digest="0" * 64,
                size=5,
            )
            hub_module = types.ModuleType("huggingface_hub")
            hub_module.snapshot_download = lambda **_kwargs: str(snapshot)
            with patch.dict(sys.modules, {"huggingface_hub": hub_module}):
                instance = worker.ChatterboxWorker(
                    model_manifest_path=manifest,
                    approved_model_root=cache,
                    approved_manifest_root=root,
                )
                with self.assertRaisesRegex(RuntimeError, "checksum does not match"):
                    instance.load_model()
            self.assertIsNone(instance.model)

    def test_injected_model_loader_remains_manifest_free(self):
        instance = worker.ChatterboxWorker(model_loader=lambda: types.SimpleNamespace(sr=24000))
        instance.load_model()
        self.assertEqual(instance.sample_rate, 24000)

    def test_missing_manifest_is_reported_as_structured_model_load_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                result = worker.main(["--approved-root", directory])
            self.assertEqual(result, 1)
            message = json.loads(output.getvalue())
            self.assertEqual(message["event"], "error")
            self.assertEqual(message["error"]["code"], "model_load_failed")


if __name__ == "__main__":
    unittest.main()
