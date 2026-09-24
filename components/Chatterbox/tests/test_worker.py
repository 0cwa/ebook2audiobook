"""Dependency-light worker validation tests."""

from __future__ import annotations

import ast
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import zipfile
import wave
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from components.Chatterbox.runtime import contract_data


ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = ROOT / "components" / "Chatterbox" / "worker.py"
SPEC = importlib.util.spec_from_file_location("chatterbox_worker_under_test", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

CANONICAL_MODEL_FILE_PATHS = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
V3_MODEL_FILE_PATHS = (
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)


class WorkerValidationTests(unittest.TestCase):
    def test_contract_data_is_dependency_light_and_worker_constants_are_shared(self):
        source = Path(contract_data.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported_modules.update(
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertTrue(imported_modules <= {"__future__", "dataclasses", "typing"})

        for name in (
            "ACTIVATION_RECEIPT_SCHEMA",
            "APPROVED_LANGUAGE_IDS",
            "CHANNELS",
            "DEVICE",
            "MAX_SEGMENTS",
            "MAX_SILENCE_SECONDS",
            "MAX_TEXT_CHARS",
            "MAX_TOTAL_TEXT_CHARS",
            "MODEL_FAMILY",
            "MODEL_RECEIPT_SCHEMA",
            "MODEL_VARIANT",
            "PKUSEG_DATA_FILENAME",
            "PKUSEG_DATA_SHA256",
            "PROTOCOL_VERSION",
            "RUNTIME_RECEIPT_SCHEMA",
            "SAMPLE_RATE",
            "SUPPORTED_LANGUAGES",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(worker, name), getattr(contract_data, name))

    @staticmethod
    def write_publication_proof(receipt: Path, label: str) -> None:
        receipt.with_name(f"{receipt.name}.published").write_text(json.dumps({
            "schema": f"ebook2audiobook.chatterbox-{label}-receipt-publication.v1",
            "status": "published",
            "receipt": str(receipt),
            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        }), encoding="utf-8")

    @staticmethod
    def create_nonregular_ambiguity_marker(receipt: Path, marker_kind: str) -> Path:
        ambiguity = receipt.with_name(f"{receipt.name}.ambiguous")
        if marker_kind == "dangling_symlink":
            ambiguity.symlink_to(ambiguity.with_name(f"{ambiguity.name}.missing"))
        elif marker_kind == "special_file":
            os.mkfifo(ambiguity)
        else:
            raise AssertionError(f"unsupported marker kind: {marker_kind}")
        return ambiguity

    def write_ready_activation_chain(self, root: Path):
        snapshot = root / "snapshot"
        snapshot.mkdir()
        runtime_receipt = root / "runtime.json"
        model_receipt = root / "model.json"
        activation_receipt = root / "activation.json"
        runtime_receipt.write_text(json.dumps({
            "schema": worker.RUNTIME_RECEIPT_SCHEMA,
            "status": "ready",
            "runtime_fingerprint": "runtime-id",
        }), encoding="utf-8")
        model_receipt.write_text(json.dumps({
            "schema": worker.MODEL_RECEIPT_SCHEMA,
            "status": "ready",
            "model_fingerprint": "model-id",
            "artifact": {"snapshot_path": str(snapshot)},
        }), encoding="utf-8")
        activation_receipt.write_text(json.dumps({
            "schema": worker.ACTIVATION_RECEIPT_SCHEMA,
            "status": "ready",
            "runtime_fingerprint": "runtime-id",
            "model_fingerprint": "model-id",
            "runtime_receipt_sha256": hashlib.sha256(runtime_receipt.read_bytes()).hexdigest(),
            "model_receipt_sha256": hashlib.sha256(model_receipt.read_bytes()).hexdigest(),
            "model_snapshot": str(snapshot),
            "checks": {"local_model_load": {"ok": True}},
        }), encoding="utf-8")
        receipts = {
            "runtime": runtime_receipt,
            "model": model_receipt,
            "activation": activation_receipt,
        }
        for label, receipt in receipts.items():
            self.write_publication_proof(receipt, label)
        environment = {
            "E2A_CHATTERBOX_RUNTIME_RECEIPT": str(runtime_receipt),
            "E2A_CHATTERBOX_MODEL_RECEIPT": str(model_receipt),
            "E2A_CHATTERBOX_ACTIVATION_RECEIPT": str(activation_receipt),
        }
        return snapshot, receipts, environment

    @staticmethod
    def write_manifest(
        path: Path,
        *,
        revision: str,
        digest: str,
        size: int,
        file_paths: tuple[str, ...] | None = None,
        variant: str = "v2",
    ) -> None:
        if file_paths is None:
            file_paths = V3_MODEL_FILE_PATHS if variant == "v3" else CANONICAL_MODEL_FILE_PATHS
        path.write_text(json.dumps({
            "sources": {
                "model": {
                    "locator": "https://huggingface.co/ResembleAI/chatterbox",
                    "variant": f"multilingual-{variant}",
                    "revision": revision,
                    "files": [
                        {"path": file_path, "size_bytes": size, "sha256": digest}
                        for file_path in file_paths
                    ],
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

    def test_v3_request_matches_only_a_v3_worker_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "protocol": 1,
                "id": "request-v3",
                "op": "synthesize",
                "model": {
                    "profile": "v3",
                    "family": "chatterbox-multilingual",
                    "fingerprint": "chatterbox-v3-aaaaaaaaaaaaaaaa",
                    "revision": "d" * 40,
                    "t3_model": "v3",
                },
                "device": "cpu",
                "language": "en",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{"kind": "text", "text": "Hello from V3"}],
                "output": {"path": str(root / "sentence.flac"), "sample_rate": 24000, "channels": 1},
            }
            normalized = worker.validate_request(
                request,
                expected_model={
                    "fingerprint": "chatterbox-v3-aaaaaaaaaaaaaaaa",
                    "revision": "d" * 40,
                    "variant": "v3",
                },
            )
            self.assertEqual(normalized["model_profile"], "v3")
            self.assertEqual(normalized["loader_kind"], "multilingual")
            self.assertEqual(normalized["fingerprint"], "chatterbox-v3-aaaaaaaaaaaaaaaa")
            with self.assertRaisesRegex(worker.WorkerRequestError, "fingerprint does not match"):
                worker.validate_request(
                    request,
                    expected_model={
                        "fingerprint": "chatterbox-v3-bbbbbbbbbbbbbbbb",
                        "revision": "d" * 40,
                        "variant": "v3",
                    },
                )
            with self.assertRaisesRegex(worker.WorkerRequestError, "profile does not match"):
                worker.validate_request(
                    request,
                    expected_model={
                        "fingerprint": "chatterbox-v3-aaaaaaaaaaaaaaaa",
                        "revision": "d" * 40,
                        "variant": "v2",
                    },
                )

    def test_turbo_request_is_english_only_preserves_native_tags_and_has_no_t3_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "protocol": 1,
                "id": "request-turbo",
                "op": "synthesize",
                "model": {
                    "profile": "turbo",
                    "loader_kind": "turbo",
                    "family": "chatterbox-turbo",
                    "fingerprint": "chatterbox-turbo-aaaaaaaaaaaaaaaa",
                    "revision": "e" * 40,
                },
                "device": "cpu",
                "language": "en",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{
                    "kind": "text",
                    "text": "That was unexpected [chuckle], but it worked.",
                }],
                "output": {
                    "path": str(root / "sentence.flac"),
                    "sample_rate": 24000,
                    "channels": 1,
                },
            }
            normalized = worker.validate_request(
                request,
                expected_model={
                    "fingerprint": "chatterbox-turbo-aaaaaaaaaaaaaaaa",
                    "revision": "e" * 40,
                    "profile": "turbo",
                    "family": "chatterbox-turbo",
                },
                supported_languages=("en",),
                minimum_prompt_seconds=5.0,
            )
            self.assertEqual(normalized["model_profile"], "turbo")
            self.assertEqual(normalized["loader_kind"], "turbo")
            self.assertEqual(
                normalized["segments"][0]["text"],
                "That was unexpected [chuckle], but it worked.",
            )

            wrong_language = json.loads(json.dumps(request))
            wrong_language["language"] = "sv"
            with self.assertRaisesRegex(worker.WorkerRequestError, "language is not supported"):
                worker.validate_request(
                    wrong_language,
                    supported_languages=("en",),
                    minimum_prompt_seconds=5.0,
                )

            encoded_as_t3 = json.loads(json.dumps(request))
            encoded_as_t3["model"]["t3_model"] = "turbo"
            with self.assertRaisesRegex(
                worker.WorkerRequestError,
                "must not be encoded as multilingual t3_model",
            ):
                worker.validate_request(
                    encoded_as_t3,
                    supported_languages=("en",),
                    minimum_prompt_seconds=5.0,
                )

    def test_turbo_prompt_must_be_strictly_longer_than_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_prompt(path: Path, seconds: float) -> None:
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(24000)
                    stream.writeframes(b"\x00\x00" * int(round(24000 * seconds)))

            prompt = root / "prompt.wav"
            request = {
                "protocol": 1,
                "id": "request-prompt",
                "op": "synthesize",
                "model": {
                    "profile": "nano",
                    "loader_kind": "turbo",
                    "family": "chatterbox-turbo",
                    "revision": "f" * 40,
                },
                "device": "cpu",
                "language": "en",
                "approved_roots": {"voice": [str(root)], "output": [str(root)]},
                "segments": [{
                    "kind": "text",
                    "text": "Hello",
                    "voice_prompt": {"path": str(prompt)},
                }],
                "output": {
                    "path": str(root / "sentence.flac"),
                    "sample_rate": 24000,
                    "channels": 1,
                },
            }

            write_prompt(prompt, 5.0)
            with self.assertRaisesRegex(worker.WorkerRequestError, "longer than 5 seconds"):
                worker.validate_request(
                    request,
                    supported_languages=("en",),
                    minimum_prompt_seconds=5.0,
                )

            write_prompt(prompt, 5.01)
            normalized = worker.validate_request(
                request,
                supported_languages=("en",),
                minimum_prompt_seconds=5.0,
            )
            self.assertEqual(
                normalized["segments"][0]["voice_prompt"]["path"],
                str(prompt),
            )

    def test_generation_kwargs_never_pass_language_id_to_turbo_family(self):
        instance = worker.ChatterboxWorker(model_loader=lambda: types.SimpleNamespace(sr=24000))
        instance.loader_kind = "turbo"
        kwargs = instance._generation_kwargs(
            {"language": "en"},
            {"kind": "text", "voice_prompt": {"path": "/approved/prompt.wav"}},
        )
        self.assertEqual(kwargs, {"audio_prompt_path": "/approved/prompt.wav"})

        instance.loader_kind = "multilingual"
        instance._v3_compat_mode = True
        kwargs = instance._generation_kwargs(
            {"language": "sv"},
            {"kind": "text", "voice_prompt": None},
        )
        self.assertEqual(
            kwargs,
            {"language_id": "sv", "repetition_penalty": 1.2},
        )

    def test_turbo_and_nano_native_loader_flags_are_explicit(self):
        calls = []

        class FakeTurbo:
            @classmethod
            def from_local(cls, checkpoint_dir, device, nano=False):
                calls.append((checkpoint_dir, device, nano))
                return types.SimpleNamespace(sr=24000)

        chatterbox_package = types.ModuleType("chatterbox")
        chatterbox_package.__path__ = []
        turbo_module = types.ModuleType("chatterbox.tts_turbo")
        turbo_module.ChatterboxTurboTTS = FakeTurbo
        models_package = types.ModuleType("chatterbox.models")
        models_package.__path__ = []
        t3_package = types.ModuleType("chatterbox.models.t3")
        t3_package.__path__ = []
        configs_module = types.ModuleType("chatterbox.models.t3.llama_configs")
        configs_module.LLAMA_CONFIGS = {"GPT2_small": {}}

        modules = {
            "chatterbox": chatterbox_package,
            "chatterbox.tts_turbo": turbo_module,
            "chatterbox.models": models_package,
            "chatterbox.models.t3": t3_package,
            "chatterbox.models.t3.llama_configs": configs_module,
        }
        with patch.dict(sys.modules, modules):
            turbo_model, turbo_compat = worker._load_local_turbo_model(
                Path("/verified/turbo"),
                "turbo",
            )
            nano_model, nano_compat = worker._load_local_turbo_model(
                Path("/verified/nano"),
                "nano",
            )

        self.assertEqual(turbo_model.sr, 24000)
        self.assertEqual(nano_model.sr, 24000)
        self.assertFalse(turbo_compat)
        self.assertFalse(nano_compat)
        self.assertEqual(
            calls,
            [
                ("/verified/turbo", "cpu", False),
                ("/verified/nano", "cpu", True),
            ],
        )

    def test_nano_falls_back_when_pinned_runtime_has_no_nano_selector(self):
        calls = []

        class OldTurbo:
            @classmethod
            def from_local(cls, checkpoint_dir, device):
                raise AssertionError("old native Turbo loader must not be used for Nano")

        chatterbox_package = types.ModuleType("chatterbox")
        chatterbox_package.__path__ = []
        turbo_module = types.ModuleType("chatterbox.tts_turbo")
        turbo_module.ChatterboxTurboTTS = OldTurbo
        compat_module = types.ModuleType("components.Chatterbox.turbo_compat")

        def load_turbo_compat(snapshot, *, device, nano):
            calls.append((str(snapshot), device, nano))
            return types.SimpleNamespace(sr=24000)

        compat_module.load_turbo_compat = load_turbo_compat
        with patch.dict(
            sys.modules,
            {
                "chatterbox": chatterbox_package,
                "chatterbox.tts_turbo": turbo_module,
                "components.Chatterbox.turbo_compat": compat_module,
            },
        ):
            model, compatibility_mode = worker._load_local_turbo_model(
                Path("/verified/nano"),
                "nano",
            )

        self.assertEqual(model.sr, 24000)
        self.assertTrue(compatibility_mode)
        self.assertEqual(calls, [("/verified/nano", "cpu", True)])

    def test_manifest_file_records_accept_profile_specific_declared_sets(self):
        v3_records = [
            {"path": path, "size_bytes": 1, "sha256": "0" * 64}
            for path in V3_MODEL_FILE_PATHS
        ]
        self.assertEqual(
            tuple(record["path"] for record in worker._manifest_file_records(v3_records)),
            V3_MODEL_FILE_PATHS,
        )
        reduced = v3_records[:3]
        self.assertEqual(len(worker._manifest_file_records(reduced)), 3)

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
                digest="0" * 64,
                size=1,
            )
            with self.assertRaisesRegex(RuntimeError, "40-character commit"):
                worker._read_model_manifest(manifest, (root,))

            self.write_manifest(
                manifest,
                revision="a" * 40,
                digest="0" * 64,
                size=1,
                file_paths=("../outside.bin", *CANONICAL_MODEL_FILE_PATHS[1:]),
            )
            with self.assertRaisesRegex(RuntimeError, "safe relative paths"):
                worker._read_model_manifest(manifest, (root,))

    def test_multi_repository_manifest_resolves_sources_and_uses_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "runtime-manifest.json"
            records = [
                {
                    "path": path,
                    "source": "base" if index < 3 else "pack",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                }
                for index, path in enumerate(CANONICAL_MODEL_FILE_PATHS)
            ]
            manifest_path.write_text(json.dumps({
                "sources": {
                    "model": {
                        "profile": "v2",
                        "loader_kind": "multilingual",
                        "family": "chatterbox-multilingual",
                        "repositories": {
                            "base": {
                                "locator": "https://huggingface.co/Example/base",
                                "revision": "a" * 40,
                            },
                            "pack": {
                                "locator": "https://huggingface.co/Example/pack",
                                "revision": "b" * 40,
                            },
                        },
                        "files": records,
                    },
                },
            }), encoding="utf-8")

            parsed = worker._read_model_manifest(manifest_path, (root,))
            self.assertIsNone(parsed["repository"])
            self.assertIsNone(parsed["revision"])
            self.assertEqual(parsed["profile"], "v2")
            self.assertTrue(parsed["fingerprint"].startswith("chatterbox-v2-"))
            self.assertEqual(parsed["files"][0]["source"], "base")
            self.assertEqual(parsed["files"][-1]["source"], "pack")

            invalid = json.loads(manifest_path.read_text(encoding="utf-8"))
            invalid["sources"]["model"]["files"][0]["source"] = "missing"
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not a declared repository"):
                worker._read_model_manifest(manifest_path, (root,))

    def test_manifest_requires_nonempty_unique_safe_declared_model_files(self):
        records = [
            {"path": path, "size_bytes": 1, "sha256": "0" * 64}
            for path in CANONICAL_MODEL_FILE_PATHS
        ]
        self.assertEqual(
            tuple(record["path"] for record in worker._manifest_file_records(records)),
            CANONICAL_MODEL_FILE_PATHS,
        )
        self.assertEqual(len(worker._manifest_file_records(records[:5])), 5)
        expanded = [*records, {"path": "nested/extra.bin", "size_bytes": 1, "sha256": "1" * 64}]
        self.assertEqual(len(worker._manifest_file_records(expanded)), 7)
        with self.assertRaisesRegex(RuntimeError, "declared artifacts"):
            worker._manifest_file_records([])
        duplicate = [dict(record) for record in records]
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            worker._manifest_file_records(duplicate)
        unsafe = [dict(record) for record in records]
        unsafe[0]["path"] = "../outside.bin"
        with self.assertRaisesRegex(RuntimeError, "safe relative paths"):
            worker._manifest_file_records(unsafe)

    def test_default_loader_is_network_blocked_and_uses_verified_local_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "verified-model"
            revision = "a" * 40
            snapshot.mkdir(parents=True)
            model_file = snapshot / CANONICAL_MODEL_FILE_PATHS[0]
            model_file.write_bytes(b"verified model")
            for relative_path in CANONICAL_MODEL_FILE_PATHS[1:]:
                (snapshot / relative_path).write_bytes(model_file.read_bytes())
            digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
            manifest = root / "runtime-manifest.json"
            self.write_manifest(
                manifest,
                revision=revision,
                digest=digest,
                size=model_file.stat().st_size,
            )
            pkuseg_home = root / "pkuseg"
            pkuseg_model = pkuseg_home / "spacy_ontonotes"
            pkuseg_model.mkdir(parents=True)
            (pkuseg_model / "features.msgpack").write_bytes(b"features")
            (pkuseg_model / "weights.npz").write_bytes(b"weights")
            pkuseg_archive = pkuseg_home / worker.PKUSEG_DATA_FILENAME
            with zipfile.ZipFile(pkuseg_archive, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("features.msgpack", b"features")
                archive.writestr("weights.npz", b"weights")
            pkuseg_digest = hashlib.sha256(pkuseg_archive.read_bytes()).hexdigest()

            from_local_calls = []

            class FakeModel:
                sr = 24000

            class FakeMultilingual:
                @classmethod
                def get_supported_languages(cls):
                    return {language: language for language in worker.SUPPORTED_LANGUAGES}

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
            hub_module.snapshot_download = lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("worker must never acquire a model")
            )

            with (
                patch.dict(sys.modules, {
                    "chatterbox": chatterbox_package,
                    "chatterbox.mtl_tts": chatterbox_module,
                    "huggingface_hub": hub_module,
                }),
                patch("urllib.request.urlopen", side_effect=AssertionError("network access is forbidden")),
                patch.object(worker, "PKUSEG_DATA_SHA256", pkuseg_digest),
                patch.dict(
                    "os.environ",
                    {"HF_TOKEN": "must-not-survive", "PKUSEG_HOME": str(pkuseg_home)},
                    clear=False,
                ),
            ):
                instance = worker.ChatterboxWorker(
                    model_manifest_path=manifest,
                    approved_model_root=snapshot,
                    approved_manifest_root=root,
                )
                instance.load_model()
                offline_value = __import__("os").environ["HF_HUB_OFFLINE"]
                token_present = "HF_TOKEN" in __import__("os").environ

            self.assertEqual(instance.model_revision, revision)
            self.assertEqual(from_local_calls, [(str(snapshot.resolve()), {"device": "cpu"})])
            self.assertEqual(instance.supported_languages, worker.SUPPORTED_LANGUAGES)
            self.assertEqual(offline_value, "1")
            self.assertFalse(token_present)

    def test_v3_loader_uses_upstream_selector_when_runtime_exposes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "verified-model"
            revision = "e" * 40
            snapshot.mkdir(parents=True)
            content = b"verified-v3"
            v3_paths = V3_MODEL_FILE_PATHS
            for relative_path in v3_paths:
                (snapshot / relative_path).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            manifest = root / "runtime-manifest-v3.json"
            self.write_manifest(
                manifest,
                revision=revision,
                digest=digest,
                size=len(content),
                variant="v3",
            )
            pkuseg_home = root / "pkuseg"
            pkuseg_model = pkuseg_home / "spacy_ontonotes"
            pkuseg_model.mkdir(parents=True)
            (pkuseg_model / "features.msgpack").write_bytes(b"features")
            (pkuseg_model / "weights.npz").write_bytes(b"weights")
            pkuseg_archive = pkuseg_home / worker.PKUSEG_DATA_FILENAME
            with zipfile.ZipFile(pkuseg_archive, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("features.msgpack", b"features")
                archive.writestr("weights.npz", b"weights")
            pkuseg_digest = hashlib.sha256(pkuseg_archive.read_bytes()).hexdigest()

            from_local_calls = []

            class FakeModel:
                sr = 24000

            class FakeMultilingual:
                @classmethod
                def get_supported_languages(cls):
                    return {language: language for language in worker.SUPPORTED_LANGUAGES}

                @classmethod
                def from_local(cls, checkpoint_dir, *, device, t3_model=None):
                    from_local_calls.append((checkpoint_dir, device, t3_model))
                    return FakeModel()

            chatterbox_package = types.ModuleType("chatterbox")
            chatterbox_package.__path__ = []
            chatterbox_module = types.ModuleType("chatterbox.mtl_tts")
            chatterbox_module.ChatterboxMultilingualTTS = FakeMultilingual

            with (
                patch.dict(sys.modules, {
                    "chatterbox": chatterbox_package,
                    "chatterbox.mtl_tts": chatterbox_module,
                }),
                patch.object(worker, "PKUSEG_DATA_SHA256", pkuseg_digest),
                patch.dict("os.environ", {"PKUSEG_HOME": str(pkuseg_home)}, clear=False),
            ):
                instance = worker.ChatterboxWorker(
                    model_manifest_path=manifest,
                    approved_model_root=snapshot,
                    approved_manifest_root=root,
                )
                instance.load_model()

            self.assertEqual(instance.model_variant, "v3")
            self.assertFalse(instance._v3_compat_mode)
            self.assertEqual(
                from_local_calls,
                [(str(snapshot.resolve()), "cpu", "v3")],
            )

    def test_v3_loader_compatibility_path_matches_pinned_runtime_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)

            class _Loadable:
                def load_state_dict(self, _state):
                    return None

                def to(self, _device):
                    return self

                def eval(self):
                    return self

            class FakeVoiceEncoder(_Loadable):
                pass

            class FakeT3(_Loadable):
                def __init__(self, _config):
                    pass

            class FakeS3Gen(_Loadable):
                pass

            class FakeT3Config:
                @staticmethod
                def multilingual():
                    return object()

            class FakeTokenizer:
                def __init__(self, path):
                    self.path = path

            class FakeConditionals:
                pass

            class FakeWatermarker:
                def __init__(self):
                    self.calls = []

                def apply_watermark(self, wav, sample_rate):
                    self.calls.append((len(wav), sample_rate))
                    return wav

            class FakeMultilingual:
                @classmethod
                def from_local(cls, checkpoint_dir, device):
                    raise AssertionError("pinned V3 runtime must use the compatibility loader")

                def __init__(self, t3, s3gen, ve, tokenizer, device, conds=None):
                    self.t3 = t3
                    self.s3gen = s3gen
                    self.ve = ve
                    self.tokenizer = tokenizer
                    self.device = device
                    self.conds = conds
                    self.watermarker = FakeWatermarker()

            chatterbox_package = types.ModuleType("chatterbox")
            chatterbox_package.__path__ = []
            models_package = types.ModuleType("chatterbox.models")
            models_package.__path__ = []
            t3_package = types.ModuleType("chatterbox.models.t3")
            t3_package.__path__ = []
            t3_package.T3 = FakeT3
            t3_module = types.ModuleType("chatterbox.models.t3.t3")
            modules_package = types.ModuleType("chatterbox.models.t3.modules")
            modules_package.__path__ = []
            t3_config_module = types.ModuleType("chatterbox.models.t3.modules.t3_config")
            t3_config_module.T3Config = FakeT3Config
            s3gen_module = types.ModuleType("chatterbox.models.s3gen")
            s3gen_module.S3Gen = FakeS3Gen
            s3tokenizer_module = types.ModuleType("chatterbox.models.s3tokenizer")
            s3tokenizer_module.S3_TOKEN_RATE = 5
            tokenizers_module = types.ModuleType("chatterbox.models.tokenizers")
            tokenizers_module.MTLTokenizer = FakeTokenizer
            voice_encoder_module = types.ModuleType("chatterbox.models.voice_encoder")
            voice_encoder_module.VoiceEncoder = FakeVoiceEncoder

            mtl_module = types.ModuleType("chatterbox.mtl_tts")
            mtl_module.ChatterboxMultilingualTTS = FakeMultilingual
            mtl_module.Conditionals = FakeConditionals
            mtl_module.drop_invalid_tokens = lambda tokens: tokens

            torch_module = types.ModuleType("torch")
            torch_module.device = lambda value: value
            torch_module.load = lambda *_args, **_kwargs: {}

            safetensors_package = types.ModuleType("safetensors")
            safetensors_package.__path__ = []
            safetensors_torch_module = types.ModuleType("safetensors.torch")
            safetensors_torch_module.load_file = lambda _path: {}

            module_map = {
                "chatterbox": chatterbox_package,
                "chatterbox.mtl_tts": mtl_module,
                "chatterbox.models": models_package,
                "chatterbox.models.t3": t3_package,
                "chatterbox.models.t3.t3": t3_module,
                "chatterbox.models.t3.modules": modules_package,
                "chatterbox.models.t3.modules.t3_config": t3_config_module,
                "chatterbox.models.s3gen": s3gen_module,
                "chatterbox.models.s3tokenizer": s3tokenizer_module,
                "chatterbox.models.tokenizers": tokenizers_module,
                "chatterbox.models.voice_encoder": voice_encoder_module,
                "torch": torch_module,
                "safetensors": safetensors_package,
                "safetensors.torch": safetensors_torch_module,
            }

            with patch.dict(sys.modules, module_map):
                model, compatibility_mode = worker._load_local_chatterbox_model(
                    snapshot,
                    "v3",
                )

                class FakeTokens:
                    shape = (1, 4)

                filtered = mtl_module.drop_invalid_tokens(FakeTokens())
                self.assertEqual(filtered.shape, (1, 4))
                watermarked = model.watermarker.apply_watermark(
                    list(range(100)),
                    sample_rate=20,
                )

            self.assertTrue(compatibility_mode)
            self.assertEqual(len(watermarked), 12)
            self.assertEqual(model.watermarker.inner.calls, [(12, 20)])
            self.assertTrue(hasattr(t3_module, "AlignmentStreamAnalyzer"))

    def test_default_loader_rejects_missing_pinned_tokenizer_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"PKUSEG_HOME": str(Path(directory) / "missing")}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "PKUSEG_HOME must be an existing local directory"):
                    worker._require_local_pkuseg_data()

    def test_default_loader_rejects_snapshot_checksum_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "verified-model"
            revision = "b" * 40
            snapshot.mkdir(parents=True)
            (snapshot / CANONICAL_MODEL_FILE_PATHS[0]).write_bytes(b"wrong")
            manifest = root / "runtime-manifest.json"
            self.write_manifest(
                manifest,
                revision=revision,
                digest="0" * 64,
                size=5,
            )
            instance = worker.ChatterboxWorker(
                model_manifest_path=manifest,
                approved_model_root=snapshot,
                approved_manifest_root=root,
            )
            with self.assertRaisesRegex(RuntimeError, "checksum does not match"):
                instance.load_model()
            self.assertIsNone(instance.model)

    def test_local_loader_rejects_partial_extra_and_symlinked_snapshots(self):
        cases = ("missing", "extra", "symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshot = root / "verified-model"
                snapshot.mkdir()
                content = b"verified model"
                model_file = snapshot / CANONICAL_MODEL_FILE_PATHS[0]
                if case != "missing":
                    if case == "symlink":
                        outside = root / "outside.bin"
                        outside.write_bytes(content)
                        model_file.symlink_to(outside)
                    else:
                        model_file.write_bytes(content)
                        (snapshot / "undeclared.bin").write_bytes(b"extra")
                manifest = root / "runtime-manifest.json"
                self.write_manifest(
                    manifest,
                    revision="c" * 40,
                    digest=hashlib.sha256(content).hexdigest(),
                    size=len(content),
                )
                instance = worker.ChatterboxWorker(
                    model_manifest_path=manifest,
                    approved_model_root=snapshot,
                    approved_manifest_root=root,
                )
                with self.assertRaises(RuntimeError):
                    instance.load_model()

    def test_injected_model_loader_remains_manifest_free(self):
        instance = worker.ChatterboxWorker(model_loader=lambda: types.SimpleNamespace(sr=24000))
        instance.load_model()
        self.assertEqual(instance.sample_rate, 24000)

    def test_worker_serializes_jsonl_responses_across_threads(self):
        instance = worker.ChatterboxWorker(model_loader=lambda: types.SimpleNamespace(sr=24000))
        state_lock = threading.Lock()
        active = 0
        peak = 0

        def observe(_message):
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1

        with patch.object(worker, "_message", side_effect=observe):
            threads = [threading.Thread(target=instance._send, args=({"id": index},)) for index in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(peak, 1)

    def test_normal_worker_requires_matching_runtime_model_and_activation_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            runtime_receipt = root / "runtime.json"
            model_receipt = root / "model.json"
            activation_receipt = root / "activation.json"
            runtime_receipt.write_text(json.dumps({
                "schema": worker.RUNTIME_RECEIPT_SCHEMA,
                "status": "ready",
                "runtime_fingerprint": "runtime-id",
            }), encoding="utf-8")
            model_receipt.write_text(json.dumps({
                "schema": worker.MODEL_RECEIPT_SCHEMA,
                "status": "ready",
                "model_fingerprint": "model-id",
                "artifact": {"snapshot_path": str(snapshot)},
            }), encoding="utf-8")
            activation_receipt.write_text(json.dumps({
                "schema": worker.ACTIVATION_RECEIPT_SCHEMA,
                "status": "ready",
                "runtime_fingerprint": "runtime-id",
                "model_fingerprint": "model-id",
                "runtime_receipt_sha256": hashlib.sha256(runtime_receipt.read_bytes()).hexdigest(),
                "model_receipt_sha256": hashlib.sha256(model_receipt.read_bytes()).hexdigest(),
                "model_snapshot": str(snapshot),
                "checks": {"local_model_load": {"ok": True}},
            }), encoding="utf-8")

            self.write_publication_proof(runtime_receipt, "runtime")
            self.write_publication_proof(model_receipt, "model")
            self.write_publication_proof(activation_receipt, "activation")
            environment = {
                "E2A_CHATTERBOX_RUNTIME_RECEIPT": str(runtime_receipt),
                "E2A_CHATTERBOX_MODEL_RECEIPT": str(model_receipt),
                "E2A_CHATTERBOX_ACTIVATION_RECEIPT": str(activation_receipt),
            }
            with patch.dict("os.environ", environment, clear=False):
                worker._validate_activation_chain(snapshot)
                model_receipt.write_text('{"schema":"stale","status":"ready"}', encoding="utf-8")
                self.write_publication_proof(model_receipt, "model")
                with self.assertRaisesRegex(RuntimeError, "model receipt is not ready"):
                    worker._validate_activation_chain(snapshot)

    def test_receipt_ambiguity_markers_require_repair_in_current_and_fresh_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = []
            for marker_kind in ("dangling_symlink", "special_file"):
                for label, schema in (
                    ("runtime", worker.RUNTIME_RECEIPT_SCHEMA),
                    ("model", worker.MODEL_RECEIPT_SCHEMA),
                    ("activation", worker.ACTIVATION_RECEIPT_SCHEMA),
                ):
                    key = f"{marker_kind}:{label}"
                    receipt = root / f"{marker_kind}-{label}.json"
                    receipt.write_text(
                        json.dumps({"schema": schema, "status": "ready"}),
                        encoding="utf-8",
                    )
                    self.write_publication_proof(receipt, label)
                    self.create_nonregular_ambiguity_marker(receipt, marker_kind)
                    receipts.append((key, label, schema, receipt))
                    with self.subTest(key=key), self.assertRaisesRegex(
                        worker.ReceiptRepairRequiredError,
                        rf"{label} receipt publication.*repair_required",
                    ):
                        worker._read_private_receipt(str(receipt), label, schema)

            script = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("fresh_chatterbox_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = {}
for key, label, schema, receipt in json.loads(sys.argv[2]):
    try:
        module._read_private_receipt(receipt, label, schema)
    except RuntimeError as exc:
        results[key] = str(exc)
    else:
        results[key] = None
print(json.dumps(results, sort_keys=True))
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(WORKER_PATH),
                    json.dumps(
                        [
                            (key, label, schema, str(receipt))
                            for key, label, schema, receipt in receipts
                        ]
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            fresh_results = json.loads(completed.stdout)
            for key, _, _, _ in receipts:
                with self.subTest(fresh_key=key):
                    self.assertIn("repair_required", fresh_results[key])

    def test_publication_markers_use_entry_presence_and_fail_closed(self):
        for label, schema in (
            ("runtime", worker.RUNTIME_RECEIPT_SCHEMA),
            ("model", worker.MODEL_RECEIPT_SCHEMA),
            ("activation", worker.ACTIVATION_RECEIPT_SCHEMA),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = root / f"{label}.json"
                receipt.write_text(
                    json.dumps({"schema": schema, "status": "ready"}),
                    encoding="utf-8",
                )
                self.write_publication_proof(receipt, label)
                publication = receipt.with_name(f"{receipt.name}.publishing")
                published = receipt.with_name(f"{receipt.name}.published")

                self.assertFalse(os.path.lexists(publication))
                self.assertTrue(published.is_file())
                self.assertEqual(
                    worker._read_private_receipt(str(receipt), label, schema)[0],
                    receipt,
                )

                publication.symlink_to(root / "missing-publishing-proof")
                with self.assertRaisesRegex(
                    worker.ReceiptRepairRequiredError,
                    rf"{label} receipt publication.*repair_required",
                ):
                    worker._read_private_receipt(str(receipt), label, schema)

                publication.unlink()
                self.assertEqual(
                    worker._read_private_receipt(str(receipt), label, schema)[0],
                    receipt,
                )

                published.unlink()
                published.symlink_to(root / "missing-published-proof")
                with self.assertRaisesRegex(
                    worker.ReceiptRepairRequiredError,
                    rf"{label} receipt publication proof.*repair_required",
                ):
                    worker._read_private_receipt(str(receipt), label, schema)

    def test_production_startup_classifies_all_nonregular_ambiguity_markers_as_repair_required(self):
        for marker_kind in ("dangling_symlink", "special_file"):
            for label in ("runtime", "model", "activation"):
                with self.subTest(marker_kind=marker_kind, label=label), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    snapshot, receipts, environment = self.write_ready_activation_chain(root)
                    self.create_nonregular_ambiguity_marker(receipts[label], marker_kind)
                    output = io.StringIO()
                    errors = io.StringIO()
                    with (
                        patch.dict("os.environ", environment, clear=False),
                        redirect_stdout(output),
                        redirect_stderr(errors),
                    ):
                        result = worker.main([
                            "--approved-root", str(root),
                            "--model-manifest", str(root / "runtime-manifest.json"),
                            "--approved-model-root", str(snapshot),
                            "--approved-manifest-root", str(root),
                        ])
                    self.assertEqual(result, 1)
                    message = json.loads(output.getvalue())
                    self.assertEqual(message["event"], "error")
                    self.assertEqual(message["error"]["code"], "repair_required")
                    self.assertIn("ReceiptRepairRequiredError", errors.getvalue())

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
