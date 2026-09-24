from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from components.Chatterbox.runtime import contract_data
import components.Chatterbox.runtime.runtime as runtime_module
from components.Chatterbox.runtime.runtime import (
    _default_repo_root,
    HostRuntimeStatus,
    ProvisioningError,
    RuntimeConfigurationError,
    RuntimeManifestError,
    RuntimeReceiptError,
    acquire_model,
    activation_status,
    build_identity_contract,
    build_paths,
    calculate_storage_plan,
    host_runtime_status,
    install_runtime,
    model_preflight,
    model_status,
    preflight,
    product_status,
    readiness_snapshot,
    receipt_identity_contracts,
    sanitized_worker_environment,
    runtime_status,
    validate_manifest_identity,
    validate_measurement_scope,
    validate_model_receipt,
    validate_receipt_identity_links,
    verified_file_credit,
    verify_lock,
    write_runtime_receipt,
)


CANONICAL_MODEL_FILE_PATHS = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)


def _manifest(lock_sha256: str | None = None, *, unresolved: list[str] | None = None) -> dict:
    return {
        "manifest_version": "2.0.0",
        "runtime_contract_version": "2.0.0",
        "target": {"os": "linux", "architecture": "x86_64", "python": "3.11.x", "backend": "cpu"},
        "product": {"profile": "chatterbox-multilingual-v2-cpu", "worker_protocol": 1},
        "sources": {
            "chatterbox_package": {
                "filename": "chatterbox_tts-0.1.7-py3-none-any.whl",
                "version": "0.1.7",
                "locator": "https://example.invalid/chatterbox.whl",
                "artifact_sha256": "0" * 64,
            },
            "chatterbox_source": {
                "locator": "https://example.invalid/chatterbox",
                "revision": "1" * 40,
                "association_to_artifact": "unverified",
            },
            "perth": {
                "filename": "resemble_perth-1.0.1-py3-none-any.whl",
                "version": "1.0.1",
                "locator": "https://example.invalid/perth.whl",
                "commit": "2" * 40,
                "artifact_sha256": "3" * 64,
                "association_to_artifact": "unverified",
            },
            "model": {
                "locator": "https://huggingface.co/Example/model",
                "variant": "multilingual-v2",
                "revision": "4" * 40,
                "files": [
                    {"path": path, "size_bytes": 7, "sha256": "5" * 64}
                    for path in CANONICAL_MODEL_FILE_PATHS
                ],
            },
        },
        "unresolved_identities": unresolved or [],
        "lock": {"path": "requirements-cpu.lock", "status": "verified", "sha256": lock_sha256},
        "storage": {
            "contract_version": "1.0.0",
            "phases": [
                {
                    "name": "runtime_artifact_acquisition",
                    "buckets": [
                        {
                            "name": "package_download_and_cache_peak",
                            "destination": "environment",
                            "required_bytes": None,
                            "measurement_status": "required_clean_disposable",
                        }
                    ],
                },
                {
                    "name": "runtime_environment_construction",
                    "buckets": [
                        {
                            "name": "expanded_environment_high_water",
                            "destination": "environment",
                            "required_bytes": None,
                            "measurement_status": "required_clean_disposable",
                        }
                    ],
                }
            ],
        },
    }


def _copy(value: dict) -> dict:
    return json.loads(json.dumps(value))


def _write_runtime(root: Path, manifest: dict) -> tuple[Path, dict]:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir()
    lock = runtime_dir / "requirements-cpu.lock"
    lock.write_text("# e2a-lock-format: 1\n--require-hashes\n", encoding="utf-8")
    manifest = _copy(manifest)
    manifest["lock"]["sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    (runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return runtime_dir, manifest


def _fresh_process_statuses(paths) -> dict:
    """Read all receipt states in a fresh interpreter with no in-process quarantine."""

    environment = {
        "XDG_DATA_HOME": str(paths.data_home),
        "XDG_STATE_HOME": str(paths.state_home),
        "E2A_ROOT": str(paths.e2a_root),
    }
    script = """
import json
import sys
from pathlib import Path
from components.Chatterbox.runtime.runtime import (
    activation_status,
    build_paths,
    model_status,
    runtime_status,
)

paths = build_paths(
    runtime_dir=Path(sys.argv[1]),
    repo_root=Path(sys.argv[2]),
    environment=json.loads(sys.argv[3]),
)
print(json.dumps({
    "runtime": runtime_status(paths),
    "model": model_status(paths),
    "activation": activation_status(paths),
    "derived_paths": {
        "environment": str(paths.environment),
        "model_namespace": str(paths.model_namespace),
        "verified_model_root": (
            str(paths.verified_model_root) if paths.verified_model_root is not None else None
        ),
    },
}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(paths.runtime_dir),
            str(paths.repo_root),
            json.dumps(environment),
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _install_fixture(root: Path) -> tuple[object, Path]:
    manifest = _manifest()
    manifest["sources"]["model"]["locator"] = "https://huggingface.co/Example/model"
    manifest["validation"] = {
        "import_modules": ["chatterbox.mtl_tts", "perth"],
        "worker_self_test_args": ["--self-test"],
    }
    manifest["storage"] = {
        "contract_version": "1.0.0",
        "phases": [
            {
                "name": "runtime_artifact_acquisition",
                "buckets": [{"name": "download", "destination": "environment", "required_bytes": 1}],
            },
            {
                "name": "runtime_environment_construction",
                "buckets": [{"name": "expanded", "destination": "environment", "required_bytes": 1}],
            },
        ],
    }
    runtime_dir, _ = _write_runtime(root, manifest)
    worker = root / "components/Chatterbox/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# synthetic worker\n", encoding="utf-8")
    paths = build_paths(
        runtime_dir=runtime_dir,
        repo_root=root,
        environment={
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_STATE_HOME": str(root / "state"),
            "E2A_ROOT": str(root / "e2a"),
        },
    )
    return paths, worker


def _model_fixture(root: Path) -> tuple[object, Path, bytes]:
    content = b"verified-v2-model"
    manifest = _manifest()
    manifest["sources"]["model"]["locator"] = "https://huggingface.co/Example/model"
    manifest["sources"]["model"]["files"] = [
        {
            "path": path,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path in CANONICAL_MODEL_FILE_PATHS
    ]
    manifest["validation"] = {
        "import_modules": ["chatterbox.mtl_tts", "perth"],
        "worker_self_test_args": ["--self-test"],
    }
    manifest["network"] = {
        "runtime_acquisition_urls": ["https://example.invalid/runtime"],
        "model_acquisition_urls": ["https://huggingface.co/Example/model"],
    }
    manifest["storage"] = {
        "contract_version": "1.0.0",
        "phases": [
            {"name": "runtime_artifact_acquisition", "buckets": [{"name": "download", "destination": "environment", "required_bytes": 1}]},
            {"name": "runtime_environment_construction", "buckets": [{"name": "expanded", "destination": "environment", "required_bytes": 1}]},
            {"name": "model_acquisition", "buckets": [
                {"name": "model", "destination": "verified_model_root", "source": "model_files", "credit_verified_files": True},
                {
                    "name": "staging",
                    "destination": "verified_model_root",
                    "required_bytes": 0,
                    "measurement_status": "measured_clean_disposable",
                    "measurement_evidence": {
                        "schema": "ebook2audiobook.chatterbox-storage-measurement.v1",
                        "evidence_id": "synthetic-model-fixture-zero",
                        "measured_bytes": 0,
                        "runs": 1,
                    },
                },
            ]},
            {"name": "activation_and_self_test", "buckets": [
                {"name": "receipt", "destination": "state", "required_bytes": 1},
                {"name": "temporary", "destination": "run", "required_bytes": 1},
            ]},
        ],
    }
    runtime_dir, _ = _write_runtime(root, manifest)
    worker = root / "components/Chatterbox/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# synthetic worker\n", encoding="utf-8")
    environment = {
        "XDG_DATA_HOME": str(root / "data"),
        "XDG_STATE_HOME": str(root / "state"),
        "E2A_ROOT": str(root / "e2a"),
    }
    paths = build_paths(runtime_dir=runtime_dir, repo_root=root, environment=environment)
    with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
        install_runtime(
            paths,
            Path(sys.executable),
            worker_script=worker,
            command_runner=_SyntheticRunner(),
            import_hook=lambda python, modules: (True, "imports passed"),
            self_test_hook=lambda python, args, env: (True, "self-test passed"),
            nonce_factory=lambda: "runtime-transaction",
        )
    return build_paths(runtime_dir=runtime_dir, repo_root=root, environment=environment), worker, content


class _SyntheticRunner:
    def __init__(self, failure: str | None = None, entered: threading.Event | None = None, release: threading.Event | None = None, python_symlink: bool = False):
        self.failure = failure
        self.entered = entered
        self.release = release
        self.python_symlink = python_symlink
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        command = [str(item) for item in args]
        self.calls.append(command)
        if "venv" in command:
            if self.entered is not None:
                self.entered.set()
            if self.release is not None:
                self.release.wait(timeout=5)
            if self.failure == "venv":
                return subprocess.CompletedProcess(command, 1, "", "failed")
            candidate = Path(command[-1])
            python = candidate / "bin/python"
            python.parent.mkdir(parents=True)
            if self.python_symlink:
                python.symlink_to(Path(sys.executable))
            else:
                python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                python.chmod(0o700)
        elif "install" in command and self.failure == "install":
            return subprocess.CompletedProcess(command, 1, "", "failed")
        elif "check" in command and self.failure == "pip_check":
            return subprocess.CompletedProcess(command, 1, "", "failed")
        return subprocess.CompletedProcess(command, 0, "", "")


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def _create_nonregular_ambiguity_marker(receipt_path: Path, marker_kind: str) -> Path:
        ambiguity = receipt_path.with_name(f"{receipt_path.name}.ambiguous")
        ambiguity.parent.mkdir(parents=True, exist_ok=True)
        if marker_kind == "dangling_symlink":
            ambiguity.symlink_to(ambiguity.with_name(f"{ambiguity.name}.missing"))
        elif marker_kind == "special_file":
            os.mkfifo(ambiguity)
        else:
            raise AssertionError(f"unsupported marker kind: {marker_kind}")
        return ambiguity

    def _assert_combined_receipt_rollback_failure_is_persistent(
        self,
        paths,
        receipt_path: Path,
        label: str,
        publish,
        status_reader,
    ) -> None:
        publication = receipt_path.with_name(f"{receipt_path.name}.publishing")
        published = receipt_path.with_name(f"{receipt_path.name}.published")
        ambiguity = receipt_path.with_name(f"{receipt_path.name}.ambiguous")
        real_fsync_directory = runtime_module._fsync_directory
        real_replace = runtime_module.os.replace
        real_link = runtime_module.os.link

        def fail_final_fsync(path):
            if Path(path) == receipt_path.parent and published.exists() and ambiguity.exists():
                raise OSError(f"injected final {label} receipt fsync failure")
            return real_fsync_directory(path)

        def fail_rollback_replace(source, destination):
            if Path(source) == published and Path(destination) == publication:
                raise OSError(f"injected {label} rollback replace failure")
            return real_replace(source, destination)

        def fail_rollback_link(source, destination, *args, **kwargs):
            if Path(source) == published and Path(destination) == publication:
                raise OSError(f"injected {label} rollback link failure")
            return real_link(source, destination, *args, **kwargs)

        with (
            mock.patch(
                "components.Chatterbox.runtime.runtime._fsync_directory",
                side_effect=fail_final_fsync,
            ),
            mock.patch(
                "components.Chatterbox.runtime.runtime.os.replace",
                side_effect=fail_rollback_replace,
            ),
            mock.patch(
                "components.Chatterbox.runtime.runtime.os.link",
                side_effect=fail_rollback_link,
            ),
        ):
            with self.assertRaisesRegex(OSError, f"injected final {label} receipt fsync failure") as raised:
                publish()

        notes = " ".join(getattr(raised.exception, "__notes__", ()))
        self.assertIn(f"{label} receipt publication rollback rename failed", notes)
        self.assertIn(f"{label} receipt publication quarantine link failed", notes)
        self.assertTrue(receipt_path.exists())
        self.assertFalse(publication.exists())
        self.assertTrue(published.exists())
        self.assertTrue(ambiguity.exists())
        self.assertEqual(status_reader(paths)["status"], "repair_required")
        self.assertEqual(_fresh_process_statuses(paths)[label]["status"], "repair_required")

    def test_default_repo_root_matches_direct_runtime_layout(self) -> None:
        runtime_dir = Path(__file__).resolve().parents[1]
        expected = Path(__file__).resolve().parents[4]
        self.assertEqual(_default_repo_root(runtime_dir), expected)

    def test_nonregular_ambiguity_markers_fail_closed_for_every_receipt(self) -> None:
        for marker_kind in ("dangling_symlink", "special_file"):
            with self.subTest(marker_kind=marker_kind), tempfile.TemporaryDirectory() as directory:
                paths, _ = _install_fixture(Path(directory))
                dimensions = (
                    ("runtime", paths.runtime_receipt_path, runtime_status),
                    ("model", paths.model_receipt_path, model_status),
                    ("activation", paths.activation_receipt_path, activation_status),
                )
                for label, receipt_path, status_reader in dimensions:
                    with self.subTest(marker_kind=marker_kind, label=label):
                        ambiguity = self._create_nonregular_ambiguity_marker(
                            receipt_path, marker_kind
                        )
                        current = status_reader(paths)
                        self.assertEqual(current["status"], "repair_required")
                        self.assertIn(str(ambiguity), current["ambiguous_paths"])
                        fresh = _fresh_process_statuses(paths)
                        self.assertEqual(fresh[label]["status"], "repair_required")
                        self.assertIn(str(ambiguity), fresh[label]["ambiguous_paths"])
                        ambiguity.unlink()

    def test_publication_markers_use_entry_presence_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _ = _model_fixture(Path(directory))
            publication = paths.runtime_receipt_path.with_name(
                f"{paths.runtime_receipt_path.name}.publishing"
            )
            published = paths.runtime_receipt_path.with_name(
                f"{paths.runtime_receipt_path.name}.published"
            )

            self.assertFalse(os.path.lexists(publication))
            self.assertTrue(published.is_file())
            self.assertEqual(runtime_status(paths)["status"], "ready")

            publication.symlink_to(Path(directory) / "missing-publishing-proof")
            current = runtime_status(paths)
            self.assertEqual(current["status"], "repair_required")
            self.assertIn(str(publication), current["ambiguous_paths"])
            self.assertEqual(
                _fresh_process_statuses(paths)["runtime"]["status"],
                "repair_required",
            )

            publication.unlink()
            self.assertEqual(runtime_status(paths)["status"], "ready")

            published.unlink()
            published.symlink_to(Path(directory) / "missing-published-proof")
            current = runtime_status(paths)
            self.assertEqual(current["status"], "repair_required")
            self.assertIn(str(published), current["ambiguous_paths"])
            self.assertEqual(
                _fresh_process_statuses(paths)["runtime"]["status"],
                "repair_required",
            )

    def test_receipt_write_guards_treat_dangling_publication_markers_as_present(self) -> None:
        writers = (
            ("runtime", runtime_module.write_runtime_receipt),
            ("model", runtime_module.write_model_receipt),
            ("activation", runtime_module.write_activation_receipt),
        )
        for label, writer in writers:
            for marker_name in ("publishing", "published"):
                with self.subTest(label=label, marker_name=marker_name), tempfile.TemporaryDirectory() as directory:
                    paths, _ = _install_fixture(Path(directory))
                    receipt_path = getattr(paths, f"{label}_receipt_path")
                    marker = receipt_path.with_name(f"{receipt_path.name}.{marker_name}")
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.symlink_to(Path(directory) / f"missing-{label}-{marker_name}")
                    with self.assertRaisesRegex(ProvisioningError, "receipt"):
                        writer(paths, {"status": "ready"})

    def test_hash_checked_lock_accepts_only_pinned_hashed_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements-cpu.lock"
            contents = "\n".join(
                [
                    "# e2a-lock-format: 1",
                    "--require-hashes",
                    "--index-url https://pypi.org/simple",
                    "--extra-index-url https://download.pytorch.org/whl/cpu",
                    f"demo==1.0 --hash=sha256:{'a' * 64}",
                    "",
                ]
            )
            path.write_text(contents, encoding="utf-8")
            digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
            result = verify_lock(path, _manifest(digest))
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["requirements"], 1)

            path.write_text(contents.replace("demo==1.0", "demo @ git+https://example.invalid/demo.git@master"), encoding="utf-8")
            rejected = verify_lock(path, _manifest(hashlib.sha256(path.read_bytes()).hexdigest()))
            self.assertFalse(rejected["ok"])
            self.assertTrue(any("mutable" in error or "pinned" in error for error in rejected["errors"]))

    def test_unresolved_manifest_is_rejected_without_inventing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements-cpu.lock"
            path.write_text("# e2a-lock-format: 1\n--require-hashes\n", encoding="utf-8")
            result = verify_lock(path, _manifest(unresolved=["perth.commit"]))
            self.assertFalse(result["ok"])
            self.assertTrue(any("unresolved" in error for error in result["errors"]))

    def test_manifest_and_runtime_contract_versions_are_explicit(self) -> None:
        manifest = _manifest("a" * 64)
        self.assertTrue(validate_manifest_identity(manifest)["ok"])
        changed_contract = _copy(manifest)
        changed_contract["runtime_contract_version"] = "2.1.0"
        self.assertNotEqual(
            build_identity_contract(changed_contract, "a" * 64)["runtime"]["fingerprint"],
            build_identity_contract(manifest, "a" * 64)["runtime"]["fingerprint"],
        )

        legacy = _copy(manifest)
        legacy["manifest_version"] = "1.0.0"
        legacy.pop("runtime_contract_version")
        result = validate_manifest_identity(legacy)
        self.assertFalse(result["ok"])
        self.assertIn("manifest_version must be 2.0.0", result["errors"])
        self.assertIn("runtime_contract_version must be 2.0.0", result["errors"])

    def test_runtime_manifest_accepts_arbitrary_declared_model_files_and_rejects_ambiguity(self) -> None:
        manifest = _manifest("a" * 64)
        self.assertTrue(validate_manifest_identity(manifest)["ok"])
        self.assertEqual(
            tuple(record["path"] for record in runtime_module._model_file_records(manifest)),
            CANONICAL_MODEL_FILE_PATHS,
        )

        reduced = _copy(manifest)
        reduced["sources"]["model"]["files"] = reduced["sources"]["model"]["files"][:3]
        self.assertTrue(validate_manifest_identity(reduced)["ok"])
        self.assertEqual(len(runtime_module._model_file_records(reduced)), 3)

        expanded = _copy(manifest)
        expanded["sources"]["model"]["files"].append(
            {"path": "nested/extra.bin", "size_bytes": 7, "sha256": "6" * 64}
        )
        self.assertTrue(validate_manifest_identity(expanded)["ok"])
        self.assertEqual(len(runtime_module._model_file_records(expanded)), 7)

        duplicate = _copy(manifest)
        duplicate["sources"]["model"]["files"][-1] = _copy(duplicate["sources"]["model"]["files"][0])
        identity = validate_manifest_identity(duplicate)
        self.assertFalse(identity["ok"])
        self.assertTrue(any("must be unique" in error for error in identity["errors"]))
        with self.assertRaisesRegex(RuntimeConfigurationError, "must be unique"):
            runtime_module._model_file_records(duplicate)

        multi = _copy(manifest)
        model = multi["sources"]["model"]
        model["profile"] = "v2"
        model["loader_kind"] = "multilingual"
        model["family"] = "chatterbox-multilingual"
        model["repositories"] = {
            "base": {"locator": "https://huggingface.co/Example/base", "revision": "4" * 40},
            "pack": {"locator": "https://huggingface.co/Example/pack", "revision": "7" * 40},
        }
        model["files"][0]["source"] = "base"
        for record in model["files"][1:]:
            record["source"] = "pack"
        self.assertTrue(validate_manifest_identity(multi)["ok"])
        records = runtime_module._model_file_records(multi)
        self.assertEqual(records[0]["source"], "base")
        self.assertEqual(records[1]["source"], "pack")

        wrong_source = _copy(multi)
        wrong_source["sources"]["model"]["files"][0]["source"] = "missing"
        identity = validate_manifest_identity(wrong_source)
        self.assertFalse(identity["ok"])
        self.assertTrue(any("source repository" in error for error in identity["errors"]))
        with self.assertRaisesRegex(RuntimeConfigurationError, "source is invalid"):
            runtime_module._model_file_records(wrong_source)

    def test_paths_and_fingerprint_are_stable_and_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            lock = runtime_dir / "requirements-cpu.lock"
            lock.write_text("# unresolved lock input\n", encoding="utf-8")
            manifest = _manifest(hashlib.sha256(lock.read_bytes()).hexdigest(), unresolved=["all final hashes"])
            (runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            environment = {
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_STATE_HOME": str(root / "state"),
                "E2A_ROOT": str(root / "e2a"),
            }
            first = build_paths(runtime_dir=runtime_dir, repo_root=root, environment=environment)
            second = build_paths(runtime_dir=runtime_dir, repo_root=root, environment=environment)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertTrue(first.environment.is_relative_to(first.env_base))
            self.assertTrue(first.model_namespace.is_relative_to(first.model_objects_dir))
            self.assertFalse(first.model_namespace.is_relative_to(first.legacy_model_cache))
            self.assertIsNone(first.verified_model_root)
            self.assertTrue(first.run_namespace.is_relative_to(first.run_dir))
            self.assertIn(first.fingerprint, first.result_path.name)

    def test_executable_identities_are_separated_from_policy_and_provenance(self) -> None:
        manifest = _manifest("a" * 64)
        baseline = build_identity_contract(manifest, "a" * 64)

        non_executable = _copy(manifest)
        non_executable["sources"]["chatterbox_source"]["revision"] = "9" * 40
        non_executable["storage"]["review_note"] = "policy-only change"
        non_executable["network"] = {"first_acquisition_urls": ["https://example.invalid/simple/"]}
        self.assertEqual(build_identity_contract(non_executable, "a" * 64), baseline)

        mirrored = _copy(manifest)
        mirrored["sources"]["chatterbox_package"]["locator"] = "https://mirror.invalid/chatterbox.whl"
        mirrored["sources"]["perth"]["locator"] = "https://mirror.invalid/perth.whl"
        self.assertEqual(
            build_identity_contract(mirrored, "a" * 64)["runtime"]["fingerprint"],
            baseline["runtime"]["fingerprint"],
        )

        lock_changed = build_identity_contract(manifest, "d" * 64)
        self.assertNotEqual(lock_changed["runtime"]["fingerprint"], baseline["runtime"]["fingerprint"])
        self.assertEqual(lock_changed["model"]["fingerprint"], baseline["model"]["fingerprint"])

        runtime_changed = _copy(manifest)
        runtime_changed["sources"]["chatterbox_package"]["artifact_sha256"] = "b" * 64
        runtime_identities = build_identity_contract(runtime_changed, "a" * 64)
        self.assertNotEqual(runtime_identities["runtime"]["fingerprint"], baseline["runtime"]["fingerprint"])
        self.assertEqual(runtime_identities["model"]["fingerprint"], baseline["model"]["fingerprint"])
        self.assertNotEqual(runtime_identities["activation"]["fingerprint"], baseline["activation"]["fingerprint"])

        model_changed = _copy(manifest)
        model_changed["sources"]["model"]["files"][0]["sha256"] = "c" * 64
        model_identities = build_identity_contract(model_changed, "a" * 64)
        self.assertEqual(model_identities["runtime"]["fingerprint"], baseline["runtime"]["fingerprint"])
        self.assertNotEqual(model_identities["model"]["fingerprint"], baseline["model"]["fingerprint"])
        self.assertNotEqual(model_identities["activation"]["fingerprint"], baseline["activation"]["fingerprint"])

        v3 = _copy(manifest)
        v3["product"]["profile"] = "chatterbox-multilingual-v3-cpu"
        v3["sources"]["model"]["variant"] = "multilingual-v3"
        for record in v3["sources"]["model"]["files"]:
            if record["path"] == "t3_mtl23ls_v2.safetensors":
                record["path"] = "t3_mtl23ls_v3.safetensors"
                break
        v3_identities = build_identity_contract(v3, "a" * 64)
        self.assertEqual(v3_identities["runtime"]["fingerprint"], baseline["runtime"]["fingerprint"])
        self.assertNotEqual(v3_identities["model"]["fingerprint"], baseline["model"]["fingerprint"])
        self.assertNotEqual(v3_identities["activation"]["fingerprint"], baseline["activation"]["fingerprint"])
        self.assertTrue(v3_identities["model"]["fingerprint"].startswith("chatterbox-mtl-v3-"))
        self.assertTrue(v3_identities["activation"]["fingerprint"].startswith("chatterbox-v3-cpu-"))

    def test_receipt_identity_links_cross_validate_without_claiming_readiness(self) -> None:
        identities = build_identity_contract(_manifest("a" * 64), "a" * 64)
        contracts = receipt_identity_contracts(identities)
        valid = validate_receipt_identity_links(identities, contracts["runtime"], contracts["model"], contracts["activation"])
        self.assertTrue(valid["identity_links_valid"], valid)
        self.assertNotIn("ready", valid)
        self.assertTrue(all("status" not in contract for contract in contracts.values()))

        stale_model = dict(contracts["model"])
        stale_model["model_fingerprint"] = "stale-model"
        rejected = validate_receipt_identity_links(identities, contracts["runtime"], stale_model, contracts["activation"])
        self.assertFalse(rejected["identity_links_valid"])
        self.assertIn("model receipt model_fingerprint does not match", rejected["errors"])

        stale_activation = dict(contracts["activation"])
        stale_activation["runtime_fingerprint"] = "stale-runtime"
        rejected = validate_receipt_identity_links(identities, contracts["runtime"], contracts["model"], stale_activation)
        self.assertFalse(rejected["identity_links_valid"])
        self.assertIn("activation receipt runtime_fingerprint does not match", rejected["errors"])

        missing_identity = validate_receipt_identity_links({}, contracts["runtime"], contracts["model"], contracts["activation"])
        self.assertFalse(missing_identity["identity_links_valid"])
        self.assertIn("runtime identity fingerprint is missing", missing_identity["errors"])

    def test_verified_file_credit_rejects_stale_missing_and_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.bin"
            stale = root / "stale.bin"
            good.write_bytes(b"good-data")
            stale.write_bytes(b"stale-dat")
            link = root / "link.bin"
            link.symlink_to(good)
            files = [
                {"path": "good.bin", "size_bytes": good.stat().st_size, "sha256": hashlib.sha256(good.read_bytes()).hexdigest()},
                {"path": "stale.bin", "size_bytes": stale.stat().st_size, "sha256": "f" * 64},
                {"path": "missing.bin", "size_bytes": 1, "sha256": "e" * 64},
                {"path": "link.bin", "size_bytes": good.stat().st_size, "sha256": hashlib.sha256(good.read_bytes()).hexdigest()},
            ]
            result = verified_file_credit(root, files)
            self.assertEqual(result["credited_bytes"], good.stat().st_size)
            self.assertEqual(result["verified"], ["good.bin"])
            reasons = {item["path"]: item["reason"] for item in result["rejected"]}
            self.assertEqual(reasons["stale.bin"], "sha256_mismatch")
            self.assertEqual(reasons["missing.bin"], "missing_or_not_regular")
            self.assertEqual(reasons["link.bin"], "missing_or_not_regular")

    def test_storage_aggregation_uses_phase_peak_per_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest["storage"] = {
                "contract_version": "1.0.0",
                "phases": [
                    {
                        "name": "one",
                        "buckets": [
                            {"name": "env-one", "destination": "environment", "required_bytes": 100},
                            {"name": "model-one", "destination": "verified_model_root", "required_bytes": 50},
                        ],
                    },
                    {
                        "name": "two",
                        "buckets": [
                            {"name": "env-two", "destination": "environment", "required_bytes": 80},
                            {"name": "model-two", "destination": "verified_model_root", "required_bytes": 100},
                        ],
                    },
                ],
            }
            runtime_dir, manifest = _write_runtime(root, manifest)
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={"XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state"), "E2A_ROOT": str(root / "e2a")},
            )

            same = calculate_storage_plan(
                paths,
                manifest,
                filesystem_resolver=lambda path: ("same", root),
                free_bytes_provider=lambda path: 1_000,
            )
            self.assertTrue(same["ok"], same)
            self.assertEqual(same["filesystems"]["same"]["required_bytes"], 180)

            insufficient = calculate_storage_plan(
                paths,
                manifest,
                filesystem_resolver=lambda path: ("same", root),
                free_bytes_provider=lambda path: 170,
            )
            self.assertEqual(insufficient["status"], "insufficient_storage")
            self.assertEqual(insufficient["filesystems"]["same"]["shortfall_bytes"], 10)

            split = calculate_storage_plan(
                paths,
                manifest,
                filesystem_resolver=lambda path: (("environment" if path == paths.environment else "model"), root),
                free_bytes_provider=lambda path: 1_000,
            )
            self.assertTrue(split["ok"], split)
            self.assertEqual(split["filesystems"]["environment"]["required_bytes"], 100)
            self.assertEqual(split["filesystems"]["model"]["required_bytes"], 100)

    def test_runtime_preflight_phase_does_not_require_model_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest["storage"] = {
                "contract_version": "1.0.0",
                "phases": [
                    {
                        "name": "runtime_artifact_acquisition",
                        "buckets": [{"name": "runtime", "destination": "environment", "required_bytes": 100}],
                    },
                    {
                        "name": "runtime_environment_construction",
                        "buckets": [{"name": "environment", "destination": "environment", "required_bytes": 200}],
                    },
                    {
                        "name": "model_acquisition",
                        "buckets": [
                            {
                                "name": "model-staging",
                                "destination": "verified_model_root",
                                "required_bytes": None,
                                "measurement_status": "required_clean_disposable",
                            }
                        ],
                    },
                ],
            }
            runtime_dir, manifest = _write_runtime(root, manifest)
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={"XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state"), "E2A_ROOT": str(root / "e2a")},
            )
            runtime_only = calculate_storage_plan(
                paths,
                manifest,
                phases=("runtime_artifact_acquisition", "runtime_environment_construction"),
                filesystem_resolver=lambda path: ("runtime", root),
                free_bytes_provider=lambda path: 1_000,
            )
            self.assertEqual(runtime_only["status"], "sufficient")
            self.assertEqual(runtime_only["selected_phases"], ["runtime_artifact_acquisition", "runtime_environment_construction"])
            self.assertEqual(calculate_storage_plan(paths, manifest, filesystem_resolver=lambda path: ("all", root))["status"], "storage_budget_unknown")

            report = preflight(
                paths,
                Path(sys.executable),
                network_checker=lambda url: (True, "synthetic reachable"),
            )
            self.assertEqual(report["checks"]["storage"]["status"], "sufficient")
            self.assertEqual(
                report["checks"]["storage"]["selected_phases"],
                ["runtime_artifact_acquisition", "runtime_environment_construction"],
            )

    def test_measurement_preflight_allows_unknown_budget_only_inside_marked_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory:
            root = Path(directory).resolve()
            (root / ".chatterbox-measurement-root").write_text(
                "observation-only-v1\n", encoding="utf-8"
            )
            outside = Path(outside_directory).resolve()
            (outside / runtime_module.DISPOSABLE_MARKER).write_text(
                runtime_module.DISPOSABLE_MARKER_CONTENT,
                encoding="utf-8",
            )
            runtime_dir, _ = _write_runtime(root, _manifest())
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_STATE_HOME": str(root / "state"),
                    "E2A_ROOT": str(root / "e2a"),
                    "E2A_MODELS_DIR": str(root / "models"),
                    "E2A_RUN_DIR": str(root / "run"),
                },
            )
            unknown = {
                "ok": False,
                "status": "storage_budget_unknown",
                "errors": [],
                "filesystems": {},
            }
            with mock.patch.object(runtime_module, "_interpreter_checks", return_value=({"ok": True}, [])), mock.patch.object(
                runtime_module,
                "verify_lock",
                return_value={"ok": True, "errors": []},
            ), mock.patch.object(runtime_module, "calculate_storage_plan", return_value=unknown):
                measurement = preflight(
                    paths,
                    Path(sys.executable),
                    measurement_root=root,
                    check_network=True,
                )
                ordinary = preflight(paths, Path(sys.executable), check_network=True)

            self.assertTrue(measurement["ok"], measurement)
            self.assertEqual(
                measurement["checks"]["network"]["reason"],
                "disposable_measurement_uses_verified_offline_inputs",
            )
            self.assertFalse(ordinary["ok"])
            self.assertTrue(any("storage_budget_unknown" in error for error in ordinary["errors"]))
            with self.assertRaisesRegex(RuntimeConfigurationError, "escapes|owned"):
                validate_measurement_scope(paths, outside)

    def test_legacy_fingerprinted_paths_are_ignored_and_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_dir, manifest = _write_runtime(root, _manifest())
            data = root / "data"
            state = root / "state"
            legacy_environment = data / "ebook2audiobook/chatterbox/envs/py311-linux-x86_64-cpu-legacy0000000000"
            legacy_environment.mkdir(parents=True)
            legacy_marker = legacy_environment / "preserve.txt"
            legacy_marker.write_text("legacy environment", encoding="utf-8")
            legacy_result = state / "ebook2audiobook/chatterbox/installation-result-py311-linux-x86_64-cpu-legacy0000000000.json"
            legacy_result.parent.mkdir(parents=True)
            legacy_result.write_text('{"legacy": true}\n', encoding="utf-8")
            before_environment = legacy_marker.read_bytes()
            before_result = legacy_result.read_bytes()

            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={"XDG_DATA_HOME": str(data), "XDG_STATE_HOME": str(state), "E2A_ROOT": str(root / "e2a")},
            )
            self.assertNotEqual(paths.environment, legacy_environment)
            self.assertNotEqual(paths.result_path, legacy_result)
            report = preflight(paths, Path(sys.executable), network_checker=lambda url: (True, "synthetic reachable"))
            self.assertFalse(any("environment path already exists" in error for error in report["errors"]))
            self.assertFalse(any("installation result already exists" in error for error in report["errors"]))
            self.assertEqual(legacy_marker.read_bytes(), before_environment)
            self.assertEqual(legacy_result.read_bytes(), before_result)
            self.assertFalse(paths.environment.exists())
            self.assertFalse(paths.result_path.exists())

    def test_runtime_receipt_is_distinct_and_legacy_result_never_means_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _install_fixture(Path(directory))
            self.assertNotEqual(paths.runtime_receipt_path, paths.result_path)
            paths.result_path.parent.mkdir(parents=True)
            paths.result_path.write_text('{"status":"passed"}\n', encoding="utf-8")
            status = runtime_status(paths)
            self.assertEqual(status["status"], "missing")
            self.assertTrue(status["legacy_result_ignored"])
            self.assertFalse(paths.runtime_receipt_path.exists())

    def test_atomic_runtime_install_publishes_artifact_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker = _install_fixture(root)
            runner = _SyntheticRunner()
            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                result = install_runtime(
                    paths,
                    Path(sys.executable),
                    worker_script=worker,
                    command_runner=runner,
                    import_hook=lambda python, modules: (True, "imports passed"),
                    self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                    nonce_factory=lambda: "transaction-success",
                )
            self.assertEqual(result["status"], "ready")
            candidate = Path(result["environment"])
            self.assertTrue(candidate.is_dir())
            self.assertTrue(candidate.is_relative_to(paths.runtime_objects_dir))
            self.assertEqual(candidate.stat().st_dev, paths.env_base.stat().st_dev)
            self.assertTrue(paths.runtime_receipt_path.is_file())
            self.assertEqual(paths.runtime_receipt_path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(paths.result_path.exists())
            status = runtime_status(paths)
            self.assertTrue(status["ok"], status)
            self.assertEqual(status["environment"], str(candidate))

            rebuilt = build_paths(
                runtime_dir=paths.runtime_dir,
                repo_root=root,
                environment={
                    "XDG_DATA_HOME": str(paths.data_home),
                    "XDG_STATE_HOME": str(paths.state_home),
                    "E2A_ROOT": str(paths.e2a_root),
                },
            )
            self.assertEqual(readiness_snapshot(rebuilt)["paths"].environment, candidate)
            receipt = json.loads(paths.runtime_receipt_path.read_text(encoding="utf-8"))
            owner_marker = candidate / ".e2a-chatterbox-runtime-owner.json"
            owner = json.loads(owner_marker.read_text(encoding="utf-8"))
            self.assertEqual(owner_marker.stat().st_mode & 0o777, 0o600)
            self.assertEqual(owner["nonce"], "transaction-success")
            self.assertEqual(owner["candidate"], str(candidate))
            self.assertEqual(receipt["artifact"]["path"], str(candidate))
            self.assertEqual(receipt["runtime_fingerprint"], paths.runtime_fingerprint)
            self.assertEqual(receipt["manifest_sha256"], paths.manifest_sha256)
            self.assertEqual(receipt["lock_sha256"], paths.lock_sha256)

    def test_disposable_runtime_install_uses_local_wheelhouse_and_measurement_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".chatterbox-measurement-root").write_text(
                "observation-only-v1\n", encoding="utf-8"
            )
            paths, worker = _install_fixture(root)
            wheelhouse = root / "wheelhouse"
            pip_cache = root / "pip-cache"
            wheelhouse.mkdir()
            preflight_calls: list[dict] = []

            def measurement_preflight(_paths, _interpreter, **kwargs):
                preflight_calls.append(kwargs)
                return {"ok": True}

            runner = _SyntheticRunner()
            with mock.patch.object(
                runtime_module,
                "preflight",
                side_effect=measurement_preflight,
            ):
                result = install_runtime(
                    paths,
                    Path(sys.executable),
                    worker_script=worker,
                    measurement_root=root,
                    wheelhouse=wheelhouse,
                    pip_cache=pip_cache,
                    command_runner=runner,
                    import_hook=lambda python, modules: (True, "imports passed"),
                    self_test_hook=lambda python, args, env: (True, "self-test passed"),
                    nonce_factory=lambda: "measurement-runtime",
                )

            self.assertTrue(result["ok"], result)
            self.assertEqual(len(preflight_calls), 2)
            self.assertTrue(all(call["measurement_root"] == root for call in preflight_calls))
            self.assertTrue(all(call["check_network"] is False for call in preflight_calls))
            pip_commands = [call for call in runner.calls if "install" in call]
            self.assertEqual(len(pip_commands), 1)
            self.assertIn("--no-index", pip_commands[0])
            self.assertIn("--find-links", pip_commands[0])
            self.assertIn(str(wheelhouse), pip_commands[0])
            self.assertIn("--cache-dir", pip_commands[0])
            self.assertIn(str(pip_cache), pip_commands[0])

    def test_venv_python_symlink_to_requested_interpreter_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            runner = _SyntheticRunner(python_symlink=True)
            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True, "errors": [], "checks": {}}):
                result = install_runtime(
                    paths,
                    Path(sys.executable),
                    worker_script=worker,
                    command_runner=runner,
                    import_hook=lambda python, modules: (True, "imports passed"),
                    self_test_hook=lambda python, args, env: (True, "self-test passed"),
                    nonce_factory=lambda: "symlink-python",
                )
            self.assertEqual(result["status"], "ready")
            self.assertTrue((Path(result["environment"]) / "bin/python").is_symlink())

    def test_unknown_runtime_capacity_stops_before_persistent_installer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            blocked = {
                "ok": False,
                "errors": ["storage_budget_unknown: measurements required"],
                "checks": {
                    "storage": {"ok": False, "status": "storage_budget_unknown"},
                    "network": {
                        "required": True,
                        "attempted": False,
                        "status": "skipped",
                        "reason": "storage_budget_unknown",
                        "results": {},
                        "ok": False,
                    },
                },
            }
            runner = _SyntheticRunner()
            with mock.patch(
                "components.Chatterbox.runtime.runtime.preflight",
                return_value=blocked,
            ) as checked:
                with self.assertRaisesRegex(ProvisioningError, "storage_budget_unknown"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=runner,
                    )
            self.assertEqual(checked.call_count, 1)
            self.assertFalse(paths.state_dir.exists())
            self.assertFalse(paths.install_lock_path.exists())
            self.assertFalse(paths.runtime_objects_dir.exists())
            self.assertEqual(runner.calls, [])

    def test_runtime_capacity_is_rechecked_under_lock_before_candidate_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            allowed = {"ok": True, "errors": [], "checks": {}}
            blocked = {
                "ok": False,
                "errors": ["storage_budget_unknown: measurements required"],
                "checks": {"storage": {"ok": False, "status": "storage_budget_unknown"}},
            }
            runner = _SyntheticRunner()
            with mock.patch(
                "components.Chatterbox.runtime.runtime.preflight",
                side_effect=(allowed, blocked),
            ) as checked:
                with self.assertRaisesRegex(ProvisioningError, "storage_budget_unknown"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=runner,
                    )
            self.assertEqual(checked.call_count, 2)
            self.assertFalse(paths.runtime_objects_dir.exists())
            self.assertFalse(paths.runtime_receipt_path.exists())
            self.assertEqual(runner.calls, [])

    def test_failure_injection_cleans_only_current_owned_candidate_and_retry_succeeds(self) -> None:
        cases = ("venv", "install", "pip_check", "imports", "self_test", "receipt_validation", "receipt_publication")
        for failure in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                paths, worker = _install_fixture(Path(directory))
                runner = _SyntheticRunner(failure if failure in {"venv", "install", "pip_check"} else None)
                import_hook = lambda python, modules: (failure != "imports", "import hook failed" if failure == "imports" else "imports passed")
                self_test_hook = lambda python, args, env: (
                    failure != "self_test",
                    "self-test hook failed" if failure == "self_test" else "self-test hook passed",
                )

                def receipt_validator(runtime_paths, receipt):
                    if failure == "receipt_validation":
                        return {"ok": False, "status": "repair_required", "errors": ["injected receipt validation failure"]}
                    from components.Chatterbox.runtime.runtime import validate_runtime_receipt
                    return validate_runtime_receipt(runtime_paths, receipt)

                def receipt_writer(runtime_paths, receipt):
                    if failure == "receipt_publication":
                        raise OSError("injected receipt publication failure")
                    from components.Chatterbox.runtime.runtime import write_runtime_receipt
                    return write_runtime_receipt(runtime_paths, receipt)

                with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                    with self.assertRaises((OSError, RuntimeError)):
                        install_runtime(
                            paths,
                            Path(sys.executable),
                            worker_script=worker,
                            command_runner=runner,
                            import_hook=import_hook,
                            self_test_hook=self_test_hook,
                            receipt_validator=receipt_validator,
                            receipt_writer=receipt_writer,
                            nonce_factory=lambda: "transaction-failure",
                        )
                self.assertFalse(paths.runtime_receipt_path.exists())
                self.assertEqual(list(paths.runtime_objects_dir.iterdir()), [])
                self.assertEqual(runtime_status(paths)["status"], "missing")

                with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                    retried = install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                        import_hook=lambda python, modules: (True, "imports passed"),
                        self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                        nonce_factory=lambda: "transaction-retry",
                    )
                self.assertEqual(retried["status"], "ready")

    def test_owner_marker_failure_rolls_back_only_new_empty_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            with (
                mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}),
                mock.patch("components.Chatterbox.runtime.runtime._write_json_file", side_effect=OSError("injected marker failure")),
            ):
                with self.assertRaisesRegex(OSError, "injected marker failure"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                        nonce_factory=lambda: "transaction-marker-failure",
                    )
            self.assertTrue(paths.runtime_objects_dir.is_dir())
            self.assertEqual(list(paths.runtime_objects_dir.iterdir()), [])
            self.assertFalse(paths.runtime_receipt_path.exists())

    def test_candidate_parent_fsync_failure_rolls_back_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            real_fsync_directory = runtime_module._fsync_directory
            failed = False

            def fail_first_candidate_parent_fsync(path):
                nonlocal failed
                if Path(path) == paths.runtime_objects_dir and not failed:
                    failed = True
                    raise OSError("injected candidate parent fsync failure")
                return real_fsync_directory(path)

            with (
                mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}),
                mock.patch(
                    "components.Chatterbox.runtime.runtime._fsync_directory",
                    side_effect=fail_first_candidate_parent_fsync,
                ),
            ):
                with self.assertRaisesRegex(OSError, "injected candidate parent fsync failure"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                        nonce_factory=lambda: "transaction-parent-fsync-failure",
                    )
            self.assertTrue(paths.runtime_objects_dir.is_dir())
            self.assertEqual(list(paths.runtime_objects_dir.iterdir()), [])
            self.assertEqual(runtime_status(paths)["status"], "missing")

            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                retried = install_runtime(
                    paths,
                    Path(sys.executable),
                    worker_script=worker,
                    command_runner=_SyntheticRunner(),
                    import_hook=lambda python, modules: (True, "imports passed"),
                    self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                    nonce_factory=lambda: "transaction-parent-fsync-retry",
                )
            self.assertEqual(retried["status"], "ready")

    def test_lock_open_and_flock_failures_stop_before_candidate_reservation(self) -> None:
        for failure in ("open", "flock"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                paths, worker = _install_fixture(Path(directory))
                if failure == "open":
                    real_open = runtime_module.os.open

                    def fail_lock_open(path, *args, **kwargs):
                        if Path(path) == paths.install_lock_path:
                            raise OSError("injected lock open failure")
                        return real_open(path, *args, **kwargs)

                    patches = [mock.patch("components.Chatterbox.runtime.runtime.os.open", side_effect=fail_lock_open)]
                    expected = "injected lock open failure"
                else:
                    if runtime_module.fcntl is None:
                        self.skipTest("fcntl is unavailable")
                    patches = [
                        mock.patch(
                            "components.Chatterbox.runtime.runtime.fcntl.flock",
                            side_effect=OSError("injected flock failure"),
                        )
                    ]
                    expected = "injected flock failure"

                with patches[0], mock.patch(
                    "components.Chatterbox.runtime.runtime.preflight",
                    return_value={"ok": True},
                ):
                    with self.assertRaisesRegex(OSError, expected):
                        install_runtime(
                            paths,
                            Path(sys.executable),
                            worker_script=worker,
                            command_runner=_SyntheticRunner(),
                        )
                self.assertFalse(paths.runtime_objects_dir.exists())
                self.assertFalse(paths.runtime_receipt_path.exists())

    def test_cleanup_failure_is_reported_and_preserved_as_repair_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            with (
                mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}),
                mock.patch(
                    "components.Chatterbox.runtime.runtime.shutil.rmtree",
                    side_effect=OSError("injected cleanup failure"),
                ),
            ):
                with self.assertRaisesRegex(ProvisioningError, "owned candidate cleanup is incomplete"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner("venv"),
                        nonce_factory=lambda: "transaction-cleanup-failure",
                    )
            candidates = list(paths.runtime_objects_dir.iterdir())
            self.assertEqual(len(candidates), 1)
            self.assertTrue((candidates[0] / ".e2a-chatterbox-runtime-owner.json").is_file())
            self.assertEqual(runtime_status(paths)["status"], "repair_required")

    def test_receipt_directory_fsync_failure_never_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            real_fsync_directory = runtime_module._fsync_directory
            state_fsync_calls = 0

            def fail_post_publication_fsync(path):
                nonlocal state_fsync_calls
                if Path(path) == paths.state_dir:
                    state_fsync_calls += 1
                    if state_fsync_calls == 2:
                        raise OSError("injected post-publication directory fsync failure")
                return real_fsync_directory(path)

            with (
                mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}),
                mock.patch(
                    "components.Chatterbox.runtime.runtime._fsync_directory",
                    side_effect=fail_post_publication_fsync,
                ),
            ):
                with self.assertRaisesRegex(OSError, "injected post-publication directory fsync failure"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                        import_hook=lambda python, modules: (True, "imports passed"),
                        self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                        nonce_factory=lambda: "transaction-receipt-fsync-failure",
                    )
            self.assertTrue(paths.runtime_receipt_path.exists())
            status = runtime_status(paths)
            self.assertFalse(status["ok"])
            self.assertEqual(status["status"], "repair_required")
            self.assertTrue(any(path.endswith(".publishing") for path in status["ambiguous_paths"]))
            self.assertEqual(len(list(paths.runtime_objects_dir.iterdir())), 1)

    def test_runtime_receipt_uncertainty_stays_fail_closed_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            publication = paths.runtime_receipt_path.with_name(f"{paths.runtime_receipt_path.name}.publishing")
            published = paths.runtime_receipt_path.with_name(f"{paths.runtime_receipt_path.name}.published")
            real_fsync_directory = runtime_module._fsync_directory
            real_replace = runtime_module.os.replace
            real_link = runtime_module.os.link

            def fail_final_fsync(path):
                if Path(path) == paths.state_dir and published.exists():
                    if publication.exists():
                        raise OSError("injected runtime rollback fsync failure")
                    raise OSError("injected final runtime receipt fsync failure")
                return real_fsync_directory(path)

            def fail_rollback_replace(source, destination):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected runtime rollback replace failure")
                return real_replace(source, destination)

            def fail_rollback_link(source, destination, *args, **kwargs):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected runtime rollback link failure")
                return real_link(source, destination, *args, **kwargs)

            with (
                mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}),
                mock.patch("components.Chatterbox.runtime.runtime._fsync_directory", side_effect=fail_final_fsync),
                mock.patch("components.Chatterbox.runtime.runtime.os.replace", side_effect=fail_rollback_replace),
                mock.patch("components.Chatterbox.runtime.runtime.os.link", side_effect=fail_rollback_link),
            ):
                with self.assertRaisesRegex(OSError, "final runtime receipt fsync failure") as raised:
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                        import_hook=lambda python, modules: (True, "imports passed"),
                        self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                        nonce_factory=lambda: "transaction-runtime-marker-failure",
                    )
            notes = " ".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn("runtime receipt publication rollback rename failed", notes)
            self.assertIn("runtime receipt publication quarantine link failed", notes)
            self.assertIn("runtime receipt publication quarantine could not be restored", notes)
            self.assertTrue(paths.runtime_receipt_path.exists())
            self.assertFalse(publication.exists())
            self.assertTrue(published.exists())
            self.assertTrue(
                paths.runtime_receipt_path.with_name(
                    f"{paths.runtime_receipt_path.name}.ambiguous"
                ).exists()
            )
            self.assertEqual(runtime_status(paths)["status"], "repair_required")
            fresh = _fresh_process_statuses(paths)
            self.assertEqual(fresh["runtime"]["status"], "repair_required")
            self.assertEqual(
                fresh["derived_paths"]["environment"],
                str(paths.runtime_objects_dir / ".unpublished"),
            )

    def test_runtime_receipt_combined_rollback_failures_stay_quarantined_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _install_fixture(Path(directory))
            self._assert_combined_receipt_rollback_failure_is_persistent(
                paths,
                paths.runtime_receipt_path,
                "runtime",
                lambda: write_runtime_receipt(paths, {"status": "ready"}),
                runtime_status,
            )

    def test_direct_receipt_writers_cannot_overwrite_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _install_fixture(Path(directory))
            barrier = threading.Barrier(2)
            successes: list[str] = []
            failures: list[BaseException] = []

            def publish(label: str) -> None:
                try:
                    barrier.wait(timeout=5)
                    write_runtime_receipt(paths, {"writer": label})
                    successes.append(label)
                except BaseException as exc:
                    failures.append(exc)

            first = threading.Thread(target=publish, args=("first",))
            second = threading.Thread(target=publish, args=("second",))
            first.start()
            second.start()
            first.join(timeout=10)
            second.join(timeout=10)

            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], ProvisioningError)
            payload = json.loads(paths.runtime_receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"writer": successes[0]})
            self.assertFalse(paths.runtime_receipt_path.with_name(f"{paths.runtime_receipt_path.name}.publishing").exists())

    def test_unowned_or_ambiguous_runtime_paths_are_preserved_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            unowned = paths.runtime_objects_dir / "foreign-candidate"
            unowned.mkdir(parents=True)
            marker = unowned / "preserve.txt"
            marker.write_text("user state", encoding="utf-8")
            before = marker.read_bytes()
            status = runtime_status(paths)
            self.assertEqual(status["status"], "repair_required")
            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                with self.assertRaisesRegex(RuntimeError, "repair_required"):
                    install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=_SyntheticRunner(),
                    )
            self.assertEqual(marker.read_bytes(), before)
            self.assertFalse(paths.runtime_receipt_path.exists())

    def test_runtime_receipt_tampering_reports_repair_required_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                result = install_runtime(
                    paths,
                    Path(sys.executable),
                    worker_script=worker,
                    command_runner=_SyntheticRunner(),
                    import_hook=lambda python, modules: (True, "imports passed"),
                    self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                    nonce_factory=lambda: "transaction-tamper",
                )
            candidate = Path(result["environment"])
            receipt = json.loads(paths.runtime_receipt_path.read_text(encoding="utf-8"))
            receipt["lock_sha256"] = "f" * 64
            paths.runtime_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            status = runtime_status(paths)
            self.assertEqual(status["status"], "repair_required")
            self.assertTrue(candidate.exists())

    def test_concurrent_installers_serialize_and_second_reuses_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker = _install_fixture(Path(directory))
            entered = threading.Event()
            release = threading.Event()
            first_runner = _SyntheticRunner(entered=entered, release=release)
            results: list[dict] = []
            failures: list[BaseException] = []

            def invoke(runner, nonce):
                try:
                    result = install_runtime(
                        paths,
                        Path(sys.executable),
                        worker_script=worker,
                        command_runner=runner,
                        import_hook=lambda python, modules: (True, "imports passed"),
                        self_test_hook=lambda python, args, env: (True, "self-test hook passed"),
                        nonce_factory=lambda: nonce,
                    )
                    results.append(result)
                except BaseException as exc:
                    failures.append(exc)

            with mock.patch("components.Chatterbox.runtime.runtime.preflight", return_value={"ok": True}):
                first = threading.Thread(target=invoke, args=(first_runner, "transaction-concurrent-one"))
                second_runner = _SyntheticRunner()
                second = threading.Thread(target=invoke, args=(second_runner, "transaction-concurrent-two"))
                first.start()
                self.assertTrue(entered.wait(timeout=5))
                second.start()
                release.set()
                first.join(timeout=10)
                second.join(timeout=10)

            self.assertFalse(failures, failures)
            self.assertEqual(sorted(item["status"] for item in results), ["already_ready", "ready"])
            self.assertEqual(second_runner.calls, [])
            self.assertEqual(len(list(paths.runtime_objects_dir.iterdir())), 1)

    def test_unknown_model_capacity_stops_before_model_lock_or_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker, _content = _model_fixture(Path(directory))
            self.assertFalse(paths.model_install_lock_path.exists())
            blocked = {
                "ok": False,
                "errors": ["storage_budget_unknown: measurements required"],
                "checks": {"storage": {"ok": False, "status": "storage_budget_unknown"}},
            }
            with self.assertRaisesRegex(ProvisioningError, "storage_budget_unknown"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: blocked,
                    download_file=lambda *_args: (_ for _ in ()).throw(
                        AssertionError("download must not start")
                    ),
                    activation_hook=lambda *_args: (_ for _ in ()).throw(
                        AssertionError("activation must not start")
                    ),
                )
            self.assertFalse(paths.model_install_lock_path.exists())
            self.assertFalse(paths.model_objects_dir.exists())
            self.assertFalse(paths.model_receipt_path.exists())
            self.assertFalse(paths.activation_receipt_path.exists())

    def test_model_acquisition_publishes_distinct_receipts_and_canonical_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)
            self.assertEqual(product_status(paths)["status"], "model_missing")
            downloads: list[tuple[str, Path]] = []

            def download(url: str, destination: Path) -> None:
                downloads.append((url, destination))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            result = acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda python, args, env: (True, "local model load passed"),
                nonce_factory=lambda: "model-transaction",
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(len(downloads), 6)
            self.assertIn("/resolve/" + "4" * 40 + "/ve.pt", downloads[0][0])
            self.assertTrue(paths.model_receipt_path.is_file())
            self.assertTrue(paths.activation_receipt_path.is_file())
            self.assertNotEqual(paths.runtime_receipt_path, paths.model_receipt_path)
            self.assertNotEqual(paths.model_receipt_path, paths.activation_receipt_path)
            model = model_status(paths)
            snapshot = Path(model["snapshot"])
            self.assertEqual(snapshot.name, "snapshot")
            self.assertEqual(snapshot.parent.parent, paths.model_objects_dir)
            self.assertFalse(snapshot.is_relative_to(paths.legacy_model_cache))
            self.assertEqual((snapshot / "ve.pt").read_bytes(), content)
            self.assertTrue(activation_status(paths)["ok"])

            rebuilt = build_paths(
                runtime_dir=paths.runtime_dir,
                repo_root=root,
                environment={
                    "XDG_DATA_HOME": str(paths.data_home),
                    "XDG_STATE_HOME": str(paths.state_home),
                    "E2A_ROOT": str(paths.e2a_root),
                },
            )
            self.assertIsNone(rebuilt.verified_model_root)
            self.assertEqual(rebuilt.model_namespace, paths.model_objects_dir / ".unpublished/snapshot")
            ready_snapshot = readiness_snapshot(rebuilt)
            selected = ready_snapshot["paths"]
            self.assertEqual(selected.verified_model_root, snapshot)
            self.assertEqual(selected.model_namespace, snapshot)
            worker_env = sanitized_worker_environment(selected, {"HF_TOKEN": "secret"})
            self.assertEqual(worker_env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(worker_env["TRANSFORMERS_OFFLINE"], "1")
            self.assertNotIn("HF_TOKEN", worker_env)

    def test_nonregular_ambiguity_markers_block_receipt_derived_path_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda python, args, env: (True, "local model load passed"),
                nonce_factory=lambda: "nonregular-ambiguity-selection",
            )
            environment = {
                "XDG_DATA_HOME": str(paths.data_home),
                "XDG_STATE_HOME": str(paths.state_home),
                "E2A_ROOT": str(paths.e2a_root),
            }
            ready = build_paths(
                runtime_dir=paths.runtime_dir,
                repo_root=root,
                environment=environment,
            )
            self.assertEqual(ready.environment, paths.runtime_objects_dir / ".unpublished")
            self.assertIsNone(ready.verified_model_root)

            selections = (
                ("runtime", paths.runtime_receipt_path),
                ("model", paths.model_receipt_path),
            )
            for marker_kind in ("dangling_symlink", "special_file"):
                for label, receipt_path in selections:
                    with self.subTest(marker_kind=marker_kind, label=label):
                        ambiguity = self._create_nonregular_ambiguity_marker(
                            receipt_path, marker_kind
                        )
                        rebuilt = build_paths(
                            runtime_dir=paths.runtime_dir,
                            repo_root=root,
                            environment=environment,
                        )
                        if label == "runtime":
                            self.assertEqual(
                                rebuilt.environment,
                                paths.runtime_objects_dir / ".unpublished",
                            )
                        else:
                            self.assertEqual(
                                rebuilt.model_namespace,
                                paths.model_objects_dir / ".unpublished/snapshot",
                            )
                            self.assertIsNone(rebuilt.verified_model_root)
                        snapshot = readiness_snapshot(rebuilt)
                        self.assertEqual(snapshot["status"], "repair_required")
                        if label == "model":
                            self.assertIsNone(snapshot["paths"].verified_model_root)
                        else:
                            self.assertIsNotNone(snapshot["paths"].verified_model_root)
                        ambiguity.unlink()

    def test_model_receipt_uncertainty_stays_fail_closed_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)
            publication = paths.model_receipt_path.with_name(f"{paths.model_receipt_path.name}.publishing")
            published = paths.model_receipt_path.with_name(f"{paths.model_receipt_path.name}.published")
            real_fsync_directory = runtime_module._fsync_directory
            real_replace = runtime_module.os.replace
            real_link = runtime_module.os.link

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            def fail_final_fsync(path):
                if Path(path) == paths.state_dir and published.exists():
                    if publication.exists():
                        raise OSError("injected model rollback fsync failure")
                    raise OSError("injected final model receipt fsync failure")
                return real_fsync_directory(path)

            def fail_rollback_replace(source, destination):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected model rollback replace failure")
                return real_replace(source, destination)

            def fail_rollback_link(source, destination, *args, **kwargs):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected model rollback link failure")
                return real_link(source, destination, *args, **kwargs)

            with (
                mock.patch("components.Chatterbox.runtime.runtime._fsync_directory", side_effect=fail_final_fsync),
                mock.patch("components.Chatterbox.runtime.runtime.os.replace", side_effect=fail_rollback_replace),
                mock.patch("components.Chatterbox.runtime.runtime.os.link", side_effect=fail_rollback_link),
            ):
                with self.assertRaisesRegex(OSError, "final model receipt fsync failure") as raised:
                    acquire_model(
                        paths,
                        worker_script=worker,
                        preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                        download_file=download,
                        activation_hook=lambda python, args, env: (True, "passed"),
                        nonce_factory=lambda: "model-marker-failure",
                    )
            notes = " ".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn("model receipt publication rollback rename failed", notes)
            self.assertIn("model receipt publication quarantine link failed", notes)
            self.assertIn("model receipt publication quarantine could not be restored", notes)
            self.assertTrue(paths.model_receipt_path.exists())
            self.assertFalse(publication.exists())
            self.assertTrue(published.exists())
            self.assertTrue(
                paths.model_receipt_path.with_name(
                    f"{paths.model_receipt_path.name}.ambiguous"
                ).exists()
            )
            self.assertEqual(model_status(paths)["status"], "repair_required")
            self.assertEqual(product_status(paths)["status"], "repair_required")
            fresh = _fresh_process_statuses(paths)
            self.assertEqual(fresh["model"]["status"], "repair_required")
            self.assertEqual(
                fresh["derived_paths"]["model_namespace"],
                str(paths.model_objects_dir / ".unpublished/snapshot"),
            )
            self.assertIsNone(fresh["derived_paths"]["verified_model_root"])

    def test_model_receipt_combined_rollback_failures_stay_quarantined_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _install_fixture(Path(directory))
            self._assert_combined_receipt_rollback_failure_is_persistent(
                paths,
                paths.model_receipt_path,
                "model",
                lambda: runtime_module.write_model_receipt(paths, {"status": "ready"}),
                model_status,
            )

    def test_activation_receipt_uncertainty_stays_fail_closed_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            with self.assertRaisesRegex(ProvisioningError, "stop before activation publication"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=download,
                    activation_hook=lambda python, args, env: (True, "passed"),
                    activation_receipt_writer=lambda *_args: (_ for _ in ()).throw(
                        ProvisioningError("stop before activation publication")
                    ),
                    nonce_factory=lambda: "activation-marker-model",
                )
            self.assertTrue(model_status(paths)["ok"])
            self.assertEqual(activation_status(paths)["status"], "missing")

            publication = paths.activation_receipt_path.with_name(
                f"{paths.activation_receipt_path.name}.publishing"
            )
            published = paths.activation_receipt_path.with_name(
                f"{paths.activation_receipt_path.name}.published"
            )
            real_fsync_directory = runtime_module._fsync_directory
            real_replace = runtime_module.os.replace
            real_link = runtime_module.os.link

            def fail_final_fsync(path):
                if Path(path) == paths.state_dir and published.exists():
                    if publication.exists():
                        raise OSError("injected activation rollback fsync failure")
                    raise OSError("injected final activation receipt fsync failure")
                return real_fsync_directory(path)

            def fail_rollback_replace(source, destination):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected activation rollback replace failure")
                return real_replace(source, destination)

            def fail_rollback_link(source, destination, *args, **kwargs):
                if Path(source) == published and Path(destination) == publication:
                    raise OSError("injected activation rollback link failure")
                return real_link(source, destination, *args, **kwargs)

            with (
                mock.patch("components.Chatterbox.runtime.runtime._fsync_directory", side_effect=fail_final_fsync),
                mock.patch("components.Chatterbox.runtime.runtime.os.replace", side_effect=fail_rollback_replace),
                mock.patch("components.Chatterbox.runtime.runtime.os.link", side_effect=fail_rollback_link),
            ):
                with self.assertRaisesRegex(OSError, "final activation receipt fsync failure") as raised:
                    acquire_model(
                        paths,
                        worker_script=worker,
                        preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                        download_file=lambda *_args: (_ for _ in ()).throw(AssertionError("model must be reused")),
                        activation_hook=lambda python, args, env: (True, "passed"),
                    )
            notes = " ".join(getattr(raised.exception, "__notes__", ()))
            self.assertIn("activation receipt publication rollback rename failed", notes)
            self.assertIn("activation receipt publication quarantine link failed", notes)
            self.assertIn("activation receipt publication quarantine could not be restored", notes)
            self.assertTrue(paths.activation_receipt_path.exists())
            self.assertFalse(publication.exists())
            self.assertTrue(published.exists())
            self.assertTrue(
                paths.activation_receipt_path.with_name(
                    f"{paths.activation_receipt_path.name}.ambiguous"
                ).exists()
            )
            self.assertEqual(activation_status(paths)["status"], "repair_required")
            self.assertEqual(product_status(paths)["status"], "repair_required")
            self.assertEqual(
                _fresh_process_statuses(paths)["activation"]["status"],
                "repair_required",
            )

    def test_activation_receipt_combined_rollback_failures_stay_quarantined_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _install_fixture(Path(directory))
            self._assert_combined_receipt_rollback_failure_is_persistent(
                paths,
                paths.activation_receipt_path,
                "activation",
                lambda: runtime_module.write_activation_receipt(paths, {"status": "ready"}),
                activation_status,
            )

    def test_model_acquisition_rejects_corrupt_partial_symlink_and_extra_content(self) -> None:
        for failure in ("wrong_size", "wrong_hash", "interrupted", "symlink", "extra"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths, worker, content = _model_fixture(root)
                outside = root / "outside-model.bin"
                outside.write_bytes(content)

                def download(_url: str, destination: Path) -> None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if failure == "wrong_size":
                        destination.write_bytes(content[:-1])
                    elif failure == "wrong_hash":
                        destination.write_bytes(b"x" * len(content))
                    elif failure == "interrupted":
                        destination.write_bytes(content[:3])
                        raise OSError("injected interrupted download")
                    elif failure == "symlink":
                        destination.symlink_to(outside)
                    else:
                        destination.write_bytes(content)
                        (destination.parent / "undeclared.bin").write_bytes(b"extra")

                with self.assertRaises((OSError, ProvisioningError)):
                    acquire_model(
                        paths,
                        worker_script=worker,
                        preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                        download_file=download,
                        activation_hook=lambda python, args, env: (True, "local model load passed"),
                        nonce_factory=lambda: f"model-{failure}",
                    )
                self.assertFalse(paths.model_receipt_path.exists())
                self.assertFalse(paths.activation_receipt_path.exists())
                self.assertEqual(list(paths.model_objects_dir.iterdir()), [])
                self.assertEqual(outside.read_bytes(), content)
                self.assertEqual(model_status(paths)["status"], "missing")

    def test_interrupted_model_acquisition_and_activation_are_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def interrupted(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content[:2])
                raise OSError("interrupted")

            with self.assertRaisesRegex(OSError, "interrupted"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=interrupted,
                    nonce_factory=lambda: "interrupted",
                )

            def complete(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            with self.assertRaisesRegex(ProvisioningError, "activation failed"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=complete,
                    activation_hook=lambda python, args, env: (False, "activation failed"),
                    nonce_factory=lambda: "complete-model",
                )
            self.assertTrue(model_status(paths)["ok"])
            self.assertEqual(activation_status(paths)["status"], "missing")
            self.assertEqual(product_status(paths)["status"], "activation_missing")

            result = acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=lambda *_args: (_ for _ in ()).throw(AssertionError("model must be reused")),
                activation_hook=lambda python, args, env: (True, "activation retry passed"),
            )
            self.assertTrue(result["ok"])

    def test_activation_only_acquisition_runs_storage_preflight_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, worker, content = _model_fixture(Path(directory))

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            with self.assertRaisesRegex(ProvisioningError, "stop before activation receipt"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=download,
                    activation_hook=lambda *_args: (True, "activation passed"),
                    activation_receipt_writer=lambda *_args: (_ for _ in ()).throw(
                        ProvisioningError("stop before activation receipt")
                    ),
                    nonce_factory=lambda: "activation-only-model",
                )
            self.assertTrue(model_status(paths)["ok"])
            self.assertEqual(activation_status(paths)["status"], "missing")

            calls: list[dict] = []
            blocked = {
                "ok": False,
                "errors": ["storage_budget_unknown: measurements required"],
                "checks": {
                    "storage": {
                        "ok": False,
                        "status": "storage_budget_unknown",
                        "selected_phases": list(runtime_module.MODEL_ACQUISITION_STORAGE_PHASES),
                    }
                },
            }

            def capacity_gate(*_args, **kwargs):
                calls.append(dict(kwargs))
                return blocked

            with self.assertRaisesRegex(ProvisioningError, "storage_budget_unknown"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=capacity_gate,
                    download_file=lambda *_args: (_ for _ in ()).throw(
                        AssertionError("ready model must be reused")
                    ),
                    activation_hook=lambda *_args: (_ for _ in ()).throw(
                        AssertionError("activation must not start before capacity passes")
                    ),
                )
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0]["check_network"])
            self.assertTrue(model_status(paths)["ok"])
            self.assertEqual(activation_status(paths)["status"], "missing")

    def test_model_receipt_mismatch_fails_closed_without_deleting_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda python, args, env: (True, "passed"),
                nonce_factory=lambda: "receipt-mismatch",
            )
            snapshot = Path(model_status(paths)["snapshot"])
            receipt = json.loads(paths.model_receipt_path.read_text(encoding="utf-8"))
            receipt["model_fingerprint"] = "stale-model"
            paths.model_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(model_status(paths)["status"], "repair_required")
            self.assertEqual(product_status(paths)["status"], "repair_required")
            self.assertTrue(snapshot.is_dir())

    def test_model_receipt_file_metadata_must_match_canonical_manifest_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda python, args, env: (True, "passed"),
                nonce_factory=lambda: "receipt-file-metadata",
            )
            receipt = json.loads(paths.model_receipt_path.read_text(encoding="utf-8"))
            receipt["files"][0]["sha256"] = "0" * 64
            result = validate_model_receipt(paths, receipt)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("file metadata does not match" in error for error in result["errors"])
            )

    def test_unowned_model_object_is_preserved_and_blocks_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)
            foreign = paths.model_objects_dir / "foreign-object"
            foreign.mkdir(parents=True)
            marker = foreign / "preserve.bin"
            marker.write_bytes(b"user state")
            self.assertEqual(model_status(paths)["status"], "repair_required")
            with self.assertRaisesRegex(ProvisioningError, "repair_required"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=lambda _url, destination: destination.write_bytes(content),
                )
            self.assertEqual(marker.read_bytes(), b"user state")
            self.assertFalse(paths.model_receipt_path.exists())

    def test_activation_receipt_hash_mismatch_never_reports_product_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda python, args, env: (True, "passed"),
                nonce_factory=lambda: "activation-mismatch",
            )
            receipt = json.loads(paths.activation_receipt_path.read_text(encoding="utf-8"))
            receipt["model_receipt_sha256"] = "0" * 64
            paths.activation_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(activation_status(paths)["status"], "repair_required")
            self.assertEqual(product_status(paths)["status"], "repair_required")

    def test_model_preflight_keeps_unknown_measurements_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_runtime = Path(__file__).resolve().parents[1]
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            manifest = json.loads((source_runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
            bucket = manifest["storage"]["phases"][2]["buckets"][1]
            bucket["required_bytes"] = None
            bucket["measurement_status"] = "required_clean_disposable"
            bucket.pop("measurement_evidence", None)
            (runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (runtime_dir / "requirements-cpu.lock").write_bytes(
                (source_runtime / "requirements-cpu.lock").read_bytes()
            )
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_STATE_HOME": str(root / "state"),
                    "E2A_ROOT": str(root / "e2a"),
                },
            )
            checked: list[str] = []

            def network_checker(url: str) -> tuple[bool, str]:
                checked.append(url)
                return True, "reachable"

            report = model_preflight(paths, network_checker=network_checker)
            self.assertFalse(report["ok"])
            self.assertEqual(report["checks"]["storage"]["status"], "storage_budget_unknown")
            self.assertEqual(checked, [])
            self.assertEqual(report["checks"]["network"]["status"], "skipped")
            self.assertEqual(report["checks"]["network"]["reason"], "storage_budget_unknown")
            self.assertFalse(report["checks"]["network"]["attempted"])

    def test_flat_two_gib_threshold_is_rejected_and_below_declared_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _manifest()
            manifest["sources"]["model"]["files"] = [
                {"path": "one.bin", "size_bytes": 2_147_483_648, "sha256": "1" * 64},
                {"path": "two.bin", "size_bytes": 1_061_468_100, "sha256": "2" * 64},
            ]
            manifest["storage"] = {"min_free_bytes": 2 * 1024 * 1024 * 1024}
            runtime_dir, manifest = _write_runtime(root, manifest)
            paths = build_paths(runtime_dir=runtime_dir, repo_root=root, environment={"XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state")})
            result = calculate_storage_plan(paths, manifest)
            self.assertEqual(sum(item["size_bytes"] for item in manifest["sources"]["model"]["files"]), 3_208_951_748)
            self.assertGreater(3_208_951_748, manifest["storage"]["min_free_bytes"])
            self.assertEqual(result["status"], "invalid_storage_contract")
            self.assertTrue(any("flat storage.min_free_bytes" in error for error in result["errors"]))

    def test_worker_environment_allowlist_removes_host_runtime_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            lock = runtime_dir / "requirements-cpu.lock"
            lock.write_text("# unresolved lock input\n", encoding="utf-8")
            manifest = _manifest(hashlib.sha256(lock.read_bytes()).hexdigest(), unresolved=["all final hashes"])
            (runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={"XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state")},
            )
            worker_env = sanitized_worker_environment(
                paths,
                {
                    "HOME": "/home/test",
                    "PATH": "/host/conda/bin",
                    "PYTHONPATH": "/host/site-packages",
                    "VIRTUAL_ENV": "/host/venv",
                    "CONDA_PREFIX": "/host/conda",
                    "HF_TOKEN": "must-not-cross-boundary",
                    "LANG": "en_US.UTF-8",
                },
            )
            self.assertEqual(worker_env["HOME"], "/home/test")
            self.assertEqual(worker_env["LANG"], "en_US.UTF-8")
            self.assertNotIn("PYTHONPATH", worker_env)
            self.assertNotIn("VIRTUAL_ENV", worker_env)
            self.assertNotIn("CONDA_PREFIX", worker_env)
            self.assertNotIn("HF_TOKEN", worker_env)
            self.assertEqual(worker_env["PYTHONNOUSERSITE"], "1")
            self.assertIn("chatterbox", worker_env["HF_HOME"])
            self.assertEqual(
                worker_env["PKUSEG_HOME"],
                str(paths.run_namespace / "cache" / "pkuseg"),
            )
            self.assertTrue(worker_env["PATH"].startswith(str(paths.environment / "bin")))

    def test_worker_command_hook_requires_and_records_success_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json\n"
                "print(json.dumps({'protocol': 1, 'event': 'model_self_test', 'ok': True}))\n",
                encoding="utf-8",
            )
            observations: list[dict] = []
            ok, detail = runtime_module._command_hook(
                Path(sys.executable),
                [str(worker), "--model-self-test"],
                {"PYTHONNOUSERSITE": "1"},
                observation_callback=observations.append,
                observation_kind="model_self_test",
            )
            self.assertTrue(ok, detail)
            self.assertEqual(detail, "self-test hook passed")
            self.assertEqual(len(observations), 1)
            observation = observations[0]
            self.assertTrue(observation["execution_observed"])
            self.assertEqual(observation["returncode"], 0)
            self.assertEqual(observation["command"][1], str(worker))
            self.assertEqual(observation["stdout_events"][0]["event"], "model_self_test")

            worker.write_text("print('exit zero without worker event')\n", encoding="utf-8")
            ok, detail = runtime_module._command_hook(
                Path(sys.executable),
                [str(worker), "--model-self-test"],
                {"PYTHONNOUSERSITE": "1"},
            )
            self.assertFalse(ok)
            self.assertIn("did not emit a successful model_self_test event", detail)

    def test_preflight_is_read_only_and_skips_network_when_storage_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            (runtime_dir / "requirements-cpu.lock").write_text("# unresolved lock input\n", encoding="utf-8")
            manifest = _manifest(unresolved=["all final hashes"])
            manifest["network"] = {"first_acquisition_urls": ["https://pypi.org/simple/", "https://download.pytorch.org/whl/cpu/"]}
            (runtime_dir / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            paths = build_paths(
                runtime_dir=runtime_dir,
                repo_root=root,
                environment={
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_STATE_HOME": str(root / "state"),
                    "E2A_ROOT": str(root / "e2a"),
                },
            )
            checked: list[str] = []

            def network_checker(url: str) -> tuple[bool, str]:
                checked.append(url)
                return True, "synthetic reachable"

            report = preflight(paths, Path(sys.executable), network_checker=network_checker)
            self.assertFalse(report["ok"])
            self.assertEqual(checked, [])
            self.assertEqual(report["checks"]["storage"]["status"], "storage_budget_unknown")
            self.assertEqual(report["checks"]["network"]["status"], "skipped")
            self.assertEqual(report["checks"]["network"]["reason"], "storage_budget_unknown")
            self.assertFalse(report["checks"]["network"]["attempted"])
            self.assertFalse(paths.environment.exists())

    def test_checked_in_manifest_identity_lock_and_storage_contract(self) -> None:
        runtime_dir = Path(__file__).resolve().parents[1]
        manifest_path = runtime_dir / "runtime-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = validate_manifest_identity(manifest)
        self.assertTrue(identity["ok"], identity)
        self.assertEqual(manifest["manifest_version"], "2.0.0")
        self.assertEqual(manifest["runtime_contract_version"], "2.0.0")
        lock = verify_lock(runtime_dir / "requirements-cpu.lock", manifest)
        self.assertTrue(lock["ok"], lock)
        self.assertEqual(lock["requirements"], 107)
        self.assertEqual(lock["sha256"], "b615b428258c69771b52da7310d954040aa5340efbb5b7f988e598347c56206d")
        paths = build_paths(runtime_dir=runtime_dir)
        storage = calculate_storage_plan(paths, manifest)
        self.assertIn(storage["status"], {"storage_budget_unknown", "sufficient", "insufficient_storage"})
        self.assertIsNone(paths.verified_model_root)

    def test_readiness_snapshot_inspects_each_dimension_once_and_passes_results_to_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _worker, _content = _model_fixture(Path(directory))
            calls: list[tuple[str, object]] = []
            runtime = {"ok": True, "status": "ready", "environment": str(paths.environment)}
            model = {"ok": True, "status": "ready", "snapshot": str(paths.model_namespace)}
            activation = {"ok": True, "status": "ready"}

            def activation_check(selected, *, runtime_result, model_result):
                calls.append(("activation", (selected, runtime_result, model_result)))
                return activation

            with (
                mock.patch.object(runtime_module, "runtime_status", side_effect=lambda value: calls.append(("runtime", value)) or runtime),
                mock.patch.object(runtime_module, "model_status", side_effect=lambda value: calls.append(("model", value)) or model),
                mock.patch.object(runtime_module, "activation_status", side_effect=activation_check),
            ):
                snapshot = readiness_snapshot(paths)

            self.assertEqual([name for name, _value in calls], ["runtime", "model", "activation"])
            selected = snapshot["paths"]
            self.assertEqual(selected.environment, paths.environment)
            self.assertEqual(selected.model_namespace, paths.model_namespace)
            self.assertIs(selected, calls[-1][1][0])
            self.assertIs(calls[-1][1][1], runtime)
            self.assertIs(calls[-1][1][2], model)
            self.assertEqual(snapshot["status"], "ready")

    def test_readiness_snapshot_hashes_each_model_once_per_call_and_worker_verifies_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            acquire_model(
                paths,
                worker_script=root / "components/Chatterbox/worker.py",
                preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                download_file=download,
                activation_hook=lambda _python, _args, _env: (True, "passed"),
                nonce_factory=lambda: "hash-count-model",
            )
            first = readiness_snapshot(paths)
            snapshot = Path(first["model"]["snapshot"])
            model_files = {snapshot / relative for relative in CANONICAL_MODEL_FILE_PATHS}
            counts: dict[Path, int] = {path: 0 for path in model_files}
            original_hash = runtime_module.sha256_file

            def counting_hash(path: Path) -> str:
                resolved = Path(path).resolve()
                if resolved in counts:
                    counts[resolved] += 1
                return original_hash(path)

            with mock.patch.object(runtime_module, "sha256_file", side_effect=counting_hash):
                second = readiness_snapshot(paths)
                self.assertTrue(second["ok"])
                self.assertEqual(set(counts.values()), {1})
                third = readiness_snapshot(paths)
                self.assertTrue(third["ok"])
                self.assertEqual(set(counts.values()), {2})

            worker_path = Path(__file__).resolve().parents[2] / "worker.py"
            spec = importlib.util.spec_from_file_location("chatterbox_worker_readiness_test", worker_path)
            assert spec is not None and spec.loader is not None
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            records = runtime_module._model_file_records(manifest)
            self.assertEqual(worker._verify_model_snapshot(snapshot, records, snapshot), snapshot)

    def test_model_preflight_and_activation_only_acquisition_credit_verified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, worker, content = _model_fixture(root)

            def download(_url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            with self.assertRaisesRegex(ProvisioningError, "activation failed"):
                acquire_model(
                    paths,
                    worker_script=worker,
                    preflight_runner=lambda *_args, **_kwargs: {"ok": True},
                    download_file=download,
                    activation_hook=lambda _python, _args, _env: (False, "activation failed"),
                    nonce_factory=lambda: "preflight-credit-model",
                )

            model = model_status(paths)
            snapshot = Path(model["snapshot"])
            observed: list[Path | None] = []

            def observe_preflight(preflight_paths, **_kwargs):
                observed.append(preflight_paths.verified_model_root)
                return {"ok": True}

            direct_report = model_preflight(paths, check_network=False)
            self.assertTrue(direct_report["checks"]["storage"]["ok"])

            result = acquire_model(
                paths,
                worker_script=worker,
                preflight_runner=observe_preflight,
                download_file=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("activation-only acquisition must reuse the verified model")
                ),
                activation_hook=lambda _python, _args, _env: (True, "passed"),
            )
            self.assertTrue(result["ok"])
            self.assertGreaterEqual(len(observed), 2)
            self.assertTrue(all(value == snapshot for value in observed))

    def test_capacity_fixtures_cover_unknown_sufficient_and_insufficient_without_checked_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for state, required, free in (("storage_budget_unknown", None, 10_000), ("sufficient", 10, 10_000), ("insufficient_storage", 10, 5)):
                with self.subTest(state=state):
                    manifest = _manifest()
                    for phase in manifest["storage"]["phases"]:
                        for bucket in phase["buckets"]:
                            bucket["required_bytes"] = required
                            if required is None:
                                bucket["measurement_status"] = "required_clean_disposable"
                            else:
                                bucket.pop("measurement_status", None)
                    fixture_root = root / state
                    fixture_root.mkdir()
                    runtime_dir, manifest = _write_runtime(fixture_root, manifest)
                    paths = build_paths(runtime_dir=runtime_dir, repo_root=fixture_root)
                    result = calculate_storage_plan(
                        paths,
                        manifest,
                        free_bytes_provider=lambda _path, value=free: value,
                        filesystem_resolver=lambda path, base=root / state: ("fixture", base),
                    )
                    self.assertEqual(result["status"], state)

            invalid_manifest = _manifest()
            invalid_manifest["storage"] = {"min_free_bytes": 1}
            invalid_root = root / "invalid"
            invalid_root.mkdir()
            invalid_runtime, invalid_manifest = _write_runtime(invalid_root, invalid_manifest)
            invalid_paths = build_paths(runtime_dir=invalid_runtime, repo_root=invalid_root)
            invalid = calculate_storage_plan(invalid_paths, invalid_manifest)
            self.assertEqual(invalid["status"], "invalid_storage_contract")

    def test_contract_data_is_stdlib_only_immutable_and_profile_aware(self) -> None:
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
        self.assertIs(runtime_module.TARGET_PYTHON, contract_data.TARGET_PYTHON)
        self.assertIs(runtime_module.RUNTIME_RECEIPT_SCHEMA, contract_data.RUNTIME_RECEIPT_SCHEMA)
        self.assertEqual(contract_data.SUPPORTED_MODEL_PROFILES, ("v2", "v3", "turbo", "nano"))
        self.assertEqual(contract_data.model_profile_spec("v2").loader_kind, "multilingual")
        self.assertEqual(contract_data.model_profile_spec("v3").family, "chatterbox-multilingual")
        self.assertEqual(contract_data.model_profile_spec("turbo").loader_kind, "turbo")
        self.assertEqual(contract_data.model_profile_spec("turbo").supported_languages, ("en",))
        self.assertEqual(contract_data.model_profile_spec("turbo").minimum_prompt_seconds, 5.0)
        self.assertEqual(contract_data.model_profile_spec("nano").fixed_language_id, "en")
        self.assertEqual(len(contract_data.SUPPORTED_LANGUAGES), 23)
        self.assertEqual(contract_data.MODEL_VARIANT, "v2")
        self.assertEqual(contract_data.TARGET_BACKEND, "cpu")
        with self.assertRaises(FrozenInstanceError):
            contract_data.CONTRACT_DATA.model_variant = "v3"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            contract_data.model_profile_spec("v2").profile = "changed"  # type: ignore[misc]

    def test_host_status_is_frozen_read_only_and_does_not_expose_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            runtime_dir, _manifest_value = _write_runtime(root / "fixture", _manifest())
            paths = build_paths(runtime_dir=runtime_dir, repo_root=root / "fixture")

            def inventory() -> dict[str, tuple[bool, int, int, int]]:
                result: dict[str, tuple[bool, int, int, int]] = {}
                for path in root.rglob("*"):
                    stat = path.lstat()
                    result[str(path.relative_to(root))] = (
                        path.is_dir(),
                        stat.st_mode,
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                return result

            before = inventory()
            status = host_runtime_status(paths)
            after = inventory()

            self.assertIsInstance(status, HostRuntimeStatus)
            self.assertEqual(status.status, "provisioning")
            self.assertEqual(status.capacity_status, "storage_budget_unknown")
            self.assertFalse(status.ok)
            self.assertEqual(status.model_revision, "4" * 40)
            self.assertEqual(status.manifest_path, paths.manifest_path)
            self.assertEqual(status.runtime_root, paths.runtime_dir)
            self.assertEqual(status.manifest_root, paths.runtime_dir)
            self.assertFalse(hasattr(status, "paths"))
            self.assertNotIn("paths", status.as_dict())
            self.assertNotIn(paths, status.as_dict().values())
            self.assertEqual(before, after)
            with self.assertRaises(FrozenInstanceError):
                status.status = "ready"  # type: ignore[misc]

            with (
                mock.patch.object(runtime_module.platform, "system", return_value="Darwin"),
                mock.patch.object(
                    runtime_module,
                    "readiness_snapshot",
                    side_effect=AssertionError("unsupported targets must stop before readiness"),
                ),
            ):
                unsupported = host_runtime_status(paths)
            self.assertFalse(unsupported.supported)
            self.assertEqual(unsupported.status, "unsupported")
            self.assertEqual(unsupported.capacity_status, "not_checked")

            incomplete = {
                "ok": False,
                "status": "runtime_missing",
                "runtime": {"ok": False, "status": "missing", "errors": []},
                "model": {"ok": False, "status": "missing", "errors": []},
                "activation": {"ok": False, "status": "missing", "errors": []},
                "paths": paths,
            }
            with (
                mock.patch.object(runtime_module, "readiness_snapshot", return_value=incomplete),
                mock.patch.object(
                    runtime_module,
                    "calculate_storage_plan",
                    return_value={"ok": False, "status": "insufficient_storage", "errors": []},
                ),
            ):
                insufficient = host_runtime_status(paths)
            self.assertEqual(insufficient.status, "missing")
            self.assertEqual(insufficient.capacity_status, "insufficient_storage")
            self.assertIn("insufficient storage", insufficient.error or "")

    def test_host_status_ready_bypasses_capacity_and_selects_only_approved_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            runtime_dir, _manifest_value = _write_runtime(root / "fixture", _manifest())
            paths = build_paths(runtime_dir=runtime_dir, repo_root=root / "fixture")
            environment = root / "data" / "approved-env"
            model_root = root / "data" / "approved-model"
            snapshot = {
                "ok": True,
                "status": "ready",
                "runtime": {"ok": True, "status": "ready", "environment": str(environment)},
                "model": {"ok": True, "status": "ready", "snapshot": str(model_root)},
                "activation": {"ok": True, "status": "ready"},
                "paths": paths,
            }
            with (
                mock.patch.object(runtime_module, "readiness_snapshot", return_value=snapshot),
                mock.patch.object(
                    runtime_module,
                    "calculate_storage_plan",
                    side_effect=AssertionError("ready receipt chain must bypass capacity"),
                ),
            ):
                status = host_runtime_status(paths)

            self.assertTrue(status.ok)
            self.assertTrue(status.supported)
            self.assertEqual(status.status, "ready")
            self.assertEqual(status.capacity_status, "not_required")
            self.assertEqual(status.interpreter, environment / "bin/python")
            self.assertEqual(status.verified_model_root, model_root)
            self.assertIs(status.model_root, status.verified_model_root)
            self.assertEqual(status.model_revision, "4" * 40)
            self.assertEqual(status.manifest_path, paths.manifest_path)
            self.assertIsNotNone(status.environment)
            with self.assertRaises(TypeError):
                status.environment["PATH"] = "/unsafe"  # type: ignore[index]

    def test_host_status_classifies_expected_errors_and_propagates_unexpected_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            runtime_dir, _manifest_value = _write_runtime(root / "fixture", _manifest())
            paths = build_paths(runtime_dir=runtime_dir, repo_root=root / "fixture")
            expected_errors = (
                (RuntimeManifestError("bad manifest"), "manifest"),
                (RuntimeReceiptError("bad receipt"), "receipt"),
                (RuntimeConfigurationError("bad configuration"), "configuration"),
                (OSError("filesystem unavailable"), "filesystem"),
            )
            for error, kind in expected_errors:
                with self.subTest(kind=kind), mock.patch.object(
                    runtime_module,
                    "readiness_snapshot",
                    side_effect=error,
                ):
                    status = host_runtime_status(paths)
                self.assertFalse(status.ok)
                self.assertEqual(status.status, "repair_required")
                self.assertEqual(status.error_kind, kind)
                self.assertIn(str(error), status.error or "")

            with (
                mock.patch.object(
                    runtime_module,
                    "readiness_snapshot",
                    side_effect=AssertionError("unexpected programming defect"),
                ),
                self.assertRaises(AssertionError),
            ):
                host_runtime_status(paths)


if __name__ == "__main__":
    unittest.main()
