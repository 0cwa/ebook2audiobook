from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.Chatterbox.runtime.contract_data import canonical_model_file_paths
from components.Chatterbox.runtime.measurement import (
    CANONICAL_MODEL_FILE_PATHS,
    DISPOSABLE_MARKER,
    DISPOSABLE_MARKER_CONTENT,
    InjectedMeasurementFault,
    MeasurementConfigurationError,
    MeasurementSession,
    REQUIRED_PATH_NAMES,
    VerifiedInput,
    offline_environment,
    validate_disposable_paths,
)
from components.Chatterbox.runtime.runtime import build_paths, calculate_storage_plan, preflight


ZERO_EVIDENCE = {
    "schema": "ebook2audiobook.chatterbox-storage-measurement.v1",
    "evidence_id": "synthetic-measured-zero",
    "measured_bytes": 0,
    "runs": 3,
}


def _verified(name: str, path: Path, content: bytes) -> VerifiedInput:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return VerifiedInput(name, path, len(content), hashlib.sha256(content).hexdigest())


class MeasurementTests(unittest.TestCase):
    def _surface(self, root: Path):
        (root / DISPOSABLE_MARKER).write_text(DISPOSABLE_MARKER_CONTENT, encoding="utf-8")
        paths = {name: root / name for name in REQUIRED_PATH_NAMES}
        for path in paths.values():
            path.mkdir()
        wheelhouse = paths["cache"] / "wheelhouse"
        wheelhouse.mkdir()
        lock = _verified("requirements-cpu.lock", paths["cache"] / "requirements-cpu.lock", b"synthetic lock\n")
        packages = [_verified("package.whl", wheelhouse / "package.whl", b"synthetic wheel")]
        models = [
            _verified(name, paths["model"] / name, f"model:{name}".encode())
            for name in CANONICAL_MODEL_FILE_PATHS
        ]
        return paths, wheelhouse, lock, packages, models

    def _session(self, root: Path, **overrides) -> MeasurementSession:
        paths, wheelhouse, lock, packages, models = self._surface(root)
        values = {
            "disposable_root": root,
            "paths": paths,
            "wheelhouse": wheelhouse,
            "lock_input": lock,
            "package_inputs": packages,
            "expected_package_count": 1,
            "model_inputs": models,
        }
        values.update(overrides)
        return MeasurementSession(**values)

    def test_v3_session_uses_v3_canonical_model_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths, wheelhouse, lock, packages, _models = self._surface(root)
            v3_models = [
                _verified(name, paths["model"] / name, f"model:{name}".encode())
                for name in canonical_model_file_paths("v3")
            ]
            session = MeasurementSession(
                disposable_root=root,
                paths=paths,
                wheelhouse=wheelhouse,
                lock_input=lock,
                package_inputs=packages,
                expected_package_count=1,
                model_inputs=v3_models,
                model_variant="v3",
            )
            self.assertEqual(session.model_variant, "v3")
            self.assertEqual(
                session.expected_model_file_paths,
                canonical_model_file_paths("v3"),
            )
            self.assertEqual(
                {record["name"] for record in session.inputs["models"]},
                set(canonical_model_file_paths("v3")),
            )

    def test_disposable_root_and_every_path_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {name: root / name for name in REQUIRED_PATH_NAMES}
            with self.assertRaisesRegex(MeasurementConfigurationError, "marker"):
                validate_disposable_paths(root, paths)
            (root / DISPOSABLE_MARKER).write_text(DISPOSABLE_MARKER_CONTENT, encoding="utf-8")
            incomplete = dict(paths)
            incomplete.pop("run")
            with self.assertRaisesRegex(MeasurementConfigurationError, "map mismatch"):
                validate_disposable_paths(root, incomplete)

    def test_path_escape_relative_traversal_alias_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths, _wheelhouse, _lock, _packages, _models = self._surface(root)
            for bad in (Path("relative"), root / "runtime/../escape", root.parent / "escape"):
                candidate = dict(paths)
                candidate["runtime"] = bad
                with self.subTest(path=bad), self.assertRaises(MeasurementConfigurationError):
                    validate_disposable_paths(root, candidate)
            candidate = dict(paths)
            candidate["run"] = candidate["state"]
            with self.assertRaisesRegex(MeasurementConfigurationError, "alias"):
                validate_disposable_paths(root, candidate)
            target = root / "real-target"
            target.mkdir()
            link = root / "linked"
            link.symlink_to(target, target_is_directory=True)
            candidate = dict(paths)
            candidate["run"] = link / "run"
            with self.assertRaisesRegex(MeasurementConfigurationError, "symlink"):
                validate_disposable_paths(root, candidate)

    def test_explicit_inputs_are_verified_and_confined(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory:
            root = Path(directory).resolve()
            paths, wheelhouse, lock, packages, models = self._surface(root)
            outside = _verified("outside.whl", Path(outside_directory).resolve() / "outside.whl", b"outside")
            with self.assertRaisesRegex(MeasurementConfigurationError, "escapes"):
                MeasurementSession(
                    disposable_root=root,
                    paths=paths,
                    wheelhouse=wheelhouse,
                    lock_input=lock,
                    package_inputs=[outside],
                    expected_package_count=1,
                    model_inputs=models,
                )
            corrupt = list(models)
            corrupt[0] = VerifiedInput(corrupt[0].name, corrupt[0].path, corrupt[0].size_bytes, "0" * 64)
            with self.assertRaisesRegex(MeasurementConfigurationError, "failed verification"):
                MeasurementSession(
                    disposable_root=root,
                    paths=paths,
                    wheelhouse=wheelhouse,
                    lock_input=lock,
                    package_inputs=packages,
                    expected_package_count=1,
                    model_inputs=corrupt,
                )

    def test_sparse_package_and_model_inputs_are_rejected(self) -> None:
        for kind in ("package", "model"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                paths, wheelhouse, lock, packages, models = self._surface(root)
                target = packages[0] if kind == "package" else models[0]
                target.path.unlink()
                sparse_size = 1024 * 1024
                with target.path.open("wb") as stream:
                    stream.seek(sparse_size - 1)
                    stream.write(b"\0")
                sparse = VerifiedInput(
                    target.name,
                    target.path,
                    sparse_size,
                    hashlib.sha256(target.path.read_bytes()).hexdigest(),
                )
                if kind == "package":
                    packages = [sparse]
                else:
                    models = [sparse if model.name == target.name else model for model in models]
                with self.assertRaisesRegex(
                    MeasurementConfigurationError, "sparse|under-allocated"
                ):
                    MeasurementSession(
                        disposable_root=root,
                        paths=paths,
                        wheelhouse=wheelhouse,
                        lock_input=lock,
                        package_inputs=packages,
                        expected_package_count=1,
                        model_inputs=models,
                    )

    def test_offline_contract_and_observation_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            session = self._session(root)
            session.record_event("baseline")
            session.record_event("candidate_reserved", detail={"scenario": "runtime"})
            with self.assertRaisesRegex(MeasurementConfigurationError, "readiness-like"):
                session.record_event("verification_complete", detail={"status": "ready"})
            report = session.observations()
            self.assertEqual(report["mode"], "observation_only")
            self.assertFalse(report["offline"]["public_network_authorized"])
            self.assertIn("--no-index", report["offline"]["pip_arguments"])
            self.assertIn("--find-links", report["offline"]["pip_arguments"])
            self.assertNotIn("--index-url", report["offline"]["pip_arguments"])
            self.assertEqual([event["name"] for event in report["events"]], ["baseline", "candidate_reserved"])
            self.assertGreaterEqual(report["events"][0]["paths"]["model"]["allocated_bytes"], 0)
            self.assertTrue(report["events"][0]["filesystems"])
            encoded = json.dumps(report, sort_keys=True)
            for forbidden in ('"ok"', '"status"', '"ready"', '"readiness"'):
                self.assertNotIn(forbidden, encoded)

            environment = offline_environment({"PATH": "/bin", "HTTPS_PROXY": "http://proxy", "HF_TOKEN": "secret"})
            self.assertEqual(environment["PIP_NO_INDEX"], "1")
            self.assertEqual(environment["NO_PROXY"], "*")
            self.assertNotIn("HTTPS_PROXY", environment)
            self.assertNotIn("HF_TOKEN", environment)

    def test_fault_retry_and_concurrency_hooks_only_add_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            session = self._session(
                root,
                fault_plan={("receipt_publishing", 2, "writer-a"): "synthetic rename failure"},
            )
            session.record_concurrency("writer-a", started=True)
            session.record_retry(attempt=2, reason="first attempt interrupted", participant="writer-a")
            with self.assertRaisesRegex(InjectedMeasurementFault, "synthetic rename failure"):
                session.record_event("receipt_publishing", attempt=2, participant="writer-a")
            names = [event["name"] for event in session.observations()["events"]]
            self.assertEqual(names, ["concurrency_started", "retry_started", "receipt_publishing", "fault_injected"])

    def test_null_and_unproven_zero_are_unknown_but_measured_zero_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime_dir = root / "runtime-source"
            runtime_dir.mkdir()
            lock = runtime_dir / "requirements-cpu.lock"
            lock.write_text("# e2a-lock-format: 1\n--require-hashes\n", encoding="utf-8")
            manifest = {
                "manifest_version": "2.0.0",
                "runtime_contract_version": "2.0.0",
                "target": {"os": "linux", "architecture": "x86_64", "python": "3.11.x", "backend": "cpu"},
                "product": {"profile": "test", "worker_protocol": 1},
                "sources": {"model": {"files": [{"path": "model.bin", "size_bytes": 1, "sha256": "1" * 64}]}},
                "lock": {"path": "requirements-cpu.lock", "status": "verified", "sha256": hashlib.sha256(lock.read_bytes()).hexdigest()},
                "storage": {"contract_version": "1.0.0", "phases": [{"name": "runtime_artifact_acquisition", "buckets": []}]},
            }
            environment = {
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_STATE_HOME": str(root / "state"),
                "E2A_ROOT": str(root / "e2a"),
            }
            for expected, bucket in (
                ("storage_budget_unknown", {"name": "bucket", "destination": "environment", "required_bytes": None, "measurement_status": "required_clean_disposable"}),
                ("storage_budget_unknown", {"name": "bucket", "destination": "environment", "required_bytes": 0}),
                ("sufficient", {"name": "bucket", "destination": "environment", "required_bytes": 0, "measurement_status": "measured_clean_disposable", "measurement_evidence": ZERO_EVIDENCE}),
            ):
                with self.subTest(expected=expected, bucket=bucket):
                    manifest["storage"]["phases"][0]["buckets"] = [bucket]
                    manifest_path = runtime_dir / "runtime-manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    paths = build_paths(runtime_dir=runtime_dir, repo_root=root, environment=environment)
                    result = calculate_storage_plan(
                        paths,
                        manifest,
                        filesystem_resolver=lambda _path: ("fixture", root),
                        free_bytes_provider=lambda _path: 1,
                    )
                    self.assertEqual(result["status"], expected, result)

    def test_normal_preflight_remains_fail_closed_and_measurement_is_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_runtime = Path(__file__).resolve().parents[1]
            runtime_dir = root / "runtime"
            runtime_dir.mkdir()
            manifest = json.loads((source_runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
            bucket = manifest["storage"]["phases"][0]["buckets"][0]
            bucket["required_bytes"] = None
            bucket["measurement_status"] = "required_clean_disposable"
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
            network_calls: list[str] = []
            before = {path: path.exists() for path in (paths.state_dir, paths.install_lock_path, paths.environment)}
            with mock.patch("components.Chatterbox.runtime.runtime._probe_interpreter", return_value=({"version": [3, 11, 0], "implementation": "CPython"}, None)):
                report = preflight(paths, Path(os.sys.executable), network_checker=lambda url: network_calls.append(url) or (True, "unexpected"))
            self.assertEqual(report["checks"]["storage"]["status"], "storage_budget_unknown")
            self.assertEqual(network_calls, [])
            self.assertEqual(before, {path: path.exists() for path in before})

        repository = Path(__file__).resolve().parents[4]
        host_files = [
            repository / "lib/classes/tts_engines/chatterbox.py",
            repository / "lib/core.py",
            repository / "lib/gradio.py",
        ]
        for host_file in host_files:
            self.assertNotIn("runtime.measurement", host_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
