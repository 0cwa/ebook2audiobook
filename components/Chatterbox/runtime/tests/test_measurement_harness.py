from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from components.Chatterbox.runtime.measurement import (
    CANONICAL_MODEL_FILE_PATHS,
    DISPOSABLE_MARKER,
    DISPOSABLE_MARKER_CONTENT,
    MeasurementConfigurationError,
    MeasurementSession,
    REQUIRED_PATH_NAMES,
    VerifiedInput,
)
from components.Chatterbox.runtime.measurement_harness import (
    CANONICAL_FILESYSTEM_LAYOUTS,
    CANONICAL_SCENARIOS,
    HARNESS_SCHEMA,
    CURRENT_LOCK_REQUIREMENT_COUNT,
    REQUIRED_ENVIRONMENT_PATHS,
    REQUIRED_OBSERVATION_PATHS,
    TransactionHooks,
    aggregate_repeated_runs,
    build_disposable_environment,
    capture_snapshot,
    capture_root_inventory,
    cleanup_owned_artifacts,
    _cleanup_evidence_is_complete,
    invoke_transaction_hook,
    materialize_pip_arguments,
    run_disposable_measurement,
    run_process_phase,
    _validate_worker_observations,
    validate_complete_wheelhouse,
)
from components.Chatterbox.runtime.contract_data import PKUSEG_DATA_FILENAME


def _verified(name: str, path: Path, content: bytes) -> VerifiedInput:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return VerifiedInput(name, path, len(content), hashlib.sha256(content).hexdigest())


def _wheel(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.txt", content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeasurementHarnessTests(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory).resolve()
        (root / DISPOSABLE_MARKER).write_text(DISPOSABLE_MARKER_CONTENT, encoding="utf-8")
        return root

    def _environment_paths(self, root: Path) -> dict[str, Path]:
        parent = root / "environment"
        parent.mkdir(exist_ok=True)
        return {name: parent / name for name in REQUIRED_ENVIRONMENT_PATHS}

    def _session(self, root: Path) -> MeasurementSession:
        paths = {name: root / "surface" / name for name in REQUIRED_PATH_NAMES}
        for path in paths.values():
            path.mkdir(parents=True)
        wheelhouse = paths["cache"] / "wheelhouse"
        wheelhouse.mkdir()
        lock = _verified(
            "requirements-cpu.lock",
            paths["cache"] / "requirements-cpu.lock",
            b"# e2a-lock-format: 1\n--require-hashes\n",
        )
        package = _verified("fixture.whl", wheelhouse / "fixture.whl", b"fixture")
        models = [
            _verified(name, paths["model"] / name, f"model:{name}".encode())
            for name in CANONICAL_MODEL_FILE_PATHS
        ]
        return MeasurementSession(
            disposable_root=root,
            paths=paths,
            wheelhouse=wheelhouse,
            lock_input=lock,
            package_inputs=[package],
            expected_package_count=1,
            model_inputs=models,
        )

    def _wheelhouse_fixture(self, root: Path):
        wheelhouse = root / "wheelhouse"
        wheelhouse.mkdir()
        hashes = {
            "alpha-pkg": _wheel(wheelhouse / "alpha_pkg-1.0-py3-none-any.whl", b"alpha"),
            "beta": _wheel(wheelhouse / "beta-2.0-py3-none-any.whl", b"beta"),
        }
        lock = root / "requirements-cpu.lock"
        lock.write_text(
            "\n".join(
                (
                    "# e2a-lock-format: 1",
                    "--require-hashes",
                    f"alpha-pkg==1.0 --hash=sha256:{hashes['alpha-pkg']}",
                    f"beta==2.0 --hash=sha256:{hashes['beta']}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        sizes = {
            digest: next(path.stat().st_size for path in wheelhouse.iterdir() if hashlib.sha256(path.read_bytes()).hexdigest() == digest)
            for digest in hashes.values()
        }
        return wheelhouse, lock, hashlib.sha256(lock.read_bytes()).hexdigest(), sizes

    def _execution_fixture(self, root: Path, *, include_worker_data: bool = False):
        paths = {name: root / "surface" / name for name in REQUIRED_PATH_NAMES}
        for path in paths.values():
            path.mkdir(parents=True)
        wheelhouse = paths["cache"] / "wheelhouse"
        wheelhouse.mkdir()
        package_inputs = []
        requirement_lines = ["# e2a-lock-format: 1", "--require-hashes"]
        sizes = {}
        for index in range(CURRENT_LOCK_REQUIREMENT_COUNT):
            project = f"fixture-pkg-{index:03d}"
            artifact = wheelhouse / f"fixture_pkg_{index:03d}-1.0-py3-none-any.whl"
            digest = _wheel(artifact, project.encode())
            requirement_lines.append(f"{project}==1.0 --hash=sha256:{digest}")
            sizes[digest] = artifact.stat().st_size
            package_inputs.append(
                VerifiedInput(artifact.name, artifact, artifact.stat().st_size, digest)
            )
        lock_path = paths["cache"] / "requirements-cpu.lock"
        lock_path.write_text("\n".join((*requirement_lines, "")), encoding="utf-8")
        lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        lock_input = VerifiedInput(
            lock_path.name, lock_path, lock_path.stat().st_size, lock_sha
        )
        model_inputs = [
            _verified(name, paths["model"] / name, (f"model:{name}".encode() * 512))
            for name in CANONICAL_MODEL_FILE_PATHS
        ]
        worker_data_inputs = []
        if include_worker_data:
            worker_data_path = paths["cache"] / "worker-data" / PKUSEG_DATA_FILENAME
            worker_data_digest = _wheel(worker_data_path, b"tokenizer-data")
            worker_data_inputs.append(
                VerifiedInput(
                    PKUSEG_DATA_FILENAME,
                    worker_data_path,
                    worker_data_path.stat().st_size,
                    worker_data_digest,
                )
            )
        session = MeasurementSession(
            disposable_root=root,
            paths=paths,
            wheelhouse=wheelhouse,
            lock_input=lock_input,
            package_inputs=package_inputs,
            expected_package_count=CURRENT_LOCK_REQUIREMENT_COUNT,
            model_inputs=model_inputs,
            worker_data_inputs=worker_data_inputs,
        )
        evidence = validate_complete_wheelhouse(
            root=root,
            wheelhouse=wheelhouse,
            lock_path=lock_path,
            expected_lock_sha256=lock_sha,
            expected_artifact_sizes=sizes,
        )
        return session, evidence, lock_sha, model_inputs

    def test_environment_is_confined_and_strips_network_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            paths = self._environment_paths(root)
            environment = build_disposable_environment(
                root,
                paths,
                base={"PATH": "/bin", "HTTPS_PROXY": "http://proxy", "HF_TOKEN": "secret"},
            )
            for variable in (
                "HOME",
                "XDG_CACHE_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
                "XDG_RUNTIME_DIR",
                "TMPDIR",
                "PYTHONPYCACHEPREFIX",
                "PIP_CACHE_DIR",
            ):
                Path(environment[variable]).relative_to(root)
                self.assertTrue(Path(environment[variable]).is_dir())
            self.assertEqual(environment["PIP_NO_INDEX"], "1")
            self.assertNotIn("HTTPS_PROXY", environment)
            self.assertNotIn("HF_TOKEN", environment)

    def test_environment_path_escape_alias_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = self._root(directory)
            for name, bad in (
                ("escape", Path(outside).resolve()),
                ("traversal", root / "environment/home/../escape"),
            ):
                with self.subTest(name=name):
                    paths = self._environment_paths(root)
                    paths["home"] = bad
                    with self.assertRaises(MeasurementConfigurationError):
                        build_disposable_environment(root, paths)
            paths = self._environment_paths(root)
            paths["pip_cache"] = paths["tmpdir"]
            with self.assertRaisesRegex(MeasurementConfigurationError, "alias"):
                build_disposable_environment(root, paths)
            target = root / "target"
            target.mkdir()
            link = root / "linked"
            link.symlink_to(target, target_is_directory=True)
            paths = self._environment_paths(root)
            paths["home"] = link / "home"
            with self.assertRaisesRegex(MeasurementConfigurationError, "symlink"):
                build_disposable_environment(root, paths)

    def test_snapshot_counts_normal_symlinks_without_following_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = self._root(directory)
            observed = root / "observed"
            observed.mkdir()
            target = Path(outside) / "target.bin"
            target.write_bytes(b"outside")
            link = observed / "python"
            link.symlink_to(target)

            snapshot = capture_snapshot(root, {"observed": observed})

            record = snapshot["paths"]["observed"]
            self.assertEqual(record["entries"], 2)
            self.assertEqual(record["logical_bytes"], observed.stat().st_size + link.lstat().st_size)

    def test_snapshot_ignores_a_file_vanishing_during_walk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            observed = root / "observed"
            observed.mkdir()
            vanished = observed / "temporary.pyc"
            vanished.write_bytes(b"transient")
            real_lstat = Path.lstat

            def lstat(path: Path):
                if path == vanished:
                    raise FileNotFoundError(path)
                return real_lstat(path)

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.Path.lstat",
                autospec=True,
                side_effect=lstat,
            ):
                snapshot = capture_snapshot(root, {"observed": observed})

            self.assertEqual(snapshot["paths"]["observed"]["entries"], 1)

    def test_complete_lock_correspondence_and_no_index_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            wheelhouse, lock, lock_sha, sizes = self._wheelhouse_fixture(root)
            with self.assertRaisesRegex(MeasurementConfigurationError, "size map"):
                validate_complete_wheelhouse(
                    root=root,
                    wheelhouse=wheelhouse,
                    lock_path=lock,
                    expected_lock_sha256=lock_sha,
                    expected_artifact_sizes={},
                    expected_requirement_count=2,
                )
            evidence = validate_complete_wheelhouse(
                root=root,
                wheelhouse=wheelhouse,
                lock_path=lock,
                expected_lock_sha256=lock_sha,
                expected_artifact_sizes=sizes,
                expected_requirement_count=2,
            )
            self.assertEqual(len(evidence.requirements), 2)
            self.assertEqual({item["project"] for item in evidence.artifacts}, {"alpha-pkg", "beta"})
            paths = self._environment_paths(root)
            build_disposable_environment(root, paths)
            arguments = materialize_pip_arguments(evidence, paths["pip_cache"], root)
            self.assertIn("--no-index", arguments)
            self.assertIn("--find-links", arguments)
            self.assertIn("--cache-dir", arguments)
            self.assertNotIn("--index-url", arguments)
            self.assertNotIn("--extra-index-url", arguments)

    def test_incomplete_foreign_and_sparse_wheelhouses_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            wheelhouse, lock, lock_sha, sizes = self._wheelhouse_fixture(root)
            (wheelhouse / "beta-2.0-py3-none-any.whl").unlink()
            with self.assertRaisesRegex(MeasurementConfigurationError, "artifact count"):
                validate_complete_wheelhouse(
                    root=root,
                    wheelhouse=wheelhouse,
                    lock_path=lock,
                    expected_lock_sha256=lock_sha,
                    expected_artifact_sizes=sizes,
                    expected_requirement_count=2,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            wheelhouse, lock, lock_sha, sizes = self._wheelhouse_fixture(root)
            foreign = wheelhouse / "foreign-1.0-py3-none-any.whl"
            _wheel(foreign, b"foreign")
            (wheelhouse / "beta-2.0-py3-none-any.whl").unlink()
            with self.assertRaisesRegex(MeasurementConfigurationError, "foreign"):
                validate_complete_wheelhouse(
                    root=root,
                    wheelhouse=wheelhouse,
                    lock_path=lock,
                    expected_lock_sha256=lock_sha,
                    expected_artifact_sizes=sizes,
                    expected_requirement_count=2,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            sparse = wheelhouse / "alpha-1.0-py3-none-any.whl"
            with sparse.open("wb") as stream:
                stream.seek(1024 * 1024 - 1)
                stream.write(b"\0")
            digest = hashlib.sha256(sparse.read_bytes()).hexdigest()
            lock = root / "requirements-cpu.lock"
            lock.write_text(
                f"--require-hashes\nalpha==1.0 --hash=sha256:{digest}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MeasurementConfigurationError, "sparse|under-allocated"):
                validate_complete_wheelhouse(
                    root=root,
                    wheelhouse=wheelhouse,
                    lock_path=lock,
                    expected_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                    expected_artifact_sizes={digest: sparse.stat().st_size},
                    expected_requirement_count=1,
                )

    def test_execution_refuses_unverified_network_and_observes_every_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, lock_sha, _models = self._execution_fixture(root)
            environment_paths = self._environment_paths(root)
            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.CURRENT_LOCK_SHA256",
                lock_sha,
            ), mock.patch("subprocess.Popen") as popen:
                report = run_process_phase(
                    session=session,
                    environment_paths=environment_paths,
                    wheelhouse_evidence=evidence,
                    command=(sys.executable, "-c", "raise SystemExit('must not run')"),
                    phase="runtime",
                    run_id="runtime-baseline-1",
                    scenario="runtime-baseline",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                )
            popen.assert_not_called()
            self.assertEqual(report["execution"]["status"], "refused")
            self.assertEqual(report["network_isolation"]["status"], "network_unverified")
            self.assertFalse(report["measurement_evidence_eligible"])
            self.assertEqual(set(report["path_filesystems"]), set(REQUIRED_OBSERVATION_PATHS))
            self.assertEqual(set(report["high_water"]["paths"]), set(REQUIRED_OBSERVATION_PATHS))
            for snapshot in report["snapshots"]:
                self.assertEqual(set(snapshot["paths"]), set(REQUIRED_OBSERVATION_PATHS))
            self.assertTrue(report["cleanup"]["filesystem_complete"], report["cleanup"])
            self.assertFalse(report["cleanup"]["complete"])
            self.assertEqual(report["cleanup"]["completion_scope"], "filesystem_only")
            self.assertEqual(
                report["cleanup"]["process_observation"],
                "not_observed_execution_refused",
            )
            self.assertIsNone(report["cleanup"]["process_residue"])
            self.assertIsNone(report["cleanup"]["worker_residue"])
            self.assertIsNone(report["cleanup"]["deleted_open_files"])
            for path in environment_paths.values():
                self.assertFalse(path.exists())

    def test_worker_observation_is_required_in_addition_to_receipts(self) -> None:
        report = _validate_worker_observations(
            (),
            scenario="model-baseline",
            worker_script=None,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(len(report["errors"]), 2)

    def test_cleanup_removes_only_new_owned_entries_and_reports_unowned_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline = capture_root_inventory(root)
            (owned / "partial.bin").write_bytes(b"partial")
            unowned = root / "outside-owned-surface.bin"
            unowned.write_bytes(b"preserve")
            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={"owned": owned},
                baseline_inventory=baseline,
            )
            self.assertFalse((owned / "partial.bin").exists())
            self.assertTrue(unowned.is_file())
            self.assertFalse(cleanup["complete"])
            self.assertIn(unowned.name, cleanup["unowned_residue"])

    def test_cleanup_never_deletes_an_undeclared_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            baseline = capture_root_inventory(root)
            undeclared_parent = root / "undeclared-parent"
            owned = undeclared_parent / "owned"
            owned.mkdir(parents=True)
            (owned / "partial.bin").write_bytes(b"partial")
            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={"owned": owned},
                baseline_inventory=baseline,
            )
            self.assertFalse((owned / "partial.bin").exists())
            self.assertFalse(owned.exists())
            self.assertTrue(undeclared_parent.is_dir())
            self.assertIn("undeclared-parent", cleanup["unowned_residue"])
            self.assertNotIn("undeclared-parent", cleanup["removed"])

    def test_cleanup_removes_nested_owned_additions_through_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline = capture_root_inventory(root)
            nested_file = owned / "first" / "second" / "partial.bin"
            nested_file.parent.mkdir(parents=True)
            nested_file.write_bytes(b"partial")

            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={"owned": owned},
                baseline_inventory=baseline,
            )

            self.assertTrue(cleanup["filesystem_complete"], cleanup)
            self.assertEqual(cleanup["filesystem_cleanup_status"], "complete")
            self.assertEqual(
                cleanup["deletion_strategy"], "descriptor_anchored_no_follow"
            )
            self.assertEqual(cleanup["descriptor_cleanup_capability"], "available")
            self.assertFalse(nested_file.exists())
            self.assertTrue(owned.is_dir())
            self.assertEqual(
                cleanup["removed"],
                [
                    "owned/first",
                    "owned/first/second",
                    "owned/first/second/partial.bin",
                ],
            )

    def test_cleanup_allows_owned_directory_identity_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline = capture_root_inventory(root)
            after = copy.deepcopy(baseline)
            after["owned"]["inode"] += 1

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.capture_root_inventory",
                side_effect=[after, after],
            ):
                cleanup = cleanup_owned_artifacts(
                    root=root,
                    owned_paths={"owned": owned},
                    baseline_inventory=baseline,
                )

            self.assertTrue(cleanup["filesystem_complete"], cleanup)
            self.assertEqual(cleanup["changed_baseline_entries"], [])
            self.assertEqual(
                cleanup["expected_owned_directory_changes"], ["owned"]
            )

    def test_cleanup_evidence_allows_unavailable_optional_observations(self) -> None:
        cleanup = {
            "complete": True,
            "filesystem_complete": True,
            "filesystem_cleanup_status": "complete",
            "completion_scope": "filesystem_and_worker",
            "filesystem_residue": [],
            "unowned_residue": [],
            "missing_baseline_entries": [],
            "changed_baseline_entries": [],
            "errors": [],
            "process_residue": None,
            "worker_residue": None,
            "deleted_open_files": None,
            "process_observation": "not_observed_execution_refused",
            "worker_observation": "complete",
            "deleted_open_file_observation": "not_observed_execution_refused",
        }
        self.assertTrue(_cleanup_evidence_is_complete(cleanup))

        cleanup["deleted_open_file_observation"] = "observed"
        cleanup["deleted_open_files"] = ["worker.pid:fd=3"]
        self.assertFalse(_cleanup_evidence_is_complete(cleanup))

    def test_cleanup_refuses_intermediate_symlink_swap_without_outside_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline = capture_root_inventory(root)
            intermediate = owned / "intermediate"
            intermediate.mkdir()
            (intermediate / "partial.bin").write_bytes(b"inside")
            outside_root = Path(outside).resolve()
            outside_victim = outside_root / "victim.bin"
            outside_victim.write_bytes(b"preserve")
            displaced = root / "displaced-intermediate"
            real_capture = capture_root_inventory
            capture_count = 0

            def capture_then_swap(captured_root: Path):
                nonlocal capture_count
                inventory = real_capture(captured_root)
                capture_count += 1
                if capture_count == 1:
                    intermediate.rename(displaced)
                    intermediate.symlink_to(outside_root, target_is_directory=True)
                return inventory

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.capture_root_inventory",
                side_effect=capture_then_swap,
            ):
                cleanup = cleanup_owned_artifacts(
                    root=root,
                    owned_paths={"owned": owned},
                    baseline_inventory=baseline,
                )

            self.assertEqual(capture_count, 2)
            self.assertEqual(outside_victim.read_bytes(), b"preserve")
            self.assertTrue(intermediate.is_symlink())
            self.assertEqual(cleanup["removed"], [])
            self.assertFalse(cleanup["filesystem_complete"])
            self.assertEqual(cleanup["filesystem_cleanup_status"], "incomplete")
            self.assertTrue(cleanup["errors"], cleanup)
            self.assertIn("displaced-intermediate", cleanup["unowned_residue"])

    def test_cleanup_refuses_when_descriptor_capabilities_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline = capture_root_inventory(root)
            partial = owned / "partial.bin"
            partial.write_bytes(b"preserve")

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness._descriptor_cleanup_capability_error",
                return_value="injected descriptor capability gap",
            ), mock.patch(
                "components.Chatterbox.runtime.measurement_harness.os.unlink"
            ) as unlink, mock.patch(
                "components.Chatterbox.runtime.measurement_harness.os.rmdir"
            ) as rmdir:
                cleanup = cleanup_owned_artifacts(
                    root=root,
                    owned_paths={"owned": owned},
                    baseline_inventory=baseline,
                )

            unlink.assert_not_called()
            rmdir.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"preserve")
            self.assertEqual(cleanup["removed"], [])
            self.assertFalse(cleanup["filesystem_complete"])
            self.assertEqual(cleanup["filesystem_cleanup_status"], "refused")
            self.assertEqual(cleanup["descriptor_cleanup_capability"], "unavailable")
            self.assertIn("injected descriptor capability gap", cleanup["errors"][0]["error"])

    def test_cleanup_detects_same_size_baseline_content_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline_file = owned / "baseline.bin"
            baseline_file.write_bytes(b"before")
            baseline = capture_root_inventory(root)
            baseline_file.write_bytes(b"after!")
            self.assertEqual(baseline_file.stat().st_size, 6)
            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={"owned": owned},
                baseline_inventory=baseline,
            )
            self.assertTrue(baseline_file.is_file())
            self.assertIn("owned/baseline.bin", cleanup["changed_baseline_entries"])
            self.assertFalse(cleanup["filesystem_complete"])
            self.assertEqual(cleanup["completion_scope"], "filesystem_only")
            self.assertEqual(
                cleanup["deleted_open_file_observation"],
                "not_observed_execution_refused",
            )

    def test_cleanup_detects_middle_replacement_in_large_baseline_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            owned = root / "owned"
            owned.mkdir()
            baseline_file = owned / "large-baseline.bin"
            file_size = 512 * 1024
            baseline_file.write_bytes(b"a" * file_size)
            baseline = capture_root_inventory(root)
            self.assertEqual(
                baseline["owned/large-baseline.bin"]["fingerprint_method"],
                "sha256-full-v1",
            )

            with baseline_file.open("r+b") as stream:
                stream.seek(file_size // 2)
                stream.write(b"b")
            self.assertEqual(baseline_file.stat().st_size, file_size)

            cleanup = cleanup_owned_artifacts(
                root=root,
                owned_paths={"owned": owned},
                baseline_inventory=baseline,
            )
            self.assertTrue(baseline_file.is_file())
            self.assertIn(
                "owned/large-baseline.bin", cleanup["changed_baseline_entries"]
            )
            self.assertFalse(cleanup["filesystem_complete"])

    def _filesystem_metadata(self, filesystem_id: str, free_bytes: int) -> dict:
        device = int(filesystem_id.partition(":")[2])
        return {
            "filesystem_id": filesystem_id,
            "probe_path": "/measurement",
            "device": device,
            "block_size": 4096,
            "fragment_size": 4096,
            "free_bytes": free_bytes,
            "total_bytes": 100_000,
            "source": f"/dev/loop{device}",
            "mount_point": f"/measurement/fs{device}",
            "filesystem_type": "ext4",
            "mount_options": ["rw"],
            "super_options": ["rw"],
            "metadata_source": "/proc/self/mountinfo",
            "quota_indicators": [],
            "cow_indicators": [],
            "compression_indicators": [],
            "reflink_support": "unverified",
            "overlay": False,
        }

    def _aggregate_report(
        self,
        run_id: str,
        value: int,
        *,
        scenario: str = "runtime-baseline",
        phase: str = "runtime",
        layout: str = "same-filesystem",
        filesystem_drops: dict[str, int] | None = None,
    ):
        if filesystem_drops is None:
            filesystem_drops = {"device:1": value}
        filesystem_ids = tuple(filesystem_drops)
        path_filesystems = {
            name: (
                filesystem_ids[1]
                if layout == "split-filesystem"
                and name.startswith("environment.")
                else filesystem_ids[0]
            )
            for name in REQUIRED_OBSERVATION_PATHS
        }
        path_locations = {
            name: f"/measurement/{layout}/{name}" for name in REQUIRED_OBSERVATION_PATHS
        }
        configured_paths = {
            name: path_locations[f"environment.{name}"]
            for name in REQUIRED_ENVIRONMENT_PATHS
        }
        start_paths = {
            name: {
                "path": path_locations[name],
                "entries": 0,
                "logical_bytes": 0,
                "allocated_bytes": 0,
            }
            for name in REQUIRED_OBSERVATION_PATHS
        }
        high_paths = {
            name: {
                "path": path_locations[name],
                "entries": 1 if value else 0,
                "logical_bytes": value,
                "allocated_bytes": value,
            }
            for name in REQUIRED_OBSERVATION_PATHS
        }
        start_filesystems = {
            filesystem_id: self._filesystem_metadata(filesystem_id, 10_000)
            for filesystem_id in filesystem_ids
        }
        high_filesystems = {
            filesystem_id: self._filesystem_metadata(
                filesystem_id, 10_000 - filesystem_drops[filesystem_id]
            )
            for filesystem_id in filesystem_ids
        }
        snapshots = [
            {
                "timestamp_ns": 1,
                "stage": "start",
                "paths": start_paths,
                "path_filesystems": path_filesystems,
                "filesystems": start_filesystems,
                "owned_logical_bytes": 0,
                "owned_allocated_bytes": 0,
            },
            {
                "timestamp_ns": 2,
                "stage": "high_water",
                "paths": high_paths,
                "path_filesystems": path_filesystems,
                "filesystems": high_filesystems,
                "owned_logical_bytes": value * len(REQUIRED_OBSERVATION_PATHS),
                "owned_allocated_bytes": value * len(REQUIRED_OBSERVATION_PATHS),
            },
        ]
        value_classification = "observed_zero" if value == 0 else "observed_nonzero"
        maximum_stages = ["start", "high_water"] if value == 0 else ["high_water"]
        return {
            "schema": HARNESS_SCHEMA,
            "mode": "report_only",
            "run_id": run_id,
            "phase": phase,
            "scenario": scenario,
            "target": {"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
            "environment": {
                "environment_id": f"clean-amd64-{layout}",
                "filesystem_layout": layout,
                "clean_target": True,
                "configured_paths": configured_paths,
            },
            "network_isolation": {"status": "enforced", "enforced": True, "capability": "future-isolated-runner"},
            "path_filesystems": path_filesystems,
            "snapshots": snapshots,
            "raw_observations": snapshots,
            "cleanup": {
                "complete": True, "filesystem_complete": True,
                "filesystem_cleanup_status": "complete",
                "completion_scope": "filesystem_and_process",
                "filesystem_residue": [], "unowned_residue": [], "process_residue": [],
                "worker_residue": [], "deleted_open_files": [], "missing_baseline_entries": [],
                "changed_baseline_entries": [], "errors": [], "process_observation": "complete",
                "worker_observation": "complete", "deleted_open_file_observation": "complete",
            },
            "high_water": {
                "owned_logical_bytes": value * len(REQUIRED_OBSERVATION_PATHS),
                "owned_allocated_bytes": value * len(REQUIRED_OBSERVATION_PATHS),
                "paths": {
                    name: {
                        "allocated_bytes": value,
                        "logical_bytes": value,
                        "observation_count": 2,
                        "logical_value_classification": value_classification,
                        "allocated_value_classification": value_classification,
                        "logical_maximum_stages": maximum_stages,
                        "allocated_maximum_stages": maximum_stages,
                    }
                    for name in REQUIRED_OBSERVATION_PATHS
                },
                "filesystems": {
                    filesystem_id: {
                        "baseline_free_bytes": 10_000,
                        "minimum_free_bytes": 10_000 - drop,
                        "maximum_free_space_drop_bytes": drop,
                        "observation_count": 2,
                        "value_classification": (
                            "observed_zero" if drop == 0 else "observed_nonzero"
                        ),
                        "minimum_free_stages": (
                            ["start", "high_water"] if drop == 0 else ["high_water"]
                        ),
                        "metadata": high_filesystems[filesystem_id],
                    }
                    for filesystem_id, drop in filesystem_drops.items()
                },
            },
            "measurement_evidence_eligible": True,
        }

    def _optional_variance_matrix_reports(self) -> list[dict]:
        reports: list[dict] = []
        matrix = (
            ("runtime-baseline", "runtime", "same-filesystem", ((100, 0), (125, 0), (110, 0))),
            ("model-baseline", "model", "same-filesystem", ((140, 0), (130, 0), (135, 0))),
            ("runtime-baseline", "runtime", "split-filesystem", ((80, 200), (90, 210), (85, 205))),
            ("model-baseline", "model", "split-filesystem", ((150, 250), (145, 240), (148, 245))),
        )
        for scenario, phase, layout, drops in matrix:
            for index, (first_drop, second_drop) in enumerate(drops, 1):
                filesystem_drops = {"device:1": first_drop}
                if layout == "split-filesystem":
                    filesystem_drops["device:2"] = second_drop
                reports.append(
                    self._aggregate_report(
                        f"{scenario}-{layout}-{index}",
                        max(filesystem_drops.values()),
                        scenario=scenario,
                        phase=phase,
                        layout=layout,
                        filesystem_drops=filesystem_drops,
                    )
                )
        return reports

    def _aggregate_arguments(self) -> dict:
        return {
            "reserve_percent": 10,
            "reserve_bytes": {"device:1": 5, "device:2": 7},
            "required_scenarios": CANONICAL_SCENARIOS,
            "required_filesystem_layouts": CANONICAL_FILESYSTEM_LAYOUTS,
        }

    def test_optional_variance_aggregation_accepts_explicit_matrix(self) -> None:
        reports = self._optional_variance_matrix_reports()
        aggregate = aggregate_repeated_runs(reports, **self._aggregate_arguments())
        self.assertEqual(len(REQUIRED_OBSERVATION_PATHS), 15)
        self.assertEqual(len(aggregate["group_maxima"]), 4)
        self.assertEqual(
            aggregate["per_filesystem_capacity"]["device:1"],
            {
                "maximum_simultaneous_high_water_bytes": 150,
                "reserve": {
                    "percent": 10,
                    "percentage_bytes": 15,
                    "fixed_bytes": 5,
                    "total_bytes": 20,
                },
                "proposed_capacity_bytes": 170,
            },
        )
        self.assertEqual(
            aggregate["per_filesystem_capacity"]["device:2"]["proposed_capacity_bytes"],
            282,
        )
        self.assertEqual(aggregate["raw_observation_evidence"]["retention"], "embedded")
        self.assertEqual(len(aggregate["raw_observation_evidence"]["runs"]), 12)
        self.assertNotIn("raw_observations_retained", aggregate)
        reports[0]["raw_observations"][0]["paths"][REQUIRED_OBSERVATION_PATHS[0]][
            "logical_bytes"
        ] = 999
        retained = aggregate["raw_observation_evidence"]["runs"][0]["observations"]
        self.assertNotEqual(
            retained[0]["paths"][REQUIRED_OBSERVATION_PATHS[0]]["logical_bytes"],
            999,
        )
        self.assertTrue(aggregate["measurement_evidence_candidate"])

    def test_optional_variance_aggregation_accepts_verified_offline_input_provenance(self) -> None:
        reports = self._optional_variance_matrix_reports()
        for report in reports:
            report["network_isolation"] = {
                "status": "offline_inputs",
                "enforced": False,
                "capability": "verified_local_inputs",
            }
        aggregate = aggregate_repeated_runs(reports, **self._aggregate_arguments())
        self.assertTrue(aggregate["measurement_evidence_candidate"])

    def test_optional_variance_aggregation_rejects_sparse_and_ambiguous_records(self) -> None:
        reports = self._optional_variance_matrix_reports()
        arguments = self._aggregate_arguments()
        for mutation, message in (
            (("run_id", reports[0]["run_id"]), "unique"),
            (("raw_observations", []), "raw"),
            (("measurement_evidence_eligible", False), "eligible"),
        ):
            broken = copy.deepcopy(reports)
            broken[1][mutation[0]] = mutation[1]
            with self.subTest(field=mutation[0]), self.assertRaisesRegex(MeasurementConfigurationError, message):
                aggregate_repeated_runs(broken, **arguments)
        incomplete_group = [
            report
            for report in reports
            if not (
                report["scenario"] == "model-baseline"
                and report["environment"]["filesystem_layout"] == "split-filesystem"
            )
        ]
        aggregate = aggregate_repeated_runs(incomplete_group, **arguments)
        self.assertEqual(aggregate["run_count"], 9)
        missing_path = copy.deepcopy(reports)
        missing_path[0]["environment"]["configured_paths"].pop("tmpdir")
        with self.assertRaisesRegex(MeasurementConfigurationError, "environment-path"):
            aggregate_repeated_runs(missing_path, **arguments)
        residue = copy.deepcopy(reports)
        residue[0]["cleanup"]["filesystem_residue"] = ["partial.bin"]
        with self.assertRaisesRegex(MeasurementConfigurationError, "residue"):
            aggregate_repeated_runs(residue, **arguments)

        incomplete_numeric = copy.deepcopy(reports)
        incomplete_numeric[0]["high_water"]["paths"][REQUIRED_OBSERVATION_PATHS[0]].pop(
            "allocated_bytes"
        )
        with self.assertRaisesRegex(MeasurementConfigurationError, "non-negative integer"):
            aggregate_repeated_runs(incomplete_numeric, **arguments)

        empty_path = copy.deepcopy(reports)
        empty_path[0]["raw_observations"][0]["paths"][REQUIRED_OBSERVATION_PATHS[0]] = {}
        with self.assertRaisesRegex(MeasurementConfigurationError, "location|per-path"):
            aggregate_repeated_runs(empty_path, **arguments)

        single_snapshot = copy.deepcopy(reports)
        single_snapshot[0]["raw_observations"] = single_snapshot[0]["raw_observations"][:1]
        with self.assertRaisesRegex(MeasurementConfigurationError, "at least two"):
            aggregate_repeated_runs(single_snapshot, **arguments)

        zero_reports = copy.deepcopy(reports)
        zero_reports[0]["high_water"]["filesystems"]["device:1"].pop(
            "value_classification"
        )
        with self.assertRaisesRegex(MeasurementConfigurationError, "zero/nonzero"):
            aggregate_repeated_runs(zero_reports, **arguments)

    def test_optional_variance_aggregation_uses_only_caller_selected_coverage(self) -> None:
        reports = self._optional_variance_matrix_reports()
        arguments = self._aggregate_arguments()

        arbitrary_scenarios = dict(arguments)
        arbitrary_scenarios["required_scenarios"] = (
            "runtime-baseline",
            "caller-invented",
        )
        with self.assertRaisesRegex(MeasurementConfigurationError, "scenario coverage"):
            aggregate_repeated_runs(reports, **arbitrary_scenarios)

        arbitrary_layouts = dict(arguments)
        arbitrary_layouts["required_filesystem_layouts"] = (
            "same-filesystem",
            "caller-layout",
        )
        with self.assertRaisesRegex(MeasurementConfigurationError, "filesystem-layout coverage"):
            aggregate_repeated_runs(reports, **arbitrary_layouts)

        arbitrary_report = copy.deepcopy(reports)
        arbitrary_report[0]["scenario"] = "caller-invented"
        with self.assertRaisesRegex(MeasurementConfigurationError, "scenario coverage"):
            aggregate_repeated_runs(arbitrary_report, **arguments)

        arbitrary_report_layout = copy.deepcopy(reports)
        arbitrary_report_layout[0]["environment"]["filesystem_layout"] = "caller-layout"
        with self.assertRaisesRegex(MeasurementConfigurationError, "filesystem-layout coverage"):
            aggregate_repeated_runs(arbitrary_report_layout, **arguments)

        wrong_phase = copy.deepcopy(reports)
        wrong_phase[0]["phase"] = "model"
        with self.assertRaisesRegex(MeasurementConfigurationError, "requires phase runtime"):
            aggregate_repeated_runs(wrong_phase, **arguments)

        missing_canonical_group = [
            report
            for report in reports
            if not (
                report["scenario"] == "model-baseline"
                and report["environment"]["filesystem_layout"] == "split-filesystem"
            )
        ]
        aggregate = aggregate_repeated_runs(missing_canonical_group, **arguments)
        self.assertEqual(aggregate["run_count"], 9)

    def test_optional_variance_aggregation_rejects_incomparable_metadata_or_requested_coverage(self) -> None:
        reports = self._optional_variance_matrix_reports()
        arguments = self._aggregate_arguments()

        incomplete_matrix = [
            report
            for report in reports
            if not (
                report["scenario"] == "model-baseline"
                and report["environment"]["filesystem_layout"] == "split-filesystem"
            )
        ]
        aggregate = aggregate_repeated_runs(incomplete_matrix, **arguments)
        self.assertEqual(aggregate["run_count"], 9)

        incomplete_group = [
            report
            for report in reports
            if not (
                report["scenario"] == "model-baseline"
                and report["environment"]["filesystem_layout"] == "split-filesystem"
            )
        ]
        aggregate = aggregate_repeated_runs(incomplete_group, **arguments)
        self.assertEqual(aggregate["run_count"], 9)

        one_per_group = [
            reports[index]
            for index in (0, 3, 6, 9)
        ]
        aggregate = aggregate_repeated_runs(one_per_group, **arguments)
        self.assertEqual(aggregate["run_count"], 4)

        incomparable_environment = copy.deepcopy(reports)
        incomparable_environment[1]["environment"]["environment_id"] = "different-target"
        with self.assertRaisesRegex(MeasurementConfigurationError, "incomparable environment"):
            aggregate_repeated_runs(incomparable_environment, **arguments)

        incomparable_path = copy.deepcopy(reports)
        for observation in incomparable_path[1]["raw_observations"]:
            observation["paths"]["surface.runtime"]["path"] = "/different/runtime"
        with self.assertRaisesRegex(MeasurementConfigurationError, "incomparable path"):
            aggregate_repeated_runs(incomparable_path, **arguments)

        incomparable_filesystem = copy.deepcopy(reports)
        for observation in incomparable_filesystem[1]["raw_observations"]:
            observation["filesystems"]["device:1"]["source"] = "/dev/loop99"
        incomparable_filesystem[1]["high_water"]["filesystems"]["device:1"]["metadata"][
            "source"
        ] = "/dev/loop99"
        with self.assertRaisesRegex(MeasurementConfigurationError, "incomparable filesystem"):
            aggregate_repeated_runs(incomparable_filesystem, **arguments)

        incomparable_layout_environment = copy.deepcopy(reports)
        for report in incomparable_layout_environment:
            if (
                report["scenario"] == "model-baseline"
                and report["environment"]["filesystem_layout"] == "same-filesystem"
            ):
                report["environment"]["environment_id"] = "different-same-layout-target"
        with self.assertRaisesRegex(MeasurementConfigurationError, "layout.*incomparable environment"):
            aggregate_repeated_runs(incomparable_layout_environment, **arguments)

        incomparable_filesystem_identity = copy.deepcopy(reports)
        for report in incomparable_filesystem_identity:
            if report["environment"]["filesystem_layout"] == "split-filesystem":
                for observation in report["raw_observations"]:
                    observation["filesystems"]["device:1"]["source"] = "/dev/loop99"
                report["high_water"]["filesystems"]["device:1"]["metadata"][
                    "source"
                ] = "/dev/loop99"
        with self.assertRaisesRegex(MeasurementConfigurationError, "filesystem identity"):
            aggregate_repeated_runs(incomparable_filesystem_identity, **arguments)

        incomplete_reserve = dict(arguments)
        incomplete_reserve["reserve_bytes"] = {"device:1": 5}
        with self.assertRaisesRegex(MeasurementConfigurationError, "reserve map"):
            aggregate_repeated_runs(reports, **incomplete_reserve)

    def test_transaction_hook_is_never_invoked_without_network_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, lock_sha, _models = self._execution_fixture(root)
            paths = self._environment_paths(root)
            called: list[str] = []

            def fake_hook(invocation):
                called.append(invocation.kind)
                return {"must_not": "run"}

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.CURRENT_LOCK_SHA256",
                lock_sha,
            ):
                report = invoke_transaction_hook(
                    session=session,
                    hooks=TransactionHooks(runtime=fake_hook),
                    kind="runtime",
                    environment_paths=paths,
                    wheelhouse_evidence=evidence,
                    run_id="hook-refusal-1",
                    scenario="runtime-baseline",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                )
            self.assertEqual(called, [])
            self.assertFalse(report["hook_invoked"])
            self.assertEqual(report["execution"]["reason"], "network_unverified")

    def test_every_attempt_revalidates_complete_non_sparse_inputs_before_hook(self) -> None:
        mutations = ("incomplete", "foreign", "wrong-size", "wrong-hash", "sparse-model")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = self._root(directory)
                session, evidence, lock_sha, models = self._execution_fixture(root)
                if mutation == "incomplete":
                    Path(evidence.artifacts[0]["path"]).unlink()
                elif mutation == "foreign":
                    _wheel(session.wheelhouse / "foreign-1.0-py3-none-any.whl", b"foreign")
                elif mutation == "wrong-size":
                    package = Path(evidence.artifacts[0]["path"])
                    package.write_bytes(package.read_bytes()[:-1])
                elif mutation == "wrong-hash":
                    package = Path(evidence.artifacts[0]["path"])
                    content = bytearray(package.read_bytes())
                    content[-1] ^= 1
                    package.write_bytes(content)
                else:
                    model = models[0]
                    model.path.unlink()
                    with model.path.open("wb") as stream:
                        stream.seek(model.size_bytes - 1)
                        stream.write(b"\0")
                called: list[str] = []

                def fake_hook(invocation):
                    called.append(invocation.kind)
                    return {}

                with mock.patch(
                    "components.Chatterbox.runtime.measurement_harness.CURRENT_LOCK_SHA256",
                    lock_sha,
                ), self.assertRaises(MeasurementConfigurationError):
                    invoke_transaction_hook(
                        session=session,
                        hooks=TransactionHooks(runtime=fake_hook),
                        kind="runtime",
                        environment_paths=self._environment_paths(root),
                        wheelhouse_evidence=evidence,
                        run_id=f"invalid-{mutation}",
                        scenario="runtime-baseline",
                        target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                        environment_id="fixture-amd64",
                        filesystem_layout="same-filesystem",
                    )
                self.assertEqual(called, [])

    def test_execution_requires_current_lock_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, _lock_sha, _models = self._execution_fixture(root)
            with self.assertRaisesRegex(MeasurementConfigurationError, "current 107-entry lock"):
                run_process_phase(
                    session=session,
                    environment_paths=self._environment_paths(root),
                    wheelhouse_evidence=evidence,
                    command=(sys.executable, "-c", "pass"),
                    phase="runtime",
                    run_id="wrong-lock",
                    scenario="runtime-baseline",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                )

    def test_disposable_execution_requires_the_marked_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, _lock_sha, _models = self._execution_fixture(root)
            (root / DISPOSABLE_MARKER).unlink()
            with self.assertRaisesRegex(MeasurementConfigurationError, "marker"):
                run_disposable_measurement(
                    session=session,
                    wheelhouse_evidence=evidence,
                    environment_paths=self._environment_paths(root),
                    runtime_paths=SimpleNamespace(),
                    interpreter=Path(sys.executable),
                    run_id="measurement-1",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                )

    def test_disposable_execution_rejects_unobserved_mutable_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, _lock_sha, _models = self._execution_fixture(root)
            environment_paths = self._environment_paths(root)
            receipts = root / "environment" / "xdg_state" / "receipts"
            runtime_paths = SimpleNamespace(
                lock_path=evidence.lock_path,
                e2a_root=root / "unobserved-runtime",
                runtime_receipt_path=receipts / "runtime.json",
                model_receipt_path=receipts / "model.json",
                activation_receipt_path=receipts / "activation.json",
            )
            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.CURRENT_LOCK_SHA256",
                _lock_sha,
            ), mock.patch(
                "components.Chatterbox.runtime.runtime.validate_measurement_scope",
                return_value=root,
            ), self.assertRaisesRegex(MeasurementConfigurationError, "outside the observed"):
                run_disposable_measurement(
                    session=session,
                    wheelhouse_evidence=evidence,
                    environment_paths=environment_paths,
                    runtime_paths=runtime_paths,
                    interpreter=Path(sys.executable),
                    run_id="measurement-unobserved-path",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                )

    def test_disposable_execution_calls_install_and_model_primitives_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            session, evidence, lock_sha, _models = self._execution_fixture(
                root, include_worker_data=True
            )
            environment_paths = self._environment_paths(root)
            state = root / "environment" / "xdg_state" / "receipts"
            runtime_receipt = state / "runtime-receipt.json"
            model_receipt = state / "model-receipt.json"
            activation_receipt = state / "activation-receipt.json"
            worker_script = root / "worker.py"
            worker_script.write_text("# disposable worker fixture\n", encoding="utf-8")
            runtime_paths = SimpleNamespace(
                lock_path=evidence.lock_path,
                run_namespace=root / "surface" / "run",
                runtime_receipt_path=runtime_receipt,
                model_receipt_path=model_receipt,
                activation_receipt_path=activation_receipt,
            )

            def emit_worker_observation(observer, *, kind, event, flag):
                if observer is not None:
                    observer(
                        {
                            "kind": kind,
                            "command": [sys.executable, str(worker_script), flag],
                            "execution_observed": True,
                            "returncode": 0,
                            "expected_event": event,
                            "stdout_events": [{"event": event, "ok": True}],
                            "stdout_tail": json.dumps({"event": event, "ok": True}),
                            "stderr_tail": "",
                        }
                    )

            def fake_install(paths, _interpreter, *, worker_observer=None, **_kwargs):
                paths.runtime_receipt_path.parent.mkdir(parents=True, exist_ok=True)
                paths.runtime_receipt_path.write_text("runtime", encoding="utf-8")
                emit_worker_observation(
                    worker_observer,
                    kind="runtime_self_test",
                    event="self_test",
                    flag="--self-test",
                )
                return {"ok": True, "status": "ready"}

            def fake_acquire(paths, *, worker_observer=None, **_kwargs):
                paths.model_receipt_path.write_text("model", encoding="utf-8")
                paths.activation_receipt_path.write_text("activation", encoding="utf-8")
                emit_worker_observation(
                    worker_observer,
                    kind="model_self_test",
                    event="model_self_test",
                    flag="--model-self-test",
                )
                return {"ok": True, "status": "ready"}

            with mock.patch(
                "components.Chatterbox.runtime.measurement_harness.CURRENT_LOCK_SHA256",
                lock_sha,
            ), mock.patch(
                "components.Chatterbox.runtime.measurement_harness.PKUSEG_DATA_SHA256",
                session.revalidate_inputs()["worker_data"][0]["sha256"],
            ), mock.patch(
                "components.Chatterbox.runtime.runtime.validate_measurement_scope",
                return_value=root,
            ), mock.patch(
                "components.Chatterbox.runtime.runtime.install_runtime",
                side_effect=fake_install,
            ) as install, mock.patch(
                "components.Chatterbox.runtime.runtime.acquire_model",
                side_effect=fake_acquire,
            ) as acquire, mock.patch(
                "components.Chatterbox.runtime.runtime.product_status",
                return_value={"ok": True, "status": "ready"},
            ):
                report = run_disposable_measurement(
                    session=session,
                    wheelhouse_evidence=evidence,
                    environment_paths=environment_paths,
                    runtime_paths=runtime_paths,
                    interpreter=Path(sys.executable),
                    run_id="measurement-1",
                    scenario="model-baseline",
                    target={"os": "linux", "architecture": "x86_64", "python": "3.11", "backend": "cpu"},
                    environment_id="fixture-amd64",
                    filesystem_layout="same-filesystem",
                    worker_script=worker_script,
                    sample_interval_seconds=0.01,
                )

            self.assertEqual(report["execution"]["status"], "completed", report)
            self.assertEqual(report["execution"]["phases"]["runtime"]["status"], "completed")
            self.assertEqual(report["execution"]["phases"]["model"]["status"], "completed")
            install.assert_called_once()
            acquire.assert_called_once()
            self.assertTrue(all(item["observed"] for item in report["execution"]["receipts"].values()))
            self.assertTrue(report["high_water"]["filesystems"])
            self.assertTrue(report["measurement_evidence_eligible"])
            self.assertTrue(report["execution"]["worker_observation"]["ok"])
            self.assertTrue(report["cleanup"]["complete"], report["cleanup"])
            self.assertEqual(
                report["cleanup"]["completion_scope"], "filesystem_and_worker"
            )
            self.assertEqual(
                report["cleanup"]["process_observation"],
                "not_observed_execution_refused",
            )
            self.assertEqual(
                report["cleanup"]["deleted_open_file_observation"],
                "not_observed_execution_refused",
            )
            self.assertFalse(runtime_receipt.exists())
            self.assertTrue((root / "surface" / "model" / CANONICAL_MODEL_FILE_PATHS[0]).is_file())

    def test_harness_has_no_normal_install_or_host_reachability(self) -> None:
        repository = Path(__file__).resolve().parents[4]
        files = (
            repository / "lib/classes/tts_engines/chatterbox.py",
            repository / "lib/core.py",
            repository / "lib/gradio.py",
        )
        for path in files:
            with self.subTest(path=path):
                self.assertNotIn("measurement_harness", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
