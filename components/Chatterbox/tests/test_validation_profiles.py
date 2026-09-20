"""Tests for the opt-in Chatterbox CPU V2 validation profiles."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from components.Chatterbox.runtime import contract_data


ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = ROOT / "components" / "Chatterbox" / "validate_cpu_v2.py"
SPEC = importlib.util.spec_from_file_location("chatterbox_cpu_v2_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

WORKER_PATH = ROOT / "components" / "Chatterbox" / "worker.py"
WORKER_SPEC = importlib.util.spec_from_file_location("chatterbox_worker_validation_profiles", WORKER_PATH)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


class ValidationProfileTests(unittest.TestCase):
    def test_short_23_cases_are_exact_and_independent(self):
        cases = validator.SHORT_23_CASES
        self.assertEqual(len(cases), 23)
        self.assertEqual(len({case.case_id for case in cases}), 23)
        self.assertEqual({case.language for case in cases}, set(worker.SUPPORTED_LANGUAGES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in cases:
                with self.subTest(case=case.case_id):
                    request = self._request(root, case)
                    normalized = worker.validate_request(request)
                    self.assertEqual(normalized["language"], case.language)
                    self.assertEqual(normalized["segments"][0]["text"], case.text)
                    self.assertEqual(normalized["output"], root / f"{case.case_id}.flac")

    def test_long_form_script_categories_are_independent(self):
        cases = validator.LONG_FORM_CASES
        self.assertEqual({case.category for case in cases}, {
            "latin", "rtl", "cjk", "devanagari-combining"
        })
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertTrue(any("़" in case.text for case in cases if case.language == "hi"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in cases:
                with self.subTest(case=case.case_id):
                    normalized = worker.validate_request(self._request(root, case))
                    self.assertEqual(normalized["language"], case.language)
                    self.assertEqual(normalized["segments"][0]["text"], case.text)
                    self.assertGreater(len(case.text), 60)

    def test_host_smoke_contract_is_separate_and_non_synthesizing(self):
        self.assertEqual(len(validator.HOST_SMOKE_CASES), 1)
        case = validator.HOST_SMOKE_CASES[0]
        self.assertEqual(case.language, "en")
        self.assertEqual(case.category, "sml-resume-combine-ffmpeg")
        self.assertIn("[break]", case.text)
        self.assertIn("[pause:0.25]", case.text)
        report = validator.profile_contract_check(("host-smoke",))
        self.assertEqual(report["status"], "passed")
        self.assertNotIn("short-23", {item["profile"] for item in report["cases"]})

    def test_full_host_gradio_import_preflight(self):
        result = validator.host_import_preflight("lib.gradio")
        if result["status"] == "skipped":
            self.skipTest(result["message"])
        self.assertEqual(result["status"], "passed", result)

    def test_static_contract_is_determinate_and_keeps_exact_model_contract(self):
        result = validator.static_contract_check()
        manifest = json.loads(validator.MANIFEST_PATH.read_text(encoding="utf-8"))
        expected_model_bytes = sum(
            record["size_bytes"]
            for record in manifest["sources"]["model"]["files"]
        )
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["model_files"], list(validator.CANONICAL_MODEL_FILES))
        self.assertEqual(result["model_bytes"], expected_model_bytes)
        self.assertEqual(result["storage_budget"], "determinate")
        self.assertEqual(result["unknown_capacity_buckets"], [])
        self.assertIs(validator.CONTRACT_DATA, contract_data.CONTRACT_DATA)
        self.assertIs(validator.CANONICAL_MODEL_FILES, contract_data.CANONICAL_MODEL_FILE_PATHS)
        self.assertEqual(
            validator.SUPPORTED_TARGET,
            {
                "os": contract_data.TARGET_OS,
                "architecture": contract_data.TARGET_ARCH,
                "backend": contract_data.TARGET_BACKEND,
            },
        )

    def test_static_contract_uses_manifest_sizes_without_a_second_size_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(validator.MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest["sources"]["model"]["files"][0]["size_bytes"] += 1
            manifest_path = root / "runtime-manifest.json"
            lock_path = root / "requirements-cpu.lock"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            lock_path.write_bytes(validator.LOCK_PATH.read_bytes())

            with (
                patch.object(validator, "MANIFEST_PATH", manifest_path),
                patch.object(validator, "LOCK_PATH", lock_path),
            ):
                result = validator.static_contract_check()

        expected = sum(
            record["size_bytes"]
            for record in manifest["sources"]["model"]["files"]
        )
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["model_bytes"], expected)

    def test_static_contract_rejects_model_variant_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(validator.MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest["sources"]["model"]["variant"] = "multilingual-v3"
            manifest_path = root / "runtime-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(validator, "MANIFEST_PATH", manifest_path):
                result = validator.static_contract_check()

        self.assertEqual(result["status"], "failed")
        self.assertIn("pinned Chatterbox V2", result["message"])

    def test_host_import_preflight_reports_available_host(self):
        completed = subprocess.CompletedProcess(
            args=[sys.executable, "-c", "...", "lib.core"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch.object(validator.subprocess, "run", return_value=completed) as run:
            result = validator.host_import_preflight("lib.core")

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["classification"], "available")
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[-1], "lib.core")

    def test_host_import_preflight_skips_declared_external_dependencies(self):
        for dependency in ("stanza", "pytesseract"):
            completed = subprocess.CompletedProcess(
                args=[sys.executable, "-c", "...", "lib.core"],
                returncode=1,
                stdout="",
                stderr=f"ModuleNotFoundError: No module named '{dependency}'\n",
            )
            with self.subTest(dependency=dependency), patch.object(
                validator.subprocess, "run", return_value=completed
            ):
                result = validator.host_import_preflight("lib.core")

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["classification"], "missing-host-dependency")
            self.assertEqual(result["missing_dependency"], dependency)

    def test_host_import_preflight_keeps_internal_and_unknown_modules_as_errors(self):
        for dependency, classification in (
            ("lib.internal", "missing-first-party-module"),
            ("unlisted_dependency", "missing-unknown-module"),
        ):
            completed = subprocess.CompletedProcess(
                args=[sys.executable, "-c", "...", "lib.core"],
                returncode=1,
                stdout="",
                stderr=f"ModuleNotFoundError: No module named '{dependency}'\n",
            )
            with self.subTest(dependency=dependency), patch.object(
                validator.subprocess, "run", return_value=completed
            ):
                result = validator.host_import_preflight("lib.core")

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["classification"], classification)
            self.assertEqual(result["missing_dependency"], dependency)

    def test_selected_profile_methods_exist(self):
        selected = tuple(dict.fromkeys(
            (*validator.COMMON_TESTS,
             *(test for tests in validator.PROFILE_TESTS.values() for test in tests),
             *validator.CORE_LIFECYCLE_TESTS)
        ))
        missing = []
        for selector in selected:
            module_name, class_name, method_name = selector.rsplit(".", 2)
            module_path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
            class_nodes = [
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            methods = {
                node.name
                for class_node in class_nodes
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                missing.append(selector)
        self.assertEqual(missing, [])

    def test_profile_plan_skips_core_lifecycle_before_collection_when_host_is_missing(self):
        missing = {
            "id": "host-import-lib-core",
            "module": "lib.core",
            "status": "skipped",
            "classification": "missing-host-dependency",
            "missing_dependency": "stanza",
            "message": "lib.core import skipped: missing host dependency stanza",
        }
        with patch.object(validator, "host_import_preflight", return_value=missing):
            tests, skips = validator.build_test_plan(("host-smoke",))

        self.assertEqual(len(tests), len(set(tests)))
        self.assertTrue(any("test_real_host_status_rejects" in test for test in tests))
        self.assertTrue(any("test_host_audio_validation_falls_back_to_ffprobe" in test for test in tests))
        self.assertFalse(any("test_core_lifecycle" in test for test in tests))
        self.assertTrue(any("test_full_host_gradio_import_preflight" in test for test in tests))
        self.assertEqual([skip["id"] for skip in skips], ["host-lifecycle-preflight"])
        self.assertEqual(skips[0]["status"], "skipped")
        self.assertEqual(skips[0]["preflight"]["missing_dependency"], "stanza")

    def test_profile_plan_collects_core_lifecycle_when_host_is_available(self):
        available = {
            "id": "host-import-lib-core",
            "module": "lib.core",
            "status": "passed",
            "classification": "available",
            "message": "lib.core imported successfully in a subprocess",
        }
        with patch.object(validator, "host_import_preflight", return_value=available):
            tests, skips = validator.build_test_plan(("host-smoke",))

        self.assertTrue(any("test_core_lifecycle" in test for test in tests))
        self.assertEqual(skips, [])

    def test_live_asset_check_skips_cleanly_without_ready_receipts(self):
        states = (
            ("runtime_missing", "missing", "missing", "missing"),
            ("model_missing", "ready", "missing", "missing"),
            ("activation_missing", "ready", "ready", "missing"),
        )
        for product_state, runtime_state, model_state, activation_state in states:
            status = {
                "ok": False,
                "status": product_state,
                "runtime": {"status": runtime_state},
                "model": {"status": model_state},
                "activation": {"status": activation_state},
            }
            fake_runtime = type("FakeRuntime", (), {
                "build_paths": staticmethod(lambda: object()),
                "product_status": staticmethod(lambda _paths, value=status: value),
            })
            with (
                self.subTest(product_state=product_state),
                patch.object(validator.platform, "system", return_value="Linux"),
                patch.object(validator.platform, "machine", return_value="x86_64"),
                patch.dict(sys.modules, {
                    "components.Chatterbox.runtime.runtime": fake_runtime,
                }),
            ):
                result = validator.live_asset_check(("short-23",))
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["product_status"], product_state)
            self.assertIn("did not provision, download, repair, or synthesize", result["message"])

    def test_live_asset_check_fails_for_repair_and_other_unsafe_states(self):
        for product_state, dimension_state in (
            ("repair_required", "repair_required"),
            ("publication_ambiguous", "ambiguous"),
        ):
            status = {
                "ok": False,
                "status": product_state,
                "runtime": {"status": dimension_state, "errors": ["unsafe receipt"]},
                "model": {"status": "ready", "errors": []},
                "activation": {"status": "ready", "errors": []},
            }
            fake_runtime = type("FakeRuntime", (), {
                "build_paths": staticmethod(lambda: object()),
                "product_status": staticmethod(lambda _paths, value=status: value),
            })
            with (
                self.subTest(product_state=product_state),
                patch.object(validator.platform, "system", return_value="Linux"),
                patch.object(validator.platform, "machine", return_value="x86_64"),
                patch.dict(sys.modules, {
                    "components.Chatterbox.runtime.runtime": fake_runtime,
                }),
            ):
                result = validator.live_asset_check(("short-23",))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["product_status"], product_state)

    def test_validator_observes_and_removes_owned_temporary_artifacts(self):
        def fake_test_plan(_tests, *, temporary_root):
            (temporary_root / "fake-worker-output.flac.part").write_bytes(b"partial")
            return {
                "id": "unittest-profile",
                "status": "passed",
                "message": "passed",
                "process_observations": [{
                    "pid": 123,
                    "returncode": 0,
                    "process_group_observation": "supported",
                    "residue_before_cleanup": False,
                    "residue_after_cleanup": False,
                }],
            }

        with (
            patch.object(validator, "run_unittest_plan", side_effect=fake_test_plan),
            patch.object(validator, "build_test_plan", return_value=((), [])),
            patch.object(validator, "_identities", return_value={"manifest_sha256": "a" * 64}),
        ):
            report, exit_code = validator.build_report(("short-23",), include_live_assets=False)

        cleanup = report["cleanup"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(cleanup["before_entries"], [])
        self.assertIn("fake-worker-output.flac.part", cleanup["after_test_entries"])
        self.assertIn("fake-worker-output.flac.part", cleanup["partial_paths_observed_after_tests"])
        self.assertTrue(cleanup["root_removed"])
        self.assertFalse(cleanup["partial_files_retained"])
        self.assertEqual(cleanup["spawned_process_residue_after_cleanup"], [])
        self.assertFalse(Path(cleanup["validator_temp_root"]).exists())

    def test_unittest_runner_observes_its_spawned_process_group(self):
        test_name = (
            "components.Chatterbox.tests.test_validation_profiles."
            "ValidationProfileTests.test_host_smoke_contract_is_separate_and_non_synthesizing"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = validator.run_unittest_plan((test_name,), temporary_root=Path(directory))
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(len(result["process_observations"]), 1)
        observation = result["process_observations"][0]
        self.assertEqual(observation["returncode"], 0)
        if os.name == "posix":
            self.assertFalse(observation["residue_before_cleanup"])
            self.assertFalse(observation["residue_after_cleanup"])

    def test_missing_and_malformed_manifests_emit_json_and_exit_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            malformed = root / "malformed.json"
            malformed.write_text("{not-json", encoding="utf-8")
            for manifest in (missing, malformed):
                with self.subTest(manifest=manifest.name):
                    result = self._validator_subprocess(manifest)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["checks"][0]["id"], "static-contract")
                    self.assertEqual(report["checks"][0]["status"], "failed")
                    self.assertIn("identity_error", report["identities"])

    def test_post_contract_exception_emits_json_and_exit_one(self):
        output = io.StringIO()
        with (
            patch.object(validator, "profile_contract_check", side_effect=RuntimeError("injected")),
            redirect_stdout(output),
        ):
            exit_code = validator.main(["--profile", "short-23"])
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(check["id"] == "validation-exception" for check in report["checks"]))

    def test_evidence_contract_never_claims_mutating_validation(self):
        with (
            patch.object(validator, "run_unittest_plan", return_value={
                "id": "unittest-profile", "status": "passed", "message": "passed"
            }),
            patch.object(validator, "_identities", return_value={"manifest_sha256": "a" * 64}),
            patch.object(validator, "host_import_preflight", return_value={
                "id": "host-import-lib-core",
                "module": "lib.core",
                "status": "skipped",
                "classification": "missing-host-dependency",
                "missing_dependency": "stanza",
                "message": "lib.core import skipped: missing host dependency stanza",
            }),
        ):
            report, exit_code = validator.build_report(("host-smoke",), include_live_assets=False)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "passed_with_skips")
        self.assertEqual(report["schema"], validator.REPORT_SCHEMA)
        self.assertEqual(report["constraints"]["network"]["status"], "unverified")
        self.assertFalse(report["constraints"]["network"]["access_requested"])
        self.assertTrue(all(
            value is False
            for key, value in report["constraints"].items()
            if key not in {"network", "generated_report_path"}
        ))
        self.assertIsNone(report["constraints"]["generated_report_path"])
        self.assertTrue(report["cleanup"]["root_removed"])
        self.assertFalse(report["cleanup"]["partial_files_retained"])

    @staticmethod
    def _validator_subprocess(manifest: Path):
        loader = """
import importlib.util
from pathlib import Path
import sys

validator_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("validator_failure_mode", validator_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.MANIFEST_PATH = manifest_path
raise SystemExit(module.main(["--profile", "short-23"]))
"""
        return subprocess.run(
            [sys.executable, "-c", loader, str(VALIDATOR_PATH), str(manifest)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _request(root: Path, case):
        return {
            "protocol": 1,
            "id": case.case_id,
            "op": "synthesize",
            "model": {
                "family": "chatterbox-multilingual",
                "revision": "validation-contract",
                "t3_model": "v2",
            },
            "device": "cpu",
            "language": case.language,
            "approved_roots": {"voice": [str(root)], "output": [str(root)]},
            "segments": [{"kind": "text", "text": case.text}],
            "output": {
                "path": str(root / f"{case.case_id}.flac.part"),
                "sample_rate": 24000,
                "channels": 1,
            },
        }


if __name__ == "__main__":
    unittest.main()
