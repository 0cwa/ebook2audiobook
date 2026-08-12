from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from components.Chatterbox.runtime.runtime import (
    _default_repo_root,
    build_paths,
    preflight,
    sanitized_worker_environment,
    verify_lock,
)


def _manifest(lock_sha256: str | None = None, *, unresolved: list[str] | None = None) -> dict:
    return {
        "manifest_version": "1.0.0",
        "target": {"os": "linux", "architecture": "x86_64", "python": "3.11.x", "backend": "cpu"},
        "sources": {
            "chatterbox_package": {"artifact_sha256": "0" * 64},
            "chatterbox_source": {"revision": "1" * 40},
            "perth": {"commit": "2" * 40, "artifact_sha256": "3" * 64},
            "model": {"revision": "4" * 40, "files": [{"path": "model.bin", "sha256": "5" * 64}]},
        },
        "unresolved_identities": unresolved or [],
        "lock": {"path": "requirements-cpu.lock", "status": "verified", "sha256": lock_sha256},
    }


class RuntimeTests(unittest.TestCase):
    def test_default_repo_root_matches_direct_runtime_layout(self) -> None:
        runtime_dir = Path(__file__).resolve().parents[1]
        expected = Path(__file__).resolve().parents[4]
        self.assertEqual(_default_repo_root(runtime_dir), expected)

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
            self.assertTrue(first.model_namespace.is_relative_to(first.models_dir))
            self.assertTrue(first.run_namespace.is_relative_to(first.run_dir))
            self.assertIn(first.fingerprint, first.result_path.name)

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
            self.assertTrue(worker_env["PATH"].startswith(str(paths.environment / "bin")))

    def test_preflight_is_read_only_and_checks_first_acquisition_network(self) -> None:
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
            self.assertEqual(len(checked), 2)
            self.assertFalse(paths.environment.exists())


if __name__ == "__main__":
    unittest.main()
