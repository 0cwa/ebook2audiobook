#!/usr/bin/env python3
"""Opt-in, network-passive validation profiles for Chatterbox Multilingual V2.

The default mode runs dependency-light contract tests only.  ``--live-assets``
adds a read-only check of the installed runtime/model/activation receipt chain;
it never provisions, downloads, repairs, or synthesizes audio.

The JSON evidence report is written to stdout.  This script deliberately has
no output-file option so generated reports are not accidentally added to Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, NamedTuple, Sequence


ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ROOT = ROOT / "components" / "Chatterbox"
MANIFEST_PATH = COMPONENT_ROOT / "runtime" / "runtime-manifest.json"
LOCK_PATH = COMPONENT_ROOT / "runtime" / "requirements-cpu.lock"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.Chatterbox.runtime.contract_data import (
    APPROVED_LANGUAGE_IDS,
    CANONICAL_MODEL_FILE_PATHS,
    CONTRACT_DATA,
    MODEL_VARIANT,
)

REPORT_SCHEMA = "ebook2audiobook.chatterbox-validation-report.v2"
SUPPORTED_TARGET = {
    "os": CONTRACT_DATA.target_os,
    "architecture": CONTRACT_DATA.target_arch,
    "backend": CONTRACT_DATA.target_backend,
}
CANONICAL_MODEL_FILES = CANONICAL_MODEL_FILE_PATHS

# These are import names, while requirements.txt contains distribution names.
# Keep the mapping here deliberately small and explicit: only a dependency
# declared by the full application may turn a host import failure into a
# structured skip.  First-party and unknown modules must still fail closed.
_REQUIREMENT_MODULE_ALIASES = {
    "beautifulsoup4": "bs4",
    "coqui-tts": "TTS",
    "hf-xet": "hf_xet",
    "iso639-lang": "iso639",
    "pillow": "PIL",
    "phonemizer-fork": "phonemizer",
    "piper-tts": "piper",
    "py-cpuinfo": "cpuinfo",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "sentence-transformers": "sentence_transformers",
    "unidic-lite": "unidic_lite",
}


def _declared_host_dependency_modules() -> frozenset[str]:
    """Return normalized import roots declared by the full application."""

    try:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return frozenset()

    modules = set()
    for raw_line in requirements.splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line or line.startswith(("-", ".", "/")):
            continue
        for delimiter in ("[", "<", ">", "=", "!", "~", ";", " "):
            line = line.split(delimiter, 1)[0]
        distribution = line.strip().lower()
        if distribution:
            module = _REQUIREMENT_MODULE_ALIASES.get(
                distribution,
                distribution.replace("-", "_"),
            )
            modules.add(module.lower())
    return frozenset(modules)


HOST_DEPENDENCY_MODULES = _declared_host_dependency_modules()


class ValidationCase(NamedTuple):
    case_id: str
    language: str
    text: str
    category: str


SHORT_23_CASES = (
    ValidationCase("short-ar", "ar", "مرحبًا بالعالم.", "short"),
    ValidationCase("short-da", "da", "Hej verden.", "short"),
    ValidationCase("short-de", "de", "Hallo Welt.", "short"),
    ValidationCase("short-el", "el", "Γεια σου κόσμε.", "short"),
    ValidationCase("short-en", "en", "Hello world.", "short"),
    ValidationCase("short-es", "es", "Hola mundo.", "short"),
    ValidationCase("short-fi", "fi", "Hei maailma.", "short"),
    ValidationCase("short-fr", "fr", "Bonjour le monde.", "short"),
    ValidationCase("short-he", "he", "שלום עולם.", "short"),
    ValidationCase("short-hi", "hi", "नमस्ते दुनिया।", "short"),
    ValidationCase("short-it", "it", "Ciao mondo.", "short"),
    ValidationCase("short-ja", "ja", "こんにちは、世界。", "short"),
    ValidationCase("short-ko", "ko", "안녕하세요, 세계.", "short"),
    ValidationCase("short-ms", "ms", "Helo dunia.", "short"),
    ValidationCase("short-nl", "nl", "Hallo wereld.", "short"),
    ValidationCase("short-no", "no", "Hei verden.", "short"),
    ValidationCase("short-pl", "pl", "Witaj świecie.", "short"),
    ValidationCase("short-pt", "pt", "Olá mundo.", "short"),
    ValidationCase("short-ru", "ru", "Привет, мир.", "short"),
    ValidationCase("short-sv", "sv", "Hej världen.", "short"),
    ValidationCase("short-sw", "sw", "Salamu, dunia.", "short"),
    ValidationCase("short-tr", "tr", "Merhaba dünya.", "short"),
    ValidationCase("short-zh", "zh", "你好，世界。", "short"),
)

LONG_FORM_CASES = (
    ValidationCase(
        "long-latin",
        "en",
        "The lantern stayed lit through the storm. After the rain passed, "
        "the travelers compared their notes, checked the map again, and "
        "continued toward the quiet station beyond the river.",
        "latin",
    ),
    ValidationCase(
        "long-rtl",
        "ar",
        "ظل المصباح مضاءً طوال العاصفة. وبعد أن توقف المطر، راجع المسافرون "
        "ملاحظاتهم وتحققوا من الخريطة مرة أخرى، ثم واصلوا الطريق نحو المحطة "
        "الهادئة خلف النهر.",
        "rtl",
    ),
    ValidationCase(
        "long-cjk",
        "zh",
        "暴风雨期间，灯一直亮着。雨停以后，旅人们重新核对笔记和地图，"
        "确认路线没有改变，然后继续前往河对岸那座安静的车站。"
        "夜色渐深时，他们终于看见了站台上的钟。",
        "cjk",
    ),
    ValidationCase(
        "long-devanagari-combining",
        "hi",
        "तूफ़ान के दौरान दीपक जलता रहा। बारिश रुकने पर यात्रियों ने अपने "
        "नोट्स और नक़्शे को फिर से जाँचा, रास्ते की पुष्टि की, और नदी के पार "
        "शांत स्टेशन की ओर आगे बढ़े।",
        "devanagari-combining",
    ),
)

HOST_SMOKE_CASES = (
    ValidationCase(
        "host-english-sml",
        "en",
        "The first sentence is resumable. [break] The second sentence is "
        "combined after conversion. [pause:0.25] FFmpeg verifies the result.",
        "sml-resume-combine-ffmpeg",
    ),
)

PROFILE_CASES = {
    "short-23": SHORT_23_CASES,
    "long-form": LONG_FORM_CASES,
    "host-smoke": HOST_SMOKE_CASES,
}

COMMON_TESTS = (
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_checked_in_manifest_identity_lock_and_storage_contract",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_runtime_manifest_requires_exactly_six_unique_canonical_model_files",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_receipt_identity_links_cross_validate_without_claiming_readiness",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_model_acquisition_publishes_distinct_receipts_and_canonical_snapshot",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_activation_receipt_hash_mismatch_never_reports_product_ready",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_nonregular_ambiguity_markers_fail_closed_for_every_receipt",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_runtime_receipt_combined_rollback_failures_stay_quarantined_after_restart",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_model_receipt_combined_rollback_failures_stay_quarantined_after_restart",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_activation_receipt_combined_rollback_failures_stay_quarantined_after_restart",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_unowned_or_ambiguous_runtime_paths_are_preserved_and_fail_closed",
    "components.Chatterbox.runtime.tests.test_runtime.RuntimeTests.test_unowned_model_object_is_preserved_and_blocks_acquisition",
    "components.Chatterbox.tests.test_worker.WorkerValidationTests.test_manifest_requires_exactly_six_unique_canonical_model_files",
    "components.Chatterbox.tests.test_worker.WorkerValidationTests.test_default_loader_is_network_blocked_and_uses_verified_local_snapshot",
    "components.Chatterbox.tests.test_worker.WorkerValidationTests.test_receipt_ambiguity_markers_require_repair_in_current_and_fresh_workers",
    "components.Chatterbox.tests.test_worker.WorkerValidationTests.test_production_startup_classifies_all_nonregular_ambiguity_markers_as_repair_required",
    "components.Chatterbox.tests.test_worker.WorkerValidationTests.test_missing_manifest_is_reported_as_structured_model_load_failure",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_static_target_contract_is_linux_x86_64_cpu_only",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_real_host_status_rejects_unsupported_os_and_arch_before_worker",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_host_status_consumes_runtime_record_without_paths_or_capacity_work",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_host_status_preserves_missing_provisioning_repair_and_storage_states",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_direct_manager_rejects_all_nonready_states_before_worker",
    "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_client_ignores_arbitrary_worker_override",
    "components.Chatterbox.tests.test_client.ClientTests.test_explicit_worker_environment_does_not_inherit_host_variables",
    "components.Chatterbox.tests.test_client.ClientTests.test_timeout_removes_randomized_worker_partial_output",
    "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_live_asset_check_fails_for_repair_and_other_unsafe_states",
    "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_missing_and_malformed_manifests_emit_json_and_exit_one",
    "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_post_contract_exception_emits_json_and_exit_one",
    "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_unittest_runner_observes_its_spawned_process_group",
    "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_validator_observes_and_removes_owned_temporary_artifacts",
)

PROFILE_TESTS = {
    "short-23": (
        "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_short_23_cases_are_exact_and_independent",
        "components.Chatterbox.tests.test_language_support.ChatterboxLanguageSupportTests.test_mapping_is_exact_and_uses_repository_language_keys",
        "components.Chatterbox.tests.test_language_support.ChatterboxLanguageSupportTests.test_adapter_request_preserves_all_short_ids",
        "components.Chatterbox.tests.test_language_support.ChatterboxLanguageSupportTests.test_fake_worker_readiness_and_forwarding_cover_all_ids",
    ),
    "long-form": (
        "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_long_form_script_categories_are_independent",
        "components.Chatterbox.tests.test_language_support.ChatterboxLanguageSupportTests.test_unicode_text_survives_adapter_request_construction",
        "components.Chatterbox.tests.test_language_support.ChatterboxLanguageSupportTests.test_unicode_text_survives_sml_request_construction_without_normalization",
    ),
    "host-smoke": (
        "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_host_smoke_contract_is_separate_and_non_synthesizing",
        "components.Chatterbox.tests.test_validation_profiles.ValidationProfileTests.test_full_host_gradio_import_preflight",
        "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_request_translation_removes_sml_and_preserves_voice_and_silence",
        "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_success_path_uses_client_and_host_decodes_valid_output",
        "components.Chatterbox.tests.test_adapter.ChatterboxAdapterTests.test_host_audio_validation_falls_back_to_ffprobe",
        "components.Chatterbox.tests.test_client.ClientTests.test_ready_ping_and_success_framing",
    ),
}

CORE_LIFECYCLE_TESTS = (
    "components.Chatterbox.tests.test_core_lifecycle.CoreLifecycleTests.test_successful_conversion_closes_engine",
    "components.Chatterbox.tests.test_core_lifecycle.CoreLifecycleTests.test_failed_conversion_closes_engine_through_unload",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_profiles(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values or "all" in values:
        return tuple(PROFILE_CASES)
    return tuple(dict.fromkeys(values))


def _storage_unknown_buckets(manifest: Mapping[str, Any]) -> list[str]:
    unknown = []
    for phase in manifest.get("storage", {}).get("phases", []):
        for bucket in phase.get("buckets", []):
            if bucket.get("required_bytes") is None and bucket.get("source") is None:
                unknown.append(f"{phase.get('name')}:{bucket.get('name')}")
    return unknown


def static_contract_check() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        model = manifest["sources"]["model"]
        records = model["files"]
        paths = tuple(record["path"] for record in records)
        total = sum(record["size_bytes"] for record in records)
        unknown = _storage_unknown_buckets(manifest)
        errors = []
        target_python = CONTRACT_DATA.target_python
        expected_target = {
            **SUPPORTED_TARGET,
            "python": f"{target_python[0]}.{target_python[1]}.x",
        }
        if manifest.get("target") != expected_target:
            errors.append("manifest target is not Linux x86_64 Python 3.11 CPU")
        if model.get("variant") != f"multilingual-{MODEL_VARIANT}":
            errors.append("manifest model variant is not pinned Chatterbox V2")
        if (
            len(records) != len(CANONICAL_MODEL_FILES)
            or set(paths) != set(CANONICAL_MODEL_FILES)
            or len(set(paths)) != len(CANONICAL_MODEL_FILES)
        ):
            errors.append("manifest does not declare exactly the six canonical V2 files")
        lock = sha256_file(LOCK_PATH)
        declared_lock = manifest.get("lock", {}).get("sha256")
        if lock != declared_lock:
            errors.append("lock SHA-256 does not match the runtime manifest")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"id": "static-contract", "status": "failed", "message": str(exc)}
    return {
        "id": "static-contract",
        "status": "passed" if not errors else "failed",
        "message": "; ".join(errors) if errors else "pinned V2 contract is internally consistent",
        "model_files": list(paths),
        "model_bytes": total,
        "storage_budget": "unknown" if unknown else "determinate",
        "unknown_capacity_buckets": unknown,
    }


def profile_contract_check(profiles: Sequence[str]) -> dict[str, Any]:
    errors = []
    cases = []
    for profile in profiles:
        profile_cases = PROFILE_CASES[profile]
        ids = [case.case_id for case in profile_cases]
        if len(ids) != len(set(ids)):
            errors.append(f"{profile} contains duplicate case IDs")
        if any(not case.text.strip() for case in profile_cases):
            errors.append(f"{profile} contains empty text")
        cases.extend({
            "profile": profile,
            "case_id": case.case_id,
            "language": case.language,
            "category": case.category,
            "text_sha256": hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
            "text_chars": len(case.text),
        } for case in profile_cases)
    short_languages = {case.language for case in SHORT_23_CASES}
    if "short-23" in profiles and len(SHORT_23_CASES) != 23:
        errors.append("short-23 does not contain exactly 23 cases")
    if "short-23" in profiles and short_languages != APPROVED_LANGUAGE_IDS:
        errors.append("short-23 language set does not match the approved worker contract")
    categories = {case.category for case in LONG_FORM_CASES}
    if "long-form" in profiles and categories != {
        "latin", "rtl", "cjk", "devanagari-combining"
    }:
        errors.append("long-form script categories are incomplete")
    return {
        "id": "profile-contract",
        "status": "passed" if not errors else "failed",
        "message": "; ".join(errors) if errors else "selected profiles are complete and independently addressable",
        "cases": cases,
    }


def build_test_plan(
    profiles: Sequence[str],
    *,
    core_import_preflight: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    tests = list(COMMON_TESTS)
    skips: list[dict[str, Any]] = []
    for profile in profiles:
        tests.extend(PROFILE_TESTS[profile])
    if "host-smoke" in profiles:
        preflight = dict(
            core_import_preflight
            if core_import_preflight is not None
            else host_import_preflight("lib.core")
        )
        if preflight["status"] == "passed":
            tests.extend(CORE_LIFECYCLE_TESTS)
        else:
            skips.append({
                "id": "host-lifecycle-preflight",
                "status": preflight["status"],
                "module": "lib.core",
                "test_group": "CoreLifecycleTests",
                "message": (
                    f"{preflight['message']}; CoreLifecycleTests were not collected"
                ),
                "preflight": preflight,
            })
    return tuple(dict.fromkeys(tests)), skips


def _network_constraint() -> dict[str, Any]:
    return {
        "status": "unverified",
        "access_requested": False,
        "mechanism": "library offline flags only; no OS-level network sandbox",
    }


def _offline_environment(temporary_root: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    })
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        environment.pop(key, None)
    if temporary_root is not None:
        environment.update({
            "TMPDIR": str(temporary_root / "tmp"),
            "XDG_CACHE_HOME": str(temporary_root / "cache"),
            "XDG_DATA_HOME": str(temporary_root / "data"),
            "XDG_STATE_HOME": str(temporary_root / "state"),
        })
        for key in ("TMPDIR", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return environment


def host_import_preflight(module_name: str) -> dict[str, Any]:
    """Import a host-stack module in a clean subprocess without providing stubs."""

    command = [
        sys.executable,
        "-c",
        "import importlib, sys; importlib.import_module(sys.argv[1])",
        module_name,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=_offline_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "id": f"host-import-{module_name.replace('.', '-')}",
            "module": module_name,
            "status": "failed",
            "classification": "preflight-error",
            "message": f"{module_name} import preflight could not start: {type(exc).__name__}: {exc}",
            "returncode": None,
        }

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    missing_dependency = None
    marker = "ModuleNotFoundError: No module named "
    for line in stderr.splitlines():
        if marker in line:
            missing_dependency = line.split(marker, 1)[1].strip().strip("'\"")
            break
    if result.returncode == 0:
        return {
            "id": f"host-import-{module_name.replace('.', '-')}",
            "module": module_name,
            "status": "passed",
            "classification": "available",
            "message": f"{module_name} imported successfully in a subprocess",
            "returncode": result.returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
        }
    if missing_dependency:
        missing_root = missing_dependency.partition(".")[0].lower()
        if missing_root in HOST_DEPENDENCY_MODULES:
            return {
                "id": f"host-import-{module_name.replace('.', '-')}",
                "module": module_name,
                "status": "skipped",
                "classification": "missing-host-dependency",
                "missing_dependency": missing_dependency,
                "message": f"{module_name} import skipped: missing host dependency {missing_dependency}",
                "returncode": result.returncode,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-4000:],
            }
        first_party = missing_root in {"components", "lib"}
        return {
            "id": f"host-import-{module_name.replace('.', '-')}",
            "module": module_name,
            "status": "failed",
            "classification": (
                "missing-first-party-module" if first_party else "missing-unknown-module"
            ),
            "missing_dependency": missing_dependency,
            "message": (
                f"{module_name} import preflight found a non-skippable missing "
                f"{'first-party' if first_party else 'unknown'} module {missing_dependency}"
            ),
            "returncode": result.returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
        }
    return {
        "id": f"host-import-{module_name.replace('.', '-')}",
        "module": module_name,
        "status": "failed",
        "classification": "import-error",
        "message": f"{module_name} import preflight failed",
        "returncode": result.returncode,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-4000:],
    }


def _process_group_alive(process_group: int) -> bool | None:
    if os.name != "posix":
        return None
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(process_group: int) -> bool | None:
    alive = _process_group_alive(process_group)
    if alive is not True:
        return alive
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            return False
        for _ in range(20):
            if _process_group_alive(process_group) is False:
                return False
            time.sleep(0.05)
    return _process_group_alive(process_group)


def run_unittest_plan(
    tests: Sequence[str], *, temporary_root: Path | None = None
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for test in tests:
        module = ".".join(test.split(".")[:-2])
        grouped.setdefault(module, []).append(test)
    commands = [
        [sys.executable, "-m", "unittest", "-v", *module_tests]
        for module_tests in grouped.values()
    ]
    started = time.monotonic()
    completed: list[tuple[subprocess.Popen[str], str, str]] = []
    process_observations: list[dict[str, Any]] = []
    for command in commands:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_offline_environment(temporary_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            _stop_process_group(process.pid)
            raise
        residue_before_cleanup = _process_group_alive(process.pid)
        residue_after_cleanup = (
            _stop_process_group(process.pid)
            if residue_before_cleanup is True
            else residue_before_cleanup
        )
        process_observations.append({
            "pid": process.pid,
            "returncode": process.returncode,
            "process_group_observation": "supported" if os.name == "posix" else "unsupported",
            "residue_before_cleanup": residue_before_cleanup,
            "residue_after_cleanup": residue_after_cleanup,
        })
        completed.append((process, stdout, stderr))
    duration = time.monotonic() - started
    stdout = "\n".join(item[1] for item in completed)
    stderr = "\n".join(item[2] for item in completed)
    returncodes = [item[0].returncode for item in completed]
    residue = any(item["residue_before_cleanup"] is True for item in process_observations)
    retained_residue = any(item["residue_after_cleanup"] is True for item in process_observations)
    passed = all(returncode == 0 for returncode in returncodes) and not residue and not retained_residue
    return {
        "id": "unittest-profile",
        "status": "passed" if passed else "failed",
        "message": "curated validation tests passed" if passed else "curated validation tests failed",
        "commands": commands,
        "test_count": len(tests),
        "returncodes": returncodes,
        "duration_seconds": round(duration, 3),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-8000:],
        "process_observations": process_observations,
    }


def live_asset_check(profile_names: Sequence[str]) -> dict[str, Any]:
    """Read the real receipt chain without provisioning, repair, or synthesis."""

    if platform.system().lower() != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        return {
            "id": "live-assets",
            "status": "skipped",
            "message": "live assets require Linux x86_64/amd64 CPU; no worker was launched",
        }
    try:
        from components.Chatterbox.runtime.runtime import build_paths, product_status

        paths = build_paths()
        status = product_status(paths)
    except Exception as exc:  # A malformed local state is evidence, not an install trigger.
        return {
            "id": "live-assets",
            "status": "failed",
            "message": f"read-only receipt inspection failed: {type(exc).__name__}: {exc}",
        }
    dimensions = {
        name: status.get(name, {}).get("status")
        for name in ("runtime", "model", "activation")
    }
    product_state = status.get("status")
    unsafe_dimension = any(
        dimension not in {"ready", "missing"}
        for dimension in dimensions.values()
    )
    missing_states = {"runtime_missing", "model_missing", "activation_missing"}
    if not status.get("ok") and product_state in missing_states and not unsafe_dimension:
        return {
            "id": "live-assets",
            "status": "skipped",
            "message": (
                f"receipt-backed assets are not ready ({product_state}); "
                "validation did not provision, download, repair, or synthesize"
            ),
            "product_status": product_state,
            "dimensions": dimensions,
        }
    if not status.get("ok"):
        return {
            "id": "live-assets",
            "status": "failed",
            "message": f"receipt-backed assets are unsafe ({product_state}); repair is required",
            "product_status": product_state,
            "dimensions": dimensions,
            "errors": {
                name: status.get(name, {}).get("errors", [])
                for name in ("runtime", "model", "activation")
            },
        }
    return {
        "id": "live-assets",
        "status": "passed",
        "message": "runtime, model, and activation receipts are ready; no worker was launched",
        "product_status": "ready",
        "dimensions": {name: status[name]["status"] for name in ("runtime", "model", "activation")},
    }


def _identities() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        identities: dict[str, Any] = {
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "lock_sha256": sha256_file(LOCK_PATH),
            "model_revision": manifest["sources"]["model"]["revision"],
            "product_profile": manifest["product"]["profile"],
        }
    except Exception as exc:
        return {"identity_error": f"{type(exc).__name__}: {exc}"}
    try:
        from components.Chatterbox.runtime.runtime import build_paths

        paths = build_paths()
        identities.update({
            "runtime_fingerprint": paths.runtime_fingerprint,
            "model_fingerprint": paths.model_fingerprint,
            "activation_fingerprint": paths.activation_fingerprint,
        })
    except Exception as exc:
        identities["path_identity_error"] = f"{type(exc).__name__}: {exc}"
    return identities


def _snapshot_owned_root(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _matching_owned_paths(paths: Sequence[str], suffixes: Sequence[str]) -> list[str]:
    return [path for path in paths if any(path.endswith(suffix) for suffix in suffixes)]


def build_report(profiles: Sequence[str], *, include_live_assets: bool) -> tuple[dict[str, Any], int]:
    checks: list[dict[str, Any]] = []
    temporary_root = Path(tempfile.mkdtemp(prefix="e2a-chatterbox-validation-"))
    before_entries = _snapshot_owned_root(temporary_root)
    after_test_entries: list[str] = []
    process_observations: list[dict[str, Any]] = []
    cleanup_error: str | None = None
    try:
        static = static_contract_check()
        checks.extend([static, profile_contract_check(profiles)])
        if static["status"] == "passed":
            core_preflight = None
            if "host-smoke" in profiles:
                core_preflight = host_import_preflight("lib.core")
                if core_preflight["status"] == "passed":
                    checks.append(core_preflight)
            tests, skips = build_test_plan(
                profiles,
                core_import_preflight=core_preflight,
            )
            checks.extend(skips)
            unittest_check = run_unittest_plan(tests, temporary_root=temporary_root)
            process_observations = list(unittest_check.get("process_observations", []))
            checks.append(unittest_check)
            if include_live_assets:
                checks.append(live_asset_check(profiles))
    except Exception as exc:
        checks.append({
            "id": "validation-exception",
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
        })
    finally:
        try:
            after_test_entries = _snapshot_owned_root(temporary_root)
            shutil.rmtree(temporary_root)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"

    root_removed = not temporary_root.exists()
    partial_paths = _matching_owned_paths(after_test_entries, (".part", ".partial", ".tmp"))
    audio_paths = _matching_owned_paths(after_test_entries, (".wav", ".flac", ".mp3", ".m4a"))
    json_paths = _matching_owned_paths(after_test_entries, (".json", ".jsonl"))
    process_residue = [
        observation for observation in process_observations
        if observation.get("residue_after_cleanup") is True
    ]
    cleanup_passed = root_removed and not process_residue and cleanup_error is None
    cleanup = {
        "validator_temp_root": str(temporary_root),
        "before_entries": before_entries,
        "after_test_entries": after_test_entries,
        "root_removed": root_removed,
        "cleanup_error": cleanup_error,
        "partial_paths_observed_after_tests": partial_paths,
        "audio_paths_observed_after_tests": audio_paths,
        "json_paths_observed_after_tests": json_paths,
        "partial_files_retained": bool(partial_paths) and not root_removed,
        "audio_files_retained": bool(audio_paths) and not root_removed,
        "json_files_retained": bool(json_paths) and not root_removed,
        "spawned_process_groups": process_observations,
        "spawned_process_residue_after_cleanup": process_residue,
    }
    checks.append({
        "id": "cleanup-observation",
        "status": "passed" if cleanup_passed else "failed",
        "message": (
            "validator-owned temporary root and spawned process groups were cleaned"
            if cleanup_passed
            else "validator-owned cleanup or spawned-process cleanup was incomplete"
        ),
    })

    failed = [check for check in checks if check["status"] == "failed"]
    live_skipped = include_live_assets and any(
        check["id"] == "live-assets" and check["status"] == "skipped"
        for check in checks
    )
    if failed:
        status, exit_code = "failed", 1
    elif live_skipped:
        status, exit_code = "skipped", 2
    elif any(check["status"] == "skipped" for check in checks):
        status, exit_code = "passed_with_skips", 0
    else:
        status, exit_code = "passed", 0

    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "profiles": list(profiles),
        "mode": "static+live-assets" if include_live_assets else "static",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "identities": _identities(),
        "constraints": {
            "network": _network_constraint(),
            "model_download": False,
            "provisioning": False,
            "repair": False,
            "synthesis": False,
            "capacity_guess": False,
            "generated_report_path": None,
        },
        "checks": checks,
        "cleanup": cleanup,
    }
    return report, exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=("all", *PROFILE_CASES),
        help="profile to validate; repeat to combine profiles (default: all)",
    )
    parser.add_argument(
        "--live-assets",
        action="store_true",
        help="also inspect already-ready local receipts/model assets; never provisions or synthesizes",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="emit only the profile/evidence contract without running tests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profiles = selected_profiles(args.profile)
    if args.describe:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "described",
            "profiles": list(profiles),
            "constraints": {
                "network": _network_constraint(),
                "model_download": False,
                "provisioning": False,
                "repair": False,
                "synthesis": False,
                "capacity_guess": False,
            },
            "profile_contract": profile_contract_check(profiles),
        }
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    try:
        report, exit_code = build_report(profiles, include_live_assets=args.live_assets)
    except Exception as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "failed",
            "profiles": list(profiles),
            "mode": "static+live-assets" if args.live_assets else "static",
            "constraints": {"network": _network_constraint()},
            "checks": [{
                "id": "validation-exception",
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
            }],
        }
        exit_code = 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
