from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "ebook2audiobook.command"
DIAGNOSTICS = REPO / "tools" / "immutable-diagnostics.sh"
CONTAINER_LAUNCHER = REPO / "tools" / "immutable-launch.sh"
PODMAN_COMPOSE = REPO / "podman-compose.yml"
DOCKER_COMPOSE = REPO / "docker-compose.yml"
MANIFEST = REPO / "immutable-installer-manifest.json"
IMMUTABLE_DOCS = REPO / "docs" / "immutable-linux.md"


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env["TMPDIR"] = str(REPO / "tmp")
    if env:
        child_env.update(env)
    return subprocess.run(
        command,
        cwd=REPO,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tripwire(directory: Path, name: str, log: Path) -> None:
    path = directory / name
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' {name} >> {str(log)!r}\n"
        "exit 97\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_fake_rootless_podman(directory: Path) -> None:
    path = directory / "podman"
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = info ]; then printf 'true\\n'; exit 0; fi\n"
        "if [ \"${1:-}\" = compose ] && [ \"${2:-}\" = version ]; then exit 0; fi\n"
        "if [ \"${1:-}\" = compose ]; then printf 'services: {}\\n'; exit 0; fi\n"
        "exit 97\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_fake_unowned_stat(directory: Path) -> None:
    real_stat = shutil.which("stat", path="/usr/bin:/bin")
    assert real_stat
    path = directory / "stat"
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = -c ] && [ \"${2:-}\" = '%u' ]; then printf '4294967294\\n'; exit 0; fi\n"
        f"exec {real_stat} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_read_only_diagnostic_json_has_stable_categories_and_no_secret_leak() -> None:
    tracked_surface = [LAUNCHER, DIAGNOSTICS, CONTAINER_LAUNCHER, PODMAN_COMPOSE]
    before = {path: digest(path) for path in tracked_surface}
    installed = REPO / ".installed"
    installed_before = installed.exists()

    with tempfile.TemporaryDirectory(prefix="immutable-diagnostics-", dir=REPO / "tmp") as temp:
        temp_root = Path(temp)
        tripwire_bin = temp_root / "bin"
        tripwire_bin.mkdir()
        tripwire_log = temp_root / "tripwire.log"
        for command in (
            "sudo",
            "usermod",
            "sg",
            "apt-get",
            "dnf",
            "rpm-ostree",
            "brew",
            "curl",
            "wget",
            "podman",
            "podman-compose",
        ):
            write_tripwire(tripwire_bin, command, tripwire_log)

        data_root = temp_root / "api-key=topsecret"
        result = run(
            [str(LAUNCHER), "--diagnose", "--json"],
            {
                "PATH": f"{tripwire_bin}:{os.environ.get('PATH', '')}",
                "HOME": str(temp_root / "home"),
                "XDG_DATA_HOME": str(temp_root / "xdg-data"),
                "E2A_DATA_ROOT": str(data_root),
                "E2A_ALLOW_SYSTEM_INSTALL": "0",
            },
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["schema"] == "e2a.immutable-diagnostics/v1"
        assert report["mode"] == "read-only"
        check_ids = {check["id"] for check in report["checks"]}
        assert {"source", "data-root", "container", "network"} <= check_ids
        assert "topsecret" not in result.stdout
        assert not tripwire_log.exists()
        assert not data_root.exists()

    assert {path: digest(path) for path in tracked_surface} == before
    assert installed.exists() == installed_before


def test_uninstall_preview_is_non_destructive_and_keeps_data() -> None:
    with tempfile.TemporaryDirectory(prefix="immutable-uninstall-", dir=REPO.parent) as temp:
        data_root = Path(temp)
        marker = data_root / "audiobooks" / "keep.txt"
        marker.parent.mkdir()
        marker.write_text("keep", encoding="utf-8")

        result = run(
            [
                str(CONTAINER_LAUNCHER),
                "uninstall",
                "--preview",
                "--data-root",
                str(data_root),
            ]
        )

        assert result.returncode == 0, result.stderr
        assert "Would retain user data by default" in result.stdout
        assert marker.read_text(encoding="utf-8") == "keep"


def test_launcher_setup_handles_cold_existing_spaces_and_symlinked_home_root() -> None:
    with tempfile.TemporaryDirectory(prefix="immutable-data-root-", dir=REPO.parent) as temp:
        temp_root = Path(temp)
        fake_bin = temp_root / "bin"
        fake_bin.mkdir()
        write_fake_rootless_podman(fake_bin)
        actual_home = temp_root / "actual home"
        actual_home.mkdir()
        symlinked_home = temp_root / "home-link"
        symlinked_home.symlink_to(actual_home, target_is_directory=True)
        data_root = symlinked_home / "data root with spaces"

        env = {
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "E2A_COMPOSE_FILE": "",
            "E2A_COMPOSE_PROJECT_NAME": "",
        }
        command = [str(CONTAINER_LAUNCHER), "setup", "--no-build", "--data-root", str(data_root)]
        first = run(command, env)
        assert first.returncode == 0, first.stderr

        canonical_root = actual_home / "data root with spaces"
        for directory in ("ebooks", "audiobooks", "models", "voices", "run", "tmp"):
            assert (canonical_root / directory).is_dir()
        marker = canonical_root / "audiobooks" / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        second = run(command, env)
        assert second.returncode == 0, second.stderr
        assert marker.read_text(encoding="utf-8") == "keep"


def test_launcher_rejects_checkout_symlink_and_unowned_data_root() -> None:
    with tempfile.TemporaryDirectory(prefix="immutable-data-safety-", dir=REPO.parent) as temp:
        temp_root = Path(temp)
        checkout_link = temp_root / "link-to-checkout"
        checkout_link.symlink_to(REPO, target_is_directory=True)
        symlink_result = run(
            [str(CONTAINER_LAUNCHER), "uninstall", "--preview", "--data-root", str(checkout_link / "data")],
            {"E2A_COMPOSE_FILE": "", "E2A_COMPOSE_PROJECT_NAME": ""},
        )
        assert symlink_result.returncode != 0
        assert "outside the source checkout" in symlink_result.stderr

        unowned_root = temp_root / "unowned"
        unowned_root.mkdir()
        fake_bin = temp_root / "fake-stat-bin"
        fake_bin.mkdir()
        write_fake_unowned_stat(fake_bin)
        unowned_result = run(
            [str(CONTAINER_LAUNCHER), "uninstall", "--preview", "--data-root", str(unowned_root)],
            {
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "E2A_COMPOSE_FILE": "",
                "E2A_COMPOSE_PROJECT_NAME": "",
            },
        )
        assert unowned_result.returncode != 0
        assert "owned by the current user" in unowned_result.stderr


def test_launcher_rejects_untrusted_compose_overrides_and_orphan_cleanup() -> None:
    with tempfile.TemporaryDirectory(prefix="immutable-compose-safety-", dir=REPO.parent) as temp:
        data_root = Path(temp) / "data root with spaces"
        base_env = {"E2A_COMPOSE_FILE": "", "E2A_COMPOSE_PROJECT_NAME": ""}
        valid = run(
            [str(CONTAINER_LAUNCHER), "uninstall", "--preview", "--data-root", str(data_root)],
            base_env,
        )
        assert valid.returncode == 0, valid.stderr
        assert "ebook2audiobook-immutable-" in valid.stdout

        project_override = run(
            [str(CONTAINER_LAUNCHER), "uninstall", "--preview", "--data-root", str(data_root)],
            {**base_env, "E2A_COMPOSE_PROJECT_NAME": "unrelated-project"},
        )
        assert project_override.returncode != 0
        assert "canonical data-root identity" in project_override.stderr

        file_override = run(
            [str(CONTAINER_LAUNCHER), "uninstall", "--preview", "--data-root", str(data_root)],
            {**base_env, "E2A_COMPOSE_FILE": "/etc/hosts"},
        )
        assert file_override.returncode != 0
        assert "canonical compose file" in file_override.stderr

    assert "--remove-orphans" not in CONTAINER_LAUNCHER.read_text(encoding="utf-8")


def test_native_missing_tools_stops_without_privilege_fallthrough() -> None:
    with tempfile.TemporaryDirectory(prefix="immutable-no-sudo-", dir=REPO / "tmp") as temp:
        temp_root = Path(temp)
        minimal_bin = temp_root / "bin"
        minimal_bin.mkdir()
        tripwire_log = temp_root / "tripwire.log"
        for command in ("sudo", "usermod", "sg", "apt-get", "dnf", "rpm-ostree", "brew"):
            write_tripwire(minimal_bin, command, tripwire_log)

        # Keep only the shell utilities needed to reach the dependency check;
        # native application tools are deliberately absent from this PATH.
        for command in ("bash", "python3", "uname", "cut", "tr", "grep", "stat", "chmod", "dirname", "pwd", "ldconfig"):
            source = shutil.which(command, path="/usr/bin:/bin")
            if source:
                (minimal_bin / command).symlink_to(source)

        installed = REPO / ".installed"
        installed_before = installed.read_bytes() if installed.exists() else None
        run_dir = REPO / "run"
        mode_before = stat.S_IMODE(run_dir.stat().st_mode)
        try:
            result = run(
                [str(LAUNCHER), "--headless", "--ebook", str(REPO / "ebooks" / "missing.txt")],
                {
                    "PATH": str(minimal_bin),
                    "HOME": str(temp_root / "home"),
                    "USER": "",
                    "E2A_ALLOW_SYSTEM_INSTALL": "0",
                },
            )
        finally:
            run_dir.chmod(mode_before)
            if installed_before is None:
                installed.unlink(missing_ok=True)
            else:
                installed.write_bytes(installed_before)

        assert result.returncode != 0
        assert "No-sudo next step" in result.stdout
        assert not tripwire_log.exists()


def test_compose_requires_external_data_and_does_not_disable_selinux() -> None:
    for compose_file in (PODMAN_COMPOSE, DOCKER_COMPOSE):
        text = compose_file.read_text(encoding="utf-8")
        assert text.count("E2A_DATA_ROOT:?") == 6
        assert "/app/ebooks" in text
        assert "/app/audiobooks" in text
        assert "/app/models" in text
        assert "/app/voices" in text
        assert "/app/run" in text
        assert "/app/tmp" in text
    podman_text = PODMAN_COMPOSE.read_text(encoding="utf-8")
    assert "security_opt:" not in podman_text
    assert "\n    - label=disable" not in podman_text


def test_manifest_matches_rootless_cpu_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["scope"] == "rootless-container-cpu"
    assert manifest["supported_targets"][0]["runtime"].startswith("rootless Podman")
    assert manifest["persistent_state"]["upgrade"].startswith("retain")
    assert "no sudo" in manifest["build"]["host_mutations"]


def test_docs_distinguish_image_trust_rejection_from_missing_podman() -> None:
    text = IMMUTABLE_DOCS.read_text(encoding="utf-8")
    assert "Source image rejected" in text
    assert "python:3.12-slim-trixie" in text
    assert "not a missing-Podman failure" in text
