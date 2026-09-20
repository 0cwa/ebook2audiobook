#!/usr/bin/env bash

# User-local rootless CPU launcher for immutable Linux. It owns only the
# declared external data root and the caller's rootless Podman storage. It
# never invokes sudo, a package manager, user/group mutation, or driver setup.

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SCRIPT_DIR="$(cd "$TOOL_DIR/.." >/dev/null 2>&1 && pwd -P)"
CANONICAL_COMPOSE_FILE="$SCRIPT_DIR/podman-compose.yml"
COMPOSE_FILE="$CANONICAL_COMPOSE_FILE"
PROJECT_NAME=""
APP_VERSION="$(<"$SCRIPT_DIR/VERSION.txt")"

if [[ -n "${E2A_DATA_ROOT:-}" ]]; then
	DATA_ROOT="$E2A_DATA_ROOT"
elif [[ -n "${XDG_DATA_HOME:-}" ]]; then
	DATA_ROOT="$XDG_DATA_HOME/ebook2audiobook"
elif [[ -n "${HOME:-}" ]]; then
	DATA_ROOT="$HOME/.local/share/ebook2audiobook"
else
	DATA_ROOT=""
fi

ACTION="run"
NO_BUILD=0
PREVIEW=0
APPLY=0
APP_ARGS=()

usage() {
	cat <<'EOF'
Usage:
  tools/immutable-launch.sh setup [options]
  tools/immutable-launch.sh run [options] [-- app-arguments]
  tools/immutable-launch.sh update [options]
  tools/immutable-launch.sh diagnose [--json] [--strict]
  tools/immutable-launch.sh uninstall [--preview|--apply]

Options:
  --data-root PATH  Absolute user-owned root for ebooks, outputs, models,
                    voices, run state, and temporary files.
  --no-build        For setup, validate the CPU compose path without building.
  --preview         Show uninstall actions without changing containers or data.
  --apply           For uninstall, remove this compose project's containers and
                    network. The external data root is retained.

The default root is $XDG_DATA_HOME/ebook2audiobook or
$HOME/.local/share/ebook2audiobook. It must be outside the checkout.
EOF
}

die() {
	echo "ERROR: $*" >&2
	exit 1
}

while (($# > 0)); do
	case "$1" in
		setup|run|update|diagnose|uninstall)
			ACTION="$1"
			shift
			;;
		--data-root)
			(($# >= 2)) || die "--data-root requires an absolute path"
			DATA_ROOT="$2"
			shift 2
			;;
		--no-build) NO_BUILD=1; shift ;;
		--preview) PREVIEW=1; shift ;;
		--apply) APPLY=1; shift ;;
		--json|--strict)
			[[ "$ACTION" == diagnose ]] || die "$1 is only valid with diagnose"
			APP_ARGS+=("$1")
			shift
			;;
		--help|-h) usage; exit 0 ;;
		--)
			shift
			APP_ARGS=("$@")
			break
			;;
		*)
			if [[ "$ACTION" == run ]]; then
				APP_ARGS+=("$1")
				shift
			else
				die "unknown option or action argument: $1"
			fi
			;;
	esac
done

if [[ "$ACTION" == diagnose ]]; then
	exec bash "$TOOL_DIR/immutable-diagnostics.sh" "${APP_ARGS[@]}"
fi

DATA_DIRS=(ebooks audiobooks models voices run tmp)
CURRENT_UID="$(id -u)"

if [[ -n "${E2A_COMPOSE_FILE:-}" && "$E2A_COMPOSE_FILE" != "$CANONICAL_COMPOSE_FILE" ]]; then
	die "E2A_COMPOSE_FILE is not supported; use the canonical compose file: $CANONICAL_COMPOSE_FILE"
fi
[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE"

canonicalize_path() {
	command -v realpath >/dev/null 2>&1 || die "realpath is required to validate the user-owned data root"
	local canonical
	canonical="$(realpath -m -- "$1" 2>/dev/null)" || die "could not canonicalize path: $1"
	[[ "$canonical" == /* ]] || die "canonical path is not absolute: $1"
	printf '%s' "$canonical"
}

assert_outside_checkout() {
	local candidate="$1"
	case "$candidate" in
		"$SCRIPT_DIR"|"$SCRIPT_DIR"/*)
			die "data path must resolve outside the source checkout: $candidate"
			;;
	esac
}

nearest_existing_parent() {
	local candidate="$1"
	local parent="$candidate"
	while [[ ! -e "$parent" ]]; do
		local next
		next="$(dirname -- "$parent")"
		[[ "$next" != "$parent" ]] || die "could not find an existing parent for: $candidate"
		parent="$next"
	done
	printf '%s' "$parent"
}

validate_owned_directory() {
	local path="$1"
	local label="$2"
	local owner
	[[ -d "$path" ]] || die "$label must resolve to a directory: $path"
	owner="$(stat -c '%u' -- "$path" 2>/dev/null)" || die "could not inspect ownership of $label: $path"
	[[ "$owner" == "$CURRENT_UID" ]] || die "$label must be owned by the current user (uid $CURRENT_UID): $path"
	[[ -r "$path" && -w "$path" && -x "$path" ]] || die "$label is not readable and writable: $path"
}

validate_path_target() {
	local requested="$1"
	local label="$2"
	local canonical
	canonical="$(canonicalize_path "$requested")"
	assert_outside_checkout "$canonical"
	if [[ -L "$requested" && ! -e "$requested" ]]; then
		die "$label is a dangling symlink: $requested"
	fi
	if [[ -e "$requested" ]]; then
		validate_owned_directory "$canonical" "$label"
	else
		validate_owned_directory "$(nearest_existing_parent "$canonical")" "parent of $label"
	fi
}

validate_data_root() {
	[[ -n "$DATA_ROOT" ]] || die "set E2A_DATA_ROOT or HOME before using the container path"
	[[ "$DATA_ROOT" == /* ]] || die "--data-root must be an absolute path"

	local canonical_root
	canonical_root="$(canonicalize_path "$DATA_ROOT")"
	assert_outside_checkout "$canonical_root"
	if [[ -L "$DATA_ROOT" && ! -e "$DATA_ROOT" ]]; then
		die "data root is a dangling symlink: $DATA_ROOT"
	fi
	if [[ -e "$DATA_ROOT" ]]; then
		validate_owned_directory "$canonical_root" "data root"
	else
		validate_owned_directory "$(nearest_existing_parent "$canonical_root")" "parent of data root"
	fi
	DATA_ROOT="$canonical_root"

	local directory target
	for directory in "${DATA_DIRS[@]}"; do
		target="$DATA_ROOT/$directory"
		validate_path_target "$target" "data subdirectory $directory"
	done
}

validate_data_root

project_name_for_data_root() {
	command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to derive the compose project identity"
	local digest
	digest="$(printf '%s' "$DATA_ROOT" | sha256sum | cut -c1-16)" || die "could not derive compose project identity"
	[[ "$digest" =~ ^[[:xdigit:]]{16}$ ]] || die "invalid compose project identity"
	printf 'ebook2audiobook-immutable-%s' "$digest"
}

PROJECT_NAME="$(project_name_for_data_root)"
if [[ -n "${E2A_COMPOSE_PROJECT_NAME:-}" && "$E2A_COMPOSE_PROJECT_NAME" != "$PROJECT_NAME" ]]; then
	die "E2A_COMPOSE_PROJECT_NAME must match the canonical data-root identity: $PROJECT_NAME"
fi

cpu_device_json() {
	local arch
	case "$(uname -m)" in
		x86_64|amd64) arch=x86_64 ;;
		aarch64|arm64) arch=aarch64 ;;
		*) die "unsupported CPU architecture for the immutable image: $(uname -m)" ;;
	esac
	printf '{"name":"cpu","os":"manylinux_2_28","arch":"%s","pyvenv":[3,12],"tag":"cpu","note":"Rootless immutable CPU build"}' "$arch"
}

prepare_data_root() {
	validate_data_root
	mkdir -p "$DATA_ROOT"
	for directory in "${DATA_DIRS[@]}"; do
		mkdir -p "$DATA_ROOT/$directory"
	done
	validate_data_root
}

rootless_podman() {
	command -v podman >/dev/null 2>&1 || die "podman is not available; install it through the host's approved user-space method"
	local report
	if ! report="$(podman info --format '{{.Host.Security.Rootless}}' 2>&1)"; then
		die "podman could not be queried without changing host policy: $report"
	fi
	report="$(printf '%s' "$report" | tr -d '[:space:]')"
	[[ "$report" == true ]] || die "podman is not rootless (reported $report); this path will not use rootful mode or change host policy"
}

compose_provider() {
	if podman compose version >/dev/null 2>&1; then
		COMPOSE=(podman compose)
	elif command -v podman-compose >/dev/null 2>&1; then
		COMPOSE=(podman-compose)
	else
		die "no Podman Compose provider found; expose 'podman compose' or 'podman-compose' without changing the host"
	fi
}

compose_env() {
	local device_json
	device_json="$(cpu_device_json)"
	E2A_DATA_ROOT="$DATA_ROOT" \
	DEVICE_TAG=cpu \
	COMPOSE_PROFILES=cpu \
	APP_VERSION="$APP_VERSION" \
	PYTHON_VERSION=3.12 \
	DOCKER_DEVICE_STR="$device_json" \
	DOCKER_PROGRAMS_STR="curl ffmpeg mediainfo nodejs npm espeak-ng sox tesseract-ocr" \
	CALIBRE_INSTALLER_URL="https://download.calibre-ebook.com/linux-installer.sh" \
	ISO3_LANG=eng \
	INSTALL_RUST=1 \
	COMPOSE_PROJECT_NAME="$PROJECT_NAME" \
	"${COMPOSE[@]}" -f "$COMPOSE_FILE" --profile cpu "$@"
}

validate_compose() {
	compose_env config >/dev/null
}

run_setup() {
	prepare_data_root
	rootless_podman
	compose_provider
	validate_compose
	if ((NO_BUILD)); then
		echo "CPU compose configuration is valid; image build was skipped."
		return 0
	fi
	echo "Building the CPU image in rootless Podman storage; external data stays at $DATA_ROOT"
	compose_env build
}

run_app() {
	prepare_data_root
	rootless_podman
	compose_provider
	validate_compose
	if ((${#APP_ARGS[@]} > 0)); then
		compose_env run --rm --service-ports ebook2audiobook-cpu "${APP_ARGS[@]}"
	else
		compose_env up
	fi
}

run_update() {
	prepare_data_root
	rootless_podman
	compose_provider
	validate_compose
	echo "Rebuilding the CPU image; external data remains at $DATA_ROOT"
	compose_env build --pull
	compose_env up --force-recreate
}

run_uninstall() {
	if ((PREVIEW || !APPLY)); then
		cat <<EOF
Uninstall preview for compose project $PROJECT_NAME

Would remove this project's containers and network:
  $COMPOSE_FILE (profile: cpu)

Would retain user data by default:
  $DATA_ROOT/{ebooks,audiobooks,models,voices,run,tmp}

No host packages, groups, security settings, or source-checkout files are changed.
Run 'tools/immutable-launch.sh uninstall --apply --data-root "$DATA_ROOT"' to
stop and remove only this compose project's runtime objects.
Delete the data root separately, only after checking the exact path and taking a
backup; data deletion is intentionally not part of this command.
EOF
		return 0
	fi
	rootless_podman
	compose_provider
	compose_env down
	echo "Removed the rootless CPU compose runtime. Preserved data: $DATA_ROOT"
}

case "$ACTION" in
	setup) run_setup ;;
	run) run_app ;;
	update) run_update ;;
	uninstall) run_uninstall ;;
	*) die "unsupported action: $ACTION" ;;
esac
