#!/usr/bin/env bash

# Read-only, dependency-light preflight for immutable Linux users. This file is
# intentionally independent of the application environment: it must be useful
# before Python, Conda, or the container image exists.

set -u -o pipefail

SCHEMA="e2a.immutable-diagnostics/v1"
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SCRIPT_DIR="$(cd "$TOOL_DIR/.." >/dev/null 2>&1 && pwd -P)"

if [[ -n "${E2A_DATA_ROOT:-}" ]]; then
	DATA_ROOT="$E2A_DATA_ROOT"
elif [[ -n "${XDG_DATA_HOME:-}" ]]; then
	DATA_ROOT="$XDG_DATA_HOME/ebook2audiobook"
elif [[ -n "${HOME:-}" ]]; then
	DATA_ROOT="$HOME/.local/share/ebook2audiobook"
else
	DATA_ROOT=""
fi

OUTPUT_JSON=0
STRICT=0

usage() {
	cat <<'EOF'
Usage: tools/immutable-diagnostics.sh [--json] [--strict]

Run a read-only immutable-Linux preflight. The command does not create paths,
contact the network, install packages, change groups, or invoke a container.

  --json    Emit the stable machine-readable report.
  --strict  Exit 1 when a check needs an action; the report is still emitted.
EOF
}

while (($# > 0)); do
	case "$1" in
		--json) OUTPUT_JSON=1; shift ;;
		--strict) STRICT=1; shift ;;
		--help|-h) usage; exit 0 ;;
		*) echo "ERROR: unknown diagnostic option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

CHECK_ID=()
CHECK_CATEGORY=()
CHECK_STATUS=()
CHECK_DETAIL=()
CHECK_ACTION=()

add_check() {
	CHECK_ID+=("$1")
	CHECK_CATEGORY+=("$2")
	CHECK_STATUS+=("$3")
	CHECK_DETAIL+=("$4")
	CHECK_ACTION+=("$5")
}

json_escape() {
	local value="$1"
	value=${value//\\/\\\\}
	value=${value//\"/\\\"}
	value=${value//$'\n'/\\n}
	value=${value//$'\r'/\\r}
	value=${value//$'\t'/\\t}
	printf '%s' "$value"
}

display_path() {
	local value="$1"
	if [[ -n "${HOME:-}" && "$value" == "$HOME"/* ]]; then
		value="\${HOME}${value#"$HOME"}"
	fi
	# Do not echo common secret-shaped path components or values supplied in
	# diagnostic environment variables. The report never dumps the environment.
	value="$(printf '%s' "$value" | sed -E 's/(token|password|secret|api[_-]?key)(=|-)[^/[:space:]]+/\1\2<redacted>/Ig')"
	printf '%s' "$value"
}

command_present() {
	command -v "$1" >/dev/null 2>&1
}

version_supported() {
	local value="$1"
	local major minor
	if [[ ! "$value" =~ ^([0-9]+)\.([0-9]+) ]]; then
		return 1
	fi
	major="${BASH_REMATCH[1]}"
	minor="${BASH_REMATCH[2]}"
	(( major == 3 && minor >= 10 && minor < 13 ))
}

host_name="$(uname -s 2>/dev/null || printf 'unknown')"
host_arch="$(uname -m 2>/dev/null || printf 'unknown')"
if [[ "$host_name" == Linux ]]; then
	add_check "host" "unsupported-host" "ok" "Linux host detected" ""
else
	add_check "host" "unsupported-host" "action-required" "This immutable guide targets Linux; detected ${host_name}" "Use the platform's documented native or container path"
fi

case "$host_arch" in
	x86_64|amd64|aarch64|arm64)
		add_check "architecture" "unsupported-host" "ok" "CPU image has an amd64/arm64 path ($host_arch)" "" ;;
	*)
		add_check "architecture" "unsupported-host" "action-required" "No CPU image contract is recorded for $host_arch" "Use a supported x86_64 or aarch64 host" ;;
esac

if [[ -d "$SCRIPT_DIR" && -w "$SCRIPT_DIR" ]]; then
	add_check "source" "unwritable-checkout" "ok" "Checkout is readable and appears writable: $(display_path "$SCRIPT_DIR")" ""
else
	add_check "source" "unwritable-checkout" "action-required" "Checkout is missing or not writable: $(display_path "$SCRIPT_DIR")" "Use the rootless container path; it does not need checkout writes"
fi

if [[ -n "$DATA_ROOT" && "$DATA_ROOT" == /* ]]; then
	if [[ -d "$DATA_ROOT" && -w "$DATA_ROOT" ]]; then
		add_check "data-root" "storage" "ok" "External data root is writable: $(display_path "$DATA_ROOT")" ""
	elif [[ ! -e "$DATA_ROOT" ]]; then
		add_check "data-root" "storage" "action-required" "External data root does not exist yet: $(display_path "$DATA_ROOT")" "Run tools/immutable-launch.sh setup --data-root <absolute-path>"
	else
		add_check "data-root" "storage" "action-required" "External data root is not writable: $(display_path "$DATA_ROOT")" "Choose a user-owned writable data root"
	fi
else
	add_check "data-root" "storage" "action-required" "No absolute user-owned data root is configured" "Set E2A_DATA_ROOT to an absolute path outside the checkout"
fi

if command_present python3; then
	python_version="$(python3 --version 2>&1 | awk '{print $2}')"
	if version_supported "$python_version"; then
		add_check "python" "missing-dependency" "ok" "Python $python_version is in the native support range" ""
	else
		add_check "python" "missing-dependency" "action-required" "Python ${python_version:-unknown} is not in the supported 3.10-3.12 range" "Use the CPU container path or provide Python 3.10-3.12 for native mode"
	fi
else
	add_check "python" "missing-dependency" "action-required" "python3 is not available" "Use the CPU container path or install Python through your host's approved user-space method"
fi

runtime_dir="$SCRIPT_DIR/python_env"
if [[ -x "$runtime_dir/bin/python" && -f "$runtime_dir/.provisioned" ]]; then
	add_check "native-runtime" "runtime-model-activation-readiness" "ok" "Native runtime is present and marked provisioned" ""
elif [[ -e "$runtime_dir" ]]; then
	add_check "native-runtime" "repair-required" "action-required" "Native runtime exists but is incomplete or unmarked" "Use the container path or repair native setup explicitly; no repair was attempted"
else
	add_check "native-runtime" "runtime-model-activation-readiness" "action-required" "Native runtime is not installed" "Use the rootless CPU setup or an approved native user-space setup"
fi

models_dir="${E2A_MODELS_DIR:-${DATA_ROOT:+$DATA_ROOT/models}}"
if [[ -n "$models_dir" && -d "$models_dir" ]]; then
	add_check "models" "runtime-model-activation-readiness" "ok" "Model root exists" ""
else
	add_check "models" "runtime-model-activation-readiness" "action-required" "Model root is not ready; first model acquisition may need network access" "Use a writable data root and run the application setup"
fi

missing_native=()
for required in ffmpeg mediainfo node espeak-ng sox tesseract; do
	if ! command_present "$required"; then
		missing_native+=("$required")
	fi
done
if ((${#missing_native[@]} == 0)); then
	add_check "native-tools" "missing-dependency" "ok" "Core native tools are present" ""
else
	add_check "native-tools" "missing-dependency" "action-required" "Native tools missing: ${missing_native[*]}" "Use the rootless CPU container; this diagnostic will not invoke a package manager"
fi

if command_present podman; then
	if command_present podman-compose; then
		add_check "container" "container-availability" "ok" "podman and podman-compose commands are present" "The launch helper will still verify rootless mode before running"
	else
		add_check "container" "container-availability" "warning" "podman is present; a compose provider will be checked at launch" "Install or expose podman compose through the host's approved user-space method"
	fi
else
	add_check "container" "container-availability" "action-required" "podman is not available" "Use an immutable-host user-space installation of rootless Podman; no host mutation is attempted"
fi

if command_present xdg-open || command_present gio || command_present x-www-browser || command_present open; then
	add_check "browser" "ui-bind-failure" "ok" "A host browser opener is available" "The GUI is served over HTTP; no display or audio device is required inside the container"
else
	add_check "browser" "ui-bind-failure" "warning" "No browser opener was detected" "Open http://127.0.0.1:7860/ manually after launch, or use headless mode"
fi

add_check "network" "network" "not-probed" "Network access was not tested by this read-only command" "Network is needed for image/model acquisition when artifacts are not already present"

overall="ok"
for status in "${CHECK_STATUS[@]}"; do
	case "$status" in
		action-required|unsupported) overall="action-required" ;;
		warning)
			[[ "$overall" == ok ]] && overall="warning"
			;;
	esac
done

if ((OUTPUT_JSON)); then
	printf '{\n'
	printf '  "schema": "%s",\n' "$SCHEMA"
	printf '  "mode": "read-only",\n'
	printf '  "overall": "%s",\n' "$overall"
	printf '  "host": {"os": "%s", "architecture": "%s"},\n' "$(json_escape "$host_name")" "$(json_escape "$host_arch")"
	printf '  "checks": [\n'
	for ((i = 0; i < ${#CHECK_ID[@]}; i++)); do
		comma=,
		(( i == ${#CHECK_ID[@]} - 1 )) && comma=""
		printf '    {"id":"%s","category":"%s","status":"%s","detail":"%s","action":"%s"}%s\n' \
			"$(json_escape "${CHECK_ID[$i]}")" \
			"$(json_escape "${CHECK_CATEGORY[$i]}")" \
			"$(json_escape "${CHECK_STATUS[$i]}")" \
			"$(json_escape "${CHECK_DETAIL[$i]}")" \
			"$(json_escape "${CHECK_ACTION[$i]}")" \
			"$comma"
	done
	printf '  ]\n}\n'
else
	printf 'ebook2audiobook immutable preflight (read-only)\n'
	printf 'Host: %s / %s\n' "$host_name" "$host_arch"
	for ((i = 0; i < ${#CHECK_ID[@]}; i++)); do
		printf '[%s] %s (%s): %s\n' "${CHECK_STATUS[$i]}" "${CHECK_ID[$i]}" "${CHECK_CATEGORY[$i]}" "${CHECK_DETAIL[$i]}"
		if [[ -n "${CHECK_ACTION[$i]}" ]]; then
			printf '  Next: %s\n' "${CHECK_ACTION[$i]}"
		fi
	done
	printf 'Overall: %s\n' "$overall"
	printf 'This report did not contact the network, install packages, change groups, repair files, or invoke a container.\n'
fi

if ((STRICT)) && [[ "$overall" != ok ]]; then
	exit 1
fi
exit 0
