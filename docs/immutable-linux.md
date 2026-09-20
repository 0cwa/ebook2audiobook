# Immutable Linux: rootless CPU path

This guide is the supported no-sudo route for an immutable Linux desktop. It is
deliberately small: a read-only application checkout, a rootless Podman CPU
image, and one explicit user-owned data root.

“No sudo” means this path does not invoke sudo, a host package manager,
usermod, sg, driver installation, or host security-policy changes. It does not
mean that the host has no prerequisites. The user must already have rootless
Podman and a Compose provider available through the host's approved user-space
mechanisms.

## Quick start

Run the read-only preflight first:

~~~bash
./ebook2audiobook.command --diagnose
./ebook2audiobook.command --diagnose --json > ./tmp/ebook2audiobook-diagnostics.json
~~~

The diagnostic command does not create the data root, write the checkout, use
the network, install anything, repair an incomplete runtime, or invoke Podman.
The JSON report has stable check IDs and categories suitable for support
bundles. It does not dump the process environment.

Choose a data root outside the checkout. The root is persistent user data, not
build scratch space:

~~~bash
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/ebook2audiobook"
~~~

Create the declared directories and build the CPU image in rootless Podman
storage:

~~~bash
./tools/immutable-launch.sh setup --data-root "$DATA_ROOT"
~~~

The existing Dockerfile may need network access for its base image and package
layers. The launcher does not install host packages. If the image or model is
already cached, no network is needed for that particular step.

On a hardened host, image acquisition can be rejected even when rootless
Podman and Compose are working. For example, secureblue may report:

~~~text
Source image rejected: Running image docker://python:3.12-slim-trixie is rejected by policy.
~~~

That message is an image-trust policy failure, not a missing-Podman failure.
Do not accept an unsigned registry, weaken the host policy, reuse an unrelated
cached image, or retry with privileged mode. Preserve the exact error and use
an approved trusted mirror or an isolated VM/build host supplied by the host
administrator. The launcher makes no host security change.

Start the browser GUI:

~~~bash
./tools/immutable-launch.sh run --data-root "$DATA_ROOT"
~~~

Open http://127.0.0.1:7860/ in a browser. The container GUI uses HTTP; it does
not need the host display server, audio D-Bus, or an audio device inside the
container.

For a one-shot headless conversion, paths after -- are interpreted inside
the container:

~~~bash
./tools/immutable-launch.sh run --data-root "$DATA_ROOT" -- \
  --headless --ebook /app/ebooks/book.epub --language eng
~~~

The launcher shorthand is equivalent:

~~~bash
./ebook2audiobook.command --container setup --data-root "$DATA_ROOT"
./ebook2audiobook.command --container run --data-root "$DATA_ROOT"
~~~

## What the launcher owns

The data root contains only declared application data:

| Host path | Container path | Purpose | Kept on update/uninstall |
| --- | --- | --- | --- |
| ebooks | /app/ebooks | input books | yes |
| audiobooks | /app/audiobooks | generated output | yes |
| models | /app/models | downloaded model/cache data | yes |
| voices | /app/voices | voice files | yes |
| run | /app/run | runtime state and temporary process files | yes by default |
| tmp | /app/tmp | application temporary data | yes by default |

The checkout is used as a read-only build context and is not mounted as
/app. The CPU compose profile requires E2A_DATA_ROOT and fails closed when it
is missing. The CPU browser port is bound to 127.0.0.1; non-CPU profiles keep
their existing binding. The compose file does not use privileged,
label=disable, or :Z; the path must not weaken SELinux or relabel user data.

The rootless Podman image is local:

~~~text
localhost/athomasson2/ebook2audiobook:cpu
~~~

The application manifest is
[immutable-installer-manifest.json](../immutable-installer-manifest.json).

## Lifecycle

### Update

Rebuild the mutable local image tag and recreate the container while retaining
the external data:

~~~bash
./tools/immutable-launch.sh update --data-root "$DATA_ROOT"
~~~

The command may download newer Dockerfile inputs. It does not replace or
relocate the data root. This is not an atomic activation and the launcher does
not write an installer receipt. Before setup or update, keep a small local
operator record outside the checkout with the source revision, data-root path,
image ID or digest, and build log path. Rollback is manual: restore a recorded
image or rebuild the recorded source revision. The launcher does not delete the
prior data root.

### Diagnose

Use the same read-only report after a failed launch:

~~~bash
./ebook2audiobook.command --diagnose --json
./tools/immutable-launch.sh diagnose --json
~~~

For a strict CI/support check, add --strict; the command still prints the
report and returns nonzero when an action is required.

Stable categories include:

| Category | Meaning | Next action |
| --- | --- | --- |
| unsupported-host | OS or architecture is outside the recorded CPU contract | Use a supported Linux CPU host or treat the row as best effort |
| missing-dependency | Native mode lacks a host command or supported Python | Use the container path; no package manager is invoked |
| unwritable-checkout | Native setup would need source writes | Use the read-only container path |
| storage | Data root is missing or not writable | Select a user-owned absolute root and run setup |
| network | Network was intentionally not probed | Provide network only for uncached image/model acquisition |
| container-availability | Podman or a Compose provider is not visible | Install/expose it through the host's approved user-space method |
| runtime-model-activation-readiness | Native runtime or model data is not ready | Use the container setup or repair native mode explicitly |
| repair-required | An incomplete native runtime was found | No repair is attempted by diagnostics |
| worker-failure | The application worker failed after launch | Inspect the container log and preserve the data root |
| ui-bind-failure | Browser or HTTP bind is unavailable | Open http://127.0.0.1:7860/ manually and inspect logs |
| output-failure | Conversion could not write or validate output | Check audiobooks, storage, and input/voice paths |

The last three categories are runtime troubleshooting labels; the dependency-light
preflight does not pretend to reproduce an application worker or conversion.

### Uninstall and cleanup

Preview first:

~~~bash
./tools/immutable-launch.sh uninstall --preview --data-root "$DATA_ROOT"
~~~

Apply removes the selected Compose project's containers and network:

~~~bash
./tools/immutable-launch.sh uninstall --apply --data-root "$DATA_ROOT"
~~~

The data root and image are retained. This is intentional: runtime cleanup and
user-data deletion are different operations. The launcher does not write an
installer receipt; use the preview and your local operator record to verify the
selected project before applying cleanup. Back up the exact root before
removing it manually. The launcher never recursively deletes user data.

## Immutable host boundary

The portable general route assumes:

- Linux x86_64 or aarch64;
- rootless Podman and either podman compose or podman-compose;
- a writable user-owned data root outside the checkout;
- network access when the image or model is not cached;
- a host browser for the HTTP GUI, unless using headless mode.

Fedora Silverblue/Kinoite and other Fedora Atomic desktops, Universal Blue
desktops, and openSUSE Aeon are the Tier 1 target family for this CPU route,
subject to row-specific evidence. NixOS, Vanilla OS, openSUSE MicroOS, Endless
OS, and similar systems are best-effort Tier 2. Systems without a usable
desktop session, rootless runtime, writable user storage, or device plumbing
are Tier 3 experimental. This is not a universal distribution certification.

On hardened hosts such as secureblue, SELinux and container policy remain
enabled. If rootless Podman reports a user-namespace, storage-label, or policy
error, preserve the exact error and stop. Do not disable SELinux, add
privileged, add label=disable, enable broad unconfined user namespaces, or
ask this launcher to make an administrator change. A host administrator may
need to provide a narrow approved prerequisite; that is outside this
user-local installer contract.

GPU profiles remain conditional. This lane proves only the CPU path. It does
not install drivers, modify device groups, or claim that CUDA, ROCm, XPU, or
Jetson works on every immutable host.

## Direct Compose use

The helper is preferred because it creates only the declared data directories
and checks rootless mode. For inspection without launching:

~~~bash
E2A_DATA_ROOT="$DATA_ROOT" DEVICE_TAG=cpu \
  podman compose -f podman-compose.yml --profile cpu config
~~~

A successful config expansion is not an application smoke test. A real
acceptance run must build the CPU image, exercise an HTTP or headless operation,
verify that a marker in the external root survives update, and run the
uninstall preview/apply cleanup without deleting that root.
