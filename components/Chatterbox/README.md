# Chatterbox Multilingual V2/V3 runtime

This component integrates Chatterbox Multilingual V2 and V3 as optional,
isolated workers. V2 remains the default for backwards compatibility. The
supported target is Linux x86_64, CPU-only, using a dedicated Python 3.11
environment. The Python version used by the main
ebook2audiobook process does not need to be Python 3.11.

The runtime and model are provisioned separately:

```bash
python components/Chatterbox/runtime/install.py preflight --python /path/to/python3.11
python components/Chatterbox/runtime/install.py install --python /path/to/python3.11
python components/Chatterbox/runtime/install.py model-preflight
python components/Chatterbox/runtime/install.py acquire-model
python components/Chatterbox/runtime/install.py status
```

V3 uses the same verified runtime but has its own model snapshot and receipt
chain:

```bash
python components/Chatterbox/runtime/install.py model-preflight --model v3
python components/Chatterbox/runtime/install.py acquire-model --model v3
python components/Chatterbox/runtime/install.py status --model v3
```

The model acquisition command is the only model-network phase. It downloads
exactly the six files declared by the selected profile
(`runtime/runtime-manifest.json` for V2 or
`runtime/runtime-manifest-v3.json` for V3) from the declared immutable
Hugging Face revision and verifies every size and SHA-256 value before
publishing readiness. Normal worker startup uses `from_local()`
with offline and telemetry-safe environment settings and never downloads a
model.

## Storage and artifact layout

Runtime environments are immutable objects beneath:

```text
$XDG_DATA_HOME/ebook2audiobook/chatterbox/envs/objects/<runtime-fingerprint>/<transaction-id>
```

Verified model snapshots use exactly one canonical layout:

```text
$XDG_DATA_HOME/ebook2audiobook/chatterbox/models/objects/<model-fingerprint>/<transaction-id>/snapshot
```

A model receipt selects one snapshot object. Existing Hugging Face cache data
under `$E2A_MODELS_DIR/tts/chatterbox` is legacy cache state: it is preserved,
never adopted as verified content, and never passed to the synthesis worker.

Readiness is split across three private receipts in
`$XDG_STATE_HOME/ebook2audiobook/chatterbox`:

- the runtime receipt binds the validated Python environment;
- the model receipt binds the exact verified six-file snapshot;
- the activation receipt binds compatible runtime and model receipts after a
  local-only model-load self-test.

All three receipts are required for complete product readiness. Directory
presence or the legacy installation-result record is not readiness proof.

The V2 six-file snapshot requires exactly 3,208,951,748 bytes; the V3
snapshot requires 3,208,951,924 bytes. The completed
disposable measurement and its allocation-aware budget formulas are recorded
in [`runtime/measurement-evidence.json`](runtime/measurement-evidence.json).
The checked-in manifest now has determinate budgets for every storage bucket,
including explicit `not_applicable` provenance for model staging and Xet
metadata. Normal preflight still fails closed for an unresolved or insufficient
budget and performs no provisioning mutation in either case.

Failures clean only the current transaction's owned candidate. Unowned,
symlinked, path-escaping, partial, or receipt-ambiguous state fails closed as
`repair_required`; it is not deleted automatically.

## Scope and provenance

These profiles intentionally exclude CUDA, MPS, ROCm, XPU, Jetson, Turbo,
Nano, single-language packs, automatic model management, cache garbage
collection, and strict offline installation. After successful model acquisition and activation,
synthesis itself is local-only.

The runtime remains pinned to Chatterbox 0.1.7. That wheel predates the public
V3 `t3_model` selector, so the isolated worker contains a narrow compatibility
loader based on upstream commit
`3f35dfc8fbe63e5b29793289dc68f1875bb317a5`: it loads the verified V3 T3
checkpoint directly and disables the V2-only alignment analyzer. If a future
pinned runtime exposes the upstream V3 selector, the worker uses it directly.

Wheel SHA-256 values identify the executable Chatterbox and Perth packages.
Recorded source commits are compatibility/provenance references and are not
claimed as reproducible build sources. The immutable model revision plus its
six file paths, sizes, and SHA-256 values identify the model.

Chatterbox and Perth are MIT-licensed. Generated audio includes the upstream
Perth perceptual watermark; see `NOTICE.md` for the bundled notice.
