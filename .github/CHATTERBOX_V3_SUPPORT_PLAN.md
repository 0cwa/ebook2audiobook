# Chatterbox Multilingual V3 support plan

This plan was added because GitHub Issues are currently disabled for this repository.

## Why this is not a one-line model switch

The current Chatterbox integration deliberately binds one immutable V2 runtime/model profile:

- `lib/classes/tts_engines/chatterbox.py` hard-codes `MODEL_VARIANT = "v2"`.
- `components/Chatterbox/runtime/contract_data.py` hard-codes V2 as the only model variant and includes `t3_mtl23ls_v2.safetensors` in the canonical six-file allowlist.
- `components/Chatterbox/worker.py` rejects any request whose `t3_model` is not V2, verifies exactly the V2 six-file snapshot, and currently calls `ChatterboxMultilingualTTS.from_local(..., device="cpu")` without a model selector.
- `components/Chatterbox/runtime/provisioning.py` has V2-specific model/activation fingerprint prefixes and one model/activation receipt chain per resolved manifest.
- `components/Chatterbox/runtime/runtime-manifest.json` pins exactly the V2 snapshot (six files, immutable HF revision, sizes and SHA-256s) and its storage measurements.
- `components/Chatterbox/README.md` explicitly excludes V3, Turbo, Nano, and language packs from the current profile.

Current upstream Chatterbox supports multilingual V3 through the same class:
`ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model="v3")`, where `v3` resolves to `t3_mtl23ls_v3.safetensors`.

Turbo/Nano are different integrations: they use `ChatterboxTurboTTS`, different Hugging Face repositories/checkpoint layouts, different generation semantics, and English-only behavior. They should be handled after V3.

## Recommended implementation

### 1. Keep V2 as the default and introduce explicit model profiles

Add explicit variants:

- `v2` (existing/default)
- `v3`

Keep `internal` as a backwards-compatible alias for V2 if existing sessions/configs depend on it.

Update:

- `lib/classes/tts_engines/presets/chatterbox_presets.py`
- `lib/classes/tts_engines/chatterbox.py`
- adapter/preset tests that currently assume only `internal` / V2.

The host adapter should store the selected variant and send it in `model.t3_model`.

### 2. Keep V2 and V3 as separate verified six-file snapshots

Do not create one mixed seven-file snapshot. Preserve the exact-snapshot model by keeping a separate profile/snapshot for each variant.

Common files:

- `ve.pt`
- `s3gen.pt`
- `grapheme_mtl_merged_expanded_v1.json`
- `conds.pt`
- `Cangjie5_TC.json`

Variant checkpoint:

- V2: `t3_mtl23ls_v2.safetensors`
- V3: `t3_mtl23ls_v3.safetensors`

The selected variant must be part of the model and activation identity/fingerprint.

### 3. Generalize `contract_data.py`

Current process-wide constants are V2-specific:

- `CANONICAL_MODEL_FILE_PATHS`
- `MODEL_VARIANT`

Replace them with manifest/profile-driven values (or a frozen mapping keyed by `v2` / `v3`) so the worker can validate the selected profile without weakening validation.

Do not simply change the global constant to V3; that would remove V2 support.

### 4. Add immutable V3 artifact identity

For the V3 profile:

1. Pin a specific 40-character Hugging Face commit for `ResembleAI/chatterbox`.
2. Record the exact six files for that revision.
3. Record exact byte sizes and SHA-256 values, especially `t3_mtl23ls_v3.safetensors`.
4. Preserve local-only synthesis; model acquisition remains the only network phase.
5. Re-run the disposable storage measurement workflow and update V3 model-acquisition/activation budget evidence.

Do not use a mutable model revision such as `main`, `master`, or `latest`.

### 5. Verify the pinned executable artifact exposes the V3 API

The runtime currently pins `chatterbox-tts==0.1.7`, while the manifest explicitly treats its checked source revision as provenance-only rather than verified association to the wheel.

Before relying on V3:

- inspect/test the exact pinned wheel;
- verify `ChatterboxMultilingualTTS.from_local(..., t3_model=...)` is present;
- if not, pin a V3-capable wheel/source artifact and update all package hashes and lock metadata.

The activation self-test should fail closed if the selected variant cannot be loaded.

### 6. Make the worker validate and load the selected variant

In `components/Chatterbox/worker.py`:

- derive the expected variant from the selected manifest/profile, not a global V2 constant;
- require the request variant to match the active profile;
- verify the exact six-file snapshot for that profile;
- load with:

```python
ChatterboxMultilingualTTS.from_local(
    str(snapshot_path),
    device=DEVICE,
    t3_model=selected_variant,
)
```

The worker must reject V3 requests against a V2 activation chain and vice versa.

### 7. Generalize model/activation fingerprints

In `components/Chatterbox/runtime/provisioning.py`:

- remove V2-specific fingerprint prefixes;
- include the selected model variant/profile in the canonical model identity payload;
- ensure V2 and V3 have distinct model fingerprints, model receipt paths, activation fingerprints, and activation receipt paths;
- ensure a valid activation receipt from one variant can never satisfy the other.

Preserve the current publication/rollback/quarantine behavior.

### 8. Resolve the selected profile from the host adapter

Make `_runtime_details()` / `chatterbox_host_status()` resolve status for the selected variant and pass the matching:

- model revision
- manifest path
- verified model snapshot
- model receipt
- activation receipt

to the client/worker.

The request model variant/revision must match the activated profile.

### 9. Add model selection to installer/status commands

Extend `components/Chatterbox/runtime/install.py` so model operations target a profile explicitly, for example:

```bash
python components/Chatterbox/runtime/install.py model-preflight --model v3
python components/Chatterbox/runtime/install.py acquire-model --model v3
python components/Chatterbox/runtime/install.py status --model v3
```

No-argument behavior should remain V2 for compatibility.

The Python runtime environment may remain shared if the exact same verified environment supports both variants.

### 10. Tests

Update/add tests in:

- `components/Chatterbox/runtime/tests/test_runtime.py`
- `components/Chatterbox/tests/test_worker.py`
- `components/Chatterbox/tests/test_adapter.py`
- `components/Chatterbox/tests/test_client.py`
- validation/profile tests as applicable.

Minimum coverage:

1. V2 remains default and existing V2 tests pass.
2. V3 request + V3 manifest/receipt chain is accepted.
3. V3 request + V2 chain is rejected.
4. V2 request + V3 chain is rejected.
5. Missing V3 checkpoint fails closed.
6. Wrong V3 size/hash fails closed.
7. Extra undeclared model files still fail closed.
8. V2/V3 model fingerprints differ.
9. V2/V3 activation fingerprints and receipt paths differ.
10. Host adapter selects the correct profile.
11. Worker calls `from_local(..., t3_model="v3")` for V3.
12. Local-only activation self-test passes for both variants.
13. Synthesis-time network access remains impossible.

### 11. Documentation

Update `components/Chatterbox/README.md` with:

- V2 + V3 support;
- V2 as the backwards-compatible default;
- per-variant acquisition/status commands;
- disk requirements for each verified snapshot;
- the existing Linux x86_64 / CPU first-slice limitation unless target scope is intentionally expanded.

## Follow-up: Turbo, Nano, and Single Language Pack

Treat these as a separate phase.

### Turbo/Nano

Upstream uses `ChatterboxTurboTTS` (Nano is selected with `nano=True`) with different repositories, checkpoint files, tokenizer assets, generation parameters, and English-only behavior. Supporting them cleanly likely requires another Chatterbox model family/worker profile rather than treating them as multilingual `t3_model` variants.

### Single Language Pack

The current upstream pack consists of dedicated per-language model repositories. Each needs its own immutable artifact identity and language/profile routing. Add these after the V2/V3 profile abstraction is in place.

## Acceptance criteria

- Existing V2 behavior is unchanged by default.
- Users can explicitly select multilingual V3.
- V2 and V3 can both be acquired/activated/selected without weakening immutable artifact verification.
- The worker loads V3 only from a verified local snapshot.
- Variant mismatches fail closed.
- Runtime/model/activation receipt semantics remain intact.
- Tests cover both variants and cross-variant rejection.
