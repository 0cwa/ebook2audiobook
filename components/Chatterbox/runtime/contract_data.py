"""Immutable, dependency-free Chatterbox contract data.

This module is deliberately limited to values that are stable across the
runtime and worker boundary.  The checked-in runtime manifest remains the
authority for model revisions, file sizes and hashes, lock contents, and
other provenance-bearing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


_COMMON_MODEL_FILE_PATHS = (
    "ve.pt",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
_MODEL_FILE_PATHS = {
    "v2": ("ve.pt", "t3_mtl23ls_v2.safetensors", *_COMMON_MODEL_FILE_PATHS[1:]),
    "v3": ("ve.pt", "t3_mtl23ls_v3.safetensors", *_COMMON_MODEL_FILE_PATHS[1:]),
}
_MODEL_VARIANT_ALIASES = {
    "v2": "v2",
    "multilingual-v2": "v2",
    "v3": "v3",
    "multilingual-v3": "v3",
}
_SUPPORTED_MODEL_VARIANTS = ("v2", "v3")
_CANONICAL_MODEL_FILE_PATHS = _MODEL_FILE_PATHS["v2"]


def normalize_model_variant(value: object) -> str | None:
    """Return the stable worker variant name for a manifest/request value."""

    if not isinstance(value, str):
        return None
    return _MODEL_VARIANT_ALIASES.get(value.strip().lower())


def canonical_model_file_paths(variant: object) -> tuple[str, ...]:
    """Return the exact six-file allowlist for one supported model variant."""

    normalized = normalize_model_variant(variant)
    if normalized is None:
        raise ValueError(f"unsupported Chatterbox model variant: {variant!r}")
    return _MODEL_FILE_PATHS[normalized]
_SUPPORTED_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
)
_APPROVED_LANGUAGE_IDS = frozenset(_SUPPORTED_LANGUAGES)

# spacy-pkuseg ships its Python extension as a wheel but downloads this model
# archive on first use.  Keep the package's own pinned identity explicit at
# the worker boundary so a local-only worker cannot silently fetch it.
PKUSEG_DATA_FILENAME: Final = "spacy_ontonotes.zip"
PKUSEG_DATA_URL: Final = (
    "https://github.com/explosion/spacy-pkuseg/releases/download/"
    "v0.0.26/spacy_ontonotes.zip"
)
PKUSEG_DATA_SHA256: Final = "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe"


@dataclass(frozen=True, slots=True)
class ContractData:
    """The stable, non-provenance portion of the V2 CPU contract."""

    manifest_version: str
    runtime_contract_version: str
    runtime_receipt_schema: str
    model_receipt_schema: str
    activation_receipt_schema: str
    storage_contract_version: str
    storage_measurement_evidence_schema: str
    target_os: str
    target_arch: str
    target_python: tuple[int, int]
    target_backend: str
    runtime_install_storage_phases: tuple[str, ...]
    model_acquisition_storage_phases: tuple[str, ...]
    canonical_model_file_paths: tuple[str, ...]
    protocol_version: int
    sample_rate: int
    channels: int
    device: str
    model_family: str
    model_variant: str
    supported_languages: tuple[str, ...]
    approved_language_ids: frozenset[str]
    max_segments: int
    max_text_chars: int
    max_total_text_chars: int
    max_silence_seconds: float


CONTRACT_DATA: Final = ContractData(
    manifest_version="2.0.0",
    runtime_contract_version="2.0.0",
    runtime_receipt_schema="ebook2audiobook.chatterbox-runtime-receipt.v1",
    model_receipt_schema="ebook2audiobook.chatterbox-model-receipt.v1",
    activation_receipt_schema="ebook2audiobook.chatterbox-activation-receipt.v1",
    storage_contract_version="1.0.0",
    storage_measurement_evidence_schema="ebook2audiobook.chatterbox-storage-measurement.v1",
    target_os="linux",
    target_arch="x86_64",
    target_python=(3, 11),
    target_backend="cpu",
    runtime_install_storage_phases=(
        "runtime_artifact_acquisition",
        "runtime_environment_construction",
    ),
    model_acquisition_storage_phases=(
        "model_acquisition",
        "activation_and_self_test",
    ),
    canonical_model_file_paths=_CANONICAL_MODEL_FILE_PATHS,
    protocol_version=1,
    sample_rate=24000,
    channels=1,
    device="cpu",
    model_family="chatterbox-multilingual",
    model_variant="v2",
    supported_languages=_SUPPORTED_LANGUAGES,
    approved_language_ids=_APPROVED_LANGUAGE_IDS,
    max_segments=32,
    max_text_chars=12000,
    max_total_text_chars=40000,
    max_silence_seconds=30.0,
)


# Named aliases are the small import surface for runtime, worker, and
# validator callers.  Every sequence/set below is owned by the frozen record,
# so callers share the exact immutable object rather than copying contract
# values at import time.
MANIFEST_VERSION: Final = CONTRACT_DATA.manifest_version
RUNTIME_CONTRACT_VERSION: Final = CONTRACT_DATA.runtime_contract_version
RUNTIME_RECEIPT_SCHEMA: Final = CONTRACT_DATA.runtime_receipt_schema
MODEL_RECEIPT_SCHEMA: Final = CONTRACT_DATA.model_receipt_schema
ACTIVATION_RECEIPT_SCHEMA: Final = CONTRACT_DATA.activation_receipt_schema
STORAGE_CONTRACT_VERSION: Final = CONTRACT_DATA.storage_contract_version
STORAGE_MEASUREMENT_EVIDENCE_SCHEMA: Final = CONTRACT_DATA.storage_measurement_evidence_schema
TARGET_OS: Final = CONTRACT_DATA.target_os
TARGET_ARCH: Final = CONTRACT_DATA.target_arch
TARGET_PYTHON: Final = CONTRACT_DATA.target_python
TARGET_BACKEND: Final = CONTRACT_DATA.target_backend
RUNTIME_INSTALL_STORAGE_PHASES: Final = CONTRACT_DATA.runtime_install_storage_phases
MODEL_ACQUISITION_STORAGE_PHASES: Final = CONTRACT_DATA.model_acquisition_storage_phases
CANONICAL_MODEL_FILE_PATHS: Final = CONTRACT_DATA.canonical_model_file_paths
PROTOCOL_VERSION: Final = CONTRACT_DATA.protocol_version
SAMPLE_RATE: Final = CONTRACT_DATA.sample_rate
CHANNELS: Final = CONTRACT_DATA.channels
DEVICE: Final = CONTRACT_DATA.device
MODEL_FAMILY: Final = CONTRACT_DATA.model_family
MODEL_VARIANT: Final = CONTRACT_DATA.model_variant
SUPPORTED_MODEL_VARIANTS: Final = _SUPPORTED_MODEL_VARIANTS
SUPPORTED_LANGUAGES: Final = CONTRACT_DATA.supported_languages
APPROVED_LANGUAGE_IDS: Final = CONTRACT_DATA.approved_language_ids
MAX_SEGMENTS: Final = CONTRACT_DATA.max_segments
MAX_TEXT_CHARS: Final = CONTRACT_DATA.max_text_chars
MAX_TOTAL_TEXT_CHARS: Final = CONTRACT_DATA.max_total_text_chars
MAX_SILENCE_SECONDS: Final = CONTRACT_DATA.max_silence_seconds


__all__ = [
    "ACTIVATION_RECEIPT_SCHEMA",
    "APPROVED_LANGUAGE_IDS",
    "CANONICAL_MODEL_FILE_PATHS",
    "CHANNELS",
    "canonical_model_file_paths",
    "CONTRACT_DATA",
    "ContractData",
    "DEVICE",
    "MANIFEST_VERSION",
    "MAX_SEGMENTS",
    "MAX_SILENCE_SECONDS",
    "MAX_TEXT_CHARS",
    "MAX_TOTAL_TEXT_CHARS",
    "MODEL_ACQUISITION_STORAGE_PHASES",
    "MODEL_FAMILY",
    "MODEL_RECEIPT_SCHEMA",
    "MODEL_VARIANT",
    "SUPPORTED_MODEL_VARIANTS",
    "normalize_model_variant",
    "PKUSEG_DATA_FILENAME",
    "PKUSEG_DATA_SHA256",
    "PKUSEG_DATA_URL",
    "PROTOCOL_VERSION",
    "RUNTIME_CONTRACT_VERSION",
    "RUNTIME_INSTALL_STORAGE_PHASES",
    "RUNTIME_RECEIPT_SCHEMA",
    "SAMPLE_RATE",
    "STORAGE_CONTRACT_VERSION",
    "STORAGE_MEASUREMENT_EVIDENCE_SCHEMA",
    "SUPPORTED_LANGUAGES",
    "TARGET_ARCH",
    "TARGET_BACKEND",
    "TARGET_OS",
    "TARGET_PYTHON",
]
