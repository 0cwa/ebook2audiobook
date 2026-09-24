"""Immutable, dependency-free Chatterbox contract data.

This module is deliberately limited to values that are stable across the
runtime and worker boundary. The checked-in runtime manifest remains the
authority for model repositories, revisions, file paths, sizes, hashes, and
other provenance-bearing artifact data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


_SUPPORTED_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
)
_APPROVED_LANGUAGE_IDS = frozenset(_SUPPORTED_LANGUAGES)


@dataclass(frozen=True, slots=True)
class ModelProfileSpec:
    """Non-artifact semantics for one Chatterbox model profile."""

    profile: str
    loader_kind: str
    family: str
    supported_languages: tuple[str, ...]
    fixed_language_id: str | None
    minimum_prompt_seconds: float | None


_MODEL_PROFILE_SPECS: Final = {
    "v2": ModelProfileSpec(
        profile="v2",
        loader_kind="multilingual",
        family="chatterbox-multilingual",
        supported_languages=_SUPPORTED_LANGUAGES,
        fixed_language_id=None,
        minimum_prompt_seconds=None,
    ),
    "v3": ModelProfileSpec(
        profile="v3",
        loader_kind="multilingual",
        family="chatterbox-multilingual",
        supported_languages=_SUPPORTED_LANGUAGES,
        fixed_language_id=None,
        minimum_prompt_seconds=None,
    ),
}
_MODEL_PROFILE_ALIASES: Final = {
    "v2": "v2",
    "multilingual-v2": "v2",
    "v3": "v3",
    "multilingual-v3": "v3",
}
_SUPPORTED_MODEL_PROFILES = tuple(_MODEL_PROFILE_SPECS)


def normalize_model_profile(value: object) -> str | None:
    """Return the stable profile ID for a manifest/request value."""

    if not isinstance(value, str):
        return None
    return _MODEL_PROFILE_ALIASES.get(value.strip().lower())


def model_profile_spec(value: object) -> ModelProfileSpec:
    """Return immutable non-artifact semantics for a supported profile."""

    normalized = normalize_model_profile(value)
    if normalized is None:
        raise ValueError(f"unsupported Chatterbox model profile: {value!r}")
    return _MODEL_PROFILE_SPECS[normalized]


# Backwards-compatible name while callers migrate from V2/V3 "variant" wording.
normalize_model_variant = normalize_model_profile


# spacy-pkuseg ships its Python extension as a wheel but downloads this model
# archive on first use. Keep the package's own pinned identity explicit at
# the worker boundary so a local-only worker cannot silently fetch it.
PKUSEG_DATA_FILENAME: Final = "spacy_ontonotes.zip"
PKUSEG_DATA_URL: Final = (
    "https://github.com/explosion/spacy-pkuseg/releases/download/"
    "v0.0.26/spacy_ontonotes.zip"
)
PKUSEG_DATA_SHA256: Final = "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe"


@dataclass(frozen=True, slots=True)
class ContractData:
    """The stable, non-provenance portion of the Chatterbox CPU contract."""

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
    protocol_version=1,
    sample_rate=24000,
    channels=1,
    device="cpu",
    model_family=_MODEL_PROFILE_SPECS["v2"].family,
    model_variant="v2",
    supported_languages=_SUPPORTED_LANGUAGES,
    approved_language_ids=_APPROVED_LANGUAGE_IDS,
    max_segments=32,
    max_text_chars=12000,
    max_total_text_chars=40000,
    max_silence_seconds=30.0,
)


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
PROTOCOL_VERSION: Final = CONTRACT_DATA.protocol_version
SAMPLE_RATE: Final = CONTRACT_DATA.sample_rate
CHANNELS: Final = CONTRACT_DATA.channels
DEVICE: Final = CONTRACT_DATA.device
MODEL_FAMILY: Final = CONTRACT_DATA.model_family
MODEL_VARIANT: Final = CONTRACT_DATA.model_variant
SUPPORTED_MODEL_PROFILES: Final = _SUPPORTED_MODEL_PROFILES
SUPPORTED_MODEL_VARIANTS: Final = SUPPORTED_MODEL_PROFILES
SUPPORTED_LANGUAGES: Final = CONTRACT_DATA.supported_languages
APPROVED_LANGUAGE_IDS: Final = CONTRACT_DATA.approved_language_ids
MAX_SEGMENTS: Final = CONTRACT_DATA.max_segments
MAX_TEXT_CHARS: Final = CONTRACT_DATA.max_text_chars
MAX_TOTAL_TEXT_CHARS: Final = CONTRACT_DATA.max_total_text_chars
MAX_SILENCE_SECONDS: Final = CONTRACT_DATA.max_silence_seconds


__all__ = [
    "ACTIVATION_RECEIPT_SCHEMA",
    "APPROVED_LANGUAGE_IDS",
    "CHANNELS",
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
    "ModelProfileSpec",
    "SUPPORTED_MODEL_PROFILES",
    "SUPPORTED_MODEL_VARIANTS",
    "model_profile_spec",
    "normalize_model_profile",
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
