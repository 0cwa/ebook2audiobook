"""User-local Chatterbox runtime provisioning primitives.

This package deliberately contains no Chatterbox imports.  It is safe to
import from the host environment and only provisions an isolated worker
environment when its immutable lock and source identities have been verified.
"""

from .runtime import (
    RuntimePaths,
    build_identity_contract,
    build_paths,
    calculate_storage_plan,
    preflight,
    receipt_identity_contracts,
    runtime_status,
    sanitized_worker_environment,
    validate_manifest_identity,
    validate_receipt_identity_links,
    validate_runtime_receipt,
    verified_file_credit,
    verify_lock,
)

__all__ = [
    "RuntimePaths",
    "build_identity_contract",
    "build_paths",
    "calculate_storage_plan",
    "preflight",
    "receipt_identity_contracts",
    "runtime_status",
    "sanitized_worker_environment",
    "validate_manifest_identity",
    "validate_receipt_identity_links",
    "validate_runtime_receipt",
    "verified_file_credit",
    "verify_lock",
]
