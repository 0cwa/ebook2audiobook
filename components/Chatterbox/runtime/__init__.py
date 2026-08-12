"""User-local Chatterbox runtime provisioning primitives.

This package deliberately contains no Chatterbox imports.  It is safe to
import from the host environment and only provisions an isolated worker
environment when its immutable lock and source identities have been verified.
"""

from .runtime import (
    RuntimePaths,
    build_paths,
    preflight,
    sanitized_worker_environment,
    verify_lock,
)

__all__ = [
    "RuntimePaths",
    "build_paths",
    "preflight",
    "sanitized_worker_environment",
    "verify_lock",
]
