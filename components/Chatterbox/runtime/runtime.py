"""Compatibility import for the Chatterbox provisioning implementation.

The host-facing import path remains ``components.Chatterbox.runtime.runtime``
for existing callers.  Implementation ownership lives in ``provisioning`` so
the public runtime name stays a small compatibility seam instead of becoming
the home for installer, receipt, model, and measurement concerns.
"""

from __future__ import annotations

import sys as _sys

from . import provisioning as _implementation

# Preserve the historical module identity for callers and for integrations
# that patch the existing ``runtime`` seam.  The implementation module owns
# the symbols and their globals; no duplicate wrappers can drift from it.
_sys.modules[__name__] = _implementation
