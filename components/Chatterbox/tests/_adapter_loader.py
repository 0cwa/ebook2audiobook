"""Component-local loader for the host adapter's optional-engine seam."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types


_LOADED_MODULE = "components.Chatterbox.tests._loaded_chatterbox_adapter"


def _load_chatterbox_presets(repository: Path):
    """Load the dependency-light preset data without retaining package stubs."""

    preset_path = repository / "lib/classes/tts_engines/presets/chatterbox_presets.py"
    spec = importlib.util.spec_from_file_location(
        "components.Chatterbox.tests._chatterbox_presets",
        preset_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Chatterbox presets from {preset_path}")
    presets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(presets)
    return presets.models


def load_chatterbox_adapter():
    """Load the adapter once without importing unrelated optional engines."""

    cached = sys.modules.get(_LOADED_MODULE)
    if cached is not None:
        return cached

    repository = Path(__file__).resolve().parents[3]
    package_name = "lib.classes.tts_engines"
    utils_name = f"{package_name}.common.utils"
    adapter_name = f"{package_name}.chatterbox"
    module_prefix = f"{package_name}."
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(module_prefix)
    }
    package = types.ModuleType(package_name)
    package.__path__ = [str(repository / "lib/classes/tts_engines")]

    from lib.conf_models import SML_TAG_PATTERN

    utils = types.ModuleType(utils_name)

    class TTSUtils:
        @staticmethod
        def _split_sentence_on_sml(sentence: str) -> list[str]:
            parts: list[str] = []
            last = 0
            for match in SML_TAG_PATTERN.finditer(sentence):
                start, end = match.span()
                if start > last:
                    parts.append(sentence[last:start])
                parts.append(match.group(0))
                last = end
            if last < len(sentence):
                parts.append(sentence[last:])
            return parts

    utils.TTSUtils = TTSUtils
    presets = _load_chatterbox_presets(repository)
    try:
        # Keep the synthetic package installed only for the adapter import.
        # The imported module is retained under the dedicated cache name, but
        # the repository namespace must be restored for later tests/imports.
        sys.modules[package_name] = package
        sys.modules[utils_name] = utils
        module = importlib.import_module(adapter_name)

        def load_engine_presets(engine: str):
            if engine != "chatterbox":
                raise ImportError(f"Unsupported test adapter preset: {engine}")
            return presets

        module.load_engine_presets = load_engine_presets
        sys.modules[_LOADED_MODULE] = module
        return module
    finally:
        current_names = {
            name
            for name in sys.modules
            if name == package_name or name.startswith(module_prefix)
        }
        for name in current_names - previous_modules.keys():
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            sys.modules[name] = previous
