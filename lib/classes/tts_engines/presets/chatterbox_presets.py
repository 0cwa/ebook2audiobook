"""Presets for the isolated Chatterbox multilingual V2/V3 worker."""

from lib.conf_models import TTS_ENGINES, default_engine_settings


_settings = default_engine_settings[TTS_ENGINES["CHATTERBOX"]]

_base = {
    "repo": _settings["repo"],
    "lang": "multi",
    "samplerate": _settings["samplerate"],
    "files": [],
    "voice": _settings["voice"],
    "voices": {},
}

models = {
    "v2": {**_base, "variant": "v2"},
    "v3": {**_base, "variant": "v3"},
}
