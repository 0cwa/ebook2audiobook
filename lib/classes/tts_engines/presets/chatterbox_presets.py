"""Presets for the isolated Chatterbox multilingual V2 worker."""

from lib.conf_models import TTS_ENGINES, default_engine_settings


_settings = default_engine_settings[TTS_ENGINES["CHATTERBOX"]]

models = {
    "internal": {
        "repo": _settings["repo"],
        "samplerate": _settings["samplerate"],
        "files": [],
        "voice": _settings["voice"],
        "voices": {},
    }
}
