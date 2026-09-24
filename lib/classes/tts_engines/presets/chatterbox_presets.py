"""Presets for the isolated Chatterbox worker profiles."""

from lib.conf_models import TTS_ENGINES, default_engine_settings


_settings = default_engine_settings[TTS_ENGINES["CHATTERBOX"]]

_base = {
    "samplerate": _settings["samplerate"],
    "files": [],
    "voice": _settings["voice"],
    "voices": {},
}

models = {
    "v2": {
        **_base,
        "repo": "ResembleAI/chatterbox",
        "lang": "multi",
        "variant": "v2",
        "profile": "v2",
        "loader_kind": "multilingual",
        "family": "chatterbox-multilingual",
        "supported_language_ids": tuple(_settings["languages"].values()),
        "minimum_prompt_seconds": None,
    },
    "v3": {
        **_base,
        "repo": "ResembleAI/chatterbox",
        "lang": "multi",
        "variant": "v3",
        "profile": "v3",
        "loader_kind": "multilingual",
        "family": "chatterbox-multilingual",
        "supported_language_ids": tuple(_settings["languages"].values()),
        "minimum_prompt_seconds": None,
    },
    "turbo": {
        **_base,
        "repo": "ResembleAI/chatterbox-turbo",
        "lang": "eng",
        "profile": "turbo",
        "loader_kind": "turbo",
        "family": "chatterbox-turbo",
        "supported_language_ids": ("en",),
        "minimum_prompt_seconds": 5.0,
    },
    "nano": {
        **_base,
        "repo": "ResembleAI/chatterbox-nano",
        "lang": "eng",
        "profile": "nano",
        "loader_kind": "turbo",
        "family": "chatterbox-turbo",
        "supported_language_ids": ("en",),
        "minimum_prompt_seconds": 5.0,
    },
}
