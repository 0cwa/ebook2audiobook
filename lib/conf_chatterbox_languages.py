"""The deliberately small language surface exposed by the first Chatterbox slice."""

from __future__ import annotations


# E2A uses ISO-639-3 language keys. Keep this mapping local to Chatterbox so
# unsupported languages cannot silently fall through to another language.
CHATTERBOX_LANGUAGES = {
    "eng": "en",
    "swe": "sv",
}


def chatterbox_language_id(language: str | None) -> str | None:
    """Return the supported Chatterbox language ID for an E2A language key."""

    if not isinstance(language, str):
        return None
    return CHATTERBOX_LANGUAGES.get(language.lower())
