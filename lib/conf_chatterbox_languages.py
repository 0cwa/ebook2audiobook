"""E2A-to-Chatterbox language IDs for the multilingual model."""

from __future__ import annotations


# E2A uses ISO-639-3 language keys. Keep this mapping local to Chatterbox so
# unsupported languages cannot silently fall through to another language.
CHATTERBOX_LANGUAGES = {
    "ara": "ar",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "spa": "es",
    "fin": "fi",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "zlm": "ms",
    "nld": "nl",
    "nob": "no",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "swe": "sv",
    "swh": "sw",
    "tur": "tr",
    "zho": "zh",
}


def chatterbox_language_id(language: str | None) -> str | None:
    """Return the supported Chatterbox language ID for an E2A language key."""

    if not isinstance(language, str):
        return None
    return CHATTERBOX_LANGUAGES.get(language.lower())
