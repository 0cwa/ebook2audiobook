"""Dependency-light regression tests for shared lib.core text helpers."""

import unittest

import lib.core as core


class CoreTextTests(unittest.TestCase):
    def test_natural_sort_key_uses_numeric_basename_runs(self):
        paths = [
            "/cache/hash-z/volume-10.epub",
            "/cache/hash-a/volume-2.epub",
            "/cache/hash-b/volume-1.epub",
        ]

        self.assertEqual(
            sorted(paths, key=core.natural_sort_key),
            [paths[2], paths[1], paths[0]],
        )

    def test_token_spacing_preserves_contractions_and_sml_boundaries(self):
        self.assertEqual(core.foreign2latin("can't", "eng"), "can't")
        self.assertEqual(
            core.foreign2latin("hello [pause] world", "eng"),
            "hello [pause] world",
        )

    def test_normalize_text_assigns_emoji_removal_result(self):
        self.assertEqual(
            core.normalize_text("Hello 😀 world", "eng", "en", "piper"),
            "Hello world",
        )


if __name__ == "__main__":
    unittest.main()
