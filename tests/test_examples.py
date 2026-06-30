from __future__ import annotations

import unittest
from pathlib import Path

from utils import decode_image


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[1] / "test_images"
EXAMPLE_FILES = (
    "victoria_justice_2018.png",
    "victoria_justice_2012.jpg",
    "nina_dobrev_2018.png",
    "nina_dobrev_2011.jpg",
)


class ExampleImageTests(unittest.TestCase):
    def test_all_documented_examples_decode_as_supported_rgb_images(self) -> None:
        for filename in EXAMPLE_FILES:
            with self.subTest(filename=filename):
                path = EXAMPLE_DIRECTORY / filename
                self.assertTrue(path.is_file())
                image = decode_image(path.read_bytes())
                self.assertEqual(image.mode, "RGB")
                self.assertGreaterEqual(min(image.size), 300)


if __name__ == "__main__":
    unittest.main()
