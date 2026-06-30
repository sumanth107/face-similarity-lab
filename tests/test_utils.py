from __future__ import annotations

import io
import unittest

from PIL import Image

from utils import (
    ImageValidationError,
    decode_image,
    draw_face_overlay,
    pair_seed,
    prepare_for_inference,
)


def image_bytes(image_format: str = "PNG", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class ImageUtilityTests(unittest.TestCase):
    def test_decode_supported_formats_to_rgb(self) -> None:
        for image_format in ("JPEG", "PNG", "WEBP"):
            with self.subTest(image_format=image_format):
                decoded = decode_image(image_bytes(image_format))
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, (32, 24))

    def test_decode_rejects_empty_corrupt_unsupported_and_oversized_files(self) -> None:
        with self.assertRaises(ImageValidationError):
            decode_image(b"")
        with self.assertRaises(ImageValidationError):
            decode_image(b"not an image")

        gif = image_bytes("GIF")
        with self.assertRaises(ImageValidationError):
            decode_image(gif)

        png = image_bytes("PNG")
        with self.assertRaises(ImageValidationError):
            decode_image(png, max_bytes=10)
        with self.assertRaises(ImageValidationError):
            decode_image(png, max_pixels=100)

    def test_prepare_for_inference_preserves_or_resizes(self) -> None:
        small = Image.new("RGB", (100, 50))
        unchanged = prepare_for_inference(small, max_side=100)
        self.assertFalse(unchanged.resized_for_inference)
        self.assertEqual(unchanged.inference_size, (100, 50))

        large = Image.new("RGB", (200, 100))
        resized = prepare_for_inference(large, max_side=100)
        self.assertTrue(resized.resized_for_inference)
        self.assertEqual(resized.inference_size, (100, 50))

    def test_overlay_and_pair_seed_are_deterministic(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        overlay = draw_face_overlay(
            image,
            [(10, 10, 50, 50), (55, 10, 95, 50)],
            [0.9, 0.8],
            selected_index=0,
        )
        self.assertEqual(overlay.size, image.size)
        self.assertNotEqual(overlay.getpixel((10, 10)), image.getpixel((10, 10)))
        self.assertEqual(pair_seed(b"a", b"b"), pair_seed(b"a", b"b"))
        self.assertNotEqual(pair_seed(b"a", b"b"), pair_seed(b"b", b"a"))


if __name__ == "__main__":
    unittest.main()
