"""Image validation, conversion, and visualization helpers."""

from __future__ import annotations

import hashlib
import io
import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_INFERENCE_SIDE = 1_600
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an upload cannot be processed safely as an image."""


@dataclass(frozen=True)
class PreparedImage:
    """A validated RGB image and the metadata needed for diagnostics."""

    image: Image.Image
    original_size: tuple[int, int]
    inference_size: tuple[int, int]
    resized_for_inference: bool


def decode_image(
    data: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Image.Image:
    """Decode an upload, validate its real format, and return an RGB image.

    Validation is based on the file signature reported by Pillow, not only the
    browser-provided filename or MIME type.
    """

    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"The image is larger than the {max_bytes // (1024 * 1024)} MB limit."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                image_format = (probe.format or "").upper()
                width, height = probe.size
                if image_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(
                        "Unsupported image format. Use JPG, JPEG, PNG, or WEBP."
                    )
                if width <= 0 or height <= 0:
                    raise ImageValidationError("The image has invalid dimensions.")
                if width * height > max_pixels:
                    raise ImageValidationError(
                        f"The image exceeds the {max_pixels / 1_000_000:.0f}-megapixel limit."
                    )
                probe.verify()

            # Reopen after verify(), which intentionally invalidates the decoder.
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.seek(0)  # Use the first frame for animated WEBP files.
                decoded.load()
                oriented = ImageOps.exif_transpose(decoded)
                return oriented.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "This file could not be decoded as a valid JPG, PNG, or WEBP image."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "The image dimensions are too large to process safely."
        ) from exc
    except Image.DecompressionBombWarning as exc:
        raise ImageValidationError(
            "The image dimensions are too large to process safely."
        ) from exc


def prepare_for_inference(
    image: Image.Image, *, max_side: int = MAX_INFERENCE_SIDE
) -> PreparedImage:
    """Return an RGB image resized to a bounded inference resolution."""

    rgb = image.convert("RGB")
    original_size = rgb.size
    if max(original_size) <= max_side:
        return PreparedImage(rgb.copy(), original_size, original_size, False)

    scale = max_side / max(original_size)
    size = (
        max(1, int(round(original_size[0] * scale))),
        max(1, int(round(original_size[1] * scale))),
    )
    resized = rgb.resize(size, Image.Resampling.LANCZOS)
    return PreparedImage(resized, original_size, size, True)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a Pillow RGB image to an OpenCV BGR array."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a Pillow RGB image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")
    return Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))


def draw_face_overlay(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    selected_index: int,
) -> Image.Image:
    """Draw all detected boxes and emphasize the selected primary face."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, int(round(min(canvas.size) / 250)))

    for index, box in enumerate(boxes):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = _clamp_box(box, canvas.size)
        selected = index == selected_index
        color = (23, 181, 99) if selected else (245, 158, 11)
        width = line_width + 2 if selected else line_width
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

        confidence = confidences[index] if index < len(confidences) else 0.0
        label = (
            f"Primary {confidence:.1%}"
            if selected
            else f"Face {index + 1} {confidence:.1%}"
        )
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 8)
        draw.rectangle(
            (
                x1,
                label_y,
                min(canvas.width, x1 + text_width + 8),
                label_y + text_height + 8,
            ),
            fill=color,
        )
        draw.text((x1 + 4, label_y + 4), label, fill=(255, 255, 255))

    return canvas


def pair_seed(first: bytes, second: bytes) -> int:
    """Create a stable, order-aware seed without retaining uploaded bytes."""

    digest = hashlib.sha256(first + b"\0face-pair\0" + second).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _clamp_box(
    box: Sequence[float], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    values = [float(value) if math.isfinite(float(value)) else 0.0 for value in box[:4]]
    x1 = max(0, min(width - 1, int(round(values[0]))))
    y1 = max(0, min(height - 1, int(round(values[1]))))
    x2 = max(x1 + 1, min(width, int(round(values[2]))))
    y2 = max(y1 + 1, min(height, int(round(values[3]))))
    return x1, y1, x2, y2
