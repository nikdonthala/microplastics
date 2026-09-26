"""Image I/O utilities.

Uploaded images are processed entirely in memory: filenames are only used to
check the extension, files are never written to disk, and no internal
filesystem paths are ever exposed to the frontend.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


def is_allowed_extension(filename: str | None) -> bool:
    """Return True when the filename has a supported image extension."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in settings.ALLOWED_EXTENSIONS


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR image.

    Raises :class:`InvalidImageError` when the bytes are empty or are not a
    decodable JPG/JPEG/PNG payload. This also protects against files that only
    *pretend* to be images by extension.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(
            "The uploaded file could not be decoded as an image. "
            "Supported formats: JPG, JPEG, PNG."
        )
    return image


def encode_image_data_uri(image: np.ndarray) -> str:
    """Encode a BGR image as a PNG data URI usable directly in an <img> tag."""
    success, buffer = cv2.imencode(".png", image)
    if not success:  # pragma: no cover - practically unreachable
        raise ValueError("Failed to encode image as PNG.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
