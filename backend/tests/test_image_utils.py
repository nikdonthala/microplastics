"""Tests for image decoding/encoding utilities."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.utils.image_utils import (
    InvalidImageError,
    decode_image_bytes,
    encode_image_data_uri,
    is_allowed_extension,
)


def test_allowed_extensions():
    assert is_allowed_extension("sample.jpg")
    assert is_allowed_extension("sample.JPEG")
    assert is_allowed_extension("sample.png")
    assert not is_allowed_extension("sample.exe")
    assert not is_allowed_extension("sample.gif")
    assert not is_allowed_extension(None)
    assert not is_allowed_extension("no_extension")


def test_decode_empty_bytes_raises():
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"")


def test_decode_invalid_bytes_raises():
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"this is definitely not an image")


def test_decode_valid_png():
    image = np.full((40, 50, 3), 128, dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    decoded = decode_image_bytes(buffer.tobytes())
    assert decoded.shape == (40, 50, 3)


def test_decode_text_file_with_image_extension_raises():
    # A file named .png that is actually text must be rejected.
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"fake png content")


def test_encode_data_uri_roundtrip():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    uri = encode_image_data_uri(image)
    assert uri.startswith("data:image/png;base64,")
