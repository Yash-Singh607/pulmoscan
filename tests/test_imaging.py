"""Image loading and OOD input-guard tests."""

import io

import numpy as np
import pytest
from PIL import Image

from chestxray.imaging import ImageDecodeError, check_input, load_image_from_bytes


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_load_png_bytes():
    arr = (np.random.rand(80, 80, 3) * 255).astype("uint8")
    img = load_image_from_bytes(_png_bytes(arr), "x.png")
    assert img.mode == "RGB"


def test_load_invalid_bytes_raises():
    with pytest.raises(ImageDecodeError):
        load_image_from_bytes(b"not an image", "x.png")


def test_check_input_grayscale_is_xray_like():
    gray = np.repeat((np.random.rand(128, 128, 1) * 255).astype("uint8"), 3, axis=2)
    check = check_input(Image.fromarray(gray, "RGB"))
    assert check.is_xray_like is True


def test_check_input_color_is_flagged():
    arr = np.zeros((128, 128, 3), dtype="uint8")
    arr[..., 0] = 220  # strong red -> high saturation
    arr[..., 2] = 20
    check = check_input(Image.fromarray(arr, "RGB"))
    assert check.is_xray_like is False
    assert "color" in check.reason.lower()


def test_check_input_too_small():
    arr = (np.random.rand(20, 20, 3) * 255).astype("uint8")
    check = check_input(Image.fromarray(arr, "RGB"))
    assert check.is_xray_like is False
