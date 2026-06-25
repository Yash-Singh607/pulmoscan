"""Image loading (incl. DICOM) and an out-of-distribution input guard.

The OOD guard is a lightweight heuristic, not a learned detector. It catches
the most common real-world failure mode — someone uploads a color photo,
screenshot, or other non-radiograph — by checking that the image is
essentially grayscale and a sensible size. It is intentionally conservative.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(Exception):
    """Raised when an uploaded file cannot be decoded as an image/DICOM."""


def _looks_like_dicom(raw: bytes, filename: str) -> bool:
    if filename.lower().endswith(".dcm"):
        return True
    # DICOM files carry the magic "DICM" at byte offset 128.
    return len(raw) > 132 and raw[128:132] == b"DICM"


def _load_dicom(raw: bytes) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImageDecodeError(
            "DICOM upload received but 'pydicom' is not installed. "
            "Install it with: pip install pydicom"
        ) from exc

    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:  # pydicom raises various errors
        raise ImageDecodeError("Could not read DICOM pixel data.") from exc

    # Normalize to 0-255.
    arr -= arr.min()
    peak = arr.max()
    if peak > 0:
        arr = arr / peak * 255.0
    arr = arr.astype(np.uint8)

    # MONOCHROME1 stores inverted intensities (white = low value).
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 255 - arr

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return Image.fromarray(arr[..., :3], mode="RGB")
    return Image.fromarray(arr).convert("RGB")


def load_image_from_bytes(raw: bytes, filename: str = "") -> Image.Image:
    """Decode an uploaded file (standard image or DICOM) to an RGB PIL image."""
    if _looks_like_dicom(raw, filename):
        return _load_dicom(raw)
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("Could not decode image.") from exc


def load_image_path(path: str) -> Image.Image:
    """Load an image or DICOM from a filesystem path."""
    with open(path, "rb") as f:
        return load_image_from_bytes(f.read(), filename=path)


@dataclass
class InputCheck:
    is_xray_like: bool
    reason: str
    saturation: float

    def to_dict(self) -> dict:
        return {
            "is_xray_like": self.is_xray_like,
            "reason": self.reason,
            "saturation": round(self.saturation, 4),
        }


def check_input(image: Image.Image, sat_threshold: float = 0.12) -> InputCheck:
    """Heuristic OOD check: chest X-rays are grayscale and reasonably sized."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    if rgb.shape[0] < 64 or rgb.shape[1] < 64:
        return InputCheck(False, "Image is too small to be a diagnostic X-ray.", 0.0)

    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1.0, None), 0.0)
    mean_sat = float(sat.mean())

    if mean_sat > sat_threshold:
        return InputCheck(
            False,
            "Image appears to be in color; chest X-rays are grayscale. "
            "Result may be unreliable.",
            mean_sat,
        )
    return InputCheck(True, "", mean_sat)
