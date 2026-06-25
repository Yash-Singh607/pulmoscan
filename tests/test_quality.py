"""Image quality assessment tests."""

import numpy as np
from PIL import Image

from chestxray.quality import assess_quality


def test_sharp_grayscale_acceptable():
    gray = np.random.randint(80, 180, (256, 256), dtype=np.uint8)
    img = Image.fromarray(gray, "L").convert("RGB")
    q = assess_quality(img)
    assert q.acceptable is True
    assert q.triage == "routine"


def test_blurry_image_flagged():
    img = Image.new("L", (256, 256), 128).convert("RGB")
    q = assess_quality(img, blur_min=200.0)
    assert q.triage in ("review", "reject")
    assert q.warnings
