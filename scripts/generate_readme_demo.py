"""Generate README demo GIF for PulmoScan (no ffmpeg required).

Usage:
    python scripts/generate_readme_demo.py
    python scripts/generate_readme_demo.py --out docs/demo.gif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "demo.gif"

W, H = 960, 540
BG = (6, 8, 15)
PANEL = (14, 18, 28)
BORDER = (40, 48, 68)
BRAND = (59, 130, 246)
BRAND2 = (139, 92, 246)
TEXT = (241, 245, 249)
DIM = (148, 163, 184)
LUNG = (226, 232, 240)
HEAT = (239, 68, 68)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1)


def _draw_xray(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float = 1.0) -> None:
    s = scale
    draw.line([(cx, cy - 120 * s), (cx, cy + 130 * s)], fill=(100, 116, 139), width=int(3 * s))
    left = [
        (cx - 70 * s, cy - 90 * s),
        (cx - 95 * s, cy),
        (cx - 75 * s, cy + 95 * s),
        (cx - 10 * s, cy + 110 * s),
        (cx - 5 * s, cy - 90 * s),
    ]
    right = [(cx + (x - cx), y) for x, y in left]
    draw.polygon(left, fill=LUNG)
    draw.polygon(right, fill=LUNG)
    draw.ellipse(
        [cx + 5 * s, cy + 10 * s, cx + 55 * s, cy + 95 * s],
        fill=(120, 130, 145),
    )
    draw.polygon(
        [
            (cx + 35 * s, cy + 20 * s),
            (cx + 85 * s, cy + 45 * s),
            (cx + 70 * s, cy + 105 * s),
            (cx + 25 * s, cy + 85 * s),
        ],
        fill=(100, 116, 139),
    )


def _frame_shell(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    _rounded_rect(draw, (24, 24, W - 24, H - 24), 18, PANEL, BORDER)
    draw.text((48, 40), "PulmoScan", font=_font(22, True), fill=TEXT)
    draw.text((48, 72), title, font=_font(28, True), fill=TEXT)
    draw.text((48, 112), subtitle, font=_font(15), fill=DIM)
    return img, draw


def frame_upload() -> Image.Image:
    img, draw = _frame_shell("Upload chest X-ray", "Drag & drop PNG, JPG, or DICOM")
    box = (180, 160, W - 180, H - 90)
    _rounded_rect(draw, box, 14, (12, 18, 34), BRAND)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    draw.line([(cx, cy - 18), (cx, cy + 18)], fill=BRAND, width=3)
    draw.line([(cx - 18, cy), (cx + 18, cy)], fill=BRAND, width=3)
    draw.text((cx - 55, cy + 36), "Drop image here", font=_font(14), fill=DIM)
    draw.text((48, H - 58), "Step 1 · Upload", font=_font(13, True), fill=BRAND)
    return img


def frame_analyze(progress: float) -> Image.Image:
    img, draw = _frame_shell("AI inference", "ResNet-50 forward pass in milliseconds")
    cx, cy = W // 2, H // 2 + 10
    _draw_xray(draw, cx, cy, 1.15)
    y = int(H * 0.28 + (H * 0.45 - H * 0.28) * progress)
    draw.line([(120, y), (W - 120, y)], fill=BRAND, width=3)
    draw.text((48, H - 58), "Step 2 · Analyze", font=_font(13, True), fill=BRAND2)
    pct = int(progress * 100)
    draw.text((W - 130, H - 58), f"{pct}%", font=_font(13, True), fill=TEXT)
    return img


def frame_gradcam(intensity: float) -> Image.Image:
    img, draw = _frame_shell("Grad-CAM explainability", "Regions that drove the prediction")
    cx, cy = W // 2, H // 2 + 10
    _draw_xray(draw, cx, cy, 1.15)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    r = int(90 + 25 * intensity)
    odraw.ellipse([cx + 20 - r, cy + 10 - r, cx + 20 + r, cy + 10 + r], fill=(*HEAT, int(120 * intensity)))
    odraw.ellipse([cx + 35 - r // 2, cy + 25 - r // 2, cx + 35 + r // 2, cy + 25 + r // 2], fill=(245, 158, 11, int(80 * intensity)))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((48, H - 58), "Step 3 · Explain", font=_font(13, True), fill=(251, 113, 133))
    return img


def frame_result() -> Image.Image:
    img, draw = _frame_shell("Clinical-style report", "Confidence gauge + export-ready output")
    cx, cy = W // 2 - 80, H // 2 + 10
    _draw_xray(draw, cx, cy, 0.95)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.ellipse([cx + 10, cy + 5, cx + 90, cy + 85], fill=(*HEAT, 110))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    card = (W // 2 + 20, 190, W - 70, H - 100)
    _rounded_rect(draw, card, 12, (12, 18, 34), BORDER)
    draw.text((card[0] + 20, card[1] + 18), "Verdict", font=_font(12), fill=DIM)
    draw.text((card[0] + 20, card[1] + 42), "PNEUMONIA", font=_font(22, True), fill=(251, 113, 133))
    draw.text((card[0] + 20, card[1] + 78), "Confidence  94%", font=_font(14), fill=TEXT)
    bar = (card[0] + 20, card[1] + 108, card[2] - 20, card[1] + 120)
    _rounded_rect(draw, bar, 6, (30, 41, 59))
    fill_bar = (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * 0.94), bar[3])
    _rounded_rect(draw, fill_bar, 6, BRAND)
    draw.text((card[0] + 20, card[1] + 138), "PDF · FHIR · Review queue", font=_font(11), fill=DIM)
    return img


def build_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    frames += [frame_upload()] * 12
    for i in range(14):
        frames.append(frame_analyze(i / 13))
    for i in range(10):
        t = 0.5 + 0.5 * math.sin(i / 9 * math.pi)
        frames.append(frame_gradcam(t))
    frames += [frame_result()] * 16
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PulmoScan README demo GIF")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    poster = frames[-1].copy()
    poster_path = args.out.with_name("demo-poster.png")
    poster.save(poster_path, optimize=True)
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    print(f"Wrote {poster_path}")


if __name__ == "__main__":
    main()
