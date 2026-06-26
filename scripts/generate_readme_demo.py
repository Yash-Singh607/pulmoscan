"""Generate a professional README demo GIF for PulmoScan (Pillow only).

Usage:
    python scripts/generate_readme_demo.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEFAULT_OUT = ROOT / "docs" / "demo.gif"

W, H = 1280, 720
BG = (4, 6, 12)
WIN = (10, 14, 24)
PANEL = (14, 18, 30)
PANEL2 = (18, 24, 38)
BORDER = (45, 55, 78)
BRAND = (59, 130, 246)
BRAND2 = (99, 102, 241)
VIOLET = (139, 92, 246)
TEXT = (241, 245, 249)
DIM = (148, 163, 184)
FAINT = (100, 116, 139)
ROSE = (251, 113, 133)
GREEN = (134, 239, 172)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"),
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for bold_path, regular_path in names:
        path = bold_path if bold else regular_path
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, ...],
    outline: tuple[int, ...] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _gradient_bg() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        r = int(_lerp(4, 12, t))
        g = int(_lerp(6, 18, t))
        b = int(_lerp(18, 42, t))
        draw.line([(0, y), (W, y)], fill=(r, g, b))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse([W // 2 - 420, -120, W // 2 + 420, 380], fill=(59, 130, 246, 28))
    gdraw.ellipse([W // 2 - 300, H - 200, W // 2 + 300, H + 120], fill=(139, 92, 246, 22))
    return Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")


def _make_xray_asset(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (6, 8, 15))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2 - 8
    for i, y_off in enumerate([-95, -55, -15, 25, 65, 105]):
        draw.arc([cx - 130, cy + y_off - 20, cx + 130, cy + y_off + 40], 190, 350, fill=(148, 163, 184, 80), width=1)
    draw.line([(cx - 2, cy - 110), (cx - 2, cy + 120)], fill=(100, 116, 139), width=3)
    draw.line([(cx + 2, cy - 110), (cx + 2, cy + 120)], fill=(100, 116, 139), width=3)
    draw.polygon(
        [(cx - 75, cy - 85), (cx - 98, cy - 10), (cx - 88, cy + 70), (cx - 45, cy + 105), (cx - 8, cy + 95), (cx - 8, cy - 85)],
        fill=(220, 228, 236),
    )
    draw.polygon(
        [(cx + 75, cy - 85), (cx + 98, cy - 10), (cx + 88, cy + 70), (cx + 45, cy + 105), (cx + 8, cy + 95), (cx + 8, cy - 85)],
        fill=(220, 228, 236),
    )
    draw.ellipse([cx + 8, cy + 5, cx + 52, cy + 78], fill=(130, 140, 155))
    draw.polygon(
        [(cx + 38, cy + 15), (cx + 88, cy + 38), (cx + 78, cy + 98), (cx + 28, cy + 82)],
        fill=(90, 100, 115),
    )
    draw.arc([cx - 95, cy - 105, cx + 95, cy - 55], 200, 340, fill=(148, 163, 184), width=4)
    return img


def _make_heatmap_asset(size: tuple[int, int]) -> Image.Image:
    w, h = size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = int(w * 0.62), int(h * 0.58)
    for r, alpha, color in [(95, 95, (239, 68, 68)), (62, 120, (245, 158, 11)), (38, 160, (220, 38, 38))]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return overlay.filter(ImageFilter.GaussianBlur(radius=6))


def _browser_shell() -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    base = _gradient_bg()
    draw = ImageDraw.Draw(base)
    win = (80, 48, W - 80, H - 48)
    _rounded_rect(draw, win, 16, WIN, BORDER, 1)
    chrome = (win[0] + 1, win[1] + 1, win[2] - 1, win[1] + 46)
    _rounded_rect(draw, chrome, 15, (16, 20, 32))
    draw.rectangle([win[0] + 1, win[1] + 30, win[2] - 1, win[1] + 46], fill=(16, 20, 32))
    for i, c in enumerate([(239, 68, 68), (250, 204, 21), (34, 197, 94)]):
        draw.ellipse([win[0] + 18 + i * 22, win[1] + 14, win[0] + 30 + i * 22, win[1] + 26], fill=c)
    url_box = (win[0] + 92, win[1] + 10, win[2] - 18, win[1] + 30)
    _rounded_rect(draw, url_box, 8, (8, 12, 22), (35, 45, 68))
    draw.text((url_box[0] + 12, url_box[1] + 5), "127.0.0.1:8000  ·  PulmoScan", font=_font(11), fill=FAINT)

    nav_y = win[1] + 46
    draw.rectangle([win[0], nav_y, win[2], nav_y + 44], fill=(8, 12, 22))
    draw.text((win[0] + 22, nav_y + 12), "PulmoScan", font=_font(17, True), fill=TEXT)
    for i, label in enumerate(["Analyzer", "Showcase", "Metrics", "Docs"]):
        draw.text((win[0] + 140 + i * 92, nav_y + 14), label, font=_font(11), fill=FAINT if i else TEXT)

    content = (win[0] + 16, nav_y + 56, win[2] - 16, win[3] - 16)
    left = (content[0], content[1], content[0] + (content[2] - content[0]) // 2 - 8, content[3])
    right = (left[2] + 16, content[1], content[2], content[3])
    _rounded_rect(draw, left, 14, PANEL, BORDER)
    _rounded_rect(draw, right, 14, PANEL, BORDER)
    return base, {"left": left, "right": right, "win": win}


def _panel_header(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], kicker: str, title: str) -> int:
    x0, y0, _, _ = box
    draw.text((x0 + 18, y0 + 16), kicker.upper(), font=_font(10, True), fill=BRAND)
    draw.text((x0 + 18, y0 + 34), title, font=_font(18, True), fill=TEXT)
    return y0 + 68


def _draw_dropzone(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    filled: bool = False,
    xray: Image.Image | None = None,
    scan_y: float | None = None,
    progress: float | None = None,
) -> None:
    x0, y0, x2, y2 = box
    dz = (x0 + 18, y0, x2 - 18, y2 - 12)
    outline = BRAND if filled else BORDER
    _rounded_rect(draw, dz, 12, PANEL2, outline, 2 if filled else 1)
    if not filled:
        cx, cy = (dz[0] + dz[2]) // 2, (dz[1] + dz[3]) // 2 - 8
        _rounded_rect(draw, (cx - 28, cy - 28, cx + 28, cy + 28), 28, (22, 30, 48), BRAND)
        draw.line([(cx, cy - 10), (cx, cy + 10)], fill=TEXT, width=2)
        draw.line([(cx - 10, cy - 4), (cx, cy - 14), (cx + 10, cy - 4)], fill=TEXT, width=2)
        draw.text((cx - 95, cy + 42), "Click to upload or drag & drop", font=_font(13, True), fill=TEXT)
        draw.text((cx - 72, cy + 64), "PNG, JPG or DICOM (.dcm)", font=_font(11), fill=FAINT)
        return

    inner = (dz[0] + 8, dz[1] + 8, dz[2] - 8, dz[3] - 8)
    _rounded_rect(draw, inner, 10, (3, 5, 10))
    if xray:
        thumb = xray.copy().resize((inner[2] - inner[0], inner[3] - inner[1]), Image.Resampling.LANCZOS)
        base = Image.new("RGB", (W, H))
        base.paste(thumb, (inner[0], inner[1]))
        return base, inner

    if progress is not None:
        cx = inner[0] + 18
        cy = inner[3] - 28
        bar = (cx, cy, inner[2] - 18, cy + 8)
        _rounded_rect(draw, bar, 4, (30, 41, 59))
        fill = (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * progress), bar[3])
        _rounded_rect(draw, fill, 4, BRAND)


def _compose_analyzer_frame(
    xray: Image.Image,
    heatmap: Image.Image,
    *,
    phase: str,
    scan_t: float = 0.0,
    heat_opacity: float = 0.0,
    show_results: bool = False,
    progress: float = 0.0,
) -> Image.Image:
    canvas, regions = _browser_shell()
    left, right = regions["left"], regions["right"]
    draw = ImageDraw.Draw(canvas)

    ly = _panel_header(draw, left, "Step 1", "Image intake")
    rz = (left[0] + 18, ly + 4, left[2] - 18, left[3] - 88)
    ry = _panel_header(draw, right, "Step 2", "Analysis report")
    result_top = ry + 8

    if phase == "upload":
        _draw_dropzone(draw, rz, filled=False)
    else:
        inner = (rz[0] + 8, rz[1] + 8, rz[2] - 8, rz[3] - 8)
        _rounded_rect(draw, rz, 12, PANEL2, BRAND, 2)
        _rounded_rect(draw, inner, 10, (3, 5, 10))
        thumb = xray.resize((inner[2] - inner[0], inner[3] - inner[1]), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (inner[0], inner[1]))
        if phase == "analyze":
            y = int(inner[1] + (inner[3] - inner[1]) * scan_t)
            scan = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(scan)
            sdraw.line([(inner[0], y), (inner[2], y)], fill=(59, 130, 246, 220), width=3)
            sdraw.rectangle([inner[0], y - 18, inner[2], y], fill=(59, 130, 246, 35))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), scan).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            _draw_dropzone(draw, rz, filled=True, progress=progress)

    btn_y = left[3] - 62
    _rounded_rect(draw, (left[0] + 18, btn_y, left[0] + 100, btn_y + 34), 9, PANEL2, BORDER)
    draw.text((left[0] + 38, btn_y + 9), "Clear", font=_font(12, True), fill=DIM)
    analyze_fill = BRAND if phase != "upload" else (35, 45, 68)
    _rounded_rect(draw, (left[2] - 118, btn_y, left[2] - 18, btn_y + 34), 9, analyze_fill)
    draw.text((left[2] - 98, btn_y + 9), "Analyze", font=_font(12, True), fill=TEXT)

    rx = (right[0] + 18, result_top, right[2] - 18, right[3] - 18)
    if not show_results:
        _rounded_rect(draw, rx, 12, PANEL2, BORDER)
        cx, cy = (rx[0] + rx[2]) // 2, (rx[1] + rx[3]) // 2
        draw.text((cx - 120, cy - 28), "Awaiting analysis", font=_font(16, True), fill=TEXT)
        draw.text((cx - 145, cy + 2), "Upload an X-ray or pick a demo sample", font=_font(12), fill=FAINT)
        return canvas

    view = (rx[0] + 12, rx[1] + 12, rx[0] + (rx[2] - rx[0]) * 55 // 100, rx[3] - 12)
    _rounded_rect(draw, view, 10, (3, 5, 10), BORDER)
    vthumb = xray.resize((view[2] - view[0], view[3] - view[1]), Image.Resampling.LANCZOS)
    canvas.paste(vthumb, (view[0], view[1]))
    if heat_opacity > 0:
        heat = heatmap.resize((view[2] - view[0], view[3] - view[1]), Image.Resampling.LANCZOS)
        heat_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        heat_layer.paste(heat, (view[0], view[1]), heat)
        alpha = heat_layer.split()[3].point(lambda p: int(p * heat_opacity))
        heat_layer.putalpha(alpha)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), heat_layer).convert("RGB")
        draw = ImageDraw.Draw(canvas)

    card = (view[2] + 16, rx[1] + 12, rx[2] - 12, rx[3] - 12)
    _rounded_rect(draw, card, 12, PANEL2, BORDER)
    draw.text((card[0] + 16, card[1] + 16), "VERDICT", font=_font(10, True), fill=FAINT)
    draw.text((card[0] + 16, card[1] + 38), "PNEUMONIA", font=_font(26, True), fill=ROSE)
    draw.text((card[0] + 16, card[1] + 78), "Confidence", font=_font(11), fill=DIM)
    draw.text((card[2] - 58, card[1] + 76), "94%", font=_font(20, True), fill=TEXT)
    bar = (card[0] + 16, card[1] + 108, card[2] - 16, card[1] + 118)
    _rounded_rect(draw, bar, 5, (30, 41, 59))
    _rounded_rect(draw, (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * 0.94), bar[3]), 5, BRAND)
    draw.text((card[0] + 16, card[1] + 132), "Grad-CAM · PDF · FHIR export", font=_font(11), fill=FAINT)
    badge = (view[0] + 10, view[1] + 10, view[0] + 118, view[1] + 32)
    _rounded_rect(draw, badge, 6, (127, 29, 29, 180) if heat_opacity > 0.4 else (30, 41, 59))
    draw.text((badge[0] + 10, badge[1] + 6), "PNEUMONIA", font=_font(10, True), fill=ROSE if heat_opacity > 0.4 else DIM)

    step = {"upload": "1 / 4  Upload", "analyze": "2 / 4  Inference", "explain": "3 / 4  Grad-CAM", "result": "4 / 4  Report"}[phase]
    draw.text((regions["win"][0] + 22, regions["win"][3] - 28), step, font=_font(11, True), fill=BRAND2)
    return canvas


def build_frames(xray: Image.Image, heatmap: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for _ in range(10):
        frames.append(_compose_analyzer_frame(xray, heatmap, phase="upload"))
    frames.append(_compose_analyzer_frame(xray, heatmap, phase="analyze", scan_t=0.0, progress=0.05))
    for i in range(16):
        t = i / 15
        frames.append(_compose_analyzer_frame(xray, heatmap, phase="analyze", scan_t=t, progress=_lerp(0.08, 0.98, t)))
    for i in range(8):
        frames.append(_compose_analyzer_frame(xray, heatmap, phase="explain", show_results=True, heat_opacity=_lerp(0.15, 0.72, i / 7)))
    for i in range(14):
        pulse = 0.72 + 0.08 * math.sin(i / 13 * math.pi)
        frames.append(_compose_analyzer_frame(xray, heatmap, phase="result", show_results=True, heat_opacity=pulse))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PulmoScan README demo GIF")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    xray = _make_xray_asset((520, 640))
    heatmap = _make_heatmap_asset((520, 640))
    xray.save(ASSETS / "demo-xray.png", optimize=True)
    heatmap.save(ASSETS / "demo-heatmap.png")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = build_frames(xray, heatmap)
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
    poster.save(args.out.with_name("demo-poster.png"), optimize=True)
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB, {len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
