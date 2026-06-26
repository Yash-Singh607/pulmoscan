"""Generate README demo GIF matching the PulmoScan web UI (Pillow only).

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
APP_BG = (6, 8, 15)
PANEL = (14, 18, 28)
PANEL2 = (18, 24, 38)
SURFACE2 = (17, 24, 39)
BORDER = (255, 255, 255)  # drawn at low opacity via separate color
BORDER_RGB = (35, 42, 58)
BRAND = (59, 130, 246)
VIOLET = (139, 92, 246)
TEXT = (241, 245, 249)
DIM = (148, 163, 184)
FAINT = (100, 116, 139)
ROSE = (251, 113, 133)
WARN = (245, 158, 11)

NAV = ["Analyzer", "How it works", "Features", "Performance"]


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


def _site_background() -> Image.Image:
    img = Image.new("RGB", (W, H), APP_BG)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    g.ellipse([W // 2 - 520, -180, W // 2 + 520, 320], fill=(59, 130, 246, 38))
    g.ellipse([W - 200, 80, W + 180, 520], fill=(139, 92, 246, 22))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid)
    for x in range(0, W, 48):
        gd.line([(x, 0), (x, H)], fill=(255, 255, 255, 8))
    for y in range(0, H, 48):
        gd.line([(0, y), (W, y)], fill=(255, 255, 255, 8))
    return Image.alpha_composite(img.convert("RGBA"), grid).convert("RGB")


def _make_xray_asset(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (6, 8, 15))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2 - 8
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
    draw.polygon([(cx + 38, cy + 15), (cx + 88, cy + 38), (cx + 78, cy + 98), (cx + 28, cy + 82)], fill=(90, 100, 115))
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


def _draw_compliance_bar(draw: ImageDraw.ImageDraw) -> int:
    draw.rectangle([0, 0, W, 30], fill=(20, 16, 8))
    draw.rectangle([0, 29, W, 30], fill=(245, 158, 11, 40))
    draw.rounded_rectangle([14, 7, 28, 21], radius=4, fill=WARN)
    draw.text((16, 8), "!", font=_font(9, True), fill=(255, 255, 255))
    msg = "Research & education only. Not for clinical diagnosis — require qualified physician review."
    draw.text((36, 8), msg, font=_font(10), fill=DIM)
    return 30


def _draw_topbar(draw: ImageDraw.ImageDraw, y0: int, active: str = "Analyzer") -> int:
    bar_h = 58
    draw.rectangle([0, y0, W, y0 + bar_h], fill=(6, 8, 15))
    draw.line([(0, y0 + bar_h - 1), (W, y0 + bar_h - 1)], fill=BORDER_RGB, width=1)
    _rounded_rect(draw, (24, y0 + 10, 58, y0 + 44), 11, BRAND)
    draw.text((34, y0 + 18), "P", font=_font(16, True), fill=(255, 255, 255))
    draw.text((68, y0 + 12), "PulmoScan", font=_font(16, True), fill=TEXT)
    draw.text((68, y0 + 32), "AI CHEST X-RAY ANALYSIS", font=_font(8, True), fill=FAINT)
    nx = 250
    _rounded_rect(draw, (nx, y0 + 12, nx + 380, y0 + 46), 10, (255, 255, 255, 4))
    for label in NAV:
        pill = label == active
        px = nx + 8 + NAV.index(label) * 92
        if pill:
            _rounded_rect(draw, (px, y0 + 16, px + 86, y0 + 42), 7, PANEL)
        draw.text((px + 10, y0 + 22), label, font=_font(10, pill), fill=TEXT if pill else DIM)
    _rounded_rect(draw, (W - 168, y0 + 16, W - 24, y0 + 42), 999, (20, 28, 22), (34, 197, 94))
    draw.ellipse([W - 158, y0 + 26, W - 150, y0 + 34], fill=(34, 197, 94))
    draw.text((W - 142, y0 + 22), "Model ready", font=_font(10, True), fill=(134, 239, 172))
    return y0 + bar_h


def _paste_xray(canvas: Image.Image, box: tuple[int, int, int, int], xray: Image.Image) -> None:
    thumb = xray.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas.paste(thumb, (box[0], box[1]))


def _paste_heat(canvas: Image.Image, box: tuple[int, int, int, int], heatmap: Image.Image, opacity: float) -> Image.Image:
    heat = heatmap.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    heat_a = heat.split()[3].point(lambda p: int(p * opacity))
    heat.putalpha(heat_a)
    layer.paste(heat, (box[0], box[1]), heat)
    return Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")


def _draw_showcase(
    canvas: Image.Image,
    xray: Image.Image,
    heatmap: Image.Image,
    heat_opacity: float,
) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    frame = (680, 118, 1220, 620)
    _rounded_rect(draw, frame, 18, PANEL, BORDER_RGB)
    tabs = (frame[0], frame[1], frame[2], frame[1] + 36)
    draw.rectangle(tabs, fill=SURFACE2)
    draw.line([(frame[0], tabs[3]), (frame[2], tabs[3])], fill=BORDER_RGB)
    draw.text((frame[0] + 24, frame[1] + 11), "Pneumonia case", font=_font(11, True), fill=TEXT)
    draw.text((frame[0] + 160, frame[1] + 11), "Normal case", font=_font(11), fill=FAINT)
    draw.line([(frame[0] + 24, tabs[3] - 2), (frame[0] + 130, tabs[3] - 2)], fill=BRAND, width=2)
    view = (frame[0], tabs[3], frame[2], frame[3] - 36)
    draw.rectangle(view, fill=(3, 5, 10))
    _paste_xray(canvas, view, xray)
    if heat_opacity > 0:
        canvas = _paste_heat(canvas, view, heatmap, heat_opacity)
        draw = ImageDraw.Draw(canvas)
    badge = (frame[2] - 148, frame[1] + 48, frame[2] - 18, frame[1] + 88)
    _rounded_rect(draw, badge, 9, (48, 18, 28), (251, 113, 133))
    draw.text((badge[0] + 12, badge[1] + 8), "PNEUMONIA", font=_font(11, True), fill=ROSE)
    draw.text((badge[0] + 12, badge[1] + 26), "94% confidence", font=_font(9), fill=ROSE)
    cap = (frame[0], frame[3] - 36, frame[2], frame[3])
    draw.rectangle(cap, fill=PANEL)
    draw.text((frame[0] + 16, frame[3] - 26), "Grad-CAM highlights regions driving the model decision", font=_font(10), fill=FAINT)
    return canvas


def frame_hero(xray: Image.Image, heatmap: Image.Image, heat_opacity: float) -> Image.Image:
    canvas = _site_background()
    draw = ImageDraw.Draw(canvas)
    y = _draw_compliance_bar(draw)
    y = _draw_topbar(draw, y, active="Analyzer")
    _rounded_rect(draw, (48, y + 24, 340, y + 48), 999, PANEL, BORDER_RGB)
    draw.text((62, y + 32), "Clinical AI · ResNet-50 · Grad-CAM · Audit trail", font=_font(10, True), fill=DIM)
    draw.text((48, y + 68), "AI-powered pneumonia", font=_font(34, True), fill=TEXT)
    draw.text((48, y + 112), "screening", font=_font(34, True), fill=TEXT)
    draw.text((48, y + 156), "with full explainability", font=_font(34, True), fill=VIOLET)
    draw.text(
        (48, y + 210),
        "Upload a chest X-ray for instant inference, calibrated confidence,\nGrad-CAM heatmaps, and export-ready clinical reports.",
        font=_font(13),
        fill=DIM,
    )
    _rounded_rect(draw, (48, y + 278, 210, y + 318), 11, BRAND)
    draw.text((68, y + 290), "Start analysis", font=_font(12, True), fill=(255, 255, 255))
    _rounded_rect(draw, (222, y + 278, 360, y + 318), 11, PANEL, BORDER_RGB)
    draw.text((248, y + 290), "Watch demo", font=_font(12, True), fill=TEXT)
    for i, chip in enumerate(["Grad-CAM", "MC-Dropout", "DICOM", "PDF · FHIR"]):
        cx = 48 + i * 118
        _rounded_rect(draw, (cx, y + 334, cx + 108, y + 358), 999, PANEL, BORDER_RGB)
        draw.text((cx + 12, y + 342), chip, font=_font(9, True), fill=DIM)
    return _draw_showcase(canvas, xray, heatmap, heat_opacity)


def _analyzer_layout(draw: ImageDraw.ImageDraw, y0: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    left = (48, y0, 612, H - 36)
    right = (628, y0, 1232, H - 36)
    _rounded_rect(draw, left, 14, PANEL, BORDER_RGB)
    _rounded_rect(draw, right, 14, PANEL, BORDER_RGB)
    return left, right


def _panel_header(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], kicker: str, title: str) -> int:
    draw.text((box[0] + 20, box[1] + 18), kicker.upper(), font=_font(9, True), fill=BRAND)
    draw.text((box[0] + 20, box[1] + 36), title, font=_font(17, True), fill=TEXT)
    return box[1] + 72


def frame_analyzer(
    xray: Image.Image,
    heatmap: Image.Image,
    *,
    phase: str,
    scan_t: float = 0.0,
    heat_opacity: float = 0.0,
    progress: float = 0.0,
) -> Image.Image:
    canvas = _site_background()
    draw = ImageDraw.Draw(canvas)
    y = _draw_compliance_bar(draw)
    y = _draw_topbar(draw, y, active="Analyzer")
    left, right = _analyzer_layout(draw, y + 16)
    ly = _panel_header(draw, left, "Step 1", "Image intake")
    ry = _panel_header(draw, right, "Step 2", "Analysis report")

    dz = (left[0] + 20, ly + 6, left[2] - 20, left[3] - 96)
    inner = (dz[0] + 10, dz[1] + 10, dz[2] - 10, dz[3] - 10)

    if phase == "upload":
        _rounded_rect(draw, dz, 12, PANEL2, BORDER_RGB)
        cx, cy = (dz[0] + dz[2]) // 2, (dz[1] + dz[3]) // 2 - 10
        _rounded_rect(draw, (cx - 30, cy - 30, cx + 30, cy + 30), 30, (22, 30, 48), BRAND)
        draw.line([(cx, cy - 12), (cx, cy + 12)], fill=TEXT, width=2)
        draw.line([(cx - 12, cy - 6), (cx, cy - 16), (cx + 12, cy - 6)], fill=TEXT, width=2)
        draw.text((cx - 108, cy + 44), "Click to upload or drag & drop", font=_font(13, True), fill=TEXT)
        draw.text((cx - 88, cy + 66), "PNG, JPG or DICOM (.dcm)", font=_font(11), fill=FAINT)
    else:
        _rounded_rect(draw, dz, 12, PANEL2, BRAND, 2)
        _rounded_rect(draw, inner, 10, (3, 5, 10))
        _paste_xray(canvas, inner, xray)
        draw = ImageDraw.Draw(canvas)
        if phase == "analyze":
            sy = int(inner[1] + (inner[3] - inner[1]) * scan_t)
            draw.line([(inner[0], sy), (inner[2], sy)], fill=BRAND, width=3)
            bar = (inner[0] + 16, inner[3] - 24, inner[2] - 16, inner[3] - 12)
            _rounded_rect(draw, bar, 4, (30, 41, 59))
            _rounded_rect(draw, (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * progress), bar[3]), 4, BRAND)

    btn_y = left[3] - 52
    _rounded_rect(draw, (left[0] + 20, btn_y, left[0] + 96, btn_y + 34), 9, PANEL2, BORDER_RGB)
    draw.text((left[0] + 38, btn_y + 9), "Clear", font=_font(11, True), fill=DIM)
    _rounded_rect(draw, (left[2] - 112, btn_y, left[2] - 20, btn_y + 34), 9, BRAND if phase != "upload" else (40, 48, 68))
    draw.text((left[2] - 92, btn_y + 9), "Analyze", font=_font(11, True), fill=TEXT)

    rx = (right[0] + 20, ry + 8, right[2] - 20, right[3] - 20)
    if phase in ("upload", "analyze"):
        _rounded_rect(draw, rx, 12, PANEL2, BORDER_RGB)
        cx = (rx[0] + rx[2]) // 2
        cy = (rx[1] + rx[3]) // 2
        draw.text((cx - 108, cy - 20), "Awaiting analysis", font=_font(15, True), fill=TEXT)
        draw.text((cx - 132, cy + 8), "Upload an X-ray or pick a demo sample", font=_font(11), fill=FAINT)
        return canvas

    view = (rx[0] + 12, rx[1] + 12, rx[0] + int((rx[2] - rx[0]) * 0.52), rx[3] - 12)
    _rounded_rect(draw, view, 10, (3, 5, 10), BORDER_RGB)
    _paste_xray(canvas, view, xray)
    canvas = _paste_heat(canvas, view, heatmap, heat_opacity)
    draw = ImageDraw.Draw(canvas)
    card = (view[2] + 14, rx[1] + 12, rx[2] - 12, rx[3] - 12)
    _rounded_rect(draw, card, 12, PANEL2, BORDER_RGB)
    draw.text((card[0] + 16, card[1] + 16), "VERDICT", font=_font(9, True), fill=FAINT)
    draw.text((card[0] + 16, card[1] + 38), "PNEUMONIA", font=_font(24, True), fill=ROSE)
    draw.text((card[0] + 16, card[1] + 76), "Confidence", font=_font(11), fill=DIM)
    draw.text((card[2] - 52, card[1] + 72), "94%", font=_font(18, True), fill=TEXT)
    bar = (card[0] + 16, card[1] + 104, card[2] - 16, card[1] + 114)
    _rounded_rect(draw, bar, 5, (30, 41, 59))
    _rounded_rect(draw, (bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * 0.94), bar[3]), 5, BRAND)
    draw.text((card[0] + 16, card[1] + 126), "Export all · PDF · JSON · FHIR", font=_font(10), fill=FAINT)
    return canvas


def build_frames(xray: Image.Image, heatmap: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for i in range(10):
        t = i / 9
        frames.append(frame_hero(xray, heatmap, _lerp(0.35, 0.65, t)))
    for _ in range(6):
        frames.append(frame_analyzer(xray, heatmap, phase="upload"))
    for i in range(14):
        t = i / 13
        frames.append(frame_analyzer(xray, heatmap, phase="analyze", scan_t=t, progress=_lerp(0.1, 1.0, t)))
    for i in range(8):
        frames.append(frame_analyzer(xray, heatmap, phase="result", heat_opacity=_lerp(0.2, 0.75, i / 7)))
    for i in range(12):
        pulse = 0.75 + 0.07 * math.sin(i / 11 * math.pi)
        frames.append(frame_analyzer(xray, heatmap, phase="result", heat_opacity=pulse))
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
    frames[-1].save(args.out.with_name("demo-poster.png"), optimize=True)
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
