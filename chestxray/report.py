"""Single-page PDF report generation (via matplotlib, no extra deps)."""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def build_pdf(
    *,
    original,
    heatmap,
    overlay,
    label: str,
    confidence: float,
    probabilities: dict,
    threshold: float,
    meta: dict,
) -> bytes:
    """Compose a one-page clinical-style PDF report and return its bytes.

    ``original``, ``heatmap``, ``overlay`` are PIL images.
    """
    buf = io.BytesIO()
    accent = "#e11d48" if label == "PNEUMONIA" else "#16a34a"

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("PulmoScan — Chest X-Ray Analysis Report", fontsize=16, fontweight="bold", y=0.97)

        fig.text(0.5, 0.935, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 ha="center", fontsize=9, color="#666")

        # Verdict band
        fig.text(0.08, 0.88, "Prediction", fontsize=10, color="#666")
        fig.text(0.08, 0.855, label, fontsize=22, fontweight="bold", color=accent)
        fig.text(0.92, 0.88, "Confidence", fontsize=10, color="#666", ha="right")
        fig.text(0.92, 0.855, f"{confidence * 100:.1f}%", fontsize=22, fontweight="bold", ha="right")

        # Images row
        for i, (img, title) in enumerate(
            [(original, "Original"), (heatmap, "Grad-CAM"), (overlay, "Overlay")]
        ):
            ax = fig.add_axes([0.08 + i * 0.30, 0.58, 0.26, 0.20])
            ax.imshow(np.asarray(img))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Probability table
        fig.text(0.08, 0.50, "Class probabilities", fontsize=11, fontweight="bold")
        y = 0.47
        for name, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
            fig.text(0.10, y, name, fontsize=10)
            fig.text(0.40, y, f"{value * 100:.2f}%", fontsize=10, family="monospace")
            ax = fig.add_axes([0.52, y - 0.002, 0.40, 0.014])
            ax.barh([0], [value], color=accent if name == label else "#9ca3af")
            ax.set_xlim(0, 1)
            ax.axis("off")
            y -= 0.045

        # Meta
        fig.text(0.08, 0.30, "Details", fontsize=11, fontweight="bold")
        lines = [
            f"File: {meta.get('file', '-')}",
            f"Decision threshold (PNEUMONIA): {threshold * 100:.0f}%",
            f"Model: {meta.get('architecture', 'ResNet-50')}   Device: {meta.get('device', '-')}",
        ]
        if meta.get("uncertainty"):
            u = meta["uncertainty"]
            lines.append(
                f"Predictive entropy: {u.get('entropy', 0):.3f}   "
                f"Abstain: {u.get('abstain', False)}"
            )
        if meta.get("input_check") and not meta["input_check"].get("is_xray_like", True):
            lines.append(f"⚠ Input warning: {meta['input_check'].get('reason', '')}")
        for i, ln in enumerate(lines):
            fig.text(0.10, 0.27 - i * 0.03, ln, fontsize=9.5)

        # Disclaimer
        fig.text(
            0.5, 0.06,
            "Research & education only. Not a medical device. "
            "Do not use for clinical diagnosis.\nAlways consult a qualified healthcare professional.",
            ha="center", fontsize=8.5, color="#b91c1c", wrap=True,
        )

        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()
