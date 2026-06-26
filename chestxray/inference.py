"""Inference: load a model and predict on images, with Grad-CAM overlays."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .calibration import apply_temperature
from .checkpoint import load_checkpoint  # noqa: E402
from .config import ModelConfig  # noqa: E402
from .data import inference_transform  # noqa: E402
from .gradcam import GradCAM  # noqa: E402
from .imaging import load_image_path  # noqa: E402
from .model import build_model  # noqa: E402
from .utils import get_device, get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class Uncertainty:
    entropy: float
    std: dict[str, float]
    abstain: bool

    def to_dict(self) -> dict:
        return {
            "entropy": round(self.entropy, 4),
            "std": {k: round(v, 4) for k, v in self.std.items()},
            "abstain": self.abstain,
        }


class Classifier:
    """Loaded model ready for repeated inference (used by CLI and API)."""

    def __init__(self, checkpoint_path: str, device: torch.device | str | None = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device)

        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        self.class_names = ckpt["class_names"]

        model_cfg = ModelConfig(**{
            k: v for k, v in ckpt["model_config"].items() if k in ModelConfig().__dict__
        })
        model_cfg.num_classes = len(self.class_names)

        self.model = build_model(model_cfg, pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        size = int(ckpt.get("image_size", 224))
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt.get("optimal_threshold", 0.5))
        self.positive_idx = self.class_names.index("PNEUMONIA") if "PNEUMONIA" in self.class_names else 1
        self.transform = inference_transform(size)
        logger.info(
            "Loaded model from %s (classes=%s, size=%d, T=%.3f, threshold=%.3f)",
            checkpoint_path,
            self.class_names,
            size,
            self.temperature,
            self.threshold,
        )

    def predict(self, image: "Image.Image | str", tta: bool = False) -> Prediction:
        """Predict a single PIL image or image path.

        With ``tta=True`` the prediction is averaged over the image and its
        horizontal mirror (test-time augmentation), which usually nudges
        accuracy up a fraction of a point at the cost of a second forward pass.
        """
        image_pil = self._to_pil(image)
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if tta:
                flipped = torch.flip(tensor, dims=[3])
                logits = (logits + self.model(flipped)) / 2
            probs = apply_temperature(logits, self.temperature)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        if self.threshold is not None and len(self.class_names) == 2:
            other = 0 if self.positive_idx == 1 else 1
            idx = self.positive_idx if probs[self.positive_idx] >= self.threshold else other
        return Prediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            probabilities={name: float(p) for name, p in zip(self.class_names, probs)},
        )

    def predict_with_gradcam(
        self, image: "Image.Image | str", save_path: str | None = None
    ) -> Prediction:
        """Predict and (optionally) write a Grad-CAM overlay figure."""
        image_pil = self._to_pil(image)
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        with GradCAM(self.model) as grad_cam:
            cam, idx, probs = grad_cam.generate(tensor)

        pred = Prediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            probabilities={name: float(p) for name, p in zip(self.class_names, probs)},
        )

        if save_path:
            self._save_overlay(image_pil, cam, pred, save_path)
        return pred

    def analyze(self, image: "Image.Image | str"):
        """Predict and return Grad-CAM visuals as in-memory PIL images.

        Returns ``(prediction, original, heatmap, overlay)`` where each image
        is a 224x224 RGB :class:`PIL.Image.Image`. Used by the web UI / API so
        nothing has to be written to disk.
        """
        image_pil = self._to_pil(image)
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        with GradCAM(self.model) as grad_cam:
            cam, idx, probs = grad_cam.generate(tensor)

        pred = Prediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            probabilities={name: float(p) for name, p in zip(self.class_names, probs)},
        )

        original = np.array(image_pil.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * original + 0.45 * heatmap).astype(np.uint8)

        return (
            pred,
            Image.fromarray(original),
            Image.fromarray(heatmap),
            Image.fromarray(overlay),
        )

    def estimate_uncertainty(
        self, image: "Image.Image | str", passes: int = 10, abstain_entropy: float = 0.5
    ) -> Uncertainty:
        """Monte-Carlo Dropout uncertainty over ``passes`` stochastic forwards.

        Keeps dropout active at inference and measures the spread of the
        resulting probability distributions. High predictive entropy or a
        low top probability triggers an abstention recommendation.
        """
        image_pil = self._to_pil(image)
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        self.model.eval()
        self._enable_dropout()
        samples = []
        with torch.no_grad():
            for _ in range(max(2, passes)):
                samples.append(torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy())
        self.model.eval()  # restore

        stacked = np.stack(samples)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        entropy = float(-(mean * np.log(mean + 1e-9)).sum())
        abstain = bool(entropy >= abstain_entropy or mean.max() < 0.6)

        return Uncertainty(
            entropy=entropy,
            std={name: float(s) for name, s in zip(self.class_names, std)},
            abstain=abstain,
        )

    def _enable_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    @staticmethod
    def _to_pil(image: "Image.Image | str") -> "Image.Image":
        if isinstance(image, str):
            return load_image_path(image)
        return image.convert("RGB")

    @staticmethod
    def _save_overlay(image_pil, cam, pred: Prediction, save_path: str) -> None:
        orig = np.array(image_pil.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * orig + 0.45 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f"Prediction: {pred.label}  ({pred.confidence * 100:.1f}% confidence)",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].imshow(orig, cmap="gray")
        axes[0].set_title("Original X-Ray")
        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved Grad-CAM overlay -> %s", save_path)
