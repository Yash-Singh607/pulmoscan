"""Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet."""

from __future__ import annotations

import torch


class GradCAM:
    """Produce class-discriminative saliency maps for a ResNet model.

    Hooks are registered on the final convolutional block (``layer4[-1]``)
    and removed via :meth:`remove` to avoid leaking handles across calls.
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.layer4[-1]

        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(target_layer.register_forward_hook(forward_hook))
        self._handles.append(target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def generate(self, input_tensor, class_idx: int | None = None):
        """Return ``(cam, class_idx, probabilities)`` for one input."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs
