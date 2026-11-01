"""Grad-CAM explanations (RGB stream, last MBConv block, target = fake score).

Interpretation limits (also repeated in every report): the map shows which regions of the RGB stream's
last feature map increased the *fake score*; it is an attention-style visualisation, not a segmentation of
manipulated pixels, and it says nothing about the noise-residual stream.
"""
from __future__ import annotations

import io
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from .model import ModelBundle
from .preprocess import denormalise_to_uint8

CAM_NOTE = ("Grad-CAM highlights regions that contributed to the RGB stream's fake score. It is not proof "
            "that those pixels were manipulated, and it does not explain the noise-residual stream.")


class Explainer:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.available = True
        try:
            import pytorch_grad_cam  # noqa: F401
        except Exception:
            self.available = False

    def heatmaps(self, batch: torch.Tensor) -> Optional[Sequence[np.ndarray]]:
        """(N,3,224,224) normalised batch -> list of (224,224,3) uint8 overlays, or None if unavailable."""
        if not self.available or batch.shape[0] == 0:
            return None
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        base = denormalise_to_uint8(batch).astype(np.float32) / 255.0
        out = []
        with self.bundle.lock:
            m = self.bundle.model
            m.eval()
            with torch.enable_grad():
                cam = GradCAM(model=m, target_layers=[m.spatial_net._blocks[-1]])
                try:
                    gray = cam(input_tensor=batch.to(self.bundle.device), targets=[ClassifierOutputTarget(0)] * batch.shape[0])
                finally:
                    try:
                        cam.activations_and_grads.release()
                    except Exception:
                        pass
        for i in range(batch.shape[0]):
            out.append(show_cam_on_image(base[i], gray[i], use_rgb=True))
        return out


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
