"""Model definition (architecture unchanged from the original project) and safe checkpoint loading."""
from __future__ import annotations

import hashlib
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from .config import DECISION_THRESHOLD, INPUT_SIZE
from .schemas import ModelInfo


class SRMLayer(nn.Module):
    """Fixed (non-trainable) high-pass residual filters. Kernels are those of the original project."""

    def __init__(self):
        super().__init__()
        k = np.array([
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        ], dtype=np.float32) / 4.0
        self.kernel = nn.Parameter(torch.tensor(k).view(3, 1, 3, 3).repeat(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1)


class TwoStreamNet(nn.Module):
    """RGB stream + SRM-noise stream (two EfficientNet-B0), concatenated 2x1280 -> 512 -> 1 -> sigmoid.
    Attribute names match the original checkpoints, so they load with strict=True."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        make = EfficientNet.from_pretrained if pretrained else EfficientNet.from_name
        self.spatial_net = make("efficientnet-b0")
        self.srm_layer = SRMLayer()
        self.noise_net = make("efficientnet-b0")
        self.fc = nn.Sequential(nn.Linear(1280 * 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1), nn.Sigmoid())

    def logits(self, x):
        """Pre-sigmoid score. forward(x) == sigmoid(logits(x)); used for numerically stable training."""
        f1 = F.adaptive_avg_pool2d(self.spatial_net.extract_features(x), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.noise_net.extract_features(self.srm_layer(x)), 1).flatten(1)
        return self.fc[:-1](torch.cat((f1, f2), dim=1))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def resolve_device(device: Optional[str] = "auto") -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; using CPU.", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")
    return dev


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ModelBundle:
    model: nn.Module
    device: torch.device
    info: ModelInfo
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """(N,3,224,224) normalised tensor -> (N,) p(fake). Thread-safe."""
        with self.lock:
            self.model.eval()
            return self.model(batch.to(self.device)).detach().cpu().numpy().reshape(-1)


def load_checkpoint_state(path: Union[str, Path], device: torch.device) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Accepts the original raw state_dict files and the new {"state_dict":..., "meta":...} files."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        obj = torch.load(p, map_location=device, weights_only=True)
    except TypeError:
        warnings.warn("torch too old for weights_only; loading without it.", RuntimeWarning, stacklevel=2)
        obj = torch.load(p, map_location=device)
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        meta = dict(obj.get("meta") or {})
        obj = obj["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}
    return state, meta


def load_model(path: Union[str, Path], device: Optional[str] = "auto") -> ModelBundle:
    dev = resolve_device(device)
    state, meta = load_checkpoint_state(path, dev)
    model = TwoStreamNet(pretrained=False)
    model.load_state_dict(state, strict=True)          # any mismatch raises
    model.to(dev).eval()
    p = Path(path)
    legacy = not meta
    info = ModelInfo(
        name=meta.get("name", "verifai-two-stream-original" if legacy else "verifai-two-stream"),
        version=str(meta.get("version", "1.0-original" if legacy else "unknown")),
        checkpoint_file=p.name,
        checkpoint_sha256=sha256_file(p),
        parameters=int(sum(v.numel() for v in state.values() if v.dtype.is_floating_point)),
        device=str(dev),
        input_size=INPUT_SIZE,
        decision_threshold=DECISION_THRESHOLD,
        trained_on=meta.get("trained_on", "Celeb-DF face crops (original prototype; training data not in repository)" if legacy else None),
        training_notes=meta.get("training_notes", "Original prototype checkpoint: frame-level random split, no augmentation." if legacy else None),
        metrics_note=meta.get("metrics_note"),
    )
    return ModelBundle(model=model, device=dev, info=info)
