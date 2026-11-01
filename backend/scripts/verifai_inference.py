"""Verif-AI inference core (Phase 1 stabilisation).

Single source of truth for the model definition and single-frame inference that previously lived
(duplicated) in cell 8 of ``preprocessing.ipynb``.

What this module is NOT: it is not an image pipeline or a video pipeline. ``run_forensic_analysis``
keeps its original behaviour of analysing only the FIRST frame of whatever path it is given.

Architecture (unchanged from the notebook): two EfficientNet-B0 streams (RGB and a fixed 3-kernel
SRM-style residual), concatenated 2x1280 -> 512 -> 1 with a sigmoid. Attribute names are identical so
the existing checkpoints load with ``strict=True``.
"""
from __future__ import annotations

import io
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

# --------------------------------------------------------------------------------------------
# Constants (all values are those used by the original notebook; none were tuned in Phase 1)
# --------------------------------------------------------------------------------------------
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DECISION_THRESHOLD = 0.5          # unchanged: label is DEEPFAKE iff p(fake) > 0.5
LABEL_FAKE = "DEEPFAKE"
LABEL_REAL = "REAL"
# Training crops were written by facenet-pytorch's ``save_img`` which, for the numpy frames passed in
# notebook cell 0, calls ``cv2.imwrite(path, BGR)`` -> OpenCV's default JPEG quality (95), and were then
# read back with PIL. Verified bit-exact (see tests/test_preprocessing.py and CHANGELOG).
TRAINING_JPEG_QUALITY = 95


# --------------------------------------------------------------------------------------------
# Model (architecture unchanged)
# --------------------------------------------------------------------------------------------
class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        # 3 standard SRM filters to capture noise residuals
        srm_kernel = np.array([
            [[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]],
            [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]],
            [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]]
        ], dtype=np.float32) / 4.0

        self.kernel = torch.tensor(srm_kernel).view(3, 1, 3, 3).repeat(1, 3, 1, 1)
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1)


class TwoStreamNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        """``pretrained=True`` reproduces the notebook (downloads/uses ImageNet weights).
        ``pretrained=False`` builds the identical architecture without any download; use it whenever a
        checkpoint will be loaded afterwards (the checkpoint overwrites every backbone weight)."""
        super(TwoStreamNet, self).__init__()
        make = EfficientNet.from_pretrained if pretrained else EfficientNet.from_name
        # Stream 1: Spatial (Standard RGB)
        self.spatial_net = make('efficientnet-b0')

        # Stream 2: Noise (SRM)
        self.srm_layer = SRMLayer()
        self.noise_net = make('efficientnet-b0')

        # Fusion Classifier
        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # RGB Stream Feature Extraction
        feat1 = self.spatial_net.extract_features(x)
        feat1 = F.adaptive_avg_pool2d(feat1, 1).view(feat1.size(0), -1)

        # Noise Stream Feature Extraction (On-the-fly SRM)
        noise = self.srm_layer(x)
        feat2 = self.noise_net.extract_features(noise)
        feat2 = F.adaptive_avg_pool2d(feat2, 1).view(feat2.size(0), -1)

        # Concatenation and Final Prediction
        combined = torch.cat((feat1, feat2), dim=1)
        return self.fc(combined)


# --------------------------------------------------------------------------------------------
# Device handling and checkpoint loading
# --------------------------------------------------------------------------------------------
def resolve_device(device: Optional[Any] = None) -> torch.device:
    """None/'auto' -> CUDA if available else CPU. An explicit CUDA request on a machine without CUDA
    falls back to CPU with a warning instead of crashing."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU.", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")
    return dev


def load_checkpoint(model: nn.Module, path: str, device: Optional[Any] = None) -> nn.Module:
    """Load a state_dict checkpoint safely onto ``device`` (CPU or GPU).

    * ``map_location=device``  -> checkpoints saved from CUDA tensors load on CPU-only machines
      (this was the failure in the original ``initialize_engine``).
    * ``weights_only=True``    -> no arbitrary pickle execution. If the installed torch is too old to
      know the argument (TypeError) it falls back to a plain load with a warning.
    * ``strict=True``          -> any missing/unexpected key raises (the notebook's cell 9 used
      ``strict=False`` which can silently hide mismatches).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    dev = resolve_device(device)
    try:
        state = torch.load(path, map_location=dev, weights_only=True)
    except TypeError:
        warnings.warn("This torch version does not support weights_only; loading without it. "
                      "Only load checkpoints you trust.", RuntimeWarning, stacklevel=2)
        state = torch.load(path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # tolerate checkpoints saved from nn.DataParallel
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(dev)


def initialize_engine(model_path: Optional[str] = None, device: Optional[Any] = None
                      ) -> Tuple[nn.Module, MTCNN, torch.device]:
    """Build (model, mtcnn, device). Signature-compatible with the notebook version.

    Behaviour changes vs the original (documented in CHANGELOG):
      * loads on CPU or GPU (``device``; default auto);
      * a ``model_path`` that does not exist now raises FileNotFoundError instead of silently running
        an untrained head;
      * when a checkpoint is supplied the backbones are built without downloading ImageNet weights.
    """
    dev = resolve_device(device)
    mtcnn = MTCNN(image_size=INPUT_SIZE, margin=20, device=dev, post_process=False)
    if model_path:
        model = TwoStreamNet(pretrained=False).to(dev)
        load_checkpoint(model, model_path, dev)
        print(f"Loaded weights from {model_path} onto {dev}")
    else:
        warnings.warn("No checkpoint given: model has ImageNet backbones and an UNTRAINED head.",
                      RuntimeWarning, stacklevel=2)
        model = TwoStreamNet(pretrained=True).to(dev)
    model.eval()
    return model, mtcnn, dev


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------
_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def training_jpeg_roundtrip(face_0_255: torch.Tensor, quality: int = TRAINING_JPEG_QUALITY) -> torch.Tensor:
    """Reproduce how the TRAINING crops were stored/read: uint8 RGB crop -> OpenCV JPEG (BGR, q95) ->
    PIL decode. Input/output: float tensor (3,H,W) with values 0..255."""
    arr = face_0_255.detach().cpu().float().clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    img = Image.open(io.BytesIO(buf.tobytes())).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()


def preprocess_face(face_0_255: torch.Tensor, match_training_jpeg: bool = True) -> torch.Tensor:
    """(3,H,W) face crop with values 0..255 (MTCNN ``post_process=False`` output) -> (1,3,224,224)
    ImageNet-normalised batch.  ``match_training_jpeg=False`` reproduces the legacy inference path
    exactly (no JPEG step)."""
    x = face_0_255.detach().cpu().float()
    if match_training_jpeg:
        x = training_jpeg_roundtrip(x)
    x = x / 255.0
    if tuple(x.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE):
        x = F.interpolate(x.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear",
                          align_corners=False).squeeze(0)
    return _NORMALIZE(x).unsqueeze(0)


def _detect_largest_face(mtcnn: MTCNN, rgb: np.ndarray
                         ) -> Tuple[Optional[torch.Tensor], Optional[List[float]], Optional[float]]:
    """Same detection/selection/crop as ``mtcnn(rgb)`` (verified tensor-identical) but also returns the
    selected bounding box (x1,y1,x2,y2 in frame pixels) and its detector confidence."""
    boxes, probs, points = mtcnn.detect(rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None, None, None
    sel_boxes, sel_probs, _ = mtcnn.select_boxes(boxes, probs, points, rgb, method=mtcnn.selection_method)
    face = mtcnn.extract(rgb, sel_boxes, None)
    if face is None:
        return None, None, None
    box = [float(v) for v in np.asarray(sel_boxes).reshape(-1, 4)[0]]
    conf = float(np.asarray(sel_probs).reshape(-1)[0])
    return face, box, conf


# --------------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------------
def _result(status: str, **kw) -> Dict[str, Any]:
    base = dict(status=status, label=None, fake_probability=None, real_probability=None,
                face_box=None, face_confidence=None, threshold=DECISION_THRESHOLD, message=None)
    base.update(kw)
    return base


def _score(model: nn.Module, face_0_255: torch.Tensor, dev: torch.device, match_training_jpeg: bool,
           threshold: float) -> Tuple[Dict[str, Any], torch.Tensor]:
    batch = preprocess_face(face_0_255, match_training_jpeg).to(dev)
    model.eval()
    with torch.no_grad():
        p_fake = float(model(batch).item())
    res = _result("ok",
                  label=LABEL_FAKE if p_fake > threshold else LABEL_REAL,
                  fake_probability=p_fake, real_probability=1.0 - p_fake, threshold=threshold,
                  preprocessing=dict(match_training_jpeg=match_training_jpeg,
                                     jpeg_quality=TRAINING_JPEG_QUALITY if match_training_jpeg else None),
                  device=str(dev))
    return res, batch


def predict_face(model: nn.Module, face_0_255: torch.Tensor, device: Optional[Any] = None,
                 match_training_jpeg: bool = True, threshold: float = DECISION_THRESHOLD) -> Dict[str, Any]:
    """Score ONE face crop ((3,H,W), values 0..255). Returns the structured result (no box info)."""
    dev = next(model.parameters()).device if device is None else resolve_device(device)
    return _score(model, face_0_255, dev, match_training_jpeg, threshold)[0]


def analyze_rgb_frame(model: nn.Module, mtcnn: MTCNN, rgb: np.ndarray, device: Optional[Any] = None,
                      match_training_jpeg: bool = True, _internals: bool = False) -> Dict[str, Any]:
    """Detect the largest face in ONE RGB frame (H,W,3 uint8) and score it. Building block used by
    ``run_forensic_analysis``; it is not a public image/video pipeline."""
    face, box, conf = _detect_largest_face(mtcnn, rgb)
    if face is None:
        return _result("no_face", message="No face detected.")
    dev = next(model.parameters()).device if device is None else resolve_device(device)
    res, batch = _score(model, face, dev, match_training_jpeg, DECISION_THRESHOLD)
    res.update(face_box=box, face_confidence=conf)
    if _internals:
        res.update(input_batch=batch, face_crop=face)
    return res


def _say(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:       # e.g. cp1252 Windows consoles cannot print emoji
        print(text.encode("ascii", "replace").decode("ascii"))


def run_forensic_analysis(video_path: str, model: nn.Module, mtcnn: MTCNN, device: Optional[Any] = None,
                          visualize: bool = True, verbose: bool = True,
                          match_training_jpeg: bool = True, save_figure_path: Optional[str] = None
                          ) -> Dict[str, Any]:
    """Original behaviour retained: analyse the FIRST frame of ``video_path`` (no sampling, no
    aggregation), print the score, optionally show a Grad-CAM figure.  NEW: returns a structured dict
    ``{status, label, fake_probability, real_probability, face_box, face_confidence, threshold, ...}``
    instead of ``None``; failures are reported via ``status`` ('read_error' / 'no_face') and ``message``.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        if verbose:
            _say("Error: Could not read video.")
        return _result("read_error", message="Could not read video.", frame_index=0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = analyze_rgb_frame(model, mtcnn, rgb, device, match_training_jpeg, _internals=True)
    res["frame_index"] = 0
    if res["status"] != "ok":
        if verbose:
            _say("Error: No face detected.")
        return res

    if verbose:
        _say(f"DEBUG - Raw Output: {res['fake_probability']:.6f}")
        _say(f"Final Score for this frame: {res['fake_probability']:.4f}")

    if visualize:
        try:
            _visualize(model, res, save_figure_path)
        except Exception as exc:     # explanation must never break the prediction
            res["visualization_error"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                _say(f"Warning: Grad-CAM visualisation skipped ({res['visualization_error']})")

    res.pop("input_batch", None)
    res.pop("face_crop", None)
    return res


def _visualize(model: nn.Module, res: Dict[str, Any], save_figure_path: Optional[str]) -> None:
    """Grad-CAM on the RGB stream's last block (unchanged behaviour). Imports are lazy so that
    prediction does not require matplotlib / pytorch_grad_cam."""
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    batch = res["input_batch"]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(batch.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(batch.device)
    vis_img = np.clip((batch * std + mean)[0].permute(1, 2, 0).cpu().numpy(), 0, 1)   # pixels the model saw

    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=[model.spatial_net._blocks[-1]])
        grayscale_cam = cam(input_tensor=batch, targets=[ClassifierOutputTarget(0)])[0, :]
    heatmap_img = show_cam_on_image(vis_img.astype(np.float32), grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img)
    plt.title(f"Result: {res['label']}\nProb: {res['fake_probability']:.4f}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_img)
    plt.title("Artifact Analysis")
    plt.axis("off")
    if save_figure_path:
        plt.savefig(save_figure_path, dpi=110, bbox_inches="tight")
    plt.show()
