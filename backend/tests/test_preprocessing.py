"""Training/inference preprocessing parity.

Training path (notebook cell 0 + cell 3): MTCNN crop (uint8) -> cv2.imwrite JPEG (default q95, BGR)
-> PIL.Image.open().convert('RGB') -> Resize((224,224)) -> ToTensor -> Normalize(ImageNet).
"""
import os

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import verifai_inference as vi


def _synthetic_crop(seed=0):
    """224x224 RGB 'crop' with smooth structure plus noise so JPEG visibly changes it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224]
    base = np.stack([128 + 90 * np.sin(xx / 17.0), 128 + 90 * np.cos(yy / 23.0), 128 + 60 * np.sin((xx + yy) / 31.0)], -1)
    return np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)


def _training_loader_tensor(rgb_u8, tmp_path):
    """Exactly how a training sample reached the model."""
    path = str(tmp_path / "frame_0.jpg")
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))          # facenet-pytorch save_img for ndarray
    img = Image.open(path).convert("RGB")                                # CelebDFDataset.__getitem__
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return tf(img).unsqueeze(0)


def _legacy_inference_tensor(face_0_255):
    """Verbatim legacy inference preprocessing from notebook cell 8 (no JPEG step)."""
    face_scaled = face_0_255 / 255.0
    face_resized = F.interpolate(face_scaled.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face_resized)


def test_new_preprocessing_equals_training_loader_bit_for_bit(tmp_path):
    crop = _synthetic_crop()
    face = torch.from_numpy(crop).permute(2, 0, 1).float()              # what MTCNN(post_process=False) returns
    mine = vi.preprocess_face(face, match_training_jpeg=True)
    theirs = _training_loader_tensor(crop, tmp_path)
    assert mine.shape == theirs.shape == (1, 3, 224, 224)
    assert torch.equal(mine, theirs)


def test_legacy_mode_reproduces_old_inference_exactly():
    crop = _synthetic_crop(1)
    face = torch.from_numpy(crop).permute(2, 0, 1).float()
    assert torch.equal(vi.preprocess_face(face, match_training_jpeg=False), _legacy_inference_tensor(face))


def test_jpeg_step_actually_changes_pixels():
    face = torch.from_numpy(_synthetic_crop(2)).permute(2, 0, 1).float()
    diff = (vi.training_jpeg_roundtrip(face) - face).abs()
    assert diff.max() > 0 and diff.mean() > 0.1


def test_jpeg_quality_constant_is_opencv_default():
    # cv2.imwrite default must equal the quality we emulate (guards against an OpenCV default change)
    crop = _synthetic_crop(3)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    ok2, buf2 = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, vi.TRAINING_JPEG_QUALITY])
    assert ok and ok2 and buf.tobytes() == buf2.tobytes()


def test_preprocess_handles_non_224_input():
    face = torch.rand(3, 160, 160) * 255
    out = vi.preprocess_face(face, match_training_jpeg=False)
    assert out.shape == (1, 3, 224, 224)


@pytest.mark.needs_media
def test_real_mtcnn_training_crop_matches_inference_crop(media_image, tmp_path):
    """Strongest check: crop a real face exactly as notebook cell 0 did (post_process=True, save_path),
    then compare with the inference path."""
    from facenet_pytorch import MTCNN
    rgb = cv2.cvtColor(cv2.imread(media_image), cv2.COLOR_BGR2RGB)
    saved = str(tmp_path / "frame_0.jpg")
    MTCNN(image_size=224, margin=20, device="cpu", post_process=True)(rgb, save_path=saved)
    train_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(
        Image.open(saved).convert("RGB")).unsqueeze(0)
    mt = MTCNN(image_size=224, margin=20, device="cpu", post_process=False)
    face, box, conf = vi._detect_largest_face(mt, rgb)
    assert face is not None and len(box) == 4
    assert torch.equal(vi.preprocess_face(face, True), train_tensor)
