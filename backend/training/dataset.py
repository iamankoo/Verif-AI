"""Dataset over validated face-crop manifests (see prepare_faces.py).

Evaluation transform (identical to inference): PIL decode of the stored q95 crop -> ToTensor -> ImageNet normalise.
Training augmentation (documented in docs/DATASET_DOCUMENTATION.md): horizontal flip (p=.5), colour jitter
(brightness/contrast 0.1, saturation 0.05), Gaussian blur (p=.2, sigma 0.1-1.0), extra JPEG re-compression
(p=.5, quality 60-95). Compression/blur augmentation exists to reduce reliance on compression statistics.
"""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


@dataclass
class AugConfig:
    flip: float = 0.5
    jitter: bool = True
    blur_p: float = 0.2
    jpeg_p: float = 0.5
    jpeg_q: tuple = (60, 95)


class RandomJPEG:
    def __init__(self, p: float, q: tuple):
        self.p, self.q = p, q

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def make_transform(train: bool, aug: AugConfig = AugConfig()) -> T.Compose:
    ops: List = [T.Resize((224, 224))]
    if train:
        ops += [T.RandomHorizontalFlip(aug.flip)]
        if aug.jitter:
            ops += [T.ColorJitter(0.1, 0.1, 0.05)]
        if aug.blur_p > 0:
            ops += [T.RandomApply([T.GaussianBlur(3, (0.1, 1.0))], p=aug.blur_p)]
        if aug.jpeg_p > 0:
            ops += [RandomJPEG(aug.jpeg_p, aug.jpeg_q)]
    return T.Compose(ops + [T.ToTensor(), NORM])


def read_manifest(path: str, splits: Sequence[str]) -> List[Dict]:
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["split"] in set(splits)]
    if not rows:
        raise ValueError(f"no rows for splits {list(splits)} in {path}")
    return rows


class FaceCropDataset(Dataset):
    def __init__(self, manifest: str, splits: Sequence[str], train: bool = False, aug: AugConfig = AugConfig(), root: str = "."):
        self.rows = read_manifest(manifest, splits)
        self.tf = make_transform(train, aug)
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["crop_path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(float(r["label"]), dtype=torch.float32)

    def labels(self) -> List[int]:
        return [int(r["label"]) for r in self.rows]
