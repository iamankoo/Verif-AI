"""Image-quality degradations used (a) as SYMMETRIC training augmentation and (b) as a fixed, deterministic evaluation condition.

Why this exists (see docs/VIDEO_FALSE_POSITIVE_INVESTIGATION.md): in the training data the fake faces are systematically softer
(lower sharpness / less high-frequency noise) than the real photographs, and the model learned "soft, low-detail face => FAKE".
Ordinary phone video frames are soft, so genuine footage was flagged. Applying the *same* random blur / down-up-scaling /
JPEG / noise / low-light degradations to real AND fake training crops makes softness an uninformative cue. The augmentation is
class-symmetric on purpose: it must never be used to favour one class.

Evaluation never touches the training augmentation: `fixed_degrade` is deterministic (named conditions) and is applied identically
to every row of a split, whatever its label.
"""
from __future__ import annotations

import io
import random
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter


def blur(im: Image.Image, sigma: float) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def down_up(im: Image.Image, factor: float) -> Image.Image:
    w, h = im.size
    small = im.resize((max(8, int(w * factor)), max(8, int(h * factor))), Image.BILINEAR, reducing_gap=None)
    return small.resize((w, h), Image.BICUBIC)


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_noise(im: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(im, dtype=np.float32)
    a = a + rng.normal(0.0, sigma, a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def low_light(im: Image.Image, gain: float, noise_sigma: float, rng: np.random.Generator) -> Image.Image:
    a = np.clip(np.asarray(im, dtype=np.float32) * gain, 0, 255)
    return add_noise(Image.fromarray(a.astype(np.uint8)), noise_sigma, rng)


# ------------------------------------------------------------------------------------------------ fixed evaluation conditions
FIXED: Dict[str, Callable[[Image.Image, np.random.Generator], Image.Image]] = {
    "blur_s1.5": lambda im, r: blur(im, 1.5),
    "blur_s2.5": lambda im, r: blur(im, 2.5),
    "downup_0.5": lambda im, r: down_up(im, 0.5),
    "downup_0.3": lambda im, r: down_up(im, 0.3),
    "jpeg_q25": lambda im, r: jpeg(im, 25),
    "lowlight_noise": lambda im, r: low_light(im, 0.5, 6.0, r),
    "phone_soft": lambda im, r: jpeg(down_up(blur(im, 1.2), 0.6), 60),
}
FIXED_NAMES = tuple(FIXED)


def fixed_degrade(im: Image.Image, name: str, seed: int = 0) -> Image.Image:
    """Deterministic named degradation. 'mixed' is resolved by the caller (see FaceCropDataset)."""
    if name not in FIXED:
        raise ValueError(f"unknown degradation {name!r}; choose from {list(FIXED)} or 'mixed'")
    return FIXED[name](im.convert("RGB"), np.random.default_rng(seed))


# ------------------------------------------------------------------------------------------------ random training augmentation
def random_degrade(im: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """One or two random degradations with random strengths. Applied with the same probability to every class."""
    rng = rng or random
    nprng = np.random.default_rng(rng.getrandbits(32))
    ops = ["blur", "downup", "jpeg", "noise", "lowlight"]
    out = im.convert("RGB")
    for op in rng.sample(ops, k=1 if rng.random() < 0.6 else 2):
        if op == "blur":
            out = blur(out, rng.uniform(0.5, 3.0))
        elif op == "downup":
            out = down_up(out, rng.uniform(0.25, 0.8))
        elif op == "jpeg":
            out = jpeg(out, rng.randint(15, 70))
        elif op == "noise":
            out = add_noise(out, rng.uniform(2.0, 10.0), nprng)
        else:
            out = low_light(out, rng.uniform(0.4, 0.9), rng.uniform(0.0, 8.0), nprng)
    return out


class RandomQuality:
    """torchvision-style transform: with probability p apply `random_degrade` (class-blind)."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, im: Image.Image) -> Image.Image:
        return random_degrade(im) if random.random() < self.p else im
