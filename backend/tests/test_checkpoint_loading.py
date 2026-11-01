import os

import numpy as np
import pytest
import torch

import verifai_inference as vi
from conftest import IMAGENET_CACHE


# ---------------------------------------------------------------- device handling
def test_resolve_device_auto_matches_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert vi.resolve_device(None).type == expected
    assert vi.resolve_device("auto").type == expected


def test_resolve_device_explicit_cpu():
    assert vi.resolve_device("cpu").type == "cpu"


def test_resolve_device_cuda_request_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        assert vi.resolve_device("cuda").type == "cpu"


# ---------------------------------------------------------------- checkpoint loading
def test_checkpoint_loads_strictly_on_cpu(model):
    n = sum(v.numel() for v in model.state_dict().values() if v.dtype.is_floating_point)
    assert n == 11_972_954          # same count verified directly in the checkpoint file
    assert not model.training


def test_regression_cuda_checkpoint_without_map_location(checkpoint_path):
    """The original bug: torch.load(path) on a CPU-only machine raised RuntimeError because the
    checkpoint holds CUDA storages. load_checkpoint must work in every environment."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            torch.load(checkpoint_path, weights_only=True)          # documents the old failure
    m = vi.TwoStreamNet(pretrained=False)
    vi.load_checkpoint(m, checkpoint_path, "cpu")                    # must not raise


def test_load_uses_weights_only_and_map_location(checkpoint_path, monkeypatch):
    seen = {}
    real_load = torch.load

    def spy(*a, **k):
        seen.update(k)
        return real_load(*a, **k)

    monkeypatch.setattr(torch, "load", spy)
    vi.load_checkpoint(vi.TwoStreamNet(pretrained=False), checkpoint_path, "cpu")
    assert seen.get("weights_only") is True
    assert str(seen.get("map_location")) == "cpu"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vi.load_checkpoint(vi.TwoStreamNet(pretrained=False), str(tmp_path / "nope.pth"), "cpu")


def test_initialize_engine_missing_path_raises_instead_of_silent_untrained(tmp_path):
    with pytest.raises(FileNotFoundError):
        vi.initialize_engine(str(tmp_path / "nope.pth"), device="cpu")


def test_strict_load_detects_incompatible_checkpoint(tmp_path):
    m = vi.TwoStreamNet(pretrained=False)
    sd = m.state_dict()
    sd.pop("fc.3.bias")
    bad = tmp_path / "bad.pth"
    torch.save(sd, bad)
    with pytest.raises(RuntimeError):
        vi.load_checkpoint(vi.TwoStreamNet(pretrained=False), str(bad), "cpu")


def test_initialize_engine_returns_ready_engine(checkpoint_path):
    model, mtcnn, dev = vi.initialize_engine(checkpoint_path, device="cpu")
    assert dev.type == "cpu" and not model.training
    assert mtcnn.image_size == 224 and mtcnn.margin == 20 and mtcnn.post_process is False


# ---------------------------------------------------------------- architecture is unchanged
def test_architecture_unchanged(model):
    assert model.spatial_net._fc.in_features == 1280          # EfficientNet-B0 feature width
    assert model.noise_net._fc.in_features == 1280
    assert tuple(model.spatial_net._conv_stem.weight.shape) == (32, 3, 3, 3)
    assert tuple(model.fc[0].weight.shape) == (512, 2560)
    assert tuple(model.fc[3].weight.shape) == (1, 512)
    assert isinstance(model.fc[-1], torch.nn.Sigmoid)
    assert model.fc[2].p == 0.5


def test_srm_kernel_values_and_frozen(model):
    k = model.srm_layer.kernel
    assert tuple(k.shape) == (3, 3, 3, 3) and not k.requires_grad
    expected = np.array([[[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                         [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                         [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], dtype=np.float32) / 4.0
    for out_ch in range(3):
        for in_ch in range(3):
            np.testing.assert_array_equal(k[out_ch, in_ch].numpy(), expected[out_ch])


@pytest.mark.skipif(not os.path.isfile(IMAGENET_CACHE), reason="ImageNet EfficientNet-B0 weights not cached")
def test_pretrained_false_is_output_equivalent_to_pretrained_true(checkpoint_path):
    """Building without ImageNet weights must not change behaviour once the checkpoint is loaded."""
    a = vi.TwoStreamNet(pretrained=True)
    b = vi.TwoStreamNet(pretrained=False)
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    vi.load_checkpoint(a, checkpoint_path, "cpu")
    vi.load_checkpoint(b, checkpoint_path, "cpu")
    a.eval(); b.eval()
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=1e-7)
