import threading

import pytest
import torch

from verifai.model import ModelBundle, TwoStreamNet, load_checkpoint_state, load_model, resolve_device, sha256_file
from verifai.preprocess import jpeg_roundtrip, normalise, preprocess_faces


def test_load_model_new_format_and_metadata(random_checkpoint):
    b = load_model(random_checkpoint, "cpu")
    assert b.info.name == "random-init-test" and b.info.version == "0.0"
    assert b.info.parameters == 11_972_954
    assert b.info.checkpoint_sha256 == sha256_file(random_checkpoint)
    assert not b.model.training and b.info.decision_threshold == 0.5


def test_load_model_legacy_raw_state_dict(tmp_path):
    p = tmp_path / "raw.pth"
    torch.save(TwoStreamNet(pretrained=False).state_dict(), p)
    b = load_model(p, "cpu")
    assert b.info.version == "1.0-original" and "Celeb-DF" in b.info.trained_on


def test_strict_loading_rejects_wrong_checkpoint(tmp_path):
    sd = TwoStreamNet(pretrained=False).state_dict(); sd.pop("fc.0.bias")
    p = tmp_path / "bad.pth"; torch.save(sd, p)
    with pytest.raises(RuntimeError):
        load_model(p, "cpu")


def test_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        load_checkpoint_state("nope.pth", torch.device("cpu"))


def test_device_resolution(monkeypatch):
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning):
        assert resolve_device("cuda").type == "cpu"
    assert resolve_device("auto").type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_cuda_inference_when_available(random_checkpoint):
    b = load_model(random_checkpoint, "cuda")
    assert b.device.type == "cuda"
    assert b.predict_batch(torch.zeros(2, 3, 224, 224)).shape == (2,)


def test_sigmoid_of_logits_equals_forward(random_bundle):
    x = torch.randn(3, 3, 224, 224, generator=torch.Generator().manual_seed(2))
    m = random_bundle.model
    with torch.no_grad():
        torch.testing.assert_close(torch.sigmoid(m.logits(x)).squeeze(1), m(x).squeeze(1), rtol=0, atol=1e-6)


def test_predict_batch_range_and_thread_safety(random_bundle):
    x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(3))
    ref = random_bundle.predict_batch(x)
    assert ((ref >= 0) & (ref <= 1)).all()
    out = []
    ts = [threading.Thread(target=lambda: out.append(random_bundle.predict_batch(x))) for _ in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all((o == ref).all() for o in out)


def test_srm_kernel_frozen_and_unchanged(random_bundle):
    k = random_bundle.model.srm_layer.kernel
    assert not k.requires_grad and tuple(k.shape) == (3, 3, 3, 3)
    assert float(k[0, 0].sum()) == 0 and float(k[1, 0].sum()) == 0 and float(k[2, 0].sum()) == 0     # zero-sum high-pass kernels


def test_new_model_matches_legacy_model_on_original_checkpoint(checkpoint_path):
    """The package model is the SAME function as the Phase-1 module on the original weights."""
    import verifai_inference as legacy
    old = legacy.TwoStreamNet(pretrained=False)
    legacy.load_checkpoint(old, checkpoint_path, "cpu"); old.eval()
    new = load_model(checkpoint_path, "cpu")
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.equal(old(x), new.model(x))
    assert new.info.version == "1.0-original"


def test_preprocessing_matches_legacy_preprocessing():
    import verifai_inference as legacy
    face = (torch.rand(3, 224, 224, generator=torch.Generator().manual_seed(5)) * 255).round()
    assert torch.equal(preprocess_faces([face], True), legacy.preprocess_face(face, True))
    assert torch.equal(preprocess_faces([face], False), legacy.preprocess_face(face, False))
    assert preprocess_faces([], True).shape == (0, 3, 224, 224)


def test_jpeg_roundtrip_changes_pixels_but_not_shape():
    face = (torch.rand(3, 224, 224, generator=torch.Generator().manual_seed(6)) * 255).round()
    out = jpeg_roundtrip(face)
    assert out.shape == face.shape and (out != face).any()
    assert normalise(out.unsqueeze(0)).shape == (1, 3, 224, 224)
