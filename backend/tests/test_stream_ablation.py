"""Stream ablation: combined == the model's own output, and neutralising a stream changes the score (tiny synthetic data, CPU)."""
import numpy as np
import torch

from test_training_smoke import make_manifest
from training.dataset import FaceCropDataset
from training.stream_ablation import collect, head, run
from verifai.model import load_model


def test_combined_matches_model_forward(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    b = load_model(random_checkpoint, "cpu")
    ds = FaceCropDataset(str(m), ["val"], train=False)
    f1, f2 = collect(b.model.eval(), ds)
    combined = head(b.model, f1, f2)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        direct = b.model(x).squeeze(1).numpy()
    assert np.allclose(combined, direct, atol=1e-5)


def test_run_reports_three_variants_and_examples(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    res = run(str(random_checkpoint), str(m), ["val"], ["none", "mixed"], ref_split="train", n_ref=4, n_examples=2, device="cpu")
    for key in ("val/none", "val/mixed"):
        block = res["results"][key]
        assert set(block["variants"]) == {"combined", "rgb_only", "srm_only"}
        assert len(block["examples"]) == 4 and {e["label"] for e in block["examples"]} == {"real", "fake"}
        for v in block["variants"].values():
            assert 0.0 <= v["false_positive_rate"] <= 1.0
    # wiring: the reported single-stream scores equal the head applied to the neutralised features
    from training.stream_ablation import reference_means
    b = load_model(random_checkpoint, "cpu"); model = b.model.eval()
    r1, r2 = reference_means(model, str(m), "train", 4, 0)
    ds = FaceCropDataset(str(m), ["val"], train=False)
    f1, f2 = collect(model, ds)
    y = np.array(ds.labels())
    exp_rgb = head(model, f1, r2.expand_as(f2))[y == 0].mean()
    exp_srm = head(model, r1.expand_as(f1), f2)[y == 0].mean()
    v = res["results"]["val/none"]["variants"]
    assert abs(v["rgb_only"]["mean_score_real"] - exp_rgb) < 1e-6 and abs(v["srm_only"]["mean_score_real"] - exp_srm) < 1e-6
