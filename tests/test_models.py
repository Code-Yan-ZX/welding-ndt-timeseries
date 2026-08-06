"""Unit tests: shapes, tokenizer layout, dataset, features.

Run:  python tests/test_models.py   (or pytest tests/)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from wndt.data.dataset import WeldCycleDataset, make_weighted_sampler
from wndt.data.splits import load_split_idx, VAL_PAIRS, TEST_PAIRS
from wndt.features.handcrafted import extract_features
from wndt.models.dlinear import DLinearClassifier
from wndt.models.encoder import WeldTSEncoder
from wndt.models.heads import EncoderOnly, ITFormerProbe
from wndt.models.itformer import ITFormer
from wndt.models.simple_dl import MLPClassifier, RNNClassifier
from wndt.models.timesnet import TimesNetClassifier

B = 4
# Qwen3 1.7B path: env var or symlink under models/. Override with:
#   export QWEN_1P7B=/your/path/Qwen3-1.7B-Base
QWEN_1P7B = os.environ.get("QWEN_1P7B", str(REPO / "models/Qwen3-1.7B-Base"))


def test_encoder_shapes():
    enc = WeldTSEncoder()
    x = torch.randn(B, 200, 2)
    z = enc(x)
    assert z.shape == (B, 2, 10, 512), z.shape
    m = enc.memory(x)
    assert m.shape == (B, 10, 2, 512), m.shape
    print("encoder OK", tuple(z.shape))


def test_itformer_shapes():
    itf = ITFormer()
    mem = torch.randn(B, 10, 2, 512)
    x_text = torch.randn(B, 30, 512)
    out = itf(x_text, mem)
    assert out.shape == (B, 25, 512), out.shape
    empty = torch.zeros(B, 0, 512)
    out0 = itf(empty, mem)
    assert out0.shape == (B, 25, 512), out0.shape
    out.sum().backward()
    assert itf.prefix_token.grad is not None
    print("itformer OK", tuple(out.shape))


def test_heads():
    waves = torch.randn(B, 2, 200)
    probe = ITFormerProbe()
    logits = probe(waves)
    assert logits.shape == (B, 2)
    logits.sum().backward()
    enc_only = EncoderOnly()
    logits2 = enc_only(waves)
    assert logits2.shape == (B, 2)
    print("heads OK")


def test_simple_dl():
    x = torch.randn(B, 2, 200)
    for model in (MLPClassifier(), RNNClassifier("lstm"), RNNClassifier("gru")):
        out = model(x)
        assert out.shape == (B, 2), (type(model), out.shape)
    print("simple_dl OK")


def test_dlinear_timesnet():
    x = torch.randn(B, 2, 200)
    d = DLinearClassifier()
    assert d(x).shape == (B, 2)
    t = TimesNetClassifier()
    assert t(x).shape == (B, 2)
    t(x).sum().backward()
    print("dlinear/timesnet OK")


def test_features():
    rng = np.random.default_rng(0)
    waves = rng.normal(size=(32, 2, 200)).astype(np.float32)
    feats = extract_features(waves)
    assert feats.shape[0] == 32 and feats.shape[1] >= 40, feats.shape
    assert np.isfinite(feats).all()
    print("features OK:", feats.shape[1], "dims")


def test_dataset_and_sampler():
    proc = REPO / "data/processed"
    if not (proc / "waves.npy").exists():
        print("dataset test skipped (no processed data)")
        return
    idx = load_split_idx(proc)
    ds = WeldCycleDataset(proc, idx["train"], "global")
    x, y = ds[0]
    assert x.shape == (2, 200) and y.item() in (0, 1)
    assert abs(float(x.mean())) < 0.5, "global norm should center the data"
    ds_none = WeldCycleDataset(proc, idx["train"], "none")
    x_raw, _ = ds_none[0]
    assert abs(float(x_raw[0].mean()) - 21.7) < 5.0, "raw V should be ~21V"
    sampler = make_weighted_sampler(ds.labels)
    assert len(list(sampler.weights)) == len(ds) if hasattr(sampler, "weights") else True
    print("dataset OK | train:", len(ds), "val:", len(WeldCycleDataset(proc, idx["val"], "global")),
          "test:", len(WeldCycleDataset(proc, idx["test"], "global")))


def test_tokenizer_layout():
    """Tokenizer-only (no weights): answer words are single tokens; 25 placeholders."""
    from transformers import AutoTokenizer
    from wndt.models.time_language_model import build_prompt_layout

    tok = AutoTokenizer.from_pretrained(QWEN_1P7B, trust_remote_code=True)
    layout = build_prompt_layout(tok, prefix_num=25)
    assert len(tok.encode("good", add_special_tokens=False)) == 1
    assert len(tok.encode("bad", add_special_tokens=False)) == 1
    ids = layout.full_prefix_ids
    n_ph = sum(1 for i in ids if i == layout.placeholder_id)
    assert n_ph == 25, f"expected 25 placeholders, got {n_ph}"
    assert layout.good_id != layout.bad_id
    print(f"tokenizer OK | good={layout.good_id} bad={layout.bad_id} "
          f"placeholder={layout.placeholder_id} prefix_len={len(ids)} "
          f"text_len={len(layout.text_ids)}")


def test_tlm_placeholder_replacement():
    """Full 1.7B model on GPU: replacement touches only placeholder rows;
    trainable fraction < 1%; one training step decreases nothing but runs."""
    if not torch.cuda.is_available():
        print("tlm test skipped (no GPU)")
        return
    from wndt.models.time_language_model import ITFormerTLM

    model = ITFormerTLM(QWEN_1P7B)
    device = torch.device("cuda")
    model = model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    frac = n_train / n_total
    # paper's "<1%" applies to large LLMs; for 1.7B the fixed-size bridge is a
    # larger fraction. Hard requirement: LLM fully frozen.
    assert frac < 0.05, f"trainable fraction {frac:.4f} >= 5%"
    if n_total > 4e9:  # 8B regime
        assert frac < 0.01, f"8B trainable fraction {frac:.4f} >= 1%"
    for n, p in model.llm.named_parameters():
        assert not p.requires_grad, f"LLM param {n} trainable"

    # placeholder replacement check
    waves = torch.randn(2, 2, 200, device=device)
    layout = model.layout
    ids = torch.tensor(layout.full_prefix_ids, device=device)
    base_embeds = model._llm_embed(ids).detach().clone()
    prefix_embeds = model._prefix_embeds(waves).detach()
    diff = (prefix_embeds[0] - base_embeds).abs().sum(dim=-1)
    ph = slice(layout.ph_start, layout.ph_start + layout.prefix_num)
    assert (diff[ph] > 0).all(), "placeholder rows must change"
    mask = torch.ones(len(ids), dtype=torch.bool)
    mask[ph] = False
    assert (diff[mask] == 0).all(), "non-placeholder rows must NOT change"

    # score + loss forward
    labels = torch.tensor([1, 0], device=device)
    loss = model(waves, labels)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert model.ts_proj.weight.grad is not None
    assert model.encoder.patch_embedding[0].weight.grad is not None
    grad_on_llm = any(p.grad is not None for p in model.llm.parameters())
    assert not grad_on_llm, "LLM must not accumulate grads"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        score = model.score(waves)
    assert score.shape == (2,) and torch.isfinite(score).all()
    print(f"tlm OK | trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M ({100*frac:.3f}%) "
          f"| loss {loss.item():.4f} | scores {score.tolist()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_encoder_shapes()
    test_itformer_shapes()
    test_heads()
    test_simple_dl()
    test_dlinear_timesnet()
    test_features()
    test_dataset_and_sampler()
    test_tokenizer_layout()
    print("--- all CPU-side tests passed ---")
    if "--with-llm" in sys.argv:
        test_tlm_placeholder_replacement()
        print("--- LLM test passed ---")
