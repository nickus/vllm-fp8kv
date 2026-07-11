# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""`verify/reference.py` and `verify/metrics.py`.

The reference is the yardstick every kernel test is measured against, so it gets
checked against something even more primitive: a dense softmax written inline
here, and torch's own fp8 cast. A wrong reference would make every other test in
this repo agree on the wrong answer.
"""

import pytest
import torch

from tests.conftest import DEVICE
from verify.metrics import Report, cosine, max_abs, topk_set_overlap
from verify.reference import (
    NOPE,
    ROPE,
    ROW_BYTES,
    quantize_to_fp8_ds_mla,
    ref_dequant_fp8_ds_mla_row,
    ref_sparse_mla_decode,
)


def test_pack_unpack_roundtrip_is_within_fp8_quantization_error():
    g = torch.Generator(device="cpu").manual_seed(4)
    src = (torch.randn(32, NOPE + ROPE, generator=g) * 0.5).to(DEVICE)

    packed = quantize_to_fp8_ds_mla(src)
    assert packed.shape == (32, ROW_BYTES)
    assert packed.dtype == torch.uint8

    got = ref_dequant_fp8_ds_mla_row(packed)
    # NoPE: fp8e4m3 has 3 mantissa bits -> ~6% worst case per element, but the
    # per-128-tile amax scaling keeps the RELATIVE error small
    rel = ((got[:, :NOPE] - src[:, :NOPE]).abs() / src[:, :NOPE].abs().clamp(1e-3)).median()
    assert rel < 0.05
    assert cosine(got[:, :NOPE], src[:, :NOPE]) > 0.999
    # RoPE: raw bf16 passthrough -> EXACT (bf16 rounding of the source, nothing more)
    assert torch.equal(got[:, NOPE:], src[:, NOPE:].to(torch.bfloat16).float())


def test_scales_are_amax_over_448_per_128_tile():
    """Pin the writer's quantization contract; a different divisor would silently
    rescale everything."""
    g = torch.Generator(device="cpu").manual_seed(5)
    src = (torch.randn(4, NOPE + ROPE, generator=g) * 2.0).to(DEVICE)
    packed = quantize_to_fp8_ds_mla(src)

    got_scales = packed[:, NOPE:NOPE + 16].contiguous().view(torch.float32)
    want = src[:, :NOPE].reshape(4, 4, 128).abs().amax(-1) / 448.0
    assert torch.allclose(got_scales, want, rtol=1e-6)


def test_reference_decode_equals_a_dense_softmax():
    """The reference itself, checked against the most naive attention possible."""
    g = torch.Generator(device="cpu").manual_seed(6)
    kv = (torch.randn(32, NOPE + ROPE, generator=g) * 0.5).to(DEVICE)
    packed = quantize_to_fp8_ds_mla(kv)
    q = (torch.randn(2, 4, NOPE + ROPE, generator=g) * 0.5).to(DEVICE)
    idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=DEVICE, dtype=torch.int32)
    sm = 1.0 / (NOPE + ROPE) ** 0.5

    got = ref_sparse_mla_decode(q, packed, idx, sm)

    k = ref_dequant_fp8_ds_mla_row(packed).float()
    want = torch.zeros_like(got)
    for t in range(2):
        sel = k[idx[t].long()]                       # [4, 576]
        w = torch.softmax((q[t].float() @ sel.T) * sm, dim=-1)
        want[t] = w @ sel[:, :NOPE]                  # V = the NoPE half
    assert torch.allclose(got, want, atol=1e-5)


def test_reference_returns_zeros_for_rows_that_select_nothing():
    g = torch.Generator(device="cpu").manual_seed(7)
    packed = quantize_to_fp8_ds_mla((torch.randn(8, NOPE + ROPE, generator=g)).to(DEVICE))
    q = torch.randn(2, 4, NOPE + ROPE, device=DEVICE)
    idx = torch.tensor([[0, 1], [-1, -1]], device=DEVICE, dtype=torch.int32)

    out = ref_sparse_mla_decode(q, packed, idx, 1.0)
    assert out[1].abs().max() == 0.0
    assert out[0].abs().max() > 0.0


def test_reference_masks_out_of_range_indices():
    g = torch.Generator(device="cpu").manual_seed(8)
    packed = quantize_to_fp8_ds_mla((torch.randn(8, NOPE + ROPE, generator=g)).to(DEVICE))
    q = torch.randn(1, 4, NOPE + ROPE, device=DEVICE)

    a = ref_sparse_mla_decode(q, packed, torch.tensor([[0, 1, 999]], dtype=torch.int32), 1.0)
    b = ref_sparse_mla_decode(q, packed, torch.tensor([[0, 1]], dtype=torch.int32), 1.0)
    assert torch.allclose(a, b), "an index past the pool must be ignored, not read"


# ------------------------------------------------------------------- metrics
def test_cosine_and_max_abs():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert cosine(a, a * 5) == pytest.approx(1.0, abs=1e-6)   # scale-INVARIANT
    assert max_abs(a, a * 5) == pytest.approx(12.0)           # ...max_abs is not
    assert cosine(a, -a) == pytest.approx(-1.0, abs=1e-6)
    assert cosine(torch.empty(0), torch.empty(0)) == 1.0
    assert max_abs(torch.empty(0), torch.empty(0)) == 0.0


def test_cosine_alone_cannot_see_a_scale_error():
    """Why RESULTS.md's retracted numbers survived their own check: an 8x-too-big
    output has cosine 1.0. Tests must gate on max_abs too."""
    a = torch.randn(64)
    assert cosine(a, a * 8) == pytest.approx(1.0, abs=1e-6)
    assert max_abs(a, a * 8) > 1.0


def test_topk_set_overlap():
    ref = torch.tensor([[1, 2, 3, -1]])
    assert topk_set_overlap(ref, ref) == 1.0
    assert topk_set_overlap(torch.tensor([[3, 2, 1, -1]]), ref) == 1.0      # order-free
    assert topk_set_overlap(torch.tensor([[1, 2, 9, -1]]), ref) == pytest.approx(2 / 3)
    assert topk_set_overlap(torch.tensor([[-1]]), torch.tensor([[-1]])) == 1.0


def test_report_counts_failures_and_formats(capsys):
    rep = Report()
    rep.check("a/ok", True, "fine")
    rep.check("a/bad", False, "broken")
    rep.skip("a/skipped", "no gpu")

    assert rep.failed == 1
    out = capsys.readouterr().out
    assert "[PASS] a/ok: fine" in out
    assert "[FAIL] a/bad: broken" in out
    assert "[SKIP] a/skipped: no gpu" in out
    assert len(rep.lines) == 3
