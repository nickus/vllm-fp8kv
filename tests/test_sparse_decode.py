# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The fp8_ds_mla sparse-MLA decode kernel, checked against the fp32 reference.

Runs on CPU under the Triton interpreter (see conftest) — kernel math is CI-able
without a GPU. The bf16 production dtype cannot be exercised there (the
interpreter's bf16 is uint16 under the hood), so those cases are GPU-gated.
"""

import pytest
import torch

from tests.conftest import DEVICE, needs_cuda
from verify.metrics import cosine, max_abs
from verify.reference import ref_sparse_mla_decode
from vllm_fp8kv.fp8_ds_mla_sparse_decode import (
    NOPE,
    ROPE,
    ROW_BYTES,
    SCALE_OFF,
    fp8_ds_mla_sparse_decode,
)

SM = 1.0 / (NOPE + ROPE) ** 0.5


def _q(seq_q=2, h_q=16, seed=1, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(seq_q, h_q, NOPE + ROPE, generator=g) * 0.5).to(DEVICE).to(dtype)


def _idx(rows, n_slots=64, topk=32, seed=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, n_slots, (rows, topk), generator=g).to(DEVICE).int()


def test_geometry_constants_match_the_format():
    """656 B/token = 512 fp8 + 4x fp32 scales + 64x bf16 rope."""
    assert ROW_BYTES == 656
    assert SCALE_OFF == 512
    assert NOPE + ROPE == 576


def test_decode_matches_fp32_reference(kv_pool):
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(), _idx(2)
    out = fp8_ds_mla_sparse_decode(q, packed.reshape(1, 64, ROW_BYTES), idx, SM)
    ref = ref_sparse_mla_decode(q, packed, idx, SM)

    assert out.shape == (2, 16, NOPE)
    assert out.isfinite().all()
    assert cosine(out, ref) > 0.999, f"cosine={cosine(out, ref)}"
    assert max_abs(out, ref) < 5e-2


def test_masked_entries_are_excluded_not_attended(kv_pool):
    """-1 sentinels must be dropped; the result must equal attending over the
    surviving indices only."""
    packed, _ = kv_pool(n_slots=64)
    q = _q(seq_q=1)
    idx = _idx(1)
    idx[0, ::3] = -1

    out = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    ref = ref_sparse_mla_decode(q, packed, idx, SM)
    assert out.isfinite().all()
    assert cosine(out, ref) > 0.999


def test_leading_fully_masked_chunk_does_not_poison_the_row(kv_pool):
    """Regression: an -inf running max makes exp2(-inf - -inf) = NaN, which
    permanently poisons the accumulator even when valid keys follow. The kernel
    substitutes a safe max for exactly this case."""
    packed, _ = kv_pool(n_slots=64)
    q = _q(seq_q=1)
    idx = torch.full((1, 32), -1, dtype=torch.int32, device=DEVICE)
    idx[0, 16] = 5   # the ONLY valid key, after the first BLOCK_N=16 chunk

    out = fp8_ds_mla_sparse_decode(q, packed, idx, SM, block_n=16)
    assert out.isfinite().all(), "NaN poisoning regressed"
    ref = ref_sparse_mla_decode(q, packed, idx, SM)
    assert cosine(out, ref) > 0.999
    assert max_abs(out, ref) < 5e-2


def test_row_selecting_nothing_yields_zeros(kv_pool):
    packed, _ = kv_pool(n_slots=64)
    q = _q(seq_q=2)
    idx = _idx(2)
    idx[1] = -1                      # row 1 selects nothing at all

    out = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    assert out.isfinite().all()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert not torch.equal(out[0], torch.zeros_like(out[0]))


def test_out_of_range_indices_are_masked(kv_pool):
    """idx >= num_slots must be treated as masked, not read out of bounds."""
    packed, _ = kv_pool(n_slots=64)
    q = _q(seq_q=1)
    idx = _idx(1)
    valid = idx.clone()
    idx[0, ::4] = 9999               # beyond the pool
    valid[0, ::4] = -1               # the reference's way of saying the same

    out = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    ref = ref_sparse_mla_decode(q, packed, valid, SM)
    assert out.isfinite().all()
    assert cosine(out, ref) > 0.999


def test_lse_is_returned_and_finite_only_for_rows_with_keys(kv_pool):
    packed, _ = kv_pool(n_slots=64)
    q = _q(seq_q=2)
    idx = _idx(2)
    idx[1] = -1

    out, lse = fp8_ds_mla_sparse_decode(q, packed, idx, SM, return_lse=True)
    assert lse.shape == (2, 16)
    assert lse[0].isfinite().all()
    assert (lse[1] == -float("inf")).all(), "empty row must have lse = -inf"

    # lse must equal log(sum(exp(scores))) of the reference scores
    from verify.reference import ref_dequant_fp8_ds_mla_row

    k = ref_dequant_fp8_ds_mla_row(packed[idx[0].long()]).float()
    want = torch.logsumexp((q[0].float() @ k.T) * SM, dim=-1)
    assert torch.allclose(lse[0], want, atol=1e-2, rtol=1e-2)


def test_3d_and_2d_indices_are_equivalent(kv_pool):
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(), _idx(2)
    a = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    b = fp8_ds_mla_sparse_decode(q, packed, idx.unsqueeze(1), SM)
    assert torch.equal(a, b)


def test_accepts_fp8_viewed_cache(kv_pool):
    """The engine may hand us the pool typed float8_e4m3fn rather than uint8."""
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(), _idx(2)
    a = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    b = fp8_ds_mla_sparse_decode(q, packed.view(torch.float8_e4m3fn), idx, SM)
    assert torch.equal(a, b)


def test_block_n_32_matches_block_n_16(kv_pool):
    """Chunk size must not change the answer (online-softmax rescaling)."""
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(), _idx(2)
    a = fp8_ds_mla_sparse_decode(q, packed, idx, SM, block_n=16)
    b = fp8_ds_mla_sparse_decode(q, packed, idx, SM, block_n=32)
    assert cosine(a, b) > 0.9999
    assert max_abs(a, b) < 5e-2


@pytest.mark.parametrize(
    "bad, msg",
    [
        ("dtype", "must be bf16"),
        ("cache_dtype", "uint8"),
        ("noncontig", "contiguous"),
        ("dim", "expected 576"),
    ],
)
def test_guard_assertions(kv_pool, bad, msg):
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(), _idx(2)
    cache = packed

    if bad == "dtype":
        q = q.to(torch.float16)          # fp16 rope bitcast would read garbage
    elif bad == "cache_dtype":
        cache = packed.to(torch.int8)
    elif bad == "noncontig":
        cache = packed.t()               # non-contiguous view of the pool
    elif bad == "dim":
        q = q[..., :512].contiguous()

    with pytest.raises(AssertionError):
        fp8_ds_mla_sparse_decode(q, cache, idx, SM)


@needs_cuda
def test_bf16_production_dtype_matches_reference(kv_pool):
    """The dtype the engine actually uses. GPU-only: the Triton interpreter
    cannot do bf16 arithmetic."""
    packed, _ = kv_pool(n_slots=64)
    q, idx = _q(dtype=torch.bfloat16), _idx(2)
    out = fp8_ds_mla_sparse_decode(q, packed, idx, SM)
    ref = ref_sparse_mla_decode(q, packed, idx, SM)
    assert cosine(out, ref) > 0.999
