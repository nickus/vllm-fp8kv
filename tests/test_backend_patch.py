# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""`vllm_fp8kv.backend_patch` — the wiring that makes vLLM's TRITON_MLA_SPARSE
backend accept fp8_ds_mla KV.

The 2026-07-11 audit found that NOTHING executed these functions: a latent
AttributeError sat in `_forward_fp8_kv`'s tuple-q branch, and the engine-level
gaps (576-wide cache shape, dtype canonicalization) were invisible because no
test ever asked the backend for a cache shape. These tests drive the real
functions against a faithful fake of vLLM's surface.
"""

import pytest
import torch

from tests import fakes
from tests.conftest import DEVICE
from vllm_fp8kv import backend_patch
from vllm_fp8kv.fp8_ds_mla_sparse_decode import NOPE, ROPE, ROW_BYTES


@pytest.fixture
def vllm(monkeypatch):
    return fakes.install(monkeypatch)


def test_apply_declares_the_fp8_dtype(vllm):
    assert "fp8_ds_mla" not in vllm.backend.supported_kv_cache_dtypes
    backend_patch.apply()
    assert "fp8_ds_mla" in vllm.backend.supported_kv_cache_dtypes
    # the bf16 dtypes must survive — this must be additive, not a replacement
    assert {"auto", "bfloat16"} <= set(vllm.backend.supported_kv_cache_dtypes)


def test_apply_declares_the_cli_fp8_aliases(vllm):
    """Load-bearing, and found only by booting a real engine: vLLM SELECTS the
    backend with the RAW `--kv-cache-dtype` value and canonicalizes only
    afterwards (mla_attention.py: get_attn_backend(...) then
    _canonicalize_sparse_mla_kv_cache_dtype(...)). So declaring `fp8_ds_mla`
    alone is not enough — the selector never sees that name, and the engine dies
    with "No valid attention backend found ... TRITON_MLA_SPARSE:
    [kv_cache_dtype not supported]". FLASHMLA_SPARSE declares "fp8" as an alias
    for exactly this reason."""
    backend_patch.apply()
    declared = set(vllm.backend.supported_kv_cache_dtypes)
    assert {"fp8", "fp8_e4m3"} <= declared, (
        "the CLI-visible fp8 names must be declared or the backend is never selected"
    )


def test_apply_fixes_the_cache_shape(vllm):
    """Without this the engine allocates 576-wide bf16 rows and the C++ writer
    scribbles past the end of every fp8 page."""
    before = vllm.backend.get_kv_cache_shape(8, 64, 1, 576, "fp8_ds_mla")
    assert before[-1] == 576, "fake does not reproduce upstream's bug"

    backend_patch.apply()
    after = vllm.backend.get_kv_cache_shape(8, 64, 1, 576, "fp8_ds_mla")
    assert after == (8, 64, ROW_BYTES)
    # bf16 must be unaffected
    assert vllm.backend.get_kv_cache_shape(8, 64, 1, 576, "auto") == (8, 64, 576)


@pytest.mark.parametrize(
    "requested, expect",
    [
        ("fp8", "fp8_ds_mla"),
        ("fp8_e4m3", "fp8_ds_mla"),
        ("auto", "auto"),           # Ampere default stays bf16, as upstream
        ("bfloat16", "bfloat16"),
    ],
)
def test_apply_canonicalizes_the_cli_dtype(vllm, requested, expect):
    """`--kv-cache-dtype fp8_e4m3` must reach the backend as fp8_ds_mla."""
    backend_patch.apply()
    canon = vllm.mla_attention._canonicalize_sparse_mla_kv_cache_dtype
    assert canon(vllm.backend, requested) == expect


def test_canonicalization_leaves_other_backends_to_upstream(vllm):
    backend_patch.apply()
    canon = vllm.mla_attention._canonicalize_sparse_mla_kv_cache_dtype

    class Sm120:
        @staticmethod
        def get_name():
            return "FLASHINFER_MLA_SPARSE_SM120"

    assert canon(Sm120, "auto") == "fp8_ds_mla"      # upstream's own rule, preserved
    assert canon(Sm120, "bfloat16") == "bfloat16"


def test_apply_is_idempotent(vllm):
    """Called twice (harness + engine), it must not stack wrappers or duplicate
    the dtype entry."""
    backend_patch.apply()
    first = vllm.mla_attention._canonicalize_sparse_mla_kv_cache_dtype
    backend_patch.apply()
    second = vllm.mla_attention._canonicalize_sparse_mla_kv_cache_dtype

    assert first is second, "canonicalization wrapper was stacked"
    assert vllm.backend.supported_kv_cache_dtypes.count("fp8_ds_mla") == 1


def _pool(n_slots=64, seed=3):
    from verify.reference import quantize_to_fp8_ds_mla

    g = torch.Generator(device="cpu").manual_seed(seed)
    src = (torch.randn(n_slots, NOPE + ROPE, generator=g) * 0.5).to(DEVICE)
    return quantize_to_fp8_ds_mla(src).contiguous()


def test_forward_fp8_kv_runs_the_kernel(vllm):
    backend_patch.apply()
    impl = vllm.impl(kv_cache_dtype="fp8_ds_mla", num_heads=16, softmax_scale=1 / 24.0)
    cache = _pool()
    q = torch.randn(2, 16, NOPE + ROPE, device=DEVICE)
    idx = torch.randint(0, 64, (2, 32), device=DEVICE).int()

    out = backend_patch._forward_fp8_kv(impl, q, cache, idx, attn_metadata=None)
    assert out.shape == (2, 16, NOPE)
    assert out.isfinite().all()


def test_forward_fp8_kv_accepts_tuple_q(vllm):
    """REGRESSION (audit M3): `num_tokens = q.shape[0]` used to run BEFORE the
    tuple was concatenated, raising AttributeError on the documented input form."""
    backend_patch.apply()
    impl = vllm.impl(kv_cache_dtype="fp8_ds_mla", num_heads=16, softmax_scale=1 / 24.0)
    cache = _pool()
    q_tuple = (
        torch.randn(2, 16, NOPE, device=DEVICE),
        torch.randn(2, 16, ROPE, device=DEVICE),
    )
    idx = torch.randint(0, 64, (2, 32), device=DEVICE).int()

    out = backend_patch._forward_fp8_kv(impl, q_tuple, cache, idx, attn_metadata=None)
    assert out.shape == (2, 16, NOPE)


def test_forward_fp8_kv_slices_to_num_heads(vllm):
    """The kernel pads heads up to BLOCK_H; the backend must return exactly the
    layer's head count."""
    backend_patch.apply()
    impl = vllm.impl(kv_cache_dtype="fp8_ds_mla", num_heads=8, softmax_scale=1 / 24.0)
    q = torch.randn(1, 16, NOPE + ROPE, device=DEVICE)
    idx = torch.randint(0, 64, (1, 32), device=DEVICE).int()

    out = backend_patch._forward_fp8_kv(impl, q, _pool(), idx, attn_metadata=None)
    assert out.shape == (1, 8, NOPE)


class _Meta:
    def __init__(self, n_tokens, block_size=64, topk=32, n_blocks=8):
        self.req_id_per_token = torch.zeros(n_tokens, dtype=torch.int32, device=DEVICE)
        self.block_table = torch.arange(n_blocks, device=DEVICE).view(1, n_blocks).int()
        self.block_size = block_size
        self.topk_tokens = topk


def test_forward_mqa_dispatches_fp8_and_bf16(vllm):
    """The whole point of the patch: fp8_ds_mla must take OUR path, everything
    else must still take upstream's."""
    backend_patch.apply()
    cache = _pool(n_slots=8 * 64)
    md = _Meta(n_tokens=2)
    q = torch.randn(2, 16, NOPE + ROPE, device=DEVICE)
    topk = torch.randint(0, 8 * 64, (2, 32), device=DEVICE).int()

    fp8_impl = vllm.impl(kv_cache_dtype="fp8_ds_mla", softmax_scale=1 / 24.0)
    fp8_impl.topk_indices_buffer = topk
    out, extra = fp8_impl.forward_mqa(q, cache, md, layer=None)
    assert extra is None
    assert out.shape == (2, 16, NOPE) and out.isfinite().all()
    assert fp8_impl.bf16_calls == 0, "fp8 request was routed to the bf16 kernel"

    bf16_impl = vllm.impl(kv_cache_dtype="auto", softmax_scale=1 / 24.0)
    bf16_impl.topk_indices_buffer = topk
    out2, _ = bf16_impl.forward_mqa(q, cache, md, layer=None)
    assert bf16_impl.bf16_calls == 1, "bf16 request was hijacked by the fp8 path"
    assert out2.shape == (2, 16, NOPE)


def test_forward_mqa_uses_upstreams_index_converter(vllm):
    """Indices reaching our kernel must be flat GLOBAL slots (block_table
    applied), not token-relative ones."""
    backend_patch.apply()
    seen = {}

    real = backend_patch._forward_fp8_kv

    def spy(self, q, cache, topk_indices, attn_metadata):
        seen["idx"] = topk_indices.clone()
        return real(self, q, cache, topk_indices, attn_metadata)

    import vllm_fp8kv.backend_patch as bp

    orig, bp._forward_fp8_kv = bp._forward_fp8_kv, spy
    try:
        impl = vllm.impl(kv_cache_dtype="fp8_ds_mla", softmax_scale=1 / 24.0)
        md = _Meta(n_tokens=1)
        md.block_table = torch.tensor([[3, 1, 0, 2, 4, 5, 6, 7]], device=DEVICE).int()
        impl.topk_indices_buffer = torch.tensor(
            [[0, 1, 64, -1] + [-1] * 28], device=DEVICE).int()
        impl.forward_mqa(
            torch.randn(1, 16, NOPE + ROPE, device=DEVICE), _pool(8 * 64), md, layer=None)
    finally:
        bp._forward_fp8_kv = orig

    idx = seen["idx"][0]
    assert int(idx[0]) == 3 * 64 + 0, "block_table not applied to token 0"
    assert int(idx[1]) == 3 * 64 + 1
    assert int(idx[2]) == 1 * 64 + 0, "second block not remapped"
    assert int(idx[3]) == -1, "the -1 sentinel must survive the conversion"


def test_apply_marks_the_fp8_spec_as_quantized(vllm):
    """Found only by booting a real engine (2026-07-12).

    `MLAAttention.get_kv_cache_spec` builds MLAAttentionSpec WITHOUT
    kv_quant_mode, so it defaults to NONE. The worker's reshape path then asks
    the backend for the *unquantized* cache shape:

        layer_cache_dtype = "auto" if spec.kv_quant_mode == NONE else cache_dtype

    ...while the pool itself was allocated from spec.page_size_bytes, which DOES
    special-case fp8_ds_mla at 656 B/token. The engine dies reshaping a 656-byte
    pool into 576-byte rows:

        RuntimeError: shape '[128811, 64, 576]' is invalid for input of size 5408001024

    Declaring the quant mode makes allocation and reshape agree.
    """
    backend_patch.apply()
    layer = vllm.MLAAttention(kv_cache_dtype="fp8_ds_mla")

    spec = layer.get_kv_cache_spec("fp8_ds_mla")
    assert spec.kv_quant_mode != "NONE", "fp8 pool still declared unquantized"

    bf16_spec = layer.get_kv_cache_spec("auto")
    assert bf16_spec.kv_quant_mode == "NONE", "bf16 path must be untouched"


def test_spec_patch_is_idempotent(vllm):
    backend_patch.apply()
    first = vllm.MLAAttention.get_kv_cache_spec
    backend_patch.apply()
    assert vllm.MLAAttention.get_kv_cache_spec is first
