# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Fakes for the slices of vLLM our code touches.

The real modules need a CUDA build of vLLM. Everything we monkeypatch or call
into is a small, well-defined surface — dtype declarations, a cache-shape
staticmethod, an index converter, a C++ cache writer — so it can be faked
faithfully and driven on CPU. That is what makes `backend_patch.py`,
`verify/contract.py`, `verify/integration.py` and `verify/capture_indices.py`
testable at all; before this they were never executed by any test.

`install(monkeypatch, ...)` inserts the fakes into `sys.modules` for one test.
"""

import dataclasses
import sys
import types

import torch

ROW_BYTES = 656
NOPE, ROPE, TILE = 512, 64, 128


# --------------------------------------------------------------------------
# faithful python re-implementation of vLLM's C++ concat_and_cache_ds_mla
# --------------------------------------------------------------------------
def write_ds_mla_pages(
    kv_c, k_pe, cache, slot_mapping, scale=None,
    scale_off=NOPE, rope_off=NOPE + 16, honour_k_scale=False,
):
    """Mirror of `concat_and_cache_ds_mla_kernel` (cache_kernels.cu).

    Per 128-element tile of the NoPE half: scale = amax/448, store fp8e4m3 and
    the fp32 scale. RoPE is copied through as RAW bf16. The layer's `k_scale`
    argument is IGNORED (that is the contract our decode depends on) — unless
    `honour_k_scale`, which simulates the day upstream changes its mind and must
    make our tripwire fire.

    `scale_off` / `rope_off` let a test simulate a re-tiled row (layout drift).
    """
    n = kv_c.shape[0]
    flat = cache.reshape(-1, ROW_BYTES)
    tiles = kv_c.float().reshape(n, NOPE // TILE, TILE)
    tile_scale = (tiles.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1.1754944e-38)
    if honour_k_scale and scale is not None:
        tiles = tiles * float(scale)
    q8 = (tiles / tile_scale).to(torch.float8_e4m3fn)

    slots = slot_mapping.long()
    flat[slots, :NOPE] = q8.view(torch.uint8).reshape(n, NOPE)
    flat[slots, scale_off:scale_off + 16] = (
        tile_scale.reshape(n, NOPE // TILE).contiguous().view(torch.uint8).reshape(n, 16)
    )
    flat[slots, rope_off:rope_off + 128] = (
        k_pe.to(torch.bfloat16).view(torch.uint8).reshape(n, 128)
    )


# --------------------------------------------------------------------------
# fake vLLM modules
# --------------------------------------------------------------------------
def _make_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def convert_req_index_to_global_index(
    req_id_per_token, block_table, token_indices, BLOCK_SIZE, NUM_TOPK_TOKENS
):
    """Mirror of upstream's Triton index converter: token-relative -> flat global
    slot, preserving negative sentinels."""
    out = torch.full_like(token_indices, -1)
    for t in range(token_indices.shape[0]):
        req = int(req_id_per_token[t])
        toks = token_indices[t]
        valid = toks >= 0
        blk = block_table[req][(toks[valid] // BLOCK_SIZE).long()]
        out[t][valid] = blk * BLOCK_SIZE + (toks[valid] % BLOCK_SIZE)
    return out


class FakeTritonMLASparseBackend:
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name():
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        return (num_blocks, block_size, head_size)      # 576-wide for EVERY dtype


class FakeTritonMLASparseImpl:
    """Only the bits `backend_patch` touches."""

    def __init__(self, kv_cache_dtype="auto", num_heads=16, softmax_scale=1.0):
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale
        self.topk_indices_buffer = None
        self.bf16_calls = 0

    def _forward_bf16_kv(self, q, kv_cache, topk_indices, attn_metadata):
        self.bf16_calls += 1
        return torch.zeros(q.shape[0], self.num_heads, NOPE, dtype=q.dtype, device=q.device)

    def forward_mqa(self, *args, **kwargs):
        """The sparse-MLA decode entry point — what capture_indices hooks."""
        return None


class FakeFlashMLASparseBackend:
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        # the backend that OWNS the fp8_ds_mla format — 656 B/token
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, ROW_BYTES)
        return (num_blocks, block_size, head_size)


@dataclasses.dataclass(frozen=True)
class FakeMLAAttentionSpec:
    """Mirrors vLLM's MLAAttentionSpec: FROZEN, and kv_quant_mode defaults to
    NONE even for an fp8_ds_mla pool — which is the bug backend_patch fixes."""

    block_size: int = 64
    num_kv_heads: int = 1
    head_size: int = 576
    cache_dtype_str: str | None = None
    kv_quant_mode: str = "NONE"


class FakeMLAAttention:
    """Only get_kv_cache_spec, which is what backend_patch wraps."""

    def __init__(self, kv_cache_dtype="auto"):
        self.kv_cache_dtype = kv_cache_dtype

    def get_kv_cache_spec(self, vllm_config):
        return FakeMLAAttentionSpec(cache_dtype_str=vllm_config)


def get_kv_quant_mode(kv_cache_dtype: str) -> str:
    return "FP8_PER_TENSOR" if str(kv_cache_dtype).startswith("fp8") else "NONE"


def canonicalize_sparse_mla_kv_cache_dtype(attn_backend, kv_cache_dtype):
    """Upstream's version: maps fp8 -> fp8_ds_mla for FLASHMLA_SPARSE / SM120 only."""
    name = attn_backend.get_name()
    if name == "FLASHMLA_SPARSE" and kv_cache_dtype.startswith("fp8"):
        return "fp8_ds_mla"
    if name == "FLASHINFER_MLA_SPARSE_SM120" and kv_cache_dtype in ("auto", "fp8", "fp8_e4m3"):
        return "fp8_ds_mla"
    return kv_cache_dtype


def install(monkeypatch, *, writer=None, sparse_kernel=None, row_bytes=ROW_BYTES):
    """Insert the fake vLLM into sys.modules for the duration of a test."""
    ops = _make_module(
        "vllm._custom_ops",
        concat_and_cache_mla=writer or (
            lambda kv_c, k_pe, cache, slot, kv_cache_dtype, scale:
                write_ds_mla_pages(kv_c, k_pe, cache, slot, scale)
        ),
    )
    flashmla = _make_module(
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        FlashMLASparseBackend=FakeFlashMLASparseBackend,
        FlashMLASparseImpl=type("FlashMLASparseImpl", (), {"forward": lambda self, *a, **k: None}),
    )
    triton_sparse = _make_module(
        "vllm.v1.attention.backends.mla.triton_mla_sparse",
        TritonMLASparseBackend=FakeTritonMLASparseBackend,
        TritonMLASparseImpl=FakeTritonMLASparseImpl,
    )
    xpu_sparse = _make_module(
        "vllm.v1.attention.backends.mla.xpu_mla_sparse",
        triton_convert_req_index_to_global_index=convert_req_index_to_global_index,
    )
    mla_attention = _make_module(
        "vllm.model_executor.layers.attention.mla_attention",
        _canonicalize_sparse_mla_kv_cache_dtype=canonicalize_sparse_mla_kv_cache_dtype,
        MLAAttention=FakeMLAAttention,
    )
    kv_cache_iface = _make_module(
        "vllm.v1.kv_cache_interface", get_kv_quant_mode=get_kv_quant_mode
    )
    kernel_mod = _make_module(
        "vllm.v1.attention.ops.triton_mla_sparse_kernel",
        triton_mla_sparse_attention=sparse_kernel or _unpatched_sparse_attention,
    )

    mods = {
        "vllm": _make_module("vllm", _custom_ops=ops),
        "vllm._custom_ops": ops,
        "vllm.model_executor": _make_module("vllm.model_executor"),
        "vllm.model_executor.layers": _make_module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.attention": _make_module(
            "vllm.model_executor.layers.attention", mla_attention=mla_attention),
        "vllm.model_executor.layers.attention.mla_attention": mla_attention,
        "vllm.v1": _make_module("vllm.v1"),
        "vllm.v1.attention": _make_module("vllm.v1.attention"),
        "vllm.v1.attention.backends": _make_module("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.mla": _make_module(
            "vllm.v1.attention.backends.mla",
            triton_mla_sparse=triton_sparse,
            flashmla_sparse=flashmla,
            xpu_mla_sparse=xpu_sparse,
        ),
        "vllm.v1.attention.backends.mla.triton_mla_sparse": triton_sparse,
        "vllm.v1.attention.backends.mla.flashmla_sparse": flashmla,
        "vllm.v1.attention.backends.mla.xpu_mla_sparse": xpu_sparse,
        "vllm.v1.kv_cache_interface": kv_cache_iface,
        "vllm.v1.attention.ops": _make_module("vllm.v1.attention.ops"),
        "vllm.v1.attention.ops.triton_mla_sparse_kernel": kernel_mod,
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # backend class attributes are mutated by apply(); restore them afterwards
    monkeypatch.setattr(
        FakeTritonMLASparseBackend, "supported_kv_cache_dtypes",
        ["auto", "float16", "bfloat16"], raising=False,
    )
    monkeypatch.setattr(
        FakeTritonMLASparseBackend, "get_kv_cache_shape",
        FakeTritonMLASparseBackend.__dict__["get_kv_cache_shape"], raising=False,
    )
    # apply() rebinds this classmethod; restore it between tests
    monkeypatch.setattr(FakeMLAAttention, "get_kv_cache_spec",
                        FakeMLAAttention.__dict__["get_kv_cache_spec"], raising=False)
    return types.SimpleNamespace(
        backend=FakeTritonMLASparseBackend,
        impl=FakeTritonMLASparseImpl,
        mla_attention=mla_attention,
        kernel_mod=kernel_mod,
        MLAAttention=FakeMLAAttention,
    )


def _unpatched_sparse_attention(q, kv, indices, sm_scale, **kw):
    """Upstream's kernel WITHOUT our patch: bf16 KV only, no fp8 dispatch."""
    raise NotImplementedError("bf16 only")
