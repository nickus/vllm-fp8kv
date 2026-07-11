# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Wire the fp8 sparse-MLA decode kernel into vLLM's TRITON_MLA_SPARSE backend.

Upstream today (PR #47629, `triton_mla_sparse.py` + its XPU base):

    def forward_mqa(...):
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 kv is not supported with XPU MLA Sparse yet")
        ...
        topk_indices_global = triton_convert_req_index_to_global_index(...)
        attn_out = self._forward_bf16_kv(q, kv_cache, topk_indices_global, md)

    class TritonMLASparseBackend(AttentionBackend):
        supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

...i.e. the gap is spelled out in the source. This module removes it:

  * declares `fp8_ds_mla` as a supported KV dtype on the backend,
  * makes `get_kv_cache_shape` allocate the 656-byte `fp8_ds_mla` page
    (mirroring `flashmla_sparse.py`'s handling) instead of a 576-wide bf16 row,
  * maps `--kv-cache-dtype fp8 / fp8_e4m3` -> `fp8_ds_mla` for this backend in
    `_canonicalize_sparse_mla_kv_cache_dtype` (upstream only does so for
    FLASHMLA_SPARSE and SM120), and
  * adds `_forward_fp8_kv`, dispatching the (already flat, already global)
    top-k slot indices into our software-dequant Triton kernel.

Nothing else changes: the cache is written by vLLM's own C++
`concat_and_cache_ds_mla` kernel (which runs on sm_86 unmodified — verified),
the index conversion is upstream's own
`triton_convert_req_index_to_global_index`, and the page layout is untouched.

`apply()` monkeypatches a live vLLM for testing; `patches/
triton_mla_sparse_fp8.patch` is the equivalent kernel diff for upstream.

STATUS (2026-07-11 review): the decode kernel and cache roundtrip are verified
on hardware (see RESULTS.md); a full engine boot (`LLM()` with
`--kv-cache-dtype fp8_e4m3`) through this patch has NOT yet been run — that is
open item B (engine-level AC1).
"""

import torch

from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode

# vLLM's name for the 656-byte DeepSeek-MLA fp8 page format
FP8_DS_MLA = "fp8_ds_mla"


def _forward_fp8_kv(self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata):
    """Sparse MLA decode over fp8_ds_mla pages (sm_80 / sm_86).

    `topk_indices` are already FLAT GLOBAL SLOT indices with -1 sentinels —
    upstream's `triton_convert_req_index_to_global_index` produced them
    (block_table[req, tok // B] * B + tok % B), which is precisely our kernel's
    contract. The cache arrives as (num_blocks, block_size, 656) uint8.
    """
    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)
    num_tokens = q.shape[0]
    out = fp8_ds_mla_sparse_decode(
        q,
        kv_c_and_k_pe_cache,
        topk_indices.view(num_tokens, 1, -1),
        sm_scale=self.softmax_scale,
    )
    return out[:, : self.num_heads, :]


def _forward_mqa_fp8_aware(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
    """Replacement for the base `forward_mqa`: fp8_ds_mla is no longer refused."""
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)

    num_actual_toks = q.shape[0]
    assert self.topk_indices_buffer is not None
    topk_indices = self.topk_indices_buffer[:num_actual_toks]

    topk_indices_global = triton_convert_req_index_to_global_index(
        attn_metadata.req_id_per_token,
        attn_metadata.block_table,
        topk_indices,
        BLOCK_SIZE=attn_metadata.block_size,
        NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
    )

    if self.kv_cache_dtype == FP8_DS_MLA:
        attn_out = _forward_fp8_kv(
            self, q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
        )
    else:
        attn_out = self._forward_bf16_kv(
            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
        )
    return attn_out, None


def _get_kv_cache_shape_fp8_aware(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    """fp8_ds_mla pages are 656 B/token, not head_size wide.

    Mirrors `flashmla_sparse.py::get_kv_cache_shape`. Without this the engine
    allocates a (num_blocks, block_size, 576) cache and the C++ writer scribbles
    past every row.
    """
    if cache_dtype_str == FP8_DS_MLA:
        return (num_blocks, block_size, 656)
    return (num_blocks, block_size, head_size)


def apply() -> None:
    """Monkeypatch a live vLLM (test/bench use; the .patch file is for upstream)."""
    from vllm.model_executor.layers.attention import mla_attention as mla_mod
    from vllm.v1.attention.backends.mla import triton_mla_sparse as tms

    impl, backend = tms.TritonMLASparseImpl, tms.TritonMLASparseBackend

    impl._forward_fp8_kv = _forward_fp8_kv
    impl.forward_mqa = _forward_mqa_fp8_aware
    backend.get_kv_cache_shape = staticmethod(_get_kv_cache_shape_fp8_aware)

    if FP8_DS_MLA not in backend.supported_kv_cache_dtypes:
        backend.supported_kv_cache_dtypes = [
            *backend.supported_kv_cache_dtypes,
            FP8_DS_MLA,
        ]

    # `--kv-cache-dtype fp8 / fp8_e4m3` must canonicalize to fp8_ds_mla for
    # this backend, as upstream already does for FLASHMLA_SPARSE and SM120.
    # ("auto" intentionally stays bf16, matching upstream's Ampere default.)
    if getattr(mla_mod._canonicalize_sparse_mla_kv_cache_dtype,
               "_vllm_fp8kv", False):
        return  # apply() called twice — don't stack wrappers
    _upstream_canon = mla_mod._canonicalize_sparse_mla_kv_cache_dtype

    def _canon_fp8_aware(attn_backend, kv_cache_dtype):
        if attn_backend.get_name() == "TRITON_MLA_SPARSE" and kv_cache_dtype in (
            "fp8",
            "fp8_e4m3",
        ):
            return FP8_DS_MLA
        return _upstream_canon(attn_backend, kv_cache_dtype)

    _canon_fp8_aware._vllm_fp8kv = True
    mla_mod._canonicalize_sparse_mla_kv_cache_dtype = _canon_fp8_aware

    print(f"[vllm_fp8kv] TRITON_MLA_SPARSE now supports {FP8_DS_MLA} "
          f"(kv dtypes: {backend.supported_kv_cache_dtypes}; "
          f"cache shape + dtype canonicalization patched)")
