# Copyright 2026 Nick / vllm-fp8kv contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Softmax/skeleton structure follows vLLM's own pure-Triton sparse-MLA decode
# kernel (vllm/v1/attention/ops/xpu_mla_sparse.py, Apache-2.0). The fp8
# in-register software dequant is ported from nickus/dsa-3090 (Apache-2.0),
# itself derived from renning22/glm-5.2-4090 (Apache-2.0).
"""Sparse-MLA decode over **fp8_ds_mla** KV pages, for GPUs without hardware
FP8 (CUDA sm_80 / sm_86 — A100, RTX 3090, A40, A6000).

Why this exists
---------------
vLLM's Ampere sparse-MLA path (TRITON_MLA_SPARSE, PR #38476/#47629) is
bf16-KV-only, because Triton cannot compile `fp8e4nv` load/convert on sm_8x
("Triton fp8e4nv store ... does not compile on SM80" — #47629). That doubles
KV bytes per token (1152 B vs 656 B) and is the binding constraint on 24 GB
consumer cards.

This kernel removes that constraint **without asking Triton for fp8 at all**:
the standard `fp8_ds_mla` pages are loaded as raw `uint8` and decoded to fp16
in-register with bit-math that is bit-exact against `torch.float8_e4m3fn`
(see `vllm_fp8kv/fp8_dequant.py`). Nothing about the cache layout changes —
the C++ writer (`concat_and_cache_ds_mla_kernel`) already runs on sm_86.

This is the same architecture vLLM already accepts for XPU in
`vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py` ("dequantize FP8 KV
cache pages to BF16 on the fly, then reuse the BF16 sparse MLA attention
kernel"), brought to CUDA sm_8x.

The fp8_ds_mla row (656 B/token), authoritative per
csrc/libtorch_stable/cache_kernels.cu:446 and the golden parser in
tests/v1/attention/test_sparse_mla_backends.py:84 —

    [   0 : 512 )  512 x fp8e4m3   NoPE latent (kv_lora_rank)
    [ 512 : 528 )    4 x fp32      per-128-element tile scales (amax / 448)
    [ 528 : 656 )   64 x bf16      RoPE, stored RAW — never quantized

MLA shares K and V: V is the NoPE half of the same row, so the dequantized
NoPE block is used twice (transposed) and loaded once.

Scale contract (trap): `fp8_ds_mla` carries its scales INLINE, per 128-tile.
The layer's `_k_scale` is threaded into the C++ writer but that kernel ignores
it. This kernel therefore applies the inline tile scales ONLY. Applying
`_k_scale` as well would double-scale and silently degrade quality.
"""

import torch

from vllm_fp8kv.fp8_dequant import dequant_bitmath_triton

try:
    from vllm.triton_utils import tl, triton
except ImportError:  # standalone / test use without vLLM installed
    import triton
    import triton.language as tl

# fp8_ds_mla geometry — asserted by vLLM at cache_kernels.cu:864-873
NOPE = 512      # kv_lora_rank
ROPE = 64       # qk_rope_head_dim
TILE = 128      # elements per fp32 scale
ROW_BYTES = NOPE + (NOPE // TILE) * 4 + ROPE * 2   # 512 + 16 + 128 = 656
SCALE_OFF = NOPE                                   # byte offset of the scales
ROPE_OFF = NOPE + (NOPE // TILE) * 4               # byte offset of the rope


@triton.jit
def _fp8_ds_mla_sparse_decode_kernel(
    q_ptr,            # [seq_q, h_q, NOPE + ROPE]  fp16/bf16
    kv_ptr,           # [num_slots, 656]           uint8  (blocks x block_size flattened)
    indices_ptr,      # [seq_q, kv_groups, topk]   int32, -1 = masked slot
    out_ptr,          # [seq_q, h_q, NOPE]         fp16/bf16
    lse_ptr,          # [seq_q, h_q]               fp32
    seq_q,
    num_slots,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_kv_slot,
    stride_out_token,
    stride_out_head,
    stride_lse_token,
    stride_idx_token,
    stride_idx_head,
    sm_scale,          # already multiplied by log2(e) by the caller
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # geometry as constexpr params: Triton JIT cannot read plain module globals
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    N_TILES: tl.constexpr,
    SCALE_OFF: tl.constexpr,
    ROPE_OFF: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    # int64: with h_q=128, stride_q_token=73728 -> cur_q * stride overflows
    # int32 at ~29k tokens. Mixed-batch mode pushes every batched token through
    # here, so max_num_batched_tokens >= 32k hits it. Same disease we already
    # cured on the kv side; cure it here too.
    cur_q = tl.program_id(1).to(tl.int64)
    cur_head_id = tl.program_id(0)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_d = tl.arange(0, NOPE)       # NoPE dims
    offs_r = tl.arange(0, ROPE)       # RoPE dims

    # q: [BLOCK_H, NOPE] and [BLOCK_H, ROPE]
    q_base = cur_q * stride_q_token + cur_head[:, None] * stride_q_head
    q_nope = tl.load(q_ptr + q_base + offs_d[None, :], mask=mask_h[:, None], other=0.0)
    q_rope = tl.load(q_ptr + q_base + NOPE + offs_r[None, :], mask=mask_h[:, None], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, NOPE], dtype=tl.float32)

    for start_idx in range(0, index_topk, BLOCK_N):
        offs_n = start_idx + tl.arange(0, BLOCK_N)
        idx = tl.load(
            indices_ptr + cur_q * stride_idx_token
            + cur_kv_head_id * stride_idx_head + offs_n,
            mask=offs_n < index_topk,
            other=-1,
        )
        mask_kv = (idx >= 0) & (idx < num_slots)
        # int64: slot * 656 exceeds int32 past ~3.3M slots (a 5M-token pool is
        # routine on this rig). dsa-3090 shipped this bug once — never again.
        row = idx.to(tl.int64) * stride_kv_slot                       # [BLOCK_N]

        # ---- NoPE: raw uint8 bytes -> fp16 (bit-math) -> * per-tile fp32 scale
        # byte address of element d of slot n:  row[n] + d
        kb = tl.load(
            kv_ptr + row[None, :] + offs_d[:, None],
            mask=mask_kv[None, :], other=0,
        )                                                             # [NOPE, BLOCK_N] uint8
        kd = dequant_bitmath_triton(kb).to(tl.float32)                # exact fp8e4m3 -> fp32
        # 4 x fp32 tile scales at byte SCALE_OFF. Load them as [N_TILES, BLOCK_N]
        # (64 values), NOT as a [NOPE, BLOCK_N] gather: the latter is a 128x
        # redundant load that Triton stages in smem as 32 KB/stage and blows
        # sm_86's 100 KB cap. Broadcast over the tile via reshape instead.
        offs_t = tl.arange(0, N_TILES)
        sc = tl.load(
            (kv_ptr + row[None, :] + SCALE_OFF).to(tl.pointer_type(tl.float32))
            + offs_t[:, None],
            mask=mask_kv[None, :], other=0.0,
        )                                                             # [N_TILES, BLOCK_N]
        kd3 = tl.reshape(kd, (N_TILES, TILE, BLOCK_N))
        k_nope = tl.reshape(kd3 * sc[:, None, :], (NOPE, BLOCK_N)).to(q_nope.dtype)

        # ---- RoPE: stored RAW bf16 (never quantized) at byte ROPE_OFF
        k_rope = tl.load(
            (kv_ptr + row[None, :] + ROPE_OFF).to(tl.pointer_type(tl.bfloat16))
            + offs_r[:, None],
            mask=mask_kv[None, :], other=0.0,
        ).to(q_rope.dtype)                                            # [ROPE, BLOCK_N]

        qk = tl.dot(q_nope, k_nope) + tl.dot(q_rope, k_rope)          # [BLOCK_H, BLOCK_N]
        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -float("inf"))

        # ---- online softmax (exp2 form; sm_scale carries the log2(e) factor)
        # NaN GUARD (load-bearing): a BLOCK_N chunk that is fully masked BEFORE
        # any valid chunk leaves the running max at -inf, and exp2(-inf - -inf)
        # = exp2(NaN) = NaN, which then permanently poisons acc — the later
        # `e_sum > 0` guard cannot undo it. Fully-masked chunks are NOT exotic:
        # upstream post-masks empty rows for exactly this reason
        # (flashinfer_mla_sparse.py:523), and the DCP path emits interleaved -1s.
        # Substituting 0.0 for a still--inf max is exact: acc stays 0, e_sum
        # stays 0, LSE stays -inf.
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        safe_max = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp2(e_max - safe_max)
        p = tl.exp2(qk - safe_max[:, None])
        acc *= re_scale[:, None]

        # MLA: V IS the NoPE half of the same rows — reuse, don't re-load.
        v = tl.trans(k_nope)                                          # [BLOCK_N, NOPE]
        acc += tl.dot(p.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # rows that selected nothing: emit zeros, not NaN (now actually true — see
    # the NaN guard above; this alone was never sufficient)
    safe = e_sum > 0
    out = acc / tl.where(safe, e_sum, 1.0)[:, None]
    tl.store(
        out_ptr + cur_q * stride_out_token + cur_head[:, None] * stride_out_head
        + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_h[:, None],
    )
    if WRITE_LSE:
        lse = tl.where(safe, e_max * 0.6931471805599453 + tl.log(e_sum), -float("inf"))
        tl.store(lse_ptr + cur_q * stride_lse_token + cur_head, lse, mask=mask_h)


def fp8_ds_mla_sparse_decode(
    q: torch.Tensor,            # [seq_q, h_q, 576]  fp16/bf16
    kv_cache: torch.Tensor,     # [num_blocks, block_size, 656] uint8 (or fp8-viewed)
    indices: torch.Tensor,      # [seq_q, kv_groups, topk] int32; -1 = masked
    sm_scale: float,
    return_lse: bool = False,
    block_n: int = 16,
    block_h: int = 16,
    num_stages: int = 2,        # 3 (Triton default) overflows sm_86's 100 KB smem
    num_warps: int = 4,
):
    """Sparse MLA decode reading fp8_ds_mla pages, dequantized in-register.

    `indices` are FLAT SLOT indices into the paged cache (block_idx *
    block_size + offset) — the same contract vLLM's sparse backends use after
    the block-table transform. Returns `out` [seq_q, h_q, 512] (+ lse).
    """
    assert kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn), kv_cache.dtype
    # reshape() on a non-contiguous pool would silently COPY the whole cache
    # every call; and the kernel hard-assumes unit last-dim strides.
    assert kv_cache.is_contiguous(), "kv_cache must be contiguous"
    assert q.stride(-1) == 1 and indices.stride(-1) == 1, "need unit last-dim strides"
    # rope is bitcast as bf16 (the writer stores it as the model dtype); an fp16
    # model would make that bitcast read garbage. Also keeps us clear of the
    # Triton-3.6 fp16 tl.dot miscompile on RTX 3090 (triton #9830).
    # fp32 is additionally allowed: it is numerically safe (bf16 rope widens to
    # fp32 exactly) and it is the only dtype the CPU Triton interpreter can do
    # arithmetic in, which is what makes this kernel testable without a GPU.
    assert q.dtype in (torch.bfloat16, torch.float32), (
        f"rope is stored bf16; q must be bf16 (fp32 allowed for testing), got {q.dtype}"
    )

    kv_u8 = kv_cache.view(torch.uint8).reshape(-1, ROW_BYTES)
    num_slots = kv_u8.shape[0]

    seq_q, h_q, dim_qk = q.shape
    assert dim_qk == NOPE + ROPE, f"expected {NOPE + ROPE}, got {dim_qk}"
    if indices.dim() == 2:
        indices = indices.unsqueeze(1)          # [seq_q, 1, topk]
    kv_groups, topk = indices.shape[1], indices.shape[2]
    assert h_q % kv_groups == 0, f"h_q={h_q} not divisible by kv_groups={kv_groups}"

    out = torch.empty((seq_q, h_q, NOPE), dtype=q.dtype, device=q.device)
    lse = torch.empty((seq_q, h_q), dtype=torch.float32, device=q.device)

    kv_group_num = h_q // kv_groups
    # grid axis 0 = head-blocks so that sibling CTAs (same token, different
    # heads) launch adjacently and re-hit the same gathered KV rows in L2;
    # the reverse order re-streams ~1.3 MB per head-block from HBM.
    grid = (triton.cdiv(kv_group_num, block_h) * kv_groups, seq_q)

    _fp8_ds_mla_sparse_decode_kernel[grid](
        q, kv_u8, indices, out, lse,
        seq_q, num_slots, h_q,
        q.stride(0), q.stride(1),
        kv_u8.stride(0),
        out.stride(0), out.stride(1),
        lse.stride(0),
        indices.stride(0), indices.stride(1),
        sm_scale * 1.4426950408889634,          # fold log2(e) for the exp2 softmax
        kv_group_num=kv_group_num,
        index_topk=topk,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        NOPE=NOPE,
        ROPE=ROPE,
        TILE=TILE,
        N_TILES=NOPE // TILE,
        SCALE_OFF=SCALE_OFF,
        ROPE_OFF=ROPE_OFF,
        WRITE_LSE=return_lse,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return (out, lse) if return_lse else out
