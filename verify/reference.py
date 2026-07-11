# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Golden references for fp8_ds_mla sparse MLA — fp32 math, runs anywhere.

`ref_dequant_fp8_ds_mla_row` is deliberately written to mirror vLLM's OWN
golden parser (tests/v1/attention/test_sparse_mla_backends.py:84
`_dequantize_fp8_ds_mla_entry`), so our kernel is verified against upstream's
definition of the format rather than our own re-reading of it.

Layout (656 B/token), per csrc/libtorch_stable/cache_kernels.cu:446:
    [  0:512)  512 x fp8e4m3  NoPE latent
    [512:528)    4 x fp32     per-128-tile scales (amax/448)
    [528:656)   64 x bf16     RoPE, RAW (never quantized)
"""

import torch

NOPE, ROPE, TILE = 512, 64, 128
ROW_BYTES = 656


def ref_dequant_fp8_ds_mla_row(rows_u8: torch.Tensor) -> torch.Tensor:
    """(..., 656) uint8 -> (..., 576) fp32 [NoPE dequantized | RoPE raw]."""
    flat = rows_u8.reshape(-1, ROW_BYTES)
    n = flat.shape[0]
    n_tiles = NOPE // TILE

    nope_q = flat[:, :NOPE].view(torch.float8_e4m3fn).float().reshape(n, n_tiles, TILE)
    scales = flat[:, NOPE : NOPE + n_tiles * 4].contiguous().view(torch.float32).reshape(n, n_tiles, 1)
    nope = (nope_q * scales).reshape(n, NOPE)          # dequant = fp8 * scale

    rope = flat[:, NOPE + n_tiles * 4 :].contiguous().view(torch.bfloat16).float().reshape(n, ROPE)
    return torch.cat([nope, rope], dim=-1).reshape(*rows_u8.shape[:-1], NOPE + ROPE)


def ref_sparse_mla_decode(
    q: torch.Tensor,          # [seq_q, h_q, 576]
    kv_u8: torch.Tensor,      # [num_slots, 656] uint8
    indices: torch.Tensor,    # [seq_q, topk] int, -1 = masked
    sm_scale: float,
) -> torch.Tensor:
    """fp32 reference: dequantize, then dense-attend over the selected slots.

    MLA: K = full 576 (nope+rope), V = the NoPE half (first 512) of the same
    rows. Rows selecting nothing produce zeros (not NaN).
    """
    seq_q, h_q, _ = q.shape
    rows = kv_u8.reshape(-1, ROW_BYTES)
    n_slots = rows.shape[0]
    qf = q.float()
    out = torch.zeros(seq_q, h_q, NOPE, dtype=torch.float32, device=q.device)

    for t in range(seq_q):
        sel = indices[t]
        sel = sel[(sel >= 0) & (sel < n_slots)].long()
        if sel.numel() == 0:
            continue
        # gather THEN dequantize: dequantizing the whole pool first would
        # materialize num_slots x 576 fp32 (8 GB at a 3.5M-slot pool) — the
        # reference must not be more expensive than the kernel it checks.
        k = ref_dequant_fp8_ds_mla_row(rows[sel]).float()   # [S, 576]
        scores = (qf[t] @ k.T) * sm_scale               # [h_q, S]
        w = torch.softmax(scores, dim=-1)
        out[t] = w @ k[:, :NOPE]                        # V = NoPE half
    return out


def quantize_to_fp8_ds_mla(kv: torch.Tensor) -> torch.Tensor:
    """fp32/bf16 (..., 576) -> packed (..., 656) uint8, mirroring the C++
    writer (`concat_and_cache_ds_mla_kernel`): per-128-tile amax/448 scales on
    the NoPE half; RoPE copied through as raw bf16.

    Used only to BUILD test data; the kernel under test never calls it. The
    round-trip through `ref_dequant_fp8_ds_mla_row` (an independent parser)
    is what makes a symmetric pack/parse bug unable to hide.
    """
    flat = kv.reshape(-1, NOPE + ROPE).float()
    n = flat.shape[0]
    n_tiles = NOPE // TILE

    tiles = flat[:, :NOPE].reshape(n, n_tiles, TILE)
    scale = (tiles.abs().amax(dim=-1, keepdim=True) / 448.0).clamp(min=1.1754944e-38)
    q8 = (tiles / scale).to(torch.float8_e4m3fn)

    out = torch.empty(n, ROW_BYTES, dtype=torch.uint8, device=kv.device)
    out[:, :NOPE] = q8.view(torch.uint8).reshape(n, NOPE)
    out[:, NOPE : NOPE + n_tiles * 4] = (
        scale.reshape(n, n_tiles).contiguous().view(torch.uint8).reshape(n, -1))
    out[:, NOPE + n_tiles * 4 :] = (
        flat[:, NOPE:].to(torch.bfloat16).view(torch.uint8).reshape(n, -1))
    return out.reshape(*kv.shape[:-1], ROW_BYTES)
