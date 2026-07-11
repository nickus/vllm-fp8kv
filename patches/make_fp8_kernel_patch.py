#!/usr/bin/env python3
# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Generate the fp8-KV patch for vLLM's TRITON_MLA_SPARSE kernel (PR #47629).

Design decision (D2, recorded): rather than ship a *competing* sparse-MLA
kernel, we add an `IS_FP8` branch to upstream's `_sparse_mla_compute_tile` —
the single function through which BOTH their single-pass and split-KV kernels
load KV. That means fp8 inherits, for free and by construction:

  * their split-KV parallelism (`_sparse_mla_kernel_split` + merge kernel) —
    without which a bs=1 decode runs on ONE CTA of 82 SMs (measured: our
    standalone kernel was 0.4x their bf16 at topk=2048 purely from occupancy),
  * their `@triton.autotune` configs and CUDA-graph-safe warmup,
  * their masking/NEG_LARGE softmax sentinel and index conversion.

and the upstream diff stays small and reviewable: one flag, one branch, one
dispatcher arg.

The fp8 branch loads the standard `fp8_ds_mla` 656-byte page as raw uint8 and
decodes in-register — Triton is never asked to convert `fp8e4nv`, which is the
whole reason sm_80/86 was excluded (#47629: "Triton fp8e4nv store ... does not
compile on SM80").

Usage:  python patches/make_fp8_kernel_patch.py <path-to-vllm-src>
Writes: patches/triton_mla_sparse_fp8.patch  (git-applyable against #47629)
"""

import os
import subprocess
import sys

TARGET = "vllm/v1/attention/ops/triton_mla_sparse_kernel.py"

# --- 1. imports + fp8 geometry -------------------------------------------------
IMPORT_ANCHOR = "from vllm.utils.platform_utils import num_compute_units"
IMPORT_NEW = """from vllm.utils.platform_utils import num_compute_units

# fp8_ds_mla page geometry (csrc/libtorch_stable/cache_kernels.cu:446):
#   [   0 : 512 )  512 x fp8e4m3  NoPE latent
#   [ 512 : 528 )    4 x fp32     per-128-element tile scales (amax / 448)
#   [ 528 : 656 )   64 x bf16     RoPE, stored RAW (never quantized)
# tl.constexpr so the @triton.jit bodies below can read them (Triton forbids
# plain module globals inside kernels).
_FP8_ROW_BYTES = tl.constexpr(656)
_FP8_SCALE_OFF = tl.constexpr(512)
_FP8_ROPE_OFF = tl.constexpr(528)
_FP8_TILE = tl.constexpr(128)
_FP8_N_TILES = tl.constexpr(4)
_FP8_ROW_BYTES_PY = 656   # plain int for host-side shape math


@triton.jit
def _decode_fp8e4m3(b):
    \"\"\"fp8e4m3fn bytes -> fp16, in-register, without Triton's fp8e4nv.

    The fp8 payload dropped into the fp16 bit layout is off by a fixed 2^8 for
    normals AND denormals alike, so one bitcast + one exact fp16 multiply
    decodes it; 0x7F/0xFF (fp8e4m3*fn* has no inf) are NaN and need a select.
    Bit-exact vs torch.float8_e4m3fn over all 256 byte values.
    \"\"\"
    u = b.to(tl.uint16)
    sgn = (u & 0x80) << 8
    mag = (u & 0x7F) << 7
    val = (sgn | mag).to(tl.float16, bitcast=True) * 256.0
    nan = (sgn | 0x7F80).to(tl.float16, bitcast=True)
    return tl.where((u & 0x7F) == 0x7F, nan, val.to(tl.float16))"""

# --- 2. compute-tile signature: add the IS_FP8 flag ---------------------------
SIG_ANCHOR = """    BLOCK_DPE: tl.constexpr,
):
    \"\"\"Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators.\"\"\""""
SIG_NEW = """    BLOCK_DPE: tl.constexpr,
    IS_FP8: tl.constexpr = False,
):
    \"\"\"Shared stage-1 body: load Q, run the sparse online-softmax loop over
    `[split_start, split_end)` of the topk axis, return accumulators.

    With IS_FP8, `k_buffer` is the standard `fp8_ds_mla` cache as raw uint8
    (656 B/token) and is decoded in-register — no fp8e4nv conversion, so this
    path compiles and runs on sm_80/sm_86 where native fp8 does not exist.\"\"\""""

# --- 3. the loads: branch bf16 vs fp8 ------------------------------------------
LOADS_ANCHOR = """        offs_k = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        offs_kpe = (
            indices[None, :] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dpe[:, None]
        )
        kpe = tl.load(
            k_buffer + offs_kpe,
            mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
            other=0.0,
        )
        qk += tl.dot(qpe, kpe.to(q.dtype))"""
LOADS_NEW = """        if IS_FP8:
            # int64: slot * 656 exceeds int32 past ~3.3M slots, which a 24 GB
            # fp8 pool reaches (26 M tokens/16 GiB).
            row = indices.to(tl.int64) * _FP8_ROW_BYTES
            kb = tl.load(
                k_buffer + row[None, :] + offs_d[:, None],
                mask=mask_kv[None, :],
                other=0,
            )
            kd = _decode_fp8e4m3(kb).to(tl.float32)
            # 4 fp32 tile scales per row; broadcast over the 128-element tile
            # (gathering them per-element would stage 32 KB/stage of smem and
            # blow sm_86's 100 KB cap).
            sc = tl.load(
                (k_buffer + row[None, :] + _FP8_SCALE_OFF).to(
                    tl.pointer_type(tl.float32)
                )
                + tl.arange(0, _FP8_N_TILES)[:, None],
                mask=mask_kv[None, :],
                other=0.0,
            )
            kd3 = tl.reshape(kd, (_FP8_N_TILES, _FP8_TILE, BLOCK_N))
            k = tl.reshape(kd3 * sc[:, None, :], (BLOCK_DMODEL, BLOCK_N)).to(q.dtype)
            qk = tl.dot(q, k)

            # RoPE half is stored RAW bf16 — never quantized, never scaled
            kpe = tl.load(
                (k_buffer + row[None, :] + _FP8_ROPE_OFF).to(
                    tl.pointer_type(tl.bfloat16)
                )
                + tl.arange(0, BLOCK_DPE)[:, None],
                mask=mask_kv[None, :],
                other=0.0,
            ).to(q.dtype)
            qk += tl.dot(qpe, kpe)
        else:
            offs_k = (
                indices[None, :] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_d[:, None]
            )
            k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
            qk = tl.dot(q, k.to(q.dtype))

            offs_kpe = (
                indices[None, :] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))"""

# --- 4. V load: MLA shares K/V, so fp8 reuses the decoded NoPE block -----------
V_ANCHOR = """        offs_v = (
            indices[:, None] * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)"""
V_NEW = """        if IS_FP8:
            # MLA: V IS the NoPE half of the same rows — reuse the block we
            # already decoded instead of re-reading and re-decoding it.
            v = tl.trans(k)
        else:
            offs_v = (
                indices[:, None] * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dv[None, :]
            )
            v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)"""


def _thread_is_fp8(s: str) -> str:
    """Add IS_FP8 to both calling kernels' signatures and their compute_tile
    calls. Both call sites end the arg list with `BLOCK_DPE,` — anchor on that.
    """
    # 1) the two @triton.jit kernels each declare `BLOCK_DPE: tl.constexpr,`
    #    in their signature; give them IS_FP8 too.
    n_sig = s.count("    BLOCK_DPE: tl.constexpr,\n")
    if n_sig < 3:  # compute_tile (already done) + final + split
        sys.exit(f"expected >=3 'BLOCK_DPE: tl.constexpr,' decls, found {n_sig}")
    s = s.replace(
        "    BLOCK_DPE: tl.constexpr,\n",
        "    BLOCK_DPE: tl.constexpr,\n    IS_FP8: tl.constexpr,\n",
    )
    # compute_tile already has its own IS_FP8 (with a default) from SIG_NEW —
    # remove the duplicate the blanket replace just added to it.
    s = s.replace(
        "    BLOCK_DPE: tl.constexpr,\n    IS_FP8: tl.constexpr,\n"
        "    IS_FP8: tl.constexpr = False,\n",
        "    BLOCK_DPE: tl.constexpr,\n    IS_FP8: tl.constexpr = False,\n",
        1,
    )
    # 2) both `_sparse_mla_compute_tile(...)` calls end with `        BLOCK_DPE,`
    n_call = s.count("        BLOCK_DPE,\n    )")
    if n_call != 2:
        sys.exit(f"expected exactly 2 compute_tile call sites, found {n_call}")
    s = s.replace(
        "        BLOCK_DPE,\n    )",
        "        BLOCK_DPE,\n        IS_FP8,\n    )",
    )
    return s


# --- 5. autotuning: NO key change is needed (2026-07-11 review) ---------------
# An earlier revision added IS_FP8 to the @triton.autotune keys, believing the
# fp8 path would otherwise inherit bf16's cached config. That was wrong twice
# over: (a) IS_FP8 is a tl.constexpr, so each value compiles a SEPARATE kernel
# specialization with its own autotune cache; (b) Triton >= 3.x appends every
# tensor argument's dtype to the autotune cache key anyway
# (triton/runtime/autotuner.py, `key.append(str(arg.dtype))`), and the fp8
# cache arrives as uint8 vs bf16.
#
# The REAL open item is the config LISTS: upstream's _FINAL_AUTOTUNE_CONFIGS
# (BLOCK_N=16 only) and _SPLIT_AUTOTUNE_CONFIGS (BLOCK_N=32 only) were chosen
# for bf16's 1152 B rows; whether fp8's 656 B rows + scale load want larger
# blocks/more warps is unmeasured on the real kernel — extending those lists
# is deferred until measured on hardware (see RESULTS.md).

DISPATCH_ANCHOR = """    assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK"""
DISPATCH_NEW = """    # fp8_ds_mla: the cache arrives as raw uint8 pages (656 B/token) instead of
    # bf16 [seq_kv, 1, 576]. Decoded in-register by the kernel — no fp8e4nv, so
    # this works on sm_80/sm_86 where native fp8 conversion does not exist.
    is_fp8 = kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    if is_fp8:
        kv = kv.view(torch.uint8).reshape(-1, 1, _FP8_ROW_BYTES_PY)
        assert kv.shape[2] == _FP8_ROW_BYTES_PY
    else:
        assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/root/vllm-src"
    path = os.path.join(src, TARGET)
    orig = open(path).read()
    s = orig

    for anchor, new, name in [
        (IMPORT_ANCHOR, IMPORT_NEW, "imports+decoder"),
        (SIG_ANCHOR, SIG_NEW, "IS_FP8 flag"),
        (LOADS_ANCHOR, LOADS_NEW, "K/RoPE loads"),
        (V_ANCHOR, V_NEW, "V reuse"),
        (DISPATCH_ANCHOR, DISPATCH_NEW, "dispatcher fp8 detect"),
    ]:
        if anchor not in s:
            sys.exit(f"ANCHOR NOT FOUND ({name}) — upstream drifted; re-derive")
        s = s.replace(anchor, new, 1)
    s = _thread_is_fp8(s)
    # the two kernel launches must pass IS_FP8=is_fp8
    n_launch = s.count("        BLOCK_DPE=_BLOCK_DPE,")
    if n_launch != 2:
        sys.exit(f"expected exactly 2 kernel launches, found {n_launch}")
    s = s.replace("        BLOCK_DPE=_BLOCK_DPE,", "        BLOCK_DPE=_BLOCK_DPE,\n        IS_FP8=is_fp8,")

    open(path, "w").write(s)
    diff = subprocess.run(["git", "-C", src, "diff", "--", TARGET],
                          capture_output=True, text=True).stdout
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "triton_mla_sparse_fp8.patch")
    if not diff.strip():
        # `src` is not a git worktree (e.g. site-packages), so git produced
        # nothing. The file above IS patched, but overwriting the shipped diff
        # with an empty one would silently destroy the artifact.
        sys.exit(f"patched {path}, but `git diff` in {src} is empty — not a git "
                 f"checkout? REFUSING to overwrite {out}. To apply the shipped "
                 f"patch to a non-git tree use: patch -p1 -d {src} < {out}")
    open(out, "w").write(diff)
    print(f"patched {path}\nwrote {out} ({len(diff.splitlines())} diff lines)")


if __name__ == "__main__":
    main()
