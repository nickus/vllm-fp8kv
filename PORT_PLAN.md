# Port plan — fp8-KV for vLLM's Ampere DSA path

Derived from a 4-agent code audit of vLLM main @ `1bd8f80a` (2026-07-11) and
the `dsa-3090` codebase. **The headline finding rewrites the scope.**

## The scope-changing discovery

**vLLM already has the exact byte layouts `dsa-3090` implements.** Both derive
from DeepSeek's reference, so they are byte-for-byte identical:

| | dsa-3090 (sglang) | vLLM |
|---|---|---|
| MLA KV row | 656 B: `[512 fp8 nope][16 B = 4×fp32 per-128-tile scales][128 B = 64×bf16 rope]` | **identical** — cache dtype `fp8_ds_mla`, `csrc/libtorch_stable/cache_kernels.cu:446` `concat_and_cache_ds_mla_kernel`, shape `(num_blocks, block_size, 656)` |
| Indexer K page | SoA: `[block×128 fp8][block×fp32]` = 8448 B @ page 64 | **identical** — `indexer_k_quant_and_cache_kernel`, cache_kernels.cu:549 |
| Tile scale | `amax / 448` | **identical** — `kFp8ScaleDivisor = 448.f`, cache_kernels.cu:28 |
| Dequant direction | `fp8 * scale` | **identical** — quant_utils.cuh:297 |

**Consequence: the entire cache-WRITE half of dsa-3090 is NOT needed.** vLLM's
C++ writers already produce exactly the pages our decode kernel knows how to
read, and they **already compile and run on sm_86** (`ENABLE_FP8` is gated on
CUDA ≥ 11.8, *not* on arch — `cmake/utils.cmake:136`; the intrinsic is only
guarded against `__CUDA_ARCH__ < 800`). Do **not** port `fp8_quant.py`,
`quantize_k_cache_sm86`, or `act_quant_sm86`. *(Must confirm on hardware:
`__nv_cvt_float_to_fp8` on sm_86 — first bench-box task.)*

**And there is no vLLM policy against fp8 KV on Ampere.** `supports_fp8()`
(=`has_device_capability(89)`, cuda.py:555) is never consulted for the KV
cache. The only hard refusal lives in the non-MLA `TRITON_ATTN` backend
(triton_attn.py:501) and it is a *Triton* limitation — `fp8e4nv` needs SM89 —
not a design decision. **That is exactly the wall we already broke.**

So the project reduces to: **make the sm_8x sparse-MLA decode + indexer
kernels READ fp8 pages using software dequant, and declare the dtype.**

## Work items

| D | Was | Now |
|---|---|---|
| **D1 cache write** | port dsa-3090 write path | **~free**: declare `fp8_ds_mla` in the new backend's `supported_kv_cache_dtypes` + `_canonicalize_sparse_mla_kv_cache_dtype` (mla_attention.py:323). Verify the C++ writer on sm_86. |
| **D2 decode reads fp8** | the real work | Write a Triton sparse-MLA decode kernel that dequants 656-B pages in-register. Template: `vllm/v1/attention/ops/xpu_mla_sparse.py::_bf16_mla_sparse_kernel` (the tree's only pure-Triton sparse-MLA decode). Dequant: `dsa-3090/sm86_dsa/fp8_dequant.py::dequant_bitmath_triton` (ports verbatim — zero sglang coupling). tilelang is NOT a vLLM dep, so the dsa-3090 decode *route* is unportable; its *semantics* are the spec. |
| **D3 indexer** | fp8 vs bf16 decision | Indexer cache is fp8 **by model design** (independent of `--kv-cache-dtype`). #47629's Triton MQA-logits does `tl.dot` on fp8 operands → won't compile on sm_80/86. Fix with our uint8-load + bit-math decode (dsa-3090's `_fp8_mqa_logits_kernel_sm86` / `_paged_idx_kernel_sm86` are the spec). |
| **D4 rig combo** | TP2+DP+PP+MTP+graph | Same gates as before, plus **next_n>1** (see traps). |
| **D5 harness** | adapt | `verify/` ports nearly as-is — **its layout constants ARE vLLM's constants**. vLLM's own golden parser (`tests/v1/attention/test_sparse_mla_backends.py:84` `_dequantize_fp8_ds_mla_entry`) matches `verify/reference.py:140` functionally. |

## Base branch

Fork **#47629's head** (`triton-mla-sparse-sm80`, MERGEABLE, 2026-07-07) — not
#38476 (CONFLICTING/needs-rebase, author inactive). Re-check merge status each
session; upstream moves fast.

## Traps (each one is a silent-corruption class)

1. **Double-applied scale.** `is_quantized_kv_cache('fp8_ds_mla')` is True, so
   `_k_scale` is loaded from checkpoint and threaded into
   `concat_and_cache_mla(..., scale=k_scale)` — where **the ds_mla kernel
   ignores it** (cache_kernels.cu:460 declares it, never dereferences it) and
   computes its own per-tile scales. Our decode must apply **only** the inline
   per-tile scales. Applying `_k_scale` too = silent quality loss.
2. **RoPE is NOT quantized** in `fp8_ds_mla` (bytes [528:656) are raw bf16).
   Treating the row as all-fp8, or scaling the rope half, is wrong. (Plain
   `fp8`/`fp8_e4m3` MLA *does* quantize rope — different dtype, different rule.)
3. **Indexer-cache doc is wrong.** `vllm/utils/deep_gemm.py:586` documents the
   paged indexer cache as **AoS** ("last 4 bytes per (block,pos)"); the C++
   writer (cache_kernels.cu:598) is **SoA**. Coding from the doc reproduces
   exactly the >2048-context corruption renning22 hit. Trust the writer.
4. **next_n > 1 (MTP).** No dsa-3090 kernel handles it — `fp8_paged_mqa_logits`
   does `q_fp8[:, 0]` (sm86_dsa.py:160), silently dropping draft tokens. AC4
   requires MTP, so this must be built in from the start, not retrofitted.
5. **SMEM overflow.** The template kernel launches with Triton's default
   `num_stages=3` → ~104,448 B > sm_86's 101,376 B cap at DSA dims. Pass
   `num_stages=2` explicitly. (Also: `acc[16,512]` fp32 = 64 regs/thread →
   ~1 block/SM; expect occupancy pressure.)
6. **Don't reach for `tl.float8e4b15`.** vLLM's TurboQuant uses it on SM<8.9 —
   it is a *different bit layout* (the ktransformers#1999 alias trap in new
   clothes). Software bit-math decode is the only correct route.
7. **CUDA-graph**: declare `UNIFORM_BATCH` support, not
   `UNIFORM_SINGLE_TOKEN_DECODE` (gpu_model_runner takes the min across groups
   and silently downgrades). The pure-torch fallbacks in-tree call `.item()` in
   a per-batch loop — not capturable; do not lift them.
8. **deep_gemm hard-raise**: `SparseAttnIndexer.__init__` raises if
   `not has_deep_gemm()` on CUDA (sparse_attn_indexer.py:728). Same class as
   the sglang shim; vLLM's interception point is cleaner.
9. **Triton 3.6 fp16 `tl.dot` miscompile on RTX 3090** (triton #9830). Gate
   every block-size/autotune change with the bit-exactness harness.

## Upstream framing (AC7)

Merged PR **#43914** already hard-codes "fp8 KV requires SM89+" for the Triton
*attention* path, and open PR **#47060** would mirror that into Triton *MLA*.
Our RFC must therefore draw the distinction explicitly:

> **fp8-as-storage with software dequant ≠ native fp8e4nv compute.**
> We do not ask Triton to convert `fp8e4nv` on sm_8x. We store the standard
> `fp8_ds_mla` pages, load them as `uint8`, and decode in-register with
> bit-math that is bit-exact against `torch.float8_e4m3fn`.

Precedent already in-tree and accepted: `vllm/models/deepseek_v4/xpu/
xpu_sparse_decode_fp8.py` does exactly this ("dequantize FP8 KV cache pages to
BF16 on the fly, then reuse the BF16 sparse MLA attention kernel… keeps the
external KV cache layout identical to CUDA/ROCm") — but only for XPU. We bring
the same architecture to CUDA sm_8x.

Demand is documented: a 2026-07-07 field report on #47629 runs GLM-5.2 on a
fleet including **8× RTX 3090** and hits the bf16-KV ceiling (1152 vs 656
B/token = **1.76× capacity left on the table**); a 2026-04-20 comment on
#38476 reports OOM at 128 K context on 8×A100-80G with bf16 KV.

## Bonus finding (report upstream regardless)

`TRITON_MLA` (dense) declares `supported_kv_cache_dtypes = [... 'fp8',
'fp8_e4m3']` **and** `supports_compute_capability → True` (triton_mla.py:85,
131) — so `--kv-cache-dtype fp8` with TRITON_MLA on Ampere is *accepted* today
but its read path lowers to `fp8e4nv` conversion, which Triton rejects on
sm_8x. Likely broken-on-arrival; verify on the bench box and file.
