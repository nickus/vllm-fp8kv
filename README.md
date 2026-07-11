# vllm-fp8kv — fp8 KV cache for vLLM's Ampere DSA path

vLLM's Ampere sparse-MLA backend (`TRITON_MLA_SPARSE`, PR #47629) is
**bf16-KV-only**, because *"Triton fp8e4nv store … does not compile on SM80"*.
That costs 1152 B/token instead of 656 B and is the binding constraint on 24 GB
consumer cards.

This repo removes that constraint **without asking Triton for fp8 at all**: the
standard `fp8_ds_mla` pages are loaded as raw `uint8` and decoded in-register
with bit-math that is bit-exact against `torch.float8_e4m3fn`.

## Status — read this before citing any number

**Verified on RTX 3090 (kernel level):** cosine **0.999996** through vLLM's own
C++ `ds_mla` cache writer → vLLM's own index converter → our fp8 decode kernel;
fp8 dequant bit-exact over all 256 byte values; correct at the int32-overflow
slot boundary; `k_scale` semantics pinned by a tripwire test.

**Not verified:** engine-level boot (`--kv-cache-dtype fp8_e4m3` through a real
`LLM()`) and decode speed. Measured decode on the patched upstream kernel is
**0.45–0.92× of bf16**, not parity.

An independent audit on 2026-07-11 ([REVIEW-2026-07-11.md](REVIEW-2026-07-11.md))
retracted this project's earlier parity and autotune-bug claims; the corrections
are folded into [RESULTS.md](RESULTS.md) and [UPSTREAM.md](UPSTREAM.md). If you
are here for numbers, those two files are the honest ones.

Architecture precedent: vLLM already accepts exactly this approach for XPU
(`models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`: *"dequantize FP8 KV cache
pages to BF16 on the fly, then reuse the BF16 sparse MLA attention kernel"*),
and PR #47629 itself ships a software-fp8 (uint8 LUT) decode on SM80 in its
indexer. We bring the same technique to the KV path on CUDA sm_80/sm_86.

## Layout

```
patches/triton_mla_sparse_fp8.patch   the upstream-able diff (224 lines) —
                                      adds an IS_FP8 branch to upstream's own
                                      _sparse_mla_compute_tile, so fp8 inherits
                                      their split-KV, autotune and merge
patches/make_fp8_kernel_patch.py      re-derives it against a moving upstream
                                      (anchored; fails loudly on drift)
vllm_fp8kv/fp8_dequant.py             the fp8e4m3fn software decode (bit-exact)
vllm_fp8kv/fp8_ds_mla_sparse_decode.py  standalone kernel (reference impl / tests)
vllm_fp8kv/backend_patch.py           backend wiring: dtype declaration, 656-B
                                      cache shape, dtype canonicalization, forward
verify/                               model-free harness: golden refs, contract
                                      tests vs vLLM ITSELF, component checks, bench
REVIEW-2026-07-11.md  RESULTS.md  UPSTREAM.md  PREFLIGHT.md  PORT_PLAN.md
```

## Reproduce

Needs an sm_80/sm_86 GPU box with CUDA 12.x (`-devel` image; `nvcc` on PATH):

```bash
git clone https://github.com/nickus/vllm-fp8kv && cd vllm-fp8kv
./setup.sh          # nightly vLLM + PR #47629 overlay + our patch + harness
```

`setup.sh` pins PR #47629 at head `bbe2ab4d6` and — importantly — applies the
patch to the **installed site-packages copy**, which is what the verify scripts
import. Patching only a git clone leaves the runtime unpatched.

Nightly vLLM is required: the 0.24.0 release lacks the `_C_stable_libtorch`
symbols PR #47629's Python code calls.

To re-derive the patch against a newer PR head:

```bash
python patches/make_fp8_kernel_patch.py /path/to/vllm-git-clone
```

## Provenance

renning22/glm-5.2-4090 → [nickus/dsa-3090](https://github.com/nickus/dsa-3090)
(the sm_86 DSA port; software fp8 dequant originates there) → this repo.
Apache-2.0 throughout; see `NOTICE`.

Filed upstream from this work: vLLM
[#48364](https://github.com/vllm-project/vllm/issues/48364) /
[#48366](https://github.com/vllm-project/vllm/pull/48366) — NaN poisoning in
`xpu_mla_sparse` on fully-masked leading index chunks.
