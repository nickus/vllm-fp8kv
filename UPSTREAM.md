# Upstream package

## Filed

| # | What | Status |
|---|---|---|
| [#48364](https://github.com/vllm-project/vllm/issues/48364) | **Bug:** fully-masked leading index chunks NaN-poison `xpu_mla_sparse` (`exp2(-inf − -inf)`), silently corrupting attention output for any row whose first 16 topk entries are all masked | filed |
| [#48366](https://github.com/vllm-project/vllm/pull/48366) | **Fix PR:** finite sentinel + regression test; verified bidirectionally (fails on the unfixed kernel, passes on the fixed one) | filed |
| [#48374](https://github.com/vllm-project/vllm/issues/48374) | **RFC:** fp8 KV cache for the Ampere sparse-MLA path via software dequant — with the honest 0.92×/0.45× decode table and the two open items (backend wiring, engine boot) stated up front | filed |

> **Rewritten 2026-07-11** after the adversarial audit
> ([REVIEW-2026-07-11.md](REVIEW-2026-07-11.md)). The previous version of this
> file proposed a second contribution — a "dtype-blind autotune key" bug report
> — which was **withdrawn before filing: its central mechanism is false.** See
> below.

---

## Withdrawn: "sparse-MLA autotune key is blind to the KV dtype"

The claim was that upstream's `key=["index_topk", "kv_group_num"]` would make
a second KV dtype silently inherit bf16's cached config. **False on the
pinned stack, twice over:**

1. Triton ≥ 3.x appends `str(arg.dtype)` of every tensor argument to the
   autotune cache key (`triton/runtime/autotuner.py`); a uint8 fp8 cache and a
   bf16 cache can never share a cache entry.
2. Our `IS_FP8` is a `tl.constexpr`, which compiles a separate kernel
   specialization with its own autotune cache regardless.

The 3.0× (608 → 201 µs) evidence compared two *different* kernels, one of
which had an invalid split-softmax merge (see RESULTS.md corrections). The
salvageable, honest observation — worth a note in the eventual RFC, not a bug
report — is that upstream's autotune **config lists** are bf16-oriented
(`_FINAL_AUTOTUNE_CONFIGS`: BLOCK_N=16 only; `_SPLIT_AUTOTUNE_CONFIGS`:
BLOCK_N=32 only) and an fp8 path may want its own entries. Unmeasured on the
real kernel as of this writing.

---

## Filed: fp8 KV cache on sm_80 / sm_86 (`fp8_ds_mla`) — RFC #48374

**Patch:** `patches/triton_mla_sparse_fp8.patch` (224 lines; one flag, one
branch, one dispatcher arg). Regenerates **byte-identically** from
`patches/make_fp8_kernel_patch.py` against PR #47629 head `bbe2ab4d6`,
`git apply --check`s cleanly, and the result byte-compiles.

### The gap, in upstream's own words (verified verbatim, safe to cite)

`TRITON_MLA_SPARSE` declares `supported_kv_cache_dtypes = ["auto", "float16",
"bfloat16"]`, its decode entry point is literally `_forward_bf16_kv`, and
`forward_mqa` opens with:

```python
if is_quantized_kv_cache(self.kv_cache_dtype):
    raise NotImplementedError("FP8 kv is not supported with XPU MLA Sparse yet")
```

The stated reason for excluding fp8 on Ampere is that *"Triton fp8e4nv store …
does not compile on SM80"*. Meanwhile `FLASHINFER_MLA_SPARSE_SM120` **requires**
fp8 KV. On 24 GB cards this is the binding constraint: bf16 costs 1152 B/token
vs fp8_ds_mla's 656 B — 1.756× of KV capacity left on the table (format
arithmetic; ≈1.63× effective once the indexer K-cache is counted).

### The proposal: fp8-as-storage ≠ native fp8 compute (safe to cite)

We never ask Triton to convert `fp8e4nv`. The standard `fp8_ds_mla` pages are
loaded as raw `uint8` and decoded in-register with bit-math that is **bit-exact
against `torch.float8_e4m3fn` over all 256 byte values**. Nothing about the
cache layout changes; the existing C++ `concat_and_cache_ds_mla` writer already
runs on sm_86 unmodified (`ENABLE_FP8` is CUDA-version-gated, not arch-gated).

Merged PR #43914 and open PR #47060 enforce "native FP8 requires SM89+" —
correctly, for native conversion; software dequant is a different mechanism.
Precedent already in-tree: `models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`
(dequant-on-the-fly, identical architecture, XPU). Note for framing: PR #47629
itself ships a uint8-LUT software-fp8 decode on SM80 in its indexer
(`mqa_logits_triton.py`) — the technique is already accepted *inside this very
PR*; our contribution extends it to the KV path.

### Reviewer notes that remain valid (safe to cite)

* `fp8_ds_mla` carries its scales **inline, per 128-element tile**; the layer's
  `_k_scale` is threaded into the writer but that kernel ignores it. Our decode
  therefore applies only the inline tile scales. (Verified with a tripwire:
  `k_scale=2.0` into the writer leaves the roundtrip unaffected.)
* The RoPE half of the row is **raw bf16, never quantized** — it must not be
  scaled.
* Slot offsets are computed in **int64**: `slot × 656` exceeds int32 past
  ~3.3 M slots, which a 24 GB fp8 pool reaches.

### What the RFC states honestly (and what is still open)

The RFC leads with the decode cost rather than burying it: **0.92× at bs=8,
0.45× at bs=32** vs bf16 on the same kernel — a capacity-for-speed trade, not a
free win. The config-list hypothesis (upstream's `BLOCK_N=16/32`-only autotune
lists are bf16-shaped) is flagged **as a hypothesis, not a finding**, because it
has not been measured on the real kernel.

Open items, both stated in the RFC:

1. **Backend wiring** — the shipped patch is kernel-only and inert alone
   (`supported_kv_cache_dtypes`, the 656-B `get_kv_cache_shape`, and
   `_canonicalize_sparse_mla_kv_cache_dtype`). Working out-of-tree in
   `vllm_fp8kv/backend_patch.py` (~40 lines, now tested).
2. **Engine boot** — no `LLM(kv_cache_dtype="fp8_e4m3")` run yet; all
   correctness is kernel-level (0.999996 through vLLM's own writer + converter).

Both are gated on the next GPU session, and on upstream saying the feature is
wanted at all.

### Provenance

The software-dequant technique comes from
[nickus/dsa-3090](https://github.com/nickus/dsa-3090) (Apache-2.0), the sm_86
DSA port, itself derived from renning22/glm-5.2-4090.
