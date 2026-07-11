# Results — fp8-KV for vLLM's Ampere DSA path

Measured on **RTX 3090 (sm_86)**, driver 580.159, CUDA 12.8, torch 2.13,
Triton 3.7, vLLM nightly + PR #47629 (Python-only overlay, head `bbe2ab4d6`).

> **CORRECTED 2026-07-11** after an independent adversarial audit
> ([REVIEW-2026-07-11.md](REVIEW-2026-07-11.md)), whose four core findings were
> re-verified against primary sources before this rewrite. What changed:
>
> 1. The **decode-parity table (0.85–1.02×) and the 0.999873 figure are
>    retracted.** They were measured on a scratch split-KV kernel whose merge
>    step (`tl.atomic_add` of locally-normalized per-split outputs, no LSE
>    weighting) does not compute a global softmax. Cosine similarity is
>    scale-invariant and the i.i.d. Gaussian test data made per-split means
>    nearly parallel, so the error surfaced only as the drop from 0.999996 to
>    0.999873 — misread at the time as quantization noise.
> 2. The **"upstream autotune key is dtype-blind" bug report is withdrawn.**
>    Triton ≥ 3.x appends every tensor argument's dtype to the autotune cache
>    key (`triton/runtime/autotuner.py`), and `IS_FP8` is a `tl.constexpr`,
>    which specializes the kernel anyway. The 608 → 201 µs table compared two
>    different kernels — one of them the invalid one above — at a config that
>    is not in upstream's config space.
> 3. **AC1 was never verified at engine level.** No vLLM engine has booted
>    with fp8 KV through this patch; all correctness numbers are kernel-level
>    on synthetic data. (The monkeypatch now also covers `get_kv_cache_shape`
>    and dtype canonicalization, which a real boot needs — previously missing.)
> 4. The shipped `.patch` had drifted from its generator (the generator had
>    grown the misguided autotune hunks without regeneration; one of them
>    silently no-opped). The generator has been reverted to match the shipped
>    224-line patch, which regenerates byte-identically and is the correct
>    artifact.

## What is verified (kernel level, on hardware)

**fp8-KV sparse-MLA decode runs on Ampere.** vLLM's `TRITON_MLA_SPARSE` backend
is bf16-KV-only because *"Triton fp8e4nv store … does not compile on SM80"*
(#47629). We removed that constraint by never asking Triton for fp8: the
standard `fp8_ds_mla` pages are loaded as raw `uint8` and decoded in-register
with bit-math that is bit-exact vs `torch.float8_e4m3fn`.

| Check | Result |
|---|---|
| fp8 dequant, all 256 byte values, on device | **bit-exact** |
| Standalone kernel vs fp32 golden ref (topk 2048, 64K-slot pool, 128 heads) | **cosine 0.999996** |
| Decode over pages written by **vLLM's own C++ `ds_mla` writer** | **cosine 0.999996** |
| `k_scale=2.0` passed to their writer (tripwire) | correctly **ignored** → our no-double-scale decision is right |
| Leading-masked chunks (NaN-poisoning case) | finite, 0.999996 |
| 3.5M-slot pool (int32 offset overflow boundary) | **cosine 0.999997** |
| Upstream's patched split-KV kernel + their index converter | cosine 0.999996 — measured by a scratch script on the (destroyed) bench box; **not reproducible from the shipped harness**, re-verification queued |

**vLLM's C++ fp8 cache writer runs on sm_86 unmodified** — verified by
execution (`ENABLE_FP8` is gated on CUDA version, not architecture).

The verify harness (16 checks in the current tree) passed on-device; the raw
logs were lost with the rented boxes, so treat "harness green" as historical
until the queued re-run.

## AC2 — KV pool capacity: format arithmetic, target missed as written

| pool | bf16 KV | fp8 KV |
|---|---|---|
| 16 GiB | 14.91 M tokens | **26.19 M tokens** |
| 20 GiB | 18.64 M tokens | **32.74 M tokens** |

`1152 / 656 = 1.756×`. Two honesty notes the original version of this file
did not make:

* **1.756× is arithmetic, not a measurement** — no engine allocated these
  pools. The AC target of ≥1.9× is arithmetically unreachable for
  `fp8_ds_mla` (512 fp8 + 16 B inline scales + 128 B raw-bf16 RoPE = 656 B vs
  1152 B bf16), so the AC as written is **missed**; 1.756× is the format's
  ceiling.
* At engine level the DSA **indexer K-cache** (~132 B/token, same dtype in
  both configs) dilutes the effective pool gain to **≈1.63×**.

## AC3 — decode throughput: NOT at parity; real numbers below

The only decode timings ever taken on the **real patched upstream kernel**
(scratch `test_patched.py`, commit `12a6a75`, RTX 3090, topk=2048):

| bs | fp8 vs bf16 |
|---|---|
| 8 | **0.92×** |
| 32 | **0.45×** |

AC3 (≥95% @8K, ≥100% @64K) is therefore **not met**, and — a further audit
point — it was never measured on its own axis: the AC specifies context-length
gates, while all timings swept batch size at one fixed pool.

The retracted parity table suggested the gap could be closed with fp8-specific
launch configs (larger BLOCK_N, more warps). That is now an **unverified
hypothesis**: upstream's `_FINAL_AUTOTUNE_CONFIGS`/`_SPLIT_AUTOTUNE_CONFIGS`
lists are bf16-oriented (BLOCK_N=16/32 only), and extending them for fp8's
656-byte rows is plausible but must be measured on the real kernel. Queued for
the next GPU session, before any upstream RFC cites a speed number.

The three ALU-reduction experiments (straight-to-bf16 decode, 256-entry LUT,
coalesced-V) all failed to help in their day-of runs, suggesting the cost is
load-stream- rather than ALU-bound — but the shipped `variants.py` only
exercises two of the five variants, so this too is anecdote, not evidence.

## Honest summary

| AC | Status |
|---|---|
| AC1 fp8 boots + logits parity | ❌ **not verified at engine level** — kernel-level correctness only (0.999996 on synthetic data); no vLLM boot with fp8 KV has run |
| AC2 pool ≥1.9× | ❌ as written (target exceeds the format ceiling). Format arithmetic: **1.756×**; effective engine-level ≈1.63× incl. indexer cache |
| AC3 speed ≥95%/100% | ❌ **0.92× (bs=8) / 0.45× (bs=32)** on the real patched kernel; parity claim retracted; config-tuning hypothesis unmeasured |
| AC5 no non-Ampere regression | ◻ argued statically (`IS_FP8` defaults False, bf16 branch source-identical) — **never tested on non-Ampere hardware** |
| AC7 upstream package | ⚠️ patch (224 lines) regenerates byte-identically and apply-checks vs `bbe2ab4d6`; RFC held back pending re-measurement (see UPSTREAM.md) |
| AC9 pre-flight | ✅ gap real and upstream-declared |
| AC4 (TP+DP+PP+MTP+graph), AC6 (128K), AC8 | not reached |

The question the project set out to answer — *can Ampere read fp8 KV pages
correctly inside vLLM's kernels?* — is answered **yes** at kernel level, with
a verified 0.999996 correctness chain over vLLM's own writer and layout. The
questions that remain open are **engine boot** (AC1) and **decode speed**
(currently 0.45–0.92×, with an untested tuning hypothesis). On a 24 GB card
the capacity-for-speed trade may already be worth it; the honest numbers to
decide with are the ones above.
