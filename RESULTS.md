# Results — fp8-KV for vLLM's Ampere DSA path

Measured on **RTX 3090 (sm_86)**, driver 595, CUDA 12.8, torch 2.11, Triton 3.6,
vLLM 0.25.0 nightly + PR #47629 (head `bbe2ab4d6`, Python-only overlay).

> **2026-07-12 — everything below was re-measured on hardware.** The numbers in
> the version of this file dated 2026-07-11 came partly from a scratch kernel
> whose split-KV merge was invalid (see [REVIEW-2026-07-11.md](REVIEW-2026-07-11.md));
> they are gone. Every timing here comes from **upstream's own patched kernel**
> (its split-KV, its LSE merge, its autotune), and **every timed configuration is
> correctness-gated on cosine AND max_abs** — the missing max_abs gate is exactly
> how an 8×-wrong kernel passed last time.

## AC1 — engine boot + logits parity: **PASS**

The acceptance criterion this project had never actually run. `verify/engine_boot.py`
boots two real vLLM engines on a GlmMoeDsa model and compares them:

| Check | Result |
|---|---|
| Engine boots with `--kv-cache-dtype fp8_e4m3` | **yes**, backend = `TRITON_MLA_SPARSE` |
| KV pages are really fp8 | **656 B/token** (bf16 run: 576 B) |
| Our fp8 decode actually executed | **16 calls**, bf16 decode 0 |
| Baseline really used upstream's bf16 path | 16 calls, fp8 0 |
| Logits parity vs bf16 (teacher-forced) | **cosine 0.999999 / 0.999997** |
| Top-1 agreement (teacher-forced) | **0.80 / 1.00** |

Parity is measured **teacher-forced** — the same prompt positions scored by both
engines. Free-running greedy tokens are *not* a valid gate: on a random-weight
toy the logits are near-flat over a 50k vocab, so a 2 % fp8 perturbation flips a
near-tie, and after one divergence the two runs are scoring different sequences.
(That trap is why the first version of this test "failed" for the wrong reason.)

**Three engine-level bugs had to be fixed to get here**, none of which any
kernel test could have found — see `vllm_fp8kv/backend_patch.py`:

1. **The CLI dtype is never seen by the selector.** vLLM picks the backend with
   the *raw* `--kv-cache-dtype` and canonicalizes afterwards, so a backend that
   declares only `fp8_ds_mla` is never selected: *"No valid attention backend
   found … TRITON_MLA_SPARSE: [kv_cache_dtype not supported]"*. FLASHMLA_SPARSE
   declares `"fp8"` as an alias for exactly this reason; we now do too.
2. **The KV spec claims to be unquantized.** `MLAAttention.get_kv_cache_spec`
   builds `MLAAttentionSpec` without `kv_quant_mode`, so it defaults to `NONE`.
   The pool is then *allocated* from `page_size_bytes` (which does special-case
   fp8_ds_mla at 656 B) but *reshaped* using the unquantized 576-wide shape:
   `RuntimeError: shape '[128811, 64, 576]' is invalid for input of size 5408001024`.
3. **The worker is a separate process.** vLLM V1 runs EngineCore out-of-process,
   so an in-process monkeypatch never reaches it. A real deployment needs a
   plugin entry point or `sitecustomize.py` (the same lesson dsa-3090 learned
   with sglang's subprocess workers).

## Kernel correctness (harness, on-device)

| Check | Result |
|---|---|
| fp8 dequant, all 256 byte values | **bit-exact** vs `torch.float8_e4m3fn` |
| Decode vs fp32 reference (up to 64K pool, topk 2048, 128 heads) | **cosine 0.999996** |
| Over pages written by vLLM's own C++ `ds_mla` writer | **cosine 0.999996** |
| `k_scale=2.0` tripwire | correctly **ignored** by the writer |
| Leading fully-masked chunks (the NaN case) | finite, 0.999996 |
| 3.5 M-slot pool (int32 offset overflow boundary) | **cosine 0.999997** |
| Harness | **16/16, 0 failures** |
| Test suite on GPU | **90 passed**, 91.6 % coverage |

## AC2 — KV pool capacity: 1.756× (format ceiling), ~1.63× effective

`1152 / 656 = 1.756×`. Arithmetic, not a measurement — and the AC target of
≥1.9× is **unreachable** for this format (512 fp8 + 16 B inline scales + 128 B
raw-bf16 RoPE). At engine level the DSA indexer's own K-cache (same size in both
configs) dilutes the gain to **≈1.63×**. Recorded as a **miss against the AC as
written**, with the honest ceiling stated.

## AC3 — decode throughput: **0.84–0.93×** of bf16, and it is not tunable away

Upstream's patched kernel, fp8 vs bf16, only the KV dtype differing:

| context (pool) | bs | fp8 | bf16 | fp8/bf16 | correctness |
|---|---|---|---|---|---|
| 8 K | 1 | 112.8 µs | 104.6 µs | **0.93×** | 0.999996 |
| 16 K | 1 | 123.9 µs | 106.3 µs | 0.86× | 0.999996 |
| 64 K | 1 | 123.1 µs | 103.9 µs | 0.84× | 0.999996 |
| 128 K | 1 | 125.0 µs | 108.1 µs | 0.87× | 0.999996 |
| 64 K | 8 | 567.7 µs | 310.7 µs | 0.55× | 0.999996 |
| 64 K | 32 | 1471 µs | 1293 µs | 0.88× | 0.999996 |
| 64 K | 64 | 3017 µs | 2591 µs | 0.86× | 0.999996 |

AC3 asked for ≥95 % @8K and ≥100 % @64K: **missed** (0.93× and 0.84×). fp8 costs
roughly **10–15 % of decode speed** across context lengths, with a worse dip at
bs=8 (0.55×) where the split-KV heuristic changes regime.

### The config-list hypothesis is REFUTED

RFC #48374 flagged an open question: are upstream's autotune config lists
(`BLOCK_N=16` for the single-pass kernel, `32` for split-KV) simply bf16-shaped,
so that fp8's 656-byte rows want bigger blocks? Measured, at bs=32:

| BLOCK_N | warps | stages | fp8 | bf16 | fp8/bf16 |
|---|---|---|---|---|---|
| 16 | 4 | 3 | **1415 µs** | 1210 µs | 0.85× |
| 32 | 4 | 3 | 1546 µs | **1088 µs** | 0.70× |
| **64 / 128** | any | any | **does not compile** | — | shared-memory OOM on sm_86 |

**Bigger blocks are not available on this hardware** — `BLOCK_N ≥ 64` exceeds
sm_86's 100 KB shared-memory cap. (The retracted "BLOCK_N=64, 8 warps wins"
config could never have run in the real kernel at all.) fp8's best config is
essentially what upstream already ships; bf16's best is `BLOCK_N=32`. Best-vs-best
is **0.77×**. There is no free tuning win here.

## Honest summary

| AC | Status |
|---|---|
| AC1 engine boots + logits parity | ✅ **PASS** — 656 B pages, our decode ran, cosine 0.999999 teacher-forced |
| AC2 pool ≥1.9× | ❌ as written — **1.756×** is the format's ceiling (~1.63× effective) |
| AC3 speed ≥95 %/100 % | ❌ — **0.84–0.93×**; not recoverable by tuning (BLOCK_N ≥ 64 does not fit sm_86) |
| AC5 no non-Ampere regression | ◻ argued statically (`IS_FP8` defaults False); never tested on non-Ampere hardware |
| AC7 upstream package | ✅ [#48364](https://github.com/vllm-project/vllm/issues/48364) + [#48366](https://github.com/vllm-project/vllm/pull/48366) (NaN bug + fix) and [#48374](https://github.com/vllm-project/vllm/issues/48374) (RFC) |
| AC9 pre-flight | ✅ gap real and upstream-declared |
| AC4 (TP+DP+PP+MTP+graph), AC6 (128K e2e), AC8 | not reached |

**The trade, stated plainly:** on a 24 GB card, fp8 KV buys **1.76× KV capacity
(≈1.63× in practice) for ~10–15 % of decode speed**, with correctness verified
end-to-end inside a real vLLM engine (cosine 0.999999). Whether that is worth it
depends on whether you are context-bound or latency-bound. On a 3090 serving
long-context GLM-5.2, it is: the alternative is not "faster decode", it is "does
not fit".
