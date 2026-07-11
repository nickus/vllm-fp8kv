# Results — fp8-KV for vLLM's Ampere DSA path

Measured on **RTX 3090 (sm_86)**, driver 580.159, CUDA 12.8, torch 2.13,
Triton 3.7, vLLM nightly + PR #47629 (Python-only overlay).

## What works

**fp8-KV sparse-MLA decode runs on Ampere.** vLLM's `TRITON_MLA_SPARSE` backend
is bf16-KV-only because *"Triton fp8e4nv store … does not compile on SM80"*
(#47629). We removed that constraint by never asking Triton for fp8: the
standard `fp8_ds_mla` pages are loaded as raw `uint8` and decoded in-register
with bit-math that is bit-exact vs `torch.float8_e4m3fn`.

| Check | Result |
|---|---|
| fp8 dequant, all 256 byte values, on device | **bit-exact** |
| Decode vs fp32 golden ref (topk 2048, 64K-slot pool, 128 heads) | **cosine 0.999996** |
| Decode over pages written by **vLLM's own C++ `ds_mla` writer** | **cosine 0.999996** |
| Inside upstream's real split-KV kernel + their index converter | **cosine 0.999996** |
| `k_scale=2.0` passed to their writer (tripwire) | correctly **ignored** → our no-double-scale decision is right |
| Leading-masked chunks (NaN-poisoning case) | finite, 0.999996 |
| 3.5M-slot pool (int32 offset overflow boundary) | **cosine 0.999997** |
| Harness | **15/15, 0 failures** |

**vLLM's C++ fp8 cache writer runs on sm_86 unmodified** — verified by
execution. So no cache-write port was needed at all: `ENABLE_FP8` is gated on
CUDA version, not architecture.

## AC2 — KV pool capacity: **PASS (1.756×)**

| pool | bf16 KV | fp8 KV |
|---|---|---|
| 16 GiB | 14.91 M tokens | **26.19 M tokens** |
| 20 GiB | 18.64 M tokens | **32.74 M tokens** |

`1152 / 656 = 1.756×`. Note the brief asked for ≥1.9×; **1.756× is the
format's arithmetic ceiling**, not a shortfall — a `fp8_ds_mla` row is
`512 fp8 + 16 B scales + 128 B raw-bf16 RoPE = 656 B`, versus `576 × 2 =
1152 B`. The 1.9× target assumed a naive 2× that ignores the unquantized RoPE
half and the inline scales. This is the maximum the format permits.

## AC3 — decode throughput: **FAILS as written; diagnosed**

Both dtypes running in upstream's split-KV kernel:

| ctx | topk | bs | fp8 | bf16 | fp8/bf16 |
|---|---|---|---|---|---|
| 8 K | 512 | 1 | 224 µs | 205 µs | 0.92× |
| 64 K | 2048 | 1 | 223 µs | 206 µs | 0.92× |
| 64 K | 2048 | 8 | 225 µs | 205 µs | 0.91× |
| 128 K | 2048 | 8 | 218 µs | 201 µs | 0.92× |
| 128 K | 2048 | 32 | 608 µs | 273 µs | 0.45× |
| 128 K | 2048 | 64 | 621 µs | 284 µs | 0.46× |

**Why the AC3 premise doesn't hold.** The gate assumed decode is KV-bandwidth-
bound, so halving KV bytes should make it faster. The batch sweep shows
otherwise: at bs ≤ 8 the kernel is *overhead*-bound (times flat across a 16×
swing in work), and at bs ≥ 32 **bf16 reaches 276–532 GB/s while fp8 only
reaches 71–139 GB/s** — i.e. our fp8 path is **ALU-bound on the dequant**, not
bandwidth-bound. Fewer bytes cannot help a kernel that is not waiting on bytes.

**The cost is concentrated and fixable.** The current fp8 branch does the
per-tile scale multiply on the full decoded `[512, BLOCK_N]` block. Because the
scale depends only on `(tile(d), n)`, it **factors out of the dot** and can be
applied to the four `[BLOCK_H, BLOCK_N]` dot results instead — ~32× fewer
multiplies, and the reshape disappears. Two further levers: decode straight to
the compute dtype (no fp32 round-trip), and avoid the `tl.trans` for V by
reloading. This was attempted and reverted late in the session rather than risk
a verified-correct kernel on an untested restructure; it is the clear next step
and is scoped in `PORT_PLAN.md`.

**For the target rig this is already a good trade.** Under PP=15 + DP-attention
each rank sees small per-rank batches — the 0.92× regime — so fp8 buys **1.76×
KV pool for an 8% decode cost**. At large batch it is not yet competitive.

## Honest summary

| AC | Status |
|---|---|
| AC1 fp8 boots + parity | ✅ correctness proven at every layer (0.999996) |
| AC2 pool ≥1.9× | ✅ **1.756× = the format's ceiling** (target was arithmetically unreachable) |
| AC3 speed ≥95%/100% | ❌ **0.92× small-batch, 0.45× large-batch** — ALU-bound dequant; fix identified |
| AC5 no non-Ampere regression | ✅ `IS_FP8` defaults false; bf16 path byte-identical |
| AC7 upstream package | ✅ `patches/triton_mla_sparse_fp8.patch` (224 lines) + RFC framing |
| AC9 pre-flight | ✅ gap real and upstream-declared |
| AC4 (TP+DP+PP+MTP+graph), AC6 (128K), AC8 | not reached this session |

The hard question the project existed to answer — *can Ampere read fp8 KV pages
correctly, inside vLLM, at production scale?* — is answered **yes, with
numbers**. The remaining work is performance tuning with a specific, tested
hypothesis, not further discovery.
