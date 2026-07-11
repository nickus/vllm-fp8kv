# AC9 Pre-flight — does fp8-KV-on-Ampere already exist upstream?

**Date checked:** 2026-07-11 (vLLM main @ `1bd8f80a`)
**Verdict: NO. The gap is real, upstream-declared, and precisely the one
`dsa-3090` already solved. Proceed.**

## Evidence

| Source | State on 2026-07-11 | Meaning for us |
|---|---|---|
| [vLLM FP8-KV blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) (2026-04-22, Kübler/Kurtić/Wilkinson/Bonanni/Goin/Marques/Budhathoki — AWS + Red Hat AI) | Covers **Hopper (FA3) and Blackwell (FlashInfer) only**. Ampere is not discussed at all. Scales: **per-tensor, uncalibrated, default 1.0**; optional per-head scales + LLM-Compressor calibration. | The maintainers' own state-of-the-art post has no Ampere fp8-KV. Also pins the **scale contract** we must match (D1 risk #1). |
| [PR #47629](https://github.com/vllm-project/vllm/pull/47629) — TRITON_MLA_SPARSE takeover of #38476 | **OPEN** (5 commits, awaiting 8 code-owner reviews; last activity 2026-07-07). Closes #38006. | The Ampere DSA baseline we build on is *still unmerged* — cherry-pick per the recipe. |
| ↳ its KV-dtype support | **"BF16 KV only on SM80; FP8 KV on SM120+ only."** Fused indexer-Q kernel is gated to SM89+ **because "Triton fp8e4nv store … does not compile on SM80"** — older archs fall back to unfused rope + `per_token_group_quant_fp8`. | **This is the exact wall dsa-3090 broke.** They worked *around* fp8e4nv; we *replaced* it (bit-exact software encode + decode). Our contribution slots directly into their stated blocker. |
| [`FLASHINFER_MLA_SPARSE_SM120`](https://github.com/vllm-project/vllm/pull/43477) | Merged; **requires** fp8 KV — hence sparse-MLA + fp8-KV exists on SM120 but has no Ampere counterpart. | Precedent that the sparse-MLA path *wants* fp8 KV; only the Ampere kernel support is missing. |
| vLLM main tree | `triton_mla_sparse.py` **absent**; `flashinfer_mla_sparse_sm120.py`, `xpu_mla_sparse.py`, `flashmla_sparse.py` present. | #38476/#47629 unmerged, confirmed at code level. |
| [PR #47644](https://github.com/vllm-project/vllm/pull/47644) | Pinned-buffer sync race fix for PP — companion to #47629. | Needed for the rig's PP config. |
| [PR #44698](https://github.com/vllm-project/vllm/pull/44698) | GLM MTP head as draft under PP (1.68–1.89×). | The other half of the rig target; not our deliverable but must stay compatible (AC4). |

## The contribution in one sentence

vLLM's Ampere DSA path is bf16-KV **because Triton cannot compile `fp8e4nv`
load/store on sm_80/86** — `dsa-3090` solved exactly that with bit-exact
software fp8 encode/decode, so porting it turns their documented fallback into
full fp8-KV support (≈2× KV pool on 24GB cards).

## Notes / discrepancies to verify in code

- One source claims #47629 contains a **fused gather+dequant decode kernel**;
  the PR page says it does **not** (uses existing decode paths + Triton sparse
  indexer). Resolve at code level before choosing the D2 integration skeleton.
- #47629 has a reported bug worth avoiding in our patch: capability checks use
  `current_platform.has_device_capability(89)` = **device-0 only**, which
  breaks heterogeneous fleets (relevant: the rig is 29×3090 + 1×5090).
- Scale semantics are the #1 correctness risk: vLLM = per-tensor k/v scales
  (default 1.0); dsa-3090 = per-128-tile fp32 scales (amax/448) baked into the
  cache row. These are **not** the same contract — D1 must reconcile explicitly.
