#!/usr/bin/env python3
"""AC1, engine level: boot vLLM with fp8 KV on the Ampere sparse-MLA path and
compare logits against a bf16 run of the SAME model.

This is the check the whole vllm-fp8kv project never did. It is deliberately
paranoid about self-deception:

  * it ASSERTS the selected backend is TRITON_MLA_SPARSE (not a silent fallback
    to some other MLA backend, which would make "fp8 works!" meaningless);
  * it ASSERTS the allocated KV cache is 656 B/token (not 576 x bf16), i.e. that
    the fp8 pages are real;
  * it ASSERTS our fp8 decode path actually executed (a counter in the forward),
    so a backend that quietly ignored the dtype cannot pass;
  * it compares LOGITS, not just "did it emit tokens".

Usage:  python engine_boot.py --model <path> [--max-len 2048]
"""
import argparse
import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
# vLLM V1 runs EngineCore in a SEPARATE PROCESS by default, so an in-process
# monkeypatch (backend_patch.apply()) never reaches the worker: the backend
# still refuses fp8 ("TRITON_MLA_SPARSE: kv_cache_dtype not supported") and the
# decode spies never fire. Run the engine in-process so the patch applies.
# (A real deployment would install it via a vLLM plugin entry point or a
# sitecustomize.py, exactly as dsa-3090 does for sglang's subprocess workers.)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch


def build_toy_dsa_model(out_dir: str) -> str:
    """A tiny GlmMoeDsa checkpoint: real architecture, random weights.

    Small enough to boot in seconds; the point is the CODE PATH (sparse MLA
    decode over an fp8 KV pool), not the quality of the output.
    """
    import json

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    cfg = AutoConfig.for_model(
        "glm_moe_dsa",
        num_hidden_layers=2,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        kv_lora_rank=512,
        q_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        # upstream's index converter asserts NUM_TOPK_TOKENS % 128 == 0
        index_topk=128,
        index_n_heads=8,
        index_head_dim=128,
        # must cover the tokenizer's id space: the gpt2 tokenizer emits ids up
        # to 50256, and a smaller vocab makes the embedding index out of bounds
        # (device-side assert, far from the real cause)
        vocab_size=50304,
        max_position_embeddings=4096,
        torch_dtype="bfloat16",
        # vLLM only materializes gate.e_score_correction_bias when
        # topk_method == "noaux_tc" (deepseek_v2.py:313). Real GLM-5.2 sets it;
        # a toy without it makes vLLM's loader KeyError on a weight the
        # transformers checkpoint does contain. Match the real model.
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.save_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.save_pretrained(out_dir)

    # vLLM's GLM/DeepSeek MoE loader requires `mlp.gate.e_score_correction_bias`
    # on every routed-MoE layer (the noaux_tc routing bias). transformers does
    # not emit it for a from_config model, so the checkpoint is incomplete and
    # the engine dies with KeyError at weight load. Add it explicitly.
    from safetensors.torch import load_file, save_file

    st = os.path.join(out_dir, "model.safetensors")
    sd = load_file(st)
    added = 0
    for i in range(cfg.num_hidden_layers):
        if i < cfg.first_k_dense_replace:
            continue                            # dense layer, no gate
        key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if key not in sd:
            sd[key] = torch.zeros(cfg.n_routed_experts, dtype=torch.float32)
            added += 1
    if added:
        save_file(sd, st, metadata={"format": "pt"})

    with open(os.path.join(out_dir, "config.json")) as f:
        c = json.load(f)
    print(f"[toy] {c['architectures']} layers={c['num_hidden_layers']} "
          f"index_topk={c.get('index_topk')} experts={c.get('n_routed_experts')} "
          f"(+{added} e_score_correction_bias)")
    return out_dir


def run(model: str, kv_dtype: str, max_len: int, prompts: list[str]):
    """Boot an engine, capture prompt logprobs, and report what really ran."""
    import vllm_fp8kv.backend_patch as bp

    bp.apply()

    from vllm import LLM, SamplingParams

    # instrument: prove which decode path executed
    from vllm.v1.attention.backends.mla import triton_mla_sparse as tms

    counters = {"fp8": 0, "bf16": 0, "row_bytes": None, "backend": None}
    real_fp8 = bp._forward_fp8_kv
    real_bf16 = tms.TritonMLASparseImpl._forward_bf16_kv

    def fp8_spy(self, q, kv, idx, md):
        counters["fp8"] += 1
        counters["row_bytes"] = int(kv.shape[-1])
        counters["backend"] = type(self).__name__
        return real_fp8(self, q, kv, idx, md)

    def bf16_spy(self, q, kv, idx, md):
        counters["bf16"] += 1
        counters["row_bytes"] = int(kv.shape[-1])
        counters["backend"] = type(self).__name__
        return real_bf16(self, q, kv, idx, md)

    bp._forward_fp8_kv = fp8_spy
    tms.TritonMLASparseImpl._forward_bf16_kv = bf16_spy

    llm = LLM(
        model=model,
        kv_cache_dtype=kv_dtype,
        max_model_len=max_len,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    # TEACHER-FORCED comparison. `prompt_logprobs` scores every prompt position
    # against the SAME inputs in both runs, so bf16 and fp8 are compared on
    # identical sequences. Generated-token logprobs are NOT comparable once the
    # two runs pick different tokens — after that they are scoring different
    # sequences, and on a random-weight toy (near-flat logits over a 50k vocab)
    # a 2% fp8 perturbation flips near-ties routinely. That divergence says
    # nothing about the kernel; the teacher-forced logits do.
    sp = SamplingParams(max_tokens=8, temperature=0.0, prompt_logprobs=5, logprobs=0)
    outs = llm.generate(prompts, sp)

    logits, top1 = [], []
    for o in outs:
        row, best = [], []
        for pl in (o.prompt_logprobs or []):
            if not pl:
                continue                       # first prompt position has no logprobs
            items = sorted(pl.items(), key=lambda kv: -kv[1].logprob)
            row.append([lp.logprob for _, lp in items[:5]])
            best.append(items[0][0])
        logits.append(torch.tensor(row))
        top1.append(best)
    toks = [o.outputs[0].token_ids for o in outs]

    del llm
    torch.cuda.empty_cache()
    return logits, top1, toks, counters


PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then",
    "In a shocking finding, scientists discovered that",
]


def _worker(model, kv_dtype, max_len, out_json):
    """One engine per PROCESS. Two engines in one process cannot both allocate a
    KV pool (the first one's memory is not returned to the allocator in time),
    and vLLM's own profiler trips over the wobble."""
    import json as _json

    logits, top1, toks, counters = run(model, kv_dtype, max_len, PROMPTS)
    with open(out_json, "w") as f:
        _json.dump({
            "logits": [t.tolist() for t in logits],
            "top1": top1,
            "tokens": [list(t) for t in toks],
            "counters": counters,
        }, f)
    print(f"[{kv_dtype}] backend={counters['backend']} "
          f"kv_row_bytes={counters['row_bytes']} "
          f"bf16_calls={counters['bf16']} fp8_calls={counters['fp8']}")


def main():
    import json as _json
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--kv", default=None, help="internal: run ONE engine")
    ap.add_argument("--out", default=None, help="internal: json result path")
    a = ap.parse_args()

    if a.kv:                                        # child process
        _worker(a.model, a.kv, a.max_len, a.out)
        return

    model = a.model or build_toy_dsa_model("/root/toy-glm-dsa")

    results = {}
    for tag, kv in (("bf16", "auto"), ("fp8", "fp8_e4m3")):
        print(f"\n===== {kv} KV =====", flush=True)
        out = f"/tmp/engine_{tag}.json"
        r = subprocess.run(
            [sys.executable, __file__, "--model", model, "--max-len", str(a.max_len),
             "--kv", kv, "--out", out],
            capture_output=True, text=True,
        )
        for line in r.stdout.splitlines():
            if line.startswith("[") or "TRITON_MLA_SPARSE attention backend" in line:
                print(line)
        if r.returncode != 0:
            print(f"[{tag}] ENGINE FAILED (rc={r.returncode})")
            tail = [ln for ln in (r.stdout + r.stderr).splitlines()
                    if any(k in ln for k in ("Error", "error", "assert", "raise"))][-6:]
            print("\n".join(tail))
            sys.exit(1)
        with open(out) as f:
            results[tag] = _json.load(f)

    l_bf16 = [torch.tensor(x) for x in results["bf16"]["logits"]]
    l_fp8 = [torch.tensor(x) for x in results["fp8"]["logits"]]
    p1_bf16, p1_fp8 = results["bf16"]["top1"], results["fp8"]["top1"]
    t_bf16, t_fp8 = results["bf16"]["tokens"], results["fp8"]["tokens"]
    c_bf16, c_fp8 = results["bf16"]["counters"], results["fp8"]["counters"]

    print("\n===== AC1 verdict =====")
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    check("engine/boots-with-fp8-kv", c_fp8["fp8"] > 0 or c_fp8["bf16"] > 0,
          "engine booted and ran the sparse-MLA backend")
    check("engine/fp8-path-executed", c_fp8["fp8"] > 0 and c_fp8["bf16"] == 0,
          f"our fp8 decode ran {c_fp8['fp8']}x, bf16 decode ran {c_fp8['bf16']}x")
    check("engine/kv-pages-are-656B", c_fp8["row_bytes"] == 656,
          f"fp8 KV row = {c_fp8['row_bytes']} B (bf16 run: {c_bf16['row_bytes']} B)")
    check("engine/baseline-used-bf16-path", c_bf16["bf16"] > 0 and c_bf16["fp8"] == 0,
          f"bf16 decode ran {c_bf16['bf16']}x")

    for i, (a_, b_) in enumerate(zip(l_bf16, l_fp8)):
        n = min(len(a_), len(b_))
        if n == 0:
            continue
        x, y = a_[:n].flatten(), b_[:n].flatten()      # [positions x top5] logprobs
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        mad = (x - y).abs().mean().item()
        check(f"logits/parity-prompt{i}", cos > 0.99,
              f"cosine={cos:.6f} mean_abs_diff={mad:.4f} over {n} teacher-forced "
              f"positions x top-5 logprobs")

    # teacher-forced top-1 agreement: same inputs, so this IS comparable
    for i, (a_, b_) in enumerate(zip(p1_bf16, p1_fp8)):
        agree = sum(x == y for x, y in zip(a_, b_)) / max(len(a_), 1)
        check(f"top1/teacher-forced-prompt{i}", agree >= 0.75,
              f"{agree:.3f} over {len(a_)} prompt positions")

    for i, (a_, b_) in enumerate(zip(t_bf16, t_fp8)):
        agree = sum(x == y for x, y in zip(a_, b_)) / max(len(a_), 1)
        print(f"[note] free-running greedy tokens prompt{i}: {agree:.3f} agreement "
              f"(NOT a gate — random-weight toy, near-flat logits: once the two "
              f"runs differ at one step they score different sequences)")

    print(f"\n== AC1 {'PASS' if ok else 'FAIL'} ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
