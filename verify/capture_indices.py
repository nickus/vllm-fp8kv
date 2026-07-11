# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Capture REAL index tensors from a stock vLLM sparse-MLA run.

Reading a kernel signature tells you the shapes; it does not tell you the
PADDING SEMANTICS — what the backend actually puts in the unused topk slots
when a sequence is shorter than index_topk, whether indices are token-relative
or flat slot ids, whether the sentinel is -1 or num_slots or something else.
That is exactly where "the kernel is wrong at the interface" lives.

So: monkeypatch the stock sparse-MLA decode entry point, run one tiny
generation, dump every argument it received, and replay those tensors as
fixtures against our fp8 kernel (verify/contract.py::check_indices_contract).

Usage (on a box with vLLM + a DSA model):
    python verify/capture_indices.py --model <toy-or-real-glm-dsa> \
        [--out /root/idx_trace.pt]

Works with whichever sparse-MLA backend the platform selects; we hook the
generic MLA sparse impl rather than one concrete backend.
"""

import argparse
import os

import torch

CAPTURED = {}


def _install_hook(out_path: str):
    """Wrap the sparse-MLA decode call and record its inputs (first call only)."""
    import vllm.v1.attention.backends.mla.flashmla_sparse as fms

    target_cls = None
    for name in dir(fms):
        obj = getattr(fms, name)
        if isinstance(obj, type) and name.endswith("Impl"):
            target_cls = obj
            break
    if target_cls is None:
        raise RuntimeError("could not locate the sparse-MLA Impl class to hook")

    orig = target_cls.forward

    def traced(self, *args, **kwargs):
        if not CAPTURED:
            rec = {}
            for i, a in enumerate(args):
                if torch.is_tensor(a):
                    rec[f"arg{i}"] = a.detach().cpu()
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    rec[k] = v.detach().cpu()
            # the topk indices are the int tensor with a topk-sized last dim
            for k, v in list(rec.items()):
                if v.dtype in (torch.int32, torch.int64) and v.dim() >= 2 and v.shape[-1] >= 64:
                    rec["indices"] = v
                    rec["indices_from"] = k
            # cache geometry, so we can tell flat-slot from token indices
            for k, v in list(rec.items()):
                if v.dtype == torch.uint8 and v.dim() >= 2:
                    rec["num_slots"] = int(v.shape[0] * (v.shape[1] if v.dim() > 2 else 1))
            CAPTURED.update(rec)
            torch.save(rec, out_path)
            keys = {k: (tuple(v.shape), str(v.dtype)) for k, v in rec.items() if torch.is_tensor(v)}
            print(f"[capture] wrote {out_path}")
            print(f"[capture] tensors: {keys}")
            if "indices" in rec:
                idx = rec["indices"]
                print(f"[capture] INDICES: shape={tuple(idx.shape)} dtype={idx.dtype} "
                      f"min={int(idx.min())} max={int(idx.max())} "
                      f"negatives={int((idx < 0).sum())} "
                      f"unique_negatives={sorted(set(idx[idx < 0].tolist()))[:5]}")
        return orig(self, *args, **kwargs)

    target_cls.forward = traced
    print(f"[capture] hooked {target_cls.__name__}.forward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/root/idx_trace.pt")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="run the STOCK path (bf16/auto) — we want their contract, "
                         "not ours")
    a = ap.parse_args()

    _install_hook(a.out)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        kv_cache_dtype=a.kv_cache_dtype,
        enforce_eager=True,           # keep the call path simple for tracing
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
    )
    llm.generate(["The quick brown fox"], SamplingParams(max_tokens=4, temperature=0.0))
    print("[capture] done" if os.path.exists(a.out) else "[capture] NOTHING CAPTURED — hook missed")


if __name__ == "__main__":
    main()
