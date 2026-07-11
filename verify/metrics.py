# Copyright 2026 Nick / sm86-dsa contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Comparison metrics + tolerance gates (brief §6)."""

import torch


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item() if a.numel() else 0.0


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def topk_set_overlap(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean per-row |got ∩ ref| / |ref| over valid (>=0) indices. Ties may
    reorder between implementations; sets ignore order."""
    assert got.shape == ref.shape
    total, hits = 0, 0
    for r in range(got.shape[0]):
        rs = set(ref[r][ref[r] >= 0].tolist())
        gs = set(got[r][got[r] >= 0].tolist())
        total += len(rs)
        hits += len(rs & gs)
    return hits / total if total else 1.0


class Report:
    """Collects PASS/FAIL lines; run_all exits nonzero if any check failed."""

    def __init__(self):
        self.failed = 0
        self.lines = []

    def check(self, name: str, ok: bool, detail: str):
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failed += 1
        line = f"[{tag}] {name}: {detail}"
        self.lines.append(line)
        print(line, flush=True)

    def skip(self, name: str, why: str):
        line = f"[SKIP] {name}: {why}"
        self.lines.append(line)
        print(line, flush=True)
