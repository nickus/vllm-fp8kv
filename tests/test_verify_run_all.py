# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""`verify/run_all.py` — the harness itself, run on CPU.

It used to skip its entire decode section without a GPU, which meant CI verified
nothing at all. It now runs the same code paths at interpreter-sized shapes, so
these tests assert that the harness both RUNS and CATCHES a broken kernel.
"""

import pytest
import torch

from tests.conftest import DEVICE
from verify import run_all
from verify.metrics import Report


def test_dequant_and_layout_sections_pass():
    rep = Report()
    run_all.section_dequant(rep, DEVICE)
    run_all.section_layout(rep, DEVICE)
    assert rep.failed == 0, rep.lines
    assert any("dequant/bitmath-256" in ln for ln in rep.lines)
    assert any("layout/rope-raw-bf16-exact" in ln for ln in rep.lines)


def test_decode_section_runs_without_a_gpu():
    """The regression that mattered: `if device != "cuda": skip` meant the
    kernel was never exercised in CI."""
    rep = Report()
    run_all.section_decode(rep, DEVICE)
    assert rep.failed == 0, rep.lines

    names = " ".join(rep.lines)
    for check in (
        "decode/empty-selection-is-zero",
        "decode/leading-masked-chunks-no-nan",
        "decode/paged-3d-cache-shape",
        "decode/lse-matches-logsumexp",
    ):
        assert check in names, f"{check} did not run"
    assert "SKIP" not in names, "decode section still skips on CPU"


def test_decode_section_CATCHES_a_broken_kernel(monkeypatch):
    """Test-the-test: if the kernel returned something wrong, the harness must
    go red. (Here: an 8x-scaled output — the exact error class that hid behind
    cosine in the retracted benchmarks.)"""
    import vllm_fp8kv.fp8_ds_mla_sparse_decode as mod

    real = mod.fp8_ds_mla_sparse_decode

    def scaled(*a, **kw):
        out = real(*a, **kw)
        return (out[0] * 8, out[1]) if isinstance(out, tuple) else out * 8

    monkeypatch.setattr(mod, "fp8_ds_mla_sparse_decode", scaled)

    rep = Report()
    run_all.section_decode(rep, DEVICE)
    assert rep.failed > 0, (
        "an 8x-wrong kernel passed the harness — the max_abs gate is missing"
    )


def test_bigpool_section_skips_on_cpu():
    rep = Report()
    run_all.section_bigpool(rep, DEVICE)
    assert rep.failed == 0
    if DEVICE == "cpu":
        assert any("SKIP" in ln and "bigpool" in ln for ln in rep.lines)


def test_main_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        run_all.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "0 failure(s)" in out
    assert "[FAIL]" not in out
