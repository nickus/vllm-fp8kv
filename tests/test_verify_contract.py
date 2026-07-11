# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""`verify/contract.py` — and, crucially, TESTS OF THE TESTS (task B2).

A canary that cannot fail is not a canary. Each check here is driven twice:
once against a faithful fake of vLLM's cache writer (must PASS) and once
against a deliberately broken one (must FAIL). The broken variants are the two
futures we actually fear:

  * upstream RE-TILES the 656-byte row  -> our decode would read scales/rope
    from the wrong bytes and return plausible garbage;
  * upstream starts HONOURING `k_scale` -> our decode would double-scale and
    silently lose accuracy (a bias no toy-agreement test can see).
"""

import functools

import pytest
import torch

from tests import fakes
from tests.conftest import DEVICE
from verify import contract
from verify.metrics import Report


@pytest.fixture
def vllm(monkeypatch):
    return fakes.install(monkeypatch)


# ---------------------------------------------------------------- layout canary
def test_layout_canary_passes_on_the_real_geometry(vllm):
    rep = Report()
    contract.check_layout_contract(rep)
    assert rep.failed == 0
    assert any("layout-row-bytes" in ln and "PASS" in ln for ln in rep.lines)


def test_layout_canary_FIRES_when_vllm_retiles_the_row(monkeypatch):
    """B2 negative: if vLLM's own backend reports a different row size, the
    canary must fail loudly rather than let the kernel read garbage."""
    class Drifted(fakes.FakeFlashMLASparseBackend):
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                               cache_dtype_str="auto"):
            return (num_blocks, block_size, 704)      # re-tiled row

    fakes.install(monkeypatch)
    import sys
    monkeypatch.setattr(
        sys.modules["vllm.v1.attention.backends.mla.flashmla_sparse"],
        "FlashMLASparseBackend", Drifted,
    )

    rep = Report()
    contract.check_layout_contract(rep)
    assert rep.failed == 1, "the drift canary did not fire"
    assert any("FAIL" in ln and "704" in ln for ln in rep.lines)


def test_layout_canary_skips_without_vllm(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.mla.flashmla_sparse", None)
    rep = Report()
    contract.check_layout_contract(rep)
    assert rep.failed == 0
    assert any("SKIP" in ln for ln in rep.lines)


# -------------------------------------------------------------- writer roundtrip
def test_writer_roundtrip_passes_against_a_faithful_writer(vllm):
    rep = Report()
    contract.check_writer_roundtrip(rep, device=DEVICE)
    assert rep.failed == 0, rep.lines

    names = " ".join(rep.lines)
    assert "scale-offset-observed" in names and "rope-offset-observed" in names
    assert "writer-nope-roundtrip" in names and "decode-over-vllm-written-pages" in names


def test_k_scale_tripwire_FIRES_if_the_writer_starts_honouring_k_scale(monkeypatch):
    """B1/B2 negative: contract.py passes k_scale=2.0 and asserts the roundtrip
    is unaffected (today the ds_mla writer ignores it and computes inline tile
    scales). The day that changes, our decode would be 2x off — this must fail."""
    honouring = functools.partial(fakes.write_ds_mla_pages, honour_k_scale=True)
    fakes.install(
        monkeypatch,
        writer=lambda kv_c, k_pe, cache, slot, kv_cache_dtype, scale:
            honouring(kv_c, k_pe, cache, slot, scale),
    )

    rep = Report()
    contract.check_writer_roundtrip(rep, device=DEVICE)
    assert rep.failed >= 1, "the k_scale tripwire did not fire"
    assert any("writer-nope-roundtrip" in ln and "FAIL" in ln for ln in rep.lines)


def test_offset_canary_FIRES_when_the_writer_moves_the_scales(monkeypatch):
    """B2 negative: a writer that stores the fp32 tile scales somewhere else must
    be caught by the offset search, not silently mis-read."""
    # a plausible re-tiling that still fits the 656-byte row: rope right after
    # the fp8 latent, the 4 fp32 scales moved to the tail
    moved = functools.partial(fakes.write_ds_mla_pages, scale_off=640, rope_off=512)
    fakes.install(
        monkeypatch,
        writer=lambda kv_c, k_pe, cache, slot, kv_cache_dtype, scale:
            moved(kv_c, k_pe, cache, slot, scale),
    )

    rep = Report()
    contract.check_writer_roundtrip(rep, device=DEVICE)
    assert any("scale-offset-observed" in ln and "FAIL" in ln for ln in rep.lines), (
        "the scale-offset canary did not fire on a re-tiled row"
    )
    assert any("rope-offset-observed" in ln and "FAIL" in ln for ln in rep.lines)


def test_writer_roundtrip_reports_a_writer_that_raises(monkeypatch):
    """On sm_86 the C++ writer might not launch at all — that must be a FAIL with
    the exception text, not a crash."""
    def boom(*a, **kw):
        raise RuntimeError("no kernel image is available")

    fakes.install(monkeypatch, writer=boom)
    rep = Report()
    contract.check_writer_roundtrip(rep, device=DEVICE)
    assert rep.failed == 1
    assert any("writer raised" in ln for ln in rep.lines)


def test_writer_roundtrip_skips_without_vllm(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", None)
    rep = Report()
    contract.check_writer_roundtrip(rep, device=DEVICE)
    assert rep.failed == 0
    assert any("SKIP" in ln for ln in rep.lines)


# ------------------------------------------------------------- indices contract
def test_indices_contract_skips_without_a_trace(tmp_path):
    rep = Report()
    contract.check_indices_contract(rep, trace_path=str(tmp_path / "nope.pt"))
    assert rep.failed == 0
    assert any("SKIP" in ln and "indices-replay" in ln for ln in rep.lines)


def test_indices_contract_accepts_a_valid_trace(tmp_path):
    trace = {
        "indices": torch.randint(0, 512, (4, 1, 128), dtype=torch.int32),
        "num_slots": 512,
    }
    trace["indices"][0, 0, ::7] = -1
    p = tmp_path / "idx.pt"
    torch.save(trace, p)

    rep = Report()
    contract.check_indices_contract(rep, trace_path=str(p))
    assert rep.failed == 0, rep.lines


def test_indices_contract_FIRES_on_token_relative_indices(tmp_path):
    """If a future backend hands us token-relative indices (bigger than the
    pool) instead of flat slots, our kernel would mask them all away and return
    zeros. The replay must catch that."""
    trace = {
        "indices": torch.full((2, 1, 64), 99999, dtype=torch.int32),
        "num_slots": 512,
    }
    p = tmp_path / "idx.pt"
    torch.save(trace, p)

    rep = Report()
    contract.check_indices_contract(rep, trace_path=str(p))
    assert rep.failed == 1, "the flat-slot contract check did not fire"


def test_main_runs_end_to_end(vllm, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        contract.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "CONTRACT tests" in out and "FAIL" not in out
