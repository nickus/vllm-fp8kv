# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""`verify/integration.py` and `verify/capture_indices.py`.

Both were shipped un-executed. integration.py's headline check — "does the
PATCHED UPSTREAM kernel decode fp8 correctly?" — is the one the upstream patch
actually stands on, and it must fail loudly when the runtime kernel is NOT
patched (the exact trap the 2026-07-11 audit fell into: the git clone was
patched, site-packages was not).

capture_indices.py hooked a backend that is never selected on Ampere, so its
hook could never fire. These tests pin the selection logic.
"""

import sys

import pytest
import torch

from tests import fakes
from tests.conftest import DEVICE
from verify import capture_indices, integration


def _patched_sparse_attention(q, kv, indices, sm_scale, **kw):
    """Stand-in for upstream's kernel WITH our patch applied: dispatches on the
    cache dtype (`is_fp8`) and decodes the 656-byte pages."""
    is_fp8 = kv.dtype in (torch.uint8, torch.float8_e4m3fn)   # the marker the check greps
    assert is_fp8, "fake only implements the fp8 branch"
    from vllm_fp8kv.fp8_ds_mla_sparse_decode import fp8_ds_mla_sparse_decode

    return fp8_ds_mla_sparse_decode(q, kv.reshape(-1, 656).contiguous(), indices, sm_scale)


def test_integration_main_is_green_with_a_patched_runtime(monkeypatch, capsys):
    fakes.install(monkeypatch, sparse_kernel=_patched_sparse_attention)
    with pytest.raises(SystemExit) as e:
        integration.main()
    assert e.value.code == 0, capsys.readouterr().out

    out = capsys.readouterr().out
    assert "[PASS] integration/dtype-declared" in out
    assert "[PASS] integration/indices-are-flat-global-slots" in out
    assert "[PASS] integration/full-chain-decode" in out
    assert "[PASS] integration/patched-upstream-kernel" in out


def test_integration_FAILS_when_the_runtime_kernel_is_unpatched(monkeypatch, capsys):
    """The audit's trap: patching the clone but importing an unpatched
    site-packages. The harness must say so instead of reporting green."""
    fakes.install(monkeypatch)          # default kernel = unpatched
    with pytest.raises(SystemExit) as e:
        integration.main()
    assert e.value.code >= 1

    out = capsys.readouterr().out
    assert "[FAIL] integration/patched-upstream-kernel" in out
    assert "UNPATCHED" in out


def test_integration_skips_without_pr47629(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.mla.triton_mla_sparse", None)
    with pytest.raises(SystemExit) as e:
        integration.main()
    assert e.value.code == 0
    assert "SKIP" in capsys.readouterr().out


# --------------------------------------------------------------- capture_indices
def test_hook_targets_the_backend_ampere_actually_selects(monkeypatch, tmp_path):
    """REGRESSION (audit M8): the hook used to be installed on flashmla_sparse
    only — a backend never selected on sm_86 — so it could never fire."""
    fakes.install(monkeypatch)
    capture_indices.CAPTURED.clear()

    out = tmp_path / "trace.pt"
    capture_indices._install_hook(str(out))

    tms = sys.modules["vllm.v1.attention.backends.mla.triton_mla_sparse"]
    assert tms.TritonMLASparseImpl.forward.__name__ == "traced", (
        "the backend selected on Ampere was not hooked"
    )


def test_hook_records_the_index_tensor_and_pool_size(monkeypatch, tmp_path):
    fakes.install(monkeypatch)
    capture_indices.CAPTURED.clear()
    out = tmp_path / "trace.pt"

    tms = sys.modules["vllm.v1.attention.backends.mla.triton_mla_sparse"]
    calls = []
    tms.TritonMLASparseImpl.forward = lambda self, *a, **k: calls.append(1)

    capture_indices._install_hook(str(out))

    idx = torch.randint(0, 512, (2, 1, 128), dtype=torch.int32)
    idx[0, 0, ::5] = -1
    cache = torch.zeros(8, 64, 656, dtype=torch.uint8)
    impl = tms.TritonMLASparseImpl()
    impl.forward(torch.randn(2, 16, 576), cache, indices=idx)

    assert calls == [1], "the original forward must still run"
    assert out.exists()
    rec = torch.load(out)
    assert torch.equal(rec["indices"], idx)
    assert rec["indices_from"] == "indices"
    assert rec["num_slots"] == 8 * 64
    assert capture_indices.CAPTURED, "module-level guard not set"


def test_hook_raises_when_no_sparse_backend_exists(monkeypatch):
    for name in capture_indices._SPARSE_MLA_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match="no sparse-MLA Impl"):
        capture_indices._install_hook("/tmp/never-written.pt")


def test_captured_trace_replays_through_the_contract_check(monkeypatch, tmp_path):
    """The capture -> replay loop closes: what capture_indices writes is exactly
    what contract.check_indices_contract consumes."""
    from verify import contract
    from verify.metrics import Report

    fakes.install(monkeypatch)
    capture_indices.CAPTURED.clear()
    out = tmp_path / "trace.pt"
    capture_indices._install_hook(str(out))

    tms = sys.modules["vllm.v1.attention.backends.mla.triton_mla_sparse"]
    idx = torch.randint(0, 512, (2, 1, 128), dtype=torch.int32)
    tms.TritonMLASparseImpl().forward(torch.randn(2, 16, 576, device=DEVICE),
                                      torch.zeros(8, 64, 656, dtype=torch.uint8),
                                      indices=idx)

    rep = Report()
    contract.check_indices_contract(rep, trace_path=str(out))
    assert rep.failed == 0, rep.lines
