# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""AC2's capacity arithmetic, and the patch generator driven in-process.

The AC2 numbers were only ever *printed*, never asserted — so "1.756x" was a
claim, not a check. Here it is a check, together with the effective (engine-
level) ratio that the indexer K-cache dilutes it to.

The generator is also exercised in-process (the subprocess tests in
test_patch_generator.py prove the CLI contract; these prove the internals).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_patch_generator import FIXTURE, TARGET
from verify.bench import (
    BF16_ROW,
    FP8_ROW,
    effective_pool_ratio,
    kv_pool_tokens,
)


def test_row_sizes_are_the_documented_ones():
    assert FP8_ROW == 656 == 512 + 16 + 128      # fp8 latent + tile scales + raw rope
    assert BF16_ROW == 1152 == 576 * 2


def test_kv_pool_capacity_matches_the_published_table():
    bf16_16, fp8_16, ratio = kv_pool_tokens(16 * 2**30)
    assert bf16_16 / 1e6 == pytest.approx(14.91, abs=0.01)
    assert fp8_16 / 1e6 == pytest.approx(26.19, abs=0.01)
    assert ratio == pytest.approx(1.756, abs=0.001)

    bf16_20, fp8_20, _ = kv_pool_tokens(20 * 2**30)
    assert bf16_20 / 1e6 == pytest.approx(18.64, abs=0.01)
    assert fp8_20 / 1e6 == pytest.approx(32.74, abs=0.01)


def test_the_19x_target_is_above_the_format_ceiling():
    """AC2 asked for >=1.9x. The format cannot deliver it: 16 B of inline scales
    and a 128 B raw-bf16 RoPE half do not shrink. This is why AC2 is recorded as
    a MISS, not a pass."""
    _, _, ratio = kv_pool_tokens(2**30)
    assert ratio < 1.9


def test_effective_ratio_accounts_for_the_indexer_cache():
    """RESULTS.md's ~1.63x: the DSA indexer's own K-cache is the same size in
    both configs, so it dilutes the gain an engine actually sees."""
    assert effective_pool_ratio() == pytest.approx(1.63, abs=0.01)
    assert effective_pool_ratio() < kv_pool_tokens(2**30)[2]
    # no indexer cache -> the raw format ratio
    assert effective_pool_ratio(indexer_row=0) == pytest.approx(1.756, abs=0.001)


# ------------------------------------------------------- generator, in-process
def _tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "vllm/v1/attention/ops").mkdir(parents=True)
    (src / TARGET).write_text(FIXTURE.read_text())
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    subprocess.run(["git", "add", "-A"], cwd=src, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
        cwd=src, check=True,
    )
    return src


def _load_generator(patch_out: Path):
    """Import the generator with its output path redirected into tmp."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_inproc", Path(__file__).resolve().parents[1] / "patches" / "make_fp8_kernel_patch.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.__file__ = str(patch_out)          # main() writes next to __file__
    return gen


def test_generator_main_writes_the_patch(tmp_path, monkeypatch, capsys):
    src = _tree(tmp_path)
    out_dir = tmp_path / "patches"
    out_dir.mkdir()
    gen = _load_generator(out_dir / "make_fp8_kernel_patch.py")

    monkeypatch.setattr(sys, "argv", ["make_fp8_kernel_patch.py", str(src)])
    gen.main()

    patch = out_dir / "triton_mla_sparse_fp8.patch"
    assert patch.exists()
    body = patch.read_text()
    assert "IS_FP8" in body and "_decode_fp8e4m3" in body
    assert "diff --git" in body
    assert "patched" in capsys.readouterr().out


def test_generator_main_defaults_to_root_vllm_src(tmp_path, monkeypatch):
    """No argv -> the historical /root/vllm-src default. Off the bench box that
    path is missing (or unreadable), and the generator must die on it rather
    than write an empty patch. OSError covers both FileNotFound and Permission."""
    gen = _load_generator(tmp_path / "make_fp8_kernel_patch.py")
    monkeypatch.setattr(sys, "argv", ["make_fp8_kernel_patch.py"])
    with pytest.raises((OSError, SystemExit)):
        gen.main()
    assert not (tmp_path / "triton_mla_sparse_fp8.patch").exists()


def test_thread_is_fp8_rejects_a_tree_with_too_few_signatures(tmp_path):
    gen = _load_generator(tmp_path / "make_fp8_kernel_patch.py")
    with pytest.raises(SystemExit, match="BLOCK_DPE"):
        gen._thread_is_fp8("def k(\n    BLOCK_DPE: tl.constexpr,\n):\n    pass\n")
