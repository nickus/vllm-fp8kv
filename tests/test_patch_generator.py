# Copyright 2026 Nick / vllm-fp8kv contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The upstream patch generator: does it produce the artifact we ship, and does
it FAIL LOUDLY rather than silently mis-patching?

The 2026-07-11 audit found a string-replace in this file that matched nothing
and raised no error (upstream's split-kernel autotune key had drifted), plus a
checked-in patch that no longer matched its generator. Both are regressions this
file exists to prevent.

`tests/fixtures/triton_mla_sparse_kernel_pr47629.py` is vLLM's kernel at PR
#47629 head bbe2ab4d6 (Apache-2.0), vendored so these tests are hermetic.
"""

import importlib.util
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "patches" / "make_fp8_kernel_patch.py"
SHIPPED_PATCH = REPO / "patches" / "triton_mla_sparse_fp8.patch"
FIXTURE = Path(__file__).parent / "fixtures" / "triton_mla_sparse_kernel_pr47629.py"
TARGET = "vllm/v1/attention/ops/triton_mla_sparse_kernel.py"


def _upstream_tree(tmp_path: Path, git: bool = True, mutate=None) -> Path:
    """A minimal vLLM source tree containing just the kernel we patch."""
    src = tmp_path / "vllm-src"
    (src / "vllm/v1/attention/ops").mkdir(parents=True)
    text = FIXTURE.read_text()
    if mutate is not None:
        text = mutate(text)
    (src / TARGET).write_text(text)
    if git:
        subprocess.run(["git", "init", "-q"], cwd=src, check=True)
        subprocess.run(["git", "add", "-A"], cwd=src, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
            cwd=src, check=True,
        )
    return src


def _run_generator(src: Path, out_patch: Path):
    """Run the generator as a subprocess, with the patch output redirected."""
    shutil.copy(GENERATOR, out_patch.parent / "make_fp8_kernel_patch.py")
    return subprocess.run(
        [sys.executable, str(out_patch.parent / "make_fp8_kernel_patch.py"), str(src)],
        capture_output=True, text=True,
    )


def test_regenerates_the_shipped_patch_byte_for_byte(tmp_path):
    """The checked-in .patch must be exactly what the generator produces today."""
    src = _upstream_tree(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    r = _run_generator(src, patches_dir / "triton_mla_sparse_fp8.patch")
    assert r.returncode == 0, r.stderr

    def _body(p: str) -> list[str]:
        # drop the `index <old>..<new> <mode>` line: git abbreviates blob hashes
        # to as few chars as are unambiguous IN THAT REPO, so a 1-file temp repo
        # legitimately prints shorter hashes than the real vLLM clone.
        return [ln for ln in p.splitlines() if not ln.startswith("index ")]

    got = (patches_dir / "triton_mla_sparse_fp8.patch").read_text()
    assert _body(got) == _body(SHIPPED_PATCH.read_text()), (
        "patches/triton_mla_sparse_fp8.patch is STALE vs its generator"
    )
    assert len(_body(got)) == 223, "patch size changed unexpectedly"


def test_patched_kernel_compiles_and_threads_the_flag(tmp_path):
    src = _upstream_tree(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    assert _run_generator(src, patches_dir / "p.patch").returncode == 0

    patched = (src / TARGET).read_text()
    py_compile.compile(str(src / TARGET), doraise=True)

    # the flag must reach BOTH kernels' signatures, BOTH compute_tile calls and
    # BOTH launches — the split kernel is the one decode actually takes
    assert patched.count("IS_FP8: tl.constexpr,") == 2, "both kernels must declare IS_FP8"
    assert patched.count("IS_FP8: tl.constexpr = False,") == 1, "compute_tile default"
    assert patched.count("        IS_FP8,\n    )") == 2, "both compute_tile call sites"
    assert patched.count("IS_FP8=is_fp8,") == 2, "both kernel launches"
    assert "_decode_fp8e4m3" in patched and "_FP8_ROW_BYTES" in patched


def test_does_not_touch_the_autotune_keys(tmp_path):
    """The 'dtype-blind autotune key' theory was false (Triton >=3 already keys
    on tensor dtypes, and IS_FP8 is a constexpr specialization). The generator
    must not reintroduce that hunk."""
    src = _upstream_tree(tmp_path)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _run_generator(src, patches_dir / "p.patch")

    patched = (src / TARGET).read_text()
    assert 'key=["index_topk", "kv_group_num"]' in patched, "final-kernel key altered"
    assert 'key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"]' in patched, (
        "split-kernel key altered"
    )
    assert "IS_FP8" not in patched.split("def _sparse_mla_kernel_final")[0].split(
        "@triton.autotune"
    )[-1].split(")")[0], "IS_FP8 must not appear in an autotune key"


@pytest.mark.parametrize(
    "drop",
    [
        "from vllm.utils.platform_utils import num_compute_units",   # import anchor
        "        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)",
        "    assert kv.shape[1] == 1 and kv.shape[2] == _DIM_QK",     # dispatcher anchor
    ],
)
def test_missing_anchor_fails_loudly(tmp_path, drop):
    """Upstream drift must abort the generator, never produce a half-patch."""
    src = _upstream_tree(tmp_path, mutate=lambda t: t.replace(drop, "# drifted"))
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    r = _run_generator(src, patches_dir / "p.patch")
    assert r.returncode != 0
    assert "ANCHOR NOT FOUND" in (r.stdout + r.stderr)


def test_missing_call_site_fails_loudly(tmp_path):
    """The un-anchored multi-replaces are count-guarded: if upstream refactors a
    compute_tile call away, we must abort, not silently patch one of two."""
    def mutate(t):
        return t.replace("        BLOCK_DPE,\n    )", "        BLOCK_DPE, )", 1)

    src = _upstream_tree(tmp_path, mutate=mutate)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    r = _run_generator(src, patches_dir / "p.patch")
    assert r.returncode != 0
    assert "expected exactly 2 compute_tile call sites, found 1" in (r.stdout + r.stderr)


def test_missing_launch_fails_loudly(tmp_path):
    def mutate(t):
        return t.replace("        BLOCK_DPE=_BLOCK_DPE,", "        BLOCK_DPE=_BLOCK_DPE ,", 1)

    src = _upstream_tree(tmp_path, mutate=mutate)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    r = _run_generator(src, patches_dir / "p.patch")
    assert r.returncode != 0
    assert "expected exactly 2 kernel launches, found 1" in (r.stdout + r.stderr)


def test_refuses_to_clobber_the_patch_when_target_is_not_a_git_tree(tmp_path):
    """Pointing the generator at site-packages produced an EMPTY git diff, which
    would have silently overwritten the shipped patch with nothing."""
    src = _upstream_tree(tmp_path, git=False)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    out = patches_dir / "triton_mla_sparse_fp8.patch"
    out.write_text("PRECIOUS ARTIFACT\n")

    r = _run_generator(src, out)
    assert r.returncode != 0
    assert "REFUSING to overwrite" in (r.stdout + r.stderr)
    assert out.read_text() == "PRECIOUS ARTIFACT\n", "generator destroyed the patch"


def test_shipped_patch_applies_to_the_pinned_upstream_head(tmp_path):
    """`git apply --check` of the artifact we actually ship."""
    src = _upstream_tree(tmp_path)
    r = subprocess.run(
        ["git", "apply", "--check", str(SHIPPED_PATCH)], cwd=src, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr

    subprocess.run(["git", "apply", str(SHIPPED_PATCH)], cwd=src, check=True)
    py_compile.compile(str(src / TARGET), doraise=True)


def test_generator_module_is_importable_and_declares_its_anchors():
    """Cheap guard: the anchors are module constants, not buried literals."""
    spec = importlib.util.spec_from_file_location("gen", GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.TARGET == TARGET
    for name in ("IMPORT_ANCHOR", "SIG_ANCHOR", "LOADS_ANCHOR", "V_ANCHOR", "DISPATCH_ANCHOR"):
        assert getattr(gen, name), f"{name} is empty"
    assert not hasattr(gen, "AUTOTUNE_ANCHOR_1"), "the withdrawn autotune hunk is back"
