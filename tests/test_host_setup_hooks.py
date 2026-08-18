"""Unit tests for the host-setup parallelism hooks.

These back the GPU-batched emulator's parallel host setup (one spawn worker per
profile, each with a private FastChem tree). Three independent, additive hooks:

  1. RATE-PARSE CACHE — `ReadRate.read_rate` memoises the network parse per
     process. A cache hit must reproduce the no-cache RunState bit-for-bit
     (the parse output depends only on the network file + `use_ion`).
  2. skip_chem_warmup — skipping the chem-RHS JIT warmup must leave the
     returned RunState byte-identical (the warmup result is discarded).
  3. $VULCAN_JAX_FASTCHEM_DIR — redirects FastChem's working tree when set
     before import (checked in a subprocess; no FastChem run needed).

All CPU; mirror `test_vmap_while_loop` for the fast photo-off setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _pin_cfg():
    """Pin vulcan_cfg for a fast, photo-off, FastChem-free pre-loop setup."""
    import vulcan_jax.legacy_io as op

    vulcan_cfg = op.default_config()

    vulcan_cfg.count_max = 40
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_print_prog = False
    vulcan_cfg.use_live_plot = False
    vulcan_cfg.use_live_flux = False
    vulcan_cfg.use_photo = False
    vulcan_cfg.use_ion = False
    vulcan_cfg.atm_type = "isothermal"
    vulcan_cfg.Kzz_prof = "Pfunc"
    vulcan_cfg.Tiso = 1000.0
    # These hooks are orthogonal to the initial-abundance solver, so use the
    # in-process `const_lowT` Newton solver instead of the FastChem subprocess
    # to keep the unit tests fast, quiet, and deterministic.
    vulcan_cfg.ini_mix = "const_lowT"
    return vulcan_cfg


def _build_rs(vulcan_cfg, *, skip_chem_warmup=False):
    from vulcan_jax.state import RunState

    return RunState.with_pre_loop_setup(vulcan_cfg, skip_chem_warmup=skip_chem_warmup)


def _rate_fields(rs) -> dict:
    """The rate-parse-derived fields that a cache miss-restore would corrupt."""
    return {
        "k": np.asarray(rs.rate.k, dtype=np.float64),
        "Rf": dict(rs.metadata.Rf),
        "n_branch": dict(rs.metadata.n_branch),
        "photo_sp": frozenset(rs.metadata.photo_sp),
    }


def _assert_rate_fields_equal(a: dict, b: dict) -> None:
    np.testing.assert_array_equal(a["k"], b["k"])
    assert a["Rf"] == b["Rf"]
    assert a["n_branch"] == b["n_branch"]
    assert a["photo_sp"] == b["photo_sp"]


def test_rate_parse_cache_matches_no_cache():
    """A warm cache hit reproduces the no-cache RunState's rate metadata."""
    import vulcan_jax.legacy_io as lio

    cfg = _pin_cfg()

    # Reference: cache OFF.
    os.environ["VULCAN_JAX_RATE_CACHE"] = "0"
    lio._RATE_PARSE_CACHE.clear()
    assert not lio._rate_cache_enabled()
    ref = _rate_fields(_build_rs(cfg))

    # Cache ON: first build fills the cache (miss), second restores it (hit).
    os.environ["VULCAN_JAX_RATE_CACHE"] = "1"
    lio._RATE_PARSE_CACHE.clear()
    assert lio._rate_cache_enabled()
    cold = _rate_fields(_build_rs(cfg))
    assert len(lio._RATE_PARSE_CACHE) == 1  # the parse got memoised
    warm = _rate_fields(_build_rs(cfg))

    _assert_rate_fields_equal(cold, ref)
    _assert_rate_fields_equal(warm, ref)  # the decisive one: restore == parse
    os.environ.pop("VULCAN_JAX_RATE_CACHE", None)


def test_skip_chem_warmup_runstate_identical():
    """skip_chem_warmup leaves every RunState array leaf byte-identical."""
    import jax

    cfg = _pin_cfg()
    rs_warm = _build_rs(cfg, skip_chem_warmup=False)
    rs_skip = _build_rs(cfg, skip_chem_warmup=True)

    # Drop `metadata` (its `start_time` leaf is a wall-clock time.time() that
    # differs per build) and `photo_static` (None here). The warmup touches
    # none of the RunState's runtime/atm/rate arrays, which is what we compare.
    leaves_warm = jax.tree_util.tree_leaves(
        rs_warm._replace(metadata=None, photo_static=None)
    )
    leaves_skip = jax.tree_util.tree_leaves(
        rs_skip._replace(metadata=None, photo_static=None)
    )
    assert len(leaves_warm) == len(leaves_skip)
    compared = 0
    for lw, ls in zip(leaves_warm, leaves_skip):
        aw = np.asarray(lw)
        if aw.dtype.kind not in ("f", "i", "u", "b"):
            continue
        np.testing.assert_array_equal(aw, np.asarray(ls))
        compared += 1
    assert compared > 0


def test_fastchem_dir_env_override(tmp_path):
    """$VULCAN_JAX_FASTCHEM_DIR redirects the FastChem working tree at import."""
    override = tmp_path / "fc_scratch"
    override.mkdir()
    env = dict(os.environ)
    env["VULCAN_JAX_FASTCHEM_DIR"] = str(override)
    code = (
        "import vulcan_jax.ini_abun as ia;"
        "print(str(ia._FC_DIR));"
        "print(str(ia._FC_BIN));"
        "print(str(ia._FC_VULCAN_EQ))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    fc_dir, fc_bin, fc_eq = out.stdout.split()
    assert Path(fc_dir).resolve() == override.resolve()
    assert Path(fc_bin) == override.resolve() / "fastchem"
    # All derived write paths live under the override (isolation guarantee).
    assert str(Path(fc_eq)).startswith(str(override.resolve()))


def test_fastchem_dir_default_is_package(tmp_path):
    """Without the env var, FastChem stays anchored to the package tree."""
    env = dict(os.environ)
    env.pop("VULCAN_JAX_FASTCHEM_DIR", None)
    code = (
        "import vulcan_jax.ini_abun as ia;"
        "print(str(ia._FC_DIR.name));"
        "print(str(ia._ROOT))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    name = out.stdout.split("\n")[0]
    assert name == "fastchem_vulcan"
