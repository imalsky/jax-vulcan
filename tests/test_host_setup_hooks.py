"""Unit tests for the host-setup parallelism hooks.

These back the GPU-batched emulator's parallel host setup (one spawn worker per
profile, each with a private FastChem tree). Two independent, additive hooks:

  1. skip_chem_warmup — skipping the chem-RHS JIT warmup must leave the
     returned RunState byte-identical (the warmup result is discarded).
  2. $VULCAN_JAX_FASTCHEM_DIR — redirects FastChem's working tree when set
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
from _helpers import fast_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _pin_cfg():
    """Pin vulcan_cfg for a fast, photo-off, FastChem-free pre-loop setup.

    These hooks are orthogonal to the initial-abundance solver, so use the
    in-process `const_lowT` Newton solver instead of the FastChem subprocess
    to keep the unit tests fast, quiet, and deterministic.
    """
    return fast_cfg(count_max=40, Tiso=1000.0, ini_mix="const_lowT")
def _build_rs(vulcan_cfg, *, skip_chem_warmup=False):
    from vulcan_jax.state import RunState

    return RunState.with_pre_loop_setup(vulcan_cfg, skip_chem_warmup=skip_chem_warmup)


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
