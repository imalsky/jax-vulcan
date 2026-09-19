"""Unit test for the host-setup parallelism hook.

This backs the GPU-batched emulator's parallel host setup (one spawn worker
per profile): skipping the chem-RHS JIT warmup must leave the returned
RunState byte-identical, because the warmup result is discarded.

All CPU; mirrors `test_vmap_while_loop` for the fast photo-off setup.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from _helpers import fast_cfg

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _pin_cfg():
    """Pin vulcan_cfg for a fast, photo-off pre-loop setup.

    This hook is orthogonal to the initial-abundance solver, so use the
    5-mol `const_lowT` Newton solve instead of the full equilibrium seed to
    keep the unit test fast and quiet.
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
