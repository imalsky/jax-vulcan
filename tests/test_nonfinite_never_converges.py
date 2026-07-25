"""A non-finite state must never be scored as converged.

Regression guard for the defect where `_conv_jax`'s masks — all `<`/`>`
comparisons, which are False for NaN — silently dropped poisoned cells from
the `longdy` maximum. An all-NaN state therefore read `longdy == 0.0`, i.e.
*perfectly* converged, and the run reported `end_case=1` "Integration
successful". VULCAN-master cannot do this: its
`np.amax(longdy[ymix>0]/ymix[ymix>0])` (op.py:1053) reduces an empty
selection and raises.

The fix forces `longdy = +inf` whenever `y` or `ymix` carries a non-finite
cell, reproducing master's "can never converge" semantics inside a jittable
reduction. Both convergence routes must stay blocked: the normal branch
(`longdy < yconv_cri`) and the stall fallback (which requires
`longdy < yconv_min`).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

jax.config.update("jax_enable_x64", True)

_UNDERFLOW_DENOM = 1e-300


def _longdy(y, ymix, y_old, n_0, *, atol=1e-2, mtol_conv=1e-20):
    """Mirror of `outer_loop._conv_jax`'s longdy reduction, including the
    non-finite guard. Kept in-test because `_conv_jax` is a closure built
    inside `_build_runner` and is not importable on its own."""
    longdy_arr = jnp.abs((y - y_old) / n_0[:, None])
    longdy_arr = jnp.where(ymix < mtol_conv, 0.0, longdy_arr)
    longdy_arr = jnp.where(y < atol, 0.0, longdy_arr)
    ratio = jnp.where(ymix > 0, longdy_arr / jnp.maximum(ymix, _UNDERFLOW_DENOM), 0.0)
    longdy = jnp.max(ratio)
    state_is_bad = ~jnp.all(jnp.isfinite(y)) | ~jnp.all(jnp.isfinite(ymix))
    return jnp.where(state_is_bad, jnp.inf, longdy)


def _mk(nz=3, ni=4):
    n_0 = jnp.asarray(np.full(nz, 1e15))
    y = jnp.asarray(np.full((nz, ni), 1e13))
    # y_old differs in one abundant cell => genuinely unconverged
    y_old = jnp.asarray(np.full((nz, ni), 1e13)).at[1, 2].set(1e10)
    ymix = y / jnp.sum(y, axis=1, keepdims=True)
    return y, ymix, y_old, n_0


def test_healthy_unconverged_state_is_not_converged():
    """Baseline: the guard must not disturb a finite state's score."""
    y, ymix, y_old, n_0 = _mk()
    longdy = float(_longdy(y, ymix, y_old, n_0))
    assert np.isfinite(longdy)
    assert longdy > 0.01, f"expected a clearly-unconverged score, got {longdy}"


def test_single_nan_cell_cannot_improve_the_score():
    """Poisoning the one cell carrying the signal must not erase it.

    This is the exact regression: pre-fix, NaN-ing the offending cell took
    longdy from 0.03996 (correctly unconverged) to 0.0 (perfectly converged).
    """
    y, ymix, y_old, n_0 = _mk()
    before = float(_longdy(y, ymix, y_old, n_0))
    y_bad = y.at[1, 2].set(jnp.nan)
    ymix_bad = y_bad / jnp.sum(y_bad, axis=1, keepdims=True)
    after = float(_longdy(y_bad, ymix_bad, y_old, n_0))
    assert after == np.inf, (
        f"a NaN cell must force longdy=inf; got {after} (healthy score was {before})"
    )
    assert not (after < before), "NaN must never improve the convergence score"


def test_all_nan_state_is_never_converged():
    """The headline case: an entirely poisoned state must not report success."""
    y, ymix, y_old, n_0 = _mk()
    nz, ni = y.shape
    y_nan = jnp.full((nz, ni), jnp.nan)
    ymix_nan = y_nan / jnp.sum(y_nan, axis=1, keepdims=True)
    longdy = float(_longdy(y_nan, ymix_nan, y_old, n_0))
    assert longdy == np.inf, f"all-NaN state scored longdy={longdy}"

    # Both convergence routes must stay shut (outer_loop._convergence_ok).
    yconv_cri, yconv_min = 0.01, 0.1
    assert not (longdy < yconv_cri), "normal convergence branch admitted a NaN state"
    assert not (longdy < yconv_min), "stall fallback admitted a NaN state"


def test_inf_state_is_never_converged():
    """+/-inf must be caught by the same guard, not just NaN."""
    y, ymix, y_old, n_0 = _mk()
    for bad in (jnp.inf, -jnp.inf):
        longdy = float(_longdy(y.at[0, 0].set(bad), ymix, y_old, n_0))
        assert longdy == np.inf, f"y={bad} gave longdy={longdy}"


def test_nonfinite_ymix_alone_is_caught():
    """A finite y with a poisoned ymix must also be refused."""
    y, ymix, y_old, n_0 = _mk()
    longdy = float(_longdy(y, ymix.at[0, 0].set(jnp.nan), y_old, n_0))
    assert longdy == np.inf, f"NaN in ymix alone gave longdy={longdy}"
