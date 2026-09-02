"""A non-finite state must never be scored as converged.

Regression guard for the defect where `_conv_jax`'s masks — all `<`/`>`
comparisons, which are False for NaN — silently dropped poisoned cells from
the `longdy` maximum, so an all-NaN state read `longdy == 0.0` (*perfectly*
converged) and the run reported `end_case=1` "Integration successful".
VULCAN-master cannot do this: its `np.amax(longdy[ymix>0]/ymix[ymix>0])`
(op.py:1055) reduces an empty selection and raises. The fix forces
`longdy = +inf` on any non-finite `y`/`ymix` cell, reproducing master's "can
never converge" semantics inside a jittable reduction.
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


def _longdy(y, ymix, y_old, n_0, *, atol=1e-2, mtol_conv=1e-20):
    """The shipped reduction, imported so deleting the guard fails this file."""
    from vulcan_jax.outer_loop import _longdy_reduce

    return _longdy_reduce(y, ymix, y_old, n_0, atol=atol, mtol_conv=mtol_conv)[0]


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

    Pre-fix, NaN-ing that cell took longdy from 0.03996 to 0.0.
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


def _delta(sol, delta_arr, ymix_old, *, atol=1e-2, mtol=1e-22):
    """The shipped step-acceptance reduction (same import rule as `_longdy`)."""
    from vulcan_jax.outer_loop import _make_aggregate_delta_fn

    agg = _make_aggregate_delta_fn(mtol, atol, False, jnp.zeros(sol.shape, dtype=bool))
    return agg(sol, delta_arr, ymix_old)


def test_sub_atol_cell_gives_a_finite_batch_independent_tangent():
    """A density that is positive but below atol (and above the 1e-300 floor)
    is masked out of both maxima, yet its cell passes the `> 0` guard. If the
    masked numerator is divided by the raw tiny denominator, the division's
    tangent is `0 * den**-2 = 0 * inf = NaN` there, and the `jnp.max` JVP
    multiplies tangents by a 0/1 indicator (it does not select), so the max's
    tangent is NaN -- except that XLA rewrites the unbatched multiply into a
    select and hides it. The reductions must not rely on that: plain and
    vmapped tangents are finite and equal (TOI-7169 b, S8 at 1e-160 cm^-3)."""
    y, ymix, y_old, n_0 = _mk()
    y = y.at[0, 1].set(1e-160)
    t = jnp.ones_like(y)

    def longdy_of(y_):
        ymix_ = y_ / jnp.sum(y_, axis=1, keepdims=True)
        return _longdy(y_, ymix_, y_old, n_0)

    def delta_of(sol):
        return _delta(sol, jnp.abs(sol - y_old) * 1e-3, ymix)

    def tangents(f):
        plain = jax.jvp(f, (y,), (t,))[1]
        batched = jax.vmap(lambda tt: jax.jvp(f, (y,), (tt,))[1])(t[None])[0]
        return float(plain), float(batched)

    for f in (longdy_of, delta_of):
        plain, batched = tangents(f)
        assert np.isfinite(plain) and np.isfinite(batched), (f.__name__, plain, batched)
        np.testing.assert_allclose(batched, plain, rtol=1e-12, err_msg=f.__name__)


def test_end_case_is_not_success_for_a_frozen_or_yielded_lane():
    """`termination_reason` 0 (chunk yield) and 5 (non-finite freeze) both
    stop below both caps, so neither cap fires and the fall-through used to
    report end_case=1 "Integration successful". Conversely a step that
    converges on the same step it hits count_max is a success (master's
    stop() tests convergence first), never end_case=3."""
    from vulcan_jax.outer_loop import OuterLoop

    class _S:
        accept_count, count_max_dyn = 10, 1000
        t, runtime_dyn = 1.0, 1e10
        y = jnp.ones((2, 2))

        def __init__(self, reason):
            self.termination_reason = reason

    clf = OuterLoop._classify_end_case
    assert clf(None, _S(1)) == 1
    for reason in (0, 5):
        assert clf(None, _S(reason)) == 5, f"reason {reason} reported as success"
    bad = _S(1)
    bad.y = jnp.asarray([[1.0, jnp.nan], [1.0, 1.0]])
    assert clf(None, bad) == 5
    at_cap = _S(1)
    at_cap.accept_count = at_cap.count_max_dyn + 1
    assert clf(None, at_cap) == 1, "converged at count_max must be a success"
    assert clf(None, _S(3)) == 3 and clf(None, _S(2)) == 2
