"""A non-finite state must never be scored as converged.

`<`/`>` masks are False for NaN. Without a guard, a poisoned cell drops out
of the longdy maximum and an all-NaN state scores 0 (converged). Master's
`np.amax(longdy[ymix>0]/...)` (op.py:1055) raises on the empty selection;
`_longdy_reduce` instead forces +inf on any non-finite y/ymix cell.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _poisoned(case, y, ymix):
    """`(y, ymix)` with the non-finite cells `case` names."""
    if case in ("nan_signal_cell", "all_nan"):
        y = y.at[1, 2].set(jnp.nan) if case == "nan_signal_cell" else jnp.full(y.shape, jnp.nan)
        return y, y / jnp.sum(y, axis=1, keepdims=True)
    if case == "nan_ymix_only":
        return y, ymix.at[0, 0].set(jnp.nan)
    return y.at[0, 0].set(jnp.inf if case == "pos_inf" else -jnp.inf), ymix


@pytest.mark.parametrize(
    "case", ["nan_signal_cell", "all_nan", "pos_inf", "neg_inf", "nan_ymix_only"]
)
def test_nonfinite_state_is_never_converged(case):
    """Any non-finite y or ymix cell forces longdy = +inf, so neither
    convergence branch can admit it; a NaN on the one signal-carrying cell
    must not erase the signal."""
    y, ymix, y_old, n_0 = _mk()
    longdy = float(_longdy(*_poisoned(case, y, ymix), y_old, n_0))
    assert longdy == np.inf, f"{case}: longdy={longdy}"


def _delta(sol, delta_arr, ymix_old, *, atol=1e-2, mtol=1e-22):
    """The shipped step-acceptance reduction (same import rule as `_longdy`)."""
    from vulcan_jax.outer_loop import _make_aggregate_delta_fn

    agg = _make_aggregate_delta_fn(mtol, atol, False, jnp.zeros(sol.shape, dtype=bool))
    return agg(sol, delta_arr, ymix_old)


def test_sub_atol_cell_gives_a_finite_batch_independent_tangent():
    """A positive sub-atol cell (TOI-7169 b S8 at 1e-160) must give finite
    plain and vmapped tangents that agree. A masked numerator over the raw tiny
    denominator gives 0 * inf in the batched max tangent, which the unbatched
    select rewrite hides."""
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
    """`termination_reason` 0 (still running) and 5 (non-finite freeze) both
    stop below both caps, so neither cap fires and a fall-through would
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


_COUNT_MAX = 8  # small: the control run must exhaust it in seconds


@pytest.mark.strict_isolation
@pytest.mark.parametrize("poison,want_reason", [(False, 3), (True, 5)])
def test_single_profile_run_stops_on_a_nonfinite_state(poison, want_reason):
    """End to end: the single-profile `cond_fn` must bail on a non-finite
    state with the batched path's reason 5 (`body_fn_batch`) instead of
    burning the whole step budget. The finite case is the control: unchanged
    predicate, so it still runs out its budget (reason 3).
    """
    from _helpers import fast_cfg

    from vulcan_jax import legacy_io, op_jax
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.state import RunState

    cfg = fast_cfg(
        count_max=_COUNT_MAX, Tiso=900.0, use_vm_mol=False, use_hybrid_vm_mol=False
    )
    integ = OuterLoop(op_jax.Ros2JAX(), legacy_io.Output())
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))
    if poison:
        # Poison y AND y_prev: a rejected Ros2 step reverts y to y_prev, so
        # poisoning y alone self-heals (same trick as test_vmap_while_loop).
        state = state._replace(
            y=state.y.at[0, 0].set(jnp.nan),
            y_prev=state.y_prev.at[0, 0].set(jnp.nan),
        )
    final = integ._runner(state, atm_static)
    steps = int(final.accept_count)
    assert int(final.termination_reason) == want_reason, (
        f"poison={poison}: reason {int(final.termination_reason)} after "
        f"{steps} steps (want {want_reason})"
    )
    assert (steps <= 1) if poison else (steps > _COUNT_MAX), (
        f"poison={poison}: {steps} accepted steps against count_max "
        f"{_COUNT_MAX}"
    )
