"""Gustafsson PI step-size controller (`use_pi_controller`, off by default).

Four guards, cheapest first:

1. Off-path identity — `_step_size` without PI args is bit-identical to the
   pre-PI I-controller formula, and the PI code path with the -1.0 no-history
   sentinel reproduces it bitwise (the runner's force-accept path relies on
   this).
2. neoVULCAN oracle — a NumPy mirror of neoVULCAN `ode_solver.step_size` /
   `step_reject` history semantics is driven through an accept / reject /
   zero-delta event sequence and the JAX controller must match every dt.
3. Forward-mode AD safety — jvp of `_step_size` w.r.t. delta is finite and
   FD-matched on all three branches (valid history, sentinel fallback,
   zero-delta substitution). The sharp edge is the sentinel branch: a naive
   `(delta_prev/delta)**(beta/2)` would put a NaN in the untaken select branch.
4. Runner-level — 50 accepted HD189 steps with the controller off vs on:
   the off run leaves the carry's `delta_prev` at the inert -1.0 sentinel,
   the on run takes a different dt trajectory (flag is live) with valid
   history, and both stay finite with baseline-scale atom loss.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

# Match the production defaults (vulcan_cfg / neoVULCAN).
RTOL = 0.2
DT_VAR_MIN = 0.2
DT_VAR_MAX = 2.0
DT_MIN = 1e-14
DT_MAX = 1e12
SAFETY = 0.9
ZERO_DELTA_FRAC = 0.01
PI_ALPHA = 0.7
PI_BETA = 0.4


def _numpy_step_size(dt, delta, delta_prev, use_pi):
    """NumPy mirror of neoVULCAN ode_solver.step_size (error_order p=2)."""
    if delta == 0.0:
        delta = ZERO_DELTA_FRAC * RTOL
    if use_pi and delta_prev > 0.0:
        h_factor = (
            SAFETY
            * (RTOL / delta) ** (PI_ALPHA / 2.0)
            * (delta_prev / delta) ** (PI_BETA / 2.0)
        )
    else:
        h_factor = SAFETY * (RTOL / delta) ** 0.5
    h_factor = min(max(h_factor, DT_VAR_MIN), DT_VAR_MAX)
    return min(max(dt * h_factor, DT_MIN), DT_MAX), delta  # (new dt, new delta_prev)


def _jax_step_size(dt, delta, delta_prev=None):
    import jax.numpy as jnp
    from vulcan_jax.outer_loop import _step_size

    kwargs = {}
    if delta_prev is not None:
        kwargs = dict(
            delta_prev=jnp.float64(delta_prev), pi_alpha=PI_ALPHA, pi_beta=PI_BETA
        )
    return _step_size(
        jnp.float64(dt),
        jnp.float64(delta),
        RTOL,
        DT_VAR_MIN,
        DT_VAR_MAX,
        DT_MIN,
        DT_MAX,
        SAFETY,
        ZERO_DELTA_FRAC,
        **kwargs,
    )


def test_pi_off_bit_identical_to_i_control():
    """No PI args == pre-PI formula; sentinel history == I-control, bitwise."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        dt = 10.0 ** rng.uniform(-12, 8)
        delta = 0.0 if rng.random() < 0.1 else 10.0 ** rng.uniform(-6, 1)
        base = float(_jax_step_size(dt, delta))
        sentinel = float(_jax_step_size(dt, delta, delta_prev=-1.0))
        expected, _ = _numpy_step_size(dt, delta, -1.0, use_pi=False)
        assert base == sentinel, (dt, delta)
        np.testing.assert_allclose(base, expected, rtol=1e-15)


def test_pi_matches_neovulcan_oracle_sequence():
    """dt trajectory through accept/reject/zero-delta events matches the mirror.

    Replays the runner's history rule: a kept step stores the effective delta,
    a reject stores the -1.0 sentinel (so the next accepted step is I-control),
    exactly neoVULCAN's step_size / step_reject split.
    """
    rng = np.random.default_rng(42)
    dt_np = dt_jax = 1e-8
    delta_prev = -1.0
    n_pi_engaged = 0
    for step in range(300):
        event = rng.random()
        if event < 0.15:  # rejection: dt *= dt_var_min, history invalidated
            dt_np = max(dt_np * DT_VAR_MIN, DT_MIN)
            dt_jax = max(dt_jax * DT_VAR_MIN, DT_MIN)
            delta_prev = -1.0
            continue
        delta = 0.0 if event > 0.95 else 10.0 ** rng.uniform(-4, np.log10(RTOL))
        if delta_prev > 0.0:
            n_pi_engaged += 1
        dt_np, delta_prev_np = _numpy_step_size(dt_np, delta, delta_prev, use_pi=True)
        dt_jax = float(_jax_step_size(dt_jax, delta, delta_prev=delta_prev))
        np.testing.assert_allclose(dt_jax, dt_np, rtol=1e-14, err_msg=f"step {step}")
        delta_prev = delta_prev_np
    assert n_pi_engaged > 100  # the PI branch was genuinely exercised


def test_pi_step_size_forward_ad_safe():
    """jvp w.r.t. delta is finite and FD-matched on all three branches."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def f(delta, delta_prev):
        return _jax_step_size(1e2, delta, delta_prev=delta_prev)

    # delta near rtol keeps h_factor inside the [dt_var_min, dt_var_max] clip
    # window, so the tangent is not zeroed by the clip.
    cases = [
        (0.15, 0.1),  # valid history -> PI branch (interior of the clip)
        (0.15, -1.0),  # sentinel -> I-control fallback (naive impl NaNs here)
        (0.0, 0.1),  # zero-delta substitution (tangent must not blow up)
    ]
    for delta, delta_prev in cases:
        g = float(
            jax.jvp(
                lambda d: f(d, jnp.float64(delta_prev)),
                (jnp.float64(delta),),
                (jnp.float64(1.0),),
            )[1]
        )
        assert np.isfinite(g), (delta, delta_prev, g)
        if delta > 0.0:  # FD check away from the substitution kink
            eps = delta * 1e-6
            fd = (
                float(f(delta + eps, delta_prev)) - float(f(delta - eps, delta_prev))
            ) / (2 * eps)
            np.testing.assert_allclose(
                g, fd, rtol=1e-5, err_msg=str((delta, delta_prev))
            )

    # History direction: tangent w.r.t. delta_prev flows only through the PI
    # branch and is exactly zero at the sentinel (no NaN leakage).
    g_hist = float(
        jax.jvp(
            lambda p: f(jnp.float64(0.15), p), (jnp.float64(0.1),), (jnp.float64(1.0),)
        )[1]
    )
    assert np.isfinite(g_hist) and g_hist != 0.0
    g_sent = float(
        jax.jvp(
            lambda p: f(jnp.float64(0.15), p), (jnp.float64(-1.0),), (jnp.float64(1.0),)
        )[1]
    )
    assert g_sent == 0.0


def _run_hd189(hd189_state, n_steps, use_pi):
    """Drive the inner runner for n_steps accepted HD189 steps; return final carry."""
    import vulcan_jax.legacy_io as op
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.jax_step import make_atm_static

    # Pin every vulcan_cfg reference to the object the fixture's pre-loop
    # setup used (same dance as test_outer_loop_smoke) so the runner statics
    # read the knobs set here.
    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = vulcan_cfg
    outer_loop.vulcan_cfg = vulcan_cfg
    vulcan_cfg.count_max = n_steps
    vulcan_cfg.count_min = 1
    vulcan_cfg.use_pi_controller = use_pi
    vulcan_cfg.use_print_prog = False

    st = hd189_state
    integ = outer_loop.OuterLoop(st.solver, st.output, cfg=vulcan_cfg)
    integ._ensure_runner(st.var, st.atm)
    ni = outer_loop._NETWORK.ni
    nz = int(np.asarray(st.atm.Tco).shape[0])
    atm_static = make_atm_static(st.atm, ni, nz, cfg=vulcan_cfg)
    init = integ._pack_state(st.var, st.para, st.atm)
    return integ._runner(init, atm_static)


@pytest.mark.strict_isolation
def test_pi_runner_hd189_50_steps(hd189_state):
    """Controller off vs on through the real JIT'd runner (50 HD189 steps).

    `_run_hd189` only reads the fixture state (`_pack_state` and the runner
    are pure), so both runs share the same per-test fixture copy.
    """
    final_off = _run_hd189(hd189_state, 50, use_pi=False)
    final_on = _run_hd189(hd189_state, 50, use_pi=True)

    for name, fin in (("off", final_off), ("on", final_on)):
        assert int(fin.accept_count) >= 50, name
        assert np.isfinite(float(fin.dt)) and float(fin.dt) > 0, name
        assert np.isfinite(float(fin.t)) and float(fin.t) > 0, name
        assert np.all(np.isfinite(np.asarray(fin.y))), name
        # HD189 is the smooth baseline: atom loss stays at the established
        # ~4.4e-5 scale (test_outer_loop_smoke tracks the tight bound).
        assert float(np.max(np.abs(np.asarray(fin.atom_loss)))) < 1e-3, name

    # Off: the carry's history slot is an inert pass-through (-1.0 seed).
    assert float(final_off.delta_prev) == -1.0
    # On: after a run of accepted steps the history holds the last accepted
    # effective delta (positive), and the controller actually changed the
    # trajectory.
    assert float(final_on.delta_prev) > 0.0
    assert float(final_on.dt) != float(final_off.dt)
    assert float(final_on.t) != float(final_off.t)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
