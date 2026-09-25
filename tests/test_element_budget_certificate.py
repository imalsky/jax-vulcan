"""The certificate's cumulative column element budget (C23).

`loss_eps` rejects a step whose unweighted atom sum JUMPS; a slow drain walks
past it (notes §1.12: a column certified having lost 21% of its sulfur).
`budget_ok` closes that: the run accumulates the per-step change of every
element's operator-weighted column, each step measured on the grid in force
during it, normalised by the fixed t=0 column (`budget_ref`), and may end only
within `element_budget_tol`. A geometry refresh moves no atom, so it never
enters the sum, and it cannot rescale a deficit already accumulated.

The carry is seeded directly (the stall-gate idiom): the chemistry criteria
are opened up so `conv_normal` holds on the first accepted step, leaving the
budget term as the only discriminator, and the run terminates after that one
step iff the column kept its elements.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")


def _cfg():
    """HD189 photo-off, non-hybrid, with the chemistry certificate opened up."""
    import vulcan_jax.legacy_io as op

    c = op.default_config()
    c.use_photo = False
    c.use_vm_mol = False
    c.use_hybrid_vm_mol = False
    c.use_print_prog = False
    c.count_min = 5
    # Isolate the budget term: any post-step longdy/longdydt passes the tight
    # branch, so `candidate` fires on the first accepted step.
    c.yconv_cri = 1.0e3
    c.slope_cri = 1.0e3
    return c


@pytest.mark.parametrize(
    "drain, scale, reason, why",
    [
        (1.0, 1.0, 1, "an intact column certifies on its first candidate step"),
        (4.0, 1.0, 3, "a column that lost 75% of its carbon may not certify"),
        (1.0, 0.9, 1, "a column whose every element moved by one factor certifies"),
    ],
)
def test_certificate_requires_the_column_element_budget(drain, scale, reason, why):
    """A drained column must fall through to the step-count exit, not certify.

    `drain` scales the CH4 rows of the column the run is treated as having
    started from, and `budget_err` is seeded with the relative change from
    that column to the present one: the run behaves as if it had lost that
    much carbon since t=0 (-0.73 of the C column at 4x, against
    `element_budget_tol` 1e-2). `scale` multiplies the whole starting column:
    the term is measured relative to H (0.10.1), so a common factor (the
    hydrostatic renormalisation) must still certify.
    """
    import vulcan_jax.legacy_io as op
    from vulcan_jax import op_jax, outer_loop
    from vulcan_jax.ini_abun import column_atoms
    from vulcan_jax.state import RunState

    c = _cfg()
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=c), cfg=c)
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(c))
    t0 = 10.0 * float(c.trun_min)
    ch4 = outer_loop._NETWORK.species_idx["CH4"]
    started_from = column_atoms(
        (state.y * scale).at[:, ch4].multiply(drain), state.dz, integ._compo_arr
    )
    now = column_atoms(state.y, state.dz, integ._compo_arr)
    seeded = state._replace(
        t=jnp.float64(t0),
        accept_count=jnp.int32(int(c.count_min) + 5),
        count_min_dyn=jnp.int32(int(c.count_min)),
        count_max_dyn=jnp.int32(int(c.count_min) + 10),
        runtime_dyn=jnp.float64(float(c.runtime)),
        # A lookback the step actually spans (the ring is zero-filled at t=0).
        t_time_ring=jnp.full((int(c.conv_step),), float(c.st_factor) * t0),
        aflux_change=jnp.float64(0.0),
        geom_ok=jnp.bool_(False),
        budget_ok=jnp.bool_(False),
        budget_err=now / started_from - 1.0,
    )
    final = integ._runner(seeded, atm_static)
    assert int(final.termination_reason) == reason, (
        f"{why}; got reason {int(final.termination_reason)}, budget_ok "
        f"{bool(final.budget_ok)}, geom_ok {bool(final.geom_ok)}"
    )
    assert bool(final.budget_ok) is (reason == 1)
    # The geometry term is satisfied in both cases: the budget is the only
    # thing that changed the outcome.
    assert bool(final.geom_ok)


@pytest.mark.parametrize(
    "err_c, reason, why",
    [
        (0.0, 1, "a refresh alone may not manufacture a budget error"),
        (-0.012, 3, "a refresh may not dilute an accumulated deficit into tolerance"),
    ],
)
def test_a_geometry_refresh_alone_never_moves_the_budget(err_c, reason, why):
    """A refresh keeps y and changes dz, so it moves no atom and the budget
    term must not move either -- in either direction.

    The carry enters on a stale grid, compressed where the column's mass sits;
    the refresh at the certificate candidate rebuilds the true dz and the
    discrete integral grows by roughly 3x. `err_c` is the carbon error
    accumulated before that refresh, stated twice: in `budget_err` (what the
    shipped fixed-denominator term reads) and as the matching offset of
    `budget_ref` (what a running reference would read at entry). A design that
    shifts the reference on the refresh rescales the deficit by the geometry
    factor -- -0.012 dilutes to -0.004 and certifies; the shipped term cannot.
    """
    import numpy as np

    import vulcan_jax.legacy_io as op
    from vulcan_jax import op_jax, outer_loop
    from vulcan_jax.ini_abun import column_atoms
    from vulcan_jax.state import RunState

    c = _cfg()
    c.geom_conv_tol = 1.0e9  # the geometry term is not under test here
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=c), cfg=c)
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(c))
    nz = state.y.shape[0]
    h = integ._atom_order.index("H")
    ci = integ._atom_order.index("C")
    dz_seed = state.dz / jnp.linspace(3.0, 1.0, nz)
    budget_err = jnp.zeros((integ._compo_arr.shape[1],)).at[ci].set(err_c)
    col_seed = column_atoms(state.y, dz_seed, integ._compo_arr)
    budget_ref = col_seed.at[ci].divide(1.0 + err_c)

    t0 = 10.0 * float(c.trun_min)
    seeded = state._replace(
        t=jnp.float64(t0),
        accept_count=jnp.int32(int(c.count_min) + 5),
        count_min_dyn=jnp.int32(int(c.count_min)),
        count_max_dyn=jnp.int32(int(c.count_min) + 10),
        runtime_dyn=jnp.float64(float(c.runtime)),
        t_time_ring=jnp.full((int(c.conv_step),), float(c.st_factor) * t0),
        aflux_change=jnp.float64(0.0),
        geom_ok=jnp.bool_(False),
        budget_ok=jnp.bool_(False),
        dz=dz_seed,
        budget_ref=budget_ref,
        budget_err=budget_err,
    )
    final = integ._runner(seeded, atm_static)
    assert int(final.termination_reason) == reason and bool(final.budget_ok) is (
        reason == 1
    ), (why, int(final.termination_reason), np.asarray(final.budget_drift))
    # The deficit survives the refresh undiluted.
    assert float(final.budget_drift[ci]) == pytest.approx(err_c, rel=0.05, abs=1e-4), (
        float(final.budget_drift[ci])
    )
    # The reference is fixed, and the stored drift is its accumulator read
    # relative to H.
    assert np.array_equal(np.asarray(final.budget_ref), np.asarray(budget_ref))
    err = np.asarray(final.budget_err)
    rel = (1.0 + err) / (1.0 + err[h]) - 1.0
    assert np.allclose(rel, np.asarray(final.budget_drift), rtol=0, atol=1e-12)


def test_shipped_configs_declare_the_budget_tolerance():
    """Every shipped config must DECLARE `element_budget_tol` (the runner has
    no default: a physics knob that silently appears is how a certificate
    stops meaning anything) and none may loosen it past the measured 1e-2."""
    undeclared, loose = [], []
    for path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        raw = yaml.safe_load(Path(path).read_text())
        if "element_budget_tol" not in raw:
            undeclared.append(os.path.basename(path))
        elif float(raw["element_budget_tol"]) > 1e-2:
            loose.append(os.path.basename(path))
    assert not undeclared and not loose, (undeclared, loose)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
