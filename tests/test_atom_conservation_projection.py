"""Lock the chemistry atom-conservation projection (`jax_step._project_chem_rhs`).

XLA's FMA fusion breaks the exact stoichiometric nullspace of the codegen RHS;
the projection distributes the residual across the reservoir species
(H2, H2O, CO, N2) so each layer conserves H/O/C/N. A VULCAN-JAX
correctness feature with no master analogue.

Runs on the always-available HD189 pre-loop state (the HD209 fixture-based
projection test skips on a fresh checkout). Pins:
  1. injecting an atom residual on a non-reservoir species and projecting
     drives the per-atom residual to ~machine-zero,
  2. the projection mutates ONLY the reservoir rows,
  3. the real codegen RHS is machine-zero conserving after projection.
"""

from __future__ import annotations

import numpy as np
from _helpers import atom_count_matrix

_ATOMS = ("H", "O", "C", "N")
_RESERVOIRS = ("H2", "H2O", "CO", "N2")
PROJECTION_TOL = 1e-12  # atom residual left by the projection, relative to its scale
LAPACK_RTOL = 1e-12  # the repair sweep vs lax.linalg.tridiagonal_solve


def test_projection_conserves_atoms_on_reservoir_rows_only(hd189_state):
    """An injected CH4 defect and the real codegen RHS both project to an atom
    residual under PROJECTION_TOL of their scale, changing reservoir columns only."""
    import jax.numpy as jnp
    import vulcan_jax.chem_funs as chem_funs
    import vulcan_jax.jax_step as jax_step

    # Active for the HD189 default: atoms H/O/C/N, all four reservoirs present.
    assert jax_step._CHEM_PROJECTION_ENABLED
    net = chem_funs._NETWORK
    atom_counts = atom_count_matrix(net, _ATOMS)
    reservoir_idx = [net.species_idx[sp] for sp in _RESERVOIRS]
    var, atm = hd189_state.var, hd189_state.atm
    raw = np.asarray(chem_funs.chem_rhs_codegen(
        jnp.asarray(var.y), jnp.asarray(atm.M), jnp.asarray(var.k_arr)))
    injected = raw.copy()
    injected[:, net.species_idx["CH4"]] += max(float(np.max(np.abs(raw))), 1.0)
    # The RHS spans ~29 orders of magnitude, so the residual is judged in
    # absolute terms against its scale (the injected defect, or the per-atom
    # production peak); a per-cell floor drowns in float64 cancellation noise.
    for rhs, scale in ((injected, np.max(np.abs(injected @ atom_counts))),
                       (raw, np.max(np.abs(raw) @ atom_counts))):
        proj = np.asarray(jax_step._project_chem_rhs(jnp.asarray(rhs)))
        assert scale > 0.0
        assert np.max(np.abs(proj @ atom_counts)) < PROJECTION_TOL * scale
        assert np.array_equal(np.delete(proj, reservoir_idx, axis=1),
                              np.delete(rhs, reservoir_idx, axis=1))


def _hd189_step_inputs(state):
    """(y, k_arr, atm_static, net_jax) from the HD189 pre-loop fixture state."""
    from vulcan_jax.config import default_config
    import vulcan_jax.chem_funs as chem_funs
    import vulcan_jax.jax_step as jax_step

    cfg = default_config()
    data_var, data_atm = state.var, state.atm
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape
    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=cfg)
    return y, np.asarray(data_var.k_arr, dtype=np.float64), atm_static, chem_funs._NET_JAX


def test_stage_vectors_satisfy_the_per_layer_element_identity(monkeypatch, hd189_state):
    """Both Ros2 stage vectors satisfy `c0 a^T k - a^T T k = a^T b_tr` in every
    layer at every dt, and with transport off a step changes no layer's element
    content by more than REPAIR_ABS_FLOOR of its density. The
    identity bar is set by the correction tridiagonal's conditioning (~T/c0),
    not by roundoff."""
    import jax
    import jax.numpy as jnp
    import vulcan_jax.jax_step as jax_step
    from _oracles import stage_defects

    y, k_arr, atm, net = _hd189_step_inputs(hd189_state)
    y, k_arr = jnp.asarray(y), jnp.asarray(k_arr)
    ac = jax_step._CHEM_ATOM_COUNTS
    zero = atm._replace(
        Kzz=0 * atm.Kzz, Dzz=0 * atm.Dzz, vz=0 * atm.vz, vm=0 * atm.vm, vs=0 * atm.vs
    )
    # These are the repair's contracts, so the trace-carrier guard
    # (_REPAIR_MAX_CELL_FRAC, test_stage_repair_guard.py) is lifted. Fresh
    # lambdas make jit trace after the patch (jax caches a trace per function).
    monkeypatch.setattr(jax_step, "_REPAIR_MAX_CELL_FRAC", float("inf"))
    defects = jax.jit(lambda *a: stage_defects(*a))
    step = jax.jit(lambda *a: jax_step.jax_ros2_step.__wrapped__(*a))
    identity = {}
    for dt in (1e8, 1e11, 1e13, 1e15):
        k1, k2, d1, d2, bound = defects(y, k_arr, jnp.float64(dt), atm, net)
        assert bool(jnp.all(jnp.isfinite(k1)) & jnp.all(jnp.isfinite(k2))), dt
        # Defect over the repair's bound; 1e3 covers the correction tridiagonal's conditioning at the 1e15 cap.
        identity[dt] = max(float(jnp.max(jnp.abs(d1) / bound)), float(jnp.max(jnp.abs(d2) / bound)))
    assert all(v < 1e3 for v in identity.values()), identity
    # The uncorrected element change per layer and atom stays under
    # REPAIR_ABS_FLOOR of the layer density at every dt (worst 4x over two
    # stages; bar 10x).
    from vulcan_jax.config import REPAIR_ABS_FLOOR
    for dt in (1e4, 1e6, 1e8, 1e11, 1e13, 1e15):
        sol, _ = step(y, k_arr, jnp.float64(dt), zero, net)
        change = jnp.abs(sol @ ac - y @ ac)                       # (nz, n_atoms)
        n_lay = jnp.sum(jnp.abs(sol), axis=1) + jnp.sum(jnp.abs(y), axis=1)
        worst = float(jnp.max(change / n_lay[:, None]))
        assert worst < 10.0 * REPAIR_ABS_FLOOR, (dt, worst)


def test_repair_tridiagonal_solve_is_lapack_gtsv_with_its_tangent(hd189_state):
    """`jax_step._tridiagonal_solve`, the repair's partial-pivoting sweep, is
    LAPACK dgtsv: backward stable on the HD189 repair matrices from dt 1e4 s to
    the 1e15 s cap, within 1e-12 of `lax.linalg.tridiagonal_solve` on
    swap-forcing systems, with a matching JVP and vmap over directions."""
    import jax
    import jax.numpy as jnp
    from jax.lax.linalg import tridiagonal_solve

    import vulcan_jax.jax_step as js

    def lapack(dl, d, du, g):
        return tridiagonal_solve(dl.T, d.T, du.T, g.T[:, :, None])[:, :, 0].T

    y, _, atm, _ = _hd189_step_inputs(hd189_state)
    y = jnp.asarray(y)
    ridx = js._CHEM_RESERVOIR_IDX
    A_e, B_e, C_e, A_m, B_m, C_m, _ = js._build_diff_coeffs_jax(y, atm, js.compute_diff_grav(atm))
    diag_d = A_e[:, None] + A_m
    pad = jnp.zeros((1, ridx.shape[0]))
    du = jnp.concatenate([-(B_e[:-1, None] + B_m[:-1])[:, ridx], pad])
    dl = jnp.concatenate([pad, -(C_e[1:, None] + C_m[1:])[:, ridx]])
    g = jax.random.normal(jax.random.PRNGKey(1), diag_d[:, ridx].shape)
    def backward_error(dl, d, du, g, x):
        # componentwise (Oettli-Prager) residual; ~1e-16 for a stable solve
        z = jnp.zeros((1, x.shape[1]))
        xu, xl = jnp.concatenate([x[1:], z]), jnp.concatenate([z, x[:-1]])
        res = jnp.abs(d * x + du * xu + dl * xl - g)
        return float(jnp.max(res / (jnp.abs(d * x) + jnp.abs(du * xu) + jnp.abs(dl * xl) + jnp.abs(g) + 1e-300)))

    def close(a, b, tol):
        return bool(jnp.all(jnp.isfinite(a))) and float(jnp.max(jnp.abs(a - b))) <= tol * float(jnp.max(jnp.abs(b)))

    # Ill-conditioned at large dt (~1e13 at the cap), so the invariant is the
    # sweep's own backward error, not agreement with another solver.
    for dt in (1e4, 1e8, 1e11, 1e13, 1e15):
        d = 1.0 / (js._ROS2_GAMMA * dt) - diag_d[:, ridx]
        x = js._tridiagonal_solve(dl, d, du, g)
        assert bool(jnp.all(jnp.isfinite(x))) and backward_error(dl, d, du, g, x) < 5e-15, dt

    # Well-conditioned, column-dominant, with five rows shrunk so |d_i| < |dl_{i+1}|
    # forces a swap there; plus the 3x3 whose unpivoted second pivot is exactly 0.
    nz, m = 62, 5
    k = jax.random.split(jax.random.PRNGKey(3), 3)
    du = -jnp.exp(jax.random.normal(k[0], (nz, m))).at[-1].set(0.0)
    dl = -jnp.exp(jax.random.normal(k[1], (nz, m))).at[0].set(0.0)
    zero = jnp.zeros((1, m))
    d = 1e-3 - jnp.concatenate([zero, du[:-1]]) - jnp.concatenate([dl[1:], zero])
    d = d.at[jnp.array([5, 20, 21, 40, 60])].multiply(0.05)
    g = jax.random.normal(k[2], (nz, m))
    assert bool(jnp.any(jnp.abs(d[:-1]) < jnp.abs(dl[1:])))
    assert close(js._tridiagonal_solve(dl, d, du, g), lapack(dl, d, du, g), LAPACK_RTOL)
    # independent directions (a common scaling of A and g has a zero tangent)
    tans = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), a.shape) * 1e-2
                 for i, a in enumerate((dl, d, du, g)))
    tans = (tans[0].at[0].set(0.0), tans[1], tans[2].at[-1].set(0.0), tans[3])
    t_new = jax.jvp(js._tridiagonal_solve, (dl, d, du, g), tans)[1]
    t_ref = jax.jvp(lapack, (dl, d, du, g), tans)[1]
    scale = float(jnp.max(jnp.abs(t_ref)))
    assert float(jnp.max(jnp.abs(t_new - t_ref))) < LAPACK_RTOL * scale
    dirs = jnp.stack([g * s for s in (1.0, -0.5, 0.25)])
    def solve_g(gg):
        return js._tridiagonal_solve(dl, d, du, gg)

    batched = jax.vmap(lambda v: jax.jvp(solve_g, (g,), (v,))[1])(dirs)
    assert close(batched, jnp.stack([jax.jvp(solve_g, (g,), (v,))[1] for v in dirs]), 1e-13)
    dl3, d3, du3 = (jnp.array(v)[:, None] for v in ((0.0, -1.0, 2.0), (2.0, 1.0, 4.0), (-2.0, -3.0, 0.0)))
    x3 = js._tridiagonal_solve(dl3, d3, du3, jnp.ones((3, 1)))
    assert bool(jnp.allclose(x3[:, 0], jnp.array([2.0, 1.5, -0.5]), rtol=0, atol=1e-15)), x3
