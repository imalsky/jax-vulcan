"""Lock the chemistry atom-conservation projection (`jax_step._project_chem_rhs`).

XLA's FMA fusion breaks the exact stoichiometric nullspace of the codegen RHS;
the projection distributes the residual across the reservoir species
(H2, H2O, CO, N2) so each layer conserves H/O/C/N exactly. A VULCAN-JAX
correctness feature with no master analogue (notes.md, atom-conservation
projection record).

Runs on the always-available HD189 pre-loop state (the HD209 fixture-based
projection test skips on a fresh checkout). Pins:
  1. injecting an atom residual on a non-reservoir species and projecting
     drives the per-atom residual to ~machine-zero,
  2. the projection mutates ONLY the reservoir rows,
  3. the real codegen RHS is machine-zero conserving after projection.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from vulcan_jax._paths import resolve_data_path

_ATOMS = ("H", "O", "C", "N")
_RESERVOIRS = ("H2", "H2O", "CO", "N2")


def _atom_count_matrix(net, atoms):
    """Species-by-atom stoichiometry, shape (ni, n_atoms). Reads with an
    explicit encoding so species names come back as str (not numpy.bytes_)."""
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    compo = np.genfromtxt(
        resolve_data_path(vulcan_cfg.com_file),
        names=True,
        dtype=None,
        encoding=None,
    )
    row_by_species = {str(row["species"]): row for row in compo}
    return np.asarray(
        [[float(row_by_species[sp][a]) for a in atoms] for sp in net.species],
        dtype=np.float64,
    )


def _capture_hd189_state():
    """Return (y, M, k_arr, net) from a fresh HD189 pre-loop pipeline run."""
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()
    import vulcan_jax.network as net_mod
    from vulcan_jax.state import RunState

    rs = RunState.with_pre_loop_setup(vulcan_cfg)
    net = net_mod.parse_network(vulcan_cfg.network)
    y = np.asarray(rs.step.y, dtype=np.float64)
    M = np.asarray(rs.atm.M, dtype=np.float64)
    k_arr = np.asarray(rs.rate.k, dtype=np.float64)
    return y, M, k_arr, net


def main() -> int:
    import jax.numpy as jnp
    import vulcan_jax.jax_step as jax_step
    import vulcan_jax.make_chem_funs as mcf

    y, M, k_arr, net = _capture_hd189_state()

    # The projection static tables must be enabled for the HD189 default
    # (atom_list = H/O/C/N and all four reservoir species present).
    assert jax_step._CHEM_PROJECTION_ENABLED, (
        "atom-conservation projection is disabled for the default config; "
        "expected it active for HD189 (atoms H/O/C/N, reservoirs H2/H2O/CO/N2)"
    )

    atom_counts = _atom_count_matrix(net, _ATOMS)
    reservoir_idx = [net.species_idx[sp] for sp in _RESERVOIRS]

    fn = mcf.build_chem_rhs(net)  # production JIT'd codegen RHS
    raw = np.asarray(fn(jnp.asarray(y), jnp.asarray(M), jnp.asarray(k_arr)))

    ok = True

    # --- 1. Operator test: inject a known conservation defect ---
    # Perturb a single non-reservoir species (CH4: C + 4H) by a magnitude
    # comparable to the RHS so the injected residual is unambiguously nonzero.
    inj_sp = (
        "CH4"
        if "CH4" in net.species_idx
        else next(sp for sp in net.species if sp not in _RESERVOIRS)
    )
    inj_idx = net.species_idx[inj_sp]
    delta = max(float(np.max(np.abs(raw))), 1.0)
    raw_def = raw.copy()
    raw_def[:, inj_idx] += delta

    resid_def = raw_def @ atom_counts  # (nz, n_atoms) — now nonzero
    projected = np.asarray(jax_step._project_chem_rhs(jnp.asarray(raw_def)))
    proj_resid = projected @ atom_counts

    # The RHS spans ~29 orders of magnitude, so judge conservation as the
    # ABSOLUTE residual relative to the injected defect (a per-cell relative
    # floor would drown in the ~1e14 float64 cancellation noise of 1e29-scale
    # terms). The projection should cut the residual by ~15 orders.
    inj_max = float(np.max(np.abs(resid_def)))
    proj_max = float(np.max(np.abs(proj_resid)))
    reduction = proj_max / max(inj_max, 1e-300)
    print(f"injected ({inj_sp}) max |residual|:  {inj_max:.3e}")
    print(f"projected max |residual|:        {proj_max:.3e}")
    print(f"residual reduction factor:       {reduction:.3e}")
    if not inj_max > 0.0:
        print("FAIL: injected residual is zero (vacuous test)")
        ok = False
    if not reduction < 1e-12:
        print("FAIL: projection did not drive the injected residual to the FP floor")
        ok = False

    # --- 2. Only reservoir rows are mutated by the projection ---
    non_reservoir_delta = np.delete(projected - raw_def, reservoir_idx, axis=1)
    max_non_reservoir = float(np.max(np.abs(non_reservoir_delta)))
    print(f"max |delta| outside reservoir species: {max_non_reservoir:.3e}")
    if max_non_reservoir != 0.0:
        print("FAIL: projection mutated a non-reservoir species")
        ok = False

    # --- 3. On the real RHS, projection conserves to the FP cancellation floor ---
    raw_proj = np.asarray(jax_step._project_chem_rhs(jnp.asarray(raw)))
    raw_proj_resid = raw_proj @ atom_counts
    # Judge against the per-atom peak production magnitude, not per cell.
    prod_peak = float(np.max(np.abs(raw) @ atom_counts))
    raw_proj_rel = float(np.max(np.abs(raw_proj_resid))) / max(prod_peak, 1e-300)
    real_non_reservoir = np.delete(raw_proj - raw, reservoir_idx, axis=1)
    print(f"real-RHS residual / production peak: {raw_proj_rel:.3e}")
    if raw_proj_rel > 1e-12:
        print("FAIL: real RHS not conserving to the FP floor after projection")
        ok = False
    if float(np.max(np.abs(real_non_reservoir))) != 0.0:
        print("FAIL: real-RHS projection mutated a non-reservoir species")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())


def _hd189_step_inputs():
    """(y, k_arr, atm_static, net_jax) from the HD189 pre-loop state."""
    from vulcan_jax.config import default_config
    import vulcan_jax.chem_funs as chem_funs
    import vulcan_jax.jax_step as jax_step
    from vulcan_jax.state import RunState, legacy_view

    cfg = default_config()
    rs = RunState.with_pre_loop_setup(cfg)
    data_var, data_atm, _ = legacy_view(rs)
    y = np.asarray(data_var.y, dtype=np.float64)
    nz, ni = y.shape
    atm_static = jax_step.make_atm_static(data_atm, ni, nz, cfg=cfg)
    return y, np.asarray(data_var.k_arr, dtype=np.float64), atm_static, chem_funs._NET_JAX


def test_stage_vectors_satisfy_the_per_layer_element_identity(monkeypatch):
    """Both Ros2 stage vectors satisfy `c0 a^T k - a^T T k = a^T b_tr` in every
    layer at every dt (the identity the stage system implies once the
    chemistry terms are projected), and with transport off a step conserves
    every layer's element content. Before the stage repair the identity
    failed by 6e-4 (dt 1e11), 6e-2 (1e13) and 3 (1e15) of the layer's own
    c0 |a^T y| on this column, which is what drained elements from long
    large-dt runs (notes.md §1.13). Ratios here are against the absolute
    size of the terms, so the floor is float64 roundoff for the closed-layer
    change; the identity residual after the repair is set by the correction
    tridiagonal's conditioning (~T/c0, growing with dt) and measured
    1e-13 at dt <= 1e11, 2e-13 to 1e-11 at the 1e15 cap depending on the
    rate table (2026-09-08, C20), so its bar sits 1e6 below the leak it
    guards, not at roundoff."""
    import jax
    import jax.numpy as jnp
    import vulcan_jax.jax_step as jax_step
    from _oracles import stage_defects

    y, k_arr, atm, net = _hd189_step_inputs()
    y, k_arr = jnp.asarray(y), jnp.asarray(k_arr)
    ac = jax_step._CHEM_ATOM_COUNTS
    zero = atm._replace(
        Kzz=0 * atm.Kzz, Dzz=0 * atm.Dzz, vz=0 * atm.vz, vm=0 * atm.vm, vs=0 * atm.vs
    )
    # Both contracts below are the REPAIR's, so the trace-carrier guard is
    # LIFTED: `jax_step._REPAIR_MAX_CELL_FRAC` (pinned by
    # test_stage_repair_guard.py) deliberately leaves a layer's identity open
    # where the fixed reservoir is a trace -- on this column the 1e-24-VMR H2O
    # of the 6000 K top, where the correction is 1e10 to 1e16 times the cell.
    # Both callees are jitted HERE through fresh lambdas so they trace after
    # the lift: jax caches the traced jaxpr per Python function, so re-jitting
    # `jax_ros2_step.__wrapped__` itself would reuse a trace made with the
    # shipped value if anything compiled it earlier in this process.
    monkeypatch.setattr(jax_step, "_REPAIR_MAX_CELL_FRAC", float("inf"))
    defects = jax.jit(lambda *a: stage_defects(*a))
    step = jax.jit(lambda *a: jax_step.jax_ros2_step.__wrapped__(*a))
    identity, closed = {}, {}
    for dt in (1e8, 1e11, 1e13, 1e15):
        k1, k2, d1, d2, bound = defects(y, k_arr, jnp.float64(dt), atm, net)
        assert bool(jnp.all(jnp.isfinite(k1)) & jnp.all(jnp.isfinite(k2))), dt
        # Residual defect over the bound the repair leaves it under (the
        # roundoff of the terms or REPAIR_ABS_FLOOR of the layer density);
        # 1e3 x roundoff covers the correction tridiagonal's conditioning
        # at the 1e15 cap (1e-11 of the terms measured, 2026-09-08).
        identity[dt] = max(float(jnp.max(jnp.abs(d1) / bound)), float(jnp.max(jnp.abs(d2) / bound)))
        sol, _ = step(y, k_arr, jnp.float64(dt), zero, net)
        closed[dt] = float(
            jnp.max(jnp.abs(sol @ ac - y @ ac) / (jnp.abs(sol) @ ac + jnp.abs(y) @ ac))
        )
    print("identity residual / bound:", {f"{d:g}": f"{v:.1e}" for d, v in identity.items()})
    print("closed-layer element change / |content|:",
          {f"{d:g}": f"{v:.1e}" for d, v in closed.items()})
    assert all(v < 1e3 for v in identity.values()), identity
    # The repair runs at every dt (no gate since 0.6.0) and leaves alone a
    # defect under config.REPAIR_ABS_FLOOR of the layer's density (roundoff
    # by measurement; correcting it stalled a small-dt column by moving
    # 1e-21 of a layer onto a 1e-21 H2S cell every stage, notes.md §1.13).
    # Contract: per layer and atom the uncorrected element change is
    # bounded by that floor times the layer density (measured <= 4x the
    # floor over two stages; 10x here), at every dt including the small
    # ones the 0.5.1 gate used to skip (LU error 1e-9 at 1e4 s, 3e-7 at
    # 1e6 s relative to content, both corrected now).
    from vulcan_jax.config import REPAIR_ABS_FLOOR
    for dt in (1e4, 1e6, 1e8, 1e11, 1e13, 1e15):
        sol, _ = step(y, k_arr, jnp.float64(dt), zero, net)
        change = jnp.abs(sol @ ac - y @ ac)                       # (nz, n_atoms)
        n_lay = jnp.sum(jnp.abs(sol), axis=1) + jnp.sum(jnp.abs(y), axis=1)
        worst = float(jnp.max(change / n_lay[:, None]))
        assert worst < 10.0 * REPAIR_ABS_FLOOR, (dt, worst)


def test_repair_tridiagonal_solve_is_lapack_gtsv_with_its_tangent():
    """`jax_step._tridiagonal_solve`, the batched partial-pivoting sweep
    the stage repair uses instead of `lax.linalg.tridiagonal_solve`
    (a per-system cuSPARSE call on the GPU, notes.md §2.9), IS LAPACK dgtsv:
    backward stable (componentwise residual < 5e-15) on the HD189 repair
    matrices `c0 - T_rho` at every dt from 1e4 s to the 1e15 s cap, and
    within 1e-12 of the CPU primitive on systems that force row
    swaps, where an unpivoted sweep meets an exact zero pivot (a wrong-sign
    sub-diagonal from the central-difference drift can do that, notes.md
    §1.13). Its tangent, taken through the sweep's `where`s, matches the
    primitive's JVP rule (a second solve) and a dense solve on a
    well-conditioned swap-forcing system; a vmap over directions is the
    per-direction result to roundoff (the retrieval's 6-direction
    program)."""
    import jax
    import jax.numpy as jnp
    from jax.lax.linalg import tridiagonal_solve

    import vulcan_jax.jax_step as js

    def lapack(dl, d, du, g):
        return tridiagonal_solve(dl.T, d.T, du.T, g.T[:, :, None])[:, :, 0].T

    y, _, atm, _ = _hd189_step_inputs()
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

    # The real systems are ill-conditioned at large dt (~1e13 at the cap), so
    # two exact solvers may differ there; the platform-free invariant is the
    # sweep's own backward error (measured 1.1e-16 to 2.9e-16 on the Mac).
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
    assert close(js._tridiagonal_solve(dl, d, du, g), lapack(dl, d, du, g), 1e-12)
    # independent directions (a common scaling of A and g has a zero tangent)
    tans = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), a.shape) * 1e-2
                 for i, a in enumerate((dl, d, du, g)))
    tans = (tans[0].at[0].set(0.0), tans[1], tans[2].at[-1].set(0.0), tans[3])
    t_new = jax.jvp(js._tridiagonal_solve, (dl, d, du, g), tans)[1]
    t_ref = jax.jvp(lapack, (dl, d, du, g), tans)[1]
    scale = float(jnp.max(jnp.abs(t_ref)))
    assert float(jnp.max(jnp.abs(t_new - t_ref))) < 1e-12 * scale
    dirs = jnp.stack([g * s for s in (1.0, -0.5, 0.25)])
    solve_g = lambda gg: js._tridiagonal_solve(dl, d, du, gg)
    batched = jax.vmap(lambda v: jax.jvp(solve_g, (g,), (v,))[1])(dirs)
    assert close(batched, jnp.stack([jax.jvp(solve_g, (g,), (v,))[1] for v in dirs]), 1e-13)
    dl3, d3, du3 = (jnp.array(v)[:, None] for v in ((0.0, -1.0, 2.0), (2.0, 1.0, 4.0), (-2.0, -3.0, 0.0)))
    x3 = js._tridiagonal_solve(dl3, d3, du3, jnp.ones((3, 1)))
    assert bool(jnp.allclose(x3[:, 0], jnp.array([2.0, 1.5, -0.5]), rtol=0, atol=1e-15)), x3
