"""Regenerate the two adjoint-state oracle fixtures under tests/data/.

Writes `adj_state_hd189.npz` (HD189, photo OFF, C-H-N-O network) and
`adj_state_w39b.npz` (WASP-39 b, photo ON, SNCHO network): a converged state
plus the operator inputs the reverse-mode adjoint tests replay, so the adjoint
linear algebra can be iterated without re-converging a run.

The network is import-locked, so the two states cannot be built in one
process; each case runs in its own process (`tests/gen_fixtures.py` handles
the fan-out; running this module directly builds exactly one case).

Run from VULCAN-JAX/:
    python tests/_gen_adj_state.py hd189 [--out DIR]
    python tests/_gen_adj_state.py w39b  [--out DIR]

Prefer `python tests/gen_fixtures.py --all`, which drives both plus the two
photo fixtures and writes the provenance manifest.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "data"

# HD189 adjoint-state knobs. DT is the body-map step the adjoint probe replays;
# N_POLISH extra normalized-map steps tighten the fixed point well past what the
# runner's own convergence criterion guarantees (the adjoint linearizes AT this
# point, so a loose fixed point shows up as adjoint residual, not as error here).
DT = 1e8
N_POLISH = 40

# The W39b state needs the sulfur network and its atom list. These are
# import-locked env vars: they must be set BEFORE the first `import vulcan_jax`,
# which is why this module imports jax lazily inside the builders.
W39B_NETWORK = "thermo/SNCHO_photo_network.txt"
W39B_ATOM_LIST = "H,O,C,N,S"


def _atm_step_arrays(atm_step) -> dict:
    """Split an AtmStatic into npz-safe array and bool groups."""
    fields = atm_step._fields
    arrays = {f"atm__{f}": np.asarray(getattr(atm_step, f))
              for f in fields if not isinstance(getattr(atm_step, f), bool)}
    bools = {f"atmbool__{f}": np.asarray(getattr(atm_step, f))
             for f in fields if isinstance(getattr(atm_step, f), bool)}
    return {**arrays, **bools}


def build_hd189(out_dir: Path) -> Path:
    """Converge HD189 with photochemistry OFF and dump the polished state."""
    t0 = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(ROOT)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from vulcan_jax.config import load_config

    cfg = load_config("default")
    cfg.use_live_plot = cfg.use_live_flux = cfg.use_print_prog = False
    cfg.use_photo = False
    cfg.yconv_cri = 1e-3
    cfg.count_max = 3000
    cfg.runtime = 1e14

    from vulcan_jax import chem as chem_mod, composition, network as net_mod
    from vulcan_jax.jax_step import jax_ros2_step, make_atm_static
    from vulcan_jax.state import RunState, legacy_view
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    rs = RunState.with_pre_loop_setup(cfg)
    network = net_mod.parse_network(cfg.network)
    net_jax = chem_mod.to_jax(network)
    _var, atm_d, para = legacy_view(rs)
    nz = atm_d.Tco.shape[0]
    ni = network.ni

    solver = op_jax.Ros2JAX()
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    _ = integ(rs)
    atm_static = make_atm_static(atm_d, ni, nz, cfg=integ._cfg)
    state0 = integ._pack_state_from_runstate(rs)
    final = integ._runner(state0, atm_static)
    y0 = final.y
    atm_step = atm_static._replace(g=final.g, dzi=final.dzi, Hpi=final.Hpi,
                                   top_flux=final.top_flux, vs=final.vs)
    kj = jnp.asarray(state0.k_arr)
    M_col = atm_step.M[:, None]
    dtj = jnp.float64(DT)

    # Normalized body map: matches the outer loop's post-step renormalize, whose
    # fixed point is much tighter than the bare step's.
    @jax.jit
    def G(y):
        sol, _ = jax_ros2_step(y, kj, dtj, atm_step, net_jax)
        return M_col * sol / jnp.sum(sol, axis=1, keepdims=True)

    y_star = y0
    fp = float("inf")
    for i in range(N_POLISH):
        y_next = G(y_star)
        fp = float(jnp.max(jnp.abs(y_next - y_star))
                   / jnp.maximum(jnp.max(jnp.abs(y_star)), 1e-300))
        y_star = y_next
        if i % 5 == 0 or fp < 1e-12:
            print(f"  polish {i:3d} fp_err={fp:.2e}", flush=True)
        if fp < 1e-13:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adj_state_hd189.npz"
    np.savez(
        path,
        y_star=np.asarray(y_star),
        k_arr=np.asarray(kj),
        dz=np.asarray(atm_d.dz),
        compo=np.asarray(composition.compo_array)[:ni],
        dt=DT, nz=nz, ni=ni,
        **_atm_step_arrays(atm_step),
    )
    print(f"[hd189] wrote {path} (nz={nz} ni={ni} fp_err={fp:.2e}) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return path


def build_w39b(out_dir: Path) -> Path:
    """Converge WASP-39 b with photochemistry ON and dump the frozen-photo state.

    Photo is held FROZEN at its converged `k_arr` (the rate table including the
    photolysis rows the runner populated), so the solver-map step-vjp gives the
    frozen-photo adjoint: the leading-order thermochemistry sensitivity of SO2.
    The dJ/dy feedback is second order and deliberately omitted.
    """
    t0 = time.time()
    # Import-locked: must precede the first `import vulcan_jax` in this process.
    os.environ["VULCAN_JAX_NETWORK"] = W39B_NETWORK
    os.environ["VULCAN_JAX_ATOM_LIST"] = W39B_ATOM_LIST
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(ROOT)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from vulcan_jax.config import load_config

    cfg = load_config("W39b")
    cfg.use_live_plot = cfg.use_live_flux = cfg.use_print_prog = False
    cfg.yconv_cri = 1e-3

    from vulcan_jax import composition, network as net_mod
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.state import RunState, legacy_view
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    rs = RunState.with_pre_loop_setup(cfg)
    _var, atm, para = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    nz = atm.Tco.shape[0]
    ni, nr = network.ni, network.nr
    sidx = network.species_idx
    if "SO2" not in sidx:
        raise RuntimeError(
            f"SO2 is not in the network parsed from {cfg.network!r} "
            f"({ni} species). The W39b adjoint fixture is an SO2 sensitivity "
            "state; check $VULCAN_JAX_NETWORK.")
    so2 = sidx["SO2"]
    print(f"[w39b] setup {time.time() - t0:.1f}s nz={nz} ni={ni} nr={nr} "
          f"SO2@{so2}", flush=True)

    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    _ = integ(rs)
    atm_static = make_atm_static(atm, ni, nz, cfg=cfg)
    state0 = integ._pack_state_from_runstate(rs)
    final = integ._runner(state0, atm_static)
    y_star = final.y
    k_frozen = final.k_arr
    atm_step = atm_static._replace(g=final.g, dzi=final.dzi, Hpi=final.Hpi,
                                   top_flux=final.top_flux, vs=final.vs)
    so2_mid = float(y_star[nz // 2, so2] / jnp.sum(y_star[nz // 2]))
    print(f"[w39b] converged {time.time() - t0:.1f}s SO2(mid)={so2_mid:.2e}",
          flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adj_state_w39b.npz"
    np.savez(
        path,
        y_star=np.asarray(y_star),
        k_arr=np.asarray(k_frozen),
        dz=np.asarray(atm.dz),
        compo=np.asarray(composition.compo_array)[:ni],
        nz=nz, ni=ni, nr=nr, so2=so2,
        **_atm_step_arrays(atm_step),
    )
    print(f"[w39b] wrote {path} ({time.time() - t0:.1f}s)", flush=True)
    return path


BUILDERS = {"hd189": build_hd189, "w39b": build_w39b}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=sorted(BUILDERS),
                    help="which adjoint state to build (one per process: the "
                         "reaction network is import-locked)")
    ap.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = ap.parse_args(argv)
    BUILDERS[args.case](args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
