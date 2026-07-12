"""Reverse-mode reaction ranking on a real HD189 column.

`steady_state_grad.steady_state_reaction_sensitivity` returns row-wise
`dL/d(ln k_r)` for every directional rate-table entry in one adjoint solve:
"which rows set the converged abundance of species X". By finite differences
this ranking would cost one re-converged model per row; reverse-mode returns
all of them at once.

The result is the mean over an ensemble of twin solves (default n_solves=3,
body_dt=1e7; the twin spread printed below is a magnitude-stability diagnostic).
This example runs both the default `solver_map="renorm"` and the legacy
`solver_map="bare"`: the default linearizes the hydrostatic-renormalized map the
runner actually iterates, so `y_star` is a tight fixed point of it and the CH4
rows are sub-percent vs the finite-difference anchors, whereas the raw-step
`"bare"` map is ~6-7% off (2026-07-03). On photochemistry-on columns pass
`photo_recompute_k` (see `make_photo_recompute_k`) to also include the dJ/dy
feedback and reach percent level on photo-coupled rows (e.g. WASP-39b OH+H2,
~11% -> ~0.2%); it is the standard companion to the renorm default there. The
function returns `k`-only directional-row sensitivities; for a physical
detailed-balance perturbation of a reversible thermal reaction, sum the forward
and reverse rows. Forward-mode (`grad_jvp_example.py`) is exact and stays the
route for any single hard row. See the `steady_state_reaction_sensitivity`
docstring and `scan_body_dt_reaction_sensitivity` for more.

This script loads a saved converged HD189 (photo-off) state
(`tests/data/adj_state_hd189.npz`, a local artifact — `*.npz` is gitignored) so
it does not re-converge the forward model. To produce that dump, converge
the runner and polish `y_star` to a tight fixed point of the renormalized body
map (see `jax_paper/scripts/adj_save_state.py`); to run it on WASP-39b SO2, set
`$VULCAN_JAX_NETWORK`/`$VULCAN_JAX_ATOM_LIST` before the first import and pick a
log10(SO2 VMR) loss at the peak-SO2 layer (see `adj_w39b_so2.py`).

Wall time is dominated by the one-time step-VJP XLA compile (~10-20 min cold;
wrap in `caffeinate -dimsu` on macOS so App Nap does not throttle it). Run from
VULCAN-JAX/ as `python examples/grad_reverse_example.py`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

FIXTURE = ROOT / "tests" / "data" / "adj_state_hd189.npz"

# Centered-FD truth for the HD189 CH4 loss (jax_paper/scripts/adj_solvermap_gmres.py).
FD_ANCHORS = {13: -5.651e-01, 14: +5.651e-01, 115: -1.919e-05, 116: +2.712e-05}


def main() -> int:
    if not FIXTURE.exists():
        print(
            f"Missing fixture {FIXTURE}: a converged HD189 state capture "
            "(run the CLI to convergence and np.savez the runner state; the "
            "maintainer's capture script lives in an internal repo)."
        )
        return 1

    import vulcan_jax.chem_funs as chem_funs
    from vulcan_jax.jax_step import AtmStatic
    from vulcan_jax.steady_state_grad import steady_state_reaction_sensitivity

    t0 = time.time()
    d = np.load(FIXTURE, allow_pickle=True)
    nz, ni = int(d["nz"]), int(d["ni"])
    y_star = jnp.asarray(d["y_star"])
    k_arr = jnp.asarray(d["k_arr"])
    dz = jnp.asarray(d["dz"])
    compo = jnp.asarray(d["compo"])
    atm = AtmStatic(
        **{k[5:]: jnp.asarray(d[k]) for k in d.files if k.startswith("atm__")},
        **{k[9:]: bool(d[k]) for k in d.files if k.startswith("atmbool__")},
    )
    net = chem_funs._NET_JAX
    rf = chem_funs._NETWORK.Rf
    ch4 = chem_funs._NETWORK.species_idx["CH4"]
    L0 = nz // 2
    print(
        f"Loaded converged HD189: nz={nz} ni={ni} nr={net.nr}  ({time.time() - t0:.1f}s)"
    )

    def loss(y):  # log10 CH4 volume mixing ratio at mid-column
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        return jnp.log10(ymix[L0, ch4])

    print("Solving the steady-state adjoint (one solve, all rate-table rows)...")
    print("  (first call per solver_map pays the step-VJP XLA compile; minutes)")

    grads = {}
    for smap in ("renorm", "bare"):  # renorm is the default; bare is legacy
        t1 = time.time()
        dLdlnk, info = steady_state_reaction_sensitivity(
            loss,
            y_star,
            k_arr,
            atm,
            net,
            compo_array=compo,
            dz=dz,
            solver_map=smap,
            lgmres_inner_m=250,
            lgmres_cycles=8,
            return_info=True,
        )
        grads[smap] = np.asarray(dLdlnk)
        print(
            f"  solver_map={smap:<6} {time.time() - t1:5.1f}s  "
            f"fp_err={info['fp_err']:.2e} resid={info['resid']:.2e} "
            f"twin_spread={info['ensemble_spread']:.2e} "
            f"(n_solves={info['n_solves']}, body_dt={info['body_dt']:.0e})"
        )

    print(f"\nTop 8 reactions setting log10(CH4 VMR) at layer {L0} (solver_map=renorm):")
    order = np.argsort(np.abs(grads["renorm"][: net.nr + 1]))[::-1][:8]
    for r in order:
        print(f"  r{int(r):4d}  dL/dln k = {grads['renorm'][r]:+.3e}   {rf.get(int(r), '?')}")

    print("\nFinite-difference anchors (adj_solvermap_gmres.py) — renorm (default) vs bare (legacy):")
    for r, fd in FD_ANCHORS.items():
        rel_b = abs(grads["bare"][r] - fd) / abs(fd)
        rel_r = abs(grads["renorm"][r] - fd) / abs(fd)
        print(f"  r{r:4d}  FD={fd:+.4e}  renorm={grads['renorm'][r]:+.4e} ({rel_r:.1%})"
              f"  bare={grads['bare'][r]:+.4e} ({rel_b:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
