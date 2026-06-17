"""Reverse-mode reaction ranking on a real HD189 column.

`steady_state_grad.steady_state_reaction_sensitivity` returns
`dL/d(ln k_r)` for every reaction in one adjoint solve — "which reactions set
the converged abundance of species X". By finite differences this ranking would
cost one re-converged model per reaction; reverse-mode returns all of them at
once.

This is a reaction-*ranking* tool, not a precision-gradient tool: accuracy is
~few % vs finite differences (a steady-state-definition ceiling, not solver
error — the FD-anchor comparison printed below shows it), photolysis is frozen
on photo-on columns, and it returns `k`-only sensitivities. Forward-mode
(`grad_jvp_example.py`) is the higher-accuracy route for end-to-end gradients.
See the `steady_state_reaction_sensitivity` docstring for the full list.

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
        print(f"Missing fixture {FIXTURE}; generate it with adj_save_state.py.")
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

    print("Solving the steady-state adjoint (one solve, all reactions)...")
    print("  (first call pays the step-VJP XLA compile; minutes)")
    t1 = time.time()
    dLdlnk, info = steady_state_reaction_sensitivity(
        loss,
        y_star,
        k_arr,
        atm,
        net,
        compo_array=compo,
        dz=dz,
        lgmres_inner_m=250,
        lgmres_cycles=8,
        return_info=True,
    )
    dLdlnk = np.asarray(dLdlnk)
    print(
        f"Done in {time.time() - t1:.1f}s  "
        f"fp_err={info['fp_err']:.2e} resid={info['resid']:.2e} "
        f"matvecs={info['n_matvec']} deflated_dims={info['n_null']}"
    )

    print(f"\nTop 8 reactions setting log10(CH4 VMR) at layer {L0}:")
    order = np.argsort(np.abs(dLdlnk[: net.nr + 1]))[::-1][:8]
    for r in order:
        print(f"  r{int(r):4d}  dL/dln k = {dLdlnk[r]:+.3e}   {rf.get(int(r), '?')}")

    print("\nFinite-difference anchors (adj_solvermap_gmres.py):")
    for r, fd in FD_ANCHORS.items():
        rel = abs(dLdlnk[r] - fd) / abs(fd)
        print(f"  r{r:4d}  adjoint={dLdlnk[r]:+.4e}  FD={fd:+.4e}  rel={rel:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
