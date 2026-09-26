"""Reverse-mode reaction ranking on a converged HD189 column.

`steady_state_reaction_sensitivity` returns dL/d(ln k_r) for every
directional rate-table row in one adjoint solve; finite differences would
need one re-converged run per row. It linearizes the hydrostatic-
renormalized map the runner iterates and averages an ensemble of twin
solves (default n_solves=3, body_dt=1e7); the printed twin spread is a
stability diagnostic. Rows are k-only: for a detailed-balance perturbation
of a reversible reaction, sum its forward and reverse rows. Forward mode
(grad_jvp_example.py) stays the route for a single row.

Needs tests/data/adj_state_hd189.npz (gitignored): build it with
`python tests/_gen_adj_state.py hd189`. The first call pays a one-time
step-VJP compile (~10-20 min). Run from VULCAN-JAX/ as
`python examples/grad_reverse_example.py`.
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
            f"Missing fixture {FIXTURE}; build it with "
            "`python tests/_gen_adj_state.py hd189`."
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
    fields = {k[5:]: jnp.asarray(d[k]) for k in d.files if k.startswith("atm__")}
    # Splice a default diff_esc_mask into fixtures that lack it (HD189 ships diff_esc: []).
    fields.setdefault("diff_esc_mask", jnp.zeros(ni, dtype=jnp.bool_))
    atm = AtmStatic(
        **fields,
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
    print("  (the first call pays the step-VJP XLA compile; minutes)")

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
    grad = np.asarray(dLdlnk)
    print(
        f"  {time.time() - t1:5.1f}s  "
        f"fp_err={info['fp_err']:.2e} resid={info['resid']:.2e} "
        f"twin_spread={info['ensemble_spread']:.2e} "
        f"(n_solves={info['n_solves']}, body_dt={info['body_dt']:.0e})"
    )

    print(f"\nTop 8 reactions setting log10(CH4 VMR) at layer {L0}:")
    order = np.argsort(np.abs(grad[: net.nr + 1]))[::-1][:8]
    for r in order:
        print(f"  r{int(r):4d}  dL/dln k = {grad[r]:+.3e}   {rf.get(int(r), '?')}")

    print("\nFinite-difference anchors (adj_solvermap_gmres.py):")
    for r, fd in FD_ANCHORS.items():
        rel = abs(grad[r] - fd) / abs(fd)
        print(f"  r{r:4d}  FD={fd:+.4e}  adjoint={grad[r]:+.4e} ({rel:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
