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
