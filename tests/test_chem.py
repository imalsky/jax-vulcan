"""Validate chem_rhs and chem_jac against VULCAN-master's chemdf and symjac.

We compare on a real (y, M, k) state captured from VULCAN-master's pipeline.
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

# Oracle test: requires VULCAN-master sibling for the upstream chemdf/symjac
# and op references. Skip cleanly when absent.
def _oracle_dir():
    """Configured upstream oracle checkout, or a non-existent sentinel path
    (so the `is_dir()` skip sites work without a None branch)."""
    import os
    from pathlib import Path as _P

    raw = os.environ.get("VULCAN_MASTER_DIR")
    return _P(raw).expanduser().resolve() if raw else _P("/nonexistent/VULCAN-oracle-unset")


# Oracle location comes from $VULCAN_MASTER_DIR, never from a sibling guess:
# an auto-detected ../VULCAN-master pins nothing. Do not restore the fallback.
VULCAN_MASTER = _oracle_dir()
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"upstream oracle not configured (looked at {VULCAN_MASTER}). Set "
        f"$VULCAN_MASTER_DIR to a clean clone at the commit pinned in "
        f"tests/science_sources.yaml; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )

warnings.filterwarnings("ignore")


def main() -> int:
    # === 1. Run VULCAN-master pipeline to get reference y/M/k state and chemdf/symjac ===
    os.chdir(VULCAN_MASTER)
    sys.path.insert(0, str(VULCAN_MASTER))

    import vulcan_cfg as cfg_v
    import store as st_v
    import build_atm as ba_v
    import op as op_v
    import chem_funs as cf_v

    data_var = st_v.Variables()
    data_atm = st_v.AtmData()
    make_atm = ba_v.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if cfg_v.use_condense:
        make_atm.sp_sat(data_atm)
    rate = op_v.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    data_var = rate.rev_rate(data_var, data_atm)
    ini = ba_v.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)

    T = np.asarray(data_atm.Tco, dtype=np.float64).copy()
    M = np.asarray(data_atm.M, dtype=np.float64).copy()
    y = np.asarray(data_var.y, dtype=np.float64).copy()
    k_dict = {i: np.asarray(v, dtype=np.float64).copy() for i, v in data_var.k.items()}
    nz, ni = y.shape
    print(f"State: nz={nz}, ni={ni}, T range [{T.min():.1f}, {T.max():.1f}]")

    # Get reference chemdf and symjac on this state
    dydt_ref = np.asarray(cf_v.chemdf(y, M, k_dict)).copy()
    J_ref = np.asarray(cf_v.symjac(y, M, k_dict)).copy()

    # === 2. Switch over to VULCAN-JAX modules (clear cached imports) ===
    for mod_name in (
        "vulcan_cfg",
        "store",
        "build_atm",
        "op",
        "chem_funs",
        "network",
        "rates",
        "gibbs",
        "chem",
    ):
        sys.modules.pop(mod_name, None)
    # Remove VULCAN-master from sys.path
    while str(VULCAN_MASTER) in sys.path:
        sys.path.remove(str(VULCAN_MASTER))
    os.chdir(ROOT)

    import vulcan_jax.network as net_mod
    import vulcan_jax.chem as chem_mod
    import jax.numpy as jnp

    from vulcan_jax._paths import resolve_data_path

    net = net_mod.parse_network(resolve_data_path(cfg_v.network))
    net_jax = chem_mod.to_jax(net)

    # Use the same k_dict from VULCAN's pipeline (already validated against ours).
    # Pack into [nr+1, nz] array.
    k_full = np.zeros((net.nr + 1, nz), dtype=np.float64)
    for i, vec in k_dict.items():
        k_full[i] = vec

    # === 3. Compute JAX chem_rhs ===
    y_j = jnp.asarray(y)
    M_j = jnp.asarray(M)
    k_j = jnp.asarray(k_full)
    # Use the codegen path that the integrator actually runs. We build it
    # against `net` (parsed from cfg_v.network) so the test isn't sensitive
    # to JAX's default vulcan_cfg.network.
    import vulcan_jax.make_chem_funs as mcf

    chem_rhs_codegen = mcf.build_chem_rhs(net)
    dydt_jax = np.asarray(chem_rhs_codegen(y_j, M_j, k_j))

    print(f"dydt_ref shape: {dydt_ref.shape}, dydt_jax shape: {dydt_jax.shape}")

    # Skip cells below 1e-6 of the species's peak |dydt|: cancellation
    # residues near zero, where XLA's FMA fusion differs from NumPy but both
    # are within ULP of zero at the species's natural scale.
    per_species_peak = np.abs(dydt_ref).max(axis=0)
    cell_floor = 1e-6 * np.maximum(per_species_peak[None, :], 1e-30)
    significant = np.abs(dydt_ref) > cell_floor
    abs_tol = 1e-30
    relerr_full = np.abs(dydt_jax - dydt_ref) / np.maximum(np.abs(dydt_ref), abs_tol)
    relerr = np.where(significant, relerr_full, 0.0)
    max_relerr = float(relerr.max())
    max_idx = np.unravel_index(int(relerr.argmax()), relerr.shape)
    print(
        f"chem_rhs max relerr (cells > 1e-6 of species peak): {max_relerr:.3e} at "
        f"layer {max_idx[0]}, species {net.species[max_idx[1]]}"
    )
    print(f"  values: jax={dydt_jax[max_idx]:.3e} ref={dydt_ref[max_idx]:.3e}")
    print(f"  species peak |dydt|: {per_species_peak[max_idx[1]]:.3e}")

    # Per-species worst-error ranking
    per_species_max = relerr.max(axis=0)
    bad_species = np.argsort(per_species_max)[::-1][:15]
    print("Top 15 worst species by max relerr:")
    for j in bad_species:
        if per_species_max[j] < 1e-12:
            break
        # Find layer with max error
        layer_idx = relerr[:, j].argmax()
        print(
            f"  {net.species[j]:>8}  relerr={per_species_max[j]:.3e}  "
            f"layer {layer_idx}: jax={dydt_jax[layer_idx, j]:.3e} ref={dydt_ref[layer_idx, j]:.3e}"
        )

    # === 4. Compute JAX chem_jac ===
    Jblk_jax = np.asarray(chem_mod.chem_jac(y_j, M_j, k_j, net_jax))
    print(f"chem_jac shape: {Jblk_jax.shape}, J_ref shape: {J_ref.shape}")

    # symjac layout: per-layer (ni, ni) blocks live at J_ref[j*ni:(j+1)*ni, j*ni:(j+1)*ni]
    max_jac_relerr = 0.0
    max_jac_idx = None
    for j in range(nz):
        block_ref = J_ref[j * ni : (j + 1) * ni, j * ni : (j + 1) * ni]
        block_jax = Jblk_jax[j]
        err = np.max(
            np.abs(block_jax - block_ref) / np.maximum(np.abs(block_ref), abs_tol)
        )
        if err > max_jac_relerr:
            max_jac_relerr = err
            max_jac_idx = j
    print(f"chem_jac max relerr: {max_jac_relerr:.3e} at layer {max_jac_idx}")

    # Bulk-species criterion: the W39b benchmark species must agree tightly
    # (a looser ~1e-4 floor once produced 0.25-0.49 dex converged-state drift).
    bulk_relerr = 0.0
    for sp in ("H2O", "CO2", "SO", "SO2", "H2", "CO", "S", "H2S"):
        if sp in net.species_idx:
            j = net.species_idx[sp]
            r = float(
                np.abs(dydt_jax[:, j] - dydt_ref[:, j]).max()
                / max(np.abs(dydt_ref[:, j]).max(), 1e-30)
            )
            bulk_relerr = max(bulk_relerr, r)
    print(f"bulk-species worst relerr: {bulk_relerr:.3e}")

    # chem_rhs is codegen-backed (master-faithful per-reaction term order).
    # On cells > 1e-6 of species peak, rtol=1e-5; on bulk species, 1e-5.
    ok = (max_relerr < 1e-5) and (bulk_relerr < 1e-5) and (max_jac_relerr < 1e-6)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Runs main() in a fresh subprocess: the master/JAX module-table swap
    only works from a cold Python start."""
    import subprocess
    from tests.oracle import oracle_worktree

    with oracle_worktree("vulcan2_ncho", "cfg_examples/vulcan_cfg_HD189.py") as master:
        env = os.environ.copy()
        env["VULCAN_MASTER_DIR"] = str(master)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
