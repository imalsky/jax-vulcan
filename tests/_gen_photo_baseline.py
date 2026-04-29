"""One-shot fixture generator for the photo-setup tests.

Run on a green tree to refresh the npz baselines, commit them, and
never run again unless the maintainer explicitly asks. The dense
`PhotoStaticInputs` pytree is the only oracle for the photo-setup
tests; any regeneration must be justified in the PR description.

Produces two fixtures under `tests/data/`:
  - `photo_setup_hd189_baseline.npz`    HD189 default (T_cross_sp=[],
                                        use_ion=False).
  - `photo_setup_hd189_T_dep.npz`       Same atm, but T_cross_sp patched
                                        to ['CO2','H2O','NH3'] so the
                                        T-dependent path is exercised.

Encoding of dict-keyed arrays: flat npz keys.
  cross[sp]               -> "cross__{sp}"
  cross_J[(sp, i)]        -> "cross_J__{sp}__{i}"
  cross_T[sp]             -> "cross_T__{sp}"
  cross_J_T[(sp, i)]      -> "cross_J_T__{sp}__{i}"
  cross_Jion[(sp, i)]     -> "cross_Jion__{sp}__{i}"
  cross_scat[sp]          -> "cross_scat__{sp}"
Plus scalars: bins, dbin1, dbin2, nbin (as 0-d / 1-d arrays).

Usage:
    cd VULCAN-JAX/
    python tests/_gen_photo_baseline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _build_state_through_read_rate():
    """Run the pre-photo VULCAN setup and return (var, atm).

    `_build_photo_static_dense` reads dict attrs that `read_rate` writes
    onto `var`, so this helper wires them up directly via the private
    legacy classes (`state._Variables` / `_AtmData`).
    """
    import legacy_io as op
    from atm_setup import Atm
    from state import _Variables, _AtmData

    data_var = _Variables()
    data_atm = _AtmData()
    make_atm = Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    rate = op.ReadRate()
    data_var = rate.read_rate(data_var, data_atm)
    return data_var, data_atm


def _dump_static(static, fname: str) -> None:
    """Encode the dense `PhotoStaticInputs` pytree as a flat npz."""
    payload: dict[str, np.ndarray] = {}
    payload["bins"] = np.asarray(static.bins, dtype=np.float64)
    payload["dbin1"] = np.asarray(static.dbin1, dtype=np.float64)
    payload["dbin2"] = np.asarray(static.dbin2, dtype=np.float64)
    payload["nbin"] = np.asarray(static.nbin, dtype=np.int64)
    for i, sp in enumerate(static.absp_sp):
        payload[f"cross__{sp}"] = np.asarray(static.absp_cross[i], dtype=np.float64)
    for i, (sp, br) in enumerate(static.branch_keys):
        payload[f"cross_J__{sp}__{br}"] = np.asarray(static.cross_J[i], dtype=np.float64)
    for i, sp in enumerate(static.scat_sp):
        payload[f"cross_scat__{sp}"] = np.asarray(static.scat_cross[i], dtype=np.float64)
    for i, sp in enumerate(static.absp_T_sp):
        payload[f"cross_T__{sp}"] = np.asarray(static.absp_T_cross[i], dtype=np.float64)
    for i, (sp, br) in enumerate(static.branch_T_keys):
        payload[f"cross_J_T__{sp}__{br}"] = np.asarray(static.cross_J_T[i], dtype=np.float64)
    for i, (sp, br) in enumerate(static.ion_branch_keys):
        payload[f"cross_Jion__{sp}__{br}"] = np.asarray(static.cross_Jion[i], dtype=np.float64)
    np.savez(fname, **payload)
    print(f"  wrote {fname}  ({len(payload)} keys)")


def main() -> int:
    import vulcan_cfg
    import photo_setup

    out_dir = ROOT / "tests" / "data"
    out_dir.mkdir(exist_ok=True)

    # ---- 1) HD189 default ----
    print("Snapshotting HD189 baseline (T_cross_sp=[], use_ion=False)...")
    var, atm = _build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    _dump_static(static, str(out_dir / "photo_setup_hd189_baseline.npz"))

    # ---- 2) HD189 with Earth-style T_cross_sp ----
    print("Snapshotting HD189 with T_cross_sp=['CO2','H2O','NH3']...")
    saved_T_cross_sp = list(vulcan_cfg.T_cross_sp)
    vulcan_cfg.T_cross_sp = ["CO2", "H2O", "NH3"]
    try:
        var, atm = _build_state_through_read_rate()
        static = photo_setup._build_photo_static_dense(var, atm)
        _dump_static(static, str(out_dir / "photo_setup_hd189_T_dep.npz"))
    finally:
        vulcan_cfg.T_cross_sp = saved_T_cross_sp

    print("\nDone. Commit the .npz files; do not regenerate without justification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
