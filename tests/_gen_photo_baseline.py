"""Regenerate the photo_setup oracle fixtures under tests/data/.

Writes `photo_setup_hd189_baseline.npz` (default HD189 cfg: T_cross_sp=[],
use_ion=False) and `photo_setup_hd189_T_dep.npz` (T_cross_sp patched to
['CO2','H2O','NH3']) with the exact key schema `tests/test_photo_setup.py`
compares against. The fixtures are gitignored; `tests/gen_fixtures.py --all`
drives this script.

Run from VULCAN-JAX/:
    python tests/_gen_photo_baseline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

FIXTURE_DIR = ROOT / "tests" / "data"


def build_state_through_read_rate():
    """Return (var, atm) after the pre-photo setup and ReadRate, in the legacy
    containers: `photo_setup._build_photo_static_dense` reads dict attrs that
    `legacy_io.ReadRate.read_rate` writes onto `var`."""
    import vulcan_jax.legacy_io as op
    from _helpers import load_tpk_state

    var, atm, _ = load_tpk_state()
    return op.ReadRate().read_rate(var, atm), atm


def _static_to_npz_dict(static) -> dict:
    """Flatten a `_PhotoStatic` into the fixture key schema."""
    out = {
        "bins": np.asarray(static.bins),
        "nbin": int(static.nbin),
        "dbin1": float(static.dbin1),
        "dbin2": float(static.dbin2),
    }
    for i, sp in enumerate(static.absp_sp):
        out[f"cross__{sp}"] = np.asarray(static.absp_cross[i])
    for i, sp in enumerate(static.absp_T_sp):
        out[f"cross_T__{sp}"] = np.asarray(static.absp_T_cross[i])
    for i, (sp, br) in enumerate(static.branch_keys):
        out[f"cross_J__{sp}__{br}"] = np.asarray(static.cross_J[i])
    for i, (sp, br) in enumerate(static.branch_T_keys):
        out[f"cross_J_T__{sp}__{br}"] = np.asarray(static.cross_J_T[i])
    for i, sp in enumerate(static.scat_sp):
        out[f"cross_scat__{sp}"] = np.asarray(static.scat_cross[i])
    for i, (sp, br) in enumerate(static.ion_branch_keys):
        out[f"cross_Jion__{sp}__{br}"] = np.asarray(static.cross_Jion[i])
    return out


def main() -> int:
    import vulcan_jax.photo_setup as photo_setup
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

    if not bool(getattr(vulcan_cfg, "use_photo", False)):
        raise SystemExit(
            "use_photo=False in vulcan_cfg; the fixtures need the photo path on."
        )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    var, atm = build_state_through_read_rate()
    static = photo_setup._build_photo_static_dense(var, atm)
    path = FIXTURE_DIR / "photo_setup_hd189_baseline.npz"
    np.savez(path, **_static_to_npz_dict(static))
    print(f"wrote {path}")

    old_T_cross_sp = vulcan_cfg.T_cross_sp
    try:
        vulcan_cfg.T_cross_sp = ["CO2", "H2O", "NH3"]
        var, atm = build_state_through_read_rate()
        static = photo_setup._build_photo_static_dense(var, atm)
        path = FIXTURE_DIR / "photo_setup_hd189_T_dep.npz"
        np.savez(path, **_static_to_npz_dict(static))
        print(f"wrote {path}")
    finally:
        vulcan_cfg.T_cross_sp = old_T_cross_sp
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
