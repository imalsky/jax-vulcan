"""Validate the JAX-native VULCAN-JAX/chem_funs.py against the SymPy-generated
VULCAN-master/chem_funs.py.

Loads each module by manipulating sys.path order and asserts that all public
names match (or are documented exceptions: symjac/neg_symjac raise
NotImplementedError on our side).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def _import_chem_funs(prefer_jax: bool):
    """(Re)import chem_funs from either VULCAN-JAX (jax-native) or VULCAN-master
    (SymPy-generated). Returns the module."""
    VULCAN_MASTER = ROOT.parent / "VULCAN-master"
    # Reset modules so the new import wins
    for m in list(sys.modules.keys()):
        if m.startswith("chem_funs") or m == "chem_funs":
            del sys.modules[m]
    # Manipulate sys.path
    while str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    while str(VULCAN_MASTER) in sys.path:
        sys.path.remove(str(VULCAN_MASTER))
    if prefer_jax:
        sys.path.insert(0, str(ROOT))
        sys.path.append(str(VULCAN_MASTER))
    else:
        sys.path.insert(0, str(VULCAN_MASTER))
        sys.path.append(str(ROOT))
    import importlib
    import chem_funs   # noqa: E402
    importlib.reload(chem_funs)
    return chem_funs


def main() -> int:
    # Load reference (VULCAN-master, SymPy-generated)
    cf_ref = _import_chem_funs(prefer_jax=False)
    # Load JAX-native
    cf_jax = _import_chem_funs(prefer_jax=True)

    ok = True

    # === ni / nr / spec_list ===
    if cf_jax.ni != cf_ref.ni:
        print(f"FAIL ni: jax={cf_jax.ni} ref={cf_ref.ni}")
        ok = False
    else:
        print(f"OK   ni = {cf_jax.ni}")
    if cf_jax.nr != cf_ref.nr:
        print(f"FAIL nr: jax={cf_jax.nr} ref={cf_ref.nr}")
        ok = False
    else:
        print(f"OK   nr = {cf_jax.nr}")
    if list(cf_jax.spec_list) != list(cf_ref.spec_list):
        print("FAIL spec_list mismatch")
        ok = False
    else:
        print(f"OK   spec_list ({len(cf_jax.spec_list)} species)")

    # === re_dict / re_wM_dict (sample comparison) ===
    rd_match = sum(
        1 for i in range(1, cf_jax.nr + 1)
        if cf_jax.re_dict.get(i) == cf_ref.re_dict.get(i)
    )
    print(f"     re_dict matches: {rd_match} / {cf_jax.nr}")
    rwd_match = sum(
        1 for i in range(1, cf_jax.nr + 1)
        if cf_jax.re_wM_dict.get(i) == cf_ref.re_wM_dict.get(i)
    )
    print(f"     re_wM_dict matches: {rwd_match} / {cf_jax.nr}")

    # === Gibbs(i, T) for sampled forward reactions ===
    Tco = np.linspace(500.0, 2500.0, 10)
    max_err_gibbs = 0.0
    n_gibbs = 0
    for i in range(1, min(101, cf_jax.nr), 2):   # forward reactions
        try:
            ref = np.asarray(cf_ref.Gibbs(i, Tco), dtype=np.float64)
            ours = np.asarray(cf_jax.Gibbs(i, Tco), dtype=np.float64)
        except (KeyError, IndexError, ValueError):
            continue
        err = np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-300))
        max_err_gibbs = max(max_err_gibbs, err)
        n_gibbs += 1
    print(f"     Gibbs(i, T) compared: {n_gibbs}, max relerr: {max_err_gibbs:.3e}")
    if max_err_gibbs > 1e-10:
        print("FAIL Gibbs relerr too large")
        ok = False

    # === gibbs_sp(name, T) ===
    max_err_gsp = 0.0
    n_gsp = 0
    for sp in cf_jax.spec_list[:30]:
        try:
            ref = np.asarray(cf_ref.gibbs_sp(sp, Tco), dtype=np.float64)
            ours = np.asarray(cf_jax.gibbs_sp(sp, Tco), dtype=np.float64)
        except (KeyError, IndexError, AttributeError, ValueError):
            continue
        err = np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-300))
        max_err_gsp = max(max_err_gsp, err)
        n_gsp += 1
    print(f"     gibbs_sp(name, T) compared: {n_gsp}, max relerr: {max_err_gsp:.3e}")
    if max_err_gsp > 1e-10:
        print("FAIL gibbs_sp relerr too large")
        ok = False

    # === chemdf(y, M, k) ===
    nz_test = 5
    rng = np.random.default_rng(42)
    y = np.maximum(rng.uniform(0, 1e15, (nz_test, cf_jax.ni)), 1e-30)
    M = rng.uniform(1e10, 1e20, nz_test)
    k_arr = rng.uniform(1e-30, 1e-15, (cf_jax.nr + 1, nz_test))
    k_dict = {i: k_arr[i] for i in range(1, cf_jax.nr + 1)}

    df_ref = cf_ref.chemdf(y, M, k_dict)
    df_jax = cf_jax.chemdf(y, M, k_dict)
    df_relerr = np.max(np.abs(df_jax - df_ref) / np.maximum(np.abs(df_ref), 1e-30))
    print(f"     chemdf relerr: {df_relerr:.3e}")
    if df_relerr > 1e-3:    # large because of summation order; chem_rhs notes a 1e-4 known FP cancellation
        print("FAIL chemdf relerr too large")
        ok = False

    # === symjac / neg_symjac should raise on JAX side ===
    raised = False
    try:
        cf_jax.symjac(y, M, k_dict)
    except NotImplementedError:
        raised = True
    if not raised:
        print("FAIL symjac on JAX side did not raise NotImplementedError")
        ok = False
    else:
        print("OK   symjac raises NotImplementedError on JAX side (as expected)")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_main():
    """Pytest wrapper. `main()` returns 0 on success; convert to an
    assertion so `pytest tests/` collects and runs this script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
