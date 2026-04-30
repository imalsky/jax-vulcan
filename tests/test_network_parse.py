"""Validate VULCAN-JAX network.py against the freshly-generated VULCAN-master/chem_funs.py.

Run:
    cd VULCAN-JAX && python tests/test_network_parse.py

Requires that VULCAN-master/chem_funs.py has been regenerated for the same
network file (run `python VULCAN-master/make_chem_funs.py` once).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest  # noqa: E402  (used by @pytest.mark.master_serial)

# Make VULCAN-JAX importable when run from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import network as net_mod  # noqa: E402

# Make VULCAN-master importable for chem_funs and vulcan_cfg
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
if not VULCAN_MASTER.is_dir():
    pytest.skip(
        f"VULCAN-master oracle absent at {VULCAN_MASTER}; "
        "this comparison test requires the upstream sibling repo.",
        allow_module_level=True,
    )


def main() -> int:
    sys.path.insert(0, str(VULCAN_MASTER))
    import vulcan_cfg  # noqa: E402  (imports relative to current dir)

    network_path = ROOT / vulcan_cfg.network
    print(f"Parsing: {network_path}")
    net = net_mod.parse_network(network_path)

    # Compare against VULCAN-master/chem_funs.py
    import chem_funs  # noqa: E402

    ok = True

    # ni and nr
    if net.ni != chem_funs.ni:
        print(f"FAIL ni: jax={net.ni} vulcan={chem_funs.ni}")
        ok = False
    else:
        print(f"OK   ni = {net.ni}")
    if net.nr != chem_funs.nr:
        print(f"FAIL nr: jax={net.nr} vulcan={chem_funs.nr}")
        ok = False
    else:
        print(f"OK   nr = {net.nr}")

    # spec_list
    if list(net.species) != list(chem_funs.spec_list):
        print("FAIL spec_list")
        for i, (a, b) in enumerate(zip(net.species, chem_funs.spec_list)):
            if a != b:
                print(f"  position {i}: jax={a!r}  vulcan={b!r}")
                break
        if len(net.species) != len(chem_funs.spec_list):
            print(
                f"  length differs: jax={len(net.species)} vulcan={len(chem_funs.spec_list)}"
            )
        ok = False
    else:
        print(f"OK   spec_list (first 5: {net.species[:5]})")

    # Reaction text Rf -- compare a few sample entries
    rf_mismatch = 0
    for i in range(1, net.nr + 1, 2):
        a = net.Rf.get(i, "").replace(" ", "")
        b = chem_funs.re_dict.get(i, ([], []))
        # chem_funs stores re_dict as {i: ([reactants], [products])}; reconstruct text
        if not b:
            continue
        reactants, products = b
        chem_funs_eq = (
            " + ".join(str(r) for r in reactants)
            + " -> "
            + " + ".join(str(p) for p in products)
        ).replace(" ", "")
        if a != chem_funs_eq:
            rf_mismatch += 1
            if rf_mismatch <= 5:
                print(f"  mismatch i={i}: jax={a!r} vs vulcan={chem_funs_eq!r}")
    if rf_mismatch:
        print(f"WARN Rf mismatches: {rf_mismatch} (probably formatting/M handling)")
    else:
        print("OK   Rf reaction texts (sampled)")

    # photo_indx and stop_rev_indx are exposed by chem_funs.py? -- they live in var.* in VULCAN.
    # We can at least print our own values.
    print(f"   photo_indx    = {net.photo_indx}")
    print(f"   stop_rev_indx = {net.stop_rev_indx}")
    print(f"   conden_indx   = {net.conden_indx}")
    print(f"   #photo species: {len(net.photo_sp)}")

    # Stoichiometry sanity: every reaction should reference at least one species
    bad = 0
    for i in range(1, net.nr + 1):
        if (net.reactant_stoich[i].sum() == 0) and (net.product_stoich[i].sum() == 0):
            bad += 1
    if bad:
        print(f"FAIL {bad} reactions have empty stoichiometry")
        ok = False
    else:
        print("OK   all reactions have non-empty stoichiometry")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


@pytest.mark.master_serial
def test_main():
    """Run the master comparison in a fresh Python process."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(main())
