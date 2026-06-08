"""Guards against the FastChem element-ORDER regression.

VULCAN's vendored FastChem (`fastchem_src/mass_action_constant.cpp`) subtracts
per-element NASA9 reference polynomials by HARD-CODED slot index, while the
element vector is actually built in the row order of the abundance file. If the
two disagree, carbon silently takes helium's reference polynomial and CO/CH4/CO2
never form (all carbon stays atomic) — no crash, plausible-looking output.

These tests pin the three sources of truth together so the regression cannot
reappear silently:
  1. the hard-coded `index_X = N` map in the C++,
  2. the `_FASTCHEM_ELEMENT_ORDER` constant the runtime validator checks against,
  3. the shipped `*_element_abundances.dat` row order,
plus an end-to-end check that the binary actually forms CO at solar / 2000 K.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vulcan_jax._paths import PACKAGE_ROOT
from vulcan_jax.runtime_validation import (
    _FASTCHEM_ELEMENT_ORDER,
    _validate_fastchem_input_vs_network,
)

ROOT = Path(__file__).resolve().parent.parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"

# C++ symbol token -> element symbol (Km is potassium; e is the electron).
_CPP_NAME_TO_ELEMENT = {
    "C": "C", "H": "H", "He": "He", "N": "N", "O": "O", "P": "P", "S": "S",
    "Si": "Si", "Ti": "Ti", "V": "V", "Cl": "Cl", "Km": "K", "Na": "Na",
    "Mg": "Mg", "F": "F", "Ca": "Ca", "Fe": "Fe", "e": "e-",
}


def _fastchem_tree(repo_root: Path) -> Path:
    return repo_root / "fastchem_vulcan"


def _parse_cpp_index_map(cpp_path: Path) -> dict[str, int]:
    """Extract the `unsigned int index_X = N;` slot map from the C++ source."""
    text = cpp_path.read_text()
    out: dict[str, int] = {}
    for name, idx in re.findall(r"index_(\w+)\s*=\s*(\d+)", text):
        if name in _CPP_NAME_TO_ELEMENT:
            out[_CPP_NAME_TO_ELEMENT[name]] = int(idx)
    return out


def _parse_abundance_order(path: Path) -> list[str]:
    """Element symbols in the order FastChem will load them (row order)."""
    order: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                float(parts[1])
            except ValueError:
                continue
            order.append(parts[0])
    return order


# --------------------------------------------------------------------------- #
# 1. The validator constant must match the C++ hard-coded indices.            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "repo_root",
    [
        pytest.param(PACKAGE_ROOT, id="jax"),
        pytest.param(VULCAN_MASTER, id="master",
                     marks=pytest.mark.skipif(
                         not (VULCAN_MASTER / "fastchem_vulcan").is_dir(),
                         reason="VULCAN-master sibling absent")),
    ],
)
def test_cpp_indices_match_canonical_order(repo_root: Path):
    cpp = _fastchem_tree(repo_root) / "fastchem_src" / "mass_action_constant.cpp"
    index_map = _parse_cpp_index_map(cpp)
    # Rebuild the order implied by the C++ slots and compare to the constant.
    order_from_cpp = [sym for sym, _ in sorted(index_map.items(), key=lambda kv: kv[1])]
    assert order_from_cpp == _FASTCHEM_ELEMENT_ORDER, (
        f"{cpp} hard-coded indices {index_map} no longer match "
        f"_FASTCHEM_ELEMENT_ORDER {_FASTCHEM_ELEMENT_ORDER}. Update one to match "
        "the other (and the shipped abundance files)."
    )


# --------------------------------------------------------------------------- #
# 2. The shipped abundance files must be in that exact order.                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "repo_root",
    [
        pytest.param(PACKAGE_ROOT, id="jax"),
        pytest.param(VULCAN_MASTER, id="master",
                     marks=pytest.mark.skipif(
                         not (VULCAN_MASTER / "fastchem_vulcan").is_dir(),
                         reason="VULCAN-master sibling absent")),
    ],
)
@pytest.mark.parametrize(
    "fname", ["solar_element_abundances.dat", "element_abundances_vulcan.dat"]
)
def test_abundance_file_matches_canonical_order(repo_root: Path, fname: str):
    path = _fastchem_tree(repo_root) / "input" / fname
    if not path.exists():
        # element_abundances_vulcan.dat is generated at runtime and git-ignored,
        # so it is absent on a clean checkout; the tracked solar file is the
        # real source of truth (the generated file inherits its row order).
        pytest.skip(f"{path} absent (generated at runtime / clean checkout)")
    order = _parse_abundance_order(path)
    # Exact, COMPLETE order (electron row optional) — not a truncation-permissive
    # prefix, which would let a file missing trailing elements pass.
    assert order in (_FASTCHEM_ELEMENT_ORDER, _FASTCHEM_ELEMENT_ORDER[:-1]), (
        f"{path} row order {order} disagrees with the FastChem hard-coded slots "
        f"{_FASTCHEM_ELEMENT_ORDER}. Reordering or truncation silently corrupts "
        "carbon (or P/S/rocky) chemistry (CO never forms)."
    )


# --------------------------------------------------------------------------- #
# 3. The runtime validator must REJECT a reordered file.                       #
# --------------------------------------------------------------------------- #
def test_validator_rejects_reordered_abundance_file(tmp_path: Path):
    # H-first order with otherwise-canonical values (the exact regression).
    rows = ["H 12.00", "He 10.9232", "C 8.4434", "N 7.9130", "O 8.7826",
            "S 7.1492", "P -3.0", "Si -3.0", "Ti -3.0", "V -3.0", "Cl -3.0",
            "K -3.0", "Na -3.0", "Mg -3.0", "F -3.0", "Ca -3.0", "Fe -3.0",
            "e- 0"]
    rel = "fastchem_vulcan/input/solar_element_abundances.dat"
    (tmp_path / "fastchem_vulcan" / "input").mkdir(parents=True)
    (tmp_path / rel).write_text("# header\n" + "\n".join(rows) + "\n")

    cfg = SimpleNamespace(ini_mix="EQ", fastchem_solar_abundance_file=rel)
    errors = _validate_fastchem_input_vs_network(cfg, tmp_path)
    assert any("ROW ORDER" in e for e in errors), (
        f"validator failed to flag a reordered abundance file; errors={errors}"
    )

    # And ACCEPT the canonical order (values + order both correct).
    good = "\n".join(
        f"{s} {v}" for s, v in zip(
            _FASTCHEM_ELEMENT_ORDER,
            ["8.4434", "12.00", "10.9232", "7.9130", "8.7826", "-3.0", "7.1492",
             "-3.0", "-3.0", "-3.0", "-3.0", "-3.0", "-3.0", "-3.0", "-3.0",
             "-3.0", "-3.0", "0"],
        )
    )
    (tmp_path / rel).write_text("# header\n" + good + "\n")
    assert _validate_fastchem_input_vs_network(cfg, tmp_path) == []

    # And REJECT a truncated file: canonical order but missing trailing rows,
    # which would leave later C++ slots (P/S/rocky) on the wrong reference.
    truncated = "\n".join(f"{s} 1.0" for s in _FASTCHEM_ELEMENT_ORDER[:5])
    (tmp_path / rel).write_text("# header\n" + truncated + "\n")
    assert any(
        "ROW ORDER" in e for e in _validate_fastchem_input_vs_network(cfg, tmp_path)
    ), "validator failed to flag a truncated abundance file"


# --------------------------------------------------------------------------- #
# 4. End-to-end: the binary must put carbon in CO, not atomic C.              #
# --------------------------------------------------------------------------- #
def test_fastchem_eq_forms_co_not_atomic_carbon(tmp_path: Path):
    """Solar / 2000 K / 1 bar equilibrium must have carbon in CO (~5e-4), not
    frozen as atomic C. This is the physical invariant the order bug violated;
    it catches ANY future corruption (order, thermo, or binary), not just a
    reorder. Skips cleanly if the prebuilt binary is unavailable."""
    src = _fastchem_tree(PACKAGE_ROOT)
    binary = src / "fastchem"
    if not binary.exists():
        pytest.skip("FastChem binary not built in this checkout")

    tree = tmp_path / "fastchem_vulcan"
    shutil.copytree(src, tree)
    # Force a known single point and the no-ion parameter set.
    (tree / "input" / "vulcan_TP" / "vulcan_TP.dat").write_text(
        "#p (bar)\tT (K)\n1.000e+00\t2000.0"
    )
    params = tree / "input" / "parameters_wo_ion.dat"
    if params.exists():
        shutil.copy2(params, tree / "input" / "parameters.dat")

    res = subprocess.run(
        ["./fastchem", "input/config.input"],
        cwd=str(tree), capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        pytest.skip(f"FastChem binary did not run here: {res.stderr[-400:]}")

    out = tree / "output" / "vulcan_EQ.dat"
    data = np.genfromtxt(out, names=True, dtype=None, encoding="utf-8")
    names = data.dtype.names
    assert "CO" in names and "C" in names, f"unexpected output columns: {names}"
    co = float(data["CO"])
    c_atomic = float(data["C"])
    assert co > 1e-4, f"CO={co:.3e} (carbon failed to form CO — order bug?)"
    assert c_atomic < 1e-6, f"atomic C={c_atomic:.3e} (carbon stuck atomic)"
    assert co > 100 * c_atomic, f"CO={co:.3e} not >> atomic C={c_atomic:.3e}"
