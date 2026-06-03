"""WASP-39 b pre-integration invariant against VULCAN-master.

This catches hidden FastChem input drift before any long W39b integration.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
from vulcan_jax._paths import PACKAGE_ROOT

JAX_CFG = PACKAGE_ROOT / "cfg_examples" / "vulcan_cfg_W39b.py"
MASTER_CFG = VULCAN_MASTER / "cfg_examples" / "vulcan_cfg_W39b.py"

EXPECTED_FREE_O = 5.37e-3 - 2.95e-3
REL_TOL = 1e-10


_JAX_SCRIPT = r"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])

os.chdir(root)
sys.path.insert(0, str(root))

spec = importlib.util.spec_from_file_location("vulcan_cfg", cfg_src)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)
sys.modules["vulcan_jax.vulcan_cfg"] = cfg
sys.modules["vulcan_cfg"] = cfg

# Clear any cached vulcan_jax submodules so chem_funs re-parses the W39b network
for mod_name in list(sys.modules):
    if mod_name.startswith("vulcan_jax.") and mod_name != "vulcan_jax.vulcan_cfg":
        del sys.modules[mod_name]
if "vulcan_jax" in sys.modules:
    del sys.modules["vulcan_jax"]

from vulcan_jax.runtime_validation import validate_runtime_config

validate_runtime_config(cfg, root)

import vulcan_jax.chem_funs as chem_funs
from vulcan_jax.atm_setup import Atm
from vulcan_jax.ini_abun import InitialAbun, _fastchem_solar_abundance_path
from vulcan_jax.state import _AtmData, _Variables

data_var = _Variables()
data_atm = _AtmData()
make_atm = Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if cfg.use_condense:
    make_atm.sp_sat(data_atm)

ini = InitialAbun()
data_var = ini.ini_y(data_var, data_atm)
data_var = ini.ele_sum(data_var)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    nr=np.int64(chem_funs.nr),
    pco=np.asarray(data_atm.pco, dtype=np.float64),
    Tco=np.asarray(data_atm.Tco, dtype=np.float64),
    Kzz=np.asarray(data_atm.Kzz, dtype=np.float64),
    y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
    atom_keys=np.array(list(data_var.atom_ini.keys()), dtype=object),
    atom_vals=np.array([float(v) for v in data_var.atom_ini.values()], dtype=np.float64),
    fastchem_source=np.array([str(_fastchem_solar_abundance_path())], dtype=object),
)
"""


_MASTER_SCRIPT = r"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])
backup_dir = Path(sys.argv[4])

backup_dir.mkdir(parents=True, exist_ok=True)
backed_up = []
for rel in ("vulcan_cfg.py", "chem_funs.py"):
    src = root / rel
    if src.exists():
        dst = backup_dir / rel
        shutil.copy2(src, dst)
        backed_up.append((src, dst))

try:
    (root / "vulcan_cfg.py").write_text(cfg_src.read_text())
    res = subprocess.run(
        [sys.executable, "make_chem_funs.py"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    # make_chem_funs.py writes chem_funs.py BEFORE its post-codegen
    # check_conserv() sanity check, which raises under numpy>=1.24
    # (str(numpy.bytes_) -> "b'OH'"). The crash is benign to the generated
    # module (master's own vulcan.py ignores the exit code via os.system), so
    # only bail if the generated chem_funs.py does not import with valid ni/nr.
    if res.returncode != 0:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import chem_funs as c; "
                "assert getattr(c, 'ni', 0) > 0 and getattr(c, 'nr', 0) > 0",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if probe.returncode != 0:
            print("make_chem_funs.py failed", res.returncode)
            print(res.stdout[-2000:])
            print(res.stderr[-2000:])
            sys.exit(res.returncode)

    os.chdir(root)
    sys.path.insert(0, str(root))

    import build_atm
    import chem_funs
    import store
    import vulcan_cfg

    data_var = store.Variables()
    data_atm = store.AtmData()
    make_atm = build_atm.Atm()
    data_atm = make_atm.f_pico(data_atm)
    data_atm = make_atm.load_TPK(data_atm)
    if vulcan_cfg.use_condense:
        make_atm.sp_sat(data_atm)

    ini = build_atm.InitialAbun()
    data_var = ini.ini_y(data_var, data_atm)
    data_var = ini.ele_sum(data_var)

    np.savez_compressed(
        out_npz,
        species=np.array(list(chem_funs.spec_list), dtype=object),
        nr=np.int64(chem_funs.nr),
        pco=np.asarray(data_atm.pco, dtype=np.float64),
        Tco=np.asarray(data_atm.Tco, dtype=np.float64),
        Kzz=np.asarray(data_atm.Kzz, dtype=np.float64),
        y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
        atom_keys=np.array(list(data_var.atom_ini.keys()), dtype=object),
        atom_vals=np.array([float(v) for v in data_var.atom_ini.values()], dtype=np.float64),
    )
finally:
    for src, dst in backed_up:
        shutil.copy2(dst, src)
"""


def _run_probe(script: str, *args: Path, skip_on_failure: bool = False):
    """Run a probe subprocess and surface stdout/stderr on failure."""
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode != 0 and skip_on_failure:
        pytest.skip(
            f"probe subprocess failed {result.returncode} "
            f"(likely numpy encoding incompatibility in master's make_chem_funs.py)"
        )
    assert result.returncode == 0, (
        f"probe failed with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def _atom_dict(data: np.lib.npyio.NpzFile) -> dict[str, float]:
    """Return the saved atom diagnostics as a normal dict."""
    return {
        str(k): float(v)
        for k, v in zip(data["atom_keys"].tolist(), data["atom_vals"].tolist())
    }


def _max_relerr(a: np.ndarray, b: np.ndarray, floor: float = 1e-30) -> float:
    """Maximum relative error with a floor for absent trace species."""
    denom = np.maximum(np.abs(b), floor)
    return float(np.max(np.abs(a - b) / denom))


@pytest.mark.master_serial
def test_w39b_fastchem_initial_state_matches_master():
    """W39b FastChem EQ init must match master before integration starts."""
    if not VULCAN_MASTER.is_dir():
        pytest.skip(
            f"VULCAN-master oracle absent at {VULCAN_MASTER}; W39b invariant skipped."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        jax_npz = tmp_path / "jax_w39b_init.npz"
        master_npz = tmp_path / "master_w39b_init.npz"
        master_backup = tmp_path / "master_backup"

        _run_probe(_JAX_SCRIPT, PACKAGE_ROOT, jax_npz, JAX_CFG)
        _run_probe(
            _MASTER_SCRIPT,
            VULCAN_MASTER,
            master_npz,
            MASTER_CFG,
            master_backup,
            skip_on_failure=True,
        )

        jax = np.load(jax_npz, allow_pickle=True)
        master = np.load(master_npz, allow_pickle=True)

        assert list(jax["species"]) == list(master["species"])
        assert int(jax["nr"]) == int(master["nr"])
        np.testing.assert_allclose(jax["pco"], master["pco"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(jax["Tco"], master["Tco"], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(jax["Kzz"], master["Kzz"], rtol=1e-14, atol=0.0)

        y_relerr = _max_relerr(jax["y_ini"], master["y_ini"])
        assert y_relerr < REL_TOL, f"W39b y_ini relerr {y_relerr:.3e}"

        jax_atoms = _atom_dict(jax)
        master_atoms = _atom_dict(master)
        assert set(jax_atoms) == set(master_atoms)
        for atom, master_val in master_atoms.items():
            relerr = abs(jax_atoms[atom] - master_val) / abs(master_val)
            assert relerr < REL_TOL, f"atom_ini[{atom}] relerr {relerr:.3e}"

        free_o = (jax_atoms["O"] - jax_atoms["C"]) / jax_atoms["H"]
        assert np.isclose(free_o, EXPECTED_FREE_O, rtol=5e-4, atol=0.0), (
            f"W39b free oxygen budget {(free_o):.6e} != {EXPECTED_FREE_O:.6e}"
        )
        source = str(jax["fastchem_source"][0])
        assert source.endswith("fastchem_vulcan/input/solar_element_abundances.dat")
