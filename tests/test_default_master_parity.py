"""Matched-step, whole-model parity checks against VULCAN-master.

The default HD189 config for 20 and 200 steps, plus the condensing column that
pins the `fix_species` reservoir. Every test stages VULCAN-master only
inside subprocesses and restores any changed config/FastChem files before
returning.

The matched-step cases run the JAX side FROM MASTER'S OWN initial column
(`_JAX_SCRIPT` swaps the `EQ` loader): the two codes seed from
different equilibrium solvers, and comparing trajectories started from
different columns would measure the seeds rather than the solvers. The seeds
have their own gate in tests/test_eq_seed.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
from _helpers import relerr
from oracle import oracle_dir_or_sentinel

# The parent verifies the pin and passes a temporary copy; the per-test
# is_dir() skips below handle an unset oracle.
VULCAN_MASTER = oracle_dir_or_sentinel()

from vulcan_jax._paths import PACKAGE_ROOT

# (count_max, update_frq, diff_esc, rtol, ymix_min). Case 1: default HD189
# for 20 steps over every cell, at the FP-accumulation floor (JAX and NumPy
# rate builds differ by 5.7e-14 and 19 stiff steps amplify it; worst 3.1e-9).
# Case 2: 200 steps with 40 hydrostatic refreshes and H escape, over cells
# with master ymix > 1e-10 (trace cells clip on different steps); a
# misapplied escape Jacobian term reads 3.7e-3.
MATCHED_CASES = [
    (19, 100, [], 5.0e-9, 0.0),
    (199, 5, ["H"], 1.0e-6, 1.0e-10),
]


_MASTER_SCRIPT = r"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

master_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
backup_dir = Path(sys.argv[3])
count_max = int(sys.argv[4])
update_frq = int(sys.argv[5])
diff_esc = [sp for sp in sys.argv[6].split(",") if sp]

TRACKED_FILES = [
    Path("vulcan_cfg.py"),
    Path("chem_funs.py"),
    Path("fastchem_vulcan/input/element_abundances_vulcan.dat"),
    Path("fastchem_vulcan/input/parameters.dat"),
    Path("fastchem_vulcan/input/vulcan_TP/vulcan_TP.dat"),
    Path("fastchem_vulcan/output/vulcan_EQ.dat"),
    Path("fastchem_vulcan/output/chem_species.dat"),
    Path("fastchem_vulcan/output/monitor_output.dat"),
]


def backup_files() -> set[Path]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    existed = set()
    for rel in TRACKED_FILES:
        src = master_root / rel
        if src.exists():
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            existed.add(rel)
    return existed


def restore_files(existed: set[Path]) -> None:
    for rel in TRACKED_FILES:
        dst = master_root / rel
        src = backup_dir / rel
        if rel in existed:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


def run() -> None:
    existed = backup_files()
    try:
        cfg_src = master_root / "cfg_examples" / "vulcan_cfg_HD189.py"
        cfg_text = cfg_src.read_text()
        overrides = (
            "\n# === default parity test overrides ===\n"
            f"count_max = {count_max}\n"
            f"count_min = {count_max + 1}\n"
            "trun_min = 1e22\n"
            "use_print_prog = False\n"
            "use_print_delta = False\n"
            "use_live_plot = False\n"
            "use_live_flux = False\n"
            "use_plot_end = False\n"
            "use_plot_evo = False\n"
            "use_save_movie = False\n"
            "use_flux_movie = False\n"
            "save_evolution = False\n"
            "plot_TP = False\n"
            "use_adapt_rtol = False\n"
            f"update_frq = {update_frq}\n"
            f"diff_esc = {diff_esc!r}\n"
        )
        (master_root / "vulcan_cfg.py").write_text(cfg_text + overrides)

        res = subprocess.run(
            [sys.executable, "make_chem_funs.py"],
            cwd=str(master_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        # make_chem_funs.py writes chem_funs.py before its check_conserv(), which
        # raises under numpy>=1.24; master's vulcan.py ignores that exit code, so
        # only the module's importability matters.
        if res.returncode != 0:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import chem_funs as c; "
                    "assert getattr(c, 'ni', 0) > 0 and getattr(c, 'nr', 0) > 0",
                ],
                cwd=str(master_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if probe.returncode != 0:
                print(res.stdout[-2000:])
                print(res.stderr[-2000:])
                sys.exit(res.returncode)

        os.chdir(master_root)
        sys.path.insert(0, str(master_root))

        # Master's OWN codegen and the cfg this script just wrote: importing
        # the port here would make the saved species list and the pre-loop
        # branches below read the side under test.
        import build_atm
        import chem_funs
        import op
        import store
        import vulcan_cfg

        data_var = store.Variables()
        data_atm = store.AtmData()
        data_para = store.Parameters()
        data_para.start_time = time.time()
        make_atm = build_atm.Atm()
        output = op.Output()

        data_atm = make_atm.f_pico(data_atm)
        data_atm = make_atm.load_TPK(data_atm)
        if vulcan_cfg.use_condense:
            make_atm.sp_sat(data_atm)

        rate = op.ReadRate()
        data_var = rate.read_rate(data_var, data_atm)
        if vulcan_cfg.use_lowT_limit_rates:
            data_var = rate.lim_lowT_rates(data_var, data_atm)
        data_var = rate.rev_rate(data_var, data_atm)
        data_var = rate.remove_rate(data_var)

        ini_abun = build_atm.InitialAbun()
        data_var = ini_abun.ini_y(data_var, data_atm)
        data_var = ini_abun.ele_sum(data_var)

        y_ini = np.asarray(data_var.y_ini, dtype=np.float64).copy()
        pco = np.asarray(data_atm.pco, dtype=np.float64).copy()
        Tco = np.asarray(data_atm.Tco, dtype=np.float64).copy()
        Kzz = np.asarray(data_atm.Kzz, dtype=np.float64).copy()

        data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
        make_atm.mol_diff(data_atm)
        make_atm.BC_flux(data_atm)

        solver = op.Ros2()
        if vulcan_cfg.use_photo:
            rate.make_bins_read_cross(data_var, data_atm)
            make_atm.read_sflux(data_var, data_atm)
            solver.compute_tau(data_var, data_atm)
            solver.compute_flux(data_var, data_atm)
            solver.compute_J(data_var, data_atm)
            data_var = rate.remove_rate(data_var)

        integ = op.Integration(solver, output)
        solver.naming_solver(data_para)
        integ(data_var, data_atm, data_para, make_atm)

        np.savez_compressed(
            out_npz,
            species=np.array(list(chem_funs.spec_list), dtype=object),
            nr=np.int64(chem_funs.nr),
            y_ini=y_ini,
            pco=pco,
            Tco=Tco,
            Kzz=Kzz,
            y=np.asarray(data_var.y, dtype=np.float64),
            ymix=np.asarray(data_var.ymix, dtype=np.float64),
            t=np.float64(data_var.t),
            dt=np.float64(data_var.dt),
            longdy=np.float64(data_var.longdy),
            count=np.int64(data_para.count),
            atom_loss_keys=np.array(list(data_var.atom_loss.keys()), dtype=object),
            atom_loss_vals=np.array(
                [float(v) for v in data_var.atom_loss.values()],
                dtype=np.float64,
            ),
        )
        print("MASTER_OK")
    finally:
        restore_files(existed)


run()
"""


_JAX_SCRIPT = r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

jax_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
count_max = int(sys.argv[3])
update_frq = int(sys.argv[4])
diff_esc = [sp for sp in sys.argv[5].split(",") if sp]
master_npz = Path(sys.argv[6])

os.chdir(jax_root)
sys.path.insert(0, str(jax_root))

from vulcan_jax.config import default_config
vulcan_cfg = default_config()

vulcan_cfg.count_max = count_max
vulcan_cfg.count_min = count_max + 1
vulcan_cfg.trun_min = 1e22
vulcan_cfg.update_frq = update_frq
vulcan_cfg.diff_esc = diff_esc
# The pinned VULCAN 2 oracle has no refreshed-interface vm and no hybrid
# flip (which extends the budget past count_max), so both are off.
vulcan_cfg.use_vm_mol = False
vulcan_cfg.use_hybrid_vm_mol = False
vulcan_cfg.use_print_prog = False

from vulcan_jax.runtime_validation import validate_runtime_config

validate_runtime_config(vulcan_cfg, root=jax_root)

# Seed from master's initial column (module docstring); everything after
# ini_y derives from it.
import vulcan_jax.ini_abun as _ini_abun

_MASTER_Y = np.asarray(
    np.load(master_npz, allow_pickle=True)["y_ini"], dtype=np.float64
)
_ini_abun._MODE_DISPATCH["EQ"] = lambda _atm: (_MASTER_Y.copy(), [])

import vulcan_jax.chem_funs as chem_funs
import vulcan_jax.legacy_io as op
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop
from vulcan_jax.state import RunState

rs = RunState.with_pre_loop_setup(vulcan_cfg)
rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())(rs)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    nr=np.int64(chem_funs.nr),
    y_ini=np.asarray(rs.metadata.y_ini, dtype=np.float64),
    pco=np.asarray(rs.atm.pco, dtype=np.float64),
    Tco=np.asarray(rs.atm.Tco, dtype=np.float64),
    Kzz=np.asarray(rs.atm.Kzz, dtype=np.float64),
    y=np.asarray(rs_out.step.y, dtype=np.float64),
    ymix=np.asarray(rs_out.step.ymix, dtype=np.float64),
    t=np.float64(rs_out.step.t),
    dt=np.float64(rs_out.step.dt),
    longdy=np.float64(rs_out.step.longdy),
    count=np.int64(rs_out.params.count),
    atom_loss_keys=np.array(list(rs_out.atoms.atom_order), dtype=object),
    atom_loss_vals=np.asarray(rs_out.atoms.atom_loss, dtype=np.float64),
)
print("JAX_OK")
"""


def _run_script(
    script: str,
    args: list[Path | int],
    *,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run a script string in a subprocess with PYTHONHASHSEED=0: master's
    matched-step trajectory depends on set order and is not reproducible
    without it."""
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )


def _atom_dict(data: np.lib.npyio.NpzFile) -> dict[str, float]:
    """Return atom-loss arrays from the npz as a dict."""
    return {
        str(key): float(value)
        for key, value in zip(
            data["atom_loss_keys"].tolist(),
            data["atom_loss_vals"].tolist(),
        )
    }


@pytest.mark.master_serial
def test_audit_refuses_a_contaminated_oracle() -> None:
    """The audit refuses an oracle checkout carrying VULCAN-JAX's own code (auditing against it is circular)."""
    from tools.audit_master_parity import _check_oracle_is_pristine

    with tempfile.TemporaryDirectory(prefix="fake_oracle_") as tmp:
        contaminated = Path(tmp)
        (contaminated / "op.py").write_text(
            "stall_window = getattr(vulcan_cfg, 'conv_stall_window', 200)\n"
        )
        errors = _check_oracle_is_pristine(contaminated)

    assert errors, "the guard did not flag a checkout containing conv_stall_window"
    assert any("conv_stall_window" in e for e in errors)
    # An unversioned copy is also flagged: provenance cannot be established.
    assert any("no .git" in e for e in errors)


@pytest.mark.master_serial
def test_audit_master_parity_against_pinned_oracle() -> None:
    """The static HD189 audit is clean against its exact, pristine oracle."""
    from oracle import require_oracle
    from tools.audit_master_parity import audit

    master = require_oracle("vulcan2_ncho")
    errors = audit(master, PACKAGE_ROOT, "vulcan2_ncho")
    assert errors == []


@pytest.mark.master_serial
@pytest.mark.parametrize(
    "count_max, update_frq, diff_esc, rtol, ymix_min", MATCHED_CASES
)
def test_default_hd189_preloop_and_matched_steps_match_master(
    count_max: int, update_frq: int, diff_esc: list[str], rtol: float, ymix_min: float
) -> None:
    """HD189 initial state and `count_max + 1` matched Ros2 steps match master."""
    from oracle import oracle_worktree

    with tempfile.TemporaryDirectory(prefix="default_parity_") as tmp, \
            oracle_worktree(
                "vulcan2_ncho",
                # The shipped default preset on both sides: an unmatched
                # composition is a different atmosphere, not a parity result.
                fastchem_abundance="solar_element_abundances.dat",
            ) as master_root:
        tmp_path = Path(tmp)
        master_npz = tmp_path / "master_hd189.npz"
        jax_npz = tmp_path / "jax_hd189.npz"
        master_backup = tmp_path / "master_backup"

        master_res = _run_script(
            _MASTER_SCRIPT,
            [
                master_root,
                master_npz,
                master_backup,
                count_max,
                update_frq,
                ",".join(diff_esc),
            ],
            timeout=900.0,
        )
        assert master_res.returncode == 0, (
            f"master subprocess failed {master_res.returncode}\n"
            f"--- stdout ---\n{master_res.stdout}\n"
            f"--- stderr ---\n{master_res.stderr}"
        )
        assert "MASTER_OK" in master_res.stdout

        jax_res = _run_script(
            _JAX_SCRIPT,
            [
                PACKAGE_ROOT,
                jax_npz,
                count_max,
                update_frq,
                ",".join(diff_esc),
                master_npz,
            ],
            timeout=900.0,
        )
        assert jax_res.returncode == 0, (
            f"JAX subprocess failed {jax_res.returncode}\n"
            f"--- stdout ---\n{jax_res.stdout}\n"
            f"--- stderr ---\n{jax_res.stderr}"
        )
        assert "JAX_OK" in jax_res.stdout

        master = np.load(master_npz, allow_pickle=True)
        jax = np.load(jax_npz, allow_pickle=True)

        assert list(jax["species"]) == list(master["species"])
        assert int(jax["nr"]) == int(master["nr"])
        # Exact because the JAX run was seeded from this very array; the two
        # SEEDS are compared, with their measured bar, in tests/test_eq_seed.py.
        np.testing.assert_array_equal(jax["y_ini"], master["y_ini"])
        np.testing.assert_array_equal(jax["pco"], master["pco"])
        np.testing.assert_array_equal(jax["Tco"], master["Tco"])
        np.testing.assert_allclose(jax["Kzz"], master["Kzz"], rtol=1e-14, atol=0.0)

        sig = np.asarray(master["ymix"]) > ymix_min
        y_relerr = relerr(jax["y"], master["y"], mask=sig)
        ymix_relerr = relerr(jax["ymix"], master["ymix"], mask=sig)
        t_relerr = abs(float(jax["t"]) - float(master["t"])) / abs(float(master["t"]))
        dt_relerr = abs(float(jax["dt"]) - float(master["dt"])) / abs(
            float(master["dt"])
        )

        max_relerr = max(y_relerr, ymix_relerr, t_relerr, dt_relerr)
        assert int(jax["count"]) == int(master["count"])
        assert max_relerr <= rtol, (
            f"HD189 matched-step relerr {max_relerr:.3e} > "
            f"{rtol:.3e}: y={y_relerr:.3e}, "
            f"ymix={ymix_relerr:.3e}, t={t_relerr:.3e}, dt={dt_relerr:.3e}, "
            f"longdy_jax={float(jax['longdy']):.3e}, "
            f"longdy_master={float(master['longdy']):.3e}, "
            f"atom_jax={_atom_dict(jax)}, atom_master={_atom_dict(master)}"
        )


# --- condensation: the `fix_species` pin snapshot ----------------------------
# Upstream's Earth methodology (cfg_examples/vulcan_cfg_Earth.py:107-120).
# No shipped config condenses, so this is the only guard on the order of
# pin snapshot and relax. It uses the smallest vendored network with a
# condensation row; isothermal const_mix with photo off needs no EQ seed or
# cross sections. The pin fires on step 45, leaving six steps.
CONDEN_STEPS = 50
CONDEN_KNOBS = {
    "atom_list": ["H", "O", "C"],
    "network": "thermo/CHO_photo_network_lowT.txt",
    "use_lowT_limit_rates": False,
    "atm_type": "isothermal",
    "Tiso": 250.0,
    "nz": 40,
    "P_b": 1.0e6,
    "P_t": 1.0e2,
    "atm_base": "H2",
    "ini_mix": "const_mix",
    "const_mix": {"H2": 0.9878, "H2O": 1.0e-2, "CH4": 1.0e-3,
                  "CO": 1.0e-3, "CO2": 2.0e-4},
    "use_photo": False,
    "use_ion": False,
    "use_Kzz": True,
    "Kzz_prof": "const",
    "const_Kzz": 1.0e5,
    "use_moldiff": True,
    "use_vz": False,
    "use_condense": True,
    "condense_sp": ["H2O"],
    "non_gas_sp": ["H2O_l_s"],
    "use_settling": True,
    "r_p": {"H2O_l_s": 1.0e-2},
    "rho_p": {"H2O_l_s": 0.9},
    "use_relax": ["H2O"],
    "humidity": 1.0,
    "start_conden_time": 0.0,
    "stop_conden_time": 1.0e3,
    "fix_species": ["H2O", "H2O_l_s"],
    "fix_species_from_coldtrap_lev": True,
    "use_sat_surfaceH2O": False,
    "use_ini_cold_trap": False,
    "use_topflux": False,
    "use_botflux": False,
    "use_fix_sp_bot": {},
    "diff_esc": [],
    "remove_list": [],
    "use_adapt_rtol": False,
    "count_max": CONDEN_STEPS,
    "count_min": CONDEN_STEPS + 1,
    "trun_min": 1.0e22,
    "use_print_prog": False,
    "save_evolution": False,
}
# Master-only knobs: `use_print_delta` and the plotter / live-UI switches,
# which JAX's default.yaml does not carry, set off so master's plotter stays
# quiet. The vm knobs are the same PRE-FLIP baseline as the HD189 cases above:
# the pinned VULCAN 2 oracle has no refreshed-interface vm, and the hybrid
# phase flip breaks the matched-count contract.
CONDEN_MASTER_ONLY = {
    "use_print_delta": False,
    "use_live_plot": False,
    "use_live_flux": False,
    "use_plot_end": False,
    "use_plot_evo": False,
    "use_save_movie": False,
    "use_flux_movie": False,
    "plot_TP": False,
}
CONDEN_JAX_ONLY = {
    "use_vm_mol": False, "use_hybrid_vm_mol": False, "high_temp_cut": False,
}
# Machine tolerance: realised 8.5e-16 on masked cells, 0.0 on the H2O_l_s column.
CONDEN_RTOL = 1.0e-12


def _cfg_lines(prefix: str, *tables: dict) -> str:
    """Render the knob tables as `<prefix><name> = <repr>` lines.

    One table, two renderings ('' for master's cfg module, 'cfg.' for the JAX
    config object), so the two sides of the comparison cannot drift apart.
    """
    return "\n".join(
        f"{prefix}{name} = {value!r}"
        for table in tables for name, value in table.items()
    )


_CONDEN_MASTER_SCRIPT = r'''
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

master_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])

(master_root / "vulcan_cfg.py").write_text(
    (master_root / "cfg_examples" / "vulcan_cfg_HD189.py").read_text()
    + "\n# === condensation pin parity overrides ===\n"
    + """%(master_cfg)s\n"""
)

# make_chem_funs.py writes chem_funs.py before its check_conserv(), which
# raises under numpy>=1.24; master's vulcan.py ignores that exit code, so
# only the module's importability matters.
res = subprocess.run([sys.executable, "make_chem_funs.py"], cwd=str(master_root),
                     capture_output=True, text=True, timeout=1800)
probe = subprocess.run(
    [sys.executable, "-c", "import chem_funs as c; assert c.ni > 0 and c.nr > 0"],
    cwd=str(master_root), capture_output=True, text=True, timeout=120)
if probe.returncode != 0:
    print(res.stdout[-2000:], res.stderr[-2000:], probe.stderr[-2000:])
    sys.exit(2)

os.chdir(master_root)
sys.path.insert(0, str(master_root))

import build_atm
import chem_funs
import op
import store

data_var = store.Variables()
data_atm = store.AtmData()
data_para = store.Parameters()
data_para.start_time = time.time()
make_atm = build_atm.Atm()
output = op.Output()

data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
make_atm.sp_sat(data_atm)
rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)
ini_abun = build_atm.InitialAbun()
data_var = ini_abun.ini_y(data_var, data_atm)
data_var = ini_abun.ele_sum(data_var)

y_ini = np.asarray(data_var.y_ini, dtype=np.float64).copy()
pco = np.asarray(data_atm.pco, dtype=np.float64).copy()
Tco = np.asarray(data_atm.Tco, dtype=np.float64).copy()

data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)

solver = op.Ros2()
solver.naming_solver(data_para)
op.Integration(solver, output)(data_var, data_atm, data_para, make_atm)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    y_ini=y_ini, pco=pco, Tco=Tco,
    y=np.asarray(data_var.y, dtype=np.float64),
    ymix=np.asarray(data_var.ymix, dtype=np.float64),
    t=np.float64(data_var.t), dt=np.float64(data_var.dt),
    count=np.int64(data_para.count),
    rejections=np.array([data_para.nega_count, data_para.loss_count,
                         data_para.delta_count], dtype=np.int64),
)
print("MASTER_OK")
'''


_CONDEN_JAX_SCRIPT = r'''
import os
import sys
from pathlib import Path

import numpy as np

package_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])

# `network` and `atom_list` are import-frozen: they must be selected before
# the first `import vulcan_jax`, which is why the JAX side needs a cold process.
os.environ["VULCAN_JAX_NETWORK"] = %(network)r
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
sys.path.insert(0, str(package_root.parent))
os.chdir(package_root)

import vulcan_jax  # noqa: F401
import vulcan_jax.chem_funs as chem_funs
import vulcan_jax.legacy_io as op
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop
from vulcan_jax.config import default_config
from vulcan_jax.state import RunState

cfg = default_config()
%(jax_cfg)s

rs = RunState.with_pre_loop_setup(cfg)
rs_out = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)(rs)
p = rs_out.params

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    y_ini=np.asarray(rs.metadata.y_ini, dtype=np.float64),
    pco=np.asarray(rs.atm.pco, dtype=np.float64),
    Tco=np.asarray(rs.atm.Tco, dtype=np.float64),
    y=np.asarray(rs_out.step.y, dtype=np.float64),
    ymix=np.asarray(rs_out.step.ymix, dtype=np.float64),
    t=np.float64(rs_out.step.t), dt=np.float64(rs_out.step.dt),
    count=np.int64(p.count),
    rejections=np.array([p.nega_count, p.loss_count, p.delta_count],
                        dtype=np.int64),
)
print("JAX_OK")
'''


@pytest.mark.master_serial
def test_conden_fix_species_pin_matches_master() -> None:
    """The `fix_species` pin freezes the reservoir master freezes.

    On the trigger step master snapshots `fix_y` from the post-solve y BEFORE
    the H2O relaxation (op.py:871-873, then op.py:898-902 relaxes inside the
    same block), so `_activate_fix_species` must run before `conden_branch`.
    Snapshotting the post-relax y pins a reservoir the relax has already
    drained, and the pin then holds that column for the rest of the run.
    """
    from oracle import oracle_worktree

    fmt = {
        "network": CONDEN_KNOBS["network"],
        "master_cfg": _cfg_lines("", CONDEN_KNOBS, CONDEN_MASTER_ONLY),
        "jax_cfg": _cfg_lines("cfg.", CONDEN_KNOBS, CONDEN_JAX_ONLY),
    }
    with tempfile.TemporaryDirectory(prefix="conden_parity_") as tmp, \
            oracle_worktree("vulcan2_ncho") as master_root:
        # Both sides must read the SAME network file: the condensation rows are
        # indexed positionally off `299 [ H2O -> H2O_l_s ]`.
        network_rel = CONDEN_KNOBS["network"]
        assert (master_root / network_rel).read_bytes() == (
            PACKAGE_ROOT / network_rel
        ).read_bytes(), f"{network_rel} differs between the oracle and this tree"

        tmp_path = Path(tmp)
        master_npz = tmp_path / "master_conden.npz"
        jax_npz = tmp_path / "jax_conden.npz"

        master_res = _run_script(
            _CONDEN_MASTER_SCRIPT % fmt, [master_root, master_npz],
            timeout=900.0,
        )
        assert master_res.returncode == 0 and "MASTER_OK" in master_res.stdout, (
            f"master subprocess failed {master_res.returncode}\n"
            f"--- stdout ---\n{master_res.stdout}\n"
            f"--- stderr ---\n{master_res.stderr}"
        )
        jax_res = _run_script(
            _CONDEN_JAX_SCRIPT % fmt, [PACKAGE_ROOT, jax_npz], timeout=900.0,
        )
        assert jax_res.returncode == 0 and "JAX_OK" in jax_res.stdout, (
            f"JAX subprocess failed {jax_res.returncode}\n"
            f"--- stdout ---\n{jax_res.stdout}\n"
            f"--- stderr ---\n{jax_res.stderr}"
        )
        # dict() forces the lazy npz reads before the temp dir goes away.
        master = dict(np.load(master_npz, allow_pickle=True))
        jax = dict(np.load(jax_npz, allow_pickle=True))

    # With use_photo=False master's codegen drops the photolysis rows and their
    # photo-only species (make_chem_funs.py:74), so its species list is a
    # subset of the port's; those extra columns must be exactly zero.
    sp_master = [str(name) for name in master["species"]]
    sp_jax = [str(name) for name in jax["species"]]
    cols = [sp_jax.index(name) for name in sp_master]
    extra = [i for i, name in enumerate(sp_jax) if name not in sp_master]
    assert np.all(np.asarray(jax["y"])[:, extra] == 0.0)
    y_jax = np.asarray(jax["y"])[:, cols]
    ymix_jax = np.asarray(jax["ymix"])[:, cols]

    np.testing.assert_array_equal(np.asarray(jax["y_ini"])[:, cols], master["y_ini"])
    np.testing.assert_array_equal(jax["pco"], master["pco"])
    np.testing.assert_array_equal(jax["Tco"], master["Tco"])
    np.testing.assert_array_equal(jax["rejections"], master["rejections"])
    assert int(jax["count"]) == int(master["count"]) == CONDEN_STEPS + 1

    # Trace cells clip to zero on different steps once dt has grown, so the
    # metric is read over master's resolved cells (as in the HD189 cases).
    sig = np.asarray(master["ymix"]) > 1.0e-10
    cond = sp_master.index("H2O_l_s")
    col_master = np.asarray(master["y"])[:, cond]
    col_jax = y_jax[:, cond]
    scores = {
        "y": relerr(y_jax, master["y"], mask=sig),
        "ymix": relerr(ymix_jax, master["ymix"], mask=sig),
        "H2O_l_s column": abs(col_jax.sum() - col_master.sum()) / col_master.sum(),
        "t": abs(float(jax["t"]) - float(master["t"])) / float(master["t"]),
        "dt": abs(float(jax["dt"]) - float(master["dt"])) / float(master["dt"]),
    }
    assert max(scores.values()) <= CONDEN_RTOL, (
        "condensation pin parity: "
        + ", ".join(f"{name}={value:.3e}" for name, value in scores.items())
        + f" > {CONDEN_RTOL:.1e}; nonzero H2O_l_s layers "
        f"master {int((col_master > 0).sum())} jax {int((col_jax > 0).sum())}"
    )
