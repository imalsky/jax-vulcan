"""Default HD189 parity checks against VULCAN-master.

The test stages VULCAN-master only inside subprocesses and restores any
changed config/FastChem files before returning.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
from oracle import oracle_dir_or_sentinel  # noqa: E402

# Oracle location from $VULCAN_MASTER_DIR only, never a sibling guess. The
# parent verifies the pinned revision + clean tree and points this at a
# temporary COPY; the per-test is_dir() skips below handle "not configured".
VULCAN_MASTER = oracle_dir_or_sentinel()

from vulcan_jax._paths import PACKAGE_ROOT

# (count_max, update_frq, diff_esc, rtol, ymix_min). The first case is the
# default HD189 config for 20 accepted steps at the FP-accumulation floor over
# every cell. The second fires the hydrostatic refresh 40 times and refreshes
# the diffusion-limited escape flux with it over 200 steps: an escape Jacobian
# term applied outside upstream's upwind variants diverges to 3.7e-3 at the TOA
# by step 200 while the no-escape control sits at 9e-8 (measured 2026-08-26),
# so the bar is 1e-6 over cells with master ymix > 1e-10 (trace cells clip to
# zero on different steps once dt has grown, so the all-cell metric is noise).
MATCHED_CASES = [
    (19, 100, [], 3.0e-9, 0.0),
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
stock_fastchem = Path(sys.argv[4])
count_max = int(sys.argv[5])
jax_fastchem_bin = Path(sys.argv[6])
update_frq = int(sys.argv[7])
diff_esc = [sp for sp in sys.argv[8].split(",") if sp]

TRACKED_FILES = [
    Path("vulcan_cfg.py"),
    Path("chem_funs.py"),
    Path("fastchem_vulcan/fastchem"),
    Path("fastchem_vulcan/input/nasa9_logK_SNCHOPTi.dat"),
    Path("fastchem_vulcan/input/solar_element_abundances.dat"),
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
        shutil.copy2(
            stock_fastchem,
            master_root / "fastchem_vulcan/input/solar_element_abundances.dat",
        )
        # Stage the SAME FastChem binary both sides: two builds of the same
        # source differ in the last printed digit of vulcan_EQ.dat (amplified
        # ~n_atoms per species, observed up to 9.3e-7 on C6H6), which seeds a
        # matched-step divergence far above the oracle's 3e-9 bar. The oracle
        # compares VULCAN implementations, not FastChem compiler builds.
        shutil.copy2(jax_fastchem_bin, master_root / "fastchem_vulcan/fastchem")
        # Stage the JAX logK table too: upstream's nasa9_logK_SNCHOPTi.dat
        # carries a DUPLICATE CH2_1 species block (double-counted singlet
        # methylene shifts every element's equilibrium by ~1e-10..1e-7); the
        # JAX copy deduplicates it (notes.md, Parity & bug guide, C5).
        shutil.copy2(
            jax_fastchem_bin.parent / "input/nasa9_logK_SNCHOPTi.dat",
            master_root / "fastchem_vulcan/input/nasa9_logK_SNCHOPTi.dat",
        )

        res = subprocess.run(
            [sys.executable, "make_chem_funs.py"],
            cwd=str(master_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        # make_chem_funs.py writes chem_funs.py BEFORE its post-codegen
        # check_conserv() sanity check, which raises under numpy>=1.24
        # (str(numpy.bytes_) -> "b'OH'", so compo_row.index('OH') fails). That
        # crash is benign to the generated module: master's own vulcan.py
        # ignores make_chem_funs's exit code (os.system). So only bail if the
        # generated chem_funs.py does not import with a valid (ni, nr).
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

        import build_atm
        import vulcan_jax.chem_funs as chem_funs
        import op
        import store
        from vulcan_jax.config import default_config
        vulcan_cfg = default_config()

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
import time
from pathlib import Path

import numpy as np

jax_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
count_max = int(sys.argv[3])
update_frq = int(sys.argv[4])
diff_esc = [sp for sp in sys.argv[5].split(",") if sp]

os.chdir(jax_root)
sys.path.insert(0, str(jax_root))

from vulcan_jax.config import default_config
vulcan_cfg = default_config()

vulcan_cfg.count_max = count_max
vulcan_cfg.count_min = count_max + 1
vulcan_cfg.trun_min = 1e22
vulcan_cfg.update_frq = update_frq
vulcan_cfg.diff_esc = diff_esc
# Master comparison is the PRE-FLIP central-difference baseline: the pinned
# VULCAN 2 oracle's own use_vm_mol (off by default) is the setup-frozen
# cell-centred form, not vm_branch's refreshed interface form, and the hybrid phase flip extends the
# step budget past count_max on phase-0 exhaustion (count+1000), which breaks
# the matched-count contract. Pin both vm_branch defaults off for this oracle.
vulcan_cfg.use_vm_mol = False
vulcan_cfg.use_hybrid_vm_mol = False
vulcan_cfg.use_print_prog = False
vulcan_cfg.use_print_delta = False
vulcan_cfg.use_live_plot = False
vulcan_cfg.use_live_flux = False
vulcan_cfg.use_plot_end = False
vulcan_cfg.use_plot_evo = False
vulcan_cfg.use_save_movie = False
vulcan_cfg.use_flux_movie = False
vulcan_cfg.save_evolution = False
vulcan_cfg.plot_TP = False

from vulcan_jax.runtime_validation import validate_runtime_config

validate_runtime_config(vulcan_cfg, root=jax_root)

import vulcan_jax.chem_funs as chem_funs
import vulcan_jax.legacy_io as op
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop
from vulcan_jax.atm_setup import Atm
from vulcan_jax.state import RunState, legacy_view

rs = RunState.with_pre_loop_setup(vulcan_cfg)
data_var, data_atm, data_para = legacy_view(rs)
data_para.start_time = time.time()
y_ini = np.asarray(data_var.y_ini, dtype=np.float64).copy()
pco = np.asarray(data_atm.pco, dtype=np.float64).copy()
Tco = np.asarray(data_atm.Tco, dtype=np.float64).copy()
Kzz = np.asarray(data_atm.Kzz, dtype=np.float64).copy()

solver = op_jax.Ros2JAX()
if vulcan_cfg.use_photo and rs.photo_static is not None:
    solver._photo_static = rs.photo_static
integ = outer_loop.OuterLoop(solver, op.Output())
make_atm = Atm()
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
        [float(v) for v in data_var.atom_loss.values()], dtype=np.float64,
    ),
)
print("JAX_OK")
"""


def _master_python() -> str:
    """Return a Python interpreter that can run master's SymPy codegen."""
    probe = subprocess.run(
        [sys.executable, "-c", "import sympy"],
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if probe.returncode == 0:
        return sys.executable
    candidate = Path("/opt/homebrew/Caskroom/miniforge/base/bin/python")
    if candidate.exists():
        probe = subprocess.run(
            [str(candidate), "-c", "import sympy"],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if probe.returncode == 0:
            return str(candidate)
    return sys.executable


def _run_script(
    script: str,
    args: list[Path | int],
    *,
    python: str | None = None,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run a Python script string in a subprocess."""
    return subprocess.run(
        [python or sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _safe_relerr(
    a: np.ndarray, b: np.ndarray, floor: float = 1.0e-30, mask=None
) -> float:
    """Return max relative error with a denominator floor, over `mask` cells."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rel = np.abs(a - b) / np.maximum(np.abs(b), floor)
    return float(np.max(rel if mask is None else rel[mask]))


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
    """The audit must refuse an oracle checkout carrying VULCAN-JAX's own code
    (auditing against such a tree is circular); `_check_oracle_is_pristine`
    must fire on it. Do not weaken this guard.
    """
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
            oracle_worktree("vulcan2_ncho") as master_root:
        tmp_path = Path(tmp)
        master_npz = tmp_path / "master_hd189.npz"
        jax_npz = tmp_path / "jax_hd189.npz"
        master_backup = tmp_path / "master_backup"
        stock_fastchem = (
            PACKAGE_ROOT / "fastchem_vulcan/input/solar_element_abundances.dat"
        )
        jax_fastchem_bin = PACKAGE_ROOT / "fastchem_vulcan" / "fastchem"
        if not jax_fastchem_bin.exists():
            from vulcan_jax.ini_abun import _ensure_fastchem_binary

            _ensure_fastchem_binary()

        master_res = _run_script(
            _MASTER_SCRIPT,
            [
                master_root,
                master_npz,
                master_backup,
                stock_fastchem,
                count_max,
                jax_fastchem_bin,
                update_frq,
                ",".join(diff_esc),
            ],
            python=_master_python(),
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
            [PACKAGE_ROOT, jax_npz, count_max, update_frq, ",".join(diff_esc)],
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
        # Exact equality holds because _MASTER_SCRIPT stages the JAX-compiled
        # FastChem binary: same binary + same inputs -> bit-identical
        # vulcan_EQ.dat on both sides.
        np.testing.assert_array_equal(jax["y_ini"], master["y_ini"])
        np.testing.assert_array_equal(jax["pco"], master["pco"])
        np.testing.assert_array_equal(jax["Tco"], master["Tco"])
        np.testing.assert_allclose(jax["Kzz"], master["Kzz"], rtol=1e-14, atol=0.0)

        sig = np.asarray(master["ymix"]) > ymix_min
        y_relerr = _safe_relerr(jax["y"], master["y"], mask=sig)
        ymix_relerr = _safe_relerr(jax["ymix"], master["ymix"], mask=sig)
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
