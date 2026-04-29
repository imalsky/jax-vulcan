"""Per-config 20-step Ros2 oracle vs VULCAN-master.

Runs each vendored ``cfg_examples/vulcan_cfg_<NAME>.py`` end-to-end (atm +
rates + photo + ini_abun + 20 Ros2 steps) on both VULCAN-master and
VULCAN-JAX, then compares the post-step ``y`` / ``ymix`` / ``t`` / ``dt``
arrays at each layer x species. Captures a baseline npz under
``tests/data/oracle_baselines/`` so future runs without master can still
validate against the captured state.

Parametrized over three configs that exercise paths HD189 doesn't:
- **Earth** — ``use_condense=True`` with ``T_cross_sp=['CO2','H2O','NH3']``
  and ``ini_mix='const_mix'``.
- **HD209** — ``ini_mix='EQ'``, no condensation, ``NCHO_photo_network.txt``
  (no S species), weaker gravity (936 cm/s^2 vs HD189's 2140).
- **Jupiter** — ``ini_mix='EQ'``, ``use_photo=True``, low-T rate caps
  active, condensation on (H2O, NH3).

Skips cleanly when:
- ``../VULCAN-master/`` is absent AND no baseline npz is on disk.
- A required input file (atm_file, sflux_file, etc.) is missing.

If master subprocess fails (rare), the JAX run still establishes a
baseline (or compares against an existing one). The skip message inlines
the master error.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
VULCAN_MASTER = ROOT.parent / "VULCAN-master"
BASELINE_DIR = ROOT / "tests" / "data" / "oracle_baselines"

# 20 matched Ros2 steps. Loose enough to absorb the documented chem_rhs
# cancellation floor (~1e-4 absolute on a few species per CLAUDE.md);
# tighter than 1e-9 on smooth pre-condensation phases would be surprising
# and worth investigating.
COUNT_MAX = 20

# Baseline-vs-JAX comparison: both runs are JAX, so the only drift is
# JIT compile order; bit-exact within float64 ULP.
RELERR_VS_BASELINE = 1e-12

# Per-config master-vs-JAX tolerance. Empirically HD209's 20-step run lands
# at ~1.7e-9 (vs HD189's 50-step 1.59e-10 baseline in CLAUDE.md): early-step
# relerr is dominated by the per-term ulp drift in chem_rhs documented in
# CLAUDE.md "Numerical hygiene" — closing it requires SymPy-faithful codegen
# which is WONTFIX. Earth and Jupiter remain at 1e-9.
ORACLE_CONFIGS = [
    pytest.param("Earth", 1e-9, id="Earth"),
    pytest.param("HD209", 3e-9, id="HD209"),
    pytest.param("Jupiter", 1e-9, id="Jupiter"),
]


# Defaults the vendored cfgs lack but VULCAN-master's runtime needs (gated
# by other flags / not exercised in this 20-step matched-step path, but
# `getattr`-style fallbacks are not used at every call site).
_MASTER_DEFAULTS = """
# === oracle test compatibility shims ===
use_adapt_rtol = False
use_fix_all_bot = False
conver_ignore = []
n_ccn = 1e2
fix_species_Ptop = []
K_deep = 0.0
EQ_ini_file = ''
photo_sp = []
sp_H = {}
g = gs
PItol = 1.0
"""


# ---------------------------------------------------------------------------
# Subprocess scripts: master and JAX runs share the cfg-pinning preamble.
# Each writes a NamedTuple-friendly npz of (y, ymix, t, dt, count, atom_loss)
# at sys.argv[2].
# ---------------------------------------------------------------------------
_MASTER_SCRIPT = r"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

master_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])
count_max = int(sys.argv[4])
backup_dir = Path(sys.argv[5])
master_extra_overrides = sys.argv[6] if len(sys.argv) > 6 else ""

# Backup master's cfg + chem_funs so we can restore on exit.
backup_dir.mkdir(parents=True, exist_ok=True)
import shutil as _shutil
for fn in ("vulcan_cfg.py", "chem_funs.py"):
    src = master_root / fn
    if src.exists():
        _shutil.copy2(src, backup_dir / fn)

# Stage the vendored cfg into master/vulcan_cfg.py, force matched-step
# integration (no convergence early-out, no live plotting / movie).
cfg_text = cfg_src.read_text()
overrides = (
    "\n# === oracle test overrides ===\n"
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
)
overrides += master_extra_overrides
(master_root / "vulcan_cfg.py").write_text(cfg_text + overrides)

os.chdir(master_root)
sys.path.insert(0, str(master_root))

# Regenerate chem_funs.py for this network (master's SymPy codegen).
import subprocess as _sp
res = _sp.run(
    [sys.executable, "make_chem_funs.py"],
    cwd=str(master_root), capture_output=True, text=True, timeout=600,
)
if res.returncode != 0:
    print("MASTER_FAIL: make_chem_funs.py exited", res.returncode)
    print("--- stdout ---", res.stdout[-1500:], sep="\n")
    print("--- stderr ---", res.stderr[-1500:], sep="\n")
    sys.exit(2)

# Master imports.
import numpy as np
import time

import vulcan_cfg
import store
import build_atm
import op

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

# Capture and persist final state.
np.savez_compressed(
    out_npz,
    y=np.asarray(data_var.y, dtype=np.float64),
    ymix=np.asarray(data_var.ymix, dtype=np.float64),
    t=np.float64(data_var.t),
    dt=np.float64(data_var.dt),
    count=np.int64(data_para.count),
    atom_loss_keys=np.array(list(data_var.atom_loss.keys()), dtype=object),
    atom_loss_vals=np.array(
        [float(v) for v in data_var.atom_loss.values()], dtype=np.float64,
    ),
    spec_list=np.array(list(__import__("chem_funs").spec_list), dtype=object),
)
print("MASTER_OK")
"""


_JAX_SCRIPT = r"""
from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

jax_root = Path(sys.argv[1])
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])
count_max = int(sys.argv[4])

os.chdir(jax_root)
sys.path.insert(0, str(jax_root))

# Pin vulcan_cfg to the vendored example before any runtime module imports
# its module-level constants.
import importlib.util
spec = importlib.util.spec_from_file_location("vulcan_cfg", cfg_src)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)
sys.modules["vulcan_cfg"] = cfg

# Force matched-step integration; convergence stays inert because
# count_min > count_max and trun_min is huge.
cfg.count_max = count_max
cfg.count_min = count_max + 1
cfg.trun_min = 1e22
cfg.use_print_prog = False
cfg.use_print_delta = False
cfg.use_live_plot = False
cfg.use_live_flux = False
cfg.use_plot_end = False
cfg.use_plot_evo = False
cfg.use_save_movie = False
cfg.use_flux_movie = False
cfg.save_evolution = False
cfg.plot_TP = False

import numpy as np

from runtime_validation import validate_runtime_config
validate_runtime_config(cfg, root=jax_root)

import legacy_io as op
from atm_setup import Atm
from ini_abun import InitialAbun
import op_jax
import outer_loop
from state import _Variables, _AtmData, _Parameters

# Sync vulcan_cfg references — outer_loop / legacy_io captured the module
# at their own import time; rebind so our cfg overrides land on the
# instance the runner actually reads.
op.vulcan_cfg = cfg
outer_loop.vulcan_cfg = cfg

import rates as _rates_mod

data_var = _Variables()
data_atm = _AtmData()
data_para = _Parameters()
data_para.start_time = time.time()
make_atm = Atm()
output = op.Output()

data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
if cfg.use_condense:
    make_atm.sp_sat(data_atm)

rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
_network = _rates_mod.setup_var_k(cfg, data_var, data_atm)

ini_abun = InitialAbun()
data_var = ini_abun.ini_y(data_var, data_atm)
data_var = ini_abun.ele_sum(data_var)

data_atm = make_atm.f_mu_dz(data_var, data_atm, output)
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)

solver = op_jax.Ros2JAX()
if cfg.use_photo:
    import photo_setup as _photo_setup
    _photo_setup.populate_photo_arrays(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    _rates_mod.apply_photo_remove(cfg, data_var, _network, data_atm)

integ = outer_loop.OuterLoop(solver, output)
solver.naming_solver(data_para)
integ(data_var, data_atm, data_para, make_atm)

import chem_funs as _cf
np.savez_compressed(
    out_npz,
    y=np.asarray(data_var.y, dtype=np.float64),
    ymix=np.asarray(data_var.ymix, dtype=np.float64),
    t=np.float64(data_var.t),
    dt=np.float64(data_var.dt),
    count=np.int64(data_para.count),
    atom_loss_keys=np.array(list(data_var.atom_loss.keys()), dtype=object),
    atom_loss_vals=np.array(
        [float(v) for v in data_var.atom_loss.values()], dtype=np.float64,
    ),
    spec_list=np.array(list(_cf.spec_list), dtype=object),
)
print("JAX_OK")
"""


def _parse_cfg_strings(cfg_path: Path) -> dict[str, str]:
    """Cheaply parse a vendored cfg's top-level string assignments.
    Returns the keys we actually consult from this test (paths and a few
    flags), without exec'ing the cfg. Quoted strings only.
    """
    src = cfg_path.read_text()
    out: dict[str, str] = {}
    for line in src.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if rhs.startswith("'") or rhs.startswith('"'):
            out[lhs] = rhs.strip().strip("'\"")
        elif rhs in ("True", "False"):
            out[lhs] = rhs
    return out


def _required_paths(cfg_path: Path) -> list[Path]:
    """Return the set of input files the cfg references that must exist
    in both VULCAN-JAX and VULCAN-master (vendored). BC flux files are
    only required when their corresponding ``use_topflux`` /
    ``use_botflux`` flag is true.
    """
    parsed = _parse_cfg_strings(cfg_path)
    keys: list[str] = ["network", "atm_file", "com_file", "gibbs_text"]
    if parsed.get("use_photo", "True") == "True":
        keys.append("sflux_file")
    if parsed.get("use_topflux", "False") == "True":
        keys.append("top_BC_flux_file")
    if parsed.get("use_botflux", "False") == "True":
        keys.append("bot_BC_flux_file")

    needed: list[Path] = []
    for k in keys:
        val = parsed.get(k, "")
        # Skip placeholders like 'atm/' (use_topflux=False on Earth).
        if val and not val.endswith("/"):
            needed.append(Path(val))
    return needed


def _check_inputs_present(cfg_path: Path, root: Path) -> Path | None:
    """Return the first missing input path, or None if all present."""
    for rel in _required_paths(cfg_path):
        if not (root / rel).exists():
            return root / rel
    return None


def _master_python() -> str:
    """Pick a Python interpreter that has sympy available (master needs
    it for ``make_chem_funs.py``). Prefer the conda ``base`` env when
    the current ``sys.executable`` lacks sympy.
    """
    probe_self = subprocess.run(
        [sys.executable, "-c", "import sympy"],
        capture_output=True, text=True, timeout=15.0,
    )
    if probe_self.returncode == 0:
        return sys.executable
    candidate = "/opt/homebrew/Caskroom/miniforge/base/bin/python"
    if Path(candidate).exists():
        probe = subprocess.run(
            [candidate, "-c", "import sympy"],
            capture_output=True, text=True, timeout=15.0,
        )
        if probe.returncode == 0:
            return candidate
    return sys.executable


def _run_subprocess(
    script: str,
    args: list[str],
    timeout: float = 600.0,
    python: str | None = None,
):
    """Execute *script* in a fresh interpreter with *args* tail-appended.
    Returns the CompletedProcess.
    """
    return subprocess.run(
        [python or sys.executable, "-c", script, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _restore_master(backup_dir: Path) -> None:
    """Restore master's vulcan_cfg.py + chem_funs.py from backup_dir."""
    if not backup_dir.exists():
        return
    for fn in ("vulcan_cfg.py", "chem_funs.py"):
        src = backup_dir / fn
        if src.exists():
            shutil.copy2(src, VULCAN_MASTER / fn)


def _run_master_subprocess(
    cfg_file: Path, master_npz: Path, backup_dir: Path,
) -> dict:
    """Attempt the master oracle subprocess. Always restores master tree
    in a finally clause. Returns a status dict.
    """
    try:
        res = _run_subprocess(
            _MASTER_SCRIPT,
            [
                str(VULCAN_MASTER), str(master_npz), str(cfg_file),
                str(COUNT_MAX), str(backup_dir), _MASTER_DEFAULTS,
            ],
            python=_master_python(),
        )
        return {
            "ran": True,
            "rc": res.returncode,
            "stdout": res.stdout[-3000:],
            "stderr": res.stderr[-3000:],
        }
    finally:
        _restore_master(backup_dir)


def _save_baseline(jax_npz: Path, baseline_file: Path) -> None:
    """Persist the JAX-final state to ``baseline_file`` for future
    standalone validation. Stored under both descriptive (`*_final`) and
    short keys so the comparator can read either form.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    jax_d = np.load(jax_npz, allow_pickle=True)
    np.savez_compressed(
        baseline_file,
        ymix_final=jax_d["ymix"],
        y_final=jax_d["y"],
        t_final=jax_d["t"],
        dt_final=jax_d["dt"],
        count_final=jax_d["count"],
        atom_loss_keys=jax_d["atom_loss_keys"],
        atom_loss_vals=jax_d["atom_loss_vals"],
        spec_list=jax_d["spec_list"],
        # Also store under raw keys so `_compare_states` can read it.
        ymix=jax_d["ymix"],
        y=jax_d["y"],
        t=jax_d["t"],
        dt=jax_d["dt"],
        count=jax_d["count"],
    )


def _safe_relerr(a: np.ndarray, b: np.ndarray, floor: float = 1e-30) -> float:
    """Max relative error |a - b| / max(|a|, floor), ignoring entries
    where both arrays are NaN (e.g. ``ymix = 0/0`` in fully empty
    layers — Jupiter's stratosphere above the cold trap). Mismatched
    NaN positions are treated as +inf to avoid silent miscomparison.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    both_nan = nan_a & nan_b
    only_one_nan = nan_a ^ nan_b
    if np.any(only_one_nan):
        return float("inf")
    a_clean = np.where(both_nan, 0.0, a)
    b_clean = np.where(both_nan, 0.0, b)
    diff = np.abs(a_clean - b_clean)
    denom = np.maximum(np.abs(a_clean), floor)
    rel = diff / denom
    if rel.size == 0:
        return 0.0
    return float(np.max(rel))


def _compare_states(ref_npz: Path, jax_npz: Path) -> dict:
    """Compute max relerr stats between ref (master or baseline) and JAX state."""
    ref_d = np.load(ref_npz, allow_pickle=True)
    jax_d = np.load(jax_npz, allow_pickle=True)

    ref_specs = list(ref_d["spec_list"])
    jax_specs = list(jax_d["spec_list"])
    if ref_specs != jax_specs:
        return {
            "status": "species_mismatch",
            "ref_n": len(ref_specs),
            "jax_n": len(jax_specs),
        }

    relerr_y = _safe_relerr(ref_d["y"], jax_d["y"])
    relerr_ymix = _safe_relerr(ref_d["ymix"], jax_d["ymix"])

    t_relerr = abs(float(ref_d["t"]) - float(jax_d["t"])) / max(
        abs(float(ref_d["t"])), 1e-30,
    )
    dt_relerr = abs(float(ref_d["dt"]) - float(jax_d["dt"])) / max(
        abs(float(ref_d["dt"])), 1e-30,
    )
    return {
        "status": "ok",
        "max_relerr_y": relerr_y,
        "max_relerr_ymix": relerr_ymix,
        "t_relerr": float(t_relerr),
        "dt_relerr": float(dt_relerr),
        "count_ref": int(ref_d["count"]),
        "count_jax": int(jax_d["count"]),
    }


@pytest.mark.master_serial
@pytest.mark.parametrize("cfg_name,relerr_vs_master", ORACLE_CONFIGS)
def test_oracle(cfg_name: str, relerr_vs_master: float) -> None:
    """Per-config 20-step Ros2 oracle. See module docstring for skips."""
    cfg_file = ROOT / "cfg_examples" / f"vulcan_cfg_{cfg_name}.py"
    baseline_file = BASELINE_DIR / f"{cfg_name.lower()}_20step.npz"

    if not cfg_file.exists():
        pytest.skip(f"Vendored cfg {cfg_file} not present.")

    missing_jax = _check_inputs_present(cfg_file, ROOT)
    if missing_jax is not None:
        pytest.skip(f"Required input file {missing_jax} not found in VULCAN-JAX.")

    have_master = VULCAN_MASTER.is_dir()
    have_baseline = baseline_file.exists()
    if not have_master and not have_baseline:
        pytest.skip(
            f"VULCAN-master sibling not present at {VULCAN_MASTER} and no "
            f"baseline at {baseline_file}; oracle skipped (VULCAN-JAX is "
            "standalone)."
        )

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"oracle_{cfg_name.lower()}_") as tmp:
        tmp_p = Path(tmp)
        master_npz = tmp_p / "master_state.npz"
        jax_npz = tmp_p / "jax_state.npz"
        backup_dir = tmp_p / "master_backup"

        # 1. Always run JAX side first — it's the authoritative kernel.
        t0 = time.time()
        res_j = _run_subprocess(
            _JAX_SCRIPT,
            [str(ROOT), str(jax_npz), str(cfg_file), str(COUNT_MAX)],
            timeout=600.0,
        )
        jax_wall = time.time() - t0

        assert res_j.returncode == 0, (
            f"JAX subprocess exited {res_j.returncode}\n"
            f"--- stdout ---\n{res_j.stdout}\n"
            f"--- stderr ---\n{res_j.stderr}"
        )
        assert "JAX_OK" in res_j.stdout, "JAX subprocess did not print JAX_OK"
        assert jax_npz.exists(), f"JAX did not produce {jax_npz}"
        print(f"[{cfg_name}] JAX wall={jax_wall:.1f}s")

        # 2. Try master. If it succeeds, use it as the oracle. Otherwise
        # fall back to the on-disk baseline (or write one from JAX).
        master_status: dict = {"ran": False}
        if have_master:
            missing_master = _check_inputs_present(cfg_file, VULCAN_MASTER)
            if missing_master is not None:
                master_status = {
                    "ran": False,
                    "skip_reason": (
                        f"input {missing_master} not in VULCAN-master tree"
                    ),
                }
            else:
                t0 = time.time()
                master_status = _run_master_subprocess(
                    cfg_file, master_npz, backup_dir,
                )
                master_status["wall"] = time.time() - t0

        master_ok = (
            master_status.get("ran")
            and master_status.get("rc") == 0
            and master_npz.exists()
        )

        if master_ok:
            cmp = _compare_states(master_npz, jax_npz)
            ref_label = "master"
            threshold = relerr_vs_master
        elif have_baseline:
            cmp = _compare_states(baseline_file, jax_npz)
            ref_label = "baseline"
            threshold = RELERR_VS_BASELINE
        else:
            # No ref available: write JAX as the new baseline so future
            # runs have something. Skip with master error inlined.
            _save_baseline(jax_npz, baseline_file)
            stdout = master_status.get("stdout", "")
            stderr = master_status.get("stderr", "")
            skip_reason = master_status.get("skip_reason", "")
            pytest.skip(
                f"VULCAN-master could not validate {cfg_name} "
                f"(rc={master_status.get('rc')!r}, "
                f"skip_reason={skip_reason!r}). Captured baseline at "
                f"{baseline_file} from JAX run for future regression "
                f"comparisons.\n--- master stdout ---\n{stdout}\n"
                f"--- master stderr ---\n{stderr}"
            )

        # 3. Validate the comparison.
        if cmp["status"] == "species_mismatch":
            pytest.skip(
                f"Species ordering differs between {ref_label} "
                f"({cmp['ref_n']}) and JAX ({cmp['jax_n']})."
            )

        max_relerr = max(
            cmp["max_relerr_y"], cmp["max_relerr_ymix"],
            cmp["t_relerr"], cmp["dt_relerr"],
        )
        master_wall = float(master_status.get("wall", 0.0)) if master_ok else 0.0
        print(
            f"[{cfg_name}] ref={ref_label} count_ref={cmp['count_ref']} "
            f"count_jax={cmp['count_jax']} master_wall={master_wall:.1f}s "
            f"max_relerr y={cmp['max_relerr_y']:.3e} "
            f"ymix={cmp['max_relerr_ymix']:.3e} "
            f"t={cmp['t_relerr']:.3e} dt={cmp['dt_relerr']:.3e}"
        )

        assert cmp["count_ref"] == cmp["count_jax"], (
            f"step count mismatch: {ref_label}={cmp['count_ref']} "
            f"jax={cmp['count_jax']}"
        )
        assert max_relerr <= threshold, (
            f"{cfg_name} oracle: max relerr {max_relerr:.3e} > "
            f"{threshold:.3e} vs {ref_label}: "
            f"y={cmp['max_relerr_y']:.3e} ymix={cmp['max_relerr_ymix']:.3e} "
            f"t={cmp['t_relerr']:.3e} dt={cmp['dt_relerr']:.3e}"
        )

        # 4. Refresh baseline only when master validated and baseline is
        # missing — preserves any earlier-captured baseline that has
        # already been peer-reviewed.
        if master_ok and not have_baseline:
            _save_baseline(jax_npz, baseline_file)
