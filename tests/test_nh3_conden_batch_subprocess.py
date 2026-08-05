"""End-to-end batched NH3-relaxation condensation: solo vs `run_batch` lanes.

Pins that the NH3 cold-trap index rides `ProfileVars` per lane (not the
runner closure): two cold Jupiter-like profiles with genuinely different
cold-trap indices are run solo and batched on profile A's runner, and each
lane must match its solo run for the abundant species (see the YMIX_FLOOR
note in the child for why bitwise all-species agreement is impossible under
vmap).

The default import-locked network has no `NH3_l_s`, so the child selects the
lowT Jupiter network via `$VULCAN_JAX_NETWORK`; config values follow master's
Jupiter example with an analytical Heng+14 T-P so no atm file is read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONDEN_NETWORK = "thermo/NCHO_photo_network_lowT_Jupiter.txt"

_CHILD = r"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["VULCAN_JAX_NETWORK"] = "thermo/NCHO_photo_network_lowT_Jupiter.txt"

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
os.chdir(repo / "src" / "vulcan_jax")

import numpy as np
import vulcan_jax
import vulcan_jax.composition as composition

sl = list(composition.species)
for need in ("NH3", "NH3_l_s", "H2O", "H2O_l_s"):
    assert need in sl, f"lowT Jupiter network missing {need!r}"

from vulcan_jax.config import default_config
cfg = default_config()

# Cold Jupiter-like gas giant (master cfg_examples/vulcan_cfg_Jupiter.py values
# where applicable); analytical T-P so the two profiles need no atm files.
cfg.atm_type = "analytical"
cfg.atm_base = "H2"
# Deterministic fixed diffusion scheme for the batched regime (the new hybrid
# default would flip schemes mid-run per lane).
cfg.use_vm_mol = False
cfg.use_hybrid_vm_mol = False
cfg.nz = 60
cfg.P_b = 5e9
cfg.P_t = 1e-2
cfg.Rp = 7.1492e9
cfg.Mp = 2479.0 * 7.1492e9**2 / 6.67430e-8  # -> g=G*Mp/Rp^2 = 2479
cfg.use_Kzz = True
cfg.Kzz_prof = "const"
cfg.const_Kzz = 1e8
cfg.use_vz = False
cfg.use_moldiff = True
cfg.use_photo = False
cfg.use_ion = False
cfg.ini_mix = "const_mix"
cfg.const_mix = {"H2": 0.864, "He": 0.134, "CH4": 1.8e-3, "NH3": 7e-4, "H2O": 1.0e-3}
cfg.use_topflux = False
cfg.use_botflux = False
cfg.use_fix_sp_bot = {}
cfg.use_ini_cold_trap = False
cfg.use_print_prog = False
cfg.use_live_plot = False
cfg.use_live_flux = False
cfg.use_condense = True
cfg.use_settling = True
cfg.condense_sp = ["H2O", "NH3"]
cfg.non_gas_sp = ["H2O_l_s", "NH3_l_s"]
cfg.r_p = {"H2O_l_s": 5e-5, "NH3_l_s": 5e-5}
cfg.rho_p = {"H2O_l_s": 0.9, "NH3_l_s": 0.7}
cfg.use_relax = ["H2O", "NH3"]
cfg.humidity = 1.0
cfg.start_conden_time = 0.0
cfg.stop_conden_time = 1e22
cfg.count_max = 40
cfg.count_min = 1

from vulcan_jax.state import RunState
import vulcan_jax.legacy_io as op
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop

# Two Heng+14 parameter sets whose temperature minima sit at different layers,
# so the per-profile NH3 cold-trap indices differ (the vacuity guard below).
PARA_A = [120.0, 130.0, 0.05, 0.02, 1.0, 1.0]
PARA_B = [100.0, 170.0, 0.2, 0.01, 1.0, 1.0]

t0 = time.time()
cfg.para_anaTP = PARA_A
rsA = RunState.with_pre_loop_setup(cfg, skip_chem_warmup=True)
integA = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
sA, atmA = integA.prepare_runstate(rsA)

cfg.para_anaTP = PARA_B
rsB = RunState.with_pre_loop_setup(cfg, skip_chem_warmup=True)
integB = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
sB, atmB = integB.prepare_runstate(rsB)

topA = int(np.asarray(sA.pv.c_nh3_conden_top))
topB = int(np.asarray(sB.pv.c_nh3_conden_top))
assert topA != topB, f"vacuous test: cold-trap indices equal ({topA})"
print(f"TOPS A={topA} B={topB} ({time.time()-t0:.0f}s setup)")

# Solo B runs on its OWN runner (closure baked from B); the batch runs on A's.
# If the cold-trap index (or any conden array) were still closure-baked, lane B
# would silently use A's value while soloB uses B's -> divergence.
soloA = integA._runner(sA, atmA)
soloB = integB._runner(sB, atmB)
batched = integA.run_batch(
    outer_loop.stack_integ_states([sA, sB]),
    outer_loop.stack_atm_statics([atmA, atmB]),
)
out = outer_loop.unstack_integ_states(batched, 2)

# Batch-vs-solo agreement is asserted on the abundant, well-conditioned
# species (ymix > 1e-4: the H2/He bath, CH4, and the condensing gases
# H2O/NH3). It CANNOT be asserted bitwise across all species: jax.vmap fuses
# reductions in a different order than the scalar runner, so the batched and
# solo trajectories diverge at the ULP level, and this stiff network amplifies
# that chaotically in ill-conditioned trace radicals (C2H4, N, NO, ... at
# <1e-10 abundance reach ~15% by convergence) while the abundant species stay
# tight (~1e-5). All per-lane inputs and the step count are identical (the
# stacked c_nh3_sat/k_arr differ by lane and accept_count matches solo) — only
# FP associativity differs. The invariant this test targets, no per-lane state
# leak (lane 0's cold-trap index / sat profile / Dg bleeding into lane 1),
# would corrupt the abundant condensing gases: a leak drives relB ~0.1+ (the
# original baked-index bug gave 0.16), far above the 1e-2 gate, so it is still
# fully caught here.
YMIX_FLOOR = 1e-4

def rel(b, r):
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    m = np.abs(r) > YMIX_FLOOR
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(b[m] - r[m]) / np.abs(r[m])))

relA = rel(out[0].ymix, soloA.ymix)
relB = rel(out[1].ymix, soloB.ymix)
differ = rel(soloB.ymix, soloA.ymix)
j = sl.index("NH3_l_s")
ice = max(float(np.asarray(soloA.ymix)[:, j].sum()),
          float(np.asarray(soloB.ymix)[:, j].sum()))
print(f"relA={relA:.3e} relB={relB:.3e} differ={differ:.3e} NH3_l_s={ice:.3e}")
print(f"total {time.time()-t0:.0f}s")
assert relA < 1e-2, f"lane A diverged from solo: {relA:.3e}"
assert relB < 1e-2, f"lane B diverged from solo: {relB:.3e}"
assert differ > 1e-6, f"profiles identical ({differ:.3e}) - vacuous"
assert ice > 0.0, "NH3 condensate never formed - relax path not exercised"
print("PASS")
"""


def test_nh3_conden_batch_subprocess():
    from vulcan_jax._paths import PACKAGE_ROOT

    if not (PACKAGE_ROOT / CONDEN_NETWORK).exists():
        pytest.skip(f"condensate network {CONDEN_NETWORK!r} not vendored")

    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": CONDEN_NETWORK,
    }
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=ROOT,
    )
    assert res.returncode == 0, (
        f"NH3 batched-conden subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    assert "PASS" in res.stdout, res.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
