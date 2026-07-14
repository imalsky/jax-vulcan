"""End-to-end smooth-rainout runtime checks on the SNCHO network (Route B
B0-2/B0-3/B0-4/B0-5), in a subprocess (the network and the reservoir
projection tables are import-locked, so the S-bearing configuration needs a
fresh process — same pattern as test_atom_conservation_s_subprocess.py).

Self-contained: isothermal T-P, const_mix init, no photo, no FastChem.

Child checks:
  1. Loud config validation: bad conden_mode, fix_species in smooth mode,
     and multi-species condense_sp all raise.
  2. Cold supersaturated column (400 K, S8 over-abundant at depth, H2S
     bottom pin): per-step ledger telescoping identity holds EXACTLY
     (led_step + led_renorm + led_bc == N_E(after) - N_E(before)), the
     sink is active (led_rain[S] > 0, led_step[S] < 0), the boundary
     ledger touches ONLY H and S (the pinned H2S's elements), the S8
     network kinetics rows stay zero (inert in this mode), the accept
     gate masks exactly {H, S}, and the direct-residual evaluator returns
     finite scaled residuals with the pinned cell excluded.
  3. Hot subsaturated column (1500 K): the sink is EXACTLY zero
     (led_rain == 0, led_step ~ conservation floor for S) and the smooth
     run's state after a fixed number of accepted steps matches a plain
     conden-off run to solver noise (gate G4 seed).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S_NETWORK = "thermo/SNCHO_photo_network.txt"

_CHILD = r"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"

from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
os.chdir(repo / "src" / "vulcan_jax")

import numpy as np
import jax.numpy as jnp

import vulcan_jax.vulcan_cfg as cfg
cfg.atom_list = ["H", "O", "C", "N", "S"]

import vulcan_jax.legacy_io as op
import vulcan_jax.op_jax as op_jax
import vulcan_jax.outer_loop as outer_loop
import vulcan_jax.jax_step as js
from vulcan_jax.state import RunState
from vulcan_jax import state as state_mod
from vulcan_jax.chem_funs import spec_list as SL
from vulcan_jax.steady_residual import residual_from_state

ATOMS = ("H", "O", "C", "N", "S")

def base_cfg(T):
    cfg.atm_type = "isothermal"; cfg.Tiso = float(T)
    cfg.atm_base = "H2"; cfg.use_moldiff = True
    cfg.use_Kzz = True; cfg.Kzz_prof = "const"; cfg.const_Kzz = 1e7; cfg.use_vz = False
    cfg.use_photo = False; cfg.use_ion = False
    cfg.ini_mix = "const_mix"
    cfg.const_mix = {"H2": 0.85, "He": 0.148, "H2O": 5e-4, "CO": 2e-4,
                     "CH4": 1e-4, "N2": 1e-4, "H2S": 3e-5, "S8": 1e-4}
    cfg.nz = 24; cfg.P_b = 7.6e6; cfg.P_t = 1e2
    cfg.use_topflux = False; cfg.use_botflux = False
    cfg.use_ini_cold_trap = False; cfg.use_print_prog = False
    cfg.use_settling = False; cfg.use_relax = []
    cfg.fix_species = []
    cfg.count_max = 500; cfg.count_min = 1; cfg.runtime = 1e30
    cfg.use_condense = True
    cfg.conden_mode = "smooth_rainout"
    cfg.conden_smooth_width = 0.1
    cfg.rainout_rate_scale = 1.0
    cfg.condense_sp = ["S8"]; cfg.non_gas_sp = ["S8_l_s"]
    cfg.r_p = {"S8_l_s": 5e-3}; cfg.rho_p = {"S8_l_s": 2.07}
    cfg.use_fix_sp_bot = {"H2S": 3e-5}

def fresh_run(chunk_steps):
    rs = RunState.with_pre_loop_setup(cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
    var, atm, para = state_mod.legacy_view(rs)
    integ._ensure_runner(var, atm)
    state = integ._pack_state_from_runstate(rs)
    atm_static = js.make_atm_static(atm, len(SL), int(cfg.nz), cfg=cfg)
    return rs, integ, state, atm_static

# ---- Check 1: loud validation ------------------------------------------------
base_cfg(400.0)
cfg.conden_mode = "not_a_mode"
try:
    fresh_run(1); raise SystemExit("bad conden_mode did not raise")
except ValueError as e:
    assert "conden_mode" in str(e)
base_cfg(400.0)
cfg.fix_species = ["S8", "S8_l_s"]
try:
    fresh_run(1); raise SystemExit("fix_species in smooth mode did not raise")
except ValueError as e:
    assert "fix_species" in str(e)
base_cfg(400.0)
cfg.condense_sp = ["H2O", "S8"]
try:
    fresh_run(1); raise SystemExit("multi-species condense_sp did not raise")
except ValueError as e:
    assert "S8" in str(e)
print("VALIDATION_OK")

# ---- Check 2: cold supersaturated column ------------------------------------
base_cfg(400.0)
rs, integ, state, atm_static = fresh_run(6)
st = integ._statics
assert bool(st.use_smooth_rainout)

# gate mask: exactly H and S are open (S8 rainout opens S; H2S pin opens H+S)
gate = np.asarray(st.gate_atom_mask)
assert list(gate) == [False, True, True, True, False], f"gate mask {gate}"

compo = np.asarray(st.compo_arr)
i_s8 = SL.index("S8")
conden_rows = np.asarray(integ._conden_static.conden_re_idx)

led_sum_hist = []
for k in range(6):
    target = int(np.asarray(state.accept_count)) + 1
    state = state._replace(chunk_target=jnp.int32(target))
    y_before = np.asarray(state.y, dtype=np.float64)
    dz_used = np.asarray(state.dz, dtype=np.float64)
    state = integ._runner(state, atm_static)
    assert int(np.asarray(state.accept_count)) == target, "chunk step did not accept exactly once"
    y_after = np.asarray(state.y, dtype=np.float64)

    N_before = np.einsum("zi,ie,z->e", y_before, compo, dz_used)
    N_after = np.einsum("zi,ie,z->e", y_after, compo, dz_used)
    led = (np.asarray(state.led_step) + np.asarray(state.led_renorm)
           + np.asarray(state.led_bc))
    # Exactness is relative to the INVENTORY: the runner's jnp.einsum and
    # this np.einsum order their sums differently, so each N_E carries
    # ~1e-16 relative noise that dominates when Delta N_E ~ 0 (conserved
    # elements). The telescoping identity itself is exact by construction.
    scale = np.maximum(np.maximum(np.abs(N_after), np.abs(N_before)), 1e-300)
    err = np.max(np.abs(led - (N_after - N_before)) / scale)
    led_sum_hist.append(err)
    assert err < 1e-12, f"ledger telescoping violated at step {k}: rel {err:.3e}"

    # k_arr conden kinetics rows must stay zero (inert network rows)
    k_np = np.asarray(state.k_arr)
    assert float(np.max(np.abs(k_np[conden_rows]))) == 0.0
    assert float(np.max(np.abs(k_np[conden_rows + 1]))) == 0.0

led_rain = np.asarray(state.led_rain)
led_step = np.asarray(state.led_step)
led_bc = np.asarray(state.led_bc)
assert led_rain[4] > 0.0, f"S rainout inactive: led_rain={led_rain}"
assert led_step[4] < 0.0, f"S not removed by the step: led_step={led_step}"
# rainout removes ONLY S (S8 is a pure-S molecule)
assert np.all(led_rain[:4] == 0.0), f"non-S rainout: {led_rain}"
# boundary enforcement touches ONLY the pinned H2S's elements (H and S),
# in H2S stoichiometry: exactly 2 H per S (the delta-array einsum measures
# the pin's own mass change with no cancellation floor)
assert led_bc[1] == 0.0 and led_bc[2] == 0.0 and led_bc[3] == 0.0, f"led_bc={led_bc}"
assert led_bc[4] != 0.0, f"H2S pin produced no S delta: led_bc={led_bc}"
np.testing.assert_allclose(led_bc[0], 2.0 * led_bc[4], rtol=1e-12)
print(f"COLD_OK led_rain_S={led_rain[4]:.3e} led_step_S={led_step[4]:.3e} "
      f"led_bc_H={led_bc[0]:.3e} led_bc_S={led_bc[4]:.3e} "
      f"telescope_max={max(led_sum_hist):.2e}")

# residual evaluator: finite, pinned bottom cell excluded from R
rep = residual_from_state(integ, state, atm_static)
assert np.all(np.isfinite(np.asarray(rep.F)))
assert np.all(np.isfinite(np.asarray(rep.R)))
assert bool(np.asarray(rep.mask)[0, SL.index("H2S")]), "pinned cell not excluded"
assert float(rep.max_R) > 0.0
print(f"RESIDUAL_OK max_R={float(rep.max_R):.3e} at z={int(rep.argmax_z)} "
      f"i={SL[int(rep.argmax_i)]}")

# ---- Check 2b: adjoint body map carries the sink -----------------------------
# make_body_terms on a smooth state must pack the carry's RainoutTerm and the
# carry pin values, and the sink must actually act inside the body map: a map
# built WITHOUT the term must differ on the supersaturated S8 cells. This is
# exactly the silent-drop bug class the old NotImplementedError guarded.
from vulcan_jax.steady_state_grad import make_body_terms, _make_body_map
from vulcan_jax import chem_funs as _cf

atm_step, terms = make_body_terms(integ, state, atm_static)
assert terms.rainout is not None, "smooth state produced no RainoutTerm"
np.testing.assert_allclose(
    np.asarray(terms.rainout.n_sat),
    np.asarray(state.pv.c_sat_n_per_re[int(integ._statics.rainout_re_row)]),
    rtol=0)
np.testing.assert_allclose(
    np.asarray(terms.bot_val),
    np.asarray(state.pv.bot_pin_mix) * float(np.asarray(state.pv.n_0)[0]),
    rtol=1e-15)
_, body_map, _, _ = _make_body_map(
    state.y, state.k_arr, atm_step, _cf._NET_JAX, 1e3, "renorm", None, terms)
_, body_map_nosink, _, _ = _make_body_map(
    state.y, state.k_arr, atm_step, _cf._NET_JAX, 1e3, "renorm", None,
    terms._replace(rainout=None))
g_with = np.asarray(body_map(state.y))
g_without = np.asarray(body_map_nosink(state.y))
assert np.all(np.isfinite(g_with))
n_s8_now = np.asarray(state.y[:, i_s8])
n_sat_now = np.asarray(terms.rainout.n_sat)
super_cells = n_s8_now > n_sat_now
assert super_cells.any(), "no supersaturated S8 cell at the probe state"
ds8 = np.abs(g_with[:, i_s8] - g_without[:, i_s8])
assert ds8[super_cells].max() > 0.0, (
    "rainout term does not act in the body map (S8 rows identical)")
# NOTE: no zero-delta assertion on subsaturated cells -- the implicit step
# couples layers (transport) and the renorm redistributes, so the sink's
# effect legitimately propagates column-wide within one map application;
# the hinge's exact one-sidedness is pinned at kernel level
# (test_smooth_rainout_kernel).
print(f"BODYMAP_OK sink_delta_max={ds8[super_cells].max():.3e} "
      f"n_super={int(super_cells.sum())}")

# ---- Check 3: hot subsaturated column — exact zero + conden-off agreement ----
N_STEPS = 30
base_cfg(1500.0)
cfg.use_fix_sp_bot = {}
rs_s, integ_s, state_s, atm_s = fresh_run(N_STEPS)
state_s = state_s._replace(chunk_target=jnp.int32(N_STEPS))
state_s = integ_s._runner(state_s, atm_s)
assert int(np.asarray(state_s.accept_count)) == N_STEPS
assert float(np.max(np.abs(np.asarray(state_s.led_rain)))) == 0.0, "hot sink not exactly zero"

base_cfg(1500.0)
cfg.use_fix_sp_bot = {}
cfg.use_condense = False
cfg.conden_mode = "master_pin"
cfg.condense_sp = []
rs_o, integ_o, state_o, atm_o = fresh_run(N_STEPS)
state_o = state_o._replace(chunk_target=jnp.int32(N_STEPS))
state_o = integ_o._runner(state_o, atm_o)
assert int(np.asarray(state_o.accept_count)) == N_STEPS

y_s = np.asarray(state_s.y); y_o = np.asarray(state_o.y)
ymix_o = np.asarray(state_o.ymix)
live = ymix_o > 1e-25
rel = np.max(np.abs(y_s[live] - y_o[live]) / np.maximum(np.abs(y_o[live]), 1e-300))
assert rel < 1e-8, f"hot smooth vs conden-off endpoint mismatch: rel {rel:.3e}"
print(f"HOT_OK rel_vs_off={rel:.3e} t_s={float(state_s.t):.3e} t_o={float(state_o.t):.3e}")

print("PASS")
"""


def test_smooth_rainout_runtime_subprocess():
    from vulcan_jax._paths import PACKAGE_ROOT

    if not (PACKAGE_ROOT / S_NETWORK).exists():
        pytest.skip(f"sulfur network {S_NETWORK!r} not vendored")

    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "VULCAN_JAX_NETWORK": S_NETWORK,
    }
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=ROOT,
    )
    assert res.returncode == 0, (
        f"smooth-rainout subprocess exited {res.returncode}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    for marker in ("VALIDATION_OK", "COLD_OK", "RESIDUAL_OK", "HOT_OK"):
        assert marker in res.stdout, res.stdout
    assert res.stdout.strip().endswith("PASS"), res.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
