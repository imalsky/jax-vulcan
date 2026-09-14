"""Guard on the Ros2 stage repair: a correction larger than its carrier cell
can carry is skipped (`jax_step._REPAIR_MAX_CELL_FRAC`, per layer and atom,
against `max(cell content, |raw stage change|)`).

The repair puts each atom's stage defect on a FIXED reservoir species
(H2, H2O, CO, N2, H2S). Where that species is a trace in the layer the
correction is 1e1 to 1e8 times the cell's own content and damages the cell
instead of healing the layer: on the converged W39b column at dt 1e11 the
unguarded repair drives H2S NEGATIVE across layers 83-94 (VMR 1e-8 to 2e-7,
above the spectroscopic floor) where the raw solve is accurate to 1.1e-3
(notes.md §1.13). A skipped correction leaves that layer's element budget
open, which the certificate's cumulative term (C23) sees.

Pins: (1) no H2S cell of that band is negative where the raw solve is
positive, at dt 1e8 and 1e11 -- and the unguarded repair fails that, so the
guard is what earns it (SNCHO, hence a subprocess: the network is
import-frozen); (2) on the healthy HD189 column at dt 1e6 the guard changes
no cell of either stage vector, while the repair itself does.

`_REPAIR_MAX_CELL_FRAC = 0.0` drops every nonzero correction, so it IS the
raw solve; `inf` is the unguarded repair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _helpers import run_child

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
S_NETWORK = "thermo/SNCHO_photo_network.txt"
BAND = slice(83, 95)  # W39b layers where H2S is a trace, not a reservoir
_GAMMA = 1.0 + 2.0**-0.5


def stage_arrays(fixture: str, cfg_name: str, dt: float, frac: float):
    """(y, k1, k2) of one Ros2 step from a saved converged state, with the
    repair guard set to `frac`. Also imported by the SNCHO child below."""
    import jax.numpy as jnp
    import vulcan_jax.chem as chem_mod
    import vulcan_jax.jax_step as jax_step
    import vulcan_jax.network as net_mod
    from vulcan_jax.config import load_config

    d = np.load(DATA / fixture)
    atm = jax_step.AtmStatic(
        **{
            f: (
                bool(d[f"atmbool__{f}"])
                if f"atmbool__{f}" in d
                else jnp.asarray(d[f"atm__{f}"])
            )
            for f in jax_step.AtmStatic._fields
        }
    )
    net = chem_mod.to_jax(net_mod.parse_network(load_config(cfg_name).network))
    y, k_arr = jnp.asarray(d["y_star"]), jnp.asarray(d["k_arr"])
    saved, jax_step._REPAIR_MAX_CELL_FRAC = jax_step._REPAIR_MAX_CELL_FRAC, frac
    try:
        k1, k2, _, _ = jax_step._ros2_stages(y, k_arr, jnp.float64(dt), atm, net, None)
    finally:
        jax_step._REPAIR_MAX_CELL_FRAC = saved
    return (np.asarray(y, dtype=np.float64),
            np.asarray(k1, dtype=np.float64),
            np.asarray(k2, dtype=np.float64))


def step_solution(y, k1, k2):
    return y + (3.0 / (2.0 * _GAMMA)) * k1 + (1.0 / (2.0 * _GAMMA)) * k2


# The child selects SNCHO (network) and an S-bearing atom_list before the first
# vulcan_jax import, then reuses this module's helper.
_CHILD = r"""
import os, sys
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "tests"))
import numpy as np
from test_stage_repair_guard import BAND, stage_arrays, step_solution

for dt in (1e8, 1e11):
    y, k1, k2 = stage_arrays("adj_state_w39b.npz", "W39b", dt, 0.0)
    raw = step_solution(y, k1, k2)
    import vulcan_jax.chem_funs as cf
    h2s = cf.spec_list.index("H2S")
    counts = {}
    for tag, frac in (("guarded", 1.0), ("unguarded", float("inf"))):
        y, k1, k2 = stage_arrays("adj_state_w39b.npz", "W39b", dt, frac)
        sol = step_solution(y, k1, k2)
        flipped = (sol[BAND, h2s] < 0.0) & (raw[BAND, h2s] > 0.0)
        counts[tag] = int(flipped.sum())
    print(f"dt={dt:.0e} band H2S flipped negative: {counts}", flush=True)
    assert counts["guarded"] == 0, (dt, counts)
    if dt >= 1e11:
        # Without the guard the whole band inverts; keeps the pin non-vacuous.
        assert counts["unguarded"] >= 10, (dt, counts)
print("PASS")
"""


@pytest.mark.skipif(
    not (DATA / "adj_state_w39b.npz").is_file(),
    reason="W39b fixture missing (npz artifacts are gitignored)",
)
def test_guard_keeps_the_w39b_trace_carrier_band_positive():
    res = run_child(_CHILD, network=S_NETWORK, label="stage-repair guard (W39b)")
    assert res.stdout.strip().endswith("PASS"), res.stdout


@pytest.mark.skipif(
    not (DATA / "adj_state_hd189.npz").is_file(),
    reason="HD189 fixture missing (npz artifacts are gitignored)",
)
def test_guard_is_a_no_op_on_the_healthy_hd189_column():
    """At dt 1e6 every correction on this column is <= 7e-6 of its carrier
    cell (notes.md §1.13), so the guard must be invisible: bit-identical
    stages to the unguarded repair, which itself is not the raw solve."""
    args = ("adj_state_hd189.npz", "default", 1e6)
    _, g1, g2 = stage_arrays(*args, 1.0)
    _, u1, u2 = stage_arrays(*args, float("inf"))
    _, r1, _ = stage_arrays(*args, 0.0)
    assert np.array_equal(g1, u1) and np.array_equal(g2, u2), (
        f"guard fired on a healthy column: max |d| "
        f"{max(np.max(np.abs(g1 - u1)), np.max(np.abs(g2 - u2))):.3e}"
    )
    assert not np.array_equal(g1, r1), "repair is inert here (vacuous test)"
