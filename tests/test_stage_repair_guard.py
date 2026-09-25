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
import-frozen); (2) on the healthy HD189 column at dt 1e6 the p99 of the
per-cell correction ratio stays under the 1.1e-2 of §1.13 and the guard
clamps only the thermospheric exception that note names: H2O at 5923 K and
6000 K, dissociated to a VMR of 5e-18 and below.

`_REPAIR_MAX_CELL_FRAC = 0.0` drops every nonzero correction, so it IS the
raw solve; `inf` is the unguarded repair.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest
from _helpers import run_child

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
W39B_FIXTURE = DATA / "adj_state_w39b.npz"
_MANIFEST = DATA / "FIXTURES.json"
BUILT_ON = (
    json.loads(_MANIFEST.read_text()).get("platform_machine")
    if _MANIFEST.is_file()
    else None
)
S_NETWORK = "thermo/SNCHO_photo_network.txt"
BAND = slice(83, 95)  # W39b layers where H2S is a trace, not a reservoir
_GAMMA = 1.0 + 2.0**-0.5

# The band's H2S cells sit at VMR 1e-8 and the fixture is a converged solve
# rebuilt on whatever machine runs the suite, so how many of them the UNGUARDED
# repair inverts is a property of THAT column: 11-12 on the column that
# motivated the guard (notes.md §1.13), 2-3 on the 0.15.0 EQ-seeded rebuild
# (the unguarded stage-1 correction exceeds the cell in 2 of the 12 band cells
# at dt 1e11, and those are the 2 the step inverts), 0 on the x86 runner of the
# oracle workflow. The non-vacuity pin below is therefore read only off a
# fixture built where it was measured AND on that machine (the arm64 fixture on
# an x86 host inverts no cell), and pins only that the unguarded repair
# inverts SOMETHING; what the guard itself must deliver -- no inverted cell --
# is pinned everywhere.
PINNED_MACHINE = "arm64"


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
# vulcan_jax import, then reuses this module's helper. It prints, per dt, how
# many band H2S cells the guarded and the unguarded repair flip negative.
_CHILD = r"""
import json, os, sys
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "tests"))
from test_stage_repair_guard import BAND, stage_arrays, step_solution
from vulcan_jax import jax_step

out = {}
for dt in (1e8, 1e11):
    y, k1, k2 = stage_arrays("adj_state_w39b.npz", "W39b", dt, 0.0)
    raw = step_solution(y, k1, k2)
    import vulcan_jax.chem_funs as cf
    h2s = cf.spec_list.index("H2S")
    counts = {}
    guarded = jax_step._REPAIR_MAX_CELL_FRAC
    for tag, frac in (("guarded", guarded), ("unguarded", float("inf"))):
        y, k1, k2 = stage_arrays("adj_state_w39b.npz", "W39b", dt, frac)
        sol = step_solution(y, k1, k2)
        flipped = (sol[BAND, h2s] < 0.0) & (raw[BAND, h2s] > 0.0)
        counts[tag] = int(flipped.sum())
    out[repr(dt)] = counts
print("COUNTS " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def w39b_band_counts():
    """{dt: {"guarded": n, "unguarded": n}} from one SNCHO child run."""
    res = run_child(_CHILD, network=S_NETWORK, label="stage-repair guard (W39b)")
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("COUNTS "))
    return {float(dt): c for dt, c in json.loads(line[len("COUNTS "):]).items()}


_NEEDS_W39B = pytest.mark.skipif(
    not W39B_FIXTURE.is_file(),
    reason="W39b fixture missing (npz artifacts are gitignored)",
)


@_NEEDS_W39B
def test_guard_keeps_the_w39b_trace_carrier_band_positive(w39b_band_counts):
    for dt, counts in w39b_band_counts.items():
        assert counts["guarded"] == 0, (dt, counts)


@_NEEDS_W39B
@pytest.mark.skipif(
    BUILT_ON != PINNED_MACHINE or platform.machine() != BUILT_ON,
    reason=(
        "non-vacuity NOT checked: the unguarded inversion count is pinned only "
        f"on a fixture built and run on {PINNED_MACHINE!r}; this fixture was "
        f"built on {BUILT_ON!r}, running on {platform.machine()!r}"
    ),
)
def test_unguarded_repair_inverts_the_w39b_band(w39b_band_counts):
    """Non-vacuity: without the guard the band inverts somewhere at dt 1e11."""
    assert w39b_band_counts[1e11]["unguarded"] >= 1, w39b_band_counts


@pytest.mark.skipif(
    not (DATA / "adj_state_hd189.npz").is_file(),
    reason="HD189 fixture missing (npz artifacts are gitignored)",
)
def test_guard_clamps_only_trace_carrier_cells_of_the_hd189_column():
    """At dt 1e6 the p99 of the per-cell correction ratio on this healthy
    column is 1.2e-5, under the 1.1e-2 of notes.md §1.13, and the guard
    clamps only the thermospheric exception that note names: H2O at 5923 K
    and 6000 K, where the carrier has dissociated (VMR 5e-18 and 3e-23) and
    the correction is 1e4 to 1e8 times the cell. Stage 1 is bit-identical to
    the unguarded repair on every other cell; stage 2 is not and carries no
    identity pin -- the two dropped stage-1 corrections reach ~9000 cells
    through the coupled solve."""
    import vulcan_jax.jax_step as jax_step
    args = ("adj_state_hd189.npz", "default", 1e6)
    y, g1, _ = stage_arrays(*args, jax_step._REPAIR_MAX_CELL_FRAC)
    _, u1, _ = stage_arrays(*args, float("inf"))
    _, raw1, _ = stage_arrays(*args, 0.0)
    ridx = np.asarray(jax_step._CHEM_RESERVOIR_IDX)

    # frac 0.0 drops every correction, so the unguarded stage minus it IS the
    # correction, and the guard's own denominator is max(cell, |raw stage|).
    ratio = np.abs(u1[:, ridx] - raw1[:, ridx]) / np.maximum(
        y[:, ridx], np.abs(raw1[:, ridx])
    )
    p99 = np.percentile(ratio, 99)
    assert p99 <= 1.1e-2, f"corrections grew on a healthy column: p99 {p99:.3e}"

    clamped = g1[:, ridx] != u1[:, ridx]
    vmr = (y / y.sum(axis=1, keepdims=True))[:, ridx]
    worst = float(np.max(vmr[clamped], initial=0.0))
    assert worst < 1e-10, (
        f"the guard clamped a real carrier: {int(clamped.sum())} cells, "
        f"worst carrier VMR {worst:.3e}"
    )
    fired = np.zeros_like(g1, dtype=bool)
    fired[:, ridx] = clamped
    assert np.array_equal(g1[~fired], u1[~fired]), (
        "the guard moved a stage-1 cell it did not clamp"
    )
    assert not np.array_equal(g1, raw1), "repair is inert here (vacuous test)"
