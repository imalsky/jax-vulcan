"""W39b (SNCHO photo-on, nr=1150) reverse-mode adjoint regression.

Runs in a subprocess because the SNCHO network must be import-locked via
`$VULCAN_JAX_NETWORK` before the first `import vulcan_jax` (the pytest worker
already holds the default network). Skips unless `VULCAN_JAX_RUN_SLOW=1` and
the local fixture `tests/data/adj_state_w39b.npz` exists (npz artifacts are
gitignored; produce it with `jax_paper/scripts/adj_save_state_w39b.py`).

Pins the 2026-07-02 hardening-battery results: the OH+H2 <-> H2O+H pair leads
the SO2 ranking with g1 = -0.682 (dt-insensitive to <1% across body_dt
3e6..1e8 on this column), twin spread ~6e-4, residuals <= 0.05, exact-null
conservation directions (null_quality ~1e-10), and healthy pair antisymmetry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "data" / "adj_state_w39b.npz"
_RUN_SLOW = os.environ.get("VULCAN_JAX_RUN_SLOW") == "1"

_CHILD = r"""
import os, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import vulcan_jax.chem_funs as chem_funs
from vulcan_jax.jax_step import AtmStatic
from vulcan_jax.steady_state_grad import steady_state_reaction_sensitivity

d = np.load(os.environ["W39B_FIXTURE"], allow_pickle=True)
nz, ni, so2 = int(d["nz"]), int(d["ni"]), int(d["so2"])
y_star = jnp.asarray(d["y_star"]); k_arr = jnp.asarray(d["k_arr"])
dz = jnp.asarray(d["dz"]); compo = jnp.asarray(d["compo"])
fields = {k[5:]: jnp.asarray(d[k]) for k in d.files if k.startswith("atm__")}
# The fixture predates the interface-centered vm; inert (use_vm_mol=False),
# splice the current-contract shape.
fields["vm"] = jnp.zeros((nz - 1, ni))
# It also predates `diff_esc_mask`; W39b ships `diff_esc: []`, so all-False.
fields.setdefault("diff_esc_mask", jnp.zeros(ni, dtype=jnp.bool_))
atm = AtmStatic(
    **fields, **{k[9:]: bool(d[k]) for k in d.files if k.startswith("atmbool__")}
)
net = chem_funs._NET_JAX
ys = np.asarray(y_star)
Lz = int(np.argmax(ys[:, so2] / ys.sum(axis=1)))

def loss(y):  # log10 SO2 VMR at the peak-SO2 layer (the paper's loss)
    return jnp.log10(y[Lz, so2] / jnp.sum(y[Lz]))

# This fixture is a saved FROZEN-photolysis state (no runner attached), so it
# cannot exercise the photochemistry-on default path (renorm + photo_recompute_k
# for the dJ/dy feedback). On a photo-on column without that feedback the
# renormalized map is internally inconsistent (pair_antisym ~1), which the
# diagnostics correctly flag -- so this regression pins the legacy
# solver_map="bare" behavior instead. The percent-level DEFAULT path
# (renorm + dJ/dy) is validated end-to-end through the runner in
# jax_paper/scripts/fd_validate_w39b_reverse.py (r1 0.17%, r691 0.07%).
g, info = steady_state_reaction_sensitivity(
    loss, y_star, k_arr, atm, net, compo_array=compo, dz=dz,
    solver_map="bare",
    lgmres_inner_m=60, lgmres_cycles=10, return_info=True,
)
g = np.asarray(g)
top = np.argsort(np.abs(g))[::-1][:4]
print("RESULT " + json.dumps({
    "g1": float(g[1]), "g2": float(g[2]), "g691": float(g[691]),
    "top4": [int(t) for t in top],
    "resids": [float(r) for r in info["resids"]],
    "spread": float(info["ensemble_spread"]),
    "pair_antisym": float(info["pair_antisym"]),
    "fp_err": float(info["fp_err"]),
    "null_quality": float(info["null_quality"]),
}))
"""


@pytest.mark.skipif(
    not _RUN_SLOW, reason="slow W39b adjoint regression; set VULCAN_JAX_RUN_SLOW=1"
)
@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="local W39b fixture missing (npz artifacts are gitignored)",
)
def test_w39b_reaction_sensitivity_regression():
    env = dict(os.environ)
    env["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"
    env["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"
    env["W39B_FIXTURE"] = str(FIXTURE)
    env.setdefault("OMP_NUM_THREADS", "1")
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    r = json.loads(line[len("RESULT ") :])

    # Ranking: the OH+H2 <-> H2O+H pair leads; SO+OH -> SO2+H in the top 4.
    assert set(r["top4"][:2]) == {1, 2}, r["top4"]
    assert 691 in r["top4"], r["top4"]
    # Values (2026-07-02 battery: g1 = -0.68209, dt-insensitive <1%).
    assert r["g1"] < 0 and r["g2"] > 0 and r["g691"] > 0
    assert abs(r["g1"] + 0.682) / 0.682 < 0.05, r["g1"]
    # Diagnostics (measured: resids <= 0.016 at the default budget, spread
    # 5.8e-4, pair_antisym <= 0.08, null_quality ~1e-10).
    assert sorted(r["resids"])[len(r["resids"]) // 2] < 0.1, r["resids"]
    assert r["spread"] < 0.01, r["spread"]
    assert r["pair_antisym"] < 0.2, r["pair_antisym"]
    assert r["fp_err"] < 1e-2
    assert r["null_quality"] < 1e-6
