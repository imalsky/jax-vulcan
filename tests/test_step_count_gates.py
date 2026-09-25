"""The CLI step-count gates: HD189 1495, HD209 1211, W39b 1202.

A physics-neutral change keeps these counts exactly (CLAUDE.md parity rule
3, notes §0). HD189_vulcan3 is left out: its 2603 is an arm64 count, and
x86-64 ends on the hybrid phase-1 budget at 4018 (notes §1.13). Each run is
a full integration (~2 min on one core), so the file is slow-gated; oracle.yml
runs it in the release gate.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import pytest

from vulcan_jax.config import load_config

_GATES = {"HD189": 1495, "HD209": 1211, "W39b": 1202}
# The CLI selects each config's import-frozen network by re-executing itself;
# an inherited selection would run the config on the wrong network.
_FROZEN_ENV = ("VULCAN_JAX_NETWORK", "VULCAN_JAX_ATOM_LIST", "VULCAN_JAX_CLI_RELAUNCHED")


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("VULCAN_JAX_RUN_SLOW") != "1",
    reason="three full CLI integrations; set VULCAN_JAX_RUN_SLOW=1",
)
@pytest.mark.parametrize("name, count", list(_GATES.items()))
def test_cli_step_count_gate(name, count, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in _FROZEN_ENV}
    res = subprocess.run(
        [sys.executable, "-m", "vulcan_jax.vulcan_jax_cli", "--config", name],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=3600, check=False,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    cfg = load_config(name)
    with open(tmp_path / cfg.output_dir / cfg.out_name, "rb") as fh:
        par = pickle.load(fh)["parameter"]
    assert int(par["count"]) == count, (name, int(par["count"]), par["end_case"])
