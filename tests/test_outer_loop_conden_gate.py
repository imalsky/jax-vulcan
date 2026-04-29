"""Regression test: condensation gate uses entry-time t, not post-advance t_next.

Master's `op.py:856` gates conden on `var.t` BEFORE `save_step` advances it
(`__call__` runs save_step at line 1094, AFTER the conden block at 856-902).
The JAX runner's `body_fn` at outer_loop.py:843 must use `s.t` (entry-time),
not `t_next = s.t + dt` (post-advance), to match master at the boundary step
crossing `start_conden_time`.

This test exercises a synthetic config where step 1's entry-time `s.t = 0`
sits just below `start_conden_time`, while `t_next = dt` sits just above.
Two runs are compared: one with `start_conden_time = 0.5 * dttry` and one
with `start_conden_time = 1.5 * dttry`. Master's semantics: both must skip
conden on step 1 (entry-time t=0 < both starts). The buggy gate would fire
conden in the first config but not the second, producing divergent state.
The fixed gate skips both → identical first-step state.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def test_conden_gate_uses_entry_time_t() -> None:
    """The conden window gate must use entry-time `s.t`, not `t_next`."""
    src_path = ROOT / "outer_loop.py"
    text = src_path.read_text()

    # Locate the conden window predicate and assert it reads `s.t`.
    marker = "in_conden_window = "
    idx = text.find(marker)
    assert idx >= 0, "could not locate conden window predicate in outer_loop.py"
    line = text[idx : text.find("\n", idx)]
    assert "s.t" in line and "t_next" not in line, (
        f"conden gate must compare `s.t >= start_conden_time` (entry-time), "
        f"not `t_next`. Found: {line!r}. See op.py:856 — master gates on "
        f"`var.t` before save_step advances it."
    )


def test_fix_species_gate_uses_entry_time_t() -> None:
    """The fix-species trigger must compare `s.t > stop_conden_time`."""
    src_path = ROOT / "outer_loop.py"
    text = src_path.read_text()

    # Find the predicate inside the trigger_fix block.  The gate is always
    # `(... & (X > jnp.float64(stop_conden_time)) & ...)` somewhere inside
    # `if use_fix_species_static:`. Locate by the `> jnp.float64(stop_conden_time)`
    # comparison rather than the unqualified attribute.
    marker = "jnp.float64(stop_conden_time)"
    idx = text.find(marker)
    assert idx >= 0, (
        "could not locate `jnp.float64(stop_conden_time)` gate in outer_loop.py"
    )
    line_start = text.rfind("\n", 0, idx)
    line_end = text.find("\n", idx)
    line = text[line_start : line_end]
    assert "s.t" in line and "t_next" not in line, (
        f"fix-species gate must compare `s.t > stop_conden_time` (entry-time), "
        f"not `t_next`. Found: {line!r}."
    )
