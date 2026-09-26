"""The condensation and fix-species gates read entry-time s.t, not t_next.

Master gates condensation on var.t before save_step advances it (op.py:856;
save_step at op.py:918), so the step that crosses start_conden_time does
not condense. Checked on the source text of outer_loop.py.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


def test_conden_gate_uses_entry_time_t() -> None:
    """The conden window gate must use entry-time `s.t`, not `t_next`."""
    src_path = ROOT / "src" / "vulcan_jax" / "outer_loop.py"
    text = src_path.read_text()

    # Locate the conden window predicate and assert it reads `s.t`.
    marker = "in_conden_window = "
    idx = text.find(marker)
    assert idx >= 0, "could not locate conden window predicate in outer_loop.py"
    line = text[idx : text.find("\n", idx)]
    assert "s.t" in line and "t_next" not in line, (
        f"conden gate must compare `s.t >= start_conden_time` (entry-time), "
        f"not `t_next`. Found: {line!r}. See op.py:856: master gates on "
        f"`var.t` before save_step advances it."
    )


def test_fix_species_gate_uses_entry_time_t() -> None:
    """The fix-species trigger must compare `s.t > stop_conden_time`."""
    src_path = ROOT / "src" / "vulcan_jax" / "outer_loop.py"
    text = src_path.read_text()

    # Locate the `> jnp.float64(stop_conden_time)` comparison in the trigger_fix block.
    marker = "jnp.float64(stop_conden_time)"
    idx = text.find(marker)
    assert idx >= 0, (
        "could not locate `jnp.float64(stop_conden_time)` gate in outer_loop.py"
    )
    line_start = text.rfind("\n", 0, idx)
    line_end = text.find("\n", idx)
    line = text[line_start:line_end]
    assert "s.t" in line and "t_next" not in line, (
        f"fix-species gate must compare `s.t > stop_conden_time` (entry-time), "
        f"not `t_next`. Found: {line!r}."
    )
