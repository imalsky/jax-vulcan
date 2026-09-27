"""The condensation and fix-species gates read entry-time s.t, not t_next.

Master gates condensation on var.t before save_step advances it (op.py:856;
save_step at op.py:918), so the step that crosses start_conden_time does
not condense. Checked on the source text of outer_loop.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "marker", ["in_conden_window = ", "jnp.float64(stop_conden_time)"]
)
def test_conden_gates_use_entry_time_t(marker) -> None:
    """The conden window (`s.t >= start_conden_time`) and the fix-species
    trigger (`s.t > stop_conden_time`) must compare entry-time `s.t`."""
    text = (ROOT / "src" / "vulcan_jax" / "outer_loop.py").read_text()
    idx = text.find(marker)
    assert idx >= 0, f"could not locate {marker!r} in outer_loop.py"
    line = text[text.rfind("\n", 0, idx) : text.find("\n", idx)]
    assert "s.t" in line and "t_next" not in line, (
        f"gate must compare entry-time `s.t`, not `t_next`. Found: {line!r}. "
        f"See op.py:856: master gates on `var.t` before save_step advances it."
    )
