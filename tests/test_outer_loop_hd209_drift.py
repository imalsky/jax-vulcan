"""HD209 column-C drift regression test.

The README's "Step-count and atom-conservation drift" section flagged
HD209 as the open scientific-parity issue: master sits at +5e-5 column
C drift while JAX accumulates to a few percent over the ~1000-step
trajectory. Recent codegen-RHS work has brought the JAX number from
the historical -8.3% down to ~+3.0%; this test pins that bound so
future regressions trip immediately.

The test runs HD209 to convergence (1000+ steps) and asserts:
    atom_loss[C]  < 0.05    (5% — current ~3.0%, headroom for ULP wobble)
    atom_loss[H]  < 0.005   (0.5% — current ~2e-4)
    atom_loss[O]  < 0.005   (0.5% — current ~6e-4)
    atom_loss[N]  < 0.005   (0.5% — current ~3e-4)

Tightening any of these requires investigating the per-step ULP source.
The goal here is to flag a *regression* — a change that pushes any of
these significantly past the documented bound is a real bug.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


# Per-atom column-drift budget. Numbers based on a clean run with the
# canonical HD209 example cfg; bounds give ~50% headroom on the C drift
# (the dominant contributor) and ~25× on the others.
ATOM_LOSS_BUDGET = {
    "H": 0.005,
    "O": 0.005,
    "C": 0.05,
    "N": 0.005,
}


@pytest.mark.strict_isolation
def test_hd209_atom_loss_bounded():
    """Run HD209 to convergence; assert per-atom column drift stays
    inside the documented bounds. ~30s on this CPU host."""
    cfg_src = ROOT / "cfg_examples" / "vulcan_cfg_HD209.py"
    if not cfg_src.is_file():
        pytest.skip(f"HD209 example cfg missing at {cfg_src}.")

    os.chdir(ROOT)

    import importlib.util
    spec = importlib.util.spec_from_file_location("vulcan_cfg", cfg_src)
    cfg = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg)
    sys.modules["vulcan_cfg"] = cfg

    cfg.use_print_prog = False
    cfg.use_print_delta = False
    cfg.use_live_plot = False
    cfg.use_live_flux = False
    cfg.use_plot_end = False
    cfg.use_plot_evo = False

    import legacy_io as op
    import outer_loop
    op.vulcan_cfg = cfg
    outer_loop.vulcan_cfg = cfg

    import op_jax
    from atm_setup import Atm
    from state import RunState, legacy_view

    rs = RunState.with_pre_loop_setup(cfg)
    data_var, data_atm, data_para = legacy_view(rs)
    data_para.start_time = time.time()
    make_atm = Atm()
    output = op.Output()

    solver = op_jax.Ros2JAX()
    if cfg.use_photo and rs.photo_static is not None:
        solver._photo_static = rs.photo_static

    integ = outer_loop.OuterLoop(solver, output)
    solver.naming_solver(data_para)

    integ(data_var, data_atm, data_para, make_atm)

    print(
        f"HD209 convergence run: count={data_para.count}, "
        f"end_case={data_para.end_case}, t={data_var.t:.3e}"
    )
    print("  atom_loss: " + " ".join(
        f"{a}={v:.3e}" for a, v in data_var.atom_loss.items()
    ))

    failures: list[str] = []
    for atom, budget in ATOM_LOSS_BUDGET.items():
        if atom not in data_var.atom_loss:
            continue
        loss = data_var.atom_loss[atom]
        if loss > budget:
            failures.append(
                f"atom_loss[{atom}] = {loss:.3e} exceeds budget {budget:.3e}"
            )

    assert not failures, "HD209 atom-loss regression:\n  " + "\n  ".join(failures)
