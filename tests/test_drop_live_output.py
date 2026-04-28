"""Phase 19: live-output flags are rejected by the runtime validator.

`use_live_plot`, `use_live_flux`, `use_save_movie`, and `use_flux_movie`
were dropped along with the matplotlib live-window / movie code path
in `legacy_io.Output` and `outer_loop._run_chunked`. The validator must
loudly reject any cfg that still sets one of these flags to True so that
silent regressions to a "running but no output" state don't happen.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _minimal_valid_cfg(**overrides):
    """Build a SimpleNamespace cfg that passes the non-live-output
    validations so the test only exercises the live-output rejection.
    """
    cfg = SimpleNamespace(
        ode_solver="Ros2",
        use_photo=False,
        use_ion=False,
        fix_species=[],
        use_condense=False,
        use_fix_H2He=False,
        use_topflux=False,
        use_botflux=False,
        network="thermo/SNCHO_photo_network_2025.txt",
        gibbs_text="thermo/gibbs_text.txt",
        com_file="thermo/all_compose.txt",
        atm_file="atm/atm_HD189_Kzz.txt",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize(
    "flag",
    ["use_live_plot", "use_live_flux", "use_save_movie", "use_flux_movie"],
)
def test_validator_rejects_dropped_live_flag(flag):
    """Each dropped live-output flag, when True, raises a clear error."""
    from runtime_validation import validate_runtime_config

    cfg = _minimal_valid_cfg(**{flag: True})
    with pytest.raises(RuntimeError) as excinfo:
        validate_runtime_config(cfg, root=ROOT)
    msg = str(excinfo.value)
    assert flag in msg, f"Validator error did not mention {flag!r}: {msg}"
    assert "no longer supported" in msg, (
        f"Validator error did not flag the drop reason: {msg}"
    )


def test_validator_accepts_live_flags_off():
    """All live-output flags False — validator passes."""
    from runtime_validation import validate_runtime_config

    cfg = _minimal_valid_cfg(
        use_live_plot=False,
        use_live_flux=False,
        use_save_movie=False,
        use_flux_movie=False,
    )
    # No exception expected.
    validate_runtime_config(cfg, root=ROOT)


def test_output_class_has_no_live_methods():
    """`legacy_io.Output` must not expose plot_update / plot_flux_update /
    plot_evo_inter — they were removed in Phase 19."""
    import legacy_io

    out = legacy_io.Output()
    for dropped in ("plot_update", "plot_flux_update", "plot_evo_inter"):
        assert not hasattr(out, dropped), (
            f"legacy_io.Output still exposes {dropped!r}; Phase 19 dropped it."
        )

    # The kept post-run plotters are still present.
    for kept in ("plot_end", "plot_evo", "plot_TP", "save_out", "save_cfg",
                 "print_prog", "print_end_msg", "print_unconverged_msg"):
        assert hasattr(out, kept), (
            f"legacy_io.Output is missing {kept!r}; Phase 19 should have "
            f"kept it."
        )


def test_parameters_has_no_live_plot_state():
    """`store.Parameters` no longer has `pic_count` or `tableau20`
    (only used by deleted live-plot methods)."""
    import store

    para = store.Parameters()
    assert not hasattr(para, "pic_count"), (
        "store.Parameters.pic_count was kept around — Phase 19 dropped it "
        "with the live-plot surface."
    )
    assert not hasattr(para, "tableau20"), (
        "store.Parameters.tableau20 was kept around — Phase 19 dropped it "
        "with the live-plot surface."
    )
