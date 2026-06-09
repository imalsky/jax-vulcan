"""make_config() is a real public API: overrides reach setup AND the runner.

Regression coverage for the config-isolation fix. Before it, make_config()
returned a separate namespace whose overrides were silently ignored — setup
read the global vulcan_cfg module and OuterLoop took no cfg at all, so the
advertised `cfg = make_config(...); rs = with_pre_loop_setup(cfg)` flow built a
default atmosphere and ran with default solver knobs. No prior test exercised a
separate-namespace cfg (vmap / host-hook tests all pin cfg == the global
module), so the regression was invisible.

This file asserts, with a make_config() namespace distinct from the global:
    1. grid overrides (nz / P_b / P_t) reach the pre-loop setup,
    2. runner overrides (count_max) reach OuterLoop's static config,
    3. nothing leaks onto the global vulcan_cfg module afterward,
    4. an import-locked-network override fails fast with a clear message,
    5. the clip normalization stays finite on a degenerate all-zero layer.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

warnings.filterwarnings("ignore")


@pytest.mark.strict_isolation
def test_grid_and_runner_overrides_reach_setup_and_runner_without_leakage():
    import numpy as np

    import vulcan_jax
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    from vulcan_jax import outer_loop
    from vulcan_jax.state import RunState, legacy_view

    # Pin the canonical module reference (mirrors test_outer_loop_smoke) so the
    # `cfg is vulcan_cfg` identity check in state._cfg_overlay is meaningful.
    vulcan_cfg = op.vulcan_cfg
    sys.modules["vulcan_jax.vulcan_cfg"] = vulcan_cfg

    global_nz_before = vulcan_cfg.nz
    global_pb_before = vulcan_cfg.P_b
    global_cmax_before = int(vulcan_cfg.count_max)

    # A namespace distinct from the global module, with non-default grid + runner knobs.
    cfg = vulcan_jax.make_config(
        nz=12, P_b=1e5, P_t=1e-2, count_max=37, use_photo=False
    )
    assert cfg is not vulcan_cfg

    rs = RunState.with_pre_loop_setup(cfg)
    pco = np.asarray(rs.atm.pco)

    # (1) grid overrides reached setup.
    assert pco.shape == (12,), f"nz override ignored: pco shape {pco.shape}"
    assert pco[0] == pytest.approx(1e5, rel=1e-9), f"P_b override ignored: {pco[0]}"
    assert pco[-1] == pytest.approx(1e-2, rel=1e-9), f"P_t override ignored: {pco[-1]}"

    # (2) runner override reaches OuterLoop's static config.
    var, atm, _para = legacy_view(rs)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(), cfg=cfg)
    assert integ._cfg is cfg
    statics = integ._build_statics(var, atm)
    assert int(statics.count_max) == 37, (
        f"count_max override did not reach the runner: {int(statics.count_max)}"
    )

    # (3) zero leakage: the global module is exactly as it was before the call.
    assert vulcan_cfg.nz == global_nz_before, "nz leaked onto the global module"
    assert vulcan_cfg.P_b == global_pb_before, "P_b leaked onto the global module"
    assert int(vulcan_cfg.count_max) == global_cmax_before, "count_max leaked"

    # And a default OuterLoop (no cfg) still resolves to the global module,
    # whose count_max is the shipped default, not the make_config override.
    default_integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output())
    assert default_integ._cfg is vulcan_cfg
    assert int(default_integ._cfg.count_max) != 37


@pytest.mark.strict_isolation
def test_alternate_network_fails_fast_with_clear_message():
    """Network is import-locked; make_config(network=...) must fail fast with a
    clear $VULCAN_JAX_NETWORK message, not a cryptic shape error 30s into setup."""
    import vulcan_jax
    from vulcan_jax.state import RunState

    cfg = vulcan_jax.make_config(network="thermo/SNCHO_photo_network.txt")
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)

    msg = str(excinfo.value)
    assert "VULCAN_JAX_NETWORK" in msg
    assert "import-locked" in msg
    # The cryptic downstream symptom must not be what the user sees.
    assert "k_arr" not in msg


@pytest.mark.strict_isolation
def test_same_species_different_nr_network_blocked():
    """A *_thermo network has the same species as the import-locked *_photo one
    but fewer reactions (nr 782 vs 878). The guard must still fail fast on nr,
    not let it slip through to the cryptic k_arr shape error."""
    import vulcan_jax
    from vulcan_jax.state import RunState

    cfg = vulcan_jax.make_config(network="thermo/NCHO_thermo_network.txt")
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)
    msg = str(excinfo.value)
    assert "import-locked" in msg and "reactions" in msg
    assert "k_arr" not in msg


@pytest.mark.strict_isolation
def test_missing_com_file_fails_fast():
    """make_config(com_file=<missing>) must raise, not silently fall back to the
    composition table already loaded at import."""
    import vulcan_jax
    from vulcan_jax.state import RunState

    cfg = vulcan_jax.make_config(com_file="thermo/DOES_NOT_EXIST.txt")
    with pytest.raises(ValueError, match="com_file"):
        RunState.with_pre_loop_setup(cfg)


def test_output_reads_cfg_not_global():
    """legacy_io.Output(cfg=cfg) honors the cfg (paths + progress-print caps),
    instead of always reading the global vulcan_cfg module."""
    import vulcan_jax
    import vulcan_jax.vulcan_cfg as g
    from vulcan_jax import legacy_io

    cfg = vulcan_jax.make_config(count_max=7)
    assert int(legacy_io.Output(cfg=cfg)._cfg.count_max) == 7
    assert int(legacy_io.Output()._cfg.count_max) == int(g.count_max)


@pytest.mark.strict_isolation
def test_reordered_network_blocked(tmp_path):
    """A network with the SAME species and reaction count but reordered
    reactions must be blocked: the import-time codegen RHS and k_arr indexing
    are order-specific, so species + nr equality is not sufficient."""
    import re

    import vulcan_jax
    from vulcan_jax._paths import PACKAGE_ROOT
    from vulcan_jax.state import RunState

    src = (PACKAGE_ROOT / "thermo" / "NCHO_photo_network.txt").read_text().splitlines()
    rxn = [i for i, ln in enumerate(src) if re.match(r"^\s*\d+\s*\[", ln)]
    assert len(rxn) >= 2
    a, b = rxn[0], rxn[1]
    src[a], src[b] = src[b], src[a]  # same species/nr, different topology
    swapped = tmp_path / "NCHO_reordered.txt"
    swapped.write_text("\n".join(src) + "\n")

    cfg = vulcan_jax.make_config(network=str(swapped))
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)
    msg = str(excinfo.value)
    assert "topology" in msg or "import-locked" in msg
    assert "k_arr" not in msg


@pytest.mark.strict_isolation
def test_renamed_species_network_blocked(tmp_path):
    """A network with the SAME numeric topology but a renamed species (H->X)
    must be blocked. The topology signature is index-based, so species identity
    has to be compared too — otherwise cfg-time species metadata would pair with
    the import-time codegen/species state."""
    import re

    import vulcan_jax
    from vulcan_jax._paths import PACKAGE_ROOT
    from vulcan_jax.state import RunState

    text = (PACKAGE_ROOT / "thermo" / "NCHO_photo_network.txt").read_text()
    renamed = re.sub(r"(?<![A-Za-z0-9_])H(?![A-Za-z0-9_])", "X", text)  # H token -> X
    assert renamed != text
    net = tmp_path / "NCHO_H_renamed.txt"
    net.write_text(renamed)

    cfg = vulcan_jax.make_config(network=str(net))
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)
    assert "import-locked" in str(excinfo.value)
    assert "k_arr" not in str(excinfo.value)


@pytest.mark.strict_isolation
def test_atom_list_import_frozen():
    """make_config(atom_list=...) different from the import-time value must fail
    fast — the reservoir-projection tables are baked at import."""
    import vulcan_jax
    from vulcan_jax.state import RunState

    cfg = vulcan_jax.make_config(atom_list=["H", "O", "C"])
    with pytest.raises(ValueError, match="atom_list"):
        RunState.with_pre_loop_setup(cfg)


def test_save_cfg_serializes_active_cfg(tmp_path, monkeypatch):
    """Output(cfg=cfg).save_cfg writes the ACTIVE cfg's values (so make_config
    runs are reproducible), not a copy of the packaged vulcan_cfg.py — and it
    writes under dname even when the cwd is somewhere else."""
    import vulcan_jax
    from vulcan_jax import legacy_io

    cfg = vulcan_jax.make_config(out_name="custom.vul", count_max=7)
    work = tmp_path / "cwd"
    dest = tmp_path / "dest"
    work.mkdir()
    dest.mkdir()
    monkeypatch.chdir(work)  # cwd deliberately != dname
    legacy_io.Output(cfg=cfg).save_cfg(str(dest))
    content = (dest / cfg.output_dir / "cfg_custom.txt").read_text()
    assert "count_max = 7" in content
    assert "custom.vul" in content


def test_clip_fn_degenerate_layer_stays_finite():
    """An all-zero (or all-negative) gas layer must not produce NaN/inf ymix in
    either the gas-mask or no-mask clip branch."""
    import jax.numpy as jnp

    from vulcan_jax.outer_loop import _make_clip_fn

    # 3 species; in the gas-mask branch the last column is a non-gas condensate.
    gas_mask = jnp.array([True, True, False])
    y = jnp.array(
        [
            [1e10, 2e10, 5e9],  # physical layer
            [1e-30, -1e-30, 3e9],  # gas clips to 0, condensate nonzero (inf risk)
            [-1e-25, -1e-25, -1e-25],  # all negative -> all clip to 0 (0/0 risk)
        ]
    )
    ymix_old = jnp.array([[0.4, 0.5, 0.1], [0.0, 0.0, 1.0], [0.3, 0.3, 0.4]])

    for non_gas_present in (False, True):
        clip = _make_clip_fn(
            non_gas_present, gas_mask, mtol=1e-20, pos_cut=1e-20, nega_cut=-1e-20
        )
        _y_clip, ymix_new, _s, _n = clip(y, ymix_old)
        assert bool(jnp.all(jnp.isfinite(ymix_new))), (
            f"clip produced non-finite ymix (non_gas_present={non_gas_present})"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
