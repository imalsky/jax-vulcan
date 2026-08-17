"""make_config() is a real public API: overrides reach setup AND the runner.

With a make_config() namespace distinct from the global, this file pins:
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

    # The process default config is what state._cfg_overlay identity-compares
    # against for its no-op fast path.
    from vulcan_jax.config import default_config

    vulcan_cfg = default_config()

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




def test_output_reads_cfg_not_global():
    """legacy_io.Output(cfg=cfg) honors the cfg (paths + progress-print caps),
    instead of always reading the global vulcan_cfg module."""
    import vulcan_jax
    from vulcan_jax import legacy_io

    g = vulcan_jax.default_config()
    cfg = vulcan_jax.make_config(count_max=7)
    assert int(legacy_io.Output(cfg=cfg)._cfg.count_max) == 7
    assert int(legacy_io.Output()._cfg.count_max) == int(g.count_max)




# Import-locked knob refusals: network/atom_list/com_file are frozen at the
# first `import vulcan_jax`, so a conflicting make_config value must fail
# FAST with a clear message ($VULCAN_JAX_NETWORK / "import-locked"), never
# the cryptic k_arr shape error 30 s into setup. One case per conflict class;
# the two synthetic networks prove species+nr equality is NOT sufficient
# (the codegen RHS and k_arr indexing are order- and identity-specific).
def _case_alternate(tmp_path):
    return dict(network="thermo/SNCHO_photo_network.txt"), "import-locked"


def _case_same_species_fewer_nr(tmp_path):
    return dict(network="thermo/NCHO_thermo_network.txt"), "import-locked"


def _case_reordered(tmp_path):
    import re
    from vulcan_jax._paths import PACKAGE_ROOT
    src = (PACKAGE_ROOT / "thermo"
           / "NCHO_photo_network.txt").read_text().splitlines()
    rxn = [i for i, ln in enumerate(src) if re.match(r"^\s*\d+\s*\[", ln)]
    src[rxn[0]], src[rxn[1]] = src[rxn[1]], src[rxn[0]]
    net = tmp_path / "NCHO_reordered.txt"
    net.write_text("\n".join(src) + "\n")
    return dict(network=str(net)), "topology|import-locked"


def _case_renamed_species(tmp_path):
    import re
    from vulcan_jax._paths import PACKAGE_ROOT
    text = (PACKAGE_ROOT / "thermo" / "NCHO_photo_network.txt").read_text()
    renamed = re.sub(r"(?<![A-Za-z0-9_])H(?![A-Za-z0-9_])", "X", text)
    net = tmp_path / "NCHO_H_renamed.txt"
    net.write_text(renamed)
    return dict(network=str(net)), "import-locked"


def _case_atom_list(tmp_path):
    return dict(atom_list=["H", "O", "C"]), "atom_list"


def _case_com_file(tmp_path):
    return dict(com_file="thermo/DOES_NOT_EXIST.txt"), "com_file"


@pytest.mark.strict_isolation
@pytest.mark.parametrize("case", [
    _case_alternate, _case_same_species_fewer_nr, _case_reordered,
    _case_renamed_species, _case_atom_list, _case_com_file,
], ids=lambda c: c.__name__[6:])
def test_import_locked_overrides_fail_fast(case, tmp_path):
    import re

    import vulcan_jax
    from vulcan_jax.state import RunState

    overrides, pattern = case(tmp_path)
    cfg = vulcan_jax.make_config(**overrides)
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)
    msg = str(excinfo.value)
    assert re.search(pattern, msg), msg
    assert "k_arr" not in msg      # never the cryptic downstream symptom




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
    # No ymix argument: master's second clip rule tests the POST-solve ymix,
    # which reduces to "zero every negative" (see _clip_prologue), so the
    # closure never needed the previous step's mixing ratios.
    for non_gas_present in (False, True):
        clip = _make_clip_fn(
            non_gas_present, gas_mask, pos_cut=1e-20, nega_cut=-1e-20
        )
        _y_clip, ymix_new, _s, _n = clip(y)
        assert bool(jnp.all(jnp.isfinite(ymix_new))), (
            f"clip produced non-finite ymix (non_gas_present={non_gas_present})"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
