"""make_config() overrides reach setup, the runner and the output writer
without leaking onto the global config; import-locked overrides fail fast.
Also covers RunState reuse and the clip on a degenerate layer."""

from __future__ import annotations

import pytest


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


def test_one_outer_loop_runs_fresh_runstates_each_on_its_own_atoms():
    """`integ(rs)` rebuilds the certificate ring, the element budget and the
    hybrid phase from zero, none of which a RunState carries, so handing it
    an integrated RunState must fail instead of resuming on a blank history.
    A second fresh RunState is fine, and its result reports its own atoms,
    not those of the profile the runner was first built for."""
    import numpy as np

    import vulcan_jax
    import vulcan_jax.legacy_io as op
    from vulcan_jax import op_jax, outer_loop
    from vulcan_jax.state import RunState

    cfg = vulcan_jax.make_config(
        nz=12, P_b=1e5, P_t=1e-2, count_max=5, use_photo=False,
        use_print_prog=False, ini_mix="const_mix",
        const_mix={"H2": 0.9, "He": 0.09, "H2O": 5e-3, "CO": 5e-3},
    )
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
    rs1 = RunState.with_pre_loop_setup(cfg)
    rs_out = integ(rs1)
    assert int(rs_out.params.count) == 6
    with pytest.raises(ValueError, match="already run 6 steps"):
        integ(rs_out)

    cfg.const_mix = {"H2": 0.9, "He": 0.09, "H2O": 1e-3, "CO": 9e-3}
    rs2 = RunState.with_pre_loop_setup(cfg)
    assert not np.array_equal(rs2.atoms.atom_ini, rs1.atoms.atom_ini)
    out2 = integ(rs2)
    np.testing.assert_array_equal(out2.atoms.atom_ini, rs2.atoms.atom_ini)
    np.testing.assert_array_equal(
        out2.atoms.atom_sum,
        (np.asarray(out2.atoms.atom_loss) + 1.0) * np.asarray(rs2.atoms.atom_ini),
    )


# Import-frozen knobs (network, atom_list, com_file): a conflicting
# make_config value must raise at setup with a clear message, not a
# downstream k_arr shape error. The reordered and renamed networks show that
# equal species and nr counts are not sufficient.
def _reordered_network(tmp_path):
    import re
    from vulcan_jax._paths import PACKAGE_ROOT
    src = (PACKAGE_ROOT / "thermo"
           / "NCHO_photo_network.txt").read_text().splitlines()
    rxn = [i for i, ln in enumerate(src) if re.match(r"^\s*\d+\s*\[", ln)]
    src[rxn[0]], src[rxn[1]] = src[rxn[1]], src[rxn[0]]
    net = tmp_path / "NCHO_reordered.txt"
    net.write_text("\n".join(src) + "\n")
    return net


def _renamed_species_network(tmp_path):
    import re
    from vulcan_jax._paths import PACKAGE_ROOT
    text = (PACKAGE_ROOT / "thermo" / "NCHO_photo_network.txt").read_text()
    renamed = re.sub(r"(?<![A-Za-z0-9_])H(?![A-Za-z0-9_])", "X", text)
    net = tmp_path / "NCHO_H_renamed.txt"
    net.write_text(renamed)
    return net


@pytest.mark.strict_isolation
@pytest.mark.parametrize("overrides, pattern", [
    pytest.param(dict(network="thermo/SNCHO_photo_network.txt"), "import-locked",
                 id="alternate"),
    pytest.param(dict(network="thermo/NCHO_thermo_network.txt"), "import-locked",
                 id="same_species_fewer_nr"),
    pytest.param(_reordered_network, "topology|import-locked", id="reordered"),
    pytest.param(_renamed_species_network, "import-locked", id="renamed_species"),
    pytest.param(dict(atom_list=["H", "O", "C"]), "atom_list", id="atom_list"),
    pytest.param(dict(com_file="thermo/DOES_NOT_EXIST.txt"), "com_file",
                 id="com_file"),
])
def test_import_locked_overrides_fail_fast(overrides, pattern, tmp_path):
    import re

    import vulcan_jax
    from vulcan_jax.state import RunState

    if callable(overrides):
        overrides = dict(network=str(overrides(tmp_path)))
    cfg = vulcan_jax.make_config(**overrides)
    with pytest.raises(ValueError) as excinfo:
        RunState.with_pre_loop_setup(cfg)
    msg = str(excinfo.value)
    assert re.search(pattern, msg), msg
    assert "k_arr" not in msg      # never the cryptic downstream symptom


def test_save_cfg_serializes_active_cfg(tmp_path, monkeypatch):
    """Output(cfg=cfg).save_cfg writes the ACTIVE cfg's values (so make_config
    runs are reproducible), not a copy of the packaged vulcan_cfg.py, and it
    writes under dname even when the cwd is somewhere else."""
    import vulcan_jax
    from vulcan_jax import legacy_io

    cfg = vulcan_jax.make_config(out_name="custom.vul", count_max=7)
    work = tmp_path / "cwd"
    dest = tmp_path / "dest"
    work.mkdir()
    dest.mkdir()
    monkeypatch.chdir(work)  # cwd != dname
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
    # No ymix argument: master's second clip rule reads the post-solve ymix,
    # which reduces to "zero every negative" (see _clip_prologue).
    for non_gas_present in (False, True):
        clip = _make_clip_fn(
            non_gas_present, gas_mask, pos_cut=1e-20, nega_cut=-1e-20
        )
        _y_clip, ymix_new, _s, _n = clip(y)
        assert bool(jnp.all(jnp.isfinite(ymix_new))), (
            f"clip produced non-finite ymix (non_gas_present={non_gas_present})"
        )
