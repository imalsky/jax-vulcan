"""Live-output flags are accepted and routed through the chunked runner.

`use_live_plot`, `use_live_flux`, `use_save_movie`, and `use_flux_movie`
each enable a host-side hook in `live_ui.py` that fires between JIT'd
step batches at `vulcan_cfg.live_plot_frq` cadence. The validator
accepts every combination; live-flux additionally requires use_photo.

These tests exercise the host-side wiring without launching the full
integration loop — `update_mix` and `update_flux` are pure host code and
can be driven against a hand-built `(var, atm, para)` shim.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# Force a non-interactive backend before any matplotlib import — the
# tests run headless in CI and we never want a window pop-up.
os.environ.setdefault("MPLBACKEND", "Agg")


def _minimal_valid_cfg(**overrides):
    """Build a SimpleNamespace cfg that passes the non-live validations."""
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
    ["use_live_plot", "use_save_movie", "use_flux_movie"],
)
def test_validator_accepts_live_flag(flag):
    """The validator no longer rejects live-output flags."""
    from runtime_validation import validate_runtime_config
    cfg = _minimal_valid_cfg(**{flag: True})
    # use_flux_movie is fine without photo on the validator side; it's
    # the dispatcher that gates flux on use_photo.
    validate_runtime_config(cfg, root=ROOT)


def test_validator_requires_photo_for_live_flux():
    """`use_live_flux=True` without `use_photo=True` is rejected — there
    are no diffuse fluxes without photochemistry."""
    from runtime_validation import validate_runtime_config
    cfg = _minimal_valid_cfg(use_live_flux=True, use_photo=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_runtime_config(cfg, root=ROOT)
    assert "use_live_flux" in str(excinfo.value)


def test_any_live_flag_on_helper():
    """`live_ui.any_live_flag_on` reports True iff any flag is set."""
    from live_ui import any_live_flag_on
    cfg = _minimal_valid_cfg()
    assert not any_live_flag_on(cfg)
    for flag in ("use_live_plot", "use_live_flux", "use_save_movie", "use_flux_movie"):
        c = _minimal_valid_cfg(**{flag: True})
        assert any_live_flag_on(c), f"any_live_flag_on missed {flag}"


def _build_var_atm_para_shim(nz: int, ni: int, nbin: int):
    """Hand-build a minimal var/atm/para shim that the live-UI port reads."""
    import chem_funs
    species = chem_funs.spec_list

    # Pick the first 3 species we have plot defaults for.
    rng = np.random.default_rng(0)
    ymix = rng.uniform(1e-15, 1e-3, size=(nz, ni))
    var = SimpleNamespace(
        ymix=ymix,
        y=ymix * 1e10,
        t=1.234e3,
        sflux=rng.uniform(1e-5, 1.0, size=(nz + 1, nbin)),
        dflux_d=rng.uniform(1e-8, 1e-2, size=(nz + 1, nbin)),
        dflux_u=rng.uniform(1e-8, 1e-2, size=(nz + 1, nbin)),
    )
    atm = SimpleNamespace(
        pco=np.geomspace(1e7, 1e1, nz),
        pico=np.geomspace(1e7, 1e1, nz + 1),
        zco=np.linspace(0.0, 5e7, nz + 1),
        zmco=np.linspace(0.0, 5e7, nz),
    )
    para = SimpleNamespace(count=42)
    return var, atm, para, species


def test_update_mix_writes_movie_frame(tmp_path, monkeypatch):
    """`use_save_movie=True` writes a PNG to `vulcan_cfg.movie_dir`."""
    import vulcan_cfg
    from live_ui import LiveUI
    monkeypatch.setattr(vulcan_cfg, "movie_dir", str(tmp_path) + "/")
    monkeypatch.setattr(vulcan_cfg, "use_live_plot", False)
    monkeypatch.setattr(vulcan_cfg, "use_save_movie", True)
    monkeypatch.setattr(vulcan_cfg, "plot_height", False)
    monkeypatch.setattr(vulcan_cfg, "use_condense", False)

    var, atm, para, _ = _build_var_atm_para_shim(nz=20, ni=93, nbin=8)
    ui = LiveUI()
    ui.update_mix(var, atm, para, save_movie=True, show=False)
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, f"no PNG written under {tmp_path}; got {list(tmp_path.iterdir())}"
    assert ui.pic_count == 1


def test_update_flux_writes_movie_frame(tmp_path, monkeypatch):
    """`use_flux_movie=True` writes a flux JPG under plot/movie/."""
    import vulcan_cfg
    from live_ui import LiveUI

    cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    try:
        monkeypatch.setattr(vulcan_cfg, "use_live_flux", False)
        monkeypatch.setattr(vulcan_cfg, "use_flux_movie", True)
        var, atm, para, _ = _build_var_atm_para_shim(nz=20, ni=93, nbin=8)
        ui = LiveUI()
        ui.update_flux(var, atm, para, save_movie=True, show=False)
        flux_dir = tmp_path / "plot" / "movie"
        jpgs = list(flux_dir.glob("flux-*.jpg"))
        assert jpgs, f"no flux frame written under {flux_dir}"
    finally:
        os.chdir(cwd)


def test_dispatch_no_op_when_all_flags_off(monkeypatch):
    """`dispatch` returns immediately when no flag is set; no matplotlib import."""
    import vulcan_cfg
    from live_ui import LiveUI

    for flag in ("use_live_plot", "use_live_flux", "use_save_movie", "use_flux_movie"):
        monkeypatch.setattr(vulcan_cfg, flag, False)

    ui = LiveUI()
    var, atm, para, _ = _build_var_atm_para_shim(nz=20, ni=93, nbin=8)
    ui.dispatch(var, atm, para)
    # No frames written, matplotlib never imported.
    assert ui._plt is None
    assert ui.pic_count == 0
