"""YAML config loader (`vulcan_jax.config`).

Covers the authored-YAML surface that replaced the hand-written `vulcan_cfg`
module: derived-value resolution, gravity from Mp/Rp, the CWD-first override,
env-frozen knobs, and caller overrides. Durable across the module deletion
(does not reference the old `vulcan_cfg` / `cfg_examples` sources).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import yaml

from vulcan_jax.atm_setup import surface_gravity
from vulcan_jax.config import Config, default_config, load_config

# Adopted surface gravity (cm/s^2) each shipped config must reproduce via
# g = G*Mp/Rp^2. default reproduces the historical HD189 Mp=1.118 m_jup value.
_EXPECTED_GS = {
    # default.yaml is HD189-flavored and must reproduce the adopted 2140
    # (an earlier 2139.770515 pin enshrined a raw-literature-Mp slip that
    # shifted photolysis J up to ~7%; see notes.md, Validation).
    "default": 2140.0,
    "HD189": 2140.0,
    "HD209": 936.0,
    "W39b": 422.0,
    # Explicit VULCAN 3 preset: HD189 physics, vm_branch numerics.
    "HD189_vulcan3": 2140.0,
}


def test_expected_gs_covers_every_shipped_config():
    """A newly added config must not slip past this file's coverage.

    `_EXPECTED_GS` is hand-maintained; without this check a new
    `configs/*.yaml` would silently go untested by every parametrized test
    below (which is how HD189_vulcan3.yaml would have been missed).
    """
    from vulcan_jax._paths import PACKAGE_ROOT

    shipped = {p.stem for p in (PACKAGE_ROOT / "configs").glob("*.yaml")}
    assert shipped == set(_EXPECTED_GS), (
        "shipped configs and _EXPECTED_GS disagree: "
        f"only on disk {sorted(shipped - set(_EXPECTED_GS))}, "
        f"only in the table {sorted(set(_EXPECTED_GS) - shipped)}"
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED_GS))
def test_shipped_config_loads_and_resolves(name):
    cfg = load_config(name)
    assert isinstance(cfg, Config)

    # Gravity is derived from Mp + Rp; `gs` is not a config attribute.
    assert not hasattr(cfg, "gs")
    assert surface_gravity(cfg) == pytest.approx(_EXPECTED_GS[name], rel=1e-9)

    # Derived values are filled by the loader.
    assert cfg.dt_max == cfg.runtime * 1e-5
    assert cfg.photo_switch_longdy_thresh == cfg.yconv_min * 10.0
    assert cfg.save_movie_rate == cfg.live_plot_frq
    assert cfg.para_anaTP == cfg.para_warm

    # count_max is a plain int (accepts scientific-notation authoring).
    assert isinstance(cfg.count_max, int)

    # Import-frozen knobs are present.
    assert cfg.network and cfg.atom_list and cfg.com_file


def test_overrides_win_over_yaml_and_derived():
    cfg = load_config("default", nz=99, runtime=1.0e20)
    assert cfg.nz == 99
    # dt_max derives from the overridden runtime (derived fills after overrides).
    assert cfg.dt_max == 1.0e20 * 1e-5
    # An explicit derived key in the overrides is not clobbered.
    cfg2 = load_config("default", dt_max=123.0)
    assert cfg2.dt_max == 123.0
    assert default_config() is default_config()   # cached singleton


def test_cwd_configs_override(tmp_path, monkeypatch):
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    # A minimal config that reuses the packaged default's frozen knobs is not
    # needed here; we only assert the CWD file is preferred and parsed.
    (cfgdir / "mine.yaml").write_text(
        textwrap.dedent(
            """
            runtime: 5.0e21
            yconv_min: 0.2
            live_plot_frq: 7
            para_warm: [1, 2, 3]
            count_max: 1e4
            network: thermo/NCHO_photo_network.txt
            atom_list: [H, O, C, N]
            com_file: thermo/all_compose.txt
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config("mine")
    assert cfg.runtime == 5.0e21
    assert cfg.dt_max == 5.0e21 * 1e-5  # derived from the CWD file
    assert cfg.count_max == 10000 and isinstance(cfg.count_max, int)
    assert cfg.para_anaTP == [1, 2, 3]


def test_loader_refuses_bad_input(tmp_path, monkeypatch):
    """One refusal per failure class (fail-fast rule): missing config, the
    removed `gs` knob (migration message), an unknown override, an unknown
    key authored in the YAML itself, and duplicate YAML keys. Scientific
    notation must parse as float (the CWD test covers the int case)."""
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config_name_xyz")
    with pytest.raises(ValueError, match=r"removed config key.*gs.*Mp"):
        load_config("W39b", gs=9999.0)
    with pytest.raises(ValueError, match=r"unknown config key.*gss"):
        load_config("W39b", gss=8888.0)
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    base = ("runtime: 1e22\ndt_min: 1e-14\nyconv_min: 0.1\nlive_plot_frq: 10\n"
            "para_warm: [1.0]\ncount_max: 30000\n"
            "network: thermo/NCHO_photo_network.txt\n"
            "atom_list: [H, O, C, N]\ncom_file: thermo/all_compose.txt\n")
    (cfgdir / "typo.yaml").write_text(base + "nz_typo: 100\n")
    (cfgdir / "dup.yaml").write_text("nz: 100\nnz: 150\n")
    (cfgdir / "sci.yaml").write_text(base)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"unknown config key.*nz_typo"):
        load_config("typo")
    with pytest.raises(yaml.YAMLError):
        load_config("dup")
    cfg = load_config("sci")
    assert isinstance(cfg.runtime, float) and cfg.runtime == 1e22
    assert isinstance(cfg.dt_min, float) and cfg.dt_min == 1e-14


def test_frozen_env_overrides_yaml():
    """$VULCAN_JAX_* select the import-frozen knobs over the YAML values.

    Run in a subprocess so the env is read at load time without touching this
    interpreter's already-import-frozen network.
    """
    src = textwrap.dedent(
        """
        from vulcan_jax.config import load_config
        cfg = load_config("default")
        assert cfg.atom_list == ["H", "O", "C", "N", "S"], cfg.atom_list
        assert cfg.network == "thermo/SNCHO_photo_network.txt", cfg.network
        print("OK")
        """
    )
    env = dict(os.environ)
    env["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"
    env["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"
    r = subprocess.run(
        [sys.executable, "-c", src], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_getattr_fallback_literals_match_default_yaml():
    """Every scalar `getattr(cfg, key, literal)` default in src/ must equal
    default.yaml's value: a config omitting the key must not silently run a
    different model. Sentinels are exempt (None/''/[]/{} presence checks;
    `use_photo` safe-off in validators that see partial configs).
    """
    import ast
    import re
    from pathlib import Path

    pkg = Path(__file__).resolve().parent.parent / "src" / "vulcan_jax"
    defaults = yaml.safe_load(open(pkg / "configs" / "default.yaml"))
    pat = re.compile(
        r'getattr\(\s*(?:cfg|_cfg|self\._cfg|self\.cfg)\s*,\s*"(\w+)"\s*,'
        r"\s*([^()]*?)\s*\)"
    )
    scalar = (bool, int, float)
    mismatches = []
    for f in sorted(pkg.glob("*.py")):
        for lineno, line in enumerate(open(f), 1):
            for m in pat.finditer(line):
                key, lit = m.groups()
                if key not in defaults or key == "use_photo":
                    continue
                try:
                    val = ast.literal_eval(lit)
                except (SyntaxError, ValueError):
                    continue  # dynamic default, not a literal
                want = defaults[key]
                if not (isinstance(val, scalar) and isinstance(want, scalar)):
                    continue  # sentinel or non-scalar key
                if isinstance(val, bool) != isinstance(want, bool):
                    continue
                if float(val) != float(want):
                    mismatches.append(
                        f"{f.name}:{lineno} {key}: literal {val!r} != "
                        f"default.yaml {want!r}"
                    )
    assert not mismatches, "\n".join(mismatches)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
