"""Smoke-test the production console entry point `vulcan_jax_cli.cli_main`.

The CLI (the `vulcan-jax` console script) is the one path a user actually runs,
yet the rest of the suite builds the pipeline by hand in fixtures and never
exercises `cli_main`. This runs it end-to-end on the default HD189 config with
a tiny step cap and asserts it produces a schema-correct `.vul` file.

The CLI loads its config by name/path (`load_config`), so the tiny-cap overrides
are delivered through a real resolved YAML written to the temp dir, and
`cli_main` is driven with an explicit argv (never the ambient pytest `sys.argv`).
The run happens in a temporary cwd so no `output/` artifacts leak.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")


@pytest.mark.strict_isolation
def test_cli_main_produces_vul():
    from vulcan_jax.config import dump_config, load_config
    import vulcan_jax.vulcan_jax_cli as cli

    overrides = {
        "count_max": 3,
        "count_min": 4,
        "trun_min": 1e22,
        "runtime": 1e22,
        "use_print_prog": False,
        "use_live_plot": False,
        "use_live_flux": False,
        "use_plot_end": False,
        "use_plot_evo": False,
        "use_save_movie": False,
        "use_flux_movie": False,
        "save_evolution": False,
        "output_dir": "output/",
    }
    cfg = load_config("default")
    for k, v in overrides.items():
        setattr(cfg, k, v)

    cwd0 = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="vj_cli_smoke_") as tmp:
        try:
            os.chdir(tmp)
            (Path(tmp) / "output").mkdir(exist_ok=True)
            (Path(tmp) / "plot").mkdir(exist_ok=True)
            cfg_path = Path(tmp) / "cli_smoke.yaml"
            dump_config(cfg, cfg_path)

            cli.cli_main(["--config", str(cfg_path)])

            out = Path(tmp) / cfg.output_dir / cfg.out_name
            assert out.is_file(), f"CLI did not write {out}"
            with out.open("rb") as handle:
                data = pickle.load(handle)
            # Public .vul schema: three top-level dicts.
            assert set(("variable", "atm", "parameter")).issubset(data.keys())
            assert "y" in data["variable"] and "ymix" in data["variable"]
            assert data["variable"]["y"].shape == data["variable"]["ymix"].shape
            assert "Tco" in data["atm"] and "pco" in data["atm"]
            assert "count" in data["parameter"]
        finally:
            os.chdir(cwd0)


def test_frozen_knob_mismatch_flags_only_non_default_networks():
    """`network`/`atom_list`/`com_file` drive the relaunch; nothing else does.

    A false negative silently runs a config against the wrong network; a false
    positive sends the CLI into a relaunch it cannot resolve.
    """
    from vulcan_jax.config import load_config
    import vulcan_jax.vulcan_jax_cli as cli

    # Same NCHO network as the import-frozen default -> no relaunch.
    for name in ("default", "HD189", "HD209"):
        assert cli._frozen_knob_mismatch(load_config(name)) == {}, name

    # Sulfur networks differ in both network and atom_list.
    w39b = cli._frozen_knob_mismatch(load_config("W39b"))
    assert w39b == {
        "VULCAN_JAX_NETWORK": "thermo/SNCHO_photo_network.txt",
        "VULCAN_JAX_ATOM_LIST": "H,O,C,N,S",
    }

    # A knob that is not import-frozen must never trigger a relaunch.
    assert cli._frozen_knob_mismatch(load_config("default", nz=77, count_max=5)) == {}


def test_relaunch_sets_env_and_refuses_to_loop(monkeypatch):
    """The relaunch passes the mismatch through the env and re-runs the config."""
    from vulcan_jax.config import load_config
    import vulcan_jax.vulcan_jax_cli as cli

    captured = {}

    def fake_execve(path, argv, env):
        captured.update(path=path, argv=argv, env=env)
        raise SystemExit(0)  # execve never returns

    monkeypatch.setattr(cli.os, "execve", fake_execve)
    monkeypatch.delenv(cli._RELAUNCH_GUARD, raising=False)

    with pytest.raises(SystemExit):
        cli._relaunch_for_frozen_knobs(load_config("W39b"), "W39b")

    assert captured["env"]["VULCAN_JAX_NETWORK"] == "thermo/SNCHO_photo_network.txt"
    assert captured["env"]["VULCAN_JAX_ATOM_LIST"] == "H,O,C,N,S"
    assert captured["env"][cli._RELAUNCH_GUARD] == "1"
    assert captured["argv"][-2:] == ["--config", "W39b"]

    # A matching config returns without touching execve.
    captured.clear()
    cli._relaunch_for_frozen_knobs(load_config("default"), "default")
    assert captured == {}

    # Already relaunched and still mismatched: raise rather than loop.
    monkeypatch.setenv(cli._RELAUNCH_GUARD, "1")
    with pytest.raises(RuntimeError, match="after a relaunch"):
        cli._relaunch_for_frozen_knobs(load_config("W39b"), "W39b")


if __name__ == "__main__":
    test_cli_main_produces_vul()
    print("PASS")
