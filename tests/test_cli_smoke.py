"""Smoke-test the production console entry point `vulcan_jax_cli.cli_main`.

The CLI (the `vulcan-jax` console script) is the one path a user actually runs,
yet the rest of the suite builds the pipeline by hand in fixtures and never
exercises `cli_main`. This runs it end-to-end on the default HD189 config with
a tiny step cap and asserts it produces a schema-correct `.vul` file.

`conftest._cfg_guard` restores every mutated `vulcan_cfg` attribute after the
test; the run happens in a temporary cwd so no `output/` artifacts leak.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")


@pytest.mark.strict_isolation
def test_cli_main_produces_vul():
    from vulcan_jax.config import default_config
    vulcan_cfg = default_config()
    import vulcan_jax.vulcan_jax_cli as cli

    overrides = {
        "count_max": 3,
        "count_min": 4,
        "trun_min": 1e22,
        "runtime": 1e22,
        "use_print_prog": False,
        "use_print_delta": False,
        "use_live_plot": False,
        "use_live_flux": False,
        "use_plot_end": False,
        "use_plot_evo": False,
        "use_save_movie": False,
        "use_flux_movie": False,
        "save_evolution": False,
        "output_dir": "output/",
    }
    saved = {k: getattr(vulcan_cfg, k) for k in overrides}
    cwd0 = os.getcwd()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vj_cli_smoke_") as tmp:
        try:
            for k, v in overrides.items():
                setattr(vulcan_cfg, k, v)
            os.chdir(tmp)
            (Path(tmp) / "output").mkdir(exist_ok=True)
            (Path(tmp) / "plot").mkdir(exist_ok=True)

            cli.cli_main()

            out = Path(tmp) / vulcan_cfg.output_dir / vulcan_cfg.out_name
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
            for k, v in saved.items():
                setattr(vulcan_cfg, k, v)


if __name__ == "__main__":
    test_cli_main_produces_vul()
    print("PASS")
