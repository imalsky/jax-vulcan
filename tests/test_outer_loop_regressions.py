"""Focused regressions for VULCAN-JAX outer-loop parity fixes."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


def test_aggregate_delta_uses_pre_step_ymix() -> None:
    """The Ros2 truncation-error mask uses the pre-step mixing ratio."""
    import outer_loop

    agg_delta = outer_loop._make_aggregate_delta_fn(
        mtol=1e-2,
        atol=0.0,
        zero_bot_row=False,
        condense_zero_mask=jnp.zeros((1, 2), dtype=jnp.bool_),
    )
    sol = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    delta_arr = jnp.asarray([[0.25, 0.50]], dtype=jnp.float64)
    ymix_pre = jnp.asarray([[0.10, 0.10]], dtype=jnp.float64)

    got = float(agg_delta(sol, delta_arr, ymix_pre))

    assert got == 0.50


def test_h2so4_use_relax_keeps_conden_rate_enabled() -> None:
    """Earth's use_relax=['H2O', 'H2SO4'] must not zero H2SO4 rates."""
    script = textwrap.dedent(
        r"""
        from __future__ import annotations

        import importlib.util
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import numpy as np

        root = Path.cwd()
        cfg_path = root / "cfg_examples" / "vulcan_cfg_Earth.py"
        spec = importlib.util.spec_from_file_location("vulcan_cfg", cfg_path)
        cfg = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cfg)
        cfg.use_condense = True
        cfg.condense_sp = ["H2O", "H2SO4"]
        cfg.use_relax = ["H2O", "H2SO4"]
        cfg.r_p = {"H2O_l_s": 1.0e-2, "H2SO4_l": 1.0e-4}
        cfg.rho_p = {"H2O_l_s": 0.9, "H2SO4_l": 1.8302}
        sys.modules["vulcan_cfg"] = cfg

        import outer_loop

        integ = outer_loop.OuterLoop(SimpleNamespace(), SimpleNamespace())
        ni = outer_loop._NETWORK.ni
        nz = 4
        atm = SimpleNamespace(
            Tco=np.full(nz, 250.0, dtype=np.float64),
            Dzz=np.ones((nz - 1, ni), dtype=np.float64),
            sat_p={
                "H2O": np.full(nz, 1.0e-6, dtype=np.float64),
                "H2SO4": np.full(nz, 1.0e-9, dtype=np.float64),
            },
            r_p=cfg.r_p,
            rho_p=cfg.rho_p,
            n_0=np.full(nz, 1.0e12, dtype=np.float64),
        )
        var = SimpleNamespace(
            conden_re_list=[7, 9],
            Rf={7: "H2O -> H2O_l_s", 9: "H2SO4 -> H2SO4_l"},
        )

        static = integ._build_conden_static(
            var, atm, jnp.ones((ni,), dtype=jnp.bool_),
        )
        coeff = np.asarray(static.coeff_per_re, dtype=np.float64)
        assert coeff.shape == (2,)
        assert coeff[0] == 0.0, coeff
        assert coeff[1] > 0.0, coeff
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
