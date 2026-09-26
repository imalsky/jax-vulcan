"""The certified column must not depend on the geometry refresh cadence.

HD209 photo-off is the case that did: at `update_frq` 100 the run certified
with its carried mu 26.3% off what the final ymix implies and its column
1.56 in log VMR away from the `update_frq` 1 run (notes §1.10). The
certificate now refreshes the geometry on its candidate step and requires
mu/g/Hp/dzi/Hpi to move by less than `geom_conv_tol`; the two cadences then
agree to 3e-3. A subprocess isolates the import-frozen chemistry network.
"""

import os
from pathlib import Path
import subprocess
import sys

CADENCE_TOL = 1e-2   # measured 3.1e-3 at geom_conv_tol 1e-3; 1.56 without the term


def _run(update_frq):
    import jax.numpy as jnp
    import numpy as np
    from vulcan_jax import atm_refresh
    from vulcan_jax.config import load_config
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.legacy_io import Output
    from vulcan_jax.network import parse_network
    from vulcan_jax.op_jax import Ros2JAX
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.state import RunState, legacy_view

    cfg = load_config("HD209", use_photo=False, use_print_prog=False,
                      count_max=6000, update_frq=update_frq)
    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    solver = Ros2JAX()
    solver.naming_solver(para)
    integ = OuterLoop(solver, Output(cfg=cfg), cfg=cfg)
    integ._ensure_runner(var, atm)
    static = make_atm_static(atm, parse_network(cfg.network).ni, len(atm.Tco),
                             cfg=integ._cfg)
    final = integ._runner(integ._pack_state_from_runstate(rs), static)
    assert int(final.termination_reason) == 1, int(final.termination_reason)
    fresh = atm_refresh.update_mu_dz_jax(final.ymix, integ._refresh_static)
    for name, new in zip(("mu", "g", "Hp", "dz", "zco", "dzi", "Hpi"), fresh):
        if name not in ("dz", "zco"):
            rel = float(jnp.max(jnp.abs(new - getattr(final, name)) / jnp.abs(new)))
            assert rel < float(integ._statics.geom_conv_tol), (name, rel)
    return np.asarray(final.ymix)


def _check():
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    y100, y1 = _run(100), _run(1)
    m = (y100 > 1e-8) & (y1 > 1e-8)
    worst = float(np.max(np.abs(np.log(y100[m] / y1[m]))))
    print(f"max |d ln VMR| between cadences: {worst:.3g}", flush=True)
    assert worst < CADENCE_TOL, worst


def test_certified_column_is_geometry_cadence_independent(tmp_path):
    env = dict(os.environ, VULCAN_JAX_NETWORK="thermo/NCHO_photo_network.txt",
               VULCAN_JAX_ATOM_LIST="H,O,C,N", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check()
