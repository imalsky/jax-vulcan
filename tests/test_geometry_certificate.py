"""The certified column does not depend on the geometry refresh cadence.

The certificate refreshes geometry on its candidate step and requires
mu/g/Hp/dzi/Hpi to move less than `geom_conv_tol`, so HD209 photo-off at
update_frq 100 and 1 certify the same column. Subprocess:
import-frozen network.
"""


CADENCE_TOL = 1e-2   # measured 3.1e-3 at geom_conv_tol 1e-3; 1.56 without the term


def _run(update_frq):
    import jax.numpy as jnp
    import numpy as np
    from vulcan_jax import atm_refresh
    from vulcan_jax.config import load_config
    from vulcan_jax.legacy_io import Output
    from vulcan_jax.op_jax import Ros2JAX
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.state import RunState

    cfg = load_config("HD209", use_photo=False, use_print_prog=False,
                      count_max=6000, update_frq=update_frq)
    integ = OuterLoop(Ros2JAX(), Output(cfg=cfg), cfg=cfg)
    state, atm_static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))
    final = integ._runner(state, atm_static)
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
    from _helpers import run_self

    run_self(__file__, network="thermo/NCHO_photo_network.txt",
             atom_list="H,O,C,N", cwd=tmp_path)


if __name__ == "__main__":
    _check()
