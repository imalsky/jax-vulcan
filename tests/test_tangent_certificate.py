"""`OuterLoop.run_jvp` certifies the sensitivity, not only the state.

A `jax.jvp` through the runner stops when the COLUMN certifies; the tangent
carried alongside has its own relaxation and nothing checked it (from a
converged warm start the column certifies at `count_min` with the tangent
unrelaxed, planner notes §1.7). `run_jvp` holds the tangent's change over
the same lookback, per cell over y, to the same two-branch tolerance as the
column. Checked here on HD209 photo-off along d/d ln Kzz (a tangent in
finite-difference-step units, `H`): the certified tangent must match a
central difference of the primal runner, and continuing the SAME carry past
the certificate must not move it. A subprocess isolates the import-frozen
chemistry network.
"""

import os
from pathlib import Path
import subprocess
import sys

H = 0.1            # ln Kzz step: the unit the tangent is certified in
VMR_FLOOR = 1e-8   # cells compared (below it FD is roundoff on 1e-20 VMRs)
FD_TOL = 0.01      # measured 6e-4 against a 0.54 signal (max |d ln VMR| per H)
FD_SIGNAL_MIN = 0.1   # the Kzz response must be there to be compared (0.54)
CONTINUE_TOL = 0.01   # measured: the tangent moves 2.8e-4 over 1000 more steps
CONTINUE_STEPS = 1000


def _build(kzz_scale=1.0):
    import numpy as np
    from vulcan_jax.config import load_config
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.legacy_io import Output
    from vulcan_jax.network import parse_network
    from vulcan_jax.op_jax import Ros2JAX
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.state import RunState, legacy_view

    cfg = load_config("HD209", use_photo=False, use_live_plot=False,
                      use_live_flux=False, use_print_prog=False,
                      count_max=6000)
    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    solver = Ros2JAX()
    solver.naming_solver(para)
    integ = OuterLoop(solver, Output(cfg=cfg), cfg=cfg)
    integ._ensure_runner(var, atm)
    static = make_atm_static(atm, parse_network(cfg.network).ni, len(atm.Tco),
                             cfg=integ._cfg)
    state = integ._pack_state_from_runstate(rs)
    if kzz_scale != 1.0:
        static = static._replace(Kzz=static.Kzz * kzz_scale)
        state = state._replace(pv=state.pv._replace(Kzz=state.pv.Kzz * kzz_scale))
    return integ, state, static


def _zero_tangent(tree):
    """Zeros on float leaves, float0 elsewhere: what jax.jvp hands back."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    def z(x):
        if getattr(x, "dtype", None) is not None and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.zeros_like(x)
        return np.zeros(np.shape(x), dtype=jax.dtypes.float0)
    return jax.tree_util.tree_map(z, tree)


def _check():
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)

    integ, state, static = _build()
    dstate = _zero_tangent(state)._replace(
        pv=_zero_tangent(state.pv)._replace(Kzz=state.pv.Kzz * H))
    datm = _zero_tangent(static)._replace(Kzz=static.Kzz * H)

    final, dfinal, tl, ok = integ.run_jvp(state, static, dstate, datm)
    n = int(final.accept_count)
    print(f"run_jvp: {n} steps, reason {int(final.termination_reason)}, "
          f"tangent_longdy {float(tl):.3g}, longdy {float(final.longdy):.3g}",
          flush=True)
    assert int(final.termination_reason) == 1 and bool(ok)
    ymix = np.asarray(final.ymix)
    dln_ad = np.asarray(dfinal.ymix) / ymix

    # central difference of the primal runner, per step H; both endpoints
    # must certify (reason 1) or the reference is a budget exit
    fp = integ._runner(*_build(float(np.exp(H)))[1:])
    fm = integ._runner(*_build(float(np.exp(-H)))[1:])
    assert int(fp.termination_reason) == 1 and int(fm.termination_reason) == 1
    yp, ym = np.asarray(fp.ymix), np.asarray(fm.ymix)
    m = (ymix > VMR_FLOOR) & (yp > 0) & (ym > 0)
    dln_fd = np.zeros_like(ymix)
    dln_fd[m] = (np.log(yp[m]) - np.log(ym[m])) / 2.0
    worst = float(np.max(np.abs(dln_ad - dln_fd)[m]))
    signal = float(np.max(np.abs(dln_fd[m])))
    print(f"certified tangent vs FD: max |diff| {worst:.3g}, "
          f"max |FD| {signal:.3g}", flush=True)
    assert signal > FD_SIGNAL_MIN, signal
    assert worst < FD_TOL, worst

    # continue the SAME carry (primal and tangent) past the certificate: the
    # certified tangent must already be the settled one
    cont = final._replace(count_min_dyn=jnp.int32(n + CONTINUE_STEPS),
                          count_max_dyn=jnp.int32(n + 3 * CONTINUE_STEPS))
    final2, dfinal2, tl2, ok2 = integ.run_jvp(cont, static, dfinal, datm)
    assert int(final2.termination_reason) == 1 and bool(ok2)
    dln_ad2 = np.asarray(dfinal2.ymix) / np.asarray(final2.ymix)
    moved = float(np.max(np.abs(dln_ad2 - dln_ad)[m]))
    print(f"continued to {int(final2.accept_count)} steps: tangent moved "
          f"{moved:.3g}", flush=True)
    assert moved < CONTINUE_TOL, moved


def test_certified_tangent_is_the_settled_sensitivity(tmp_path):
    env = dict(os.environ, VULCAN_JAX_NETWORK="thermo/NCHO_photo_network.txt",
               VULCAN_JAX_ATOM_LIST="H,O,C,N", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check()
