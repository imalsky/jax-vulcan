"""`OuterLoop.run_jvp` certifies the tangent, not only the state.

A `jax.jvp` through the runner stops when the column certifies, possibly
with the tangent unrelaxed; `run_jvp` holds the tangent's change over the
same lookback to the column's tolerance. Along d/d ln Kzz on HD209
photo-off and W39b photo-on, the certified tangent matches a central
difference and does not move when the carry continues. One subprocess per
case (import-frozen network).
"""

import sys

import pytest

# case -> (config, use_photo, network, atom_list, vmr_floor, fd_signal_min).
# vmr_floor is where a central difference still resolves the derivative;
# above 1e-8 on the photo-on column, top-layer radicals respond too
# nonlinearly to +/-H.
CASES = {
    "HD209_photo_off": ("HD209", False, "thermo/NCHO_photo_network.txt",
                        "H,O,C,N", 1e-8, 0.1),
    "W39b_photo_on": ("W39b", True, "thermo/SNCHO_photo_network.txt",
                      "H,O,C,N,S", 1e-4, 0.05),
}

H = 0.1            # ln Kzz step: the unit the tangent is certified in
FD_TOL = 0.01      # measured 6e-4 (HD209) / 6.7e-3 (W39b) per H
CONTINUE_TOL = 0.01   # measured: the tangent moves 2.8e-4 over 1000 more steps
CONTINUE_STEPS = 1000


def _build(case, kzz_scale=1.0):
    from vulcan_jax.config import load_config
    from vulcan_jax.legacy_io import Output
    from vulcan_jax.op_jax import Ros2JAX
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.state import RunState

    name, use_photo = CASES[case][:2]
    cfg = load_config(name, use_photo=use_photo, use_print_prog=False,
                      count_max=6000)
    integ = OuterLoop(Ros2JAX(), Output(cfg=cfg), cfg=cfg)
    state, static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))
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


def _check(case):
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)

    integ, state, static = _build(case)
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
    fp = integ._runner(*_build(case, float(np.exp(H)))[1:])
    fm = integ._runner(*_build(case, float(np.exp(-H)))[1:])
    assert int(fp.termination_reason) == 1 and int(fm.termination_reason) == 1
    yp, ym = np.asarray(fp.ymix), np.asarray(fm.ymix)
    vmr_floor, fd_signal_min = CASES[case][4:]
    m = (ymix > vmr_floor) & (yp > 0) & (ym > 0)
    dln_fd = np.zeros_like(ymix)
    dln_fd[m] = (np.log(yp[m]) - np.log(ym[m])) / 2.0
    worst = float(np.max(np.abs(dln_ad - dln_fd)[m]))
    signal = float(np.max(np.abs(dln_fd[m])))
    print(f"certified tangent vs FD over {int(m.sum())} cells: max |diff| "
          f"{worst:.3g}, max |FD| {signal:.3g}", flush=True)
    assert signal > fd_signal_min, signal
    assert worst < FD_TOL, worst

    # continue the SAME carry (primal and tangent) past the certificate: the
    # certified tangent must already be the settled one
    cont = final._replace(count_min_dyn=jnp.int32(n + CONTINUE_STEPS),
                          count_max_dyn=jnp.int32(n + 3 * CONTINUE_STEPS))
    final2, dfinal2, _tl2, ok2 = integ.run_jvp(cont, static, dfinal, datm)
    assert int(final2.termination_reason) == 1 and bool(ok2)
    dln_ad2 = np.asarray(dfinal2.ymix) / np.asarray(final2.ymix)
    moved = float(np.max(np.abs(dln_ad2 - dln_ad)[m]))
    print(f"continued to {int(final2.accept_count)} steps: tangent moved "
          f"{moved:.3g}", flush=True)
    assert moved < CONTINUE_TOL, moved


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_certified_tangent_is_the_settled_sensitivity(tmp_path, case):
    from _helpers import run_self

    network, atom_list = CASES[case][2:4]
    run_self(__file__, case, network=network, atom_list=atom_list, cwd=tmp_path)


if __name__ == "__main__":
    _check(sys.argv[1])
