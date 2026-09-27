"""Full adaptive-integration Kzz tangent against independently converged FD.

Uses the paper's HD189 photo-off comparison, without saved output or sibling
dependencies. A subprocess isolates the import-frozen chemistry network.
"""

import pytest


def _check_gradient():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from vulcan_jax.config import load_config
    from vulcan_jax.state import RunState
    from vulcan_jax.op_jax import Ros2JAX
    from vulcan_jax.outer_loop import OuterLoop
    from vulcan_jax.legacy_io import Output

    jax.config.update("jax_enable_x64", True)
    cfg = load_config("default", use_photo=False, use_print_prog=False,
                      yconv_cri=1e-3, yconv_min=1e-3)
    integ = OuterLoop(Ros2JAX(), Output(), cfg=cfg)
    state0, static = integ.prepare_runstate(RunState.with_pre_loop_setup(cfg))

    def run(kzz):
        final = integ._runner(state0, static._replace(Kzz=kzz))
        return final.y / jnp.sum(final.y, axis=1, keepdims=True), final

    _, tangent, nominal = jax.jvp(run, (static.Kzz,), (static.Kzz,), has_aux=True)
    eps = 1e-3
    plus, state_plus = run((1 + eps) * static.Kzz)
    minus, state_minus = run((1 - eps) * static.Kzz)
    for final in (nominal, state_plus, state_minus):
        assert integ._classify_end_case(final) == 1, "FD/JVP column did not converge"
    fd = np.asarray((plus - minus) / (2 * eps))
    tangent = np.asarray(tangent)
    assert np.isfinite(fd).all() and np.isfinite(tangent).all()
    high = np.abs(fd) >= np.quantile(np.abs(fd), 0.99)
    assert np.min(np.abs(fd[high])) > 0
    relative = np.abs((tangent[high] - fd[high]) / fd[high])
    print(f"Kzz JVP/FD: median={np.median(relative):.6g}, "
          f"max={relative.max():.6g}, cells={high.sum()}", flush=True)
    # Paper comparison is below 0.1% on these cells; cutting the adaptive
    # controller's tangent produces roughly 40% error despite identical primals.
    assert relative.max() < 1e-3


@pytest.mark.slow
def test_kzz_jvp_matches_reconverged_finite_difference(tmp_path):
    from _helpers import run_self

    run_self(__file__, network="thermo/NCHO_photo_network.txt",
             atom_list="H,O,C,N", cwd=tmp_path)


if __name__ == "__main__":
    _check_gradient()
