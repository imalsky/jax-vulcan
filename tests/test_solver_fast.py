"""The default solver path (`vulcan_jax.solver_fast`, `VULCAN_JAX_SOLVER=fast|ffi`)
against the reference `solver.block_thomas_diag_offdiag`.

What must hold, for both backends: the primal is the same solution (bit-identical
for `fast`, which runs the same scans; residual-matched for the C++ `ffi` kernel),
the `custom_linear_solve` tangent agrees with differentiating through the LU
and its linearised residual is never worse, reverse mode (steady_state_grad's
`jax.vjp` through the step) agrees, and vmap is consistent. Random systems at
the VULCAN shape, then the real HD189 blocks at four dt (the W39b blocks in a
SNCHO child, slow-gated like the other W39b children).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import vulcan_jax.solver as ref
import vulcan_jax.solver_fast as fast_mod

ROOT = Path(__file__).resolve().parent.parent
DTS = (1e2, 3.8e4, 1e6, 1e11)


@pytest.fixture(params=["fast", "ffi"])
def fast(request, monkeypatch):
    if request.param == "ffi":
        try:
            fast_mod.build()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pytest.skip(f"C++ kernel not built: {e}")
    monkeypatch.setattr(fast_mod, "BACKEND", request.param)
    return fast_mod


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a)))


def _resid(diag, sup, sub, x, b):
    return float(jnp.max(jnp.abs(fast_mod._matvec(diag, sup, sub, x) - b)) / jnp.max(jnp.abs(b)))


def _cur(diag, sup, sub, rhs):
    return ref.block_thomas_diag_offdiag(diag, sup, sub, rhs)


def _cand(diag, sup, sub, rhs):
    return fast_mod.solve(fast_mod.factor(diag, sup, sub), rhs)


def _system(nz, ni, seed, boost, scale):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.standard_normal((nz, ni, ni)) + boost * np.eye(ni)),
        jnp.asarray(rng.standard_normal((nz - 1, ni)) * scale),
        jnp.asarray(rng.standard_normal((nz - 1, ni)) * scale),
        jnp.asarray(rng.standard_normal((nz, ni))),
    )


def compare(diag, sup, sub, rhs, tans, seed=1):
    """Both arms jitted (the production condition). Returns a dict of the
    agreement numbers; the callers assert on them."""
    x0, dx0 = jax.jit(lambda *a: jax.jvp(_cur, a, tans))(diag, sup, sub, rhs)
    x1, dx1 = jax.jit(lambda *a: jax.jvp(_cand, a, tans))(diag, sup, sub, rhs)
    w = jnp.asarray(np.random.default_rng(seed).standard_normal(rhs.shape))
    g0 = jax.jit(jax.grad(lambda *a: jnp.sum(w * _cur(*a)), argnums=(0, 1, 2, 3)))(diag, sup, sub, rhs)
    g1 = jax.jit(jax.grad(lambda *a: jnp.sum(w * _cand(*a)), argnums=(0, 1, 2, 3)))(diag, sup, sub, rhs)
    batched = jax.jit(jax.vmap(_cand))(*(jnp.stack([a, a]) for a in (diag, sup, sub, rhs)))
    # the linearised equation A dx = db - dA x, satisfied by the exact tangent
    b_lin = tans[3] - fast_mod._matvec(tans[0], tans[1], tans[2], x0)
    # the rhs-cotangent solves the transposed system, A^T g = w
    gres = lambda g: _resid(jnp.swapaxes(diag, 1, 2), sub, sup, g, w)
    return {
        "primal_equal": bool(jnp.array_equal(x0, x1)),
        "primal_rel": _rel(x0, x1),
        "resid_cur": _resid(diag, sup, sub, x0, rhs),
        "resid_cand": _resid(diag, sup, sub, x1, rhs),
        "tangent_rel": _rel(dx0, dx1),
        "tres_cur": _resid(diag, sup, sub, dx0, b_lin),
        "tres_cand": _resid(diag, sup, sub, dx1, b_lin),
        "grad_rel": max(_rel(a, b) for a, b in zip(g0, g1)),
        "gres_cur": gres(g0[3]),
        "gres_cand": gres(g1[3]),
        "vmap_rel": max(_rel(batched[0], x1), _rel(batched[1], x1)),
    }


def test_fast_matches_reference_on_random_system(fast):
    diag, sup, sub, rhs = _system(120, 93, 42, boost=1e10, scale=1e-3)
    tans = _system(120, 93, 43, boost=0.0, scale=1e-3)
    r = compare(diag, sup, sub, rhs, tans)
    if fast.BACKEND == "fast":
        assert r["primal_equal"], r
    else:
        assert r["primal_rel"] < 1e-10, r
    assert r["tangent_rel"] < 1e-8, r
    assert r["grad_rel"] < 1e-8, r
    assert r["vmap_rel"] < 1e-12, r


def capture_stage1(fixture: str, cfg_name: str, dt: float):
    """The stage-1 system `_ros2_stages` assembles from a saved converged state,
    recorded by wrapping the solver entry points the step imports."""
    import vulcan_jax.chem as chem_mod
    import vulcan_jax.network as net_mod
    from vulcan_jax import jax_step
    from vulcan_jax.config import load_config

    d = np.load(ROOT / "tests" / "data" / fixture)
    atm = jax_step.AtmStatic(
        **{
            f: (bool(d[f"atmbool__{f}"]) if f"atmbool__{f}" in d else jnp.asarray(d[f"atm__{f}"]))
            for f in jax_step.AtmStatic._fields
        }
    )
    net = chem_mod.to_jax(net_mod.parse_network(load_config(cfg_name).network))
    y, k_arr = jnp.asarray(d["y_star"]), jnp.asarray(d["k_arr"])
    cap: dict = {}
    f0, s0 = jax_step.factor_block_thomas_diag_offdiag, jax_step.solve_block_thomas_diag_offdiag

    def rec_f(diag, sup, sub):
        cap["A"] = (diag, sup, sub)
        return f0(diag, sup, sub)

    def rec_s(factors, rhs):
        cap.setdefault("rhs", []).append(rhs)
        return s0(factors, rhs)

    jax_step.factor_block_thomas_diag_offdiag, jax_step.solve_block_thomas_diag_offdiag = rec_f, rec_s
    try:
        jax_step._ros2_stages(y, k_arr, jnp.float64(dt), atm, net, None)
    finally:
        jax_step.factor_block_thomas_diag_offdiag, jax_step.solve_block_thomas_diag_offdiag = f0, s0
    diag, sup, sub = cap["A"]
    return diag, sup, sub, cap["rhs"][0]


def check_real_blocks(fixture: str, cfg_name: str, backend: str):
    """Assert the agreement bars on the real blocks at DTS; returns the rows.
    Also run by the SNCHO child for W39b."""
    fast_mod.BACKEND = backend
    rows = []
    for dt in DTS:
        diag, sup, sub, rhs = capture_stage1(fixture, cfg_name, dt)
        ni = diag.shape[1]
        c0 = 1.0 / ((1.0 + 2.0**-0.5) * dt)
        ks = jax.random.split(jax.random.PRNGKey(0), 4)
        # direction: the dt shift on the diagonal plus 1e-3 relative noise on every operand
        tans = (
            -c0 * jnp.broadcast_to(jnp.eye(ni), diag.shape) + 1e-3 * jnp.abs(diag) * jax.random.normal(ks[0], diag.shape),
            1e-3 * jnp.abs(sup) * jax.random.normal(ks[1], sup.shape),
            1e-3 * jnp.abs(sub) * jax.random.normal(ks[2], sub.shape),
            1e-3 * jnp.abs(rhs) * jax.random.normal(ks[3], rhs.shape),
        )
        r = compare(diag, sup, sub, rhs, tans)
        r["dt"] = dt
        rows.append(r)
    print(f"\n{backend} {fixture}:")
    for r in rows:
        print("  " + " ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    # `fast` runs the reference scans, so its tangent differs from AD-through-LU
    # only by the rule; the C++ LU has its own roundoff on cond ~1e18 blocks, so
    # for `ffi` the direct tangent comparison is a loose sanity bar and the
    # linearised residual (tres) is the accuracy statement. The normwise
    # residuals of the tangent and cotangent are eps x (|A||x|/|b|) for ANY
    # backward-stable solve on these blocks (1e14..1e23 for a random rhs, notes
    # §1.4.1), so they are only compared between arms, never to a fixed bar.
    tangent_bar = 1e-5 if backend == "fast" else 1e-3
    for r in rows:
        if backend == "fast":
            assert r["primal_equal"], r
        else:
            assert r["resid_cand"] <= 2.0 * r["resid_cur"] + 1e-12, r
        # at dt=1e11 the reference tangent residual is itself O(1) (the
        # off-run-path conditioning regime, notes §1.4): nothing to match there
        if r["dt"] <= 1e6:
            assert r["tangent_rel"] < tangent_bar, r
            assert r["tres_cand"] <= 2.0 * r["tres_cur"] + 1e-13, r
            assert r["gres_cand"] <= 2.0 * r["gres_cur"], r
        if backend == "fast":  # the transpose runs on the same factors as today
            assert r["grad_rel"] < 1e-6, r
        assert r["vmap_rel"] < 1e-12, r
    return rows


@pytest.mark.skipif(
    not (ROOT / "tests" / "data" / "adj_state_hd189.npz").exists(),
    reason="HD189 adjoint fixture missing",
)
def test_fast_on_real_hd189_blocks(fast):
    check_real_blocks("adj_state_hd189.npz", "HD189", fast.BACKEND)


_DEFAULT_CHILD = r"""
import os, sys
os.environ.pop("VULCAN_JAX_SOLVER", None)
sys.path.insert(0, os.path.join(sys.argv[1], "tests"))
from vulcan_jax import jax_step
import vulcan_jax.solver_fast as fast_mod
print("solve_is_fast", jax_step.solve_block_thomas_diag_offdiag is fast_mod.solve)
print("factor_is_fast", jax_step.factor_block_thomas_diag_offdiag is fast_mod.factor)
"""


def test_fast_is_the_default_solver():
    """With the switch unset, `jax_step` binds the `solver_fast` entry points."""
    from _helpers import run_child

    res = run_child(
        _DEFAULT_CHILD,
        network="thermo/NCHO_photo_network.txt",
        label="solver_fast default wiring",
    )
    assert "solve_is_fast True" in res.stdout, res.stdout
    assert "factor_is_fast True" in res.stdout, res.stdout


_CHILD = r"""
import os, sys
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "tests"))
from test_solver_fast import check_real_blocks
import vulcan_jax.solver_fast as fast_mod
for backend in ("fast", "ffi"):
    if backend == "ffi":
        fast_mod.build()
    check_real_blocks("adj_state_w39b.npz", "W39b", backend)
"""


@pytest.mark.skipif(os.environ.get("VULCAN_JAX_RUN_SLOW") != "1", reason="slow W39b child; set VULCAN_JAX_RUN_SLOW=1")
def test_fast_on_real_w39b_blocks():
    from _helpers import run_child

    res = run_child(_CHILD, network="thermo/SNCHO_photo_network.txt", label="solver_fast W39b")
    print(res.stdout)
    assert res.stdout.count("dt=") == 2 * len(DTS)
