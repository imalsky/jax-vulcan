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
import re
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
    # On a CUDA device `ffi` runs libblock_thomas_cuda.so when it has been built
    # (`--cuda`); the same assertions apply, so nothing here is platform-specific.
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


def banded_solve(diag, sup, sub, b):
    """A BACKWARD-STABLE solve of the block-tridiagonal system (LAPACK gbsv).

    The yardstick the tangent and cotangent bars in `check_real_blocks` are
    read against; `compare` runs it on the same linearised right-hand side
    and the same cotangent seed as the two arms. The system is assembled banded (kl = ku = 2*ni-1) rather
    than dense: n = nz*ni is 10350 here, so the dense matrix would be 857 MB
    while the band is 34 MB. Pass the transposed operands for `A^T g = w`.
    """
    import scipy.linalg

    diag, sup, sub = np.asarray(diag), np.asarray(sup), np.asarray(sub)
    nz, ni = diag.shape[0], diag.shape[1]
    n, kl = nz * ni, 2 * ni - 1
    ab = np.zeros((2 * kl + 1, n))
    rows = np.arange(ni)
    for z in range(nz):
        r, c = z * ni + rows[:, None], z * ni + rows[None, :]
        ab[kl + r - c, c] = diag[z]
        if z + 1 < nz:
            r1, c1 = z * ni + rows, (z + 1) * ni + rows
            ab[kl + r1 - c1, c1] = sup[z]
            ab[kl + c1 - r1, r1] = sub[z]
    x = scipy.linalg.solve_banded((kl, kl), ab, np.asarray(b).reshape(n))
    return jnp.asarray(x.reshape(nz, ni))


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
    diag_t = jnp.swapaxes(diag, 1, 2)
    gres = lambda g: _resid(diag_t, sub, sup, g, w)
    # the backward-stable third arm of each, on the SAME b_lin and w
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
        "tres_dense": _resid(diag, sup, sub, banded_solve(diag, sup, sub, b_lin), b_lin),
        "gres_dense": _resid(diag_t, sub, sup, banded_solve(diag_t, sub, sup, w), w),
        "vmap_rel": max(_rel(batched[0], x1), _rel(batched[1], x1)),
        "resid_vmap": max(_resid(diag, sup, sub, batched[i], rhs) for i in (0, 1)),
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


@pytest.mark.parametrize("nested", [False, True])
def test_ffi_solve_takes_a_stack_of_rhs_on_shared_factors(fast, nested):
    """The tangent directions' pattern: several right-hand sides on ONE
    factorisation, flat and inside an outer vmap over lanes. The stack must equal
    the single solves bit for bit, and (vmap_method="expand_dims") the lowered
    custom call must keep the factors' direction axis at 1 instead of one copy of
    `lu` per direction."""
    if fast.BACKEND != "ffi":
        pytest.skip("the stacked-rhs handler is the C++ kernel's")
    nz, ni, ndir = 12, 7, 3
    if nested:  # outer vmap over 2 lanes (all operands batched) around the inner
        lanes = [_system(nz, ni, 11 + i, boost=1e3, scale=1e-3) for i in range(2)]
        f = jax.vmap(fast.factor)(*(jnp.stack([ln[j] for ln in lanes]) for j in range(3)))
        b = jnp.stack([jnp.stack([ln[3] * (1.0 + 0.1 * k) for k in range(ndir)]) for ln in lanes])
        fn = jax.jit(jax.vmap(lambda fa, bs: jax.vmap(fast.solve, in_axes=(None, 0))(fa, bs)))
        want = jnp.stack([
            jnp.stack([jax.jit(fast.solve)(jax.tree.map(lambda a, i=i: a[i], f), r) for r in b[i]])
            for i in range(2)
        ])
    else:
        diag, sup, sub, rhs = _system(nz, ni, 11, boost=1e3, scale=1e-3)
        f = fast.factor(diag, sup, sub)
        b = jnp.stack([rhs * (1.0 + 0.1 * k) for k in range(ndir)])
        fn = jax.jit(jax.vmap(fast.solve, in_axes=(None, 0)))
        want = jnp.stack([jax.jit(fast.solve)(f, r) for r in b])
    assert np.array_equal(np.asarray(fn(f, b)), np.asarray(want))

    txt = fn.lower(f, b).as_text()
    sig = next(ln for ln in txt.splitlines() if "vulcan_bt_solve" in ln).split("} : ")[-1]
    print(f"\nnested={nested} vulcan_bt_solve{sig.strip()}")
    lead = "2x1x" if nested else "1x"
    assert re.match(rf"\(tensor<{lead}{nz}x{ni}x{ni}xf64>,", sig), sig
    # ... and nothing anywhere in the module materialises `lu` once per direction
    copied = f"tensor<{'2x' if nested else ''}{ndir}x{nz}x{ni}x{ni}xf64>"
    assert copied not in txt, copied


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

    def rec_s(factors, rhs, **kw):
        cap.setdefault("rhs", []).append(rhs)
        return s0(factors, rhs, **kw)

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
    #
    # Both residuals need a THIRD arm. Measured on the HD189 fixture (normwise
    # `gres`, ref / ffi / gbsv):
    #     dt 1e2    2.546e-02 / 5.223e-02 / 3.467e-02
    #     dt 1e6    3.026e+02 / 1.957e+02 / 8.394e+01
    #     dt 1e11   2.111e+08 / 1.657e+08 / 3.843e+03
    # A backward-stable LAPACK solve of the SAME transposed system lands
    # BETWEEN the two arms at dt 1e2, so "ffi <= 2x ref" there is not a
    # statement about the kernel -- it is which of two non-backward-stable
    # residuals happens to be smaller on the column the fixture was built
    # from, and it failed by 2.6% on a rebuilt one. `10x` the gbsv residual is
    # a bar no correct kernel reaches by noise (6.6x headroom at dt 1e2) and
    # one a broken transpose cannot pass. The tangent residual `tres` has the
    # same character (ref / fast / gbsv on the rebuilt fixture, two machines
    # and the GH200 agreeing on the miss):
    #     dt 1e2    6.81e-07 / 4.46e-07 / 1.77e-07
    #     dt 3.8e4  8.52e-05 / 9.85e-05 / 8.51e-05
    #     dt 1e6    1.85e-03 / 5.01e-03 / 3.44e-03
    #     dt 1e11   4.05e+00 / 4.15e+00 / 6.95e-01
    # At dt 1e6 the `fast` arm (the reference scans under a custom_linear_solve
    # rule) sits 2.7x the reference and 1.5x gbsv while the tangents themselves
    # agree to 8.7e-7, so the 2x arm-to-arm bar failed for the same reason.
    # The kernel's tangent against AD-through-LU: 8.5e-4 (CPU kernel) and
    # 1.3e-3 (GH200, job 79354) at dt 1e6, where the reference itself is
    # 1.7e-4 from the gbsv tangent and the kernel 6.8e-4, with the three
    # residuals at 1.9e-3 / 2.0e-3 / 3.4e-3: on these blocks the residual is
    # the accuracy statement and this is the loose sanity bar.
    tangent_bar = 1e-5 if backend == "fast" else 1e-2
    for r in rows:
        if backend == "fast":
            assert r["primal_equal"], r
        else:
            assert r["resid_cand"] <= 2.0 * r["resid_cur"] + 1e-12, r
        # at dt=1e11 the reference tangent residual is itself O(1) (the
        # off-run-path conditioning regime, notes §1.4): nothing to match there
        if r["dt"] <= 1e6:
            assert r["tangent_rel"] < tangent_bar, r
            assert r["tres_cand"] <= max(
                2.0 * r["tres_cur"], 10.0 * r["tres_dense"]
            ), r
            assert r["gres_cand"] <= max(
                2.0 * r["gres_cur"], 10.0 * r["gres_dense"]
            ), r
        if backend == "fast":  # the transpose runs on the same factors as today
            assert r["grad_rel"] < 1e-6, r
        # The vmapped solve is judged by its residual, and by solution
        # agreement wherever batching does not change the arithmetic: on CPU
        # (measured 0 for both backends) and for the kernel on any device
        # (one thread block per lane, the same code batched or not). The
        # `fast` scans on the GH200 lower to a different cuSOLVER LU when
        # batched, and the solutions differ by roundoff amplified by the block
        # conditioning (1.6e-9 at dt 1e2 up to 1.3e-3 at dt 1e11, job 79350).
        assert r["resid_vmap"] <= 2.0 * r["resid_cand"] + 1e-12, r
        if backend == "ffi" or jax.default_backend() == "cpu":
            assert r["vmap_rel"] < 1e-12, r
    return rows


@pytest.mark.skipif(
    not (ROOT / "tests" / "data" / "adj_state_hd189.npz").exists(),
    reason="HD189 adjoint fixture missing",
)
def test_fast_on_real_hd189_blocks(fast):
    check_real_blocks("adj_state_hd189.npz", "HD189", fast.BACKEND)


def check_matrix_free(fixture: str, cfg_name: str, backend: str):
    """`_ros2_stages` gives the solve a matrix-free operator so a tangent never
    builds the dense dA per direction. On a real fixture, free and with pinned
    rows: (1) the operator equals the dense block's `_matvec` to roundoff and a
    pinned row is exactly `c0 x`; (2) the step's primal is unchanged and its
    tangent in y, k, dt and every float atmosphere field agrees with the
    dense operator's. The tangent passes through blocks of cond up to ~6e19
    (notes §1.4), so (2) is a wiring check at 1e-3 (worst, both at dt 1e6:
    HD189 8.5e-5 `fast` / 8.1e-6 `ffi`, W39b 1.4e-7 / 3.2e-13); a lost or
    wrong dA x reads O(1) and above. The primal is bitwise on the CPU and,
    since the gather-table Jacobian (0.16.5), on the GH200 too (0.0 in all
    12 cases of job 79750, both backends, every dt and pinning); job 79533's
    5e-8 `fast` / 5e-10 `ffi` came from the scatter Jacobian's atomics. Off
    the CPU the primal bar is 1e-9, roundoff amplified by the block
    conditioning with room to spare. dt 1e11 is not compared
    (the tangent is O(1)-conditioned there). Also run by the SNCHO child for
    W39b."""
    import vulcan_jax.chem as chem_mod
    import vulcan_jax.network as net_mod
    from vulcan_jax import jax_step
    from vulcan_jax.config import load_config

    fast_mod.BACKEND = backend
    d = np.load(ROOT / "tests" / "data" / fixture)
    atm = jax_step.AtmStatic(
        **{
            f: (bool(d[f"atmbool__{f}"]) if f"atmbool__{f}" in d else jnp.asarray(d[f"atm__{f}"]))
            for f in jax_step.AtmStatic._fields
        }
    )
    net = chem_mod.to_jax(net_mod.parse_network(load_config(cfg_name).network))
    y, k_arr = jnp.asarray(d["y_star"]), jnp.asarray(d["k_arr"])
    floats = {f: v for f in jax_step.AtmStatic._fields
              if isinstance(v := getattr(atm, f), jax.Array) and jnp.issubdtype(v.dtype, jnp.floating)}
    rng = np.random.default_rng(0)

    def nudge(x):
        return x * 1e-3 * jnp.asarray(rng.standard_normal(jnp.shape(x)))

    pins = jnp.asarray(rng.random(y.shape) < 0.02)
    probe = y * jnp.asarray(rng.standard_normal(y.shape))
    solve0 = jax_step.solve_block_thomas_diag_offdiag
    try:
        # (1) the operator the step hands the solve, against its dense block
        calls = []

        def record(factors, rhs, **kw):
            calls.append((factors, kw["matvec"]))
            return solve0(factors, rhs, **kw)

        jax_step.solve_block_thomas_diag_offdiag = record
        dt = DTS[1]
        for fix in (None, pins):
            calls.clear()
            jax_step._ros2_stages(y, k_arr, jnp.float64(dt), atm, net, fix)
            factors, matvec = calls[0]
            got = matvec(probe)
            assert _rel(fast_mod._matvec(factors.diag, factors.sup_d, factors.sub_d, probe), got) < 1e-13
            if fix is not None:
                c0 = 1.0 / (jax_step._ROS2_GAMMA * dt)
                assert jnp.array_equal(got[fix], (c0 * probe)[fix])

        # (2) the step. The arrays go in as jit ARGUMENTS: closed over, XLA
        # folds the step at compile time and the folded tangent is garbage
        # (1e42 against 1e23 on the dense operator), a harness artifact.
        def stages(primals, tangents, fix):
            return jax.jvp(
                lambda a, b, t, af: jax_step._ros2_stages(a, b, t, atm._replace(**af), net, fix)[:2],
                primals, tangents)

        cases = []
        for fix in (jnp.zeros_like(pins), pins):
            for dt in DTS[:-1]:
                primals = (y, k_arr, jnp.float64(dt), floats)
                cases.append((primals, jax.tree_util.tree_map(nudge, primals), fix))
        jax_step.solve_block_thomas_diag_offdiag = solve0
        free = jax.jit(stages)
        got = [free(*c) for c in cases]   # traced now, on the matrix-free operator
        jax_step.solve_block_thomas_diag_offdiag = lambda factors, rhs, **_: solve0(factors, rhs)
        dense = jax.jit(lambda *c: stages(*c))   # traced after the patch: the dense operator
        primal_bar = 0.0 if jax.default_backend() == "cpu" else 1e-9
        for c, (p_free, t_free) in zip(cases, got):
            p_dense, t_dense = dense(*c)
            for a, b in zip(p_free, p_dense):
                assert _rel(b, a) <= primal_bar, (float(c[0][2]), _rel(b, a))
            for a, b in zip(t_free, t_dense):
                assert _rel(b, a) < 1e-3, (float(c[0][2]), bool(c[2].any()), _rel(b, a))
            # no "dt=" here: the W39b child counts check_real_blocks' rows by it
            print(f"matrix-free {cfg_name} {backend} at dt {float(c[0][2]):.0e}, pinned "
                  f"{bool(c[2].any())}: primal {max(_rel(b, a) for a, b in zip(p_free, p_dense)):.1e}, "
                  f"tangent {max(_rel(b, a) for a, b in zip(t_free, t_dense)):.2e}")
    finally:
        jax_step.solve_block_thomas_diag_offdiag = solve0


@pytest.mark.skipif(
    not (ROOT / "tests" / "data" / "adj_state_hd189.npz").exists(),
    reason="HD189 adjoint fixture missing",
)
def test_matrix_free_operator_matches_the_dense_one(fast):
    from vulcan_jax import jax_step

    if jax_step._SOLVER == "reference":
        pytest.skip("the reference pair differentiates through the LU and takes no operator")
    check_matrix_free("adj_state_hd189.npz", "HD189", fast.BACKEND)


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
from test_solver_fast import check_matrix_free, check_real_blocks
import vulcan_jax.solver_fast as fast_mod
for backend in ("fast", "ffi"):
    if backend == "ffi":
        fast_mod.build()
    check_real_blocks("adj_state_w39b.npz", "W39b", backend)
    check_matrix_free("adj_state_w39b.npz", "W39b", backend)
"""


@pytest.mark.skipif(os.environ.get("VULCAN_JAX_RUN_SLOW") != "1", reason="slow W39b child; set VULCAN_JAX_RUN_SLOW=1")
def test_fast_on_real_w39b_blocks():
    from _helpers import run_child

    res = run_child(_CHILD, network="thermo/SNCHO_photo_network.txt", label="solver_fast W39b")
    print(res.stdout)
    assert res.stdout.count("dt=") == 2 * len(DTS)
