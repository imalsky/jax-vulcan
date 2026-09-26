"""Default block-Thomas path (`VULCAN_JAX_SOLVER=fast|ffi`, import-frozen).

Same call shape as `solver.py`: `factor(diag, sup_d, sub_d)` once, then
`solve(factors, rhs)` for each Ros2 stage. Two things differ from the plain
`solver.py` pair:

1. `solve` is a `lax.custom_linear_solve`. Its tangent is `A dx = db - dA x`
   on the PRIMAL factors, so both stages and every tangent direction share one
   factorisation, instead of differentiating through the pivoted LU (`lu`'s
   tangent rule was ~85% of the jvp linear algebra, notes §1.4). The primal is
   the same code on the same inputs and is bit-identical. Reverse mode
   transposes the primal sweep on the same factors, on either backend.
2. With `VULCAN_JAX_SOLVER=ffi` the raw factor and solve are one C++ call each
   (`csrc/block_thomas_cpu.cc`, built by `python -m vulcan_jax.solver_fast`):
   the CPU reference for the fused GPU kernel. `custom_linear_solve` is what
   makes a non-differentiable FFI call differentiable here. Reverse mode
   through `ffi` is not a supported route: the steady-state sensitivity's
   LGMRES is roundoff-marginal and the C++ factors' roundoff moves its HD189
   null_quality over the test's bar (notes §1.4.1); use `fast` or `reference`.

On a CUDA device the `ffi` backend runs `csrc/block_thomas_cuda.cu`, the same
math and layout with one thread block per lane: the `ni x ni` block and the
previous layer's inverse sit in dynamic shared memory, pivoting is in-block and
the whole `nz` loop stays in the kernel, so a Ros2 step costs one factor launch
and two solve launches. It is built only by `python -m vulcan_jax.solver_fast
--cuda` on a host with nvcc; without the library nothing changes here and a
device call under `ffi` fails loudly at dispatch. Reverse mode is unaffected:
`transpose_solve` runs the JAX sweep on whichever backend produced the factors.

`jax_step` imports this module by default; `VULCAN_JAX_SOLVER=reference`
restores the plain `solver.py` pair for A/B, and `ffi` selects the C++ kernel.
Removal: delete this file, `csrc/`, `tests/test_solver_fast.py` and the switch
in `jax_step.py`.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import solver as _ref

BACKEND = os.environ.get("VULCAN_JAX_SOLVER", "fast")
if BACKEND not in ("fast", "ffi"):
    raise ValueError(
        f"VULCAN_JAX_SOLVER={BACKEND!r}: expected 'fast' (default), 'ffi', "
        "or 'reference' (handled in jax_step)"
    )


class Factors(NamedTuple):
    """Primal LU factors and permutations (no tangent: computed on stop_gradient
    inputs) plus the bands they factor (these carry the tangents)."""

    diag_lu: jnp.ndarray
    diag_perm: jnp.ndarray
    diag: jnp.ndarray
    sup_d: jnp.ndarray
    sub_d: jnp.ndarray


def _matvec(diag, sup_d, sub_d, x):
    out = jnp.einsum("zij,zj->zi", diag, x)
    out = out.at[:-1].add(sup_d * x[1:])
    return out.at[1:].add(sub_d * x[:-1])


def _raw_factor(diag, sup_d, sub_d):
    if BACKEND == "ffi":
        return _ffi_factor(diag, sup_d, sub_d)
    f = _ref.factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)
    return f.diag_lu, f.diag_perm


def _raw_solve(lu, perm, sup_d, sub_d, rhs):
    if BACKEND == "ffi":
        return _ffi_solve(lu, perm, sup_d, sub_d, rhs)
    return _ref.solve_block_thomas_diag_offdiag(
        _ref.BlockThomasDiagFactors(lu, perm, sup_d, sub_d), rhs
    )


def factor(diag, sup_d, sub_d) -> Factors:
    sg = jax.lax.stop_gradient
    lu, perm = _raw_factor(sg(diag), sg(sup_d), sg(sub_d))
    return Factors(lu, perm, diag, sup_d, sub_d)


def solve(factors: Factors, rhs, matvec=None):
    """`matvec` is the operator as a function of x. The primal never runs it;
    the AD rules do (the tangent's `dA x`, the transpose's cotangent). The
    default is the dense bands in `factors`, whose tangent is the whole dense
    `dA` per direction; `jax_step._ros2_stages` passes a matrix-free one
    (notes §2.9)."""
    lu, perm, diag, sup_d, sub_d = factors
    sg = jax.lax.stop_gradient
    sup0, sub0 = sg(sup_d), sg(sub_d)

    if matvec is None:
        matvec = functools.partial(_matvec, diag, sup_d, sub_d)

    def solve_(_matvec, b):
        return _raw_solve(lu, perm, sup0, sub0, b)

    def transpose_solve(_vecmat, c):
        # The exact transpose of the primal sweep on the SAME factors: this is
        # what differentiating through the LU gives today for the rhs cotangent
        # (bit-identical for `fast`; componentwise backward error ~1e-14 on real
        # blocks, notes §1.4.1) and it costs no second factorisation. The FFI
        # call has no transpose rule, so the sweep is always the reference JAX
        # scan here, run on whichever factors the backend produced (same LU and
        # permutation layout). A fresh factorisation of A^T instead (eta ~1e-5)
        # made the W39b steady-state sensitivity LGMRES stagnate (§1.4.1).
        def sweep(b):
            return _ref.solve_block_thomas_diag_offdiag(
                _ref.BlockThomasDiagFactors(lu, perm, sup0, sub0), b
            )

        _, vjp = jax.vjp(sweep, c)
        return vjp(c)[0]

    return jax.lax.custom_linear_solve(
        matvec, rhs, solve_, transpose_solve=transpose_solve
    )


# --- C++ CPU kernel and its CUDA twin ---------------------------------------

_CSRC = Path(__file__).with_name("csrc")
_LIB = _CSRC / (
    "libblock_thomas_cpu" + (".dylib" if platform.system() == "Darwin" else ".so")
)
_CUDA_LIB = _CSRC / "libblock_thomas_cuda.so"
_HASH_TAG = b"vulcan-bt-source-sha256:"  # VulcanBtSourceHash, block_thomas_common.h


def _source_hash(src: Path) -> str:
    """sha256 of a kernel source and the header it includes."""
    h = hashlib.sha256(src.read_bytes())
    h.update((_CSRC / "block_thomas_common.h").read_bytes())
    return h.hexdigest()


def _is_current(lib: Path, src: Path) -> bool:
    """True if `lib` embeds the hash of the present sources. The file is read,
    not loaded: dlopen would keep a stale library cached under its path."""
    return lib.exists() and _HASH_TAG + _source_hash(src).encode() in lib.read_bytes()


def _nvcc_arch() -> str:
    """$NVCC_ARCH, else sm_XY of JAX's first CUDA device, else of the host's
    first GPU per nvidia-smi (a build shell may hide the GPU from JAX, e.g.
    with JAX_PLATFORMS=cpu)."""
    if arch := os.environ.get("NVCC_ARCH"):
        return arch
    try:
        cc = jax.devices("cuda")[0].compute_capability
    except RuntimeError:
        try:
            cc = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.split()[0]
        except (OSError, subprocess.CalledProcessError, IndexError) as e:
            raise RuntimeError(
                "no CUDA device to build the kernel for; set NVCC_ARCH (sm_90 on a GH200)"
            ) from e
    return "sm_" + cc.replace(".", "")


def build(force: bool = False, cuda: bool = False) -> Path:
    """Compile the CPU kernel with the system C++ compiler ($CXX or c++), or
    with `cuda=True` the GPU twin with nvcc ($NVCC, arch from `_nvcc_arch`).
    Returns the library at once if it embeds the hash of the present sources,
    unless `force`."""
    src = _CSRC / ("block_thomas_cuda.cu" if cuda else "block_thomas_cpu.cc")
    lib = _CUDA_LIB if cuda else _LIB
    if not force and _is_current(lib, src):
        return lib
    # -isystem: the XLA headers' own warnings are not reported. No -Werror: the
    # CPU build runs at import, where a newer compiler's warning must not fail.
    flags = ["-O2", "-std=c++17", "-shared", f"-DVULCAN_BT_SOURCE_HASH={_source_hash(src)}",
             "-isystem", jax.ffi.include_dir()]
    if cuda:
        cmd = [os.environ.get("NVCC", "nvcc"), *flags, f"-arch={_nvcc_arch()}", "-lineinfo",
               "-Xcompiler", "-fPIC,-Wall,-Wextra"]
    else:
        cmd = [os.environ.get("CXX", "c++"), *flags, "-fPIC", "-Wall", "-Wextra"]
        if platform.system() == "Darwin":
            cmd += ["-undefined", "dynamic_lookup"]
    # Compile in a private directory beside the library and rename it into
    # place, so a concurrent first build never loads a half-written file.
    tmp_dir = tempfile.mkdtemp(dir=lib.parent)
    try:
        tmp = os.path.join(tmp_dir, lib.name)
        subprocess.run(cmd + [str(src), "-o", tmp], check=True)
        os.replace(tmp, lib)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return lib


_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return
    # The CUDA twin is built only by `--cuda`, never here, so a stale one is
    # refused rather than loaded (`build` rebuilds the CPU kernel).
    cu_src = _CSRC / "block_thomas_cuda.cu"
    if _CUDA_LIB.exists() and not _is_current(_CUDA_LIB, cu_src):
        raise RuntimeError(
            f"{_CUDA_LIB} was not built from the present {cu_src.name} and "
            "block_thomas_common.h; rebuild it on the GPU host with `python -m "
            "vulcan_jax.solver_fast --cuda`, or delete it to run the CPU kernel only."
        )
    lib = ctypes.CDLL(str(build()))
    jax.ffi.register_ffi_target(
        "vulcan_bt_factor", jax.ffi.pycapsule(lib.VulcanBtFactor), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "vulcan_bt_solve", jax.ffi.pycapsule(lib.VulcanBtSolve), platform="cpu"
    )
    if _CUDA_LIB.exists():
        gpu = ctypes.CDLL(str(_CUDA_LIB))
        jax.ffi.register_ffi_target(
            "vulcan_bt_factor", jax.ffi.pycapsule(gpu.VulcanBtFactorCuda), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "vulcan_bt_solve", jax.ffi.pycapsule(gpu.VulcanBtSolveCuda), platform="CUDA"
        )
    _REGISTERED = True


def _ffi_factor(diag, sup_d, sub_d):
    _register()
    out = (
        jax.ShapeDtypeStruct(diag.shape, diag.dtype),
        jax.ShapeDtypeStruct(diag.shape[:-1], jnp.int32),
    )
    return jax.ffi.ffi_call("vulcan_bt_factor", out, vmap_method="broadcast_all")(
        diag, sup_d, sub_d
    )


def _ffi_solve(lu, perm, sup_d, sub_d, rhs):
    # expand_dims, not broadcast_all: at a vmap level where only `rhs` is
    # batched (the six tangent directions share one factorisation) the factors
    # get a size-1 axis instead of one copy per direction, and the handler
    # broadcasts their leading dimensions against the rhs's. On the GPU that is
    # 3.9 MB of `lu` per lane not copied six times per step.
    _register()
    return jax.ffi.ffi_call(
        "vulcan_bt_solve",
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        vmap_method="expand_dims",
    )(lu, perm, sup_d, sub_d, rhs)


if __name__ == "__main__":
    print(build(force="--force" in sys.argv, cuda="--cuda" in sys.argv))
