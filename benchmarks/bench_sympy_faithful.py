#!/usr/bin/env python
"""Benchmark the SymPy-faithful RHS change against ``origin/main``.

The script creates a temporary detached worktree at ``origin/main`` and
runs the same commands in the current tree and the baseline tree:

* ``python tests/profile_step.py`` for RHS/Jacobian/solver breakdowns.
* ``python benchmarks/bench_step.py`` for Ros2-step and short outer-loop
  timings.
* optionally ``python vulcan_jax.py`` for a full configured run.

Output is plain ``key=value`` lines plus slowdown ratios so results can be
pasted into audit notes without post-processing.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METRIC_RE = re.compile(r"^([A-Za-z0-9_]+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
TOTAL_WALL_RE = re.compile(r"Total wall time:\s*([0-9.eE+-]+)s")


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    cache_dir: Path,
    timeout: float,
    allow_failure: bool = False,
) -> tuple[str, float, int]:
    """Run *cmd* in *cwd* with an isolated JAX cache.

    Returns combined output, wall time, and return code. ``allow_failure`` is
    used for benchmark scripts that may fail only in optional master-oracle
    sections after printing the JAX metrics this audit needs.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.time() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed in {cwd} with exit {proc.returncode}: {' '.join(cmd)}\n"
            f"--- output ---\n{output[-4000:]}"
        )
    return output, wall, proc.returncode


def _parse_key_values(output: str) -> dict[str, float]:
    """Parse ``key=value`` metric lines from benchmark output."""
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        match = METRIC_RE.match(line.strip())
        if match is not None:
            metrics[match.group(1)] = float(match.group(2))
    wall_match = TOTAL_WALL_RE.search(output)
    if wall_match is not None:
        metrics["full_run_reported_wall_seconds"] = float(wall_match.group(1))
    return metrics


def _bench_tree(
    root: Path,
    *,
    label: str,
    cache_parent: Path,
    timeout: float,
    full_run: bool,
) -> dict[str, float]:
    """Run the audit benchmark commands in one tree."""
    metrics: dict[str, float] = {}
    cache_dir = cache_parent / f"jax_cache_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    profile_output, profile_wall, profile_rc = _run_command(
        [sys.executable, "tests/profile_step.py"],
        cwd=root,
        cache_dir=cache_dir,
        timeout=timeout,
    )
    metrics.update(_parse_key_values(profile_output))
    metrics["profile_step_wall_seconds"] = profile_wall
    metrics["profile_step_returncode"] = float(profile_rc)

    bench_output, bench_wall, bench_rc = _run_command(
        [sys.executable, "benchmarks/bench_step.py"],
        cwd=root,
        cache_dir=cache_dir,
        timeout=timeout,
        allow_failure=True,
    )
    metrics.update(_parse_key_values(bench_output))
    metrics["bench_step_wall_seconds"] = bench_wall
    metrics["bench_step_returncode"] = float(bench_rc)

    if full_run:
        full_output, full_wall, full_rc = _run_command(
            [sys.executable, "vulcan_jax.py"],
            cwd=root,
            cache_dir=cache_dir,
            timeout=timeout,
        )
        metrics.update(_parse_key_values(full_output))
        metrics["full_run_wall_seconds"] = full_wall
        metrics["full_run_returncode"] = float(full_rc)

    return metrics


def _add_origin_worktree(parent: Path, ref: str) -> Path:
    """Create a detached worktree for *ref* under *parent*."""
    worktree = parent / "origin_main_worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), ref],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    _copy_optional_runtime_assets(worktree)
    return worktree


def _copy_optional_runtime_assets(worktree: Path) -> None:
    """Copy ignored runtime assets that benchmarks need but git may omit."""
    for rel_path in ("fastchem_vulcan/output",):
        src = ROOT / rel_path
        dst = worktree / rel_path
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _remove_worktree(path: Path) -> None:
    """Remove a temporary git worktree if it still exists."""
    if path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )


def _metric(metrics: dict[str, float], *names: str) -> float:
    """Return the first available metric from *names*, else NaN."""
    for name in names:
        if name in metrics:
            return metrics[name]
    return math.nan


def _ratio(numer: float, denom: float) -> float:
    """Safe ratio helper."""
    if not math.isfinite(numer) or not math.isfinite(denom) or denom == 0.0:
        return math.nan
    return numer / denom


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"[{label}]")
    for key in sorted(metrics):
        print(f"{label}_{key}={metrics[key]:.6g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-ref",
        default="origin/main",
        help="Git ref to use as the pre-SymPy-faithful baseline.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds for each benchmark command.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Also run `python vulcan_jax.py` in each tree.",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Leave the temporary origin worktree on disk for inspection.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="vulcan_jax_sympy_bench_") as tmp:
        tmp_path = Path(tmp)
        origin_tree = _add_origin_worktree(tmp_path, args.origin_ref)
        try:
            current = _bench_tree(
                ROOT,
                label="current",
                cache_parent=tmp_path,
                timeout=args.timeout,
                full_run=args.full_run,
            )
            origin = _bench_tree(
                origin_tree,
                label="origin",
                cache_parent=tmp_path,
                timeout=args.timeout,
                full_run=args.full_run,
            )
            _print_metrics("current", current)
            _print_metrics("origin", origin)

            current_rhs = _metric(current, "chem_rhs_codegen_ms", "chem_rhs_ms")
            origin_rhs = _metric(origin, "chem_rhs_ms")
            current_segment = _metric(current, "chem_rhs_segment_sum_reference_ms")
            current_step = _metric(current, "jax_ros2_step_ms_per_step")
            origin_step = _metric(origin, "jax_ros2_step_ms_per_step")
            current_outer = _metric(current, "outer_loop_per_step_ms_cached")
            origin_outer = _metric(origin, "outer_loop_per_step_ms_cached")
            current_full = _metric(
                current, "full_run_reported_wall_seconds", "full_run_wall_seconds"
            )
            origin_full = _metric(
                origin, "full_run_reported_wall_seconds", "full_run_wall_seconds"
            )

            print("[slowdown]")
            print(f"rhs_current_vs_origin={_ratio(current_rhs, origin_rhs):.6g}")
            print(
                "rhs_codegen_vs_current_segment_sum_reference="
                f"{_ratio(current_rhs, current_segment):.6g}"
            )
            print(f"ros2_step_current_vs_origin={_ratio(current_step, origin_step):.6g}")
            print(f"outer_loop_current_vs_origin={_ratio(current_outer, origin_outer):.6g}")
            if args.full_run:
                print(f"full_run_current_vs_origin={_ratio(current_full, origin_full):.6g}")
        finally:
            if args.keep_worktree:
                print(f"kept_origin_worktree={origin_tree}")
            else:
                _remove_worktree(origin_tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
