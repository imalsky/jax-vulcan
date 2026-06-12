"""Standalone GPU throughput benchmark for VULCAN-JAX.

This script integrates a batch of HD 189-like planets to steady state in ONE
vmapped device call (`OuterLoop.run_batch`, the vmap-across-profiles path).
Each planet is the vendored HD189 atmosphere (T-P + Kzz file) with its
temperature profile scaled by a few percent, initialised from FastChem
equilibrium, photochemistry off (the emulator data-generation regime this
benchmark measures; `run_batch` also supports photo-on batches). That
workload converges in ~600 accepted Ros2 steps (~50 s single-profile on a
laptop CPU), so the default job is a few minutes, not an hour.

Design (items 1-4 were problems in the first GH200 run; item 5 came from
its batch-512 OOM analysis):

1.  **Profiles actually converge.** The real HD189 EQ-init regime needs only
    ~600 steps; the step cap defaults to 2500 (4x margin). The first GH200
    run used a synthetic cold const_mix CH4+O2 column that needs ~4100 steps
    to burn through the combustion transient, with a 2000-step cap — so
    every lane terminated on the step cap (0/512 "converged").
2.  **Parallel host setup.** RunStates are built on a spawn `ProcessPool`
    pinned to CPU (the same pattern as the emulator's GPU-batched generation),
    one task per planet, using every core PBS gives the job. Each worker gets
    a private copy of the package's vendored FastChem tree via
    `$VULCAN_JAX_FASTCHEM_DIR` so the cross-process flock never serialises
    the pool.
3.  **The GPU never waits for the CPU.** Builds for ALL batch sizes are
    submitted up front; the GPU starts integrating the smallest batch as soon
    as its planets are ready while the pool keeps building the rest.
4.  **Live progress.** The batched integration is driven in chunks (default
    250 accepted steps per device call) via the carry's `chunk_target` yield,
    so a timestamped status line (lanes done, step counts, reason breakdown)
    appears every chunk instead of one table after a long silence.
5.  **Device-batch tiling.** The on-device batch is capped (default 128 via
    `--device-batch`); larger sweep batches run as sequential sub-tiles
    sharing one XLA compile. The vmapped Jacobian-assembly transient grows
    linearly in the on-device batch and OOM'd an untiled batch 512 on a
    96 GB GH200; per-planet throughput was already near-saturated by 128,
    so 4x128 tiles recover essentially all of the 512-wide throughput.
    Per-lane results are unchanged (lanes never interact).

It is standalone: it imports only `vulcan_jax`, the stdlib, NumPy, and JAX —
no sibling repos, no `../VULCAN-master/`; the atmosphere file and the FastChem
binary both ship inside the installed package.

------------------------------------------------------------------------------
RUN IT
------------------------------------------------------------------------------
After `pip install vulcan-jax` (>= 0.1.13, nothing else):

    # Quick test: 2 HD189-like planets (CPU on a laptop, GPU if visible).
    python gpu_benchmark.py --batches 2

    # Force the GPU backend (errors early if no GPU is visible to JAX):
    JAX_PLATFORM_NAME=gpu python gpu_benchmark.py

On the GH200 / NAS (PBS):  qsub supercomputer_cmds/run_gpu_benchmark.pbs
On the edge A100 (SLURM):  sbatch supercomputer_cmds/run_gpu_benchmark.sh
Both stream this script's output to a tail-able log file; the .sh variant
adds an untiled Fix-B memory probe after the tiled sweep. For the paper's
throughput-vs-batch-size sweep:  qsub -v BATCHES="8 64 256 512" ... /
sbatch --export=ALL,BATCHES="8 64 256 512" ...

The "profiles/s" at the largest batch size is the number to watch: the GPU
amortizes a fixed per-call overhead across the whole batch, so throughput
should climb with batch size until the device saturates. On CPU it will be
roughly flat (vmap lanes time-share a few cores).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import multiprocessing
import os
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_T0 = time.perf_counter()

# Per-lane termination_reason codes set by run_batch's freeze-on-done body.
_REASON_NAMES = {
    1: "converged",
    2: "runtime",
    3: "step-cap",
    4: "stalled",
    5: "non-finite",
}
# Reasons that count as a profile genuinely reaching steady state: 1 is the
# normal yconv/slope criterion, 4 is master's stall fallback (end_case=1 too).
_OK_REASONS = (1, 4)


def log(msg: str) -> None:
    """Timestamped, flushed progress line (PBS-friendly: shows up live)."""
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config: a fast, fully-standalone, photo-off batched regime.
# ---------------------------------------------------------------------------
_HD189_ATM = "atm/atm_HD189_Kzz.txt"  # vendored in the vulcan_jax package


def build_cfg(nz: int, count_max: int):
    """The real HD189 regime minus photochemistry: vendored T-P + Kzz file
    (cfg defaults), FastChem-EQ init, photo off (the emulator regime).

    `count_max` must comfortably exceed the ~600 accepted steps this regime
    needs to converge (measured single-profile).
    """
    import vulcan_jax

    return vulcan_jax.make_config(
        ini_mix="EQ",  # vendored-FastChem equilibrium, like the real HD189 config
        # atm_type/atm_file/Kzz_prof stay at the package defaults ('file',
        # the vendored HD189 atm, Kzz from the same file); each worker swaps
        # in its own T-scaled copy of that file per planet.
        use_photo=False,  # photo-off: the emulator data-generation regime
        use_ion=False,
        use_condense=False,
        nz=nz,
        count_max=count_max,
        # count_min stays at the master default (120): EQ-initialised columns
        # can pass the longdy test in a few dozen steps, before transport has
        # had any simulated time to drive them away from equilibrium.
        use_print_prog=False,
        use_live_plot=False,
        use_live_flux=False,
        use_save_movie=False,
        use_flux_movie=False,
        use_chunked_runner=False,
    )


def make_tscales(n: int, spread: float, seed: int) -> np.ndarray:
    """Per-planet temperature-profile scale factors: 1 +/- `spread` plus a
    little jitter, so the batch is a real heterogeneous workload (HD189-like
    planets, not n copies of HD189)."""
    if n == 1:
        return np.array([1.0])
    rng = np.random.default_rng(seed)
    scales = 1.0 + spread * np.linspace(-1.0, 1.0, n)
    return scales * (1.0 + 0.005 * rng.standard_normal(n))


# ---------------------------------------------------------------------------
# Host-setup worker pool (spawn, CPU-pinned). Same pattern as the emulator's
# GPU-batched generation: spawn (not fork) because the parent has CUDA
# initialised; CPU-pinned so 70+ workers never touch the H100; each worker
# gets a private FastChem tree so the package's cross-process fcntl.flock
# never serialises the pool.
# ---------------------------------------------------------------------------
_WORKER_CFG = None  # set once per worker by _worker_init
_WORKER_DIR = None  # private scratch dir (FastChem tree + scaled atm files)
_WORKER_ATM = None  # parsed (header_lines, P, T, Kzz) of the vendored HD189 atm


def _seed_private_fastchem_tree(worker_root) -> None:
    """Point $VULCAN_JAX_FASTCHEM_DIR at a private copy of the package's
    fastchem_vulcan/ tree (binary + input data). Must run BEFORE the first
    vulcan_jax import — the env var is read at import time (>= 0.1.10)."""
    import importlib.util
    import shutil
    from pathlib import Path

    spec = importlib.util.find_spec("vulcan_jax")
    package_root = Path(spec.origin).parent
    fastchem_root = worker_root / "fastchem_vulcan"
    shutil.copytree(
        package_root / "fastchem_vulcan",
        fastchem_root,
        ignore=shutil.ignore_patterns(
            "obj", "output", ".fastchem_lock", "*.pyc", "__pycache__"
        ),
    )
    (fastchem_root / "output").mkdir(exist_ok=True)
    os.environ["VULCAN_JAX_FASTCHEM_DIR"] = str(fastchem_root)


def _worker_init(nz: int, count_max: int) -> None:
    """Pin this worker to CPU before its first vulcan_jax import, seed its
    private FastChem tree, and parse the vendored HD189 atm table once."""
    import logging
    import tempfile
    from pathlib import Path

    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_ENABLE_X64"] = "1"
    # These workers are CPU-only, but JAX still *discovers* the installed CUDA
    # plugin and calls cuInit(0), which fails with CUDA_ERROR_NO_DEVICE (no GPU
    # visible) and is logged at ERROR level. Mute xla_bridge BEFORE the first
    # vulcan_jax/jax import so that per-worker traceback never reaches the log.
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    worker_root = Path(tempfile.mkdtemp(prefix=f"vjax_bench_{os.getpid()}_"))
    _seed_private_fastchem_tree(worker_root)

    global _WORKER_CFG, _WORKER_DIR, _WORKER_ATM
    _WORKER_CFG = build_cfg(nz, count_max)  # first vulcan_jax import
    _WORKER_DIR = worker_root

    from vulcan_jax._paths import resolve_data_path

    atm_path = resolve_data_path(_HD189_ATM)
    lines = atm_path.read_text().splitlines()
    table = np.genfromtxt(atm_path, names=True, dtype=None, skip_header=1)
    _WORKER_ATM = (lines[:2], table["Pressure"], table["Temp"], table["Kzz"])


def _worker_build(task):
    """Build one HD189-like planet's RunState (pickled back to the parent).

    Writes a private copy of the HD189 atm file with the temperature column
    scaled by `tscale` (Kzz and the pressure grid unchanged) and points the
    cfg at it — an HD189-like planet, a few percent hotter or cooler.

    Stdout/stderr are silenced at the OS fd level so 512 repetitions of the
    FastChem / init chatter don't drown the parent's progress log — the
    FastChem binary is a subprocess that inherits the fds, so a Python-level
    redirect_stdout would not catch it.
    """
    import copy

    import vulcan_jax

    idx, tscale = task
    header, P, T, Kzz = _WORKER_ATM
    atm_path = _WORKER_DIR / f"atm_HD189_planet{idx}.txt"
    rows = "\n".join(
        f"{p:.6E}\t{t * float(tscale):.6f}\t {k:.6E}" for p, t, k in zip(P, T, Kzz)
    )
    atm_path.write_text("\n".join(header) + "\n" + rows + "\n")

    cfg = copy.copy(_WORKER_CFG)
    cfg.atm_file = str(atm_path)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        rs = vulcan_jax.RunState.with_pre_loop_setup(cfg, skip_chem_warmup=True)
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
    return idx, rs


def _resolve_workers(requested: int, n_jobs: int) -> int:
    """Pool size: explicit > $PBS-visible cores > os.cpu_count, capped at n_jobs."""
    n = requested
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:  # macOS
            n = os.cpu_count() or 1
    return max(1, min(n, n_jobs))


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------
def build_integ(cfg):
    """Construct the OuterLoop runner (Ros2 solver + output sink)."""
    import vulcan_jax.legacy_io as legacy_io
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    return outer_loop.OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(cfg=cfg), cfg=cfg)


def stack_batch(integ, run_states):
    """prepare_runstate each profile, then stack into one batched
    (init_state, atm_static) pair on the device."""
    import vulcan_jax.outer_loop as outer_loop

    init_states, atm_statics = [], []
    for rs in run_states:
        init_state, atm_static = integ.prepare_runstate(rs)
        init_states.append(init_state)
        atm_statics.append(atm_static)
    return (
        outer_loop.stack_integ_states(init_states),
        outer_loop.stack_atm_statics(atm_statics),
    )


def _device_mem_stats():
    """Peak / current / limit device memory in GiB, or None when the backend
    has no allocator stats (CPU, some platforms).

    `peak_bytes_in_use` is cumulative for the process — XLA's allocator has no
    public reset — so within an ascending batch sweep the value after each
    batch is that batch's peak. This is the number that verifies the chunked
    Jacobian assembly actually bounds the vmap transient on device.
    """
    import jax

    stats = getattr(jax.devices()[0], "memory_stats", lambda: None)()
    if not stats:
        return None
    gib = 2**30
    return {
        "peak": stats.get("peak_bytes_in_use", 0) / gib,
        "in_use": stats.get("bytes_in_use", 0) / gib,
        "limit": stats.get("bytes_limit", 0) / gib,
    }


def print_backend_banner():
    """Print the JAX backend + devices and a clear CPU-fallback note."""
    import jax

    backend = jax.default_backend()
    devices = jax.devices()
    print("=" * 70)
    print("VULCAN-JAX GPU throughput benchmark")
    print("=" * 70)
    try:
        import vulcan_jax

        print(f"vulcan-jax version : {vulcan_jax.__version__}")
    except Exception:
        pass
    print(f"jax version        : {jax.__version__}")
    print(f"jax backend        : {backend}")
    print(f"jax devices        : {devices}")
    mem = _device_mem_stats()
    if mem and mem["limit"] > 0:
        print(f"device memory      : {mem['limit']:.1f} GiB visible to XLA")
    on_gpu = backend in ("gpu", "cuda", "rocm")
    if not on_gpu:
        print("")
        print(">> Running on CPU. This benchmark measures the vmap-across-profiles")
        print(">> path that the GPU is for; on CPU the lanes time-share a few cores,")
        print(">> so throughput will be roughly flat in batch size.")
        print(">> To use a GPU:  JAX_PLATFORM_NAME=gpu python gpu_benchmark.py")
    else:
        print("")
        print(f">> Running on the {backend.upper()} backend. Throughput should climb")
        print(">> with batch size until the device saturates.")
    print("=" * 70, flush=True)
    return on_gpu


def _reason_breakdown(reasons: np.ndarray) -> str:
    parts = []
    for code, name in _REASON_NAMES.items():
        c = int((reasons == code).sum())
        if c:
            parts.append(f"{name}={c}")
    running = int((reasons == 0).sum())
    if running:
        parts.append(f"running={running}")
    return " ".join(parts) if parts else "none"


def integrate_chunked(integ, states_b, atm_b, n, count_max, chunk, label):
    """Drive `run_batch` to per-lane termination in `chunk`-step device calls.

    Each call returns when every live lane has either really terminated
    (converged / runtime / step cap, recorded in `termination_reason`) or
    advanced `chunk` accepted steps past its previous yield — the carry's
    `chunk_target` mechanism. Finished lanes stay frozen, so the final state
    of each lane is identical to one uninterrupted `run_batch` call; the
    chunking only buys the host a place to log progress.

    Returns (final_state, total_s, call_times).
    """
    import jax
    import jax.numpy as jnp

    cap = jnp.int32(count_max + 1)  # matches the single-profile chunked driver
    t_start = time.perf_counter()
    call_times = []
    while True:
        states_b = states_b._replace(
            chunk_target=jnp.minimum(states_b.accept_count + jnp.int32(chunk), cap)
        )
        t0 = time.perf_counter()
        states_b = integ.run_batch(states_b, atm_b)
        jax.block_until_ready(states_b)
        call_s = time.perf_counter() - t0
        call_times.append(call_s)

        done = np.asarray(states_b.is_done)
        steps = np.asarray(states_b.accept_count)
        reasons = np.asarray(states_b.termination_reason)
        log(
            f"{label}: {int(done.sum())}/{n} lanes done | steps "
            f"min/med/max {int(steps.min())}/{int(np.median(steps))}/"
            f"{int(steps.max())} of {count_max} | {_reason_breakdown(reasons)} | "
            f"chunk {call_s:.1f}s"
        )
        if bool(done.all()):
            break
    return states_b, time.perf_counter() - t_start, call_times


def benchmark_one(integ, run_states, count_max, chunk, label, device_batch):
    """Integrate every lane of one sweep batch to termination, tiled host-side.

    The on-device batch is capped at `device_batch` lanes: a sweep batch
    larger than that is split into equal sub-tiles integrated sequentially,
    each in its own `run_batch` device call. All full tiles share one XLA
    compile (same (nz, tile) shape); a final partial tile is padded with
    copies of planet 0 (padded lanes are integrated but excluded from the
    stats) so it reuses the same compile too. Per-lane results are identical
    to the untiled call — lanes never interact — so tiling only bounds the
    device's peak live memory, which is what OOM'd batch 512 on a 96 GB
    GH200 (the vmapped Jacobian-assembly transient grows linearly in the
    on-device batch).

    Returns a result dict for the summary table.
    """
    n = len(run_states)
    tile = min(n, max(1, device_batch))
    n_tiles = -(-n // tile)

    steps_parts, reasons_parts = [], []
    call_times: list = []
    total_s = 0.0
    finite = True
    for t in range(n_tiles):
        lane_states = run_states[t * tile : (t + 1) * tile]
        n_real = len(lane_states)
        if n_real < tile:
            lane_states = lane_states + [run_states[0]] * (tile - n_real)
        tlabel = label if n_tiles == 1 else f"{label} [tile {t + 1}/{n_tiles}]"

        t0 = time.perf_counter()
        states_b, atm_b = stack_batch(integ, lane_states)
        stack_s = time.perf_counter() - t0
        log(
            f"{tlabel}: stacked {len(lane_states)} planets onto the device in {stack_s:.1f}s"
        )

        states_b, tile_s, tile_calls = integrate_chunked(
            integ, states_b, atm_b, len(lane_states), count_max, chunk, tlabel
        )
        total_s += tile_s
        call_times.extend(tile_calls)

        steps_parts.append(np.asarray(states_b.accept_count)[:n_real])
        reasons_parts.append(np.asarray(states_b.termination_reason)[:n_real])
        finite = finite and bool(np.isfinite(np.asarray(states_b.ymix)[:n_real]).all())
        # Free this tile's device arrays before the next tile stacks.
        del states_b, atm_b
        gc.collect()

    steps = np.concatenate(steps_parts)
    reasons = np.concatenate(reasons_parts)
    n_ok = int(np.isin(reasons, _OK_REASONS).sum())

    # The first device call carries the one-time XLA compile for this
    # (nz, tile) shape; estimate it as that call's excess over the median
    # later call and report throughput with and without it.
    first_call_s = call_times[0]
    compile_est = (
        max(first_call_s - float(np.median(call_times[1:])), 0.0)
        if len(call_times) > 1
        else 0.0
    )
    mem = _device_mem_stats()
    result = {
        "n": n,
        "total_s": total_s,
        "first_call_s": first_call_s,
        "steady_profiles_s": n / max(total_s - compile_est, 1e-9),
        "profiles_s": n / total_s,
        "n_ok": n_ok,
        "reasons": _reason_breakdown(reasons),
        "steps_max": int(steps.max()),
        "finite": bool(finite),
        # Cumulative process peak; in an ascending sweep this is this batch's
        # peak. None on backends without allocator stats (CPU).
        "peak_gib": mem["peak"] if mem else None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone VULCAN-JAX GPU throughput benchmark "
        "(vmap-across-profiles run_batch, run to convergence).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[4],
        help="Batch sizes (number of HD189-like planets) to sweep.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=150,
        help="Vertical layers per planet (150 = the real HD189 config).",
    )
    parser.add_argument(
        "--count-max",
        type=int,
        default=2500,
        help="Max accepted integration steps per planet. Must exceed the "
        "~600 steps this regime needs, or lanes hit the step cap instead "
        "of converging.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=250,
        help="Accepted steps per device call (progress-log cadence).",
    )
    parser.add_argument(
        "--device-batch",
        type=int,
        default=128,
        help="Max planets per device call. Larger sweep batches are tiled "
        "host-side into sub-batches of this size (one shared XLA compile); "
        "per-lane results are unchanged. Bounds the vmapped Jacobian "
        "transient that OOM'd an untiled batch 512 on a 96 GB GH200. "
        "Set >= the largest batch to measure a single untiled device call.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Host-setup processes (0 = every core the job can see).",
    )
    parser.add_argument(
        "--t-spread",
        type=float,
        default=0.03,
        help="Fractional temperature-profile spread across the batch "
        "(each planet's HD189 T(P) is scaled by 1 +/- this).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # JAX must see x64 on; the package sets this at import, but make it explicit
    # for users who run the file directly without the package's import guard.
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    print_backend_banner()

    cfg = build_cfg(nz=args.nz, count_max=args.count_max)
    integ = build_integ(cfg)

    print(
        f"\nConfig: HD189-like planets (vendored {_HD189_ATM}, T(P) scaled "
        f"1±{args.t_spread:g})  ini_mix=EQ  use_photo=False  nz={args.nz}  "
        f"count_max={args.count_max}  chunk={args.chunk}  "
        f"device_batch={args.device_batch}"
    )
    print(f"Batch sizes to sweep: {args.batches}\n", flush=True)

    batches = sorted(set(args.batches))
    n_max = max(batches)
    tscales = make_tscales(n_max, float(args.t_spread), args.seed)
    n_workers = _resolve_workers(args.workers, n_max)
    log(f"host setup: building {n_max} planets on {n_workers} CPU workers")

    # Submit ALL profile builds up front; the GPU starts on the smallest batch
    # as soon as its profiles exist while the pool keeps building the rest.
    ctx = multiprocessing.get_context("spawn")
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(args.nz, args.count_max),
    )
    t_setup0 = time.perf_counter()
    futures = [pool.submit(_worker_build, (i, tscales[i])) for i in range(n_max)]

    built = {"n": 0}
    built_lock = threading.Lock()
    log_every = max(1, n_max // 8)

    def _on_built(_fut):
        with built_lock:
            built["n"] += 1
            k = built["n"]
        if k % log_every == 0 or k == n_max:
            log(f"host setup: {k}/{n_max} planets built")
            if k == n_max:
                log(
                    f"host setup: all planets done in "
                    f"{time.perf_counter() - t_setup0:.1f}s"
                )

    for f in futures:
        f.add_done_callback(_on_built)

    all_states: list = [None] * n_max
    results = []
    try:
        for bsz in batches:
            label = f"batch {bsz}"
            missing = [i for i in range(bsz) if all_states[i] is None]
            if missing:
                log(f"{label}: waiting for {len(missing)} planet builds")
            for i in range(bsz):
                if all_states[i] is None:
                    idx, rs = futures[i].result()
                    all_states[idx] = rs
            results.append(
                benchmark_one(
                    integ,
                    all_states[:bsz],
                    args.count_max,
                    args.chunk,
                    label,
                    args.device_batch,
                )
            )
            r = results[-1]
            peak = (
                f" | device peak {r['peak_gib']:.1f} GiB"
                if r["peak_gib"] is not None
                else ""
            )
            log(
                f"{label}: DONE in {r['total_s']:.1f}s "
                f"(first call {r['first_call_s']:.1f}s incl XLA compile) | "
                f"finished-ok {r['n_ok']}/{r['n']} | {r['reasons']}{peak}"
                + ("" if r["finite"] else " | [NON-FINITE ymix!]")
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    header = (
        f"{'batch':>6} | {'total s':>9} | {'1st call s':>10} | "
        f"{'prof/s':>8} | {'steady prof/s':>13} | {'peak GiB':>8} | "
        f"{'ok':>9} | reasons"
    )
    print("\n" + header)
    print("-" * (len(header) + 20))
    for r in results:
        flag = "" if r["finite"] else "  [NON-FINITE!]"
        peak = f"{r['peak_gib']:>8.1f}" if r["peak_gib"] is not None else f"{'--':>8}"
        print(
            f"{r['n']:>6} | {r['total_s']:>9.1f} | {r['first_call_s']:>10.1f} | "
            f"{r['profiles_s']:>8.3f} | {r['steady_profiles_s']:>13.3f} | "
            f"{peak} | "
            f"{r['n_ok']:>4}/{r['n']:<4} | {r['reasons']}{flag}"
        )

    if len(results) >= 2:
        base, best = results[0], max(results, key=lambda r: (r["profiles_s"], r["n"]))
        print(
            f"\nThroughput at batch {base['n']}: {base['profiles_s']:.3f} profiles/s"
            f"  ->  at batch {best['n']}: {best['profiles_s']:.3f} profiles/s"
            f"  ({best['profiles_s'] / max(base['profiles_s'], 1e-12):.1f}x)"
        )
    print(
        "\nNotes: every lane runs to its own termination (freeze-on-done), so "
        "'total s' is set by the slowest lane. '1st call s' includes the "
        "one-time XLA compile for that (nz, batch) shape; 'steady prof/s' "
        "excludes it. 'peak GiB' is the process-cumulative device-allocator "
        "peak (per-batch in an ascending sweep; '--' on backends without "
        "allocator stats). 'ok' counts converged + stalled-converged lanes "
        "(both are steady-state ends); 'step-cap' lanes need a higher "
        "--count-max.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
