"""`OuterLoop.run_queue`: the batched solve with lane refill from a job queue.

Five heterogeneous isothermal profiles (photo off) are the queue; `run_batch`
on the same five is the reference. Pins:

  1. NO REFILL — with `n_lanes >= n_jobs` nothing is refilled, so every job
     starts at tick 0 exactly as a plain-batch lane does and the result is
     bitwise `run_batch`'s.
  2. REFILL — a refilled lane enters the loop at a tick the plain batch never
     gave it, which moves the photo / geometry cadence but not the fixed
     point: same termination reason, mixing ratios equal at the convergence
     scale, and the call is deterministic.
  3. IDLE LANES — more lanes than jobs: the spare lane is frozen from the
     start and never writes a result.
  4. POISONED JOB — an all-NaN job exits with reason 5 and does not move its
     queue neighbours: with the same lane count and chunk their histories are
     the same with and without it appended, so their results are bitwise
     equal.
  5. LANE REUSE — the jobs refilled into the lane a poisoned job died in come
     back finite and on the batch's fixed point.
  6. ARGUMENT GUARDS — an empty queue and a lane count below 1 raise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_vmap_while_loop import (
    RTOL,
    _build_integ,
    _build_rs,
    _max_rel_diff,
    _pin_cfg,
)

# The bound test_vmap_while_loop holds batched-vs-solo to; a refilled lane
# differs from the plain batch for the same reason (cadence, not fixed point).
REL_MAX = RTOL
# Five isothermal columns: heterogeneous enough that the lanes finish at
# different iterations, so the queue really refills.
TISO = (900.0, 1000.0, 1100.0, 1200.0, 1300.0)


def _atm_array_fields():
    """AtmStatic fields that carry the batch axis (the four toggle flags are
    Python scalars and must never be indexed)."""
    from vulcan_jax.jax_step import AtmStatic
    from vulcan_jax.outer_loop import _ATM_STATIC_BATCH_AXES

    return tuple(
        f for f in AtmStatic._fields if getattr(_ATM_STATIC_BATCH_AXES, f) == 0
    )


def _take_jobs(jobs, sl):
    """Slice a (state, atm) job pair on the job axis."""
    st, atm = jobs
    return (
        jax.tree_util.tree_map(lambda x: x[sl], st),
        atm._replace(**{f: getattr(atm, f)[sl] for f in _atm_array_fields()}),
    )


def _cat_jobs(a, b):
    """Concatenate two (state, atm) job pairs on the job axis."""
    return (
        jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), a[0], b[0]
        ),
        a[1]._replace(
            **{
                f: jnp.concatenate([getattr(a[1], f), getattr(b[1], f)], axis=0)
                for f in _atm_array_fields()
            }
        ),
    )


def _poisoned(jobs, sl):
    """Job `sl` with y AND y_prev all-NaN: a rejected Ros2 step reverts y to
    y_prev, so poisoning y alone self-heals and the lane never goes
    non-finite (test_vmap_while_loop makes the same point)."""
    st, atm = _take_jobs(jobs, sl)
    return (
        st._replace(
            y=jnp.full_like(st.y, jnp.nan),
            y_prev=jnp.full_like(st.y_prev, jnp.nan),
        ),
        atm,
    )


def _ident(job):
    """`init_fn` for a queue whose jobs ARE the prepared (state, atm) pairs."""
    return job


def _out_fn(final):
    return (
        final.y,
        final.t,
        final.accept_count,
        final.termination_reason,
        final.is_done,
    )


@pytest.fixture(scope="module")
def prepared():
    """(integ, stacked states, stacked atmospheres) for the five profiles."""
    import vulcan_jax.outer_loop as outer_loop

    cfg = _pin_cfg()
    integ = _build_integ()
    pairs = [integ.prepare_runstate(_build_rs(cfg, Tiso=T)) for T in TISO]
    return (
        integ,
        outer_loop.stack_integ_states([s for s, _ in pairs]),
        outer_loop.stack_atm_statics([a for _, a in pairs]),
    )


def test_queue_without_refill_is_the_batch(prepared):
    integ, init_b, atm_b = prepared
    ref = integ.run_batch(init_b, atm_b)
    (y, t, acc, reason, done), n_iter = integ.run_queue(
        _ident, (init_b, atm_b), n_lanes=len(TISO), out_fn=_out_fn
    )
    print(f"[no refill] n_iter={int(n_iter)}", flush=True)
    for name, a, b in (
        ("y", y, ref.y),
        ("t", t, ref.t),
        ("accept_count", acc, ref.accept_count),
        ("termination_reason", reason, ref.termination_reason),
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
    assert bool(np.all(np.asarray(done)))


@pytest.mark.parametrize("n_lanes,chunk", [(2, 1), (2, 2), (3, 8)])
def test_queue_refills_and_matches_the_batch_at_the_convergence_scale(
    prepared, n_lanes, chunk
):
    integ, init_b, atm_b = prepared
    ref = integ.run_batch(init_b, atm_b)
    # The no-refill call is the lockstep baseline: with fewer lanes than jobs
    # the queue must take strictly more iterations, or nothing was refilled.
    _, n_iter_lockstep = integ.run_queue(
        _ident, (init_b, atm_b), n_lanes=len(TISO), out_fn=_out_fn
    )
    (y, _t, _acc, reason, done), n_iter = integ.run_queue(
        _ident, (init_b, atm_b), n_lanes=n_lanes, out_fn=_out_fn, chunk=chunk
    )
    assert int(n_iter) > int(n_iter_lockstep)
    assert np.array_equal(np.asarray(reason), np.asarray(ref.termination_reason))
    assert bool(np.all(np.asarray(done)))
    for k in range(len(TISO)):
        yk = np.asarray(y[k])
        yr = np.asarray(ref.y[k])
        rel = _max_rel_diff(
            yk / yk.sum(axis=1, keepdims=True), yr / yr.sum(axis=1, keepdims=True)
        )
        print(
            f"[lanes={n_lanes} chunk={chunk} n_iter={int(n_iter)} "
            f"(lockstep {int(n_iter_lockstep)}) job {k}] "
            f"max rel {rel:.3e}",
            flush=True,
        )
        assert rel < REL_MAX
    # A second call must land on the same numbers.
    (y2, *_), _ = integ.run_queue(
        _ident, (init_b, atm_b), n_lanes=n_lanes, out_fn=_out_fn, chunk=chunk
    )
    assert np.array_equal(np.asarray(y), np.asarray(y2))


def test_more_lanes_than_jobs(prepared):
    integ, init_b, atm_b = prepared
    three = _take_jobs((init_b, atm_b), slice(0, 3))
    ref = integ.run_batch(*three)
    (y, *_), _ = integ.run_queue(_ident, three, n_lanes=4, out_fn=_out_fn)
    assert np.array_equal(np.asarray(y), np.asarray(ref.y))


def test_poisoned_job_does_not_touch_its_neighbours(prepared):
    integ, init_b, atm_b = prepared
    abc = _take_jobs((init_b, atm_b), slice(0, 3))
    abcp = _cat_jobs(abc, _poisoned((init_b, atm_b), slice(3, 4)))
    (y3, _, _, r3, _), _ = integ.run_queue(
        _ident, abc, n_lanes=2, out_fn=_out_fn, chunk=1
    )
    (y4, _, acc4, r4, _), _ = integ.run_queue(
        _ident, abcp, n_lanes=2, out_fn=_out_fn, chunk=1
    )
    print(
        f"[poison] reasons without P {np.asarray(r3).tolist()}, with P "
        f"{np.asarray(r4).tolist()}, P accept_count {int(acc4[3])}",
        flush=True,
    )
    assert np.array_equal(np.asarray(y3), np.asarray(y4[:3]))
    assert int(r4[3]) == 5 and int(acc4[3]) == 0


def test_lane_reuse_after_a_poisoned_job(prepared):
    """One lane, the poisoned job FIRST: A and B are refilled into the lane P
    died in. A lane left holding P's non-finite state would carry the NaNs
    into the jobs that follow it. A and B enter at a later tick than the
    plain batch gives them, so they agree at the convergence scale."""
    integ, init_b, atm_b = prepared
    ab = _take_jobs((init_b, atm_b), slice(0, 2))
    pab = _cat_jobs(_poisoned((init_b, atm_b), slice(3, 4)), ab)
    ref = integ.run_batch(*ab)
    (y, _t, acc, reason, _done), n_iter = integ.run_queue(
        _ident, pab, n_lanes=1, out_fn=_out_fn, chunk=1
    )
    assert int(reason[0]) == 5 and int(acc[0]) == 0
    assert np.array_equal(np.asarray(reason[1:]), np.asarray(ref.termination_reason))
    for k in range(2):
        yk, yr = np.asarray(y[k + 1]), np.asarray(ref.y[k])
        assert np.all(np.isfinite(yk)), k
        rel = _max_rel_diff(
            yk / yk.sum(axis=1, keepdims=True), yr / yr.sum(axis=1, keepdims=True)
        )
        print(
            f"[lane reuse] job {k + 1} max rel {rel:.3e} (n_iter {int(n_iter)})",
            flush=True,
        )
        assert rel < REL_MAX


def test_empty_queue_and_lane_count_below_one_raise(prepared):
    integ, init_b, atm_b = prepared
    with pytest.raises(ValueError):
        integ.run_queue(
            _ident, _take_jobs((init_b, atm_b), slice(0, 0)), n_lanes=2,
            out_fn=_out_fn,
        )
    with pytest.raises(ValueError):
        integ.run_queue(
            _ident, _take_jobs((init_b, atm_b), slice(0, 2)), n_lanes=0,
            out_fn=_out_fn,
        )
