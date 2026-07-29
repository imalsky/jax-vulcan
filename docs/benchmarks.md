# Benchmarks

Where the per-step time goes, and which optimizations account for the speedup
over VULCAN 2.0. Moved out of the README in 2026-07.

Run the benchmark on your own machine:

```bash
python benchmarks/bench_step.py
```

Absolute times are hardware- and version-dependent. The numbers below are from
one reference CPU host, single-threaded, `jax==0.6.2`, float64, on HD 189733b.
The relative shares are robust across hosts even where the absolute times are
not.

## Per-step timing

| Step | VULCAN 2.0 (NumPy) | VULCAN-JAX | Speedup |
|---|---:|---:|---:|
| Single Ros2 step | 118.5 ms | 37.2 ms | 3.2x |
| 50-step `OuterLoop` | -- | 50.2 ms/step | -- |

End to end on a single CPU, VULCAN-JAX converges 4.4-6.7x faster than
VULCAN 2.0 across the three benchmark planets. Those quoted times come from a
fresh subprocess with an empty compilation cache, so they include the one-time
XLA compilation.

## Where VULCAN 2.0 spends a step

Profiling VULCAN 2.0's Ros2 step by operation shows the cost is dominated by
genuine numerical kernels, not by Python overhead.

| Operation | VULCAN 2.0 share | vs VULCAN-JAX | Why |
|---|---:|---:|---|
| **Linear solve** | **about 50%** | **about 2.3x cheaper** | VULCAN 2.0 calls `solve_banded` twice per step, which is two LU factorizations of the same matrix, and its band stores the species-diagonal off-blocks as if they were dense. Block-Thomas factorizes once, reuses that factorization for both Ros2 stages, and skips the zeros |
| Chemistry Jacobian | about 16% | about 6x cheaper | Analytical and stoichiometry-driven, rather than symbolic |
| Transport and chemistry right-hand side | about 18% | about 30-60x cheaper | Per-network code generation, XLA-fused, with abundance-independent gravity terms pre-baked |
| Repacking into SciPy band storage | about 7% | eliminated | The block-Thomas path never repacks |
| Python dispatch, glue, temporaries | about 3% | folded into one XLA program | VULCAN 2.0 is already well vectorized |

The headline correction to the usual "JAX removes Python overhead" story: Python
interpreter overhead is only about 3% here. VULCAN 2.0's time is real kernel
work, and the **linear solve is the single largest cost, about half the step**. So
the structure-aware, single-factorization block-Thomas solver is the dominant
lever, with the analytical Jacobian, the fused right-hand side, and the
eliminated repack stacking on top.

## Attribution inside the linear solve

The 2.3x is measured, and it splits unevenly:

- **2.10x from reusing one factorization** across both Rosenbrock stages.
- **1.10x from skipping the off-block zeros.**

Skipping the zeros cuts the floating-point operation count by more than an order
of magnitude, but at about 69 species the factorization sweep is bound by memory
latency rather than arithmetic, so the reduced operation count does not convert
into proportional wall time.

The analytical Jacobian is 7-16x faster to evaluate than VULCAN 2.0's generated
path. The range depends on whether the comparison is against the chemistry-block
evaluation alone or the full Jacobian assembly.

## Batched throughput

On one NVIDIA GH200, throughput rises from 0.010 converged profiles per second
for a single profile to 0.76 for a batch of 256, a gain of about 75x. A batch of
256 already uses tens of GB, so it is the largest batch reported.

These numbers are for a homogeneous batch of HD 189733b-like profiles,
initialized from chemical equilibrium and run without photochemistry.
Heterogeneous grids that span a range of temperatures or chemical regimes pay a
larger straggler cost, and their average throughput is correspondingly lower.

## Two measurement rules

**Guard wall-clock timings.** Single-threaded VULCAN 2.0 must show
`user + sys` roughly equal to `real` before any ratio is believable. Thermal
throttling inflates wall time several-fold while CPU time stays roughly correct,
which has produced both a retracted 16.8x "speedup" and a run that never
converged. `tools/bench_table1.sh` enforces the check. Accepted **step counts**
are load-independent, so use those for any cross-code comparison.

**Benchmark solver changes on real matrices.** The matrix VULCAN-JAX factorizes is
`I/(gamma*dt) - J`. On a converged HD 189733b state its blocks reach a condition
number of about 1.4e18 at the `dt` a run actually sits at, and about 6.7e23 at the
`dt_max=1e11` that retrieval workloads configure. A synthetic
diagonally-dominant test matrix tops out near 4e2, which is 21 orders of magnitude
short, so an accuracy result measured on one is meaningless. This is not
hypothetical: an inverse-carry optimization that measured 2.0x faster under `jvp`
with "no worse residual" on synthetic blocks turned out to be 569x worse on real
ones at `dt_max`, and was rejected. Build real blocks from `output/*.vul`,
`chem_jac_analytical_per_layer`, and `I/(gamma*dt)`.

**Retrieval workloads change which optimizations matter.** Under forward-mode
`jvp` the block-Thomas factorization is about 85% of the linear-algebra cost,
against about 50% on the primal path, because `lu_factor`'s tangent rule is
expensive. Benchmark at the retrieval species count under `jvp` when throughput
there is the target, and treat a primal-only measurement as not answering the
question.
