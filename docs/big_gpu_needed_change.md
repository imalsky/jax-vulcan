# Batch-512 GPU OOM: cause, fix sizing, and ripple risk

Context: GPU benchmark (job 61318, node cgh11n04 / GH200, jax 0.6.2, vulcan_jax
0.1.13) swept batch sizes 1, 8, 32, 128, 512 at nz=150, ini_mix=EQ, use_photo=False.
Batches 1–128 scaled cleanly; batch 512 died with `RESOURCE_EXHAUSTED` trying to
allocate 21.9 GiB on top of a live set XLA could not rematerialize below 42 GiB.

## Throughput up to the wall

| batch | total wall | per-planet | converged |
|------:|-----------:|-----------:|----------:|
| 1   | 96.7 s  | 96.7 s | 1/1 |
| 8   | 168.6 s | 21.1 s | 8/8 |
| 32  | 203.5 s | 6.4 s  | 32/32 |
| 128 | 243.0 s | 1.9 s  | 128/128 |
| 512 | —       | OOM    | crash |

Per-planet time was still dropping ~3.35×/4× at batch 128, i.e. the device was
**not saturated** — 512 would have been faster per planet if it fit. This is a
memory wall, not a compute wall. `count_max=2500` was never hit; everything
converged by ~1090–1287 steps.

## Why batch 512 needs ~60 GiB

Not the persistent state (~3 GiB at batch 512). The blow-up is a transient inside
the analytical-Jacobian build, multiplied by vmap.

`chem_jac_analytical_per_layer` (`src/vulcan_jax/chem.py:173-231`) assembles the
Jacobian reaction-by-reaction before collapsing it to the compact block:

```python
contrib   = out_stoich_signed[:, :, None] * drate_dy[:, None, :]   # (nr+1, 6, 3)
row       = broadcast_to(out_idx...,      contrib.shape)
col       = broadcast_to(reactant_idx..., contrib.shape)
J_flat    = segment_sum(contrib.reshape(-1), keys, (ni+1)**2)      # -> (69,69)
```

Network: ni=69, nr=878 (nr+1=879), max_reac=max_prod=3. Per layer the transient
is `(879, 6, 3)` ≈ 15.8k elements; the final block is only `(69,69)` ≈ 4.8k. The
un-reduced per-reaction form is ~13× bigger than the thing it produces, because
all 878 reactions get a slot before `segment_sum` folds them onto 69 species.

vmap is SIMD: no streaming over lanes, every lane's intermediate is live at once.
Stacking the two batch axes XLA carries (nz=150 layers × batch=512 lanes):

```
512 × 150 × 879 × 6 × 3 = 1.22e9 elements  ->  9.05 GiB per tensor (f64/i64)
```

`contrib`, `row`, `col`, and the `reshape(-1)` + `keys` pair feeding `segment_sum`
all coexist → 42–61 GiB. The 20.4 GiB single allocation that tipped it over is the
`segment_sum` scatter pair (2.74e9 elems ≈ 2.25× one `contrib`). Linear in batch,
which is why 128 (~15 GiB) fit and 512 (~60 GiB) did not.

Cheap things that do NOT fix it: `TF_GPU_ALLOCATOR=cuda_malloc_async` (fragmentation
only, not a 1.5× overcommit); rematerialization is already on and maxed out.

## Two fixes

### A. Benchmark device-batch tiling (cheap, risk-free) — do this first

Cap the on-device batch at ~128–256 and loop host-side over sub-tiles, reusing the
same compiled `_vrunner` (one XLA compile, amortized). Because throughput is still
near-linear at 128, 4×128 tiles recover essentially all of the 512 throughput.
Lives entirely in `supercomputer_cmds/gpu_benchmark.py` (`integrate_chunked` /
`benchmark_one`). **Zero numerics change, zero other consumers touched, zero
re-baseline.**

### B. Chunked Jacobian assembly in chem.py (proper fix, but rippling)

Bound the transient by chunking the reaction axis: `lax.scan` over reaction blocks,
each doing a partial `segment_sum` into the carried `(ni+1)²` accumulator (carry is
4900 floats). Peak transient drops from `nr` to `chunk_size` → ~10–30× smaller, so
512 fits outright and every caller benefits.

Scope is small and contained:
- One function body (`chem.py:173-231`), ~20–30 lines replacing the
  `contrib → segment_sum` tail. `drate_dy` and the leave-one-out logic stay.
- One direct solver call site (`jax_step.py:500`), signature unchanged.
- Existing oracle test `tests/test_chem_jac_sparse.py` validates the matrix to
  machine precision against the `jacrev` path.

Code effort ~30 min. The real cost is on-device verification (below).

## Ripple risk for fix B — the non-obvious parts

1. **It silently changes the gradient/sensitivity path.** `chem_jac_analytical` is
   reused in `steady_state_grad.py:179` (`_build_jacobian_blocks`), the implicit-AD
   path that gives ∂y*/∂θ — the differentiable-coupling/adjoint payoff of the JAX
   rewrite. Values stay correct (same matrix, used as a known linear operator), but
   its tests belong in the verification set, not just the solver tests.

2. **Results stop being bit-reproducible, and the Jacobian is *in* the step
   formula.** The step is Rosenbrock semi-implicit (`r = 1 + 1/√2`, `jax_step.py:482`):
   `diag = c0·eye − chem_J` is factored directly, so the Jacobian shapes the update
   `k1`, not just a converged-away Newton matrix. Chunking `segment_sum` changes the
   float summation order across 878 reactions (~1e-16), which amplifies over ~1000
   stiff steps. The fixed point (RHS=0) is unchanged, so converged mixing ratios
   still satisfy `loss_criteria=5e-4`, but they will NOT match old outputs to 1e-13.
   Concrete fallout:
   - Step counts wobble (the 1090/1067/1287 log numbers shift by a few). Anything
     asserting exact step counts breaks; the `gpu_benchmark_fix` pinned counts
     (4087/1429/606, count_min=120) move.
   - Frozen emulator training data stops reproducing byte-for-byte (quality fine,
     exact reproduction not). Golden artifacts need re-stamping.
   - `tests/test_default_master_parity.py` is the canary — tolerance-based, should
     still pass, but run it explicitly.
   - Compile-cache (`JAX_COMPILATION_CACHE_DIR`) invalidates for every shape touching
     this kernel; "incl XLA compile" numbers all shift, and `lax.scan` may compile
     slower than the flat op.

3. **XLA-undo risk.** If `scan` gets fused/unrolled back into one big op, you eat all
   the bit-level churn above and keep the OOM. Only shows up on-device — needs a
   512-batch run on the NAS to confirm. Coax with `unroll=1` or a remat on the chunk
   body if needed.

What is NOT at risk: `_project_chem_jac` atom-conservation projection (same matrix
in); the converged physics (fixed point is Jacobian-independent); correctness of the
matrix itself (oracle test guards it).

## Recommendation

Ship A now to unblock the 512 sweep. Treat B as a deliberate, separately-verified
refactor with a re-baseline pass — `test_chem_jac_sparse`, `test_default_master_parity`,
the adjoint tests, plus a before/after step-count diff on one HD189 profile, plus a
512-batch NAS run to confirm peak actually drops — not a quiet drive-by.
