# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VULCAN-JAX is a JAX-accelerated port of [VULCAN](https://github.com/exoclime/VULCAN) (Tsai et al. 2017/2021), the photochemical-kinetics solver for exoplanet atmospheres. The goal is one codebase that runs on CPU and GPU, is fully vectorized (jit + vmap), is numerically equivalent to VULCAN-master, and is clean — no dead code, no legacy fallbacks, no magic numbers outside `vulcan_cfg.py`.

VULCAN-JAX is **standalone** (Phase A). `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling. The supported Earth / Jupiter / HD189 / HD209 Ros2 runtime assets (`atm/`, `thermo/`, `fastchem_vulcan/`, `cfg_examples/`) are vendored locally, and `op.ReadRate` + `op.Output` are vendored into `legacy_io.py`. `vulcan_jax.py` and `op_jax.py` no longer `sys.path.append` upstream.

`../VULCAN-master/` is the upstream reference. When present, it serves as an *optional* validation oracle: 10 tests compare JAX outputs against upstream's NumPy reference (`op.diffdf`, `op.Ros2.solver`, `chem_funs.symjac`, etc.) and skip cleanly with a clear reason when the sibling is absent. Do not refactor it.

`../vulcan-emulator/` is an unrelated transformer-surrogate project. Out of scope for this repo.

## Conda environment

All commands run in the `vulcan` conda env. It has `jax`, `numpy`, `scipy`, `h5py`, `pytest`, `pytest-xdist`, `ruff`, and `vulture`.

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate vulcan
```

Direct binary if scripts need it: `/opt/homebrew/Caskroom/miniforge/base/envs/vulcan/bin/python`.

## Common commands

```bash
python vulcan_jax.py                                            # forward model (HD189 by default)
JAX_PLATFORM_NAME=gpu python vulcan_jax.py                      # GPU, no code changes
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py  # multi-CPU for vmap/pmap

pytest tests/                                                    # full suite (run serially)
pytest tests/ -k "ros2 or block_thomas"                          # filter
# NOTE: do not use `-n auto` — parallel runs collide on FastChem's fixed
# output path (see pytest.ini for details).

ruff check .                                                     # lint
ruff format .                                                    # format
vulture . --min-confidence 80                                    # dead-code finder

python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul   # diff vs upstream
python benchmarks/bench_step.py                                  # per-step timing
```

The `-n` flag on `vulcan_jax.py` is accepted for upstream compatibility but is a no-op — `chem_funs.py` is JAX-native and there is no SymPy regeneration step.

## Architecture

`vulcan_jax.py` mirrors `vulcan.py`'s orchestration. The hot path is fully JAX (post-Phase 10) — `outer_loop.OuterLoop` runs as a single JIT'd `lax.while_loop` on device:

```
outer_loop.OuterLoop  (single jit'd lax.while_loop — full integration)
  └── jax_step.jax_ros2_step  (jit'd, vmap-able, GPU-ready)
        ├── chem.chem_rhs                  (segment_sum, vmapped over layers)
        ├── chem.chem_jac_analytical       (Phase 11; stoichiometry-driven scatter)
        ├── jax_step.compute_diff_grav     (Phase 12; y-independent, computed
        │                                   once per Ros2 step instead of twice)
        ├── jax_step._build_diff_coeffs_jax (eddy + molecular, four modes)
        └── solver.block_thomas_diag_offdiag (Phase 12; diagonal-aware
                                              off-diagonals — O(ni²) rank update)

  Photochemistry: photo.py JAX kernels (compute_tau / compute_flux / compute_J)
                  fire inside the runner via lax.cond on update_photo_frq.
  Pre-loop setup: legacy_io.ReadRate (rates + photo bins) + Output.save_cfg /
                  save_out (.vul writer). Vendored from upstream op.py.

  Differentiability: per-step kernels are all jit/vmap/jvp/vjp compatible.
                  The runner's lax.while_loop supports jvp but NOT vjp; for
                  reverse-mode AD across the converged state, use
                  steady_state_grad.differentiable_steady_state or the
                  structured-input helpers in `steady_state_grad.py` — a
                  jax.custom_vjp using the implicit-function theorem
                  (∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)). O(1) memory
                  in step count.
```

`chem_funs.py` is a JAX-native module (Phase 9) that re-exports `ni`/`nr`/`spec_list`/`Gibbs`/`chemdf` etc. from `network.py` + `gibbs.py` + `chem.py`. No SymPy code generator; `make_chem_funs.py` is not invoked at startup. `symjac` raises `NotImplementedError` — production uses `chem.chem_jac` directly.

Numerical agreement vs VULCAN-master (HD189, SNCHO_photo_network_2025):

| Layer | Agreement |
|---|---|
| Forward rates (596 reactions) | bit-exact |
| Reverse rates (533 from Gibbs) | 1.4e-14 |
| Atmosphere setup / FastChem init | bit-exact |
| chem_jac (vs symjac) | 4.3e-13 |
| Diffusion operator | 2e-6 (FP-cancellation-bound; see Numerical hygiene) |
| Block-Thomas | 3e-15 |
| Single Ros2 step | 1.16e-15 |
| End-to-end 50-step HD189 | 1.59e-10 |

`STATUS.md` tracks per-component implementation status.

## Numerical hygiene

- **float64 is non-negotiable.** `jax.config.update('jax_enable_x64', True)` is set at module import. Rate constants span ~50 orders of magnitude; float32 silently fails. Don't relax this anywhere.
- **Production uses `chem.chem_jac_analytical`, not `chem.chem_jac`.** Phase 11 replaced `jax.jacrev(chem_rhs_per_layer)` with a stoichiometry-driven analytical build (`chem_jac_analytical`); chem_jac drops 95 ms → 2.6 ms. The `chem.chem_jac` (jacrev) path is kept as the test oracle for `test_chem_jac_sparse.py`. Don't replace the analytical path with AD without re-benchmarking — the speedup is not free, the analytical form skips structurally-zero entries that AD materialises.
- **block_thomas: production uses `block_thomas_diag_offdiag` (Phase 12).** When sup/sub are diagonal-in-species (which they always are for the diffusion Jacobian), the dense `O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)` rank update. The dense `block_thomas` stays for callers with truly dense off-diagonals.
- **`NetworkArrays` is a registered pytree** (`jax.tree_util.register_pytree_node`). `ni`/`nr` ride as static aux_data so `jit`/`vmap` don't retrace per-network and callers don't need `static_argnames` everywhere.
- **`pos_cut` / `nega_cut` clipping** is replicated explicitly in the outer loop, matching `op.py:2450-2473`. Don't move it into the JIT'd kernel — VULCAN's order matters.
- **Asymmetric M factor for dissociation reactions.** For `HNCO + M -> H + NCO` (R1051), M is on the LHS only; the forward rate is `k * y[HNCO] * M` (3-body), but the reverse `H + NCO -> HNCO` is bimolecular *without* M. VULCAN's auto-generated `chem_funs` preserves this asymmetry by iterating literal reactant/product lists. `network.py` tracks `has_M_reac` and `has_M_prod` separately, so `is_three_body[forward_i]` and `is_three_body[reverse_i+1]` may differ. Anything that assumes symmetry between forward and reverse three-body status is wrong.
- **`chem_rhs` cancellation floor.** `chem_rhs` accumulates 20–40 terms per species with production/loss cancellation; relative error vs VULCAN's `chemdf` is ~1e-4 for a few species (CH2_1, HC3N, HCCO at certain layers). This is summation-order sensitivity (`segment_sum` tree reduction vs SymPy's flat sum). The NumPy reference path gives ~3e-5 in the same cells. **Both are mathematically equivalent**; tests on `chem_rhs` use `rtol=1e-3`. The Jacobian has much less per-entry cancellation and agrees to machine precision.
- **Diffusion Jacobian: VULCAN-master vs ourselves.** `apply_diffusion` matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at a handful of diagonal cells for heavy condensables (S8, layers 5 and 25). Direct comparison with the analytical derivative of `op.diffdf` confirms our Jacobian is correct — `op.lhs_jac_tot` has a minor self-inconsistency. Impact on integration is negligible; don't try to "fix" us to match upstream there.
- **`outer_loop.runner` is forward-mode-AD-only.** `jax.lax.while_loop` supports `jvp`/`jacfwd` but raises on `vjp`/`grad`. For reverse-mode through the integration, route through `steady_state_grad.py`'s implicit-function-theorem API. The synthetic test in `tests/test_steady_state_grad.py` validates both the legacy `k_arr` wrapper and the structured-input pytree path against finite differences; the gradient accuracy is bounded by the runner's residual `||f(y*)||`, so use `validate_steady_state_solution(...)` / `steady_state_value_and_grad(...)` and tighten the forward convergence criterion when the default `yconv_cri = 0.01` residual is too loose.

## Output schema

The `.vul` file is `pickle.dump(...)` with `protocol=4`, identical to VULCAN-master's `op.Output.save_out` (lines 3219-3260). Three top-level keys:

- `'variable'` — dict from `vars(data_var)` filtered by `var.var_save`. Contains `y`, `ymix`, `t`, `dt`, `longdy`, `atom_*`, `Rf`, `k`, and (if photo) `tau`, `aflux`, `J_sp`, `n_branch`.
- `'atm'` — dict from `vars(data_atm)`: `pco`, `Tco`, `Kzz`, `Dzz`, `mu`, `n_0`, `dz`, `dzi`, BC arrays.
- `'parameter'` — dict from `vars(data_para)`: `count`, `nega_count`, `loss_count`.

Same keys, same array shapes, same dtypes (float64) as VULCAN-master. **All JAX arrays are passed through `np.asarray(...)` before pickling** so VULCAN's `plot_py/` scripts load our output unmodified. Don't break this contract; it is the cheapest end-to-end check we have.

## Test discipline

**Current state:** `pytest tests/` is green on the current tree. Most files still use a thin `def test_main(): assert main() == 0` wrapper around their existing script-style `main()`; two tests (`test_diffusion`, `test_ros2_step`) wrap the script as a subprocess because they do a deliberate VULCAN-master ↔ VULCAN-JAX module-table swap that only works from a cold Python start. The suite now also includes `compute_Jion` coverage, vendored example-config setup coverage, and an optional CPU↔GPU parity test that skips cleanly on CPU-only hosts. Per-test fixture sharing for the HD189 reference state is a future cleanup (not a correctness gap). Run serially — parallel collides on FastChem.

A function is "important enough to test" if any of these are true:
- It's on the per-step hot path (`chem_rhs`, `chem_jac`, `block_thomas`, `apply_diffusion`, photo kernels, `jax_ros2_step`).
- It's a numerical bridge to VULCAN-master (anything in the agreement table above).
- It's a `jit`/`vmap`/`grad` boundary (transform consistency is a real failure mode in JAX).
- It's a parser or schema reader (`network.py`, NASA-9 reader) — silent-mismatch territory.

Patterns to keep:
- **Equivalence vs VULCAN-master.** Load a real HD189 state once in a session-scoped `conftest` fixture; compare component outputs at machine-precision tolerances (with the cancellation floors documented above).
- **vmap consistency.** Single-call output must match `vmap(...)(stacked_input)[i]` to ~1e-12.
- **Gradient sanity.** `jax.grad`/`jvp` through the per-step kernel must be finite and finite-differenced to ~1e-6.
- **CPU vs GPU agreement.** When GPU is available, the same kernel on both backends must agree to float64 precision.

Don't add tests that just exercise plumbing or duplicate VULCAN's own physics; the bar is "could fail silently and matter."

## Style rules

- **Magic numbers belong in `vulcan_cfg.py` and nowhere else.** That file is the canonical configuration surface and intentionally mirrors VULCAN-master's format. Any literal in `*.py` source with physical or algorithmic meaning gets a named constant with a one-line docstring saying what it is and where it came from.
- **Comments:** docstrings on every public function (what it computes + array shapes + units). Inline comments only where the math, algorithm, or JAX-specific choice is non-obvious — e.g., why `jacrev` over `jacfwd`, why `segment_sum` instead of zip-and-add, why a particular `lax.scan` carry shape. Don't narrate what well-named code already says.
- **No dead code.** `vulture --min-confidence 80` should come back empty. If a function isn't called, delete it; if it's a future placeholder, don't write it yet.
- **No legacy fallbacks.** A code path that exists only because VULCAN-master had it and we haven't ported it is debt to remove, not a feature to preserve. Prefer raising a clear `NotImplementedError` over silently routing to the NumPy parent class.
- **Vectorize.** Per-layer operations should be `vmap`'d, not Python-looped. Reach for `jax.lax.scan` only when there is a true sequential dependency (e.g., block-Thomas forward elim).

## Toward fully-JAX (open work)

Phase 10 of the JAX port is complete (10.1 → 10.7): the outer loop is a single JIT'd `jax.lax.while_loop` with no NumPy hot-path code, no `op.Ros2` inheritance, no `op.Integration` parent class, and no non-Ros2 fallback. `pytest tests/` discovers and runs the suite. Track per-component progress in `STATUS.md`, not here.

Remaining open work:

- **Backend-parity benchmarking on real GPU hardware.** The kernels are GPU-ready and the repository now includes standalone backend-safe benchmark scripts, but checked-in CPU↔GPU float64 parity numbers still need to be recorded on an actual GPU machine.
- **Per-test fixture sharing**: `pytest tests/` runs the suite today via thin `def test_main(): assert main() == 0` wrappers; a follow-up cleanup could share an HD189 reference-state fixture across tests in `conftest.py` to cut per-test setup. Parallel execution (`pytest -n auto`) collides on FastChem's fixed output path — run serially.
- **Live plotting / movie output remain intentionally unsupported in the JAX runtime.** `runtime_validation.py` rejects those flags before JIT setup rather than silently degrading.
