# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VULCAN-JAX is a JAX-accelerated port of [VULCAN](https://github.com/exoclime/VULCAN) (Tsai et al. 2017/2021), the photochemical-kinetics solver for exoplanet atmospheres. The goal is one codebase that runs on CPU and GPU, is fully vectorized (jit + vmap), is numerically equivalent to VULCAN-master, and is clean — no dead code, no legacy fallbacks, no magic numbers outside `vulcan_cfg.py`.

`../VULCAN-master/` is the upstream reference and lives in this workspace **only as a validation oracle** (and as the source of `atm/`, `thermo/`, `fastchem_vulcan/` data files, currently symlinked). Do not refactor it.

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

pytest tests/                                                    # full suite (target state)
pytest tests/ -n auto                                            # parallel via pytest-xdist
pytest tests/ -k "ros2 or block_thomas"                          # filter

ruff check .                                                     # lint
ruff format .                                                    # format
vulture . --min-confidence 80                                    # dead-code finder

python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul   # diff vs upstream
python benchmarks/bench_step.py                                  # per-step timing
```

The `-n` flag on `vulcan_jax.py` is accepted for upstream compatibility but is a no-op — `chem_funs.py` is JAX-native and there is no SymPy regeneration step.

## Architecture

`vulcan_jax.py` mirrors `vulcan.py`'s orchestration. The hot path:

```
op.Integration (outer loop, NumPy — Phase 10 will port to lax.while_loop)
  └── Ros2JAX.solver  → jax_step.jax_ros2_step  (jit'd, vmap-able)
        ├── chem.chem_rhs        (segment_sum, vmapped over layers)
        ├── chem.chem_jac        (jacrev per layer, vmapped)
        ├── diffusion.apply_*    (eddy + molecular, four modes)
        └── solver.block_thomas  (two lax.scan: forward elim + back sub)

  Photochemistry: photo.py JAX kernels (compute_tau / compute_flux / compute_J)
                  wired into Ros2JAX.compute_{tau,flux,J} overrides.
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
- **`chem_jac` uses `jax.jacrev`, not `jacfwd`.** Reverse-mode handles the `segment_sum` scatter pattern more efficiently AND gives slightly better numerical accuracy on this network (4.3e-13 vs 6e-11 for jacfwd). Don't switch back without re-benchmarking on the full HD189 network.
- **`NetworkArrays` is a registered pytree** (`jax.tree_util.register_pytree_node`). `ni`/`nr` ride as static aux_data so `jit`/`vmap` don't retrace per-network and callers don't need `static_argnames` everywhere.
- **`pos_cut` / `nega_cut` clipping** is replicated explicitly in the outer loop, matching `op.py:2450-2473`. Don't move it into the JIT'd kernel — VULCAN's order matters.
- **Asymmetric M factor for dissociation reactions.** For `HNCO + M -> H + NCO` (R1051), M is on the LHS only; the forward rate is `k * y[HNCO] * M` (3-body), but the reverse `H + NCO -> HNCO` is bimolecular *without* M. VULCAN's auto-generated `chem_funs` preserves this asymmetry by iterating literal reactant/product lists. `network.py` tracks `has_M_reac` and `has_M_prod` separately, so `is_three_body[forward_i]` and `is_three_body[reverse_i+1]` may differ. Anything that assumes symmetry between forward and reverse three-body status is wrong.
- **`chem_rhs` cancellation floor.** `chem_rhs` accumulates 20–40 terms per species with production/loss cancellation; relative error vs VULCAN's `chemdf` is ~1e-4 for a few species (CH2_1, HC3N, HCCO at certain layers). This is summation-order sensitivity (`segment_sum` tree reduction vs SymPy's flat sum). The NumPy reference path gives ~3e-5 in the same cells. **Both are mathematically equivalent**; tests on `chem_rhs` use `rtol=1e-3`. The Jacobian has much less per-entry cancellation and agrees to machine precision.
- **Diffusion Jacobian: VULCAN-master vs ourselves.** `apply_diffusion` matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at a handful of diagonal cells for heavy condensables (S8, layers 5 and 25). Direct comparison with the analytical derivative of `op.diffdf` confirms our Jacobian is correct — `op.lhs_jac_tot` has a minor self-inconsistency. Impact on integration is negligible; don't try to "fix" us to match upstream there.

## Output schema

The `.vul` file is `pickle.dump(...)` with `protocol=4`, identical to VULCAN-master's `op.Output.save_out` (lines 3219-3260). Three top-level keys:

- `'variable'` — dict from `vars(data_var)` filtered by `var.var_save`. Contains `y`, `ymix`, `t`, `dt`, `longdy`, `atom_*`, `Rf`, `k`, and (if photo) `tau`, `aflux`, `J_sp`, `n_branch`.
- `'atm'` — dict from `vars(data_atm)`: `pco`, `Tco`, `Kzz`, `Dzz`, `mu`, `n_0`, `dz`, `dzi`, BC arrays.
- `'parameter'` — dict from `vars(data_para)`: `count`, `nega_count`, `loss_count`.

Same keys, same array shapes, same dtypes (float64) as VULCAN-master. **All JAX arrays are passed through `np.asarray(...)` before pickling** so VULCAN's `plot_py/` scripts load our output unmodified. Don't break this contract; it is the cheapest end-to-end check we have.

## Test discipline

**Target state:** every test file uses pytest collection (`def test_*` with `assert`), shared fixtures live in `conftest.py`, and `pytest tests/` runs everything. The 17 standalone scripts under `tests/` (each with `if __name__ == "__main__": main()`) need to migrate.

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

These are the remaining pieces to make the per-call path fully JAX with no NumPy hot-path code. Track progress in `STATUS.md`, not here.

- **Outer integration loop.** `op.Integration.__call__` is still NumPy and is called once per accepted step; condensation, ion charge balance, and step accept/reject live there. The plan is `jax.lax.while_loop` carrying `(y, t, dt, accept_count, ...)`. `integrate.py` already has a basic adaptive-dt JAX loop (validated 1.5e-16 vs the Python loop, 1.6× faster); production drop-in still inherits from `op.Integration` and remains the gap.
- **Remove the `Ros2JAX(op.Ros2)` inheritance** once the outer loop is ported. `Ros2JAX` already overrides every method on the per-step path; the parent class only contributes the outer loop and `lhs_jac_tot` (which `Ros2JAX` does not call). The subclass relationship pulls in NumPy by import alone.
- **Drop the non-Ros2 solver branch** in `vulcan_jax.py` (the `else: solver = getattr(op, solver_str)()` block). VULCAN-JAX targets Ros2; any other request should error with a clear message.
- **Decide on `atm/`/`thermo/`/`fastchem_vulcan/` symlinks.** Drop-in-compat shortcut today; long term, either inline only the data we actually need or make the upstream-tree dependency explicit at config-load time.
- **Tests migration.** Convert the 17 standalone scripts in `tests/` to pytest functions with shared fixtures (`conftest.py` for the HD189 reference state).
- **Custom analytical sparse Jacobian (Phase 11).** `chem_jac` is 96% of step time and computes a dense `ni×ni`-per-layer block where ~70% of entries are structurally zero. Stoichiometry-driven sparse assembly is the largest single perf lever (estimated 3-5×).
