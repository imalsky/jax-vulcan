# VULCAN-JAX

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

VULCAN-JAX runs the same calculation as VULCAN — same configuration files, same input data, same `.vul` output schema — but the entire integration loop is JAX-accelerated. The runtime is standalone inside this tree: the master non-code runtime inputs under `atm/` and `thermo/`, the full `thermo/photo_cross/` tree, the vendored `cfg_examples/`, and the FastChem runtime payload needed for `ini_mix = 'EQ'` are present locally. The hot path is JAX-only, and the upstream tree is used only as an optional oracle.

**Scope.** The goal is to port all the physics of VULCAN-master, not to add a parity test for every config knob. Every live runtime branch in master has a JAX implementation; proving each non-default branch with an oracle test is *not* a project goal. If a non-default branch is wrong, we'll find out by running it. See "Live but non-default config paths" below for the inventory.

The pre-loop input + runner state is a single typed pytree (`state.RunState`). `python vulcan_jax.py` is a 90-line driver that does `runstate = RunState.with_pre_loop_setup(cfg)` → `runstate = integ(runstate)` → `output.save_out(runstate, dname)`. The legacy mutable container classes (`Variables` / `AtmData` / `Parameters`) live in `state.py` as private `_Variables` / `_AtmData` / `_Parameters` and are scratch inside the constructor; new code should consume `RunState` directly via the typed slots (`rs.atm.*`, `rs.rate.k`, `rs.step.y`, `rs.atoms.atom_loss`, `rs.metadata.Rf`, …).

## Quickstart

```bash
cd VULCAN-JAX/

# 1. Edit vulcan_cfg.py exactly as you would VULCAN-master's. Same format.
#    Defaults: HD189 with the SNCHO_photo_network_2025 chemical network.
#    Additional vendored presets live in cfg_examples/ for Earth, Jupiter,
#    HD189, and HD209.

# 2. Run:
python vulcan_jax.py        # the -n flag is also accepted (no-op now;
python vulcan_jax.py -n     # chem_funs is JAX-native, no SymPy regen needed)

# 3. Output goes to output/<out_name>.vul. Same format as VULCAN-master,
#    so all of VULCAN's plot_py/ scripts work unmodified.

# 4. Compare against a VULCAN-master run:
python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul
```

## Runtime data surface

The current JAX code reads the same file categories as VULCAN-master, and the
non-code runtime payload for those categories is vendored locally.

- `thermo/*.txt` via `vulcan_cfg.network`, `gibbs_text`, and `com_file`.
  `legacy_io.ReadRate` parses the network, `gibbs.py` consumes
  `gibbs_text`, and `composition.py` reads `com_file`.
- `thermo/photo_cross/` via `vulcan_cfg.cross_folder` when `use_photo=True`.
  `photo_setup.populate_photo()` loads thresholds, cross sections,
  dissociation branches, ion branches, and Rayleigh files from this tree
  into a dense `state.PhotoStaticInputs` pytree.
- `atm/*.txt` via `vulcan_cfg.atm_file`, `top_BC_flux_file`, and
  `bot_BC_flux_file`. `atm_setup.Atm.load_TPK()` reads the TP/Kzz tables and
  `Atm.BC_flux()` reads the boundary-condition tables.
- `atm/stellar_flux/*.txt` via `vulcan_cfg.sflux_file`. The host-side
  reader is `state.load_stellar_flux(cfg)`; `atm_setup.Atm.read_sflux()`
  bins the spectrum at runtime.
- `fastchem_vulcan/fastchem` plus `fastchem_vulcan/input/*` and
  `fastchem_vulcan/output/*` via `ini_abun.InitialAbun` for
  `ini_mix = 'EQ'`. This intentionally matches VULCAN-master's external
  FastChem call; replacing FastChem itself is out of scope.

The remaining tree differences in these areas are helper scripts / README files
and the non-runtime FastChem source/build tree, not missing runtime inputs.

## What's accelerated

The full integration is JAX: one JIT'd `jax.lax.while_loop` runs from start to convergence on device, with no NumPy in the per-step path.

**Per-step kernel** (`jax_step.jax_ros2_step` — `@jax.jit`, vmap-able, GPU-ready):
- **Chemistry RHS** (`chem.chem_rhs`) — vectorized scatter-add over the reaction network, vmapped over vertical layers.
- **Chemistry Jacobian** (`chem.chem_jac_analytical`) — stoichiometry-driven analytical build; the legacy `jacrev` path remains as a test oracle only.
- **Diffusion operator + block Jacobians** — eddy + molecular diffusion, including `use_vm_mol`, settling, top/bottom flux terms, and fixed-species row handling.
- **Block-tridiagonal solver** (`solver.factor_block_thomas_diag_offdiag` / `solve_block_thomas_diag_offdiag`) — factor once per Ros2 step and reuse across both Rosenbrock stages.

**Outer loop** (`outer_loop.OuterLoop` — single `lax.while_loop` per integration):
- **Inner accept/reject** (`clip` / `loss` / `step_ok` / `step_size`) — pure-JAX. Bit-equivalent to `op.Ros2.one_step` to ~1e-15.
- **Photochemistry update** — `compute_tau` / `compute_flux` / `compute_J` (`photo.py`) gated by `lax.cond` on `update_photo_frq`. Photo state lives in carry on device.
- **Atmosphere refresh** — `atm_refresh.update_mu_dz_jax` + `update_phi_esc_jax`. Hydrostatic balance runs every accepted step.
- **Condensation** — `conden.update_conden_rates` + `apply_h2o_relax_jax` / `apply_nh3_relax_jax` cold-trap relaxers, gated on `t >= start_conden_time`.
- **Convergence + termination** — ring-buffered `y_time` / `t_time`, in-runner `op.conv` port returning `longdy` / `longdydt`. `cond_fn` carries the full `op.stop` criterion.
- **Adaptive rtol** + **photo-frequency ini→final switch** — both fire inside the body.
- **Ion charge balance**, **mid-run `fix_species`**, and **fix-all-bot / fix-sp-bot bottom clamps** — handled inside the runner with the same config gating as the upstream Ros2 path.

Current measured timings from the checked-in benchmark/profile scripts on the HD189 default case:
- upstream `VULCAN-master` Ros2 single-step: `~126.0 ms`
- local `jax_ros2_step`: `~22.6 ms`
- local `OuterLoop` 50-step smoke: `~39.4 ms/accepted step`

The main wins are the analytical chemistry Jacobian, diagonal-aware block-tridiagonal factorization reuse, and pre-baked y-independent diffusion terms.

## Differentiability

Every per-step kernel is `jit` / `vmap` / `jvp` / `vjp` compatible. `outer_loop.runner`'s `lax.while_loop` supports `jvp` (forward-mode) but raises on `vjp` (reverse-mode). For end-to-end reverse-mode AD through the converged state, use the structured-input API in `steady_state_grad.py` — a `jax.custom_vjp` that uses the implicit-function theorem (`∂y*/∂theta = -(∂f/∂y)^{-1} (∂f/∂theta)`):

```python
from steady_state_grad import (
    build_steady_state_inputs,
    steady_state_value_and_grad,
)

# 1. Run the forward integration to convergence.
y_star = run_outer_loop(k_arr, atm_static)
inputs = build_steady_state_inputs(k_arr, atm_static)

def loss_fn(y):
    return some_scalar_loss(y)

loss, grad_inputs = steady_state_value_and_grad(
    loss_fn,
    inputs,
    y_star,
    net,
    residual_rtol=1e-6,
)

g_k = grad_inputs.k_arr
```

The differentiable surface includes rate constants, transport/diffusion fields, boundary-condition fluxes, and photo inputs through a typed pytree. A residual check is required before attaching implicit AD so gradients fail loudly if the supplied `y_star` is not converged tightly enough. The legacy `differentiable_steady_state(k_arr, y_star, atm_static, net)` wrapper is still available for the rate-table-only case. Memory cost is O(1) in step count — no checkpointing needed.

## Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │ vulcan_jax.py  (entry point, mirrors vulcan.py)           │
   │   ├─ JAX-native chem_funs (no make_chem_funs.py)          │
   │   ├─ atm_setup.Atm (atmosphere setup) — JAX-native        │
   │   ├─ legacy_io.ReadRate (rate metadata) — vendored        │
   │   ├─ ini_abun.InitialAbun (5 ini_mix modes) — JAX-native  │
   │   ├─ photo_setup.populate_photo (cross sections) — NumPy  │
   │   ├─ Ros2JAX.compute_tau/flux/J — one-shot pre-loop       │
   │   └─ outer_loop.OuterLoop (JAX) — single-shot runner      │
   │       └─ jax.jit(while_loop):                             │
   │            cond_fn:  count_max | runtime | converged      │
   │            body_fn (one accept, with internal retries):   │
   │              ├─ photo branch (lax.cond on update_photo_frq)│
   │              │   compute_tau / flux / J → update k_arr    │
   │              ├─ atm refresh (lax.cond on update_frq)      │
   │              │   update_mu_dz / phi_esc → splice into atm │
   │              ├─ jax_ros2_step (chem_rhs+jac, diffusion,   │
   │              │     block_thomas) — JAX                    │
   │              ├─ clip / loss / step_ok / step_size — JAX   │
   │              ├─ conden branch (lax.cond on t / use_relax) │
   │              ├─ hydrostatic balance + ion + fix_all_bot   │
   │              ├─ ring-buffer y_time / t_time history       │
   │              ├─ adaptive rtol + photo-freq switch         │
   │              └─ in-runner conv check → longdy/longdydt    │
   └──────────────────────────────────────────────────────────┘
```

## File map

```
VULCAN-JAX/
├── vulcan_jax.py        Entry point; mirrors vulcan.py orchestration
├── vulcan_cfg.py        VULCAN-format config (drop in your own)
├── outer_loop.py        OuterLoop — standalone single-shot JAX runner
├── op_jax.py            Ros2JAX — standalone photo adapter
│                         (compute_tau/flux/J for pre-loop one-shot)
├── jax_step.py          Pure-JAX vmap'able Ros2 step (incl. JAX diffusion)
├── solver.py            Block-tridiagonal Thomas solver
├── chem.py              JAX chemistry RHS + analytical Jacobian
├── photo.py             JAX two-stream photochem (tau / flux / J kernels)
├── runtime_validation.py Pre-run runtime/config validator
├── atm_refresh.py       JAX update_mu_dz + update_phi_esc
├── conden.py            JAX condensation rates + cold-trap relax
├── steady_state_grad.py Implicit-function-theorem custom_vjp for gradients
│                         of the converged state w.r.t. structured runtime inputs
├── rates.py             Forward rate coefficients
├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
├── network.py           Network parser (text -> stoichiometry tables)
├── integrate.py         Pure-JAX fixed-dt scan loop (validation/benchmarks)
├── legacy_io.py         Vendored op.ReadRate + op.Output
│                         (pre-loop rate-coef + .vul writer)
├── atm_setup.py         JAX-native Atm.* (f_pico / load_TPK / mol_diff /
│                         BC_flux / sp_sat etc.)
├── ini_abun.py          JAX-native InitialAbun.* (5 ini_mix modes;
│                         JAX Newton replaces scipy.optimize.fsolve)
├── photo_setup.py       Host-side cross-section preprocessing →
│                         dense PhotoStaticInputs pytree
├── composition.py       Per-species composition / mass tables
├── state.py             Typed pre-loop pytrees (AtmInputs / RateInputs /
│                         PhotoInputs / PhotoStaticInputs / IniAbunOutputs /
│                         RunState) + private legacy mutable containers
├── chem_funs.py         JAX-native module (re-exports network/gibbs/chem
│                         API as ni/nr/spec_list/Gibbs/chemdf, no SymPy)
├── phy_const.py         Vendored from VULCAN-master (physical constants)
├── pytest.ini           Pytest config (parallel-safe via FastChem flock)
├── atm/, thermo/, fastchem_vulcan/   Vendored runtime data files, chemistry
│                         networks, photo cross sections, and FastChem runtime
│                         payload
├── cfg_examples/        Vendored example configs for Earth / Jupiter / HD189 / HD209
├── benchmarks/          Bench and timing utilities for the current APIs
├── tests/               Validation suite, including example-config smoke tests
│   ├── conftest.py      Shared path / cwd / cfg-snapshot setup; conditional VULCAN-master
│   └── diffusion_numpy_ref.py   NumPy reference for diffusion (test-only)
└── CLAUDE.md            Style + numerical-hygiene notes for AI collaborators
```

## Differences from VULCAN-master

VULCAN-JAX is a drop-in replacement for VULCAN-master with two categories of intentional divergence:

**Live UI is host-side, fired between JIT'd step batches.** Master fires `op.Output.plot_update` and `op.Output.plot_flux_update` from inside its Python step loop. The JAX integration runs as a single JIT'd `lax.while_loop` on device, so when any of `use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie` is set, the runner switches to chunked execution: chunks of `vulcan_cfg.live_plot_frq` accepted steps, with `live_ui.LiveUI` reading the legacy `(var, atm, para)` view between chunks. Cadence is behavior-faithful — master updates after every `live_plot_frq` accepted steps; the chunk boundary is the same predicate. Movie frames land in `vulcan_cfg.movie_dir` (mixing-ratio) and `plot/movie/` (flux). The validator only rejects `use_live_flux=True` without `use_photo=True` (no diffuse fluxes without photochemistry).

**Solver scope**: only `ode_solver = "Ros2"` is supported. `vulcan_jax.py` raises `NotImplementedError` on any other solver (master's `op.py` carries Ros2 plus commented-out SemiEU / SparSemiEU stubs).

**JAX-native chemistry pipeline**: `chem_funs.py` is a hand-written JAX module that re-exports `ni` / `nr` / `spec_list` / `Gibbs` / `chemdf` etc. from `network.py` + `gibbs.py` + `chem.py`. No SymPy code generator; `make_chem_funs.py` is a no-op shim that exits cleanly. The `-n` flag on `vulcan_jax.py` is accepted as a no-op for upstream compatibility.

**JAX-only configuration knobs** added on top of master's surface:
- `batch_steps` — accepted Ros2 steps per JAX runner call. Held at 1; the single-shot runner makes this a no-op in production.
- `batch_max_retries` — safety cap on inner accept/reject retries per accepted step inside the JAX body. Mirrors `op.py:2518`'s force-accept-and-clip fallback.

**External dependencies kept as-is**: FastChem stays as an external subprocess for `ini_mix='EQ'`. Replacing FastChem itself is out of scope.

## Design choices

These are the choices you should know about before you read the code:

- **float64 is non-negotiable**. `jax.config.update('jax_enable_x64', True)` is set at module import in every JAX module. Rate constants span ~50 orders of magnitude; float32 silently fails.
- **Analytical chemistry Jacobian**, not `jax.jacrev`. `chem.chem_jac_analytical` is a stoichiometry-driven scatter that skips structurally-zero entries; it runs ~36× faster than the AD path (95 ms → 2.6 ms on the SNCHO network) and is bit-exact (≤1e-13) vs the AD reference, which is kept as the test oracle.
- **Diagonal-aware block-tridiagonal factorization**. `solver.block_thomas_diag_offdiag` exploits the diffusion Jacobian's diagonal-in-species off-blocks: the dense `O(ni³)` matmul `C_j @ inv(A_prev) @ B_{j-1}` reduces to an `O(ni²)` rank update. The factor is computed once per Ros2 step and reused across both Rosenbrock stages.
- **Single-shot JIT'd runner**. The integration runs as one `jax.lax.while_loop` — photo, atm-refresh, condensation, ion balance, fix-all-bot, adaptive rtol, photo-frequency switch, and the convergence check all live inside the body. No Python `while not stop()` polling.
- **Implicit-AD route for reverse-mode gradients**. `jax.lax.while_loop` blocks `jax.vjp`, so end-to-end `grad` of the converged state goes through `steady_state_grad.differentiable_steady_state` (a `jax.custom_vjp` using the implicit-function theorem). O(1) memory in step count; no checkpointing.
- **Typed pytree as the runtime surface**. `state.RunState` carries every input the runner reads (`AtmInputs`, `RateInputs`, `PhotoInputs`, `PhotoStaticInputs`, `IniAbunOutputs`, runner-state slots, host-side static `metadata`). The legacy mutable container classes (`Variables` / `AtmData` / `Parameters`) live as private `_Variables` / `_AtmData` / `_Parameters` inside `state.py` because the hybrid oracle tests need master's pipeline to mutate a shared `(var, atm)` at the JAX↔master boundary.
- **JAX/NumPy boundary**. The hot-path runtime is fully JAX; everything inside `OuterLoop`'s `lax.while_loop` is `jit` / `vmap` / `jvp` / `vjp` compatible. Some setup-time code stays NumPy by design — host-side, runs once, not on any AD path:
    - `photo_setup.py` — cross-section CSV reader + `np.interp` binning. Build `PhotoStaticInputs` directly from JAX arrays if you need cross-section gradients.
    - `legacy_io.ReadRate.read_rate` — rate-file metadata parser. Rate *values* flow through JAX-friendly `rates.build_rate_array` (bit-exact output).
    - `legacy_io.Output.save_out` — pickle writer; one-shot at end of run.
    - CSV / data-table readers in `composition.py`, `atm_setup.py`, `ini_abun.py`.
    - FastChem subprocess for `ini_mix='EQ'`.
  Forward-mode (`jvp` / `jacfwd`) and reverse-mode (`vjp` / `grad` via implicit-AD on the converged state) are supported across the full physical input surface as long as you supply inputs as JAX arrays into the typed pytrees.

## Validation

VULCAN-JAX is **numerically equivalent** to VULCAN-master at every measurable level:

| Layer | Agreement |
|---|---|
| Forward rate coefficients (596 reactions) | bit-exact (relerr = 0) |
| Reverse rates (533 from Gibbs) | 1.4e-14 |
| Atmosphere structure (pco/Tco/Kzz/M/...) | bit-exact |
| Initial abundances (FastChem path) | bit-exact |
| Chemistry Jacobian (vs `chem_funs.symjac`) | 4.3e-13 |
| Diffusion operator (vs `op.diffdf`) | 2e-6 (FP-noise-bound) |
| Block-Thomas solver | 3e-15 |
| Single Ros2 step (vs `op.Ros2.solver`) | 1.16e-15 |
| compute_tau / flux / J (vs `op.compute_*`) | 8e-16 / 3.7e-11 / 1.8e-11 |
| compute_Jion / ion `k_arr` wiring | unit-tested end-to-end |
| `update_mu_dz` / `update_phi_esc` (vs `op.*`) | 3-7e-16 (bit-exact) |
| `conden` / `h2o_relax` / `nh3_relax` (vs numpy ref) | 0 (bit-exact) |
| End-to-end 50-step run (HD189) | 1.59e-10 |

**Known per-term floor in `chem_rhs`** (~1e-4 relerr for CH2_1, HC3N, HCCO at certain layers): the JAX path uses `jnp.prod(y**stoich)` while VULCAN-master uses SymPy-emitted `y[A] * y[B] * y[C]`. The two differ by ~1 ulp per multiply; with ~7K production/loss terms cancelling down to ~1e2 the ulp drift accumulates to ~1e-4 absolute. This is a real, measurable disagreement vs VULCAN-master. Closing it requires a SymPy-faithful codegen and pays a ~14× per-call cost on the per-step hot path, which is **WONTFIX** by maintainer policy — the trade is net-negative for the project. Document the difference, do not try to close it. The Jacobian (`chem_jac_analytical`) is unaffected since it has much less per-entry cancellation.

## GPU / multi-CPU

To run on GPU (no code changes):
```bash
JAX_PLATFORM_NAME=gpu python vulcan_jax.py
```

To enable JAX device-level parallelism for vmap/pmap on multi-core CPU:
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py
```

For batched parameter sweeps (e.g. running 16 atmospheres at once with different stellar fluxes), see `examples/batched_run.py`.

## Live but non-default config paths

These VULCAN-master config branches are **live in master and implemented in VULCAN-JAX**, but exercised by the default HD189 + Earth + Jupiter + HD209 runs only partially. Each one is a JAX implementation already; what's missing is parity tests, not physics. By project policy we don't add an oracle test for every one — if a non-default branch is wrong, we'll find out when it's used.

- **Initial abundance** (`ini_abun.py`): `ini_mix in {EQ, vulcan_ini, table, const_mix, const_lowT}`, `use_ini_cold_trap`, `use_solar`, custom elemental abundance file.
- **Atmosphere** (`atm_setup.py`): `atm_type in {file, isothermal, analytical, vulcan_ini, table}`, `Kzz_prof in {const, file, JM16, Pfunc}`, `vz_prof in {const, file}`, `use_Kzz`, `use_vz`, `atm_base in {H2, N2, O2, CO2}`.
- **Transport** (`jax_step.py`, `outer_loop.py`): `use_moldiff`, `use_vm_mol`, `use_settling`, `use_topflux`, `use_botflux`, `use_fix_sp_bot`, `use_fix_H2He`, `use_sat_surfaceH2O`, `diff_esc` species and `max_flux`.
- **Photo / ion** (`photo.py`, `op_jax.py`): `use_photo`, `use_ion`, T-dependent cross sections, Rayleigh species, ion branch files, `remove_list`, the `ini_update_photo_frq → final_update_photo_frq` cadence switch.
- **Condensation** (`conden.py`): all master-supported species (H2O, NH3, H2SO4, S2, S4, S8, C), `use_relax in {H2O, NH3}`, `fix_species` with `fix_species_from_coldtrap_lev`, `post_conden_rtol`, `start_conden_time` / `stop_conden_time`.
- **Output / runtime**: `save_evolution`, `use_print_prog`, the four live UI flags (host-side; see Differences section).

Genuinely dead in master and **not** ported: non-Ros2 ODE solvers (`SemiEU` etc. are commented-out stubs in `op.py`); `naming_solver()` selection of `solver_fix_all_bot` (the selection is commented out, although the underlying solver path exists).

## Tests

```bash
pytest tests/                  # 109 pass + 1 skip on a current CPU-only run
                               # with ../VULCAN-master/ present. test_backend_parity
                               # skips cleanly without a GPU. Upstream-comparison
                               # tests skip cleanly when ../VULCAN-master/ is absent.
pytest tests/ -n auto          # parallel-safe (FastChem invocations serialise
                               # via fcntl.flock).
python tests/test_foo.py       # individual scripts still work standalone
```

VULCAN-JAX is fully **standalone** — `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling. The optional sibling, when present, serves as a validation oracle for the upstream-comparison tests.

The test suite covers the JAX↔master numerical bridge (chem RHS, Jacobian, diffusion, single Ros2 step, photo kernels) and a handful of integration smoke tests. It deliberately does not parametrize across every non-default config combination from the inventory above — see "Scope" at the top of this file.

See `CLAUDE.md` for the full test-discipline rules (when to add a test, what a "real" failure looks like, etc.).

## License

VULCAN-JAX inherits its license from VULCAN (GPLv3, see VULCAN-master/GPL_license.txt).

## Citation

If you use VULCAN-JAX in published work, please cite the underlying VULCAN papers:
- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20
- Tsai, S.-M., Malik, M., Kitzmann, D., Lyons, J. R., Fateev, A., Lavvas, P., & Heng, K. 2021, ApJ, 923, 264
