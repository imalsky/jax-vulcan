# VULCAN-JAX

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

VULCAN-JAX runs the same calculation as VULCAN — same configuration files, same input data, same `.vul` output schema — but the entire integration loop is JAX-accelerated. The result is a forward model that's slightly faster than VULCAN-master on a single CPU AND directly usable on GPU/TPU AND `jax.vmap`-able for batched parameter sweeps.

## Quickstart

```bash
cd VULCAN-JAX/

# 1. Edit vulcan_cfg.py exactly as you would VULCAN-master's. Same format.
#    Defaults: HD189 with the SNCHO_photo_network_2025 chemical network.

# 2. Run:
python vulcan_jax.py        # the -n flag is also accepted (no-op now;
python vulcan_jax.py -n     # chem_funs is JAX-native, no SymPy regen needed)

# 3. Output goes to output/<out_name>.vul. Same format as VULCAN-master,
#    so all of VULCAN's plot_py/ scripts work unmodified.

# 4. Compare against a VULCAN-master run:
python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul
```

## What's accelerated

The full integration is JAX (post-Phase 10): one JIT'd `jax.lax.while_loop` runs from start to convergence on device, with no NumPy in the per-step path.

**Per-step kernel** (`jax_step.jax_ros2_step` — `@jax.jit`, vmap-able, GPU-ready):
- **Chemistry RHS** (`chem.chem_rhs`) — vectorized scatter-add over the reaction network, vmapped over vertical layers. ~1 ms/call for nz=120, ni=93.
- **Chemistry Jacobian** (`chem.chem_jac`) — `jax.jacrev` on the per-layer rate function, vmapped. ~150 ms/call.
- **Diffusion operator + block Jacobians** — eddy + molecular diffusion, JAX inline.
- **Block-tridiagonal solver** (`solver.block_thomas`) — two `jax.lax.scan`s for forward elim + back sub. ~16 ms.

**Outer loop** (`outer_loop.OuterLoop` — single `lax.while_loop` per integration):
- **Inner accept/reject** (`clip` / `loss` / `step_ok` / `step_size`) — pure-JAX. Bit-equivalent to `op.Ros2.one_step` to ~1e-15.
- **Photochemistry update** — `compute_tau` / `compute_flux` / `compute_J` (`photo.py`) gated by `lax.cond` on `update_photo_frq`. Photo state lives in carry on device.
- **Atmosphere refresh** — `atm_refresh.update_mu_dz_jax` + `update_phi_esc_jax`. Hydrostatic balance runs every accepted step.
- **Condensation** — `conden.update_conden_rates` + `apply_h2o_relax_jax` / `apply_nh3_relax_jax` cold-trap relaxers, gated on `t >= start_conden_time`.
- **Convergence + termination** — ring-buffered `y_time` / `t_time`, in-runner `op.conv` port returning `longdy` / `longdydt`. `cond_fn` carries the full `op.stop` criterion.
- **Adaptive rtol** + **photo-frequency ini→final switch** — both fire inside the body.
- **Ion charge balance** + **fix-all-bot bottom clamp** — Phase 10.6, gated by Python bools at trace time.

Total: **~47 ms/step warm** (Phase 11) — down from ~149 ms/step at Phase 10.6 and ~175 ms/step VULCAN-master. The Phase 11 win comes from `chem.chem_jac_analytical`, a stoichiometry-driven analytical Jacobian that replaces `jax.jacrev(chem_rhs_per_layer)` and drops chem_jac from 95 ms → 2.6 ms (36×). Bit-exact (≤1e-13) vs the AD path.

## Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │ vulcan_jax.py  (entry point, mirrors vulcan.py)           │
   │   ├─ JAX-native chem_funs (no make_chem_funs.py)          │
   │   ├─ build_atm (atmosphere setup) — vendored             │
   │   ├─ legacy_io.ReadRate (rate coefs) — vendored          │
   │   ├─ InitialAbun (FastChem subprocess) — vendored         │
   │   ├─ photochem setup — legacy_io.ReadRate.make_bins_read │
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
│                         (no op.Integration parent class)
├── op_jax.py            Ros2JAX — standalone photo adapter
│                         (compute_tau/flux/J for pre-loop one-shot)
├── jax_step.py          Pure-JAX vmap'able Ros2 step (incl. JAX diffusion)
├── solver.py            Block-tridiagonal Thomas solver
├── chem.py              JAX chemistry RHS + autodiff Jacobian
├── photo.py             JAX two-stream photochem (tau / flux / J kernels)
├── atm_refresh.py       JAX update_mu_dz + update_phi_esc (Phase 10.3)
├── conden.py            JAX condensation rates + cold-trap relax (Phase 10.4)
├── rates.py             Forward rate coefficients
├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
├── network.py           Network parser (text -> stoichiometry tables)
├── integrate.py         Pure-JAX fixed-dt scan loop (validation/benchmarks)
├── legacy_io.py         Vendored op.ReadRate + op.Output (Phase A);
│                         pre-loop rate-coef + photo bins setup + .vul writer
├── build_atm.py         Vendored from VULCAN-master (atmosphere setup)
├── store.py             Vendored from VULCAN-master (Variables/AtmData/Parameters)
├── chem_funs.py         JAX-native module (re-exports network/gibbs/chem
│                         API as ni/nr/spec_list/Gibbs/chemdf, no SymPy)
├── phy_const.py         Vendored from VULCAN-master (physical constants)
├── pytest.ini           Pytest config (serial; FastChem prevents -n auto)
├── atm/, thermo/, fastchem_vulcan/   Vendored data files (HD189 + SNCHO net, ~28 MB)
├── tests/               26 unit + validation tests
│   ├── conftest.py      Shared path / cwd / warning setup; conditional VULCAN-master
│   └── diffusion_numpy_ref.py   NumPy reference for diffusion (test-only)
├── STATUS.md            Detailed implementation status
└── CLAUDE.md            Style + numerical-hygiene notes for AI collaborators
```

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
| `update_mu_dz` / `update_phi_esc` (vs `op.*`) | 3-7e-16 (bit-exact) |
| `conden` / `h2o_relax` / `nh3_relax` (vs numpy ref) | 0 (bit-exact) |
| End-to-end 50-step run (HD189) | 1.59e-10 |

Run the full test suite with:
```bash
pytest tests/                  # ~60 s with ../VULCAN-master/ present: 27 pass.
                               # without it: 17 pass + 10 skip cleanly (those
                               # 10 compare JAX outputs to upstream's NumPy
                               # reference; without upstream, they're skipped
                               # with a clear "VULCAN-master oracle absent" reason).
python tests/test_foo.py       # individual scripts still work standalone
```

VULCAN-JAX is fully **standalone** — `python vulcan_jax.py` runs end-to-end
with no `../VULCAN-master/` sibling. The optional sibling, when present,
serves as a validation oracle for the 10 upstream-comparison tests.

`pytest -n auto` is **not** safe — see `pytest.ini`; FastChem writes to a fixed output path and parallel runs collide. Run serially.

## Status

Phases 0–10 are **complete**. See `STATUS.md` for the detailed breakdown.

**Done**:
- Phase 0: Scaffolding + network parser
- Phase 1: Atmosphere, rate coefficients, Gibbs energy, initial abundances
- Phase 2: Pure-JAX chemistry RHS + autodiff Jacobian
- Phase 3: Diffusion operator + block-tridiagonal Jacobian assembly
- Phase 4: Block-Thomas solver + Ros2 step kernel
- Phase 5: Photochemistry kernels (`photo.py`)
- Phase 6: End-to-end forward model + .vul output
- Phase 7: JIT + vmap parallelism, GPU readiness
- Phase 8: Condensation + ionchemistry + low-T rates (inherited from VULCAN-master)
- Phase 9: JAX-native `chem_funs.py`, photochem wired into `Ros2JAX`,
  per-step step control ported, `solver_fix_all_bot` ported
- Phase 10.1: JAX outer-loop foundation — `lax.while_loop` for accept/reject + step_size; `op.Ros2` subclass dropped; non-Ros2 fallback removed
- Phase 10.2: Photochemistry update inside the JAX runner (carry-resident photo state)
- Phase 10.3: `update_mu_dz` + `update_phi_esc` inside the JAX runner; hydrostatic balance per step
- Phase 10.4: Condensation kernels (`conden.py`) inside the JAX runner
- Phase 10.5: Convergence + stop check inside `cond_fn`; ring-buffered history; adaptive rtol + photo-frequency switch in the body; one-shot runner
- Phase 10.6: Ion charge balance + `fix_all_bot` post-step clamp; `op.Integration` parent dropped; `jax_integrate_adaptive` removed
- Phase 10.7: Pytest enablement — `pytest.ini`, `tests/conftest.py`, thin `def test_main(): assert main() == 0` wrappers; `pytest tests/` runs all 26 tests
- Phase A (standalone refactor): vendored `atm/`, `thermo/`, `fastchem_vulcan/` data files (~28 MB, HD189-only); vendored `op.ReadRate` + `op.Output` into `legacy_io.py`; dropped `sys.path.append(VULCAN_MASTER)` from runtime code; made oracle tests skip cleanly when `../VULCAN-master/` is absent. Result: VULCAN-JAX runs end-to-end with no upstream sibling.
- Phase 11 (sparse analytical Jacobian): replaced `jax.jacrev(chem_rhs_per_layer)` with `chem.chem_jac_analytical`, a stoichiometry-driven build that scatters analytical partials into the (ni, ni) Jacobian via `segment_sum`. chem_jac 95 ms → 2.6 ms (36×); full step 149 ms → 47 ms (3.2×). Bit-exact (≤1e-13) vs the AD path; new `tests/test_chem_jac_sparse.py` validates.

**Open follow-ups** (nice-to-have, not blocking):
- pmap multi-CPU production wiring (architecture supports it; not benchmarked end-to-end)
- Mid-run `fix_species` trigger (Earth/Jupiter only — HD189 has `fix_species=[]` so dormant; runs that would fire it print a one-time warning)
- `compute_Jion` (photoionisation rates) — out of scope for HD189; raises `NotImplementedError` if used
- Per-test fixture sharing in `conftest.py` (cosmetic — current wrappers work)
- Live plotting / movie output verification

## License

VULCAN-JAX inherits its license from VULCAN (GPLv3, see VULCAN-master/GPL_license.txt).

## Citation

If you use VULCAN-JAX, please cite the original VULCAN papers (Tsai et al. 2017, 2021).
