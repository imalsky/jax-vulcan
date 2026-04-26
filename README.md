# VULCAN-JAX

A JAX-accelerated, drop-in compatible port of [VULCAN](https://github.com/exoclime/VULCAN), the chemical-kinetics code for exoplanet atmospheres.

VULCAN-JAX runs the same calculation as VULCAN — same configuration files, same input data, same `.vul` output schema — but the inner Rosenbrock step (chemistry RHS, autodiff Jacobian, block-tridiagonal solve) is JAX-accelerated. The result is a forward model that's slightly faster on a single CPU than VULCAN-master AND directly usable on GPU/TPU AND `jax.vmap`-able for batched parameter sweeps.

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

The full per-step path is JAX (post-Phase 9):

- **Chemistry RHS** (`chem.chem_rhs`) — vectorized scatter-add over the reaction network, vmapped over vertical layers. ~1 ms/call for nz=120, ni=93.
- **Chemistry Jacobian** (`chem.chem_jac`) — `jax.jacrev` on the per-layer rate function, vmapped. ~150 ms/call.
- **Diffusion operator + block Jacobians** — eddy + molecular diffusion, JAX inline inside `jax_step.py`.
- **Block-tridiagonal solver** (`solver.block_thomas`) — two `jax.lax.scan`s for forward elim + back sub. ~16 ms.
- **2nd-order Rosenbrock step** (`jax_step.jax_ros2_step`) — fully `@jax.jit`'d; vmap-able and GPU-ready.
- **Photochemistry** (`photo.compute_tau_jax` / `compute_flux_jax` / `compute_J_jax`) — wired into `Ros2JAX` so production runs use pure-JAX kernels.
- **Per-step step control** (`clip` / `loss` / `step_ok` / `step_reject` / `reset_y` / `step_size`) overridden in `Ros2JAX` with vectorized NumPy/einsum equivalents (JAX-traceable).
- **BC variant** `solver_fix_all_bot` ported to JAX (block-Thomas + post-solve clamp).

Total: ~156 ms/step warm — slightly faster than VULCAN-master's ~175 ms/step on the same machine.

The integration outer loop (`op.Integration.__call__`, `update_mu_dz`, `conden`, …) is still inherited from VULCAN-master and runs every N steps (not per step). Phase 10 (planned) wraps it in `jax.lax.while_loop` for max GPU/TPU performance.

## Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │ vulcan_jax.py  (entry point, mirrors vulcan.py)           │
   │   ├─ JAX-native chem_funs (no SymPy code generator)       │
   │   ├─ build_atm (atmosphere setup) — VULCAN-master copy    │
   │   ├─ ReadRate / InitialAbun — VULCAN-master               │
   │   ├─ photo cross-section read — VULCAN-master             │
   │   └─ op.Integration loop — VULCAN-master (NumPy)          │
   │       └─ Ros2JAX (op_jax.py) — VULCAN-JAX                 │
   │           ├─ compute_tau / flux / J → photo.py (JAX)      │
   │           ├─ clip / loss / step_ok / ... → vectorized NP  │
   │           ├─ solver_fix_all_bot → solver() + bottom clamp │
   │           └─ solver:                                       │
   │               jax_step.jax_ros2_step (JIT'd, vmap-able)   │
   │                ├─ chem.chem_rhs (segment_sum)             │
   │                ├─ chem.chem_jac (vmap'd jacrev)           │
   │                ├─ JAX diffusion inline                    │
   │                └─ solver.block_thomas (lax.scan)          │
   └──────────────────────────────────────────────────────────┘
```

## File map

```
VULCAN-JAX/
├── vulcan_jax.py        Entry point; mirrors vulcan.py orchestration
├── vulcan_cfg.py        VULCAN-format config (drop in your own)
├── op_jax.py            Ros2JAX class (overrides solver, photochem,
│                         step-control, solver_fix_all_bot)
├── jax_step.py          Pure-JAX vmap'able Ros2 step (incl. JAX diffusion)
├── solver.py            Block-tridiagonal Thomas solver
├── chem.py              JAX chemistry RHS + autodiff Jacobian
├── photo.py             JAX two-stream photochem (tau / flux / J kernels)
├── rates.py             Forward rate coefficients
├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
├── network.py           Network parser (text -> stoichiometry tables)
├── integrate.py         Pure-JAX adaptive-dt loop (no-photo path)
├── build_atm.py         VULCAN-master copy (atmosphere setup)
├── store.py             VULCAN-master copy (Variables/AtmData/Parameters)
├── chem_funs.py         JAX-native module (re-exports network/gibbs/chem
│                         API as ni/nr/spec_list/Gibbs/chemdf, no SymPy)
├── phy_const.py         VULCAN-master copy (physical constants)
├── atm/, thermo/, fastchem_vulcan/   Symlinks to VULCAN-master
├── tests/               17 unit + validation tests, all PASS
│   └── diffusion_numpy_ref.py  NumPy reference for diffusion (test-only)
├── STATUS.md            Detailed implementation status
└── notes.md             Design log, architectural decisions
```

## GPU / multi-CPU

To run on GPU (no code changes):
```bash
JAX_PLATFORM_NAME=gpu python vulcan_jax.py -n
```

To enable JAX device-level parallelism for vmap/pmap on multi-core CPU:
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py -n
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
| End-to-end 50-step run (HD189, Phase 9 wired path) | 1.59e-10 |

Run the full test suite with:
```bash
for t in tests/test_*.py; do python "$t" | tail -1; done
```

## Status

Phases 0–9 are **complete**. See `STATUS.md` for the detailed breakdown and `notes.md` for design decisions.

**Done**:
- Phase 0: Scaffolding + network parser
- Phase 1: Atmosphere, rate coefficients, Gibbs energy, initial abundances
- Phase 2: Pure-JAX chemistry RHS + autodiff Jacobian
- Phase 3: Diffusion operator + block-tridiagonal Jacobian assembly
- Phase 4: Block-Thomas solver + Ros2 step kernel
- Phase 5: Photochemistry kernels (`photo.py`)
- Phase 6: End-to-end forward model + .vul output
- Phase 7: JIT + vmap parallelism, GPU readiness
- Phase 8: Condensation + ionchemistry + low-T rates (inherited)
- Phase 9: JAX-native `chem_funs.py`, photochem wired into `Ros2JAX`,
  per-step step control ported, `solver_fix_all_bot` ported

**Open follow-ups** (nice-to-have, not blocking):
- Phase 10: pure-JAX `Integration` outer loop (`jax.lax.while_loop`) for max GPU/TPU performance
- Phase 11: custom analytical sparse Jacobian (3-5x potential speedup for chem_jac); pmap multi-CPU
- Phase 12: live plotting / movie output verification

## License

VULCAN-JAX inherits its license from VULCAN (GPLv3, see VULCAN-master/GPL_license.txt).

## Citation

If you use VULCAN-JAX, please cite the original VULCAN papers (Tsai et al. 2017, 2021).
# jax-vulcan
