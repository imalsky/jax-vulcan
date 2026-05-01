# VULCAN-JAX

A JAX-accelerated, differentiable port of [VULCAN](https://github.com/exoclime/VULCAN) — the photochemical-kinetics solver for exoplanet atmospheres (Tsai et al. 2017, 2021).

VULCAN-JAX runs supported VULCAN calculations with the same configuration files, input data, and public `.vul` output schema as the upstream NumPy code. The hot path is a single JIT-compiled `lax.while_loop` running on CPU or GPU; the runtime is **standalone** — `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling required.

**Why use VULCAN-JAX over upstream VULCAN? (In dev, still checking all this)**
- **About 3× faster** for the HD189 Ros2 step on this CPU host in the latest `benchmarks/bench_step.py` run in the `vulcan` env.
- **Differentiable where the runtime is JAX**: forward-mode through the runner; reverse-mode through implicit steady-state gradients; raw readers and FastChem remain host-side.
- **Same config format and `.vul` output**: VULCAN's `plot_py/` scripts and downstream tooling work unmodified.
- **Vectorizable**: tested `vmap` support for per-step batched inputs (e.g. parameter sweeps).
- **GPU**: Not tested, but designed for this.

---

## Table of contents

1. [Quickstart](#quickstart)
2. [Installation](#installation)
3. [Capabilities](#capabilities)
4. [Configuration](#configuration)
5. [API overview](#api-overview)
6. [Comparison to VULCAN-master](#comparison-to-vulcan-master)
7. [Differentiability (forward & reverse-mode)](#differentiability-forward--reverse-mode)
8. [Architecture & file map](#architecture--file-map)
9. [Benchmarks](#benchmarks)
10. [Validation: what is and isn't tested](#validation-what-is-and-isnt-tested)
11. [Numerical notes (chem_rhs floor, step-count drift, atom conservation)](#numerical-notes)
12. [GPU & multi-CPU](#gpu--multi-cpu)
13. [Running tests](#running-tests)
14. [License & citation](#license--citation)

---

## Quickstart

```bash
cd VULCAN-JAX/

# 1. Edit vulcan_cfg.py exactly as you would VULCAN-master's. Same format,
#    same keys. Vendored presets live in cfg_examples/ for Earth, Jupiter,
#    HD189, and HD209.
cp cfg_examples/vulcan_cfg_HD189.py vulcan_cfg.py

# 2. Run the forward model:
python vulcan_jax.py

# 3. Output lands at output/<out_name>.vul. Same pickle schema as
#    VULCAN-master, so all of VULCAN's plot_py/ scripts work unmodified:
python plot_py/plot_vulcan.py output/HD189.vul

# 4. (Optional) compare against a parallel VULCAN-master run:
python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul
```

The `-n` flag on `vulcan_jax.py` is accepted as a no-op for upstream-CLI compatibility (`chem_funs.py` is JAX-native — no SymPy regeneration step exists).

---

## Installation

VULCAN-JAX does not require a pre-existing `vulcan` environment. On a new
machine, clone or copy this repository, install Miniforge/Conda, then create a
fresh Python environment from inside `VULCAN-JAX/`:

```bash
conda create -n vulcan-jax python=3.11 -y
conda activate vulcan-jax

python -m pip install --upgrade pip
python -m pip install jax numpy scipy h5py matplotlib pillow
python -m pip install pytest pytest-xdist ruff vulture
```

The first `pip install` line is the runtime plus plotting stack. The second
line is only needed for tests and development tools. For an NVIDIA GPU machine,
install the platform-specific JAX wheel instead of plain `jax` (for example
`jax[cuda13]` or the CUDA version recommended by the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

No sibling `../VULCAN-master/` checkout is needed for normal runs. It is only
used by optional validation tests that compare against upstream VULCAN.

**Vendored runtime data** (so VULCAN-JAX is fully standalone):
- `thermo/` — chemistry network files, NASA-9 thermodynamic data, photo cross sections.
- `atm/` — TP/Kzz tables and stellar-flux files.
- `cfg_examples/` — example configs for Earth, Jupiter, HD189, HD209.
- `fastchem_vulcan/` — FastChem binary + input/output payload for `ini_mix='EQ'`.

---

## Capabilities

VULCAN-JAX implements the full Ros2 runtime path of upstream VULCAN. Every live runtime branch in master has a JAX implementation here (I think).

**Chemistry & physics:**
- Ros2 (2nd-order Rosenbrock) integration with adaptive timestep
- Chemistry RHS + analytical Jacobian (stoichiometry-driven, ~36× faster than `jax.jacrev`)
- Eddy + molecular diffusion, settling, top/bottom boundary fluxes
- Photochemistry (two-stream, T-dependent cross sections, Rayleigh scattering)
- Ion chemistry with charge balance
- Condensation (H2O, NH3, H2SO4, S2, S4, S8, C) with cold-trap relaxation
- Hydrostatic balance, mean-mass refresh

**Pre-loop setup (also JAX-native where useful):**
- VULCAN-format config loading & parsing
- Forward + reverse rate setup (Gibbs/K_eq from NASA-9)
- 5 initial-abundance modes: `EQ` (FastChem), `vulcan_ini`, `table`, `const_mix`, `const_lowT`
- 5 atmosphere modes: `file`, `isothermal`, `analytical`, `vulcan_ini`, `table`
- 4 Kzz profile modes: `const`, `file`, `Pfunc`, `JM16`

**Outputs:**
- VULCAN-compatible `.vul` pickle (same public keys, shapes, dtypes — `plot_py/` scripts work unchanged)
- `save_evolution` ring buffer for trajectory snapshots
- Synthesized photo/ion diagnostics (`J_sp`, `Jion_sp`, etc.)
- Live UI hooks (mixing-ratio plot, flux plot, movie frames) — fired host-side between JIT'd step batches

**Differentiability:**
- Forward-mode (`jvp`, `jacfwd`) directly across the runner
- Reverse-mode (`grad`, `vjp`) via implicit-function-theorem `custom_vjp` on the converged steady state — **O(1) memory in step count**
- Differentiable surface: rate constants, T/P/Kzz, boundary fluxes, photo cross sections, initial conditions

**Vectorization:**
- `vmap`-able per-step kernels (parameter sweeps, ensemble runs)
- Single `lax.while_loop` driver — no Python `while not stop()` polling

---

## Configuration

VULCAN-JAX reads the same `vulcan_cfg.py` format as upstream VULCAN — a Python module with named attributes. Drop in your existing config; it should work as-is.

**JAX-only additions (with sensible defaults so you can ignore them):**

| Key | Default | Purpose |
|---|---|---|
| `batch_steps` | `1` | Accepted Ros2 steps per JAX runner call. Held at 1 in production. |
| `batch_max_retries` | `64` | Cap on inner accept/reject retries per accepted step. Mirrors master's force-accept fallback. |
| `conv_stall_window` | `200` | Stall-detector window (see [Numerical notes](#numerical-notes)). |
| `conver_ignore` | `[heavy hydrocarbons]` | Species excluded from the convergence test. Pre-populated in HD189/HD209 example cfgs to mitigate ULP-floor stalling on cancellation-prone radicals. |
| `rtol_min` / `rtol_max` | `0.02` / `2.5` | Bounds for adaptive rtol (only applied when `use_adapt_rtol=True`). |
| `adapt_rtol_*`, `photo_switch_*`, `hycean_pin_time` | varies | Per-knob defaults documented in `cfg_examples/`. |

**Non-default but supported config branches** (implemented in JAX, exercised partially by the bundled tests):

- **Atmosphere**: every `atm_type` / `Kzz_prof` / `vz_prof` mode, every `atm_base` (`H2`/`N2`/`O2`/`CO2`).
- **Transport**: `use_moldiff`, `use_vm_mol`, `use_settling`, `use_topflux`, `use_botflux`, `use_fix_sp_bot`, `use_fix_H2He`, `use_sat_surfaceH2O`, `diff_esc`, `max_flux`.
- **Photo/ion**: `use_photo`, `use_ion`, `T_cross_sp`, `scat_sp`, `remove_list`, `ini_update_photo_frq → final_update_photo_frq` switching.
- **Condensation**: every supported condensate species, `use_relax`, `fix_species`, `fix_species_from_coldtrap_lev`, `post_conden_rtol`, `start_conden_time` / `stop_conden_time`.
- **Live UI**: `use_live_plot`, `use_live_flux`, `use_save_movie`, `use_flux_movie` (host-side; force chunked execution).

I haven't tested every single branch. Please let me know if you find a major difference between this and VULCAN.

---

## API overview

There are three layers of API, in increasing order of detail. Most users only need the top one.

### 1. Driver script (`python vulcan_jax.py`)

The simplest way to run VULCAN-JAX. Reads `vulcan_cfg.py`, runs the integration to convergence, writes `output/<out_name>.vul`. ~90 lines:

```python
# vulcan_jax.py (simplified)
runstate = state.RunState.with_pre_loop_setup(vulcan_cfg)  # full pre-loop setup
solver   = op_jax.Ros2JAX()
integ    = outer_loop.OuterLoop(solver, output)
runstate = integ(runstate)                                 # JIT'd integration
output.save_out(runstate, dname)                           # .vul pickle
```

### 2. Programmatic Python API

For embedding VULCAN-JAX in a larger workflow (retrievals, parameter sweeps, optimization).

#### Building a run state

`state.RunState` is the canonical input pytree. `RunState.with_pre_loop_setup(cfg)` runs the entire pre-loop pipeline and returns a fully-populated typed pytree:

```python
import vulcan_cfg
from state import RunState

rs = RunState.with_pre_loop_setup(vulcan_cfg)

# rs.atm           — AtmInputs (Tco, pco, Kzz, M, mu, dz, ...)
# rs.rate          — RateInputs (k_arr, Rf, n_branch, ...)
# rs.photo         — PhotoInputs (sflux, top_flux, ...)
# rs.photo_static  — PhotoStaticInputs (cross sections, branch indices)
# rs.ini_abun      — IniAbunOutputs (y_ini, ymix_ini, atom_ini)
# rs.step          — StepInputs (y, ymix, t, dt, longdy, ...)
# rs.params        — ParamInputs (count, end_case, where_varies_most, ...)
# rs.atoms         — AtomInputs (atom_loss, atom_loss_prev, ...)
# rs.metadata      — host-side static (Rf strings, photo_sp, gas_indx, ...)
```

Every leaf in `rs.atm` / `rs.rate` / `rs.photo_static` / `rs.ini_abun` is a JAX array. Inputs supplied through these pytrees are on the differentiable runtime surface; raw file readers and FastChem are host-side setup.

#### Running the integration

```python
from outer_loop import OuterLoop
import op_jax

solver = op_jax.Ros2JAX()
integ  = OuterLoop(solver, output)
rs_out = integ(rs)                  # one JIT'd lax.while_loop, on device

# rs_out.params.count       — total accepted steps
# rs_out.params.end_case    — 1=converged, 2=runtime cap, 3=count_max cap
# rs_out.step.y             — final number densities (nz, ni)
# rs_out.step.ymix          — final mixing ratios
# rs_out.atoms.atom_loss    — column atom drift per atom
```

#### Per-step kernel (for custom drivers)

`jax_step.jax_ros2_step` is `@jax.jit`'d, vmap-able, and GPU-ready:

```python
from jax_step import jax_ros2_step

# Inputs: y, k_arr (rate table), dt, atm_static (closed-over geometry)
sol, delta = jax_ros2_step(y, k_arr, dt, atm_static, net)
# sol   : (nz, ni) — proposed next state
# delta : (nz, ni) — truncation-error proxy (sol - yk2)
```

#### Steady-state gradient API

`steady_state_grad.py` exposes a `jax.custom_vjp` that uses the implicit function theorem — the right way to get reverse-mode gradients through the converged state:

```python
from steady_state_grad import (
    build_steady_state_inputs,
    steady_state_value_and_grad,
    validate_steady_state_solution,
)

# 1. Run forward to convergence
y_star = run_outer_loop(k_arr, atm_static)

# 2. Validate the converged residual is small (gradient accuracy
#    is bounded by ||f(y*)||).
validate_steady_state_solution(y_star, inputs, net,
                               residual_rtol=1e-6)

# 3. Get value and gradient of a scalar loss
inputs = build_steady_state_inputs(k_arr, atm_static)
def loss_fn(y): return some_scalar(y)

loss, grad = steady_state_value_and_grad(
    loss_fn, inputs, y_star, net, residual_rtol=1e-6
)

g_k_arr = grad.k_arr        # gradients per-input-leaf
```

Memory cost is **O(1) in step count** — no checkpointing.

### 3. Low-level functional API

For when you need to bypass the typed pytree:

| Function | Purpose |
|---|---|
| `chem.chem_rhs(y, k_arr, net, atm_static)` | Chemistry RHS (vmapped over layers). |
| `chem.chem_jac_analytical(y, k_arr, net, atm_static)` | Stoichiometry-driven analytical Jacobian. |
| `solver.factor_block_thomas_diag_offdiag(...)` / `solve_*` | Block-tridiagonal factor + back-substitute. |
| `photo.compute_tau` / `compute_flux` / `compute_J` | Two-stream photochemistry kernels. |
| `atm_refresh.update_mu_dz_jax` / `update_phi_esc_jax` | Hydrostatic balance + escape flux update. |
| `conden.update_conden_rates` / `apply_h2o_relax_jax` / `apply_nh3_relax_jax` | Condensation kernels. |
| `rates.build_rate_array(...)` | Forward rate-coefficient table. |
| `gibbs.compute_K_eq(...)` | Equilibrium constants from NASA-9 polynomials. |

All are jit/vmap/jvp/vjp compatible.

---

## Comparison to VULCAN-master

VULCAN-JAX is intended as a drop-in replacement for the supported Ros2 path. The compatibility surface:

| Surface | Compatible? | Notes |
|---|---|---|
| `vulcan_cfg.py` format | yes — same keys, same format | JAX-only knobs are documented above; defaults match. |
| Network files (`thermo/*.txt`) | yes — same parser | Vendored from upstream. |
| Atmosphere files (`atm/*.txt`) | yes — same parser | Vendored. |
| Photo cross-section files | yes — same parser | Vendored. |
| FastChem subprocess (`ini_mix='EQ'`) | yes — same binary, same I/O | External subprocess. |
| `.vul` output schema | yes — same public keys, shapes, dtypes | Pickle bytes are not byte-identical. |
| `plot_py/` scripts | yes — unchanged | Same data surface. |
| Output writer `vars(data_var)` filtered by `var_save` | yes | Same filter. |
| `parameter` keys (`end_case`, `count`, `where_varies_most`, `pic_count`, `tableau20`, ...) | yes | All master public keys published. |
| Solver | partial — Ros2 only | `SemiEU` etc. are commented-out stubs in master, not ported. |
| `chem_funs.symjac` / `neg_symjac` | no — raise `NotImplementedError` | Replaced by `chem.chem_jac_analytical`. |
| `make_chem_funs.py` | no — no-op shim | Production path is JAX-native; no SymPy codegen. |
| Live plot cadence | yes — `live_plot_frq` | Master fires inside its Python loop; JAX fires between JIT'd chunks at the same predicate. |
| Byte-identical pickle | no | Public keys/shapes/dtypes match, but dict order and transient histories may not. |

### Intentional behavioral differences

- **Live UI is host-side, fired between JIT'd step batches.** When any of `use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie` is set, the runner switches to chunked execution (chunks of `live_plot_frq` accepted steps), with `live_ui.LiveUI` reading the legacy `(var, atm, para)` view between chunks. Cadence-faithful but not call-site-identical.
- **Output writer synthesizes** `J_sp` / `Jion_sp` / per-reaction `var.k` dicts from the typed JAX state at pickle time rather than incrementally during the run.
- **Convergence-detector stall fallback** (`conv_stall_window`) — see [Numerical notes](#numerical-notes). Master almost never trips it; JAX trips it when a heavy-hydrocarbon trace species oscillates around `yconv_min` for too long.

---

## Differentiability (forward & reverse-mode)

### What is differentiable

The full physical input surface — **atmospheric structure, rate constants, boundary fluxes, photo cross sections, initial conditions** — is differentiable as long as you supply inputs as JAX arrays into the typed pytrees (`AtmInputs` / `RateInputs` / `PhotoStaticInputs` / `IniAbunOutputs`).

The runner's `lax.while_loop` blocks `vjp` directly. There are two ways around that:

### Forward-mode (`jvp` / `jacfwd`) — works through the entire integration

`lax.while_loop` natively supports forward-mode AD:

```python
import jax

def integrate_fn(k_arr):
    rs = build_runstate_from_k(k_arr)          # supply k_arr as a JAX array
    rs_out = integ(rs)
    return rs_out.step.y                        # final state

# Tangent of the converged y* w.r.t. rate constants:
y_star, dy_dk = jax.jvp(integrate_fn, (k_arr,), (k_arr_tangent,))

# Or full forward Jacobian (if input dim is small enough):
J = jax.jacfwd(integrate_fn)(k_arr)
```

Forward-mode is exact (within the per-step ULP floor), but its memory cost is `O(input_dim × output_dim)` — best when input dim is small.

### Reverse-mode (`grad` / `vjp`) — via implicit-function theorem

For high-dimensional inputs (full `k_arr`, photo cross sections), use the implicit-AD route in `steady_state_grad.py`. It's a `jax.custom_vjp` that solves the linear system `(∂f/∂y) z = ∂L/∂y*` once at the converged state, costing **O(1) memory in step count** — no trajectory checkpointing.

Worked example:

```python
import jax
from steady_state_grad import (
    build_steady_state_inputs,
    steady_state_value_and_grad,
    validate_steady_state_solution,
)

# 1. Run the forward integration to a tight residual.
rs_out = integ(rs)
y_star = rs_out.step.y

# 2. Pack the structured inputs and validate convergence.
inputs = build_steady_state_inputs(rs.rate.k, atm_static, photo_static, ...)
validate_steady_state_solution(y_star, inputs, net, residual_rtol=1e-6)

# 3. Define a scalar loss on y*.
def transit_depth_residual(y):
    return jnp.sum((depth_model(y) - depth_obs) ** 2 / sigma ** 2)

# 4. value_and_grad through the converged state.
loss, grad_inputs = steady_state_value_and_grad(
    transit_depth_residual, inputs, y_star, net, residual_rtol=1e-6
)

g_rates = grad_inputs.k_arr
g_atm   = grad_inputs.atm  # gradients on T/P/Kzz/etc.
```

**Important**: gradient accuracy is bounded by the residual `||f(y*)||`. The default `yconv_cri = 0.01` is too loose for retrieval; tighten the convergence criterion when calling for gradients. See `tests/test_steady_state_grad.py` for the canonical pattern.

### What's NOT differentiable

These are host-side setup steps, by design:

- `photo_setup.py` — cross-section CSV reader. To differentiate cross sections, build `PhotoStaticInputs` directly from JAX arrays.
- `legacy_io.ReadRate.read_rate` — rate-file metadata parser (rate *values* flow through differentiable `rates.build_rate_array`).
- `composition.py`, `atm_setup.py`, `ini_abun.py` raw-file readers.
- FastChem subprocess.

If you want gradients through one of these, the answer is almost always: build the corresponding pytree with JAX arrays directly and inject.

### Vectorization (`vmap`)

Per-step kernels are `vmap`-able directly:

```python
# Run 16 atmospheres at once with different stellar fluxes
batched_y = jax.vmap(jax_ros2_step, in_axes=(0, 0, None, None, None))(
    y_batch, k_arr_batch, dt, atm_static, net
)
```

For full integration sweeps, see `examples/batched_run.py`.

---

## Architecture & file map

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ vulcan_jax.py  (entry point, ~90 lines)                     │
│   ├─ JAX-native chem_funs (no make_chem_funs.py)            │
│   ├─ atm_setup.Atm (atmosphere setup) — JAX-native          │
│   ├─ legacy_io.ReadRate (rate metadata) — vendored          │
│   ├─ ini_abun.InitialAbun (5 ini_mix modes) — JAX-native    │
│   ├─ photo_setup.populate_photo (cross sections) — NumPy    │
│   ├─ Ros2JAX.compute_tau/flux/J — one-shot pre-loop         │
│   └─ outer_loop.OuterLoop (JAX) — single-shot runner        │
│       └─ jax.jit(while_loop):                               │
│            cond_fn:  count_max | runtime | converged | stall│
│            body_fn (one accept, with internal retries):     │
│              ├─ photo branch (lax.cond on update_photo_frq) │
│              │   compute_tau / flux / J → update k_arr      │
│              ├─ atm refresh (lax.cond on update_frq)        │
│              │   update_mu_dz / phi_esc → splice into atm   │
│              ├─ jax_ros2_step (chem_rhs + analytical_jac,   │
│              │     diffusion, block_thomas) — JAX           │
│              ├─ clip / loss / step_ok / step_size — JAX     │
│              ├─ conden branch (lax.cond on t / use_relax)   │
│              ├─ hydrostatic balance + ion + fix_all_bot     │
│              ├─ ring-buffer y_time / t_time history         │
│              ├─ adaptive rtol + photo-freq switch           │
│              └─ in-runner conv check → longdy/longdydt      │
└─────────────────────────────────────────────────────────────┘
```

### File map

```
VULCAN-JAX/
├── vulcan_jax.py        Entry point; mirrors vulcan.py orchestration
├── vulcan_cfg.py        VULCAN-format config (drop in your own)
├── outer_loop.py        OuterLoop — standalone single-shot JAX runner
├── op_jax.py            Ros2JAX — standalone photo adapter
├── jax_step.py          Pure-JAX vmap'able Ros2 step (incl. JAX diffusion)
├── solver.py            Block-tridiagonal Thomas solver
├── chem.py              JAX chemistry RHS + analytical Jacobian
├── photo.py             JAX two-stream photochem (tau / flux / J kernels)
├── steady_state_grad.py Implicit-function-theorem custom_vjp for reverse-mode AD
├── runtime_validation.py Pre-run runtime/config validator
├── atm_refresh.py       JAX update_mu_dz + update_phi_esc
├── conden.py            JAX condensation rates + cold-trap relax
├── rates.py             Forward rate coefficients
├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
├── network.py           Network parser (text → stoichiometry tables)
├── integrate.py         Pure-JAX fixed-dt scan loop (validation/benchmarks)
├── legacy_io.py         Vendored op.ReadRate + op.Output (.vul writer)
├── atm_setup.py         JAX-native Atm.* (f_pico / load_TPK / mol_diff / ...)
├── ini_abun.py          JAX-native InitialAbun.* (5 ini_mix modes)
├── photo_setup.py       Host-side cross-section preprocessing
├── composition.py       Per-species composition / mass tables
├── state.py             Typed pytrees: AtmInputs / RateInputs /
│                         PhotoInputs / PhotoStaticInputs /
│                         IniAbunOutputs / RunState
├── chem_funs.py         JAX-native module (re-exports network/gibbs/chem
│                         API as ni/nr/spec_list/Gibbs/chemdf, no SymPy)
├── live_ui.py           Host-side live-plot dispatcher (matplotlib + PIL)
├── phy_const.py         Physical constants (vendored from VULCAN-master)
├── pytest.ini           Pytest config (parallel-safe via FastChem flock)
├── atm/, thermo/, fastchem_vulcan/   Vendored runtime data
├── cfg_examples/        Earth / Jupiter / HD189 / HD209 example configs
├── benchmarks/          Bench and timing utilities
├── tests/               Validation suite + standalone scripts
└── CLAUDE.md            Maintenance / numerical-hygiene spec for AI agents
```

### Design choices worth knowing

- **float64 is non-negotiable.** `jax.config.update('jax_enable_x64', True)` is set at every module import. Rate constants span ~50 orders of magnitude; float32 silently fails.
- **Analytical chemistry Jacobian, not `jax.jacrev`.** `chem.chem_jac_analytical` is a stoichiometry-driven scatter that skips structurally-zero entries; ~36× faster than the AD path. The AD path stays as a test oracle.
- **Diagonal-aware block-tridiagonal solver.** `solver.block_thomas_diag_offdiag` exploits diagonal-in-species off-blocks: the dense `O(ni³)` matmul reduces to an `O(ni²)` rank update.
- **Single-shot JIT'd runner.** Photo, atm-refresh, condensation, ion balance, fix-all-bot, adaptive rtol, photo-frequency switch, and the convergence check all live inside one `lax.while_loop` body. No Python step polling.
- **Implicit-AD for reverse-mode.** `lax.while_loop` blocks `vjp` directly; the implicit-function-theorem `custom_vjp` in `steady_state_grad.py` is the supported route.
- **Typed pytree as runtime surface.** `state.RunState` is the canonical shape. The legacy mutable container classes (`Variables` / `AtmData` / `Parameters`) live as private `_Variables` / `_AtmData` / `_Parameters` for hybrid oracle tests only.

---

## Benchmarks

Per-step kernel timing on the HD189 reference state from `python benchmarks/bench_step.py` (CPU, single-threaded):

| Step | Master (NumPy) | VULCAN-JAX | Speedup |
|---|---:|---:|---:|
| Single Ros2 step (photo + chem + diffusion + block-Thomas) | 152.3 ms | **45.1 ms** | **3.4×** |
| 50-step OuterLoop cached call (HD189) | — | 49.6 ms / accepted step | — |

Wall-time speedup depends on whether the convergence detector takes the same path on both branches (see [Numerical notes](#numerical-notes)); rerun the benchmark locally before quoting a number for another machine.

**GPU**: the architecture is fully `jit` / `vmap` compatible; setting `JAX_PLATFORM_NAME=gpu` runs on GPU with no code changes. Not measured on this host (CPU-only machine), but architectural overhead at scale is dominated by the chemistry RHS and block-Thomas solver, both of which are designed to vectorize.

**Where the speedup comes from:**
1. Analytical chemistry Jacobian (`chem_jac_analytical` vs `jacrev`) — 95 ms → 2.6 ms
2. Diagonal-aware block-Thomas (`block_thomas_diag_offdiag`) — `O(ni³)` → `O(ni²)` rank update for diffusion off-blocks
3. JIT compilation of the entire integration loop into one XLA graph — no Python overhead per step
4. Pre-baked y-independent diffusion terms (computed once per Ros2 step instead of twice)

Run `python benchmarks/bench_step.py` for a fresh per-step timing on your hardware.

---

## Validation: what is and isn't tested

### Numerical agreement vs VULCAN-master (per-component)

| Layer | Agreement |
|---|---|
| Forward rate coefficients (596 reactions) | bit-exact (relerr = 0) |
| Reverse rates (533 from Gibbs) | 1.4e-14 |
| Atmosphere structure (pco/Tco/Kzz/M/...) | bit-exact |
| Initial abundances (FastChem path) | bit-exact |
| Chemistry RHS (`chem_rhs` vs `chemdf`) | ~1e-4 for 6 cancellation-prone species, machine-precision otherwise |
| Chemistry Jacobian (vs `chem_funs.symjac`) | 4.3e-13 |
| Diffusion operator (vs `op.diffdf`) | 2e-6 (FP-noise-bound) |
| Block-Thomas solver | 3e-15 |
| Single Ros2 step (vs `op.Ros2.solver`) | 1.16e-15 |
| `compute_tau` / `compute_flux` / `compute_J` | 8e-16 / 3.7e-11 / 1.8e-11 |
| `compute_Jion` / ion `k_arr` wiring | unit-tested end-to-end |
| `update_mu_dz` / `update_phi_esc` | 3-7e-16 (bit-exact) |
| `conden` / `h2o_relax` / `nh3_relax` | 0 (bit-exact) |
| End-to-end 50-step run (HD189) | 1.59e-10 |
| End-to-end converged HD189 (median dex) | 0.004 dex (~1% relative) |

### What's covered by the test suite

`pytest tests/` runs 108 tests covering:
- JAX↔master numerical bridge (RHS, Jacobian, diffusion, single Ros2 step, photo kernels)
- `vmap` consistency (single-call vs batched output)
- Forward-mode AD (`jvp` through per-step kernels)
- Reverse-mode AD via `steady_state_grad` (validated against finite differences)
- HD189 smoke integration (50-step regression oracle)
- 20-step matched-step oracles for Earth, Jupiter, HD209
- `save_evolution` round-trip
- `.vul` output schema & RunState round-trip
- Vendored example-config setup
- Optional CPU↔GPU parity (skips on CPU-only hosts)

### What's NOT tested

- **Cartesian-product oracle sweeps** over every non-default config knob. By policy.
- **GPU parity** on this CPU-only host (skipped).
- **Long-to-convergence VULCAN-master oracles** for every vendored example.
- **Arbitrary custom networks** beyond parser/schema coverage and the bundled examples.
- **Gradients through host-side readers / FastChem internals** — by design (host-side setup, not on the AD path).
- **Invalid master configurations** that the validator rejects (`use_ion=True` without `use_photo=True`, `use_live_flux=True` without `use_photo=True`, `fix_species` without condensation).

---

## Numerical notes

### `chem_rhs` cancellation floor (~1e-4 relerr for 6 species)

The JAX path uses `jnp.prod(y**stoich)` while VULCAN-master uses SymPy-emitted `y[A] * y[B] * y[C]`. The two differ by ~1 ulp per multiply; with ~7K production/loss terms cancelling down to ~1e2 the ulp drift accumulates to ~1e-4 absolute. This is a real, measurable disagreement for `CH2_1`, `HC3N`, `HCCO` (and a handful of heavy hydrocarbons at certain layers).

Closing it requires a SymPy-faithful codegen and pays a ~14× per-call cost on the per-step hot path. Current code is not trying to fix this. The Jacobian (`chem_jac_analytical`) is unaffected.

### Step-count and atom-conservation drift (downstream of the floor above)

The 1e-4 floor is invisible at the per-step level but compounds over a long integration in two ways:

**1. JAX version may need more accepted steps to detect convergence than master.**
The convergence test fires when `longdy = max|Δy/(n_0·ymix)|` drops below `yconv_min = 0.1` and `longdydt < slope_min`. The ULP floor doesn't move bulk species (H2O, CO, CH4, NH3, HCN), but it nudges heavy-hydrocarbon trace radicals (`C6H6`, `C2H2`, `C4H5`, `C4H2`, `C3H3`, `CH3NH2`, ...) along slightly different trajectories. Whichever one is sitting at the threshold last gates termination. Both runs reach physically equivalent steady states; only the detection moment differs.

**2. JAX's column atom_loss grows roughly linearly with step count.**
With `loss_eps = 0.1`, neither code rejects the per-step ulp drift. A 2-3× longer JAX run ends with 2-3× more cumulative atom drift. On HD189 both runs sit at ~2e-4 column drift; on HD209 with no mitigation, JAX accumulated to −8.3% C while master sat at +5e-5. I need to look into this one.

### Mitigations (no code changes, just config knobs)

- **`conver_ignore`** (populated by default in `cfg_examples/vulcan_cfg_HD189.py` / `vulcan_cfg_HD209.py`):
  ```python
  conver_ignore = ['C6H6', 'C2H2', 'C6H5', 'C2H', 'C2H4', 'C2H5', 'C2H6',
                   'C3H2', 'C3H3', 'C4H5', 'CH2NH', 'CH3NH2', 'H2CCO']
  ```
  Heavy-hydrocarbon trace radicals excluded from the convergence detector. If a *new* trace radical takes over the gate on a different planet, look at `parameter['where_varies_most']` in the saved `.vul` and add it.

- **`conv_stall_window = 200`** (new safety net, default):
  Stall fallback in both branches. If `longdy_seen_min` (running min of `longdy`, only resets on a ≥5% relative drop) has been below `yconv_min` for 200 accepted steps without significant improvement *and* current `longdy` is also below `yconv_min`, declare `end_case=1`. Master almost never trips it; JAX trips it when a heavy hydrocarbon outside `conver_ignore` keeps oscillating around the threshold.

### What does NOT help

- **Tighter `loss_eps`** (e.g. 1e-5): drift is per physical time, not per step. Tighter `loss_eps` causes dt-thrashing without reducing cumulative drift.
- **Compensated summation** (Kahan/Neumaier in `chem_rhs`): empirically verified bit-identical to `math.fsum` on JAX's terms — the disagreement is in the per-term *values* JAX emits, not in the summation order. Adds ~14× to chem_rhs runtime for zero gain.
- **float32**: hard no — rate constants span 50 orders of magnitude.

### Other documented numerical points

- **Diffusion Jacobian** matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at heavy-condensable cells (S8 layers 5/25). Direct comparison with the analytical derivative confirms the JAX side is correct; master's `op.lhs_jac_tot` has a minor self-inconsistency.
- **Asymmetric M factor for dissociation reactions.** `network.py` tracks `has_M_reac` and `has_M_prod` separately so reactions like `HNCO + M → H + NCO` (3-body forward, bimolecular reverse) are handled correctly.

---

## GPU & multi-CPU

```bash
# Run on GPU (no code changes)
JAX_PLATFORM_NAME=gpu python vulcan_jax.py

# Enable multiple host CPU devices for tested vmap workflows
XLA_FLAGS=--xla_force_host_platform_device_count=8 python vulcan_jax.py
```

For batched parameter sweeps (e.g. running 16 atmospheres at once with different stellar fluxes), see `examples/batched_run.py`.

---

## Running tests

```bash
python -m pytest tests -q --tb=short -ra   # full suite. 108 passed + 3 skipped
                               # on a clean CPU-only run.
python -m pytest tests -n auto -q --tb=short -ra
                               # parallel-safe (FastChem invocations
                               # serialise via fcntl.flock).
python -m pytest tests -k "ros2 or block_thomas"   # filter
python tests/test_foo.py       # individual scripts run standalone
```

The 3 documented skips are: `test_backend_parity` (no GPU on this host) and 2 config-matrix sub-cases that require `H2O_l_s` in the network (HD189 doesn't have it). Upstream-comparison tests skip cleanly when `../VULCAN-master/` is absent.

The test suite is deliberately not parametrized across every non-default config combination — see [Capabilities](#capabilities) for the full inventory.

See `CLAUDE.md` for the full test-discipline rules (when to add a test, what a "real" failure looks like).

---

## License & citation

VULCAN-JAX inherits its license from VULCAN (GPLv3, see `VULCAN-master/GPL_license.txt`).

If you use VULCAN-JAX in published work, please cite the underlying VULCAN papers:

- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20
- Tsai, S.-M., Malik, M., Kitzmann, D., Lyons, J. R., Fateev, A., Lavvas, P., & Heng, K. 2021, ApJ, 923, 264
