# VULCAN-JAX

A JAX-accelerated, differentiable port of [VULCAN](https://github.com/exoclime/VULCAN) — the photochemical-kinetics solver for exoplanet atmospheres (Tsai et al. 2017, 2021).

VULCAN-JAX runs supported VULCAN calculations with the same configuration files, input data, and public `.vul` output schema as the upstream NumPy code. The hot path is a single JIT-compiled `lax.while_loop` running on CPU or GPU; the runtime is **standalone** — `python vulcan_jax.py` runs end-to-end with no `../VULCAN-master/` sibling required.

**Why use VULCAN-JAX over upstream VULCAN?**
- **About 5–8× faster** end-to-end for HD189/HD209 on this CPU host (cold JIT). The per-step kernel is ~3× faster via `benchmarks/bench_step.py`; the full integration speedup is larger because the JIT'd `lax.while_loop` also eliminates Python-loop overhead.
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
11. [Numerical notes (chemistry RHS parity, step-count drift, atom conservation)](#numerical-notes)
12. [GPU & multi-CPU](#gpu--multi-cpu)
13. [Running tests](#running-tests)
14. [License & citation](#license--citation)

---

## Quickstart

```bash
cd VULCAN-JAX/

# 1. Edit vulcan_cfg.py exactly as you would VULCAN-master's. Same format,
#    same keys. The committed default is a thin wrapper over the canonical
#    HD189 preset; vendored presets live in cfg_examples/ for HD189, HD209,
#    Earth, and W39b.
cp cfg_examples/vulcan_cfg_HD189.py vulcan_cfg.py

# 2. Run the forward model:
python vulcan_jax.py

# 3. Output lands at output/<out_name>.vul. Same pickle schema as
#    VULCAN-master — point any of upstream's plot_py/ scripts at this
#    file and they work unmodified.

# 4. (Optional) audit default HD189 config + input parity vs a
#    sibling VULCAN-master checkout:
python tools/audit_master_parity.py --master ../VULCAN-master
```

For a per-file map of every module and what its functions do, see
[`FILE_README.md`](FILE_README.md).

The `-n` flag on `vulcan_jax.py` is accepted as a no-op for upstream-CLI compatibility — `make_chem_funs.build_chem_rhs(net)` runs automatically at `chem_funs` import time and caches the SymPy-faithful per-network RHS in `__pycache__/chem_rhs_codegen_<hash>.py`.

---

## Installation

### pip install (recommended for using as a library)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vulcan-jax
```

Then from any Python script or notebook:

```python
import vulcan_jax

cfg = vulcan_jax.make_config(count_max=100)
rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a worked example.

### From source (for development)

Clone the repo and install in editable mode:

```bash
git clone git@github.com:imalsky/jax-vulcan.git VULCAN-JAX
cd VULCAN-JAX
pip install -e ".[dev]"
```

### Conda environment from scratch

On a new machine without the dependencies:

```bash
conda create -n vulcan-jax python=3.11 -y
conda activate vulcan-jax

python -m pip install --upgrade pip
python -m pip install jax numpy scipy h5py sympy matplotlib pillow
python -m pip install pytest pytest-xdist ruff vulture
```

For an NVIDIA GPU machine, install the platform-specific JAX wheel instead of
plain `jax` (see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

No sibling `../VULCAN-master/` checkout is needed for normal runs. It is only
used by optional validation tests that compare against upstream VULCAN.

**Vendored runtime data** (so VULCAN-JAX is fully standalone):
- `thermo/` — chemistry network files, NASA-9 thermodynamic data, photo cross sections.
- `atm/` — TP/Kzz tables and stellar-flux files.
- `cfg_examples/` — example configs for HD189, HD209, Earth, W39b.
- `fastchem_vulcan/` — FastChem binary + input/output payload for `ini_mix='EQ'`.
  All shipped configs (HD189, HD209, W39b, Earth) point at one canonical
  `solar_element_abundances.dat` (Lodders 2019 / Wogan & Tsai 2023 values
  with rocky elements pinned to -3.0). The shipped NCHO and SNCHO networks
  contain no Mg/Si/Fe species, so leaving those elements at solar dex would
  silently sequester O atoms into gas-phase oxides that `_load_eq_y` cannot
  read back. `runtime_validation._validate_fastchem_input_vs_network` enforces
  this at pre-run time. `tools/audit_master_parity.py` byte-hashes the file
  against the master sibling to catch cross-repo drift.

---

## Capabilities

VULCAN-JAX implements the full Ros2 runtime path of upstream VULCAN. Every live runtime branch in master has a JAX implementation here; see [Validation](#validation-what-is-and-isnt-tested) for which branches are exhaustively cross-tested versus partially exercised.

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
- `save_evolution` ring buffer (last `min(accept_count, conv_step)` accepted steps sampled by `save_evo_frq`) for trajectory snapshots; raise `conv_step` to keep a longer tail.
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

**JAX-only additions (with sensible defaults so you can ignore them).** Every knob below is declared in `vulcan_cfg.py` and validated by `runtime_validation.py`; old user configs that don't declare them inherit the defaults via `getattr(cfg, ..., default)` for back-compat.

| Key | Default | Purpose |
|---|---|---|
| `batch_max_retries` | `64` | Cap on inner accept/reject retries per accepted step. Mirrors master's force-accept fallback (master's `dt_min` underflow path fires at retry 14 for `dttry=1e-10`; 64 is a true safety guard). |
| `conv_stall_window` | `200` | Stall-detector window (see [Numerical notes](#numerical-notes)). |
| `conver_ignore` | `[heavy hydrocarbons]` | Species excluded from the convergence test. Pre-populated in HD189/HD209/W39b example cfgs to mitigate ULP-floor stalling on cancellation-prone radicals. |
| `loss_ex` | `[]` | Atoms excluded from the loss-criteria check (use when an element's column inventory is dominated by a non-conservative source). |
| `rtol_min` / `rtol_max` | `0.0` / `1.0` | Bounds for adaptive rtol (only applied when `use_adapt_rtol=True`). |
| `adapt_rtol_dec_period` / `adapt_rtol_inc_period` | `10` / `1000` | Adaptive-rtol cadence in accepted steps. |
| `adapt_rtol_dec` / `adapt_rtol_inc` | `0.75` / `1.25` | Multiplicative rtol decrease / increase factors. |
| `adapt_rtol_loss_mul` | `2.0` | Loss-criteria relaxation factor on rtol decrease. |
| `adapt_rtol_inc_loss_thresh` | `2e-4` | Max column atom_loss for an rtol increase. |
| `step_size_safety` / `step_size_zero_delta_frac` | `0.9` / `0.01` | Adaptive Ros2 step-size safety factor and zero-delta fallback fraction. |
| `photo_switch_longdy_thresh` / `photo_switch_longdydt_thresh` | `yconv_min*10` / `1e-6` | `update_photo_frq` ramps from `ini_*` to `final_*` when both gates trip. |
| `hycean_pin_time` | `1e6` | After `var.t > hycean_pin_time`, H2/He are pinned via `fix_sp_bot`. |
| `fastchem_solar_abundance_file` | `fastchem_vulcan/input/solar_element_abundances.dat` | Source file for FastChem elemental abundances. HD189/HD209 use the stock Lodders 2009 file. W39b sets this to the explicit rocky-suppressed file so FastChem does not hide oxygen in species outside the kinetic network. |
| `fastchem_newton_tol` / `fastchem_newton_max_iter` | `1e-12` / `50` | `_jax_newton` knobs for `ini_mix='EQ'` / `'const_lowT'`. |

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

The simplest way to run VULCAN-JAX. Reads `vulcan_cfg.py`, runs the integration to convergence, writes `output/<out_name>.vul`. ~80 lines:

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

# rs.atm           — AtmInputs (Tco, pco, Kzz, M, mu, dz, ms, alpha,
#                                Dzz, Dzz_cen, vm, vs, top/bot fluxes, ...)
# rs.rate          — RateInputs (k: (nr+1, nz) rate-constant array)
# rs.photo         — PhotoInputs (sflux_top, def_bin_min, def_bin_max)
# rs.photo_static  — PhotoStaticInputs (cross sections, branch indices)
# rs.step          — StepInputs (y, ymix, t, dt, longdy, longdydt, ...)
# rs.params        — ParamInputs (count, end_case, pic_count, ...)
# rs.atoms         — AtomInputs (atom_loss, atom_loss_prev, ratio history)
# rs.photo_runtime — PhotoRuntimeInputs (tau, aflux, J_sp, Jion_sp, ...)
# rs.fix_species   — FixSpeciesInputs (fix-species snapshot, masks)
# rs.metadata      — RunMetadata (Rf, n_branch, photo_sp, ion_sp,
#                                  pho_rate_index, ion_rate_index,
#                                  start_time, gas_indx, sat_p, r_p,
#                                  y_ini, ...)
```

Every leaf in `rs.atm` / `rs.rate` / `rs.photo_static` is a JAX array (the initial-abundance arrays land on `rs.step.y` / `rs.step.ymix` and the column atom inventory on `rs.atoms`). Inputs supplied through these pytrees are on the differentiable runtime surface; raw file readers and FastChem are host-side setup.

#### Running the integration

```python
from outer_loop import OuterLoop
import op_jax

solver = op_jax.Ros2JAX()
integ  = OuterLoop(solver, output)
rs_out = integ(rs)                  # one JIT'd lax.while_loop, on device

# rs_out.params.count       — total accepted steps
# rs_out.params.end_case    — 1=converged, 2=runtime cap, 3=count_max cap,
#                              4=wall_clock_max cap (chunked runner only)
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

# 2. Pack the differentiable input bundle.
inputs = build_steady_state_inputs(k_arr, atm_static)

# 3. Validate the converged residual is small (gradient accuracy
#    is bounded by ||f(y*)||).
validate_steady_state_solution(y_star, inputs, net,
                               residual_rtol=1e-6)

# 4. Get value and gradient of a scalar loss
def loss_fn(y): return some_scalar(y)

loss, grad = steady_state_value_and_grad(
    loss_fn, inputs, y_star, net,
    residual_rtol=1e-6,
    residual_atol=0.0,
)

g_k_arr = grad.k_arr        # gradients per-input-leaf
```

Memory cost is **O(1) in step count** — no checkpointing.

### 3. Low-level functional API

For when you need to bypass the typed pytree:

| Function | Purpose |
|---|---|
| `chem_funs.chem_rhs_codegen(y, M, k_arr)` | Production chemistry RHS, master-faithful term order. |
| `chem.chem_rhs_segment_sum(y, M, k_arr, net)` | Vectorized reference RHS for Jacobian oracles and synthetic custom-network tests. |
| `chem.chem_jac_analytical(y, M, k_arr, net)` | Stoichiometry-driven analytical Jacobian (block stack `(nz, ni, ni)`). |
| `solver.factor_block_thomas_diag_offdiag(diag, sup_d, sub_d)` / `solve_block_thomas_diag_offdiag(factors, rhs)` | Block-tridiagonal factor + back-substitute. |
| `photo.compute_tau_jax` / `compute_flux_jax` / `compute_J_jax` / `compute_Jion_jax` | Two-stream photochemistry kernels. |
| `atm_refresh.update_mu_dz_jax(ymix, st)` / `update_phi_esc_jax(y, g, Hp, top_flux_in, st)` | Hydrostatic balance + escape flux update. |
| `conden.update_conden_rates(k_arr, y, st)` / `apply_h2o_relax_jax(y, ymix, dt, st)` / `apply_nh3_relax_jax(y, ymix, dt, st)` | Condensation kernels. |
| `rates.build_rate_array(cfg, net, atm, nasa9_coeffs)` | Forward + reverse + remove rate-coefficient table. |
| `gibbs.K_eq_array(net, gibbs_sp, T)` | Equilibrium constants from NASA-9 polynomials. |
| `gibbs.compute_all_k(net, T, M, nasa9_coeffs, remove_list=None)` | Forward + reverse rate assembly. |
| `jax_step.jax_ros2_step(y, k_arr, dt, atm, net, fix_mask=None)` | One Rosenbrock-2 step. Returns `(sol, delta_arr)`. |

All are jit/vmap/jvp/vjp compatible.

---

## Comparison to VULCAN-master

VULCAN-JAX is intended as a drop-in replacement for the supported Ros2 path. The compatibility surface:

The default parity target is **canonical HD189**: `vulcan_cfg.py` matches
`cfg_examples/vulcan_cfg_HD189.py` with VULCAN-master-equivalent physics,
abundance, solver, tolerance, and input-data defaults. UI/output flags are
excluded from numerical parity. Use this audit before publishing a
default-mode comparison:

```bash
python tools/audit_master_parity.py --master ../VULCAN-master
```

| Surface | Compatible? | Notes |
|---|---|---|
| `vulcan_cfg.py` format | yes — same keys, same format | JAX-only knobs are documented above; defaults match. |
| Network files (`thermo/*.txt`) | yes — same parser | Vendored from upstream. |
| Atmosphere files (`atm/*.txt`) | yes — same parser | Vendored. |
| Photo cross-section files | yes — same parser | Vendored. |
| FastChem subprocess (`ini_mix='EQ'`) | yes — same binary, same I/O | External subprocess. |
| `.vul` output schema | yes — same public keys, shapes, dtypes | Pickle bytes are not byte-identical. |
| `y_time` / `t_time` history | partial — last `min(accept_count, conv_step)` accepted steps (sampled by `save_evo_frq`) | Master appends an unbounded list of accepted-step states then post-subsamples at write time. JAX preallocates a fixed ring (`conv_step` default 500) so the JIT'd runner stays shape-static; tail is captured, early-phase steps are overwritten for runs > `conv_step` accepted steps. |
| `plot_py/` scripts | yes — unchanged | Same data surface. |
| Output writer `vars(data_var)` filtered by `var_save` | yes | Same filter. |
| `parameter` keys (`end_case`, `count`, `where_varies_most`, `pic_count`, `tableau20`, ...) | yes | All master public keys published. |
| Solver | partial — Ros2 only | `SemiEU` etc. are commented-out stubs in master, not ported. |
| `chem_funs.symjac` / `neg_symjac` | no — raise `NotImplementedError` | Replaced by `chem.chem_jac_analytical`. |
| `make_chem_funs.py` | yes — JAX codegen shim | Emits a per-network SymPy-faithful JAX RHS cache; no upstream `chem_funs.py` rewrite step is required. |
| Live plot cadence | yes — `live_plot_frq` | Master fires inside its Python loop; JAX fires between JIT'd chunks at the same predicate. |
| Byte-identical pickle | no | Public keys/shapes/dtypes match, but dict order and transient histories may not. |

### Intentional behavioral differences

- **Live UI is host-side, fired between JIT'd step batches.** When any of `use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie` is set, the runner switches to chunked execution (chunks of `live_plot_frq` accepted steps), with `live_ui.LiveUI` reading the legacy `(var, atm, para)` view between chunks. Cadence-faithful but not call-site-identical.
- **Output writer synthesizes** `J_sp` / `Jion_sp` / per-reaction `var.k` dicts from the typed JAX state at pickle time rather than incrementally during the run.
- **Convergence-detector stall fallback** (`conv_stall_window`) — see [Numerical notes](#numerical-notes). Master almost never trips it; JAX trips it when a heavy-hydrocarbon trace species oscillates around `yconv_min` for too long.

---

## Differentiability (forward & reverse-mode)

### What is differentiable

The full physical input surface — **atmospheric structure, rate constants, boundary fluxes, photo cross sections, initial conditions** — is differentiable as long as you supply inputs as JAX arrays into the typed pytrees (`AtmInputs` / `RateInputs` / `PhotoStaticInputs`, plus `StepInputs.y` for initial conditions).

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
│ vulcan_jax.py  (entry point, ~80 lines)                     │
│   └─ RunState.with_pre_loop_setup(cfg) — one call:          │
│       ├─ atm_setup (JAX-native: TPK, mu/dz/g, mol_diff, …)  │
│       ├─ rates.setup_var_k → build_rate_array               │
│       ├─ ini_abun.compute_initial_abundance                 │
│       │       (5 ini_mix modes; FastChem via fcntl.flock)   │
│       ├─ photo_setup.populate_photo                         │
│       │       (host-side CSV → PhotoStaticInputs pytree)    │
│       ├─ rates.apply_photo_remove                           │
│       └─ chem_funs / make_chem_funs.build_chem_rhs(net)     │
│              (codegen RHS, content-hashed cache)            │
│                                                              │
│   integ = OuterLoop(Ros2JAX(), Output())                    │
│   runstate = integ(runstate)                                │
│   └─ one JIT'd lax.while_loop body:                         │
│        cond_fn:  count_max | runtime | converged | stall    │
│        body_fn (one accept, with internal retries):         │
│          ├─ photo branch (lax.cond on update_photo_frq)     │
│          │   compute_tau / flux / J → update k_arr          │
│          ├─ atm refresh (lax.cond on update_frq)            │
│          │   update_mu_dz / phi_esc → splice into atm       │
│          ├─ jax_ros2_step (chem_rhs codegen +               │
│          │     analytical_jac, diffusion, block_thomas)     │
│          ├─ clip / loss / step_ok / step_size               │
│          ├─ conden branch (lax.cond on t / use_relax)       │
│          ├─ hydrostatic balance + ion + fix_all_bot         │
│          ├─ ring-buffer y_time / t_time history             │
│          ├─ adaptive rtol + photo-freq switch               │
│          └─ in-runner conv check → longdy/longdydt          │
│                                                              │
│   output.save_out(runstate, dname) → output/<out_name>.vul  │
└─────────────────────────────────────────────────────────────┘
```

### File map

```
VULCAN-JAX/
├── vulcan_jax.py        Entry point (~80 lines); mirrors vulcan.py orchestration
├── vulcan_cfg.py        VULCAN-format config (default: HD189 via cfg_examples)
├── state.py             Typed pytrees: RunState, AtmInputs, RateInputs,
│                         PhotoStaticInputs, IniAbunOutputs, ParamInputs,
│                         StepInputs, AtomInputs, PhotoRuntimeInputs,
│                         FixSpeciesInputs, RunMetadata
├── outer_loop.py        OuterLoop — single-JIT lax.while_loop runner
├── op_jax.py            Ros2JAX — standalone photo adapter
├── jax_step.py          Vmap'able Ros2 single-step kernel + JAX diffusion
├── solver.py            Block-tridiagonal Thomas solvers
│                         (diagonal-aware + dense fallback)
├── chem.py              JAX chemistry RHS + analytical Jacobian
├── chem_funs.py         JAX-native module exposing master-shaped public
│                         surface (ni/nr/spec_list/Gibbs/chemdf), backed
│                         by make_chem_funs.build_chem_rhs(net)
├── make_chem_funs.py    Per-network codegen for the chem_rhs Python source
├── photo.py             JAX two-stream photochem (tau / flux / J kernels)
├── steady_state_grad.py Implicit-function-theorem custom_vjp for reverse-mode AD
├── runtime_validation.py Pre-run runtime/config validator
├── atm_refresh.py       JAX update_mu_dz + update_phi_esc kernels
├── conden.py            JAX condensation rates + cold-trap relax kernels
├── rates.py             Forward rate coefficients (Arrhenius / Lindemann /
│                         3-body / Troe), low-T caps, remove-list
├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
├── network.py           Network parser (text → stoichiometry tables)
├── integrate.py         Pure-JAX fixed-dt scan loop (validation/benchmarks)
├── legacy_io.py         Vendored op.ReadRate + .vul writer (polymorphic
│                         save_out: accepts RunState or legacy triple)
├── atm_setup.py         JAX-native atmosphere setup (f_pico / load_TPK /
│                         mol_diff / sat_p / hydrostatic refresh / ...)
├── ini_abun.py          JAX-native initial-abundance setup (5 ini_mix modes;
│                         FastChem via fcntl.flock)
├── photo_setup.py       Host-side cross-section preprocessing
├── composition.py       Per-species composition / mass tables
├── live_ui.py           Host-side live-plot dispatcher (matplotlib + PIL)
├── phy_const.py         Physical constants (kb, Navo, hc, au, r_sun, ...)
├── pytest.ini           Pytest config
├── atm/, thermo/, fastchem_vulcan/   Vendored runtime data
├── cfg_examples/        HD189 / HD209 / Earth / W39b example configs
├── benchmarks/          Per-step timing benchmark
├── examples/            Usage examples (vmap, forward-mode AD, implicit AD)
├── tools/               End-user data-prep + parity-audit utilities
├── tests/               Curated validation suite
├── FILE_README.md       Per-file index of every module and its functions
├── README.md            This file
└── CLAUDE.md            Maintenance / numerical-hygiene notes
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

Run `python benchmarks/bench_step.py` for a fresh per-step timing on your
hardware.

---

## Validation: what is and isn't tested

### Numerical agreement vs VULCAN-master (per-component)

| Layer | Agreement |
|---|---|
| Forward rate coefficients (596 reactions) | bit-exact (relerr = 0) |
| Reverse rates (533 from Gibbs) | 1.4e-14 |
| Atmosphere structure (pco/Tco/Kzz/M/...) | bit-exact |
| Initial abundances (FastChem path) | bit-exact |
| Chemistry RHS (`chem_rhs_codegen` vs `chemdf`) | <=1e-5 on significant/bulk cells; cancellation residues are tested with per-species floors |
| Chemistry Jacobian (`chem_jac_analytical` vs `chem.chem_jac` jacrev oracle) | 4.3e-13 |
| Diffusion operator (vs `op.diffdf`) | 2e-6 (FP-noise-bound) |
| Block-Thomas solver | 3e-15 |
| Single Ros2 step (vs `op.Ros2.solver`) | 1.16e-15 |
| `compute_tau_jax` / `compute_flux_jax` / `compute_J_jax` | 8e-16 / 3.7e-11 / 1.8e-11 |
| `compute_Jion_jax` / ion `k_arr` wiring | unit-tested end-to-end |
| `update_mu_dz_jax` / `update_phi_esc_jax` | 3-7e-16 (bit-exact) |
| `update_conden_rates` / `apply_h2o_relax_jax` / `apply_nh3_relax_jax` | 0 (bit-exact) |
| End-to-end 50-step run (HD189) | 1.59e-10 |
| End-to-end converged HD189 (median dex) | 0.004 dex (~1% relative) |

### What's covered by the test suite

`pytest tests/` runs the curated suite covering:
- JAX↔master numerical bridge (RHS, Jacobian, diffusion, single Ros2 step, photo kernels)
- `vmap` consistency (single-call vs batched output)
- Forward-mode AD (`jvp` through per-step kernels)
- Reverse-mode AD via `steady_state_grad` (validated against finite differences)
- HD189 smoke integration (50-step regression oracle)
- Default HD189 parity audit plus bit-exact pre-loop initial state and
  20-step matched Ros2 oracle vs VULCAN-master
- 20-step matched-step oracles for Earth + HD209
- `save_evolution` round-trip
- `.vul` output schema & RunState round-trip
- Vendored example-config setup
- W39b FastChem invariant snapshot

### What's NOT tested

- **Cartesian-product oracle sweeps** over every non-default config knob. By policy.
- **GPU parity** (CPU-only host).
- **Long-to-convergence VULCAN-master oracles** for every vendored example.
- **Arbitrary custom networks** beyond parser/schema coverage and the bundled examples.
- **Gradients through host-side readers / FastChem internals** — by design (host-side setup, not on the AD path).
- **Invalid master configurations** that the validator rejects (`use_ion=True` without `use_photo=True`, `use_live_flux=True` without `use_photo=True`, `fix_species` without condensation).

---

## Numerical notes

### Chemistry RHS parity and cancellation residues

The production JAX path now uses `make_chem_funs.build_chem_rhs(net)` to emit
per-network code in the same order as VULCAN-master's SymPy-generated
`chemdf`: odd/even reaction pairing, stoich-repeated multiply chains,
asymmetric third-body `M`, and products-before-reactants accumulation.

The old vectorized `segment_sum` RHS is still kept as
`chem.chem_rhs_segment_sum` for Jacobian oracles and synthetic custom-network
tests. It is not the production Ros2 RHS. On cancellation-prone trace cells,
the two RHS implementations can still differ in relative terms because both
are subtracting large production/loss rates down to tiny residues. Bulk and
significant-cell agreement is guarded by `tests/test_chem_rhs_codegen.py`
(at `rtol=1e-13` against the NumPy oracle and `rtol=1e-12` against master)
and `tests/test_chem.py` (at `rtol=1e-12`). Use `python benchmarks/bench_step.py`
to quantify the local speed cost on a specific machine. The analytical
Jacobian (`chem_jac_analytical`) remains stoichiometry-driven and is
unaffected by the codegen RHS.

### Step-count and atom-conservation drift (downstream of the floor above)

The 1e-4 floor is invisible at the per-step level but compounds over a long integration in two ways:

**1. JAX version may need more accepted steps to detect convergence than master.**
The convergence test fires when `longdy = max|Δy/(n_0·ymix)|` drops below `yconv_min = 0.1` and `longdydt < slope_min`. The ULP floor doesn't move bulk species (H2O, CO, CH4, NH3, HCN), but it nudges heavy-hydrocarbon trace radicals (`C6H6`, `C2H2`, `C4H5`, `C4H2`, `C3H3`, `CH3NH2`, ...) along slightly different trajectories. Whichever one is sitting at the threshold last gates termination. Both runs reach physically equivalent steady states; only the detection moment differs.

**2. XLA compilation breaks the stoichiometric nullspace (resolved).**
The generated Python RHS source is stoichiometrically correct — under `jax.disable_jit()` it matches master's NumPy RHS to machine epsilon. But `jax.jit` lets XLA fuse and reorder floating-point operations, so large production/loss terms (e.g., CH4 ~-2.4e+1 vs CH3 ~+2.3e+1 C-inventory/s at HD209's bottom layer) no longer cancel to the same rounded value. The per-step C-atom residual (~5e-7 of per-layer budget) integrates over long timesteps. `jax.lax.optimization_barrier` in `make_chem_funs.py` reduces but does not eliminate this.

**Fix**: `jax_step._project_chem_rhs` / `_project_chem_jac` enforce exact H/O/C/N conservation after each RHS evaluation by distributing the per-layer atom residual across abundant reservoir species (H2, H2O, CO, N2). The correction is ~5e-13 relative per step, preserving the physical trajectory. The projected Jacobian keeps the Rosenbrock implicit solve consistent. Overhead is ~3% per step, no change in step count or convergence behavior.

**Current state**: HD209 atom_loss matches master within 2× across all atoms (H: 2.01e-4, O: 2.44e-4, C: 6.6e-5, N: 1.99e-4 vs master H: 2.01e-4, O: 2.57e-4, C: 1.05e-4, N: 2.00e-4). `atom_ini` matches master to machine epsilon (1e-16 relative). `tests/test_oracle.py` validates the 20-step trajectory at relerr ≤ 1e-4 against master (or baseline when master's codegen is unavailable).

### Mitigations (no code changes, just config knobs)

- **`conver_ignore`** (populated by default in `cfg_examples/vulcan_cfg_HD189.py` / `vulcan_cfg_HD209.py`):
  ```python
  conver_ignore = ['C6H6', 'C2H2', 'C6H5', 'C2H', 'C2H4', 'C2H5', 'C2H6',
                   'C3H2', 'C3H3', 'C4H5', 'CH2NH', 'CH3NH2', 'H2CCO']
  ```
  Heavy-hydrocarbon trace radicals excluded from the convergence detector. If a *new* trace radical takes over the gate on a different planet, look at `parameter['where_varies_most']` in the saved `.vul` and add it.

- **`conv_stall_window = 200`** (new safety net, default):
  Stall fallback in both branches. If `longdy_seen_min` (running min of `longdy`, only resets on a ≥5% relative drop) has been below `yconv_min` for 200 accepted steps without significant improvement *and* current `longdy` is also below `yconv_min`, declare `end_case=1`. Master almost never trips it; JAX trips it when a heavy hydrocarbon outside `conver_ignore` keeps oscillating around the threshold.

### Investigated and ruled out

- **Tighter `loss_eps`** (e.g. 1e-5): drift is per physical time, not per step. Tighter `loss_eps` causes dt-thrashing without reducing cumulative drift.
- **Compensated summation** (Kahan/Neumaier in `chem_rhs`): empirically verified bit-identical to `math.fsum` on JAX's terms — the disagreement is in the per-term *values* JAX emits, not in the summation order. Adds ~14× to chem_rhs runtime for zero gain.
- **float32**: hard no — rate constants span 50 orders of magnitude.
- **Globally pivoted banded LU in place of `block_thomas_diag_offdiag`** (investigated 2026-05-22): rejected. Pure-JAX scalar banded solver ran ~96× slower; LAPACK `dgbtrf`/`dgbtrs` host callback produced **+2.63e-2 C drift, worse than block-Thomas**. Per-block partial pivoting is not the drift source.

### Other documented numerical points

- **Diffusion Jacobian** matches `op.diffdf` to 2e-6 (FP noise from extracting small residues from `c0~1e10` cancellations). Block diagonals match `op.lhs_jac_tot` to machine precision for sup/sub blocks but disagree at heavy-condensable cells (S8 layers 5/25). Direct comparison with the analytical derivative confirms the JAX side is correct; master's `op.lhs_jac_tot` has a minor self-inconsistency.
- **Asymmetric M factor for dissociation reactions.** The parser sets `network.is_three_body[i]` and `network.is_three_body[i+1]` independently for the forward and reverse slots of each reaction, so reactions like `HNCO + M → H + NCO` (3-body forward, bimolecular reverse) are handled correctly without forced forward/reverse symmetry.

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
python -m pytest tests -q --tb=short -ra   # curated suite
python -m pytest tests -n auto -q --tb=short -ra
                               # parallel-safe (FastChem invocations
                               # serialise via fcntl.flock).
python -m pytest tests -k "ros2 or block_thomas"   # filter
python -m pytest tests/test_default_master_parity.py tests/test_w39b_fastchem_invariant.py tests/test_oracle.py -q --tb=short -ra
python tools/audit_master_parity.py --master ../VULCAN-master
```

Master-comparison tests skip cleanly when `../VULCAN-master/` is absent;
two config-matrix sub-cases also skip on networks without `H2O_l_s`.

The test suite is deliberately not parametrized across every non-default
config combination — see [Capabilities](#capabilities) for the full
inventory.

---

## License & citation

VULCAN-JAX inherits its license from VULCAN (GPLv3, see `VULCAN-master/GPL_license.txt`).

If you use VULCAN-JAX in published work, please cite the underlying VULCAN papers:

- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20
- Tsai, S.-M., Malik, M., Kitzmann, D., Lyons, J. R., Fateev, A., Lavvas, P., & Heng, K. 2021, ApJ, 923, 264
