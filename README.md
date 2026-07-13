# VULCAN-JAX

A JAX-accelerated, differentiable port of [VULCAN](https://github.com/exoclime/VULCAN) -- the photochemical-kinetics solver for exoplanet atmospheres (Tsai et al. 2017, 2021).

VULCAN-JAX runs the same configuration files, input data, and `.vul` output schema as upstream VULCAN. The hot path is a single JIT-compiled `lax.while_loop` on CPU or GPU. The runtime is standalone -- no `../VULCAN-master/` sibling required.

**Why use this over upstream VULCAN?**
- ~3x faster per-step on CPU (single-threaded; see Benchmarks); end-to-end speedup is workload-dependent
- Differentiable: **forward-mode works end-to-end** through the full converged model (validated vs finite differences); **reverse-mode** returns reaction-importance sensitivities (`dL/d ln k` for all reactions in one adjoint solve) at the converged state — percent-level by default (renormalized-map operator, plus automatic photolysis feedback when runner context is supplied for photochemistry-on models; see [Differentiability](#differentiability))
- Same config format and `.vul` output: VULCAN's `plot_py/` scripts work unmodified
- Vectorizable: tested `vmap` support for batched parameter sweeps
- GPU-ready architecture (not yet benchmarked)

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Project structure](#project-structure)
4. [Running the forward model](#running-the-forward-model)
5. [Configuration](#configuration)
6. [API overview](#api-overview)
7. [Differentiability](#differentiability)
8. [Benchmarks](#benchmarks)
9. [Running tests](#running-tests)
10. [Comparison to VULCAN-master](#comparison-to-vulcan-master)
11. [Numerical notes](#numerical-notes)
12. [License & citation](#license--citation)

---

## Installation

### Option 1: pip install from TestPyPI (recommended for library use)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vulcan-jax
```

Then from any Python script or notebook:

```python
import vulcan_jax

cfg = vulcan_jax.make_config(count_max=100)
rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)
```

### Option 2: from source (for development)

```bash
git clone git@github.com:imalsky/jax-vulcan.git VULCAN-JAX
cd VULCAN-JAX
pip install -e ".[dev,plot]"
```

### Option 3: conda environment from scratch

```bash
conda create -n vulcan python=3.12 -y
conda activate vulcan
pip install -e ".[dev,plot]"
```

For NVIDIA GPU support, install the platform-specific JAX wheel instead of
plain `jax` (see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

### Dependencies

Core (installed automatically): `jax>=0.4.31`, `numpy>=1.24`, `scipy>=1.12`

Optional extras:
- `pip install -e ".[dev]"` adds `pytest`, `pytest-xdist`, `ruff`, `vulture`, `build`, `twine`, `sympy` (`sympy` is only used to run the VULCAN-master oracle tests)
- `pip install -e ".[plot]"` adds `matplotlib`, `Pillow` (needed for live plots and the quickstart notebook)

### Verifying the install

```bash
python -c "import vulcan_jax; print(vulcan_jax.__version__)"
```

---

## Quickstart

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a worked notebook that builds an HD189733b initial state and plots VMR profiles.

```python
import vulcan_jax

# Build a run state with default HD189 config
cfg = vulcan_jax.make_config(count_max=10, use_print_prog=False)
rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)

# Access the atmosphere and abundances
import numpy as np
pressure_bar = np.asarray(rs.atm.pco) / 1e6   # pressure in bar
ymix = np.asarray(rs.step.ymix)                # volume mixing ratios (nz, ni)
species = vulcan_jax.chem_funs.spec_list        # species names
```

For the full forward model from the command line:

```bash
cd VULCAN-JAX/
python -m vulcan_jax.vulcan_jax_cli        # or: vulcan-jax (entry point)
```

Output lands at `output/<out_name>.vul` -- same pickle schema as VULCAN-master.

---

## Project structure

```
VULCAN-JAX/
├── src/vulcan_jax/          Python package (pip-installable)
│   ├── __init__.py          Public API: RunState, make_config, vulcan_cfg
│   ├── vulcan_jax_cli.py    CLI entry point (vulcan-jax command)
│   ├── vulcan_cfg.py        Default config (HD189); same format as VULCAN-master
│   ├── state.py             Typed JAX pytrees (RunState, AtmInputs, etc.)
│   ├── outer_loop.py        Single-JIT lax.while_loop integration runner
│   ├── jax_step.py          Vmap-able Ros2 single-step kernel
│   ├── solver.py            Block-tridiagonal Thomas solvers
│   ├── chem.py              Chemistry RHS + analytical Jacobian
│   ├── chem_funs.py         Public surface (ni/nr/spec_list/chemdf)
│   ├── make_chem_funs.py    Per-network codegen for chemistry RHS
│   ├── photo.py             JAX two-stream photochemistry kernels
│   ├── steady_state_grad.py Reverse-mode reaction sensitivities (solver-map adjoint)
│   ├── atm_jax.py           Differentiable on-graph atmosphere builder (PhysicalInputs -> AtmStatic)
│   ├── rates_jax.py         Differentiable T -> k rate builder (forward-mode)
│   ├── rates.py             Rate coefficients (Arrhenius/Lindemann/Troe)
│   ├── gibbs.py             NASA-9 Gibbs / K_eq / reverse rates
│   ├── network.py           Network file parser
│   ├── atm_setup.py         Atmosphere setup (TP profiles, diffusion, BCs)
│   ├── ini_abun.py          Initial abundances (5 modes incl. FastChem)
│   ├── photo_setup.py       Cross-section preprocessing (host-side)
│   ├── composition.py       Species composition / mass tables
│   ├── legacy_io.py         .vul writer + vendored ReadRate
│   ├── atm_refresh.py       Hydrostatic balance update kernels
│   ├── conden.py            Condensation rate + cold-trap kernels
│   ├── op_jax.py            Ros2JAX solver adapter
│   ├── integrate.py         Fixed-dt scan loop (benchmarks/validation)
│   ├── live_ui.py           Host-side live plot dispatcher
│   ├── runtime_validation.py Pre-run config validator
│   ├── phy_const.py         Physical constants
│   ├── _paths.py            Package data path resolution
│   ├── _version.py          Version string
│   ├── atm/                 TP/Kzz tables, stellar flux, BC files
│   ├── thermo/              Network files, NASA-9 data, photo cross sections
│   ├── fastchem_vulcan/     FastChem C++ source + I/O for ini_mix='EQ' (binary auto-built on first use)
│   └── cfg_examples/        Example configs (HD189, HD209, Earth, W39b)
│
├── tests/                   Validation suite (see "Running tests")
├── examples/                Usage examples (see below)
├── benchmarks/              Per-step timing benchmark
├── tools/                   Data-prep and parity-audit utilities
├── output/                  Forward-model outputs (.vul files, not tracked)
├── plot/                    Generated figures (not tracked)
├── docs/                    Project docs (file_organization.md, notes.md)
│
├── pyproject.toml           Build metadata and dependencies
├── setup.py                 Compatibility shim
├── release.sh               TestPyPI release script
└── README.md                This file
```

### Directory purposes

| Directory | What's in it | Tracked in git? |
|---|---|---|
| `examples/` | Worked examples: quickstart notebook, batched runs, forward/reverse-mode AD | Yes |
| `benchmarks/` | `bench_step.py` -- per-step kernel timing vs NumPy | Yes |
| `tests/` | Curated pytest suite: JAX-master parity, vmap, AD, integration smoke tests | Yes |
| `tools/` | `audit_master_parity.py` (parity audit vs upstream), `make_mix_table.py`, `make_spectra_in_nm.py`, `print_actinic_flux.py` | Yes |
| `docs/` | `file_organization.md` (per-module function index), `notes.md` (implementation + end-to-end AD notes) | Yes |
| `output/` | `.vul` pickle files from forward-model runs | No (gitignored) |
| `plot/` | Generated figures from plot scripts | No (gitignored) |

---

## Running the forward model

### From the command line

```bash
# Default HD189 config:
vulcan-jax

# Or equivalently:
python -m vulcan_jax.vulcan_jax_cli

# GPU (no code changes):
JAX_PLATFORM_NAME=gpu vulcan-jax
```

Command-line flags like `-n` are ignored -- the CLI reads its configuration from `vulcan_cfg.py` -- so upstream `vulcan.py -n` habits still run.

### As a library

```python
import vulcan_jax

# Use defaults with overrides
cfg = vulcan_jax.make_config(
    count_max=5000,
    use_photo=True,
    use_live_plot=False,
)
rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)

# Run integration -- pass the SAME cfg to the runner so its solver knobs
# (count_max, rtol, dt bounds, convergence criteria, ...) honor your overrides.
from vulcan_jax import outer_loop, op_jax, legacy_io
solver = op_jax.Ros2JAX()
output = legacy_io.Output(cfg=cfg)  # cfg-aware output paths + progress prints
integ = outer_loop.OuterLoop(solver, output, cfg=cfg)
rs_out = integ(rs)

# Check results
print("Converged:", rs_out.params.end_case == 1)
print("Steps:", int(rs_out.params.count))
```

`make_config(...)` overrides are honored end-to-end: `with_pre_loop_setup(cfg)`
applies them to the pre-loop setup, `OuterLoop(solver, output, cfg=cfg)` to the
integration, and `legacy_io.Output(cfg=cfg)` to the output paths and progress
prints. `cfg` defaults to the global `vulcan_cfg` module, so the bare
`OuterLoop(solver, output)` / `Output()` (the CLI form) is unchanged.

**Import-frozen knobs `make_config` cannot change.** A few structural inputs are
read once at the first `import vulcan_jax` and cannot be changed afterward: the
reaction `network` (`ni` / `nr` / `spec_list`), the composition table `com_file`,
and `atom_list` (the reservoir-projection tables). Passing a different
`cfg.network` or `cfg.com_file` raises a clear error rather than silently using
the import-time value (a same-content copy at a different path is accepted). To
change the network, set the `VULCAN_JAX_NETWORK` environment variable to its
path **before** the first `import vulcan_jax` (or use the subprocess driver);
set `com_file` / `atom_list` in the config used at that first import.

### Batched runs (vmap across profiles)

`OuterLoop.run_batch` integrates a whole batch of *different* atmospheric
profiles in one vmapped device call — the GPU-parallel emulator-data and
parameter-sweep path:

```python
states, atms = zip(*(integ.prepare_runstate(rs) for rs in run_states))
batched = integ.run_batch(
    outer_loop.stack_integ_states(list(states)),
    outer_loop.stack_atm_statics(list(atms)),
)
results = outer_loop.unstack_integ_states(batched, len(run_states))
```

Every lane runs with freeze-on-done semantics: each profile's result is
identical to running it alone, and the call returns when the slowest lane
terminates (per-lane `termination_reason` / `ymix` on the unstacked states).
Supported physics matches the single-profile runner, including
**photochemistry** (per-profile T-interpolated cross sections ride the carry;
all lanes must share the star, wavelength grid, network, and config scalars —
only the T-P profile varies, enforced by `prepare_runstate`) and **NH3/H2O
relaxation condensation** (the per-profile NH3 cold-trap index rides the
carry). All profiles in one batch must share `nz` and the config
toggle-combo. `examples/gpu_benchmark.py` is the full worked driver (parallel
host setup, chunked progress, `--device-batch` tiling for large batches);
`tests/test_vmap_while_loop.py`, `tests/test_vmap_photo_batch.py`, and
`tests/test_nh3_conden_batch_subprocess.py` pin the solo-vs-batch equivalence.

### Known first-run behavior

On the first import with a given network, `make_chem_funs` generates and caches a per-network chemistry RHS source file. This takes a few seconds and prints informational messages. Subsequent imports reuse the cache.

The "Element # not found. Neglected!" messages from FastChem are normal -- the solar abundance file has heavy elements not in the C-H-O-N network, which are safely ignored.

The FastChem binary is **not** vendored as a pre-built executable; the C++ source and makefiles are. The first time `ini_mix='EQ'` runs, `ini_abun._ensure_fastchem_binary()` compiles it from source (`make` in `fastchem_vulcan/`, which creates the `obj/` build dir; `input/` and `output/` are vendored) and reuses it thereafter. The build is serialized by the same `fcntl.flock` that guards the EQ subprocess, so `pytest -n auto` and parallel host-setup workers won't race to build. A C++ toolchain (`c++`/`make`) must be on `PATH`; if the build fails, run `make` manually under `fastchem_vulcan/`.

---

## Configuration

VULCAN-JAX reads the same `vulcan_cfg.py` format as upstream VULCAN. Drop in your existing config; it should work as-is. Example configs ship in `src/vulcan_jax/cfg_examples/`.

JAX-only config additions (all have sensible defaults):

| Key | Default | Purpose |
|---|---|---|
| `batch_max_retries` | `64` | Cap on inner retries per accepted step |
| `conv_stall_window` | `200` | Stall-detector window for convergence |
| `conver_ignore` | `[heavy hydrocarbons]` | Species excluded from convergence test |
| `rtol_min` / `rtol_max` | `0.0` / `1.0` | Bounds for adaptive rtol |
| `loss_criteria` | `5e-4` | Max-column atom-loss gate for the adaptive-rtol controller |
| `step_size_safety` | `0.9` | Ros2 step-size safety factor |
| `use_pi_controller` | `False` | Gustafsson (1991) PI step-size controller (ported from neoVULCAN). Off = master-faithful I-control. When on, accepted steps use `h_factor = safety * (rtol/delta)^(alpha/2) * (delta_prev/delta)^(beta/2)`, falling back to I-control on the first step and after any rejection. Forward-mode AD-safe |
| `pi_controller_alpha` / `pi_controller_beta` | `0.7` / `0.4` | PI controller exponents (divided by the Ros2 error order p=2) |
| `fastchem_newton_tol` | `1e-12` | Newton solver tolerance for `ini_mix='EQ'` |
| `wall_clock_max` | `None` | Wall-clock budget (s); a positive value forces the chunked runner and bails between chunks (`end_case=4`) |

See the example configs and `vulcan_cfg.py` for the full list of supported knobs.

---

## API overview

### RunState pytree

`RunState` is the canonical runtime surface -- a JAX pytree with typed fields:

```python
rs = vulcan_jax.RunState.with_pre_loop_setup(cfg)

rs.atm            # AtmInputs (Tco, pco, Kzz, M, mu, dz, ...)
rs.rate            # RateInputs (k: (nr+1, nz) rate-constant array)
rs.step            # StepInputs (y, ymix, t, dt, longdy, ...)
rs.photo_static    # PhotoStaticInputs (cross sections, branch indices)
rs.params          # ParamInputs (count, end_case, ...)
rs.atoms           # AtomInputs (atom_loss, ...)
rs.metadata        # RunMetadata (Rf, species info, static data)
```

### Low-level functional API

All kernels are `jit`/`vmap`/`jvp`/`vjp` compatible:

| Function | Purpose |
|---|---|
| `chem_funs.chem_rhs_codegen(y, M, k_arr)` | Production chemistry RHS |
| `chem.chem_jac_analytical(y, M, k_arr, net)` | Analytical Jacobian |
| `jax_step.jax_ros2_step(y, k_arr, dt, atm, net)` | One Rosenbrock-2 step |
| `solver.block_thomas_diag_offdiag(...)` | Block-tridiagonal solve |
| `photo.compute_tau_jax` / `compute_flux_jax` / `compute_J_jax` | Photochemistry |
| `rates.build_rate_array(cfg, net, atm, nasa9_coeffs)` | Rate-coefficient table |
| `gibbs.K_eq_array(net, gibbs_sp, T)` | Equilibrium constants |

---

## Differentiability

**The rule:** a quantity is differentiable **iff it reaches the runtime as a JAX
array** — either because you supply it directly into the runtime pytrees
(`AtmStatic` / `RateInputs` / initial `y`; most of `PhotoStaticInputs`, but note
some photo fields are closure-baked into the runner's photo branch — see the
table below), or because we
provide an **on-graph builder** for it: `rates_jax` for `T -> k`, and
`atm_jax.build_atm_static` for the whole atmosphere structure (`pco`, `Tco`,
gravity, composition `-> M`, `dz`, `Hp`, `Dzz`, `vm`, `vs`, ...). Drive the inner
`integ._runner` (not `OuterLoop.__call__`, which copies to host and breaks
tracing); forward-mode (`jvp`/`jacfwd`) then runs end-to-end through the converged
integration (FD-validated <0.1%). A scalar *parameter* that a **host-side setup
formula** expands into those arrays is differentiable once that formula is on the
graph — which, after `build_atm_static`, now covers the atmosphere cascade.

### What you CAN differentiate now (forward-mode, end-to-end)

| Physical input | How |
|---|---|
| Reaction rates `k` (forward **and** reverse) | supply `k_arr`; reverse-mode reaction ranking via `steady_state_reaction_sensitivity` (all reactions, one solve) |
| Temperature `T` (per-layer array) | `atm_jax.build_atm_static` rebuilds `M`/`dz`/`Hp`/`Dzz`/`vm`/`vs` on-graph from `Tco`; also rebuild `k(T)` with `rates_jax.build_rate_array` for the rate path (`use_lowT_caps=True` on cool networks) |
| **Surface gravity `gs`, planet radius `Rp`** | `build_atm_static` — `gs`/`Rp` drive the hydrostatic height integration (`g`, `Hp`, `dz`, `dzi`) on-graph |
| **Pressure grid (`P_b`, `P_t`)** | `atm_jax.pco_from_endpoints(P_b, P_t, nz)` -> `pco` leaf of `PhysicalInputs`; reaches `M`, `Dzz`, `dz` |
| **Molecular/thermal diffusion `Dzz`, `vm`, `vs` (T-/g-driven)** | `build_atm_static` ports the `T -> Dzz` (Moses fit), `vm`, and Cloutman settling formulae on-graph — a `T`- or `g`-driven change now flows through |
| Rate coefficients — Arrhenius `a`/`n`/`E`, NASA-9 thermo | `rates_jax.build_rate_array(..., rate_coeffs={"a": ...})`; NASA-9 via `nasa9_coeffs` (one hardcoded Troe row excepted) |
| Eddy diffusion `Kzz`, advection `vz` | `atm._replace(Kzz=...)`, or `atm_setup.kzz_profile_jax` for `∂/∂K_deep`/`K_max` |
| Boundary fluxes / deposition velocity | supply `top_flux` / `bot_flux` / `bot_vdep` |
| Initial abundances `y0` | perturb `y0` directly |
| **Metallicity `[M/H]`, C/O ratio** | a `y0` *tangent*: scale metal-bearing species for `[M/H]`, or C-bearing vs O-bearing for C/O (example below). This is the correct derivative for a closed column — `Z` is the conserved metal inventory and the steady state depends on element totals, not on the initial speciation. It is exactly Fig. 9 (`∂ln VMR/∂ln Z`, SO₂ `∝ Z^2.6`). |

Build the differentiable atmosphere with `phys, spec = atm_jax.make_physical_inputs(cfg, var, atm, species_list)`,
then `atm_jax.build_atm_static(phys._replace(Tco=...), spec)` — it reproduces the
production `make_atm_static` field-for-field (machine precision) for the default
configuration (`atm_type` `file`/`analytical`/`isothermal` with `use_moldiff=on`,
which is what the runner uses) while carrying tangents w.r.t. `phys`. See
`examples/grad_physical_example.py`. (Two non-default modes differ — both because
`build_atm_static` is the *more* self-consistent of the two: `atm_type='table'`
recomputes the interface pressures from the rewritten grid where production keeps
a stale `pico`, and `use_moldiff=off` computes `Ti`/`Hpi` as interface averages
where production leaves them at legacy defaults; the latter is runtime-inert.)

**Condensation follows a live `T(P)` too.** `conden.make_conden_spec` (host, once
per config) extracts the temperature-independent metadata — species identity,
particle `m/(ρ_p r_p²)` coefficients, relax/fix flags — and
`conden.build_conden_profile(spec, Tco, pco, n_0, Dzz)` rebuilds every
temperature/structure-dependent condensation array on-graph: per-reaction
saturation number densities (`sat_p_jax(T)/k_B T`, humidity-weighted for H2O),
growth/diffusion `Dg` terms from the live `Dzz`, the H2O/NH3 relax inputs, the
NH3 cold-trap `argmin` index, and the fix-species saturation mixing ratios
(`min(1, sat_p/p)`). The function is jit/vmap/jvp-compatible; the runner already
reads these arrays from the `ProfileVars` carry every step (`s.pv.c_*`,
`s.pv.fix_species_sat_mix`), so splicing the rebuilt `CondenProfile` into `pv`
makes the whole condensation path consistent with — and differentiable w.r.t. —
the proposed temperature. `OuterLoop._build_conden_static` delegates to the same
builder (verified bit-exact against the pre-refactor host packer on isothermal
and non-isothermal columns), so host setup and on-graph rebuild share one
implementation; `tests/test_conden_profile_builder.py` pins the formulas against
an independent NumPy oracle plus the jit/vmap/jvp contracts. The one discrete
output is the cold-trap `argmin` (integer, no tangent) — a `T` tangent moves the
saturation *curves* smoothly but the active-layer set and cold-trap index change
layer-by-layer, so forward-mode derivatives are only valid away from those
switches (same caveat as any phase boundary).

### What you CANNOT differentiate yet

| Blocked knob | Why it's blocked | Workaround today |
|---|---|---|
| TP-profile parameter `∂L/∂T_irr` via Heng+14 | `analytical_TP_H14` is on-graph, but its `jax.scipy.special.expn` forward-mode is very slow over a deep column's many decades | differentiate the per-layer `Tco` (or `Tco`-scale) leaf, or use a cheaper `T(P)` parameterisation |
| Stellar-flux scale / spectrum | the stellar flux (`PhotoInputs.sflux_top`) and the room-T cross sections (`cross_J`, `absp_cross`) are **closure-baked** into the runner's photo branch (`outer_loop._make_photo_branch`), not read from a runtime pytree | perturb them requires a runner-level input, not a pytree field — not yet exposed |
| Photo cross-section **`T`-rebake** | `photo_setup._bin_T_dependent` re-interpolates cross-sections per layer on host at setup | the *T-dependent* cross sections do ride the `ProfileVars` carry (`s.pv.p_cross_J_T` / `p_absp_T_cross`), so they are differentiable as arrays via the carry; the static cross sections and the `T`→cross-section map are not |

**FastChem is the one true wall** (a subprocess): you cannot differentiate the
scalar `[M/H] -> t=0 equilibrium speciation` map. But you almost never need to —
a converged closed column forgets the initial speciation, so the metallicity /
C-O derivatives above (via `y0` tangents) are the scientifically correct ones.
(`ini_abun`'s `const_lowT` Newton *residual* (`_abun_lowT_residual`) is
differentiable w.r.t. the elemental ratios `O_H`/`C_H`/`He_H`/`N_H` for the
reduced H₂/H₂O/CH₄/He/NH₃ system, but the shipped `ini_abun` entry point reads
them as Python floats — call the solver directly with JAX-array ratios to get
that gradient.)

```python
# Metallicity in one forward pass (Fig. 9). Closed column => scaling the
# metal-bearing initial abundances is the [M/H] knob. compo_array column 0 is H
# (default atom_list order), so [:, 1:] selects metal atoms.
import jax, jax.numpy as jnp
from vulcan_jax import composition

metal = jnp.asarray((composition.compo_array[:ni, 1:].sum(1) > 0).astype(float))

def run_from_y0(y0):                       # converged VMR from an initial state
    final = integ._runner(state0._replace(y=y0), atm)
    return final.y / final.y.sum(1, keepdims=True)

_, dlnVMR_dlnZ = jax.jvp(run_from_y0, (y0,), (y0 * metal[None, :],))
```

**Reverse-mode is reaction-ranking only.** The single reverse-mode entry point is
`steady_state_reaction_sensitivity` (`dL/d ln k` for every reaction). It is *not*
general reverse-mode through arbitrary physical inputs — for those, use
forward-mode. See its limitations under
[Reverse-mode](#reverse-mode-solver-map-steady-state-adjoint) below.

### Forward-mode (works through the entire integration)

`lax.while_loop` supports `jvp`, so one forward pass differentiates the whole
converged integration. Drive the traced inner runner directly --- the public
`OuterLoop.__call__` copies state to the host for `.vul` output, which breaks
tracing:

```python
import jax
from vulcan_jax.jax_step import make_atm_static

state0 = integ._pack_state_from_runstate(rs)
atm    = make_atm_static(data_atm, ni, nz, cfg=integ._cfg)

def run(Kzz):                       # converged composition vs eddy mixing
    final = integ._runner(state0, atm._replace(Kzz=Kzz))
    return final.y / final.y.sum(axis=1, keepdims=True)

# tangent = Kzz  =>  d(VMR)/d(ln Kzz) for every species/level, in one pass
ymix, dymix = jax.jvp(run, (atm.Kzz,), (atm.Kzz,))
```

Validated end-to-end on a full HD 189733b production run (photochemistry on,
~1300 accepted steps): the `jvp` tangent matches re-converged centered finite
differences to <0.1% on the responding levels (correlation >0.9999). This route
never inverts `df/dy`, so it stays well posed even where the reverse-mode adjoint
below does not. See `examples/grad_jvp_example.py`.

**Temperature-profile gradients** are a special case: the runner's `k_arr` is
frozen at setup (host-side NumPy `rates.build_rate_array`), so a `d/dT` jvp must
rebuild it on the AD graph with `rates_jax.build_rate_array(net, T, M, nasa9,
remove_list)` (the differentiable port of `rates`+`gibbs`, bit-exact to ~5e-14
vs the NumPy build) and recompute the structural cascade. `atm_jax.build_atm_static`
now rebuilds `M = pco/(kb*T)`, `dz`, `Hp`, and the molecular-diffusion `Dzz(T)`
on-graph from `Tco`, so those are no longer frozen; only the host-side photo
cross-section T-interpolation (`photo_setup._bin_T_dependent`) stays frozen
(second-order). Forward-mode `d/dT` is validated against finite differences
(HD189 dominant species to 3–4 sig figs; WASP-39b SO2 to correlation 1.0; the
validation scripts live in the maintainer's internal manuscript repo, not in
this package).

### Reverse-mode (solver-map steady-state adjoint)

Reverse-mode answers the many-inputs/one-output question: *which of the
network's reactions set the converged abundance of a species*. One adjoint solve
returns `dL/d(ln k_r)` for every reaction, where finite differences would cost
one re-converged model each.

```python
import jax.numpy as jnp
from vulcan_jax import composition
from vulcan_jax.steady_state_grad import steady_state_reaction_sensitivity

def loss(y):                       # log10 SO2 VMR at its peak layer L
    return jnp.log10(y[L, so2] / y[L].sum())

dL_dlnk = steady_state_reaction_sensitivity(   # (nr+1,)
    loss, y_star, k_arr, atm, net,
    compo_array=composition.compo_array[:ni], dz=dz,
    integ=integ, converged_state=final_state,  # enables default dJ/dy on photo-on runs
)
```

`lax.while_loop` blocks `vjp`, so this is the steady-state adjoint of the body
map, not a backprop through the loop: at convergence `G(y*) = y*`, and
`(I - dG/dy)^T z = v` is solved with the integrator's own regularized step as the
operator, in log-abundance coordinates, with the conserved-mass null space
deflated, by LGMRES (an augmented Krylov method — restarted GMRES oscillates and
a raw Neumann iteration diverges on this indefinite, singular operator). Earlier
attempts that took the adjoint of the bare residual `df/dy` directly all failed —
on a closed column it is both singular (mass conservation) and severely
ill-conditioned (stiff chemistry) — which is why the solver-map route exists.

**Accuracy — percent-level by default; a legacy raw-step map is the only ~few-% case.**

- **Default `solver_map="renorm"`: percent level.** HD189 CH4 **~0.7%** vs
  finite differences (`y*` is a ~1e-9 fixed point of the renormalized map the
  loop actually iterates); HD209 forward rows 35% → 1%. A 2026-07-03 campaign
  (HD189/HD209/WASP-39b, checked against re-converged FD *and* forward-mode
  `jvp`, which agree to 0.02%) established that the residual error is a
  **linearized-map** effect, not the "convergence-criterion mismatch" once
  documented here: FD is invariant across `yconv` 1e-2..1e-4, and stricter
  convergence, a `body_dt` scan, and a bigger LGMRES budget do not move it.
- **On photochemistry-on columns, the default `photo_recompute_k="auto"` reaches
  percent level** when given the finished runner context
  (`runner_photo_static=integ._photo_static, converged_state=final`, or
  `integ=integ, converged_state=final`). It rebuilds `J(y)` through the runner's
  two-stream RT each application so the operator carries `dJ/dy`, removing the
  frozen-photolysis error that otherwise leaves those rows leading-order (~11%).
  With the default renorm map, **WASP-39b SO2 dominant rows reach r1 0.2% /
  r691 0.1%** vs re-converged FD — the paper's science case at percent level. It
  costs an RT solve per Krylov matvec. Pass `photo_recompute_k=None` only to
  reproduce the legacy frozen-photolysis result.
- **Legacy `solver_map="bare"`** linearizes the raw Ros2 step (`y*` only a ~1e-4
  fixed point) and lands at ~few % (HD189 CH4 ~6.6%, WASP-39b OH+H2 ~11%); kept
  only to reproduce the pre-2026-07 behavior.
- **Genuinely hard residue:** HD209 CH2OH near-equilibrium *reverse* rows stay
  ill-conditioned (LGMRES resid ~0.1 even with renorm) and are flagged by the
  default-on diagnostics — treat as ranking there; forward-mode is exact.
- **The solver regime is set by `body_dt`** (an adjoint-only probe-step knob —
  the forward model is untouched). The default `body_dt=1e7` sits in the
  measured low-residual regime (HD189: resid 0.04–0.15, twins land 0.3–6% from
  FD, mean 3.5%); at `body_dt≥1e8` (the old default) the solve *stagnates*
  (resid 0.2–0.7 — the body map has unstable top-layer H/H₂ eigenmodes and the
  matvec's FP floor grows with dt) and single-solve magnitudes bounce ~±25%;
  `body_dt~3e6` converges fully but deterministically underweights slow
  chemistry (~28% bias). The window is column-dependent: scan a few dt values
  and keep the lowest `info["resid"]` (dt map in `BODY_MAP_DT`'s comment).
  The gradient is returned as the **mean over an `n_solves` twin ensemble**
  (default 3, deterministic seeded RHS perturbations), with the twin spread in
  `info["ensemble_spread"]` as the honest magnitude error bar. The *ranking* is
  robust in every non-divergent regime: dominant reactions stand 1-2 orders of
  magnitude above the noise with stable signs.
- **Cross-regime validation (2026-07-02 battery).** WASP-39b (SNCHO photo-on,
  1150 reactions, SO₂ loss) is an *easy* regime: residuals 0.005–0.05 at every
  `body_dt` in 3e6–1e8, answers dt-insensitive to <1%, twin spread ~6e-4, and
  the ranking reproduces the paper exactly. On HD189, three loss regimes
  degrade and **all are flagged by default-on diagnostics**: buffered species
  (H₂O/CO mid-column — spread warns on the twin-noisy tail; the insensitivity
  conclusion itself is robust), upper-atmosphere losses (true stagnation —
  median-residual warns), and losses coupled to the unstable top-layer H/H₂
  modes (tiny residuals but `ensemble_spread` ~0.9 and
  `info["pair_antisym"]` ~1 — the forward/reverse pair-antisymmetry check
  catches internal inconsistency that residuals miss). Mid-column
  composition losses (the design use case) are safe because their cotangent
  is orthogonal to the unstable subspace.
- **Photolysis feedback is the default** on photochemistry-on columns when the
  finished runner context is supplied; without that context the default raises
  instead of silently returning the leading-order frozen-photolysis result.
- **Physical-input gradients via the same solve:**
  `steady_state_input_sensitivity(loss, y_star, k_arr, atm, net, p0, rebuild, ...)`
  returns `dL/dp` for an arbitrary input pytree — e.g. a full (nz,)
  temperature profile in ONE adjoint solve plus one VJP — given a
  differentiable `rebuild(p) -> (k(p), atm(p))` (`rates_jax` + `_replace`;
  non-thermal rows spliced frozen). The rebuild is consistency-checked at
  `p0` (warn/refuse on mismatch), the renorm is differentiated through
  `atm(p).M` (temperature moves the rebalance), and the chemistry path
  `∂L/∂y*·dy*/dp` is returned — a spectrum loss's direct `∂L/∂p` term (T in
  the RT) is added separately by the caller. Forward-mode `jvp` stays the
  exact route for a handful of directions.
- **Needs a genuine fixed point** (`y_star` tight under the *chosen* body map —
  tight for `"renorm"`, ~1e-4 for `"bare"`) and a `body_dt` in the safe regime
  (the danger zone is guarded).
- **The body map contains `ros2_step (+ renorm) (+ photo) (+ body_terms)`.**
  The optional `body_terms` (build with
  `make_body_terms(integ, converged_state, atm_static)` — it also returns the
  correctly spliced `atm`, including a live `vm` for `use_vm_mol`) carries the
  per-step processes a non-default config turns on: the in-window
  **condensation** composite (conden/evap k-rows recomputed from `y` — the
  dk/dy feedback analogous to photolysis dJ/dy — plus the H2O/NH3 relax
  kernels and the gas-only partial rebalance), the **fix_species** pins
  (species clamped inside the Ros2 step), and the **layer-0 boundary pins**
  (`use_fix_all_bot` / `use_fix_sp_bot` / tripped hycean H2-He). Everything
  else — clip (identity a.e.), ion charge balance (unsupported, raises), the
  escape-flux recompute and the composition→μ atm-refresh feedback (frozen,
  second-order) — stays outside the linearization. **A fingerprint guard
  raises instead of returning a silently wrong gradient**: a state converged
  with condensation active (nonzero conden rows or populated `*_l_s`
  condensates) is refused without matching terms, active ion rows are always
  refused, and frozen photolysis warns.
  **Run `audit_adjoint_scope(y_star, k_arr, atm, net, cfg=..., final_state=...,
  loss_fn=..., body_terms=...)` before trusting the gradient on any
  non-default config:** it classifies every dropped process for that config
  (error/warning/info), verifies the converged geometry was spliced into
  `atm`, and measures the **per-cell** fixed-point defect `|G(y*)−y*|/y*` of
  the exact map the solver uses — which catches unmodeled active processes
  that the global `fp_err` max-norm structurally masks (a pinned bottom trace
  row can be 100% off while `fp_err` reads 1e-9), plus the defect inside the
  loss's own footprint.

**Forward-mode (above) is the higher-accuracy route** for end-to-end gradients and
the right tool when the number of input directions is small; reverse-mode is the
right tool for all reactions at once. Validated on HD189 (CH4) and WASP-39b (SO2;
paper Fig 8). The solve is host-side scipy LGMRES (JAX has no LGMRES), one-shot
post-convergence, off the hot path; warm-start cycles stop early once scipy
reports convergence. Diagnostics are default-on: warnings fire on a poor LGMRES
residual, a loose fixed point, or a large twin-ensemble spread, an LGMRES
breakdown or a rank-deficient
deflation basis raises instead of returning garbage, and `info["null_quality"]`
reports how null the deflated conserved-mass directions actually are, relative
to the operator's scale (~3e-5 on the healthy closed HD189 column — the atom-count
vectors are only *approximately* null because the diffusion discretization is not
exactly conservative under the dz weights; O(1) means broken conservation, e.g.
open boundary fluxes). See `examples/grad_reverse_example.py`,
`tests/test_steady_state_reaction_sensitivity.py`, and the full log in `docs/notes.md`.

**What's NOT differentiable** (by design): host-side file readers (`photo_setup.py`, `composition.py`, `atm_setup.py` CSV loaders), FastChem subprocess. To differentiate through these, build the corresponding pytree directly with JAX arrays.

---

## Benchmarks

Per-step kernel timing on HD189 from `python benchmarks/bench_step.py`. Numbers
below are from one reference CPU host (single-threaded, `jax==0.6.2`, float64);
they are hardware- and version-dependent, so re-run the benchmark on your machine.

| Step | Master (NumPy) | VULCAN-JAX | Speedup |
|---|---:|---:|---:|
| Single Ros2 step | 118.5 ms | 37.2 ms | 3.2x |
| 50-step OuterLoop | -- | 50.2 ms/step | -- |

Speedup comes from: analytical Jacobian (95 ms -> 2.6 ms), diagonal-aware block-Thomas (O(ni^3) -> O(ni^2)), JIT compilation of the full loop, and pre-baked y-independent diffusion terms.

### Where the per-step time actually goes

Profiling master's Ros2 step *by operation* (single-threaded CPU, HD189) shows
the cost is dominated by genuine numerical kernels, not Python overhead. The
relative shares below are host-robust even though absolute ms are not:

| Operation | Master share | vs VULCAN-JAX | Why |
|---|---:|---:|---|
| **Linear solve** | **~50% of the step** | **~5x cheaper** | master calls `solve_banded` **twice** per step (two LU factorizations of the *same* matrix) and the band stores the species-diagonal off-blocks as if dense; block-Thomas factorizes once, reuses it for both Ros2 stages, and skips those zeros |
| Chemistry Jacobian | ~16% | ~6x cheaper | analytical (stoichiometry-driven) vs master's symbolic Jacobian |
| Transport + chemistry RHS | ~18% | ~30-60x cheaper | per-network codegen, XLA-fused, `y`-independent gravity pre-baked out |
| Banded repack into SciPy storage | ~7% | eliminated | the block-Thomas path never repacks into band storage |
| Python dispatch / glue / temporaries | ~3% | folded into one XLA program | master is already well-vectorized |

The headline correction to the usual "JAX removes Python overhead" story: Python
interpreter overhead is only ~3% here. Master's time is real kernel work, and the
**linear solve is the single biggest cost (~half the step)** -- so the
structure-aware, single-factorization block-Thomas is the dominant lever, with the
analytical Jacobian, fused RHS, and the eliminated banded repack stacking on top.

```bash
python benchmarks/bench_step.py   # run on your hardware
```

---

## Running tests

```bash
# Full suite
python -m pytest tests/ -q --tb=short -ra

# Parallel (FastChem serializes via fcntl.flock)
python -m pytest tests/ -n auto -q --tb=short -ra

# Filter
python -m pytest tests/ -k "ros2 or block_thomas"

# Parity audit vs upstream (requires ../VULCAN-master/)
python tools/audit_master_parity.py --master ../VULCAN-master
```

The suite imports the *installed* `vulcan_jax` (src layout), so development requires an **editable** install (`pip install -e .`). A non-editable install (e.g. `pip install .` from a release) would shadow the checkout and silently test stale code; `tests/conftest.py` fails collection with a clear message if `import vulcan_jax` resolves outside the repo's `src/`.

Master-comparison tests (those comparing against `../VULCAN-master/`) require the sibling checkout and skip cleanly when it's absent. These tests run master imports in isolated subprocesses.

The slowest test is `tests/test_nh3_conden_batch_subprocess.py` (~10 min cold): a fresh subprocess parses and compiles the 1141-reaction lowT-Jupiter network to prove batched NH3 condensation matches solo runs end-to-end. JAX's persistent compile cache makes identical reruns much cheaper.

The Earth example config (`cfg_examples/vulcan_cfg_Earth.py`) ships but cannot run — **in VULCAN-master either**. It lists Ar in `atom_list` / `const_mix`, but Ar appears in no reaction of the SNCHO network, so it is not a network species; master's `build_atm.ini_y` calls `species.index(sp)` unconditionally and crashes with the identical `ValueError: 'Ar' is not in list` (`build_atm.py:200`, reproduced end-to-end on the shipped Earth example). Inert background gases without network reactions were never live master physics, so VULCAN-JAX does not invent them; `runtime_validation` rejects such a `const_mix` upfront with an explanation instead of failing mid-setup (`tests/test_validation_const_mix_conden.py`). The Earth config is kept verbatim as upstream ships it — running it means removing Ar from `const_mix`/`atom_list` (master would additionally NaN-poison its atom-conservation diagnostics for any `atom_list` atom carried by no species, via 0/0 in `atom_loss`).

---

## Comparison to VULCAN-master

### Numerical agreement (per-component)

Measured on the default `NCHO_photo_network` (69 species, 878 reactions, 439
forward) against the VULCAN-master oracle. Each row is reproduced by the named
backing test; re-run those for the current numbers on your host.

| Layer | Agreement (max relative error) | Backing test |
|---|---|---|
| Forward rate coefficients (439 forward) | bit-exact | `test_rates` |
| Reverse rates (Gibbs-derived) | 1.4e-14 | `test_gibbs` |
| Atmosphere structure (pco/Tco/Kzz/M) | bit-exact | `test_default_master_parity` |
| Initial abundances (FastChem path) | bit-exact | `test_default_master_parity` |
| Chemistry RHS (`chem_rhs_codegen` vs oracle) | ~2e-13 worst cell; bulk species ~1e-16 | `test_chem_rhs_codegen` |
| Chemistry Jacobian (analytical vs jacrev oracle) | 2.8e-15 | `test_chem_jac_sparse` |
| Diffusion operator (vs `op.diffdf`) | ~1e-5 (FP-cancellation-bound); Jacobian blocks bit-exact | `test_diffusion` |
| Upwind molecular diffusion (`use_vm_mol`, vs `op.diffdf_vm`) | drift `vm` bit-identical to the shami `vm_branch` formula; operator ~2.7e-6 (FP-cancellation-bound) | `test_diffusion_variants`, `test_diffusion_production_kernel` |
| Block-Thomas solver | 3e-15 | `test_block_thomas_diag` |
| Single Ros2 step (vs `op.Ros2.solver`) | 1.6e-9 (full step) | `test_ros2_step` |
| Photo kernels (tau/flux/J) | 7e-16 / 1.2e-11 / 6.8e-12 | `test_photo` |
| End-to-end converged HD189 (median dex) | ~0.004 dex (~1%); no automated convergence oracle yet | -- |

### Compatibility surface

| Surface | Compatible? |
|---|---|
| `vulcan_cfg.py` format | Yes -- same keys, same format |
| Network / atmosphere / cross-section files | Yes -- same parsers, vendored data |
| `.vul` output schema | Yes -- same public keys, shapes, dtypes |
| `plot_py/` scripts | Yes -- unchanged |
| Solver | Ros2 only (non-Ros2 solvers were dead code in master) |

### Intentional differences

- Live UI fires between JIT'd step batches (cadence-faithful, not call-site-identical)
- Output writer synthesizes `J_sp`/`Jion_sp` at pickle time rather than incrementally
- Convergence stall fallback (`conv_stall_window`) handles heavy-hydrocarbon oscillation
- Upfront config validation is stricter than master: non-network `const_mix` keys and unsupported `condense_sp` entries fail at validate time with an explanation (master crashes deep in setup for the former and silently zero-rates the latter)
- `use_print_delta` (master's largest-truncation-error print inside the solver) is declared for config-surface compatibility but not consumed — a per-step host print is impractical inside the JIT'd runner
- Upwind molecular diffusion (`use_vm_mol`) uses the interface-centered drift velocity from the shami `vm_branch` `op.update_mu_dz` (shape `(nz-1, ni)`, harmonic-mean interface scale height). The drift `vm` is **recomputed in-loop** (every `update_frq` steps) from the live mean molecular weight, mirroring upstream `op.update_mu_dz`'s "# Also update vm" — it is *not* frozen at setup like `Dzz`/`Ti`, because `vm ∝ (… − 1/Hpi …)` and `Hpi` tracks `mu`. Freezing it (an earlier bug) biased a molecular-diffusion-dominated upper atmosphere (low Kzz) by up to ~1.7 dex; the refresh collapses the new-vs-upstream gap to ≤0.16 dex (the stiff-regime convergence floor) and matches upstream's step count. VULCAN-JAX keeps `vm` consistent at the bottom boundary (`j=0`) in every mode; upstream `op.diffdf_settling_vm` drops `vm` at `j=0` (a self-inconsistency vs its own `op.diffdf_vm`), so the doubly-non-default `use_vm_mol + use_settling` combination differs from upstream only at that one cell. We also port the *correct* `axis=0` layer-averaging form (`op.update_mu_dz`); the `build_atm.py` copy of that formula omits `axis=0`, a latent species-mixing bug — see CLAUDE.md.

---

## Numerical notes

### Chemistry RHS parity

The production JAX path uses `make_chem_funs.build_chem_rhs(net)` to emit per-network code in the same order as VULCAN-master's SymPy-generated `chemdf`: paired reactions, stoich-repeated multiply chains, asymmetric third-body M, products-before-reactants accumulation. Bit-faithful to master's `chemdf` to ~1 ULP per multiply chain.

### Step-count drift

JIT compilation lets XLA reorder floating-point operations, so large production/loss cancellations don't round identically. The per-step C-atom residual (~5e-7 of per-layer budget) is corrected by `jax_step._project_chem_rhs`, which enforces exact per-element conservation after each RHS evaluation — H/O/C/N on the C-H-N-O networks, and additionally S (via the H2S reservoir) on the sulfur network. Overhead is ~3% per step.

### float64 is non-negotiable

Rate constants span ~50 orders of magnitude. float32 silently fails. `jax_enable_x64 = True` is set at import time.

### Correctness fixes in JAX relative to VULCAN-master

The JAX port corrects several issues present in master:

- **H2S saturation-pressure conversion.** Upstream master converts the Giauque & Blue (1936) saturation-vapor-pressure formula — which is expressed in cm Hg — with the mm Hg constant (`0.001333` instead of `0.01333`), a factor-of-ten error in the H2S saturation pressure. (Upstream `build_atm.py`; the sibling `../VULCAN-master/` copy used as the test oracle carries the fix too.)

- **NH3 ice molecular weight.** Upstream `thermo/all_compose.txt` lists `NH3_l_s` at 16.023 instead of 17.031. (Fixed in the vendored composition table and in the sibling master copy.)

- **Diffusion Jacobian self-consistency.** Master's `op.lhs_jac_tot` disagrees with the analytical derivative of `op.diffdf` at a handful of diagonal cells for heavy condensable species (S8, layers 5 and 25). JAX's block-diagonal diffusion Jacobian matches the analytical derivative to machine precision. Impact on integration is negligible.

- **Atom conservation projection.** XLA's floating-point fusion breaks the stoichiometric nullspace of the chemistry RHS (production and loss terms that should cancel exactly don't, due to FMA rewriting). `jax_step._project_chem_rhs` distributes the per-layer atom residual (~5e-13 relative per step) across one abundant reservoir species per conserved element — H2, H2O, CO, N2, and H2S on the sulfur network — after each RHS evaluation, enforcing exact conservation of H/O/C/N (and S where the network carries it). The reservoir/atom pairing is selected dynamically from `atom_list` (`jax_step._ATOM_RESERVOIRS`), so any atom subset with an abundant reservoir is conserved, not a hard-coded H/O/C/N set. Master does not have this correction; its atom drift is comparable in magnitude but arises from a different source (Python evaluation order).

---

## License & citation

VULCAN-JAX inherits its license from VULCAN (GPLv3).

If you use VULCAN-JAX in published work, please cite:

- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20
- Tsai, S.-M., et al. 2021, ApJ, 923, 264
