# VULCAN-JAX

A JAX-accelerated, differentiable port of [VULCAN](https://github.com/exoclime/VULCAN) -- the photochemical-kinetics solver for exoplanet atmospheres (Tsai et al. 2017, 2021).

VULCAN-JAX runs the same configuration files, input data, and `.vul` output schema as upstream VULCAN. The hot path is a single JIT-compiled `lax.while_loop` on CPU or GPU. The runtime is standalone -- no `../VULCAN-master/` sibling required.

**Why use this over upstream VULCAN?**
- ~3x faster per-step on CPU (single-threaded; see Benchmarks); end-to-end speedup is workload-dependent
- Differentiable: forward-mode through the runner, reverse-mode via implicit steady-state gradients
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

Core (installed automatically): `jax>=0.4.31`, `numpy>=1.24`, `scipy>=1.10`, `h5py>=3.8`, `sympy>=1.12`

Optional extras:
- `pip install -e ".[dev]"` adds `pytest`, `pytest-xdist`, `ruff`, `vulture`, `build`, `twine`
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
│   ├── steady_state_grad.py Implicit-FT custom_vjp for reverse-mode AD
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
├── docs/                    Project docs (file_organization.md, BUGS_FOUND.md)
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
| `docs/` | `file_organization.md` (per-module function index), `BUGS_FOUND.md` (validation bug log) | Yes |
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

The `-n` flag is accepted as a no-op for upstream compatibility.

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

### Known first-run behavior

On the first import with a given network, `make_chem_funs` generates and caches a per-network chemistry RHS source file. This takes a few seconds and prints informational messages. Subsequent imports reuse the cache.

The "Element # not found. Neglected!" messages from FastChem are normal -- the solar abundance file has heavy elements not in the C-H-O-N network, which are safely ignored.

The FastChem binary is **not** vendored as a pre-built executable; the C++ source and makefiles are. The first time `ini_mix='EQ'` runs, `ini_abun._ensure_fastchem_binary()` compiles it from source (`make` in `fastchem_vulcan/`, creating `obj/` and the runtime `input/`/`output/` dirs as needed) and reuses it thereafter. The build is serialized by the same `fcntl.flock` that guards the EQ subprocess, so `pytest -n auto` and parallel host-setup workers won't race to build. A C++ toolchain (`c++`/`make`) must be on `PATH`; if the build fails, run `make` manually under `fastchem_vulcan/`.

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
| `step_size_safety` | `0.9` | Ros2 step-size safety factor |
| `fastchem_newton_tol` | `1e-12` | Newton solver tolerance for `ini_mix='EQ'` |

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

### Forward-mode (works through entire integration)

```python
import jax

def integrate_fn(k_arr):
    rs = build_runstate_from_k(k_arr)
    return integ(rs).step.y

y_star, dy_dk = jax.jvp(integrate_fn, (k_arr,), (k_arr_tangent,))
```

### Reverse-mode (via implicit-function theorem)

For high-dimensional inputs, use `steady_state_grad.py` -- O(1) memory in step count:

```python
from vulcan_jax.steady_state_grad import steady_state_value_and_grad

loss, grad_inputs = steady_state_value_and_grad(
    loss_fn, inputs, y_star, net, residual_rtol=1e-6
)
```

See `examples/grad_implicit_example.py` and `examples/grad_jvp_example.py` for worked examples. See `tests/test_steady_state_grad.py` for the canonical validation pattern.

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

Master-comparison tests (those comparing against `../VULCAN-master/`) require the sibling checkout and skip cleanly when it's absent. These tests run master imports in isolated subprocesses.

The Earth example config (`cfg_examples/vulcan_cfg_Earth.py`) ships but is not covered by the setup/oracle tests. It lists Ar (and other inert background gases) in `atom_list` / `const_mix`, but Ar is chemically inert and is therefore not a species in any reaction network, so `ini_abun`'s `const_mix` initializer raises `'Ar' is not in list`. Running it needs the `const_mix` path to handle inert atoms that have no network species (an open item); until then the config is provided as a starting point, not a tested path.

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

---

## Numerical notes

### Chemistry RHS parity

The production JAX path uses `make_chem_funs.build_chem_rhs(net)` to emit per-network code in the same order as VULCAN-master's SymPy-generated `chemdf`: paired reactions, stoich-repeated multiply chains, asymmetric third-body M, products-before-reactants accumulation. Bit-faithful to master's `chemdf` to ~1 ULP per multiply chain.

### Step-count drift

JIT compilation lets XLA reorder floating-point operations, so large production/loss cancellations don't round identically. The per-step C-atom residual (~5e-7 of per-layer budget) is corrected by `jax_step._project_chem_rhs`, which enforces exact H/O/C/N conservation after each RHS evaluation. Overhead is ~3% per step.

### float64 is non-negotiable

Rate constants span ~50 orders of magnitude. float32 silently fails. `jax_enable_x64 = True` is set at import time.

### Correctness fixes in JAX relative to VULCAN-master

The JAX port corrects several issues present in master:

- **Self-consistent gravity in atmosphere refresh.** Master's `f_mu_dz` (build_atm.py:552) computes `g[i] = gs * (Rp/(Rp + zco[i]))^2` using the *previous refresh cycle's* `zco`, not the value just computed in the current scan. JAX's `update_mu_dz_jax` uses a sequential `lax.scan` where each layer's `g` is computed from the freshly-updated `zco` carry. This produces self-consistent hydrostatic profiles; the difference is ~1.8% at the top of atmosphere for HD189.

- **Diffusion Jacobian self-consistency.** Master's `op.lhs_jac_tot` disagrees with the analytical derivative of `op.diffdf` at a handful of diagonal cells for heavy condensable species (S8, layers 5 and 25). JAX's block-diagonal diffusion Jacobian matches the analytical derivative to machine precision. Impact on integration is negligible.

- **Atom conservation projection.** XLA's floating-point fusion breaks the stoichiometric nullspace of the chemistry RHS (production and loss terms that should cancel exactly don't, due to FMA rewriting). `jax_step._project_chem_rhs` distributes the per-layer atom residual (~5e-13 relative per step) across reservoir species (H2, H2O, CO, N2) after each RHS evaluation, enforcing exact H/O/C/N conservation. Master does not have this correction; its atom drift is comparable in magnitude but arises from a different source (Python evaluation order).

---

## License & citation

VULCAN-JAX inherits its license from VULCAN (GPLv3).

If you use VULCAN-JAX in published work, please cite:

- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., & Heng, K. 2017, ApJS, 228, 20
- Tsai, S.-M., et al. 2021, ApJ, 923, 264
