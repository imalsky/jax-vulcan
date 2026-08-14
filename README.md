# VULCAN-JAX

VULCAN-JAX is a JAX port of
[VULCAN](https://github.com/exoclime/VULCAN), a one-dimensional chemical
kinetics model for planetary atmospheres. It supports exoplanet
photochemistry, vertical transport, condensation, and ion chemistry.

VULCAN-JAX reads the same reaction network files, atmosphere files, and
configuration names as VULCAN, and writes the same `.vul` output format. It runs
the integration loop with JAX on a CPU or GPU.

The vendored network files come from upstream at the exact revisions recorded in
[`tests/science_sources.yaml`](tests/science_sources.yaml), which also lists
every vendored scientific input with its SHA-256 and every deliberate divergence
from upstream with its reason.

**One unresolved network decision:** the N-C-H-O network here contains
`NH3 + CH -> NH2 + CH2`, which upstream removed in commit `39f1906` and which is
absent from both current upstream branches. It is unchanged pending a decision
from the upstream author, because adding or removing a reaction changes the
chemistry. This is why the manifest pins the NCHO oracle family to `80f75b9`,
that commit's parent: reaction indices are positional, so comparing against a
later revision shifts every index past this reaction and produces failures
unrelated to the ported kernels. Full context: the **Validation** section below.

## Main capabilities

- Just-in-time (JIT) compiled Rosenbrock integration for stiff chemical kinetics
- Forward-mode differentiation through the compiled integration loop (host-side
  setup is excluded; see the **Differentiability** section below)
- Reverse-mode reaction sensitivity at a converged state
- Batched atmosphere runs with `jax.vmap`
- Analytical chemistry Jacobians
- Photochemistry and molecular or eddy diffusion
- Optional condensation and ion chemistry
- VULCAN-compatible `.vul` output

## Requirements

- Python 3.10 or later
- JAX 0.4.31 or later. Continuous integration runs JAX 0.6.2, which is the
  version the numerical tests are calibrated against. Newer JAX runs the model
  correctly, but two tests compare against tolerances that assume the older
  compiler and report differences of about 1e-6 on it. See **Running the
  tests** below.
- NumPy 1.24 or later
- SciPy 1.12 or later
- PyYAML 6 or later
- A C++ compiler and `make` for FastChem equilibrium initialization

VULCAN-JAX always uses 64-bit floating-point values. The chemical rate
constants span too large a range for 32-bit values.

## Install

### Install the package

The current release is on TestPyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vulcan-jax
```

Check the installation:

```bash
python -c "import vulcan_jax; print(vulcan_jax.__version__)"
```

### Install for development

```bash
git clone https://github.com/imalsky/jax-vulcan.git
cd jax-vulcan

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plot]"
```

The editable install is required to run the tests. The test suite imports the
installed package, so a released copy in `site-packages` would shadow your
checkout and test the wrong code.

For an NVIDIA GPU, install the correct JAX GPU package for your CUDA version.
See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick start

Run the default HD 189733 b model:

```bash
vulcan-jax --config default
```

The run writes three files under `output/`:

- `HD189.vul` contains the model result
- `HD189.vul.config.yaml` contains the complete resolved configuration
- `cfg_HD189.txt` is a plain-text snapshot of the same configuration, written
  for compatibility with VULCAN. Use the YAML file to reproduce a run; it is the
  one `vulcan-jax --config` can read back.

The first run is slower, by roughly half a minute on a laptop CPU. It compiles
FastChem (about 10 seconds, once per installation), generates the chemistry
function for the network (under a second), and compiles the JAX program (about
4 seconds). Later runs reuse all three, and the compile step drops to under
2 seconds.

The compiled program is cached in `~/.cache/jax_vulcan`. Delete that directory
to force a rebuild.

Run any other supplied configuration the same way:

```bash
vulcan-jax --config W39b
```

### When a run stops

A run ends when the atmosphere reaches a photochemical steady state. VULCAN-JAX
measures this with two numbers. `longdy` is the largest relative change in any
mixing ratio over the convergence lookback window. `longdydt` is that same
change per unit of model time.

The run has converged once both numbers fall below their thresholds
(`yconv_cri` and `slope_cri`) and the actinic flux has stopped changing
(`flux_cri`). A minimum model time (`trun_min`) and a minimum step count
(`count_min`) must also have passed. An atmosphere therefore cannot converge
before it has had time to evolve.

A run that does not converge stops at a limit instead. There are three: the
model runtime, the step count, and the wall-clock budget. The final message says
which one it hit.

The output file records why the run stopped twice, and neither field covers
every case on its own.

`end_case` is `1` for converged, `2` for runtime, `3` for step count, and `4`
for the wall-clock budget.

`termination_reason` is finer where it applies: `0` still running, `1` converged
normally, `2` runtime, `3` step count, `4` the stall fallback, `5` a non-finite
state. It is the only field that separates a normal convergence from a stall
convergence, which `end_case` reports as `1` in both cases.

Two limits of `termination_reason` are worth knowing:

- A wall-clock exit leaves it at `0`. The host stops the run between compiled
  chunks, so the loop never evaluates its own stopping test. Read `end_case`
  for that case; it reports `4`.
- Reason `5` is only ever set by the batched runner (`OuterLoop.run_batch`),
  which freezes a lane at its last finite state. A single-profile run does not
  produce it.

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) is the gentlest
introduction. It builds an HD 189733 b model and plots the results.

## Use VULCAN-JAX from Python

This example builds a configuration and runs the model:

```python
import vulcan_jax
from vulcan_jax import legacy_io, op_jax, outer_loop

cfg = vulcan_jax.make_config(
    count_max=5000,
    use_photo=True,
    use_print_prog=False,
)

state = vulcan_jax.RunState.with_pre_loop_setup(cfg)
output = legacy_io.Output(cfg=cfg)
solver = op_jax.Ros2JAX()
model = outer_loop.OuterLoop(solver, output, cfg=cfg)
result = model(state)

print("Converged:", result.params.end_case == 1)
print("Accepted steps:", int(result.params.count))
print("Mixing-ratio shape:", result.step.ymix.shape)
```

Pass the same `cfg` object to setup, output, and integration. This keeps all
parts of the run consistent.

## Configuration

VULCAN-JAX uses YAML configuration files. The package supplies:

| Name | Target | Network | Mode | Result |
| --- | --- | --- | --- | --- |
| `default` | HD 189733 b baseline, the recommended first run | NCHO | VULCAN 2 parity | Converges in 1495 steps |
| `HD189` | HD 189733 b | NCHO | VULCAN 2 parity | Converges in 1495 steps |
| `HD209` | HD 209458 b | NCHO | VULCAN 2 parity | Converges in 1206 steps |
| `W39b` | WASP-39 b | SNCHO (sulfur) | VULCAN 2 parity | see below |
| `HD189_vulcan3` | HD 189733 b | NCHO | **VULCAN 3** | see below |

Every one of these runs without error and reaches a steady state. `W39b` needs a
reaction network other than the default, which the command line selects for you;
see **Select a different network** below.

### VULCAN 2 parity versus VULCAN 3

The configurations come in two flavours, and the difference is deliberate.

**VULCAN 2 parity** configurations reproduce upstream VULCAN's *numerical
settings*. As fetched on 2026-07-30: molecular diffusion is central-difference,
`conver_ignore` is empty, `high_temp_cut` is off, adaptive `rtol` is off, and the
JAX-only stall fallback is off. Use these when you want a result comparable with
published VULCAN 2 work.

**"Parity" here means settings, not inputs.** These configurations still use
VULCAN-JAX's default elemental abundances, which are not upstream's. That alone
moves HD 189733 b by roughly 20 percent in the median, which is far more than any
solver difference. For a true like-for-like comparison you must also point
`fastchem_solar_abundance_file` at the full-solar file, on both sides. See
**Elemental abundances** below.

**VULCAN 3** configurations use the newer numerics from `shami-EEG/VULCAN`
`vm_branch`: hybrid molecular diffusion (first-order upwind to convergence, then
central difference), the high-temperature bottom cut, and adaptive `rtol`. Each
such setting in `HD189_vulcan3.yaml` carries a comment citing the `vm_branch`
line it came from. `HD189_vulcan3` is the same planet as `HD189`, so you can run
both and compare the two numerical schemes directly.

**The stall fallback is off in both flavours.** `use_conv_stall` has no upstream
counterpart. Measured on every case tried, it never fires: each one either
converges on the normal criterion first, or never gets close enough for the
fallback's gate to open. So no shipped configuration decides convergence on a
criterion VULCAN 2.0 lacks. The switch stays available if you want it, and the
code default is off, so a configuration that omits the key does not get it by
accident.

Use `termination_reason` to see which exit a run took: 0 still running, 1
converged normally, 2 runtime, 3 step count, 4 the stall fallback, 5 a non-finite
state. `end_case` cannot tell you this, because it reports 1 for both a normal
and a stall convergence.

### Elemental abundances

Two composition files ship, and which one you pick changes the answer more than
the solver does:

| `fastchem_solar_abundance_file` | Contents |
| --- | --- |
| `fastchem_vulcan/input/solar_element_abundances.dat` | Lodders 2019, rocky elements (P, Si, Ti, V, Cl, K, Na, Mg, F, Ca, Fe) suppressed to `-3.0`. **The default.** |
| `fastchem_vulcan/input/solar_element_abundances_lodders2009.dat` | Lodders 2009, every element at solar. **What upstream VULCAN ships.** |

The suppression is right for the networks shipped here. They contain no Mg, Si or
Fe species, so at solar abundance FastChem locks oxygen into MgO, SiO2 and FeO
that the chemistry can never release. For a carbon-hydrogen-nitrogen-oxygen
network the only value that actually differs between the two files is helium,
which takes part in no reaction. That is why the difference is easy to miss: it
shows up as a flat offset deep in the atmosphere rather than as anything that
looks like a chemistry problem.

It matters. On HD 189733 b, VULCAN-JAX and upstream VULCAN differ by about 20
percent in the median if you leave them on different files, and by 3.8e-06 once
you match them. **Match this file before comparing against any other code.**

Every supplied configuration states which file it uses. A file that is neither
preset is rejected at validation time rather than accepted quietly.

The step counts above are for the parity configurations. They are reproducible
but not portable claims: a step count depends on every convergence knob, so
quote it together with the configuration it came from.

Every shipped configuration converges. Two that did not, `Earth` and `K2-18b`,
were removed on 2026-07-30 rather than shipped with a warning, because a
configuration that converges in neither code is misleading. Both are documented
in the **Validation** section below, including what blocks each one.

Use a file path to run a custom configuration:

```bash
vulcan-jax --config path/to/my_config.yaml
```

For a bare name, the loader first checks `./configs/<name>.yaml`. It then
checks the configurations inside the installed package. Each run saves the
complete resolved configuration next to the `.vul` result, so any run can be
repeated exactly.

### Select a different network

The reaction network, the atom list, and the composition table are read once.
This happens when Python first imports VULCAN-JAX. The parsed network, the
species tables, and the generated chemistry function are all built from them.
No later setting can change them in the same process.

On the command line you do not have to do anything. If the configuration you
asked for names a different network, `vulcan-jax` sets the three environment
variables below and restarts itself once, then reports what it did:

```
Config 'W39b' needs import-frozen settings that differ from the defaults; relaunching with VULCAN_JAX_ATOM_LIST=H,O,C,N,S VULCAN_JAX_NETWORK=thermo/SNCHO_photo_network.txt
```

Set the variables yourself when you drive VULCAN-JAX from Python, or when you
want a network that no configuration file names. They must be set before the
first import, and a value you set explicitly always wins:

```bash
VULCAN_JAX_NETWORK=/absolute/path/to/network.txt VULCAN_JAX_ATOM_LIST=H,O,C,N,S VULCAN_JAX_COM_FILE=/absolute/path/to/all_compose.txt vulcan-jax --config path/to/my_config.yaml
```

Restart Python before you change one of these values in a notebook or
interactive session.

One thing to know about the reaction number written at the start of each line in
a network file: VULCAN-JAX ignores it and counts the reaction's position
instead. VULCAN treats that column as output rather than input. It rewrites the
file in place on the first run so the numbers become 1, 3, 5, and so on
(`make_chem_funs.py:71-72,109`). VULCAN-JAX does not rewrite vendored data, so a
file that has never been run through VULCAN still carries whatever numbers its
author left in it. Six of the networks shipped here are in that state. This does
not affect the rates, but it does mean `remove_list` is a list of **positions**,
not of the numbers printed in the file. If you write `remove_list` by reading
numbers off an un-renumbered network you will silently disable the wrong
reactions, in VULCAN as well as here. VULCAN-JAX warns when it loads such a
file.

VULCAN has the same one-network-per-process restriction, but reaches it another
way. There you edit `network` and `atom_list` in `vulcan_cfg.py`. `vulcan.py`
then runs `make_chem_funs.py` to write a new `chem_funs.py` before importing it.
VULCAN-JAX is an installed package with a cached chemistry function, not a
script sitting next to an editable `vulcan_cfg.py`. It therefore reads the same
choice from the environment instead of from a file it would have to rewrite.

## Differentiation

Use forward-mode differentiation when the model has a small number of input
parameters. The complete `lax.while_loop` supports `jax.jvp` and `jax.jacfwd`,
so a tangent can travel through the whole integration.

This example differentiates the converged mixing ratios with respect to eddy
diffusion. Run the model once to compile the integration loop, then push a
tangent through the compiled loop with `jax.jvp`:

```python
import jax
import jax.numpy as jnp

import vulcan_jax
from vulcan_jax import legacy_io, op_jax, outer_loop
from vulcan_jax.jax_step import make_atm_static
from vulcan_jax.state import legacy_view

cfg = vulcan_jax.make_config(use_print_prog=False)
state = vulcan_jax.RunState.with_pre_loop_setup(cfg)
model = outer_loop.OuterLoop(op_jax.Ros2JAX(), legacy_io.Output(cfg=cfg), cfg=cfg)
model(state)  # one run compiles the integration loop

_var, atm, _para = legacy_view(state, cfg=cfg)
nz, ni = state.step.y.shape
atm_static = make_atm_static(atm, ni, nz, cfg=cfg)
state0 = model._pack_state_from_runstate(state)


def ymix_from_Kzz(Kzz):
    final = model._runner(state0, atm_static._replace(Kzz=Kzz))
    return final.y / jnp.sum(final.y, axis=1, keepdims=True)


# The tangent is Kzz itself, so the result is d(ymix) / d(ln Kzz).
Kzz = atm_static.Kzz
ymix, dymix = jax.jvp(ymix_from_Kzz, (Kzz,), (Kzz,))
print("sensitivity shape:", dymix.shape)
```

Differentiate `model._runner`, not `model`. The outer object copies arrays back
to the host between steps, which breaks tracing. To differentiate with respect
to something else, substitute a different field of `atm_static`, or the rate
array. Temperature needs one extra step, because the rate constants are built
before the loop starts. See the Differentiability section below.

Longer examples:

- [`examples/grad_jvp_example.py`](examples/grad_jvp_example.py)
- [`examples/grad_physical_example.py`](examples/grad_physical_example.py)

Use the steady-state adjoint when you need sensitivities to many reaction
rates. It calculates reaction importance after the atmosphere has converged.
Always inspect the returned residual and stability diagnostics. See
[`examples/grad_reverse_example.py`](examples/grad_reverse_example.py).

The following operations are not differentiable through the full setup:

- FastChem equilibrium initialization
- Host-side file loading
- Temperature-dependent photolysis cross-section setup
- The current condensation pinning procedure

For the full scope and the accuracy measurements, see the **Differentiability**
section below.

## Batched runs

`OuterLoop.run_batch` runs several atmosphere profiles in one device call.
Profiles in one batch must use the same reaction network, vertical-grid size,
stellar spectrum, wavelength grid, and configuration switches. Only the
temperature-pressure profile may vary.

Start with:

- [`examples/batched_run.py`](examples/batched_run.py) for a small example
- [`examples/gpu_benchmark.py`](examples/gpu_benchmark.py) for large GPU batches

## Tests and benchmarks

Run the automated tests from an editable installation:

```bash
python -m pytest tests -q
```

Run tests in parallel:

```bash
python -m pytest tests -n auto -q
```

Some tests compare against upstream VULCAN. They need `$VULCAN_MASTER_DIR`
pointing at a clean clone at the commit `tests/science_sources.yaml` pins for
that test's oracle family; there is deliberately no sibling-directory fallback.
A wrong revision, a dirty tree, or an unset variable skips with the exact clone
commands, and `VULCAN_JAX_REQUIRE_ORACLE=1` turns those skips into failures.

The manifest pins two families at different commits, so one run cannot satisfy
both. Measured 2026-08-14: 304 passed and 7 skipped on `80f75b9`
(`vulcan2_ncho`), and the three tests that skip there pass on `8970337`
(`vulcan2_fastchem_ps_order`). Slow tests, including the adjoint ones, run only
when `VULCAN_JAX_RUN_SLOW=1`.

Two tests are sensitive to the JAX version and the processor, and they are the
reason continuous integration pins JAX 0.6.2. Both compare a number against a
tolerance that was set on one compiler, so neither indicates a wrong model.

- `test_vmap_photo_batch` requires a profile inside a batch to match the same
  profile run on its own to one part in a billion. On JAX 0.10 and later the
  batched and unbatched forms of the same arithmetic round differently in the
  last digit, the adaptive step-size controller resolves one comparison the
  other way, and the mixing ratios end up about one part in a million apart.
  Everything else in the run, including the rate constants and the radiation
  field, still agrees to thirteen digits.
- `test_forward_jvp_physical` cross-checks a derivative against a coarse
  one-step finite difference. The derivative is stable everywhere tested; the
  finite difference moves with the processor and overshoots the check's 30
  percent bound on x86 Linux while passing on arm64 macOS.

If you run the suite on newer JAX and see only these two, the installation is
fine.

Run the per-step benchmark on your computer:

```bash
python benchmarks/bench_step.py
```

Timings depend on hardware and software versions. Run the benchmark yourself
rather than quoting numbers from elsewhere.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/vulcan_jax/` | Main Python package |
| `src/vulcan_jax/configs/` | Supplied YAML configurations |
| `src/vulcan_jax/atm/` | Atmosphere profiles, boundary data, and stellar spectra |
| `src/vulcan_jax/thermo/` | Reaction networks and photochemical data |
| `src/vulcan_jax/fastchem_vulcan/` | FastChem source and input data |
| `examples/` | Single, batched, and differentiation examples |
| `tests/` | Unit, integration, parity, and regression tests |
| `benchmarks/` | Performance and molecular-diffusion benchmarks |
| `tools/` | Data preparation and VULCAN parity tools |

## Known limits

- Only the Rosenbrock-2 solver is supported.
- The full integration loop supports forward-mode differentiation. Standard
  reverse-mode differentiation cannot pass through the `lax.while_loop`.
- Reverse-mode steady-state sensitivities use an adjoint solve. Their
  diagnostics can show when a result is not reliable.
- Condensation can run in the forward model, but the condensation path is not
  validated for gradient inference.
- One process can use only one reaction network. See **Select a different
  network**.
- VULCAN-JAX contains documented corrections and intentional differences from
  VULCAN. Review the **Parity & bug guide** section below before a strict parity study.

## Validation

Everything about how VULCAN-JAX compares to VULCAN 2.0, in three sections
(consolidated 2026-08-02 from three separate docs; merged into this README
2026-08-11):

1. **Validation** (this section): how closely VULCAN-JAX reproduces VULCAN 2.0,
   what is compatible, and where the two codes differ on purpose.
2. **Benchmarks**: where the per-step time goes, and which optimizations
   account for the speedup.
3. **Parity & bug guide** (formerly `corrections_to_original_code.md`): the
   parity policy, every deliberate divergence that fixes a confirmed master
   bug, and the register of inherited and not-inherited upstream defects.

### Per-component agreement

Measured on the default `NCHO_photo_network` (69 species, 878 reactions, 439 of
them forward) against the VULCAN 2.0 oracle. Each row names the test that
reproduces it; re-run those tests for current numbers on your own host.

| Layer | Maximum relative error | Backing test |
|---|---|---|
| Forward rate coefficients (439 forward) | bit-exact | `test_rates` |
| Reverse rates, Gibbs-derived | 1.4e-14 | `test_gibbs` |
| Atmosphere structure (`pco`, `Tco`, `Kzz`, `M`) | bit-exact | `test_default_master_parity` |
| Initial abundances, FastChem path | bit-exact | `test_default_master_parity` |
| Chemistry right-hand side | about 2e-13 worst cell; bulk species about 1e-16 | `test_chem_rhs_codegen` |
| Chemistry Jacobian, analytical vs `jacrev` | 2.8e-15 | `test_chem_jac_sparse` |
| Diffusion operator | about 1e-5, bound by floating-point cancellation; Jacobian blocks bit-exact | `test_diffusion` |
| Upwind molecular diffusion (`use_vm_mol`) | drift `vm` bit-identical to the reference formula; operator about 2.7e-6, cancellation-bound | `test_diffusion_variants`, `test_diffusion_production_kernel` |
| Block-Thomas solver | 3e-15 | `test_block_thomas_diag` |
| Single Ros2 step | 1.6e-9 | `test_ros2_step` |
| Photochemistry kernels (tau, flux, J) | 7e-16, 1.2e-11, 6.8e-12 | `test_photo` |

### Converged-run agreement

Forcing both codes through the same accepted Ros2 steps from the same
HD 189733b state, the maximum relative difference stays between about 1e-15 and
2e-13 over a 1-20 step integration. That is the narrowest test: it checks only
that one step evaluates the same way in both codes.

For fully converged runs, the median fractional difference in volume mixing ratio
across all species is:

| Planet | Median fractional difference |
|---|---|
| WASP-39b | 1.1e-9 |
| HD 189733b | 3.3e-6 |
| HD 209458b | 1.4e-4 |

The largest single-cell differences reach order unity, but only in trace species
sitting at the numerical floor. For cells with a mixing ratio above 1e-10, the
largest single-cell difference in any run is 5.2e-4.

**These numbers only mean anything with the composition matched.** VULCAN-JAX and
upstream ship different `solar_element_abundances.dat` files; unmatched, HD 189733b
disagrees by ~20% median rather than 3.3e-6. See "Elemental abundances are a
config choice" below.

Maximum atom-sum drift is 2.6e-5 for HD 189733b, 3.1e-4 for HD 209458b, and
1.7e-5 for WASP-39b. All are far below VULCAN 2.0's default fractional
atomic-loss budget of 1e-1.

### Compatibility surface

| Surface | Compatible |
|---|---|
| Config format | YAML in `configs/*.yaml`, with the same knob names as VULCAN 2.0 |
| Network, atmosphere, and cross-section files | Yes, same parsers and vendored data |
| `.vul` output schema | Yes, same public keys, shapes, and dtypes |
| `plot_py/` scripts | Yes, unchanged |
| Solver | Ros2 only. The non-Ros2 solvers were dead code in VULCAN 2.0 |

### Deliberate differences

- **Live plotting fires between JIT-compiled step batches.** The cadence is
  faithful, but the call site is not identical.
- **The output writer synthesizes `J_sp` and `Jion_sp` at pickle time** rather
  than accumulating them incrementally.
- **A convergence stall fallback** (`use_conv_stall`) exists but is **off in
  every shipped config** and has no upstream counterpart. Measured 2026-07-30, it
  never fires on any shipped case, so nothing shipped decides convergence on a
  criterion VULCAN 2.0 lacks. `termination_reason` (0 running, 1 converged,
  2 runtime, 3 step count, 4 stall, 5 non-finite) reports which exit was taken;
  `end_case` cannot, because it reports 1 for both a normal and a stall
  convergence.
- **Config validation is stricter.** Non-network `const_mix` keys and unsupported
  `condense_sp` entries fail at validation time with an explanation. VULCAN 2.0
  crashes deep in setup for the first and silently zero-rates the second.
- **`use_print_delta` is not supported.** A per-step host print is impractical
  inside the JIT-compiled runner. The key is removed from the config surface and
  rejected with a migration message, as are `fix_species_time` (the pin gate is
  `stop_conden_time`) and `gs`.

#### Upwind molecular diffusion

`use_vm_mol` uses the interface-centered drift velocity from
`op.update_mu_dz`, with shape `(nz-1, ni)` and a harmonic-mean interface scale
height.

The drift `vm` is **recomputed in the loop**, every `update_frq` steps, from the
live mean molecular weight. This mirrors upstream's "also update vm" behavior. It
is deliberately not frozen at setup the way `Dzz` and `Ti` are, because `vm`
depends on `1/Hpi` and `Hpi` tracks the mean molecular weight.

Freezing it was an earlier bug, and it mattered: it biased a
molecular-diffusion-dominated upper atmosphere at low `Kzz` by up to about
1.7 dex. The refresh collapses the gap to 0.16 dex or less, which is the
stiff-regime convergence floor, and it matches upstream's step count.

Two smaller notes. VULCAN-JAX keeps `vm` consistent at the bottom boundary
(`j=0`) in every mode; upstream `op.diffdf_settling_vm` drops `vm` at `j=0`,
which is inconsistent with its own `op.diffdf_vm`, so the doubly-non-default
combination of `use_vm_mol` with `use_settling` differs from upstream at that one
cell. VULCAN-JAX also ports the correct `axis=0` layer-averaging form from
`op.update_mu_dz`; the copy of that formula in `build_atm.py` omits `axis=0`,
which is a latent species-mixing bug.

#### Hybrid molecular diffusion

`use_hybrid_vm_mol` is **off in the VULCAN 2 parity configs and on only in
`HD189_vulcan3.yaml`**. It integrates on the robust upwind scheme, then switches
to central difference and finishes, matching the sequence upstream describes:
first-order upwind, then convergence, then central difference for a bounded
number of further steps.

Because the runner is one JIT-compiled `lax.while_loop`, this is implemented as an
**in-loop phase flip** rather than a host-side two-stage driver. A carry blend
factor `hybrid_use_vm` starts at 1.0 for upwind and flips to 0.0 for central.

**The flip has three triggers, not one.** Phase 0 can never end a run, so it
hands over when it converges, when it exhausts the model runtime, or when it
exhausts the step count. Priority matches the reference driver's `stop()`:
convergence, then runtime, then step count. The budget the next phase receives
depends on which one fired:

| phase 0 ended by | phase 1 step budget | runtime |
|---|---|---|
| convergence | `count + 2000` | unchanged |
| model runtime | `count + 1000` | `runtime * 1.1` |
| step count | `count + 1000` | unchanged |

The flip also resets the convergence trackers, so phase 1 must satisfy the
criteria on its own.

**What the returned state does and does not guarantee.** A hybrid run that
terminates through the loop's own stopping test always ends in the
central-difference phase, because phase 0 is excluded from that test. It is a
central-difference **fixed point** only when phase 1 also converged, that is when
`end_case` is 1. A phase-1 run that exhausts its extended budget returns a
central-difference state that is not converged, and two paths can return while
still in phase 0: a wall-clock exit, which the host takes between compiled chunks
without consulting the phase, and the batched runner freezing a lane on a
non-finite step.

Forward-mode `jvp` through the runner and the steady-state adjoint apply to a
converged phase-1 state. Check `end_case == 1` and the terminal `hybrid_use_vm`
before treating a hybrid result as a differentiable steady state. Set
`use_vm_mol=False` for a fixed central-difference scheme, which is what batched
emulator generation wants, because it is deterministic.

### Numerical notes

#### Chemistry right-hand-side parity

`make_chem_funs.build_chem_rhs(net)` emits per-network code in the same order as
VULCAN 2.0's SymPy-generated `chemdf`: paired reactions, stoichiometry-repeated
multiply chains, asymmetric third-body M, and products-before-reactants
accumulation. It is faithful to about 1 unit in the last place per multiply chain.

#### Atom conservation projection

JIT compilation lets XLA reorder floating-point operations, so the large
production and loss terms that should cancel exactly do not round identically.
Fused multiply-add rewriting breaks the stoichiometric null space of the
chemistry right-hand side.

`jax_step._project_chem_rhs` distributes the per-layer atom residual, about 5e-13
relative per step, across one abundant reservoir species per conserved element:
H2, H2O, CO, N2, and H2S on the sulfur network. This enforces exact conservation
of H, O, C, and N, plus S where the network carries it. The reservoir-to-atom
pairing is selected dynamically from `atom_list` via `jax_step._ATOM_RESERVOIRS`,
so any atom subset with an abundant reservoir is conserved rather than a
hardcoded set. Overhead is about 3% per step.

This is not only a diagnostic fix. Because the corrected right-hand side is used
inside each Rosenbrock stage, the guard changes the integrated solution. For an
HD 189733b test, disabling the projection raises the median fractional difference
against VULCAN 2.0 from 7.3e-6 to 3.9e-4.

VULCAN 2.0 has no equivalent correction. Its atom drift is comparable in
magnitude but arises from Python evaluation order instead.

#### float64 is required

Rate constants span about 50 orders of magnitude, and float32 fails silently.
`jax_enable_x64 = True` is set at import time.

### Test-suite notes

The suite imports the **installed** `vulcan_jax`, because the package uses a src
layout. Development therefore requires an editable install. A non-editable
install would shadow the checkout and silently test stale code, so
`tests/conftest.py` fails collection with a clear message when `import
vulcan_jax` resolves outside the repo's `src/`.

Tests that compare against VULCAN 2.0 need `$VULCAN_MASTER_DIR` at the pinned
commit (see **Tests and benchmarks**) and skip cleanly with the clone commands
when it is unset. They run the VULCAN 2.0 imports in isolated subprocesses,
against a disposable copy of the checkout.

The slowest test is `tests/test_nh3_conden_batch_subprocess.py`, about 10 minutes
cold. A fresh subprocess parses and compiles the 1141-reaction low-temperature
Jupiter network to prove that batched NH3 condensation matches solo runs end to
end. JAX's persistent compilation cache makes identical re-runs much cheaper.

#### The K2-18b configuration was removed on 2026-07-30

It converged in neither code: VULCAN-JAX reaches longdy 25.80 (hybrid) or 0.459
(pure central) at 30000 steps and upstream VULCAN 2.0 reaches 1.00 at 3000,
against a criterion of 0.01. The cause is a timestep that pins near 1e-4 s
against a runtime of 1e16 s, not the VULCAN 3 hybrid scheme. Do not re-add it
without a converging run. Measurements and the falsified hypotheses: `notes.md`.

#### The Earth configuration was removed on 2026-07-30

Upstream's own example is unrunnable as shipped (correction C13 below). Made
runnable at a matched 2000-step budget it converges in neither code: VULCAN 2.0
longdy 0.880 vs VULCAN-JAX 54.33, criterion 0.01. Its large hydrogen
`atom_loss` is not a conservation failure; the diagnostic assumes a closed
column and Earth is open. Measurements: `notes.md`.

#### Elemental abundances are a config choice

VULCAN-JAX and upstream VULCAN do not start from the same composition, and this
is the single most common way to get a wrong cross-code number.

| file | contents |
|---|---|
| `fastchem_vulcan/input/solar_element_abundances.dat` | Lodders 2019, rocky elements (P, Si, Ti, V, Cl, K, Na, Mg, F, Ca, Fe) suppressed to -3.0. **The default.** |
| `fastchem_vulcan/input/solar_element_abundances_lodders2009.dat` | Lodders 2009, every element at solar. **What upstream ships.** |

`fastchem_solar_abundance_file` selects, and every shipped config declares it
explicitly. `runtime_validation` accepts either preset exactly and rejects
anything else, naming both, because a hand-edited composition otherwise fails
silently.

Suppression is right for the shipped networks: they have no Mg, Si or Fe species,
so at solar dex FastChem locks oxygen into MgO/SiO2/FeO that the kinetics can
never release. For a C-H-N-O network the only value that actually differs is
**helium** (10.9864 upstream, 10.9232 here), which is inert, which is why the
difference hid as a flat offset in the deep column rather than as anything that
looked like a chemistry bug.

Measured on HD 189733b, both codes converged, floor 1e-12:

| | median | max |
|---|---|---|
| upstream defaults | 2.0e-01 | 8.8e+01 |
| + VULCAN-JAX network | 2.0e-01 | 8.8e+01 |
| **+ network AND composition** | **3.8e-06** | 4.5e-02 |
| control: upstream vs upstream | 7.0e-06 | 1.7e-02 |

Selecting the upstream preset in VULCAN-JAX reproduces the expected direction at
the deep layer: He +13.1%, H2O -19.3%, CO2 -29.1%, CO -14.1%, H2 -2.2%. It also
converges in 911 accepted steps instead of 1495, so composition moves convergence
as well as abundances.

The upstream preset carries upstream's values verbatim with **one** deviation:
P and S are swapped back into the order FastChem's own
`mass_action_constant.cpp` expects (`index_P=5`, `index_S=6`). Upstream ships
them the other way round; that is correction C12, and copying the file verbatim
would have shipped the bug.

#### The hybrid scheme costs ~40% more steps and returns the same atmosphere

Measured 2026-07-31 on HD 189733 b, four runs, everything matched except the
diffusion scheme. VULCAN 2.0 is `shami-EEG/VULCAN` `vm_branch` (HEAD `84d010d`)
in BOTH rows, with only `use_vm_mol` / `use_hybrid_vm_mol` flipped, so the scheme
is never confounded with a code version. VULCAN 3.0 is `HD189.yaml` with only the
same two knobs flipped, NOT the full `HD189_vulcan3.yaml` preset.

| | central difference | upwind then central | penalty |
|---|---|---|---|
| VULCAN 2.0 | 1590 | 2201 | +611 |
| VULCAN 3.0 | 1495 | 2102 | +607 |

The two independent implementations pay almost exactly the same price, and
VULCAN 2.0's central run reproduces the Table 1 value of 1590 exactly.

**The scheme does not change the answer here.** Worst species of the whole
network, central versus hybrid within one code: VULCAN 2.0 0.008 dex,
VULCAN 3.0 0.000 dex. That is by construction, not a coincidence: the hybrid
converges on upwind and then *switches to central and finishes there*, so it
lands back on the central fixed point. Its value is stability, not accuracy. The
accuracy claim is the Zhang+2013 benchmark instead (central 0.8%, first-order
upwind 46%).

HD 189733 b is therefore NOT a case where the new scheme changes results: it runs
a constant `Kzz` of 1e10, so eddy mixing dominates and molecular diffusion has
little to act on. The scheme bites where molecular diffusion competes with eddy
mixing, meaning LOW `Kzz` and the upper atmosphere, where an earlier measurement
found up to 1.7 dex. No shipped config is in that regime. Do not present a
high-`Kzz` planet as evidence that the scheme matters.

Code-to-code agreement in this experiment: 0.001 dex worst species under central,
0.008 dex under the hybrid, both far below any physical significance.

The same four-run protocol on WASP-39 b (S-C-H-N-O network, `W39b.yaml` with
only the two scheme knobs flipped; VULCAN 2.0 again the fresh `vm_branch` clone
at `84d010d`, composition matched):

| | central difference | upwind then central | penalty |
|---|---|---|---|
| VULCAN 2.0 | 1202 | 1301 | +99 |
| VULCAN 3.0 | 1202 | 1301 | +99 |

Here the two codes take the same step count under both schemes, neither forced
to follow the other, which is the free-convergence cross-check the retracted
2026-07-27 claim wanted to be (that one had used the contaminated local master;
this one uses the fresh clone). It also resolves the 2026-07-24 "8% apart"
note: 1202 vs 1301 was central vs hybrid, not code vs code. Worst species
between the codes is 0.0000 dex under either scheme; worst SO2 difference in
every pairing is 0.0000 dex; the scheme itself moves NH3 by at most 0.0009 dex.
Figure: `jax_paper/figures/moldiff_scheme_comparison_W39b.png` from
`jax_paper/scripts/fig_moldiff_scheme_comparison.py`.


---

## Benchmarks

Where the per-step time goes, and which optimizations account for the speedup
over VULCAN 2.0. Moved out of the README in 2026-07; merged back 2026-08-11.

Run the benchmark on your own machine:

```bash
python benchmarks/bench_step.py
```

Absolute times are hardware- and version-dependent. The numbers below are from
one reference CPU host, single-threaded, `jax==0.6.2`, float64, on HD 189733b.
The relative shares are robust across hosts even where the absolute times are
not.

### Per-step timing

| Step | VULCAN 2.0 (NumPy) | VULCAN-JAX | Speedup |
|---|---:|---:|---:|
| Single Ros2 step | 118.5 ms | 37.2 ms | 3.2x |
| 50-step `OuterLoop` | -- | 50.2 ms/step | -- |

End to end on a single CPU, VULCAN-JAX converges 4.4-6.7x faster than
VULCAN 2.0 across the three benchmark planets. Those quoted times come from a
fresh subprocess with an empty compilation cache, so they include the one-time
XLA compilation.

### Where VULCAN 2.0 spends a step

Profiling VULCAN 2.0's Ros2 step by operation shows the cost is dominated by
genuine numerical kernels, not by Python overhead.

| Operation | VULCAN 2.0 share | vs VULCAN-JAX | Why |
|---|---:|---:|---|
| **Linear solve** | **about 50%** | **about 2.3x cheaper** | VULCAN 2.0 calls `solve_banded` twice per step, which is two LU factorizations of the same matrix, and its band stores the species-diagonal off-blocks as if they were dense. Block-Thomas factorizes once, reuses that factorization for both Ros2 stages, and skips the zeros |
| Chemistry Jacobian | about 16% | about 6x cheaper | Analytical and stoichiometry-driven, rather than symbolic |
| Transport and chemistry right-hand side | about 18% | about 30-60x cheaper | Per-network code generation, XLA-fused, with abundance-independent gravity terms pre-baked |
| Repacking into SciPy band storage | about 7% | eliminated | The block-Thomas path never repacks |
| Python dispatch, glue, temporaries | about 3% | folded into one XLA program | VULCAN 2.0 is already well vectorized |

The headline correction to the usual "JAX removes Python overhead" story: Python
interpreter overhead is only about 3% here. VULCAN 2.0's time is real kernel
work, and the **linear solve is the single largest cost, about half the step**. So
the structure-aware, single-factorization block-Thomas solver is the dominant
lever, with the analytical Jacobian, the fused right-hand side, and the
eliminated repack stacking on top.

### Attribution inside the linear solve

The 2.3x is measured, and it splits unevenly:

- **2.10x from reusing one factorization** across both Rosenbrock stages.
- **1.10x from skipping the off-block zeros.**

Skipping the zeros cuts the floating-point operation count by more than an order
of magnitude, but at about 69 species the factorization sweep is bound by memory
latency rather than arithmetic, so the reduced operation count does not convert
into proportional wall time.

The analytical Jacobian is 7-16x faster to evaluate than VULCAN 2.0's generated
path. The range depends on whether the comparison is against the chemistry-block
evaluation alone or the full Jacobian assembly.

### Batched throughput

On one NVIDIA GH200, throughput rises from 0.010 converged profiles per second
for a single profile to 0.76 for a batch of 256, a gain of about 75x. A batch of
256 already uses tens of GB, so it is the largest batch reported.

These numbers are for a homogeneous batch of HD 189733b-like profiles,
initialized from chemical equilibrium and run without photochemistry.
Heterogeneous grids that span a range of temperatures or chemical regimes pay a
larger straggler cost, and their average throughput is correspondingly lower.

### Two measurement rules

**Guard wall-clock timings.** Single-threaded VULCAN 2.0 must show
`user + sys` roughly equal to `real` before any ratio is believable. Thermal
throttling inflates wall time several-fold while CPU time stays roughly correct,
which has produced both a retracted 16.8x "speedup" and a run that never
converged. `tools/bench_table1.sh` enforces the check. Accepted **step counts**
are load-independent, so use those for any cross-code comparison.

**Benchmark solver changes on real matrices.** The matrix VULCAN-JAX factorizes is
`I/(gamma*dt) - J`. On a converged HD 189733b state its blocks reach a condition
number of about 1.4e18 at the `dt` a run actually sits at, and about 6.7e23 at the
`dt_max=1e11` that retrieval workloads configure. A synthetic
diagonally-dominant test matrix tops out near 4e2, which is 21 orders of magnitude
short, so an accuracy result measured on one is meaningless. This is not
hypothetical: an inverse-carry optimization that measured 2.0x faster under `jvp`
with "no worse residual" on synthetic blocks turned out to be 569x worse on real
ones at `dt_max`, and was rejected. Build real blocks from `output/*.vul`,
`chem_jac_analytical_per_layer`, and `I/(gamma*dt)`.

**Retrieval workloads change which optimizations matter.** Under forward-mode
`jvp` the block-Thomas factorization is about 85% of the linear-algebra cost,
against about 50% on the primal path, because `lu_factor`'s tangent rule is
expensive. Benchmark at the retrieval species count under `jvp` when throughput
there is the target, and treat a primal-only measurement as not answering the
question.


---

## Parity & bug guide

The VULCAN-JAX <-> VULCAN-master parity & bug guide.

This is the single source of truth for where VULCAN-JAX intentionally diverges
from upstream VULCAN, and for confirmed bugs in VULCAN-master. It is also the
"bug guide" the standing rules point at.

### Policy (LLMs: read this first)

1. **The goal is parity with VULCAN-master.** VULCAN-JAX must reproduce
   VULCAN-master's science. Match master by default. Any intentional divergence
   must (a) fix a **confirmed, results-affecting** defect and (b) be recorded in
   this guide with `file:line` on both sides. If a behavior is not listed here, do
   not "improve" it -- reproduce master.

2. **VULCAN-master is the oracle; do not edit or refactor it.** If you find a real
   bug in VULCAN-master, do **not** patch master. Document it here: under
   "Corrected in the JAX port" if the JAX port fixes it (also allowlist any
   data-file divergence in `tools/audit_master_parity.py`), or under "Bugs still
   present" if the port inherits it. Give `file:line` and the measured or
   estimated impact.

2b. **"What does upstream do?" is answered by FETCHING upstream, never by reading
   `../VULCAN-master/`.** That directory is an unversioned copy (`git rev-parse`
   fails in it) and it is **contaminated**: verified 2026-07-30, it contains
   VULCAN-JAX's own stall detector (`store.py:204-209`, `op.py:1078-1104`), its
   `conv_stall_window` knob (`vulcan_cfg.py:141`,
   `cfg_examples/vulcan_cfg_HD189.py:122`), its `wall_clock_max`/`end_case=4`
   exit (`op.py:1119-1126`), and a 13-species `conver_ignore`
   (`vulcan_cfg.py:183`) that exists in no upstream repository. Citing it as
   upstream is how a wrong config change shipped on 2026-07-26 (`eebc8a5`) and
   2026-07-30 (`2fdc66b`). Fetch instead:
   `raw.githubusercontent.com/exoclime/VULCAN/master/<path>` for VULCAN 2, and
   `shami-EEG/VULCAN` `vm_branch` for VULCAN 3 features.
   `tools/audit_master_parity.py` now refuses to run against a checkout carrying
   VULCAN-JAX-only identifiers, and `tests/test_default_master_parity.py` skips
   (loudly, not passes) rather than reporting a circular parity verdict.
   The copy is still useful as a *numerical* oracle for per-step kernel
   comparisons, which do not depend on the contaminated convergence/config code.

3. **Only real, results-affecting bugs belong here.** A "real bug" changes a
   number a user relies on, crashes, or silently corrupts a result on a path that
   can actually run. Do **not** log or report comment typos, stale docstrings,
   dead code, style, defensible approximations, or issues on paths no shipped
   config selects. The test is: *would this change a result someone trusts?* If
   no, drop it silently. Keeping this guide short is deliberate -- it is meant to
   be read end to end without wading through trivia.

Conventions: locations are `file:line`. "master" = the workspace
`../VULCAN-master` validation oracle **for per-step numerical comparisons only**
(see Policy 2b — it is not a provenance source for any config or convergence
question). "fetched master" / "fetched vm_branch" mean the files pulled from
`raw.githubusercontent.com` on the date given. The JAX port was ported from
`shami-EEG/VULCAN vm_branch @ 362cfa2`; a few entries note where that branch and
the workspace oracle differ. None of the items below affect the default
(gas-only, HD189) validated results unless stated.

### Corrected in the JAX port

Deliberate divergences that fix a confirmed master bug. All are live in the
code today: each one is why a VULCAN-JAX file differs from master right now.

#### C1 — CH2CN + H + M -> CH3CN + M low-pressure rate (data typo)
- **master:** the 10.2025 typo fix (correct `k0 = 1.00E-29`, `k_inf = 1.00E-10`)
  was applied to master's **NCHO** network (the default) but not to its sulfur
  files. On the workspace oracle, `thermo/SNCHO_photo_network.txt:520` and
  `thermo/SNCHO_photo_network_C3.txt:535` ship `k0 = 1.00E-20`, and
  `thermo/SNCHO_DMS_photo_network_Tsai2024.txt:544` ships k0/k_inf fully swapped
  (`1.00E-10` / `1.00E-29`). With a too-large `k0` the association never falls
  off (pinned at `k_inf`), wrong by up to ~1e7x at the model top; trace nitrile
  channel, small spectral effect.
- **JAX:** all three sulfur files ship the corrected values
  (`SNCHO_photo_network.txt:520`; the C3 and DMS files fixed 2026-08-02). All
  three allowlisted in `tools/audit_master_parity.py`
  (`KNOWN_THERMO_DIVERGENCES`). No shipped config selects the C3 or DMS variant
  (`W39b.yaml` uses the base file), so no published number moves.
- **Upstream status (fetched 2026-08-02):** shami-EEG `vm_branch` ships the base
  SNCHO fix; exoclime master does not (base + C3 still `1.00E-20`, DMS still
  swapped; the C3 and DMS files do not exist on vm_branch). Reported to the
  maintainer by email 2026-08-02.
- **Note:** master's DEFAULT config uses NCHO (already `1e-29`), so default master
  runs are unaffected; this divergence only appears on sulfur runs.

#### C2 — S2 / S8 condensate molecular masses (copy-paste error)
- **master:** `op.py:1282` `45.019/Navo` (S2) and `op.py:1328` `360.152/Navo`
  (S8). Correct: 64.12 and 256.48 g/mol (2x/8x atomic S = 32.06,
  `all_compose.txt:126,129`). `45.019` is ~the HCS mass and `360.152 = 8x45.019`.
- **Effect:** the condensation rate scales with this mass, so master biased the
  S2 rate 0.702x and S8 1.404x (sulfur-cloud runs only).
- **JAX:** `src/vulcan_jax/conden.py::GAS_MASS_G_PER_MOL` S2 -> 64.12, S8 -> 256.48
  (S4 was already correct).

#### C3 — H2O saturation vapour pressure = 0 at exactly 273 K
- **master:** `build_atm.py:844-874` (`sp_sat`) writes `(T<0)*ice + (T>0)*water`
  (T in Celsius); at 273.0 K both masks are False, so the value is exactly 0 (an
  artificial cold trap; neighbours are ~6111/6112 dyne/cm^2).
- **JAX:** `src/vulcan_jax/atm_setup.py::sat_p_jax` uses one
  `jnp.where(T_C < 0, ice, liquid)` -- identical except at the buggy point.

#### C4 — sflux-epseri.txt surface-flux normalization (R_star multiplied, not divided)
- **master:** `atm/make_spectra_in_nm.py:25` converts the observed-at-Earth
  HST eps Eri UV spectrum (`obs_spectra/h_epseri_uvsum_spc.txt`) to stellar
  surface flux with `flux = F_earth*10.*(10.475*63241*au/r_sun*0.735)**2` --
  by operator precedence R_star = 0.735 R_sun MULTIPLIES where the conversion
  `F_surface = F_earth*(d/R_star)^2` divides. The shipped
  `atm/stellar_flux/sflux-epseri.txt` is therefore low by exactly
  `R_star^4 = 0.735^4 = 0.291843` (factor 3.426499 too small). The sibling
  builder `atm/stellar_flux/read_muscles_spectra_in_nm.py:38` uses the correct
  `(d/(r_sun*0.2064))**2` form for GJ1214, so this is a one-off
  parenthesization slip, verbatim in upstream exoclime/VULCAN (found by the
  2026-07-21 jwst-tool science audit, S2-01).
- **JAX:** `src/vulcan_jax/atm/stellar_flux/sflux-epseri.txt` rebuilt from the
  raw HST file with the corrected normalization; construction otherwise
  IDENTICAL to master's builder (positive-only filter, ERROR/DQ columns
  ignored, 3-decimal nm wavelengths, 2-sig-fig flux, 115.000-282.999 nm span,
  20 duplicate wavelengths retained). Every flux value is the master value
  times 3.4265 (2-sig-fig rounding); wavelength column byte-identical.
  Verified in `tools/audit_master_parity.py` (`KNOWN_SFLUX_RESCALES` +
  `_known_sflux_rescale_only`: wavelengths must match, every ratio must sit at
  the documented factor).
- **Impact:** only consumers selecting the eps Eri spectrum (jwst-tool's
  WASP-107 b default; no shipped VULCAN-JAX config uses it). Deliberately NOT
  fixed here: the 115-283 nm coverage (bands outside the file span are omitted
  from the photolysis grid by the standard clamp, master-identical) and the
  positive-only/DQ-blind construction -- measured photolysis-integral
  sensitivity of those choices is 2-6% for H2O/CH4/H2S/SO2/HCN (HO2 ~2x on
  the signed variant), small against the 3.43x normalization. jwst-tool side:
  decision record S2-01 in that repo's `notes.md`,
  cache bust via `forward._VERSION` 22 (the UV file is cache-keyed by name,
  not content).

#### C6 — H2S saturation-pressure unit conversion (mm Hg constant for a cm Hg formula)
- **upstream:** `build_atm.py:857` `saturate_p * 0.001333 * 1.e6` under upstream's own
  comment "from Giauque and Blue(1936) in cmHg" — the mm Hg constant for a cm Hg
  formula, 10x low. Same bug in shami-EEG vm_branch (`build_atm.py:920`).
- **JAX:** `src/vulcan_jax/atm_setup.py:943` `sat_p * 0.01333 * 1e6`. Anchor: at the
  H2S boiling point (212.8 K) the formula gives 76.1 cmHg -> 1.015 bar ~ 1 atm with
  0.01333; 0.1 atm with the upstream constant. **Workspace master is patched too**
  (the paper's comparison copy carries this fix), unlike C1-C4.
- Verified against upstream HEAD 2026-07-21. Previously recorded only in README.

#### C7 — NH3 ice molecular weight (NH2's mass)
- **upstream:** `thermo/all_compose.txt:167` `NH3_l_s ... 16.023` — exactly NH2's
  mass (line 40), a copy-paste of the row above. Correct: 17.031. Same in vm_branch.
- **JAX:** vendored `all_compose.txt:167` -> 17.031. **Workspace master patched too.**
- Impact channel is mean molecular weight + molecular-diffusion mass only (both
  codes hardcode ~17 g/mol in the NH3 condensation RATE), and only when NH3_l_s is
  nonzero — real but tiny. Verified against upstream HEAD 2026-07-21; previously
  recorded only in README.

#### C5 — Duplicated CH2_1 entry in the FastChem NASA-9 logK data
- **master:** `fastchem_vulcan/input/nasa9_logK_SNCHOTi.dat`,
  `nasa9_logK_SNCHOTi_ion.dat`, `nasa9_logK_SNCHOPTi.dat` each list the
  `CH2_1 : H 2 C 1` (singlet methylene) entry TWICE, with byte-identical
  coefficient lines. Still present on the workspace oracle.
- **JAX:** the vendored copies under `src/vulcan_jax/fastchem_vulcan/input/`
  keep one entry. Because the duplicate coefficients are identical, no
  measured benchmark impact (W39b V2-vs-V3 parity is 1.1e-9 median with the
  dedup in place on the JAX side only); recorded as a data-file divergence.
  Mechanism (verified 2026-07-21): FastChem's `init_add_species.cpp:132` has no
  duplicate check, so the second entry does enter the element-conservation sums;
  downstream, `build_atm.py`'s `np.genfromtxt(names=True)` renames the second
  output column `CH2_1_1` and VULCAN reads the first — trace-level at most.
  **CLOSED UPSTREAM 2026-07-28, re-verified 2026-07-31 against a fresh
  `exoclime/VULCAN` clone.** Commit `8970337` deletes the second `CH2_1` block
  from all three files; `git show 8970337^:<file> | grep -c '^CH2_1'` returns 2
  and `git show 8970337:<file>` returns 1 for each. Upstream HEAD and the
  vendored JAX copies now agree at one entry apiece, so this is no longer a
  divergence — only the unversioned workspace `../VULCAN-master/` copy still
  carries the duplicate (2 entries in SNCHOTi.dat).
- **Note (same diff, separate):** the vendored `nasa9_logK_SNCHOTi.dat` also
  adds an SiO2 entry absent from master's copy; and
  `element_abundances_vulcan.dat` differs because it is a per-run scratch
  file rewritten by the EQ initialization, not a correction.

#### C12 — FastChem element row order vs hard-coded C++ slots (P/S swap) (logged 2026-07-29)
- **Mechanism:** FastChem builds its `elements` vector in **abundance-file row
  order** — `init_read_files.cpp:204` calls `addAtom(symbol)` once per line and
  `init_add_species.cpp:55-57` does `elements.push_back` +
  `index = elements.size()-1`. But `mass_action_constant.cpp:380-399` subtracts the
  per-element NASA-9 reference polynomial by **hard-coded slot index** under its own
  comment "in the order of the element_abundances.dat file": `index_P = 5`,
  `index_S = 6`. Abundance *values* are set by symbol
  (`init_read_files.cpp:202 -> setElementAbundance`) and are therefore always right;
  what the row order controls is which reference polynomial each stoichiometric
  coefficient is multiplied by.
- **upstream:** `fastchem_vulcan/input/solar_element_abundances.dat` lists
  `... O, S, P, Si ...` — S at slot 5, P at slot 6, the exact transpose of the C++.
  So every S-bearing molecule's `log_K` is built with `log_P` and every P-bearing
  one with `log_S`. Introduced by shami-EEG `604ca6e` (2025-12-17, "PHO network
  running"), whose diff is literally a two-row `P`/`S` swap in that file. Verified
  present in **exoclime/VULCAN HEAD** as of 2026-07-29, not just the shami fork.
- **Reachable path:** `ini_mix='EQ'` only (FastChem sets the initial mixing ratios;
  it is not on the kinetics path). Live on `W39b.yaml`, which is
  `ini_mix: EQ` on an SNCHO network. `build_atm.py:82-129` rewrites the file
  **preserving its row order**, so the config layer cannot mask or fix this.
- **JAX:** `src/vulcan_jax/fastchem_vulcan/input/solar_element_abundances.dat:13-`
  ships the canonical `C,H,He,N,O,P,S,...` order the C++ expects (fixed in
  `45cc4c7`, 2026-06-08, alongside the port-introduced C/H/He regression).
  Guarded two ways so a future reorder fails loudly rather than silently:
  `runtime_validation.py:50,126` (`_FASTCHEM_ELEMENT_ORDER`, a row-ORDER check, not
  values-only) and `tests/test_fastchem_element_order.py` (pins the C++ indices, the
  validator constant, the shipped files, plus an end-to-end "CO forms" probe).
  `ini_abun.py:192-199` documents that `fc_list` order is inert by contrast.
- **Measured impact (2026-07-29, direct two-way FastChem run, identical binary and
  element->value map, row order the only difference).** The error lands almost
  entirely on whichever species is *not* the dominant reservoir, because element
  conservation pins the reservoir and the polynomial error is absorbed by the
  minor one:
  - **On the real W39b evening column** (nz=150, W39b.yaml abundances, 7.6 bar /
    2246 K to 4e-6 bar / 726 K): every molecular species is **unaffected** —
    SO2, H2S, SO, S2, COS, CS all <= 0.004 dex, H2O/CO/CH4/CO2/NH3/HCN at 0.000 dex.
    **Atomic S is wrong by 2.6-3.1 dex in all 150 layers** (median 2.58 dex across
    the 1e-4..1e-2 bar transit slab).
  - **Only where atomic S is itself a major reservoir** (hot *and* rarefied,
    T >~ 2000 K at P <~ 1e-2 bar — off the W39b profile) does it reach the
    molecules: up to 0.85 dex on SO2/H2S, 1.7 dex on S2, 3.4 dex on S4.
  - Because this is an **initial condition** and VULCAN integrates to a
    photochemical steady state — and atomic S has a very short chemical timescale —
    the converged W39b science is not expected to move. **Not quantified:** the
    effect of the wrong initial atomic-S on the convergence path/step count.
- **Upstream is not fixed and now believes it is.** shami-EEG `8970337`
  (2026-07-28) reordered `fc_list` to `C,N,O,P,S,...` with the comment "need to be
  exactly the same order as element_abundances_vulcan.dat". `fc_list` is used only
  for membership (`sp in fc_list`, `build_atm.py:113`), so that edit is **inert**,
  and it does not touch the abundance file that actually sets the slots (membership
  test at upstream `build_atm.py:114`). O1 remains open upstream. If reporting this,
  point at the .dat row order, not `fc_list`.
- **Workspace-oracle note:** `../VULCAN-master`'s copy of this file **has been
  patched** to the corrected order, so `tools/audit_master_parity.py`'s
  byte-identity check on the abundance file passes against a *patched* oracle. A
  fresh upstream clone will trip that check; the drift message is expected and this
  entry is the explanation. See "Workspace-oracle patches" below.

#### C13 — Earth example lists argon, which no network contains (logged 2026-07-30)
- **master:** `cfg_examples/vulcan_cfg_Earth.py:6` puts `'Ar'` in `atom_list` and
  line 39 puts `'Ar':9.34e-3` in `const_mix`, but the network that same file
  selects (`thermo/SNCHO_full_photo_network.txt`, 99 species) has no `Ar`
  species. `build_atm.py:200` does
  `y_ini[:,species.index(sp)] = gas_tot*const_mix[sp]` over every `const_mix`
  key, so the run dies on `ValueError: 'Ar' is not in list` during setup. The
  shipped upstream Earth example therefore does not run at all.
- **JAX:** VULCAN-JAX no longer ships an Earth config at all (removed
  2026-07-30: it converged in neither code, see `validation.md`). While it did,
  it dropped `Ar` from both `atom_list` and `const_mix` and changed nothing else.
  `runtime_validation.py` still rejects any config whose `const_mix` names a
  non-network species, with this finding as the message, so a user writing their
  own Earth case is told upfront rather than crashing in setup.
- **Effect:** the `const_mix` total falls from 0.98974 to 0.98040. Neither sums
  to 1 — upstream leaves the remainder unassigned either way — so this shifts
  the initial condition by 0.93% of the column and lowers the initial mean
  molecular weight slightly, argon being heavier than the N2 it sat beside.
  Argon is chemically inert and absent from the network, so it could not have
  participated in any reaction; the loss is in the initial normalization only,
  and Earth integrates to a photochemical steady state from there.
- **Note:** the alternative (fold 9.34e-3 into N2) preserves the total but
  misstates the composition, so the deficit was kept visible instead.

#### C14 — photolysis reaction index read from the file's id column (logged 2026-07-30)
- **master:** the id column of a network file is *output*, not input.
  `make_chem_funs.py:71-72` rewrites every reaction line as
  `'{:<4d} {:s}'.format(i, ...)` — the comment says "updating the numerical
  index in the network (1, 3, ...)" — and line 109
  (`with open(vulcan_cfg.network, 'w+') as f: f.write(new_network)`) writes the
  file back **in place**. Only after that does `op.py:245` do
  `pho_rate_index[(columns[0], int(columns[1]))] = Rindx[i]`, reading back the
  number it just wrote. So in master's workflow the id column always equals the
  parser position, and reading either is the same thing.
- **JAX:** VULCAN-JAX deliberately does not rewrite vendored network files, so
  the id column is whatever the file shipped with. It is stale in **6 of the 18
  vendored networks** — a file fetched from a remote or hand-edited but never run
  through upstream has never been renumbered. `network.py:288` already indexed by
  `parser_i` (position) and `rates.apply_remove_list` masks positionally, but
  `legacy_io.py` had been ported verbatim from `op.py:245` and used the id
  column, so the two parsers disagreed on exactly those files.
- **Effect:** `op_jax.compute_J` writes `var.k_arr[ridx, :]`, and `k_arr` is a
  dense `[nr+1, nz]` array. Three networks raised `IndexError` (e.g.
  `NCHO_photo_network_lowT.txt`: 16 photolysis rows past `nr = 674`), and three
  more — `CHO_photo_network.txt` (36 rows), `NCHO_full_photo_network.txt` (57),
  `TiSNCHO_photo_network.txt` (65) — had every photolysis id in range but wrong,
  so each J landed on an unrelated reaction with no error at all. Master's
  `var.k` is a dict, so the same stale file there silently drops or misassigns
  the rate instead of raising.
- **Fix:** `legacy_io.py` now indexes `pho_rate_index` / `ion_rate_index` by the
  parser position, matching `network.py`, and `_warn_stale_reaction_ids` raises a
  `RuntimeWarning` naming the file when its ids disagree — because a
  `cfg.remove_list` written by reading ids off an un-renumbered file selects the
  wrong reactions in **both** codes. Pinned by
  `tests/test_network_reaction_ids.py`.
- **No shipped result moves:** every network selected by a shipped config
  (`NCHO_photo_network.txt`, `SNCHO_photo_network.txt`,
  `SNCHO_photo_network_2025.txt`, `SNCHO_full_photo_network.txt`) is renumbered,
  so position and id already agreed there. A test pins that too.

### Live constraints left behind by fixed port regressions

The regressions themselves are fixed and are no longer described here. What
survives is the invariant each fix left behind.

#### C11 — `batch_max_retries` must be 110, not 64
All five shipped configs cite this anchor; keep it. A withdrawn K2-18b config
omitted the key and silently inherited the code default of 64, which is exactly
the failure this anchor exists to prevent, so declare it in every new config. `lax.while_loop` cannot retry
unboundedly, so the runner carries `batch_max_retries` as a deadlock backstop and
force-accepts on `dt_underflow | retry_exhausted` (`outer_loop.py:1097-1103`).
Master's give-up condition is a dt floor with no retry count (`op.py:2549-2560`).
Walking `dt` from `dt_max=1e17` to `dt_min=1e-14` at `dt_var_min=0.5` takes 103
rejects, so any cap below that makes the COUNT fire first and force-accepts a step
master would still be halving. 110 keeps `dt_underflow` the operative trigger.
**If `dt_max`, `dt_min` or `dt_var_min` change, redo this arithmetic and update all
five configs.**

#### Known gap — no NaN-specific termination on the single-profile path
`_conv_jax` (`outer_loop.py:918`) forces `longdy=+inf` on any non-finite `y`/`ymix`,
so a poisoned state can never be scored as converged. But `_real_terminate`
(`:957`) and `cond_fn` (`:1004`) carry no `isfinite` test of their own, so a
single-profile NaN run exhausts its budget and reports budget exhaustion rather
than a NaN reason. Only the batched path sets `termination_reason=5`
(`:1690-1693`).

### Bugs still present in the JAX port (inherited from master)

Confirmed master bugs the port faithfully carries. None affect the default
(gas-only, HD189) validated results -- they bite the convergence machinery or the
non-default condensation/photochemistry paths.

#### Unweighted atom-conservation diagnostic
- **Where:** master `op.py:2551-2569` (`ODESolver.loss`), `build_atm.py:287-292`;
  JAX `src/vulcan_jax/outer_loop.py:285-290` (`_compute_atom_loss`).
- **What:** "atom loss" is `sum over layers of compo*y` with **no `dz` weighting**,
  so on a non-uniform grid it can read exactly-conservative vertical redistribution
  as loss. It feeds step acceptance and adaptive rtol.
- **Status:** deferred. A hot-path convergence heuristic; a `dz`-weighted column
  inventory diverges from master and needs a re-baseline. JAX matches master to
  ~0.02 dex and validates conservation via the reservoir projection
  (`test_atom_conservation_projection.py`), so it is not producing wrong science
  on tested cases.

#### Two-stream particular-solution pole
- **Where:** master `op.py` `compute_flux` (`ll = -w0/(1/mu^2 - (1-w0)/edd^2)`);
  JAX `src/vulcan_jax/photo.py:197`.
- **What:** `ll` is singular at `w0 = 1 - edd^2/cos^2(sl_angle)`. With `edd=0.5`
  the pole lies inside [0,1] for `sl_angle <= 60 deg`: **reachable** at the 48 deg
  configs (HD189 / HD209 / default; pole at `w0 ~ 0.44`) and Earth (58 deg,
  `w0 ~ 0.11`); **not reachable** at the 83 deg W39b config (pole at `w0 = -15.8`).
  Near the pole `g_p`/`g_m` spike and can flip sign, corrupting the DIFFUSE actinic
  flux (the direct Beer's-law beam is unaffected). master's `ll` clip is dead code
  (applied AFTER `g_p`/`g_m`, and `ll` is never read again); JAX has no `ll` clip
  (it guards `chi` instead). Neither regularizes the blow-up.
- **Impact:** only the scattered-flux correction, only for `w0` within a thin band
  of the pole. Quantified over the converged HD189/HD209 runs (150 layers x 2588
  wavelength bins, `jax_paper/scripts/two_stream_pole_margin.py`): `w0` spans [0,1]
  and does reach the pole (closest cell `w0 = 0.44161`; `min|1/mu^2-(1-w0)/edd^2|
  ~ 1e-4`), with ~0.1-0.5% of layer/wavelength cells in a thin band of it, where
  `|ll| = w0/margin` spikes to ~5200 (HD189) / ~3580 (HD209) vs O(1) elsewhere.
  Because it touches only the diffuse actinic flux and both codes carry it
  identically, the gas-only parity comparison is unbiased; the effect on the
  absolute converged abundances remains unquantified (no evidence it moves the
  validated results). Not discussed in the paper (small, shared, inherited).
- **Status:** DEFERRED, and a fix is NOT a drop-in. The correct fix is the analytic
  resonant limit (or a stable boundary-value solve) and needs its own RT
  validation. A naive `|denominator|` floor or `g_p`/`g_m` clip changes the
  near-pole physics and would break parity with master there, so do this as a
  deliberate, validated task -- not a reflexive clip. (master's own `min/max` clip
  on `ll` does not even help: near the pole `|ll|` is large but well below `1e10`,
  so the clip never fires on the damaging values.)

#### Condensate handling in the solver (active-condensation only)
Three inherited condensation-path issues; all no-op when `non_gas_sp` is empty
(the default, `use_condense=False`), so they only matter for cloud runs. The
sibling JWST-tool and retrieval both refuse condensation on their gradient paths,
so these do not reach any inference result.
- **Condensate mass in the gas mean molecular weight** — master `op.py:2544-2546`,
  `build_atm.py:522` (`mean_mass` sums every species; the gas-only `exc_conden`
  sum at `build_atm.py:267` is used for normalization but NOT for `mu`); JAX
  `atm_setup.py:363`. Condensate mass can pollute the gas `mu` (scale height) when
  condensation is active. Fix: restrict the `mu` sum to `gas_indx`. This is the one
  physically-wrong item of the three.
- **Condensates excluded from the convergence metric** — master `op.py:1046-1063`;
  JAX `outer_loop.py:649` (`condense_zero_conv_mask` zeroes `non_gas_sp` columns in
  `longdy`). A cloud column can be declared steady while condensates still evolve.
  (Defensible: pinned-to-saturation species are set, not relaxing, so a `longdy`
  residual on them is not a steady-state signal -- documented as a known trade-off,
  not clearly a bug.)
- **Condensates excluded from the local error norm** — master `op.py:3023-3037`;
  JAX `outer_loop.py:721-751` (`non_gas_present` gas-only error denominator). The
  adaptive step can accept a step whose largest error is in a settling condensate.

#### `atm_type='table'` stale `pico`
- **Where:** master `vulcan.py:118->120->148`, `build_atm.py:406` (pco rewrite),
  `:530/554/562` (stale-pico integration); JAX production setup path.
- **What:** in `table` mode setup runs `f_pico` (pico from the logspace `P_b..P_t`
  grid) BEFORE `load_TPK` overwrites `pco` from the table, and never recomputes
  `pico`. So `f_mu_dz` integrates `dz`/`dzi`/`pref_indx` from a stale `pico`:
  g/Hp ~1% off, `dzi` up to ~12% off when the table grid differs from logspace.
  Masked when the grids match, which is the common restart case.
- **Not live on any shipped config** — all six use `atm_type: file`. The on-graph
  `atm_jax.build_atm_static` (the retrieval path) recomputes `pico`
  self-consistently, so it does not carry this.
- **Status:** deferred, deliberately. Keep production matching master for parity;
  do NOT silently fix VULCAN-JAX alone. The real fix is one line upstream. Revisit
  if upstream fixes the `f_pico`/`pco` ordering.

#### Off-path data typo
- `src/vulcan_jax/thermo/SNCHO_photo_network_C3.txt` still carries the C1 CH2CN
  `1.00E-20` typo. No shipped config or sibling repo selects this variant, so it
  changes nothing today; fix it if C3 chemistry is ever activated.

### Master-only, already better in the JAX port

Not a JAX bug; recorded so no one "fixes" JAX to match master's weaker behavior.

#### Optical-depth vs single-scattering-albedo opacity inconsistency
- **master:** `compute_tau` sums absorption over `photo_sp ∪ ion_sp` with
  T-dependent cross sections; `compute_flux`'s `w0` uses only `photo_sp` with the
  T-INDEPENDENT table (`op.py` ~2621-2672). So the optical depth and the single-
  scattering albedo are built from different opacities.
- **JAX:** `compute_tau_jax` and `compute_flux_jax` build absorption from the SAME
  `PhotoData` arrays (`absp_idx` + `absp_T_idx` + `scat_idx`), so `tau` and `w0`
  are self-consistent (`src/vulcan_jax/photo.py:106-184`). Minor (`ion_sp` empty by
  default; only the scattered flux is affected), but JAX is the correct one -- do
  not "restore parity" by reintroducing the inconsistency.

### The oracle must be a clean upstream clone

Oracle tests resolve upstream through `$VULCAN_MASTER_DIR` and check the exact
commit pinned in `tests/science_sources.yaml` plus a clean worktree. There is
deliberately no sibling-directory fallback, and each test runs against a
disposable copy so upstream setup code cannot rewrite the checkout.

A hand-patched local copy is not an oracle. The one on this project's machine
has corrections C6, C7 and C12 written into it, so any comparison against it is
circular; that is how a wrong step count was once published. Details: `notes.md`.

### Upstream defects NOT inherited by VULCAN-JAX

Recorded so they are not re-investigated and so they can be reported upstream.
Each is confirmed present in exoclime/VULCAN HEAD as of 2026-07-29; none is a
VULCAN-JAX bug, so none needs a fix here.

| Upstream defect | Location (upstream) | Why JAX is clear |
|---|---|---|
| FastChem uses fixed shared I/O filenames, and `rm`s its own output; two concurrent runs in one checkout clobber each other | `build_atm.py:129,139,154,170` | per-run isolation + `fcntl.flock` (`ini_abun`), `$VULCAN_JAX_FASTCHEM_DIR` per-worker trees |
| Shipped Earth example sets `const_mix` `Ar` and `atom_list` `Ar`, but `SNCHO_full_photo_network.txt` has no Ar species, so the example raises on init | `cfg_examples/vulcan_cfg_Earth.py:6,39` | `Earth.yaml` does not request Ar |
| Flux-convergence test calls `np.nanmax` on an empty selection for a fully dark/filtered column | `op.py:2737` | explicitly guarded: `outer_loop.py:538` `jnp.where(jnp.any(mask), max, 0.0)`; `op_jax.py:79-88` has the `else: 0.0` branch |
| `S4` has a saturation-pressure branch but is absent from `sat_sp_list`, so selecting S4 condensation raises before the implemented code runs | `build_atm.py:788` vs `:833` | `S4` present in both `atm_setup.py:884` and `conden.py:51,68,77` |
| Kinetic NH3 condensation looks for reaction label `NH3 -> NH3_l`; the low-T network defines `NH3 -> NH3_l_s`, so the rate never attaches | `op.py:1151` vs `thermo/NCHO_photo_network_lowT_Jupiter.txt:441` | JAX matches on `NH3_l_s` |
| `check_conserv` runs `np.genfromtxt(dtype=None)` then `str(sp)`; under NumPy < 2 byte-string behavior names become `b'OH'` and the post-codegen conservation check raises, leaving an unchecked generated kernel | `make_chem_funs.py:719-725` | JAX uses content-hashed codegen + a fail-fast network guard; also does not reproduce under NumPy 2.3.5 |

### Provenance

Upstream-side items above were cross-checked against a collaborator audit,
`~/Desktop/VULCAN_original_code_error_audit.md` (2026-07-29), which also lists 16
historical upstream defects that upstream has already fixed. Those are upstream
history, not parity items, and are deliberately not restated here.

Verified independently 2026-07-29: the C12 mechanism and its `604ca6e` introduction;
C1/C2/C3/C6/C7 still open at exoclime/VULCAN HEAD; every row of the not-inherited
table. One correction to that audit: it lists the dark-column `nanmax` crash as
"inherited in JAX" — it is not, both JAX paths guard the empty selection.

### Scope / verification

- C1/C2/C3 verified against the fetched `vm_branch` source and the workspace
  oracle; they keep `tests/test_conden_profile_builder.py` green and H2O
  saturation continuous through 273 K. The still-present items were verified to
  exist in the JAX code.
- Upstream defects that JAX does not inherit now have their own section above
  ("Upstream defects NOT inherited by VULCAN-JAX") — dark-column `nanmax`, FastChem
  I/O concurrency, `NH3 -> NH3_l`, `S4` saturation, the Earth `Ar` example, and the
  `check_conserv` byte-string failure. Also deliberately NOT logged: the stale
  generated-kernel / `-n` regeneration workflow (JAX uses content-hashed codegen +
  a fail-fast network guard), the atomic-P / pressure column collision (JAX reads
  `fc["P_1"]`), the dense ~1 GiB Jacobian (JAX bands it), plus approximations,
  data-file duplicates, and documentation/packaging items.
- **Initial elemental composition differs from upstream, and it dominates any
  cross-code comparison (recorded 2026-07-30).** Two separate, deliberate
  divergences live in the same file,
  `fastchem_vulcan/input/solar_element_abundances.dat`:
  (a) **Lodders 2019 (Wogan & Tsai 2023) values instead of upstream's Lodders
  2009.** For the C-H-N-O networks the only changed value is helium,
  `He 10.9864` (upstream) -> `10.9232` (JAX), i.e. He/H lower by 0.063 dex.
  Sulfur also moves, `7.12` -> `7.1492`. C, H, N and O are identical.
  (b) rocky suppression to `-3.0` (the bullet below).
  **Measured consequence, HD 189733 b, both codes converged, everything else
  matched (same network, same cfg to 1 UI flag, same TP/Kzz/stellar files):**
  median relative difference across all species above a 1e-12 floor is
  **2.0e-01**, and it is present in the deep well-mixed column, not just the
  photochemical upper atmosphere: inert He differs by a uniform 11.6%, H2O by
  25%, CO2 by 42% below 1 bar. For scale, upstream compared against *itself*
  (same code, one reaction changed, different step counts) agrees to
  **7e-06** median. So the composition file, not the solver, is what separates
  the two codes on a free-convergence run.
  **This is an initial-condition choice, not a defect** — the rocky suppression
  is deliberate for the truncated networks, and the Lodders update is a data
  refresh. But it means *"VULCAN-JAX vs VULCAN 2.0"* numbers are only meaningful
  if this file is matched first. Any parity figure or step-count comparison must
  say which composition it used.
- Not a bug, do not "fix": FastChem retains rocky elements (Mg/Si/Fe/…) at solar
  in its equilibrium (`build_atm.py` solar and customized branches both), so a few
  percent of O can sit in untracked gas species. This is a modeling choice, not a
  defect, and it is identical in both config branches. VULCAN-JAX ships a rocky-
  suppressed abundance file, which is a legitimate initial-condition difference for
  the truncated NCHO/SNCHO networks. (There is no `use_other_ele` flag in either
  codebase.)

## Differentiability

This section states what VULCAN-JAX can differentiate, how to do it, and how
accurate the result is.

The second part is the project-wide **condensation differentiation contract**
(F1-F5), which also governs vulcan-retrieval and vulcan-jwst-tool. The shelved
Route B decision records are summarized in the untracked dev log (`notes.md`,
Route B record).

### The rule

A quantity is differentiable **when it reaches the runtime as a JAX array**. That
happens in one of two ways:

1. You supply it directly into the runtime pytrees (`AtmStatic`, `RateInputs`,
   the initial `y`, and most of `PhotoStaticInputs`).
2. An **on-graph builder** produces it. `rates_jax` builds `T -> k`.
   `atm_jax.build_atm_static` builds the whole atmosphere structure (`pco`,
   `Tco`, gravity, composition to `M`, `dz`, `Hp`, `Dzz`, `vm`, `vs`).

A scalar parameter that a host-side setup formula expands into those arrays
becomes differentiable once that formula is on the graph. After
`build_atm_static`, this covers the atmosphere cascade.

Drive the inner `integ._runner`, not `OuterLoop.__call__`. The public call copies
state to the host to write `.vul` output, which breaks tracing.

### Condensation is not differentiable through

With `use_condense=True` the converged state comes from a finite condensation
window plus a `fix_species` pin that snapshots the condensate reservoir at a
transient moment. That snapshot is not a smooth steady state. The pinned-species
forward-mode tangent disagrees with re-converged finite differences at order
unity (about 0.91 relative). The set of actively condensing layers and the NH3
cold-trap level also switch discretely with temperature.

The low-level kernels (`conden.sat_p_jax`, `conden.build_conden_profile`) stay
differentiable. The completed pinned model does not:

- `steady_state_input_sensitivity` refuses a condensation state.
- `steady_state_reaction_sensitivity` returns only a ranking that is conditional
  on the frozen reservoir.
- There is no supported Fisher or retrieval-inference path through condensation.

The full scope and rationale are in the condensation differentiation contract,
which follows at the end of this section.

### What you can differentiate (forward mode, end to end)

| Physical input | How |
|---|---|
| Reaction rates `k`, forward and reverse | Supply `k_arr`. For a reaction ranking use `steady_state_reaction_sensitivity` |
| Temperature `T` (per-layer array) | `atm_jax.build_atm_static` rebuilds `M`, `dz`, `Hp`, `Dzz`, `vm`, `vs` from `Tco`. Also rebuild `k(T)` with `rates_jax.build_rate_array` |
| Surface gravity, planet radius `Rp` | `build_atm_static`. `gs` is resolved as `G*Mp/Rp^2` by `atm_setup.surface_gravity` and enters the graph as the resolved leaf. There is no `gs` knob |
| Pressure grid (`P_b`, `P_t`) | `atm_jax.pco_from_endpoints(P_b, P_t, nz)` gives the `pco` leaf, which reaches `M`, `Dzz`, and `dz` |
| Molecular and thermal diffusion `Dzz`, `vm`, `vs` | `build_atm_static` carries the Moses `T -> Dzz` fit, `vm`, and the Cloutman settling formulae on the graph |
| Arrhenius coefficients, NASA-9 thermodynamic data | `rates_jax.build_rate_array(..., rate_coeffs={"a": ...})` and `nasa9_coeffs`. One hardcoded Troe row is excepted |
| Eddy diffusion `Kzz`, advection `vz` | `atm._replace(Kzz=...)`, or `atm_setup.kzz_profile_jax` for the deep and maximum `Kzz` |
| Boundary fluxes, deposition velocity | Supply `top_flux`, `bot_flux`, `bot_vdep` |
| Initial abundances `y0` | Perturb `y0` directly |
| Metallicity `[M/H]`, C/O ratio | A `y0` tangent (see below) |

#### Metallicity and C/O are y0 tangents

A converged closed column forgets the initial speciation. The steady state
depends on the conserved elemental totals, not on how those atoms were first
distributed. So the correct metallicity derivative scales the metal-bearing
initial abundances, and the correct C/O derivative scales C-bearing species
against O-bearing ones.

```python
import jax, jax.numpy as jnp
from vulcan_jax import composition

# compo_array column 0 is H in the default atom_list order, so [:, 1:] is metals.
metal = jnp.asarray((composition.compo_array[:ni, 1:].sum(1) > 0).astype(float))

def run_from_y0(y0):
    final = integ._runner(state0._replace(y=y0), atm)
    return final.y / final.y.sum(1, keepdims=True)

_, dlnVMR_dlnZ = jax.jvp(run_from_y0, (y0,), (y0 * metal[None, :],))
```

This is the derivative behind the published `d ln SO2 / d ln Z = 2.6` result for
WASP-39b.

#### Building the differentiable atmosphere

```python
phys, spec = atm_jax.make_physical_inputs(cfg, var, atm, species_list)
atm_static = atm_jax.build_atm_static(phys._replace(Tco=new_T), spec)
```

`build_atm_static` reproduces the production `make_atm_static` field for field to
machine precision for the default configuration, which is `atm_type` `file`,
`analytical`, or `isothermal` with `use_moldiff` on. That is what the runner
uses. See `examples/grad_physical_example.py`.

Two non-default modes differ, and in both cases `build_atm_static` is the more
self-consistent of the two. With `atm_type='table'` it recomputes the interface
pressures from the rewritten grid, where production keeps a stale `pico`. With
`use_moldiff` off it computes `Ti` and `Hpi` as interface averages, where
production leaves them at legacy defaults; that difference is inert at runtime.

#### Condensation follows a live temperature profile

`conden.make_conden_spec` extracts the temperature-independent metadata once per
config on the host. `conden.build_conden_profile(spec, Tco, pco, n_0, Dzz)` then
rebuilds every temperature- and structure-dependent condensation array on the
graph: saturation number densities, the growth and diffusion `Dg` terms, the H2O
and NH3 relaxation inputs, the NH3 cold-trap index, and the fix-species
saturation mixing ratios.

The builder is jit-, vmap-, and jvp-compatible, and the runner already reads
these arrays from the `ProfileVars` carry every step.
`OuterLoop._build_conden_static` delegates to the same builder, so host setup and
on-graph rebuild share one implementation.

The cold-trap index is an `argmin`, so it is an integer with no tangent. A
temperature tangent moves the saturation curves smoothly, but the active-layer
set and the cold-trap index change layer by layer. Forward-mode derivatives are
therefore valid only away from those switches, the same caveat as any phase
boundary.

### What you cannot differentiate yet

| Blocked input | Why | What to do instead |
|---|---|---|
| `d/d T_irr` through the Heng et al. (2014) profile | `analytical_TP_H14` is on the graph, but forward mode through `jax.scipy.special.expn` is very slow over a deep column | Differentiate the per-layer `Tco` leaf, or use a cheaper `T(P)` parameterization |
| Stellar flux scale and spectrum | `sflux_top` and the room-temperature cross sections `cross_J` and `absp_cross` are closure-baked into `outer_loop._make_photo_branch`, not read from a runtime pytree | Not exposed. This needs a runner-level input, not a pytree field |
| The cross-section temperature rebake | `photo_setup._bin_T_dependent` re-interpolates cross sections per layer on the host at setup | The temperature-dependent cross sections do ride the carry (`s.pv.p_cross_J_T`, `p_absp_T_cross`), so they are differentiable as arrays. The static cross sections and the `T`-to-cross-section map are not |

FastChem is a hard wall because it is a subprocess: the scalar map from `[M/H]`
to the equilibrium speciation at `t=0` is not differentiable. This rarely
matters, because a converged closed column forgets the initial speciation and the
`y0` tangents above are the scientifically correct derivatives.

The `const_lowT` Newton residual (`ini_abun._abun_lowT_residual`) is
differentiable with respect to the elemental ratios `O_H`, `C_H`, `He_H`, and
`N_H` for the reduced H2/H2O/CH4/He/NH3 system. The shipped `ini_abun` entry
point reads them as Python floats, so call the solver directly with JAX arrays to
get that gradient.

Host-side file readers (`photo_setup.py`, `composition.py`, and the CSV loaders
in `atm_setup.py`) are not differentiable by design. Build the corresponding
pytree directly with JAX arrays instead.

### Forward mode

`lax.while_loop` supports `jvp`, so one forward pass differentiates the whole
converged integration.

```python
import jax
from vulcan_jax.jax_step import make_atm_static

state0 = integ._pack_state_from_runstate(rs)
atm    = make_atm_static(data_atm, ni, nz, cfg=integ._cfg)

def run(Kzz):
    final = integ._runner(state0, atm._replace(Kzz=Kzz))
    return final.y / final.y.sum(axis=1, keepdims=True)

ymix, dymix = jax.jvp(run, (atm.Kzz,), (atm.Kzz,))
```

This is validated end to end on a full HD 189733b production run with
photochemistry on and about 1300 accepted steps. The `jvp` tangent matches
re-converged centered finite differences to better than 0.1% on the responding
levels, with correlation above 0.9999. The route never inverts `df/dy`, so it
stays well posed where the reverse-mode adjoint does not. See
`examples/grad_jvp_example.py`.

**Temperature gradients need the rate rebuild.** The runner's `k_arr` is frozen
at setup by the host-side NumPy `rates.build_rate_array`, so a `d/dT` jvp must
rebuild it on the graph with `rates_jax.build_rate_array`, which is bit-exact to
about 5e-14 against the NumPy build. `atm_jax.build_atm_static` rebuilds `M`,
`dz`, `Hp`, and `Dzz(T)`, so those are no longer frozen. Only the host-side
cross-section temperature interpolation stays frozen, and that is second order.
Forward-mode `d/dT` is validated against finite differences: HD 189733b dominant
species to 3-4 significant figures, and WASP-39b SO2 to correlation 1.0.

### Reverse mode: the steady-state adjoint

Reverse mode answers the many-inputs, one-output question: which reactions set
the converged abundance of a species. One adjoint solve returns
`dL/d ln k_r` for every reaction, where finite differences would need one
re-converged model each.

```python
import jax.numpy as jnp
from vulcan_jax import composition
from vulcan_jax.steady_state_grad import steady_state_reaction_sensitivity

def loss(y):                       # log10 SO2 mixing ratio at its peak layer L
    return jnp.log10(y[L, so2] / y[L].sum())

dL_dlnk = steady_state_reaction_sensitivity(   # shape (nr+1,)
    loss, y_star, k_arr, atm, net,
    compo_array=composition.compo_array[:ni], dz=dz,
    integ=integ, converged_state=final_state,
)
```

#### How it works

`lax.while_loop` blocks `vjp`, so this is the steady-state adjoint of the body
map, not backpropagation through the loop. At convergence `G(y*) = y*`, and
`(I - dG/dy)^T z = v` is solved with the integrator's own regularized step as the
operator, in log-abundance coordinates, with the conserved-mass null space
deflated.

The solve uses LGMRES. That choice is measured, not incidental: restarted GMRES
oscillates on this operator and a raw Neumann iteration diverges, because the
operator is indefinite and singular. Earlier attempts that took the adjoint of
the bare residual `df/dy` all failed. On a closed column that residual is both
singular, from mass conservation, and severely ill-conditioned, from stiff
chemistry. That is why the solver-map route exists.

#### Accuracy

- **Default `solver_map="renorm"` reaches percent level.** HD 189733b CH4 lands
  about 0.7% from finite differences, because `y*` is a roughly 1e-9 fixed point
  of the renormalized map the loop actually iterates. HD 209458b forward rows
  improve from 35% to 1%. A 2026-07-03 campaign across HD 189733b, HD 209458b,
  and WASP-39b, checked against both re-converged finite differences and
  forward-mode `jvp` (which agree to 0.02%), showed the residual error is a
  linearized-map effect. It is not a convergence-criterion mismatch: finite
  differences are invariant across `yconv` from 1e-2 to 1e-4, and stricter
  convergence, a `body_dt` scan, and a larger LGMRES budget do not move it.
- **With photochemistry on, supply the runner context.** The default
  `photo_recompute_k="auto"` reaches percent level when given
  `runner_photo_static=integ._photo_static` and `converged_state=final`, or
  `integ=integ` and `converged_state=final`. It rebuilds `J(y)` through the
  runner's two-stream radiative transfer on each application, so the operator
  carries `dJ/dy`. Without it, frozen photolysis leaves those rows at leading
  order, about 11% off. With the default renorm map, WASP-39b SO2 dominant rows
  reach 0.2% and 0.1%. The cost is one radiative-transfer solve per Krylov
  matrix-vector product. Pass `photo_recompute_k=None` only to reproduce the
  older frozen-photolysis result.
- **Legacy `solver_map="bare"` reaches a few percent.** It linearizes the raw
  Ros2 step, for which `y*` is only a roughly 1e-4 fixed point: HD 189733b CH4
  about 6.6%, WASP-39b OH and H2 about 11%. It is kept only to reproduce
  pre-2026-07 behavior.
- **One case stays hard.** HD 209458b CH2OH near-equilibrium reverse rows remain
  ill-conditioned, with an LGMRES residual near 0.1 even with the renorm map.
  The default diagnostics flag them. Treat those as a ranking, and use forward
  mode where an exact value is needed.

#### The `body_dt` probe step

`body_dt` sets the solver regime. It is an adjoint-only knob; the forward model
is untouched. The default `body_dt=1e7` sits in the measured low-residual regime:
on HD 189733b the residual is 0.04-0.15 and the twin ensemble lands 0.3-6% from
finite differences, with a mean of 3.5%.

At `body_dt` of 1e8 or more, which was the old default, the solve stagnates. The
residual rises to 0.2-0.7, because the body map has unstable top-layer H and H2
eigenmodes and the matrix-vector product's floating-point floor grows with `dt`;
single-solve magnitudes then bounce by about 25%. At `body_dt` near 3e6 the solve
converges fully but deterministically underweights slow chemistry, biasing the
result by about 28%.

The safe window is column-dependent. Scan a few values and keep the lowest
`info["resid"]`; the map is recorded in the comment on `BODY_MAP_DT`.

The gradient is returned as the mean over an `n_solves` twin ensemble, three by
default, using deterministic seeded right-hand-side perturbations. The twin
spread in `info["ensemble_spread"]` is the honest error bar on the magnitude. The
**ranking** is robust in every non-divergent regime: dominant reactions stand one
to two orders of magnitude above the noise with stable signs.

#### Cross-regime validation (2026-07-02)

WASP-39b with the SNCHO network, photochemistry on, 1150 reactions, and an SO2
loss is an easy regime. Residuals are 0.005-0.05 at every `body_dt` from 3e6 to
1e8, answers are `dt`-insensitive to better than 1%, twin spread is about 6e-4,
and the ranking reproduces the paper exactly.

On HD 189733b three loss regimes degrade, and the default-on diagnostics flag all
three:

- **Buffered species** (H2O and CO in the mid column). The spread warns on the
  twin-noisy tail, but the insensitivity conclusion itself is robust.
- **Upper-atmosphere losses.** True stagnation; the median residual warns.
- **Losses coupled to the unstable top-layer H and H2 modes.** Residuals are
  tiny, but `ensemble_spread` is about 0.9 and `info["pair_antisym"]` is about 1.
  That forward/reverse pair-antisymmetry check catches internal inconsistency
  that residuals miss.

Mid-column composition losses, which are the design use case, are safe because
their cotangent is orthogonal to the unstable subspace.

#### Physical-input gradients through the same solve

`steady_state_input_sensitivity(loss, y_star, k_arr, atm, net, p0, rebuild, ...)`
returns `dL/dp` for an arbitrary input pytree, for example a full `(nz,)`
temperature profile, in one adjoint solve plus one VJP. It needs a
differentiable `rebuild(p) -> (k(p), atm(p))`, which `rates_jax` plus `_replace`
provides; non-thermal rows are spliced in frozen.

The rebuild is consistency-checked at `p0` and warns or refuses on a mismatch.
The renormalization is differentiated through `atm(p).M`, because temperature
moves the rebalance. The function returns the chemistry path
`dL/dy* * dy*/dp`; a spectrum loss's direct `dL/dp` term, such as temperature in
the radiative transfer, is added separately by the caller. Forward-mode `jvp`
remains the exact route for a handful of directions.

#### What the body map contains

The body map is `ros2_step`, plus renormalization, plus photochemistry, plus
optional `body_terms`. Build the last with
`make_body_terms(integ, converged_state, atm_static)`, which also returns the
correctly spliced `atm`, including a live `vm` when `use_vm_mol` is on.

`body_terms` carries the per-step processes that a non-default config turns on:

- The in-window **condensation** composite. Condensation and evaporation rate
  rows are recomputed from `y`, giving the `dk/dy` feedback that is analogous to
  the photolysis `dJ/dy`, plus the H2O and NH3 relaxation kernels and the
  gas-only partial rebalance.
- The **`fix_species`** pins, for species clamped inside the Ros2 step.
- The **layer-0 boundary pins** (`use_fix_all_bot`, `use_fix_sp_bot`, and a
  tripped hycean H2-He).

Everything else stays outside the linearization: clipping is the identity almost
everywhere, ion charge balance is unsupported and raises, and the escape-flux
recompute and the composition-to-mu atmosphere refresh are frozen and second
order.

**A fingerprint guard raises rather than return a silently wrong gradient.** A
state converged with condensation active is refused without matching terms,
active ion rows are always refused, and frozen photolysis warns.

Before trusting the gradient on any non-default config, run:

```python
audit_adjoint_scope(y_star, k_arr, atm, net, cfg=..., final_state=...,
                    loss_fn=..., body_terms=...)
```

It classifies every dropped process for that config as error, warning, or info,
confirms the converged geometry was spliced into `atm`, and measures the
**per-cell** fixed-point defect `|G(y*) - y*|/y*` of the exact map the solver
uses. That per-cell measurement matters: a pinned bottom trace row can be 100%
off while the global max-norm `fp_err` reads 1e-9. It also reports the defect
inside the loss's own footprint.

#### Diagnostics

Diagnostics are on by default. Warnings fire on a poor LGMRES residual, a loose
fixed point, or a large twin-ensemble spread. An LGMRES breakdown or a
rank-deficient deflation basis raises instead of returning garbage.

`info["null_quality"]` reports how null the deflated conserved-mass directions
actually are, relative to the operator's scale. It is about 3e-5 on a healthy
closed HD 189733b column. The atom-count vectors are only approximately null,
because the diffusion discretization is not exactly conservative under the `dz`
weights. A value of order unity means conservation is broken, for example by open
boundary fluxes.

The solve itself is host-side SciPy LGMRES, because JAX has no LGMRES. It runs
once after convergence, off the hot path, and warm-start cycles stop early once
SciPy reports convergence.

See `examples/grad_reverse_example.py` and
`tests/test_steady_state_reaction_sensitivity.py`.


---

### Condensation contract: project-wide report and scope decision (F1-F5)

Date:    2026-07-15
Status:  Implemented (F1-F5 landed with guard tests).
Scope:   VULCAN-JAX, vulcan-retrieval, vulcan-jwst-tool.
Related: notes.md, "Route B record" (the shelved open-system smooth-rainout
         plan + B0A decision record); steady_state_grad.py (the first-order
         adjoint machinery).

#### 1. The question, and the short answer

Goal: make condensation "usable for differentiation" across all three repos.

The answer depends entirely on what "usable" means, and the two meanings have
opposite cost:

- **(A) Honest, bounded differentiation.** Forward-mode `jvp` works on a fixed
  smooth branch; the reverse-mode reaction adjoint gives a conditional
  (frozen-reservoir) ranking; everything that is not reliably differentiable
  hard-errors with a clear explanation. This is the **simpler fix**: about 40 to
  60 lines of guards, labels, and one bypass close, plus tests, spread over the
  three repos. It touches no solver code and no physics. **It does not need
  Route B.**

- **(B) Trustworthy total derivatives through condensation.** Reliable
  `d(spectrum)/dT`, `d(SO2)/d(rate)` including the reservoir-capture history, i.e.
  what a Fisher matrix or gradient-MALA actually consumes. This requires
  replacing the pin with a smooth open-system steady state. **That is Route B**,
  and it reached a measured no-go.

**Recommendation: adopt (A), hard-error (B).** Do not resurrect Route B unless
the science specifically requires open-system rainout physics and someone owns
the flux-closure problem that failed. The (B) cases are not a "simpler fix" away;
they are ill-posed with the current pin, which is exactly why Route B had to
change the physics to attempt them.

#### 2. Why the pin is not differentiable (root cause)

The upstream `master_pin` methodology (the only condensation path on `main`) is:
run a condensation window, snapshot the condensable reservoir at the first
accepted step after `stop_conden_time`, then pin those abundances with
`fix_species` for the rest of the integration. Three independent obstructions
follow, and they are separate problems:

1. **Transient snapshot / path sensitivity.** The snapshot rides the adaptive
   accepted-step sequence. A small parameter change shifts that sequence, so the
   perturbed run captures a slightly different drainage state. Forward-mode
   differentiates the branch the unperturbed run took; finite differences
   reconverge a different branch. Measured disagreement for the pinned S8 /
   S8_l_s tangents: relative error about 0.91, i.e. the tangent is roughly
   91% wrong -- an order-unity failure, NOT a 0.91 agreement ratio and NOT a
   9% mismatch (`tests/test_condensation_live_tp.py`).

2. **Phase-boundary nonsmoothness.** `max(0, y - y_sat)` switches condensation on
   and off; the set of condensing layers changes discretely; the NH3 cold trap
   uses an integer `argmin` that carries no tangent. Away from these switches the
   smooth formulas are fine; at them the derivative is undefined.

3. **Closed column vs open physics.** The pin conserves sulfur by freezing it;
   real rainout removes it. Neither is the derivative of a smooth physical
   steady state, because the pinned state is not one.

The low-level kernels (`conden.sat_p_jax`, `conden.build_conden_profile`) are
genuinely differentiable and rebuild every saturation quantity from the live
temperature. The problem is never the vapor-pressure formula; it is the
completed, pinned solution.

#### 3. Consumer inventory: what each needs, and whether the simpler fix suffices

| # | Consumer | Repo | What it needs | Simpler fix (A) suffices? | Route B needed? |
|---|---|---|---|---|---|
| 1 | Forward model, condensation on, no AD | VULCAN-JAX / retrieval synth / jwst forward | Config hardening only | Yes | No |
| 2 | Forward-mode `jvp` on a fixed smooth branch (d comp / d ln Kzz, away from switches) | VULCAN-JAX | Works today; a "validate your column" caveat | Yes | No |
| 3 | Reverse-mode reaction ranking `dSO2/d ln k`, conden on | VULCAN-JAX / paper | A conditional-on-frozen-reservoir label | Yes, as a conditional derivative | Only for the total (history-inclusive) derivative |
| 4 | Input sensitivity `dL/dT`, `dL/dKzz`, conden on | VULCAN-JAX | Hard error | Yes (the fix is to refuse) | Route B is the only path that would deliver it; it failed |
| 5 | Retrieval gradient-MALA inference, conden on | vulcan-retrieval | Refuse (resolved-config gate) | Yes (refuse) | Route B (failed) |
| 6 | JWST Fisher, conden on | vulcan-jwst-tool | Already refused | Yes (done) | Route B (failed) |
| 7 | Hessian, condensation off | VULCAN-JAX / paper | Wire the implicit-root recipe into production | Independent of condensation | No |
| 8 | Hessian, through condensation | VULCAN-JAX | C2 smoothing | No fix | More than Route B (its sink is C1) |

Read across the table: the simpler fix makes every consumer either work (2, 3),
correctly refuse (4, 5, 6), or become config-hardened (1). The only capabilities
Route B would add are the total-derivative versions of rows 3 to 6, and those are
exactly what its B0C feasibility gate no-go'd.

#### 4. The required fixes and guards (the simpler-fix work items)

Most of the guard architecture already exists. What is verified present today:
core validation of the `condense_sp` support tier and `fix_species`/`use_condense`
consistency (`runtime_validation.py:373,403-419`); the reaction adjoint hard-errors
on a condensation state passed without body terms (`steady_state_grad.py:804-814`);
`make_body_terms` packs both condensation regimes correctly
(`steady_state_grad.py:1561-1585`); `audit_adjoint_scope` emits error/warning/info
findings and sets `ok=False` on any error (`steady_state_grad.py:1685-1854,2073`);
the retrieval refuses conden inference behind `allow_condense_inference`
(`config_schema.py:494-504`); the retrieval forward wrapper validates
`use_moldiff` / empty / `use_sat_surfaceH2O` / inert `condense_sp`
(`vulcan_chem.py:212-251`); and the jwst-tool hard-gates `use_condense` before
Fisher (`forward.py:240-252`, `app.py:271-276`).

The remaining delta is five items:

**F1 (VULCAN-JAX). Input-sensitivity guard: fix the keyed field, and hard error.**
`steady_state_input_sensitivity` warns "leading-order only" only when
`conden_static is not None` (`steady_state_grad.py:1328-1336`), i.e. the in-window
regime. The regime a real converged condensing run ends in is post-pin, where
`make_body_terms` sets `conden_static=None` and `fix_mask=<pins>`
(`1568-1585`); there the warning never fires and `_guard_unmodeled_processes`
passes it (`terms_pins=True` satisfies the guard at `804`). So `dL/dT` through a
pinned condensation column returns silently, missing both `d(sat)/dT` and the
reservoir-capture path. Change: hard-error on any active condensation
(`conden_static is not None` or `fix_mask is not None` or the `*_l_s`
fingerprint), with the explanation. Recommended default: raise, with an explicit
`allow_frozen_condensation_input_grad=True` escape hatch for a knowing user.
~15 lines.

**F2 (VULCAN-JAX). Reaction adjoint: label the conditional case.** With
`body_terms.fix_mask` set, the body map holds the reservoir at `fix_y`
(`steady_state_grad.py:477-479`), so the result is `dL/d ln k` at fixed captured
reservoir: a valid partial derivative, not the total. It proceeds silently today.
Set `info["conditional_on_fixed_reservoir"]=True` /
`info["includes_condensation_history"]=False` plus a one-shot warning. This is the
most defensible condensation-AD case (rates do not move the saturation curve
directly), so label rather than forbid; an opt-in `allow_conditional_fixed_reservoir`
is optional. ~10 lines.

**F3 (VULCAN-JAX). Core forward hardening.** Add to the existing `if use_condense:`
block in `validate_runtime_config` (`runtime_validation.py:403`): `use_moldiff=False`
raises (confirmed universal: `Dzz` is zeroed at `atm_setup.py:696` and
`atm_jax.py:248`, so the growth term `Dg=0` and nothing condenses silently); empty
`condense_sp` raises; `stop_conden_time < start_conden_time` raises (not checked
anywhere today). Do not lift the `use_sat_surfaceH2O` refusal into core: that
constraint is specific to the retrieval's live-`T(P)` rebuild; the standalone
forward model legitimately supports it. ~10 lines.

**F4 (VULCAN-JAX). Doc cleanup.** Trim the stale `conden_mode` / `smooth_rainout`
bullet from `CLAUDE.md:154`. That Route B code is not on `main` (zero occurrences
in `src/`), so the contract should not describe it. ~1 line.

**F5 (vulcan-retrieval). Resolved-config inference gate.** The gate keys on
`cfg.cfg_overrides.get("use_condense")` (`config_schema.py:494`), but the resolver
loads a base config first (`load_config(vulcan_cfg_name)`, `vulcan_chem.py:156`) and
`Earth.yaml` defaults `use_condense: true` (`Earth.yaml:105`). A case pointing at
such a base without restating the flag in overrides sails past the gate. Keep the
fast early gate and add an authoritative one on the resolved signal right after
`chem = build_chem_model(...)` (`retrieval_forward.py:55`): if
`chem.conden_spec is not None and cfg.run_inference and not cfg.allow_condense_inference:
raise`. `conden_spec` is the resolved truth (`vulcan_chem.py:651`). ~4 lines.

**Tests.** One unit test per guard's raise path, plus a "condensation contract"
test that pins the whole policy: a forward run with condensation works; input
sensitivity raises; the reaction adjoint sets the conditional flags; and the
retrieval inference gate refuses via the `Earth.yaml`-base bypass.

Total production change is roughly 40 lines plus tests, all in the moderate-to-small
band. The library changes (F1 to F4) carry the most leverage because the two
sibling repos inherit them.

#### 5. The resulting contract (what "usable" then means)

| Operation | Condensation policy after F1-F5 |
|---|---|
| Forward VULCAN run, condensation on | Supported (config-hardened) |
| JIT / vmap of forward runs | Supported |
| Low-level smooth kernels (`sat_p_jax`, `build_conden_profile`) | Differentiable |
| Forward-mode `jvp` on a fixed smooth branch | Supported; validate your column |
| Reaction adjoint after the pin | Conditional on the frozen reservoir (labeled) |
| Input adjoint (`dL/dT`, `dL/dKzz`), condensation active | Hard error |
| Retrieval / MALA inference, condensation on | Hard error |
| JWST Fisher, condensation on | Hard error |
| Hessian, condensation active | Hard error (and no production Hessian entry point exists) |

That is an honest, complete contract: condensation works as a forward model, its
smooth components stay composable in JAX, and every unreliable full-model
derivative fails loudly instead of returning a plausible but wrong number.

#### 6. The Hessian (separate and independent)

There is no production Hessian entry point today (only the paper demo). The recipe
exists and is validated in `jax_paper/scripts/hessian_demo/hessian_lib.py`:
`hessian = jacfwd(jacfwd(f))` (forward-over-forward, which the runner's
`lax.while_loop` supports because both orders are `jvp`), plus an `implicit_root`
wrapper (`lax.custom_root`, implicit-function theorem) that does second-order
implicit differentiation through a fixed point, checked against FD in
`_selfcheck_implicit`.

- **Off condensation:** to make the Hessian easy in production, wire that recipe
  into a `steady_state_hessian` (a `custom_root`-wrapped runner reusing the
  adjoint's log-scale and null-space deflation at second order). Moderate,
  self-contained, and entirely independent of the condensation work. It is cheap
  only for the low-dimensional Hessian the science wants (a few T-P / Z / C-O
  directions for Laplace evidence or Fisher curvature), not for the full
  1150-reaction space.
- **Through condensation:** the first derivative is already piecewise and
  path-sensitive, so its derivative is undefined at the switches. Even the shelved
  Route B sink is C1 (a "one-sided C1 hinge"), and its deep-boundary lookup is C0
  as built (trilinear in ln x, notes.md Route B record, B0A item 3). A
  meaningful condensation Hessian needs a C2 hinge and a C1 boundary, which is
  strictly more than Route B attempted.

F1 transitively hard-errors any condensation Hessian, since it would build on the
input-sensitivity gradient.

#### 7. Route B and the alternatives considered

Route B (open-system rainout plus an imposed deep sulfur reservoir) is not
"condensation made differentiable": it replaces the physical model, because the
mass-conserving pin cannot be linearized. About 1250 lines across 15 files.
Status: the B0C feasibility gate reached a **no-go** (flux-closure residual
~26.4%, the reference column exhausted its step budget). Shelved to branch
`research/smooth-rainout-fisher`, tag `smooth-rainout-b0c-no-go-2026-07-14`, in
both `jax-vulcan` and `vulcan-retrieval`. Fisher through condensation stays
disabled.

No cheap middle path delivers reliable total derivatives through condensation.
The three considered and rejected (criterion-gated pin, differentiate only the
frozen branch, smooth surrogate) and the full sequencing record are in
`notes.md`, "Condensation contract".

## Support

Open a [GitHub issue](https://github.com/imalsky/jax-vulcan/issues) for a bug or
question. Include the configuration, command, software versions, final
termination message, and full error message.

## License and citation

VULCAN-JAX uses the [GNU General Public License v3.0](LICENSE).

If you use VULCAN-JAX in published work, cite the VULCAN papers:

- Tsai, S.-M., Lyons, J. R., Grosheintz, L., Rimmer, P. B., Kitzmann, D., and
  Heng, K. 2017, *ApJS*, 228, 20
- Tsai, S.-M., et al. 2021, *ApJ*, 923, 264
