# VULCAN-JAX

VULCAN-JAX is a JAX port of
[VULCAN](https://github.com/exoclime/VULCAN), a one-dimensional chemical
kinetics model for planetary atmospheres. It supports exoplanet
photochemistry, vertical transport, condensation, and ion chemistry.

VULCAN-JAX reads the same reaction network files, atmosphere files, and
configuration names as VULCAN, and writes the same `.vul` output format. It runs
the integration loop with JAX on a CPU or GPU.

The vendored network files come from upstream. One difference is open: the
N-C-H-O network here contains `NH3 + CH -> NH2 + CH2`, which is absent from both
current upstream branches. It has not been changed pending a decision from the
upstream author, because adding or removing a reaction changes the chemistry.

## Main capabilities

- Just-in-time (JIT) compiled Rosenbrock integration for stiff chemical kinetics
- Forward-mode differentiation through the compiled integration loop (host-side
  setup is excluded; see
  [`docs/differentiability.md`](docs/differentiability.md))
- Reverse-mode reaction sensitivity at a converged state
- Batched atmosphere runs with `jax.vmap`
- Analytical chemistry Jacobians
- Photochemistry and molecular or eddy diffusion
- Optional condensation and ion chemistry
- VULCAN-compatible `.vul` output

## Requirements

- Python 3.10 or later
- JAX 0.4.31 or later
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
in [`docs/validation.md`](docs/validation.md), including what blocks each one.

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
before the loop starts. See the differentiability document below.

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

For the full scope and the accuracy measurements, see
[`docs/differentiability.md`](docs/differentiability.md).

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

Some tests require a sibling VULCAN checkout at `../VULCAN-master/`. These
tests skip cleanly when it is not available. Slow tests, including the adjoint
ones, run only when `VULCAN_JAX_RUN_SLOW=1`.

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
| `docs/` | Detailed implementation and validation records |

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
  VULCAN. Review the parity & bug guide in
  [`docs/validation.md`](docs/validation.md) before a strict parity study.

## Documentation

| File | Contents |
| --- | --- |
| [`docs/validation.md`](docs/validation.md) | Agreement with VULCAN layer by layer, benchmarks (where the time goes), and the parity & bug guide |
| [`docs/differentiability.md`](docs/differentiability.md) | What is differentiable, how, and to what accuracy, plus the condensation contract |

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
