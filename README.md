# VULCAN-JAX

VULCAN-JAX is a JAX port of
[VULCAN](https://github.com/exoclime/VULCAN), a one-dimensional chemical
kinetics model for planetary atmospheres. It supports exoplanet
photochemistry, vertical transport, condensation, and ion chemistry.

VULCAN-JAX uses the same reaction networks, atmosphere files, configuration
names, and `.vul` output format as VULCAN. It runs the integration loop with
JAX on a central processing unit (CPU) or graphics processing unit (GPU).

## Main capabilities

- Just-in-time (JIT) compiled Rosenbrock integration for stiff chemical kinetics
- Forward-mode differentiation through the complete integration
- Reverse-mode reaction sensitivity at a converged state
- Batched atmosphere runs with `jax.vmap`
- Analytical chemistry Jacobians
- Photochemistry and molecular or eddy diffusion
- Optional condensation and ion chemistry
- VULCAN-compatible `.vul` output

This is research software. Check convergence and run the relevant validation
tests before you use a result in a publication.

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

The run writes two files under `output/`:

- `HD189.vul` contains the model result
- `HD189.vul.config.yaml` contains the complete resolved configuration

The first run can be slower. VULCAN-JAX must compile FastChem, generate the
chemistry function for the selected network, and compile the JAX program.
Later runs reuse these files.

Run another configuration:

```bash
vulcan-jax --config HD209
```

The supplied `W39b` configuration uses a sulfur network. Select this network
before Python imports VULCAN-JAX:

```bash
VULCAN_JAX_NETWORK=thermo/SNCHO_photo_network.txt VULCAN_JAX_ATOM_LIST=H,O,C,N,S vulcan-jax --config W39b
```

Do not assume that a stopped run has converged. Check the final message or the
`end_case` value in the output. A value of `1` means that the run converged.
The other values mean that the run hit a limit: `2` the model runtime, `3` the
step count, `4` the wall-clock budget.

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

| Name | Target | Status |
| --- | --- | --- |
| `default` | HD 189733 b baseline | Recommended first run |
| `HD189` | HD 189733 b | Ready |
| `HD209` | HD 209458 b | Ready |
| `W39b` | WASP-39 b | Requires the sulfur-network environment variables |
| `Earth` | Upstream Earth example | Does not run unchanged; see **Known limits** |

Use a file path to run a custom configuration:

```bash
vulcan-jax --config path/to/my_config.yaml
```

For a bare name, the loader first checks `./configs/<name>.yaml`. It then
checks the configurations inside the installed package. Each run saves the
complete resolved configuration next to the `.vul` result, so any run can be
repeated exactly.

### Select a different network

The reaction network, atom list, and composition table are fixed when Python
first imports VULCAN-JAX. Set their environment variables before the first
import:

```bash
VULCAN_JAX_NETWORK=/absolute/path/to/network.txt VULCAN_JAX_ATOM_LIST=H,O,C,N,S VULCAN_JAX_COM_FILE=/absolute/path/to/all_compose.txt vulcan-jax --config path/to/my_config.yaml
```

Restart Python before you change one of these values in a notebook or
interactive session.

## Differentiation

Use forward-mode differentiation when the model has a small number of input
parameters. The complete `lax.while_loop` supports `jax.jvp` and `jax.jacfwd`.
See:

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
- The supplied `Earth` configuration contains argon as a fixed gas, but its
  reaction network does not contain argon. VULCAN has the same problem.
  Remove argon from `const_mix` and `atom_list` before you run this case.
- VULCAN-JAX contains documented corrections and intentional differences from
  VULCAN. Review
  [`docs/corrections_to_original_code.md`](docs/corrections_to_original_code.md)
  before a strict parity study.

## Documentation

| File | Contents |
| --- | --- |
| [`docs/differentiability.md`](docs/differentiability.md) | What is differentiable, how, and to what accuracy |
| [`docs/validation.md`](docs/validation.md) | Agreement with VULCAN, layer by layer |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Where the time goes in a step |
| [`docs/corrections_to_original_code.md`](docs/corrections_to_original_code.md) | Corrections made in the port, and the parity policy |
| [`docs/vulcan_jax_file_organization.md`](docs/vulcan_jax_file_organization.md) | A file-by-file guide to the source tree |

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
