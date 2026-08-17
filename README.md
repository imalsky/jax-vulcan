# VULCAN-JAX

VULCAN-JAX is a JAX port of [VULCAN](https://github.com/exoclime/VULCAN)
(Tsai et al. 2017, 2021), a one-dimensional photochemical-kinetics model for
planetary atmospheres. It reads the same network, atmosphere, and
configuration inputs as VULCAN and writes the same `.vul` output, with a
JIT-compiled Rosenbrock integrator that runs on CPU or GPU, forward-mode
differentiation through the full integration loop, reverse-mode reaction
sensitivities at a converged state, batched runs with `jax.vmap`,
photochemistry, transport, condensation, and ion chemistry. Chemistry agrees
with VULCAN at machine precision on the ported kernels; vendored inputs are
revision-pinned in `tests/science_sources.yaml`.

## Install

Requires Python 3.10+, JAX (CI pins 0.6.2), NumPy, SciPy, PyYAML, and a C++
compiler for FastChem. Everything runs in float64.

```bash
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vulcan-jax
```

For development (an editable install is required to run the tests):

```bash
git clone https://github.com/imalsky/jax-vulcan.git
cd jax-vulcan
python -m pip install -e ".[dev,plot]"
```

## Quick start

```bash
vulcan-jax --config default
vulcan-jax --config W39b
```

Results land in `output/` as a `.vul` file plus the resolved configuration.
Configuration is YAML only (`src/vulcan_jax/configs/`); a CWD
`./configs/<name>.yaml` overrides the shipped one. The reaction network is
fixed per process; select another with `$VULCAN_JAX_NETWORK` before the first
import. `examples/quickstart.ipynb` shows the Python API.

## Tests

```bash
python -m pytest tests -q
```

Master-oracle comparisons need a pinned upstream checkout via
`$VULCAN_MASTER_DIR` and skip cleanly without one.

## Known limits

Only the Rosenbrock-2 solver is ported. Reverse-mode cannot pass through the
integration loop; use forward-mode, or the steady-state adjoint at a
converged state. Condensation runs in the forward model but is not validated
for gradient inference. VULCAN-JAX contains documented corrections and
deliberate differences from VULCAN; review them before a strict parity study.

## Support

Open a [GitHub issue](https://github.com/imalsky/jax-vulcan/issues) with the
configuration, command, versions, and the full error message.

## License and citation

GPLv3. If you use VULCAN-JAX in published work, cite the VULCAN papers:
Tsai, S.-M., et al. 2017, ApJS, 228, 20 and Tsai, S.-M., et al. 2021, ApJ,
923, 264.
