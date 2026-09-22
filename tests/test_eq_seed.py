"""The `ini_mix: EQ` Gibbs seed: agreement with upstream, and its own invariants.

Since 0.15.0 the seed is a JAX Gibbs minimizer over the loaded network's own
gas species, using the same NASA-9 polynomials as the reverse rates, in place
of a FastChem subprocess. Two independent things need proving:

1. It is the SAME equilibrium. `test_seed_matches_upstream_fastchem` runs the
   pinned upstream VULCAN's own initializer on the same column with the same
   elemental abundances and compares dex by dex. The two codes share no
   algorithm, no species list and no thermochemical table, so the bar is the
   measured agreement (see MAX_DEX below), not machine precision.

2. It is EXACT where it must be: element ratios, normalisation, excluded
   species, and the jit / vmap / jvp boundary. Those need no oracle.

Both sides run in subprocesses: the network is import-frozen, so the W39b
(SNCHO) case cannot share a process with the NCHO default, and master's
modules must never be imported into a pytest worker.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
from oracle import oracle_dir_or_sentinel

# Oracle location from $VULCAN_MASTER_DIR only, never a sibling guess. The
# parent verifies the pinned revision + clean tree and points the probe at a
# temporary COPY; the per-test is_dir() skip below handles "not configured".
VULCAN_MASTER = oracle_dir_or_sentinel()

from vulcan_jax._paths import PACKAGE_ROOT

# Measured as max |log10(y_jax / y_master)| over cells where master's mixing
# ratio exceeds YMIX_FLOOR, against upstream at the pinned vulcan2_ncho
# revision:
#     HD189 (NCHO)   max 7.86e-03  median 5.84e-05  p99 6.44e-04  (3337 cells)
#     W39b  (SNCHO)  max 7.72e-03  median 9.06e-05  p99 7.89e-04  (4897 cells)
# The bar is 3x the worse of those. Both maxima are HO2, a 1e-12 radical.
# The floor of the disagreement is FastChem's own convergence accuracy
# (1.0e-4 in its parameters.dat, ~4.3e-5 dex) plus two different
# thermochemical tables.
#
# The full-solar Lodders 2009 preset is deliberately NOT a case here: with
# every rocky element at solar, upstream's FastChem locks oxygen into
# MgO/SiO2/FeO, and a Gibbs minimizer over an NCHO network has no species to
# do that with. Measured divergence 0.166 dex on H2O/CH4/CO2 (max 0.370),
# entirely master-side -- the two presets carry identical C/N/O rows, so this
# port's seed barely moves between them. notes.md 2.9.
MAX_DEX = 2.4e-2
YMIX_FLOOR = 1.0e-15
ELEMENT_RTOL = 3.2e-8

# (case id, VULCAN-JAX config, abundance preset)
CASES = [
    ("hd189", "default", "solar_element_abundances.dat"),
    ("w39b", "W39b", "solar_element_abundances.dat"),
]


_JAX_PROBE = r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

root = Path(sys.argv[1])       # PACKAGE_ROOT (.../src/vulcan_jax)
out_npz = Path(sys.argv[2])
config_name = sys.argv[3]
abundance = sys.argv[4]

os.chdir(root)

# The network is import-frozen: set $VULCAN_JAX_* from the config's own keys
# BEFORE importing vulcan_jax so a non-default network is parsed.
raw = yaml.safe_load((root / "configs" / f"{config_name}.yaml").read_text())
os.environ["VULCAN_JAX_NETWORK"] = raw["network"]
os.environ["VULCAN_JAX_ATOM_LIST"] = ",".join(raw["atom_list"])
os.environ["VULCAN_JAX_COM_FILE"] = raw["com_file"]

from vulcan_jax.config import default_config, load_config

cfg = default_config()
for key, value in vars(load_config(config_name)).items():
    setattr(cfg, key, value)
cfg.fastchem_solar_abundance_file = f"thermo/{abundance}"

from vulcan_jax.runtime_validation import validate_runtime_config

validate_runtime_config(cfg, root)

import vulcan_jax.chem_funs as chem_funs
from vulcan_jax.atm_setup import Atm
from vulcan_jax.ini_abun import InitialAbun
from vulcan_jax.state import _AtmData, _Variables

data_var, data_atm = _Variables(), _AtmData()
make_atm = Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
data_var = InitialAbun().ini_y(data_var, data_atm)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    pco=np.asarray(data_atm.pco, dtype=np.float64),
    Tco=np.asarray(data_atm.Tco, dtype=np.float64),
    M=np.asarray(data_atm.M, dtype=np.float64),
    y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
)
"""


_MASTER_PROBE = r"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])       # the DISPOSABLE oracle copy
out_npz = Path(sys.argv[2])
cfg_src = Path(sys.argv[3])

(root / "vulcan_cfg.py").write_text(cfg_src.read_text())
# A failed generator must not be mistaken for success by importing the
# checkout's stale, pre-generated module.
(root / "chem_funs.py").unlink(missing_ok=True)
res = subprocess.run(
    [sys.executable, "make_chem_funs.py"],
    cwd=str(root), capture_output=True, text=True, timeout=900,
)
# make_chem_funs.py writes chem_funs.py BEFORE its post-codegen check_conserv(),
# which raises under numpy>=1.24 (str(numpy.bytes_) -> "b'OH'"). That crash is
# benign to the generated module -- master's own vulcan.py ignores the exit
# code -- so only bail if the module does not import with a valid (ni, nr).
if res.returncode != 0:
    probe = subprocess.run(
        [sys.executable, "-c",
         "import chem_funs as c; assert c.ni > 0 and c.nr > 0"],
        cwd=str(root), capture_output=True, text=True, timeout=300,
    )
    if probe.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        sys.exit(res.returncode)

os.chdir(root)
sys.path.insert(0, str(root))

import build_atm
import chem_funs
import store

data_var, data_atm = store.Variables(), store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
data_var = build_atm.InitialAbun().ini_y(data_var, data_atm)

np.savez_compressed(
    out_npz,
    species=np.array(list(chem_funs.spec_list), dtype=object),
    pco=np.asarray(data_atm.pco, dtype=np.float64),
    Tco=np.asarray(data_atm.Tco, dtype=np.float64),
    M=np.asarray(data_atm.M, dtype=np.float64),
    y_ini=np.asarray(data_var.y_ini, dtype=np.float64),
)
print("MASTER_OK")
"""


_TRACER_PROBE = r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

root = Path(sys.argv[1])       # PACKAGE_ROOT (.../src/vulcan_jax)
os.chdir(root)

from vulcan_jax import ini_abun
from vulcan_jax.atm_setup import Atm
from vulcan_jax.state import _AtmData

make_atm = Atm()
data_atm = make_atm.load_TPK(make_atm.f_pico(_AtmData()))
Tco = jnp.asarray(np.asarray(data_atm.Tco, dtype=np.float64))
p_bar = jnp.asarray(np.asarray(data_atm.pco, dtype=np.float64) / 1.0e6)
b = jnp.asarray(ini_abun._element_vector())

# `_element_vector` needs the element order, so drop the setup again: the
# first seed CALL has to be the traced one, so the lazy setup is built inside
# that trace.
ini_abun._SEED = None
y = np.asarray(jax.jit(ini_abun.eq_seed)(Tco, p_bar, b))
assert np.isfinite(y).all()
fm = ini_abun._SEED[0].formula_matrix
assert not isinstance(fm, jax.core.Tracer), f"formula_matrix is a {type(fm)}"

# Anything later must be able to reuse the cached setup: another trace at
# another nz, and an untraced call. A tracer anywhere in the cached setup
# (the formula matrix, or the NASA-9 coefficients the hvector closure holds)
# raises UnexpectedTracerError here.
k = 40
np.asarray(jax.jit(ini_abun.eq_seed)(Tco[:k], p_bar[:k], b))
np.asarray(ini_abun.eq_seed(Tco, p_bar, b))
print("TRACER_PROBE_OK")
"""


def _run(script: str, *args) -> None:
    """Run a probe subprocess and surface both streams on failure."""
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True, text=True, timeout=1800, check=False,
    )
    assert result.returncode == 0, (
        f"probe exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _master_cfg(master_root: Path, config_name: str, scratch: Path) -> Path:
    """Write the master-side `vulcan_cfg.py` for one case.

    Upstream has no W39b config, so that one is resolved from the shipped
    YAML on top of the oracle's own legacy surface (old modules import
    derived knobs such as `gs` that VULCAN-JAX no longer carries), and the
    checksummed W39b inputs are staged into the disposable copy.
    """
    from vulcan_jax.config import load_config

    if config_name != "W39b":
        return master_root / "cfg_examples" / "vulcan_cfg_HD189.py"

    cfg = load_config("W39b")
    legacy = (master_root / "vulcan_cfg.py").read_text()
    lines = [
        f"{key} = {getattr(cfg, key)!r}"
        for key in sorted(vars(cfg))
        if not key.startswith("_")
    ]
    lines.append(f"gs = {6.67430e-8 * float(cfg.Mp) / float(cfg.Rp) ** 2!r}")
    cfg_path = scratch / "vulcan_cfg_W39b_resolved.py"
    cfg_path.write_text(
        legacy + "\n# resolved W39b overrides\n" + "\n".join(lines) + "\n"
    )
    for rel in (
        "thermo/SNCHO_photo_network.txt",
        "atm/atm_W39b_evening_TP_Kzz.txt",
        "atm/stellar_flux/sflux-W39b_Tsai2023.txt",
    ):
        target = master_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE_ROOT / rel, target)
    return cfg_path


@pytest.mark.master_serial
@pytest.mark.parametrize(
    "config_name, abundance", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_seed_matches_upstream_fastchem(config_name: str, abundance: str) -> None:
    """The Gibbs seed reproduces upstream's FastChem seed to MAX_DEX."""
    from oracle import oracle_worktree

    with tempfile.TemporaryDirectory(prefix="eq_seed_") as tmp, \
            oracle_worktree(
                "vulcan2_ncho",
                # Both sides must start from the SAME composition: an
                # unmatched preset compares two atmospheres, not two codes.
                fastchem_abundance=abundance,
            ) as master_root:
        scratch = Path(tmp)
        jax_npz, master_npz = scratch / "jax.npz", scratch / "master.npz"
        cfg_path = _master_cfg(master_root, config_name, scratch)

        _run(_JAX_PROBE, PACKAGE_ROOT, jax_npz, config_name, abundance)
        _run(_MASTER_PROBE, master_root, master_npz, cfg_path)

        jax = np.load(jax_npz, allow_pickle=True)
        master = np.load(master_npz, allow_pickle=True)

    assert list(jax["species"]) == list(master["species"])
    np.testing.assert_array_equal(jax["pco"], master["pco"])
    np.testing.assert_allclose(jax["Tco"], master["Tco"], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(jax["M"], master["M"], rtol=1e-14, atol=0.0)

    m = np.asarray(master["y_ini"]) / np.asarray(master["M"])[:, None]
    j = np.asarray(jax["y_ini"]) / np.asarray(jax["M"])[:, None]
    sig = m > YMIX_FLOOR
    dex = np.abs(np.log10(np.where(sig, j, 1.0) / np.where(sig, m, 1.0)))
    worst = int(np.argmax(np.where(sig, dex, -1.0)))
    z, i = np.unravel_index(worst, dex.shape)
    assert dex[sig].max() <= MAX_DEX, (
        f"seed vs upstream FastChem: max {dex[sig].max():.3e} dex > "
        f"{MAX_DEX:.3e} on {list(jax['species'])[i]} at layer {z} "
        f"(T={float(jax['Tco'][z]):.1f} K, p={float(jax['pco'][z]) / 1e6:.3e} bar, "
        f"jax={j[z, i]:.6e}, master={m[z, i]:.6e}); median "
        f"{np.median(dex[sig]):.3e} over {int(sig.sum())} cells"
    )


@pytest.mark.parametrize("Tiso", [None, 300.0, 3000.0])
def test_seed_reproduces_its_element_vector_and_normalisation(Tiso) -> None:
    """The seed's own exact invariants, on the nominal column and two corners.

    Element ratios recovered from the output must be the requested `b`, the
    mixing ratios must sum to one, and species the seed excludes must be
    exactly zero. Nothing here depends on an oracle.

    ELEMENT_RTOL is 3x the worst recovery measured over these three columns
    (1.04e-08, the isothermal 300 K one); it is set by `fastchem_newton_tol`,
    whose 1e-12 is the floor this kernel reaches -- 1e-14 does not converge.
    """
    import jax.numpy as jnp

    from vulcan_jax import composition, ini_abun

    Tco, p_bar = _hd189_column()
    if Tiso is not None:
        Tco = np.full_like(Tco, Tiso)
    b = ini_abun._element_vector()
    ymix = np.asarray(
        ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b))
    )

    assert np.isfinite(ymix).all()
    seed_idx = set(ini_abun._seed()[1].tolist())
    excluded = [i for i in range(len(composition.species)) if i not in seed_idx]
    assert np.all(ymix[:, excluded] == 0.0)
    np.testing.assert_allclose(ymix.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)

    counts = np.asarray(composition.compo_array)
    elements = ini_abun.seed_elements()
    cols = [composition.atom_list.index(e) for e in elements]
    got = ymix @ counts[:, cols]                       # (nz, E)
    got = got / got[:, [elements.index("H")]]
    np.testing.assert_allclose(
        got, np.broadcast_to(b / b[0], got.shape), rtol=ELEMENT_RTOL, atol=0.0
    )


def test_ratios_path_matches_the_config_path_and_moves_carbon() -> None:
    """The per-lane `ratios` seed equals the config path, bit for bit.

    Then doubling C/H must move the carbon carriers and leave helium alone:
    the retrieval's cold start drives exactly this path.
    """
    import jax.numpy as jnp

    from vulcan_jax import composition, ini_abun
    from vulcan_jax.config import default_config

    cfg = default_config()
    elements = [a for a in cfg.atom_list if a != "H"]
    ratios = {a: float(getattr(cfg, a + "_H")) for a in elements}
    # use_solar=False makes the config path read the same <X>_H the ratios
    # dict carries; with use_solar=True it would read the preset file instead
    # and the two would agree only to the preset's four-decimal dex.
    cfg.use_solar = False
    try:
        b_cfg = ini_abun._element_vector()
        np.testing.assert_array_equal(ini_abun._element_vector(ratios), b_cfg)
        idx = ini_abun.ratio_indices(elements)
        np.testing.assert_array_equal(
            np.asarray(ini_abun.element_vector(
                jnp.asarray([ratios[a] for a in elements]), idx)),
            b_cfg,
        )
        # No-override case, both branches of `use_solar`: the traced path
        # must read the same vector the host path reads, preset or <X>_H.
        empty = ini_abun.ratio_indices([])
        np.testing.assert_array_equal(
            np.asarray(ini_abun.element_vector(jnp.zeros(0), empty)), b_cfg
        )
        b_c2 = ini_abun._element_vector({**ratios, "C": 2.0 * ratios["C"]})
    finally:
        cfg.use_solar = True

    np.testing.assert_array_equal(
        np.asarray(ini_abun.element_vector(jnp.zeros(0), empty)),
        ini_abun._element_vector(),
    )
    # A partial override under use_solar=True, the retrieval's case: every
    # element it does not name keeps the preset on both paths (before 5657a25
    # the host path took <X>_H for them instead).
    c_only = {"C": 2.0 * ratios["C"]}
    np.testing.assert_array_equal(
        np.asarray(ini_abun.element_vector(
            jnp.asarray(list(c_only.values())), ini_abun.ratio_indices(list(c_only)))),
        ini_abun._element_vector(c_only),
    )

    Tco, p_bar = _hd189_column()
    y_cfg = np.asarray(
        ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b_cfg))
    )
    y_c2 = np.asarray(
        ini_abun.eq_seed(jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b_c2))
    )
    sp = composition.species
    assert np.all(y_c2[:, sp.index("CH4")] > y_cfg[:, sp.index("CH4")])
    np.testing.assert_allclose(
        y_c2[:, sp.index("He")], y_cfg[:, sp.index("He")], rtol=2e-2
    )


def test_solver_controls_force_a_retrace() -> None:
    """A tighter or looser config must not reuse the trace that baked in the
    old controls.

    `eq_seed` reads the tolerance and the iteration cap at TRACE time, and
    JAX keys its trace cache on the traced function plus the argument shapes,
    so a jit wrapper per key is not by itself enough. The observable: after a
    converged run at the shipped 450 iterations, the same column at
    `max_iter = 1` must come back all-NaN (the seed's "did not converge"
    signal), not silently repeat the converged answer.
    """
    import jax.numpy as jnp

    from vulcan_jax import ini_abun
    from vulcan_jax.config import default_config

    Tco, p_bar = _hd189_column()
    b = ini_abun._element_vector()
    args = (jnp.asarray(Tco), jnp.asarray(p_bar), jnp.asarray(b))

    cfg = default_config()
    warm = np.asarray(ini_abun._seed_jit()(*args))
    assert np.isfinite(warm).all(), "the shipped controls must converge"

    keep = cfg.fastchem_newton_max_iter
    cfg.fastchem_newton_max_iter = 1
    try:
        one = np.asarray(ini_abun._seed_jit()(*args))
    finally:
        key = (float(cfg.fastchem_newton_tol), 1)
        cfg.fastchem_newton_max_iter = keep
        ini_abun._SEED_JIT.pop(key, None)
        ini_abun._OPTIONS_CACHE.pop(key, None)

    seeded = ini_abun._seed()[1]
    assert np.isnan(one[:, seeded]).all(), (
        "max_iter=1 reused the converged trace: the config's solver controls "
        "are baked in at trace time and a new key must retrace"
    )


def test_seed_setup_survives_being_built_inside_a_trace() -> None:
    """The lazy setup is cached, so it must hold no tracers.

    vulcan-forward calls `eq_seed` from a jitted forward model, so the first
    call of a process can be a traced one; a JAX array built there escapes
    into every later trace. Runs in a subprocess because the check needs a
    process whose first seed call is the traced one.
    """
    _run(_TRACER_PROBE, PACKAGE_ROOT)


def test_seed_is_batchable_and_carries_a_zero_tangent() -> None:
    """`vmap` over columns equals the solo calls, and `jvp` gives zero.

    The seed runs per retrieval lane under `vmap`, and a `jvp` must never
    reach ExoGibbs's kernel (it has a custom_vjp and no forward rule), so the
    seed declares a zero tangent instead.
    """
    import jax
    import jax.numpy as jnp

    from vulcan_jax import ini_abun

    Tco, p_bar = _hd189_column()
    b = ini_abun._element_vector()
    columns = jnp.stack([jnp.asarray(Tco), jnp.asarray(Tco) * 1.05])
    pressures = jnp.stack([jnp.asarray(p_bar)] * 2)
    elements = jnp.stack([jnp.asarray(b)] * 2)

    batched = np.asarray(jax.vmap(ini_abun.eq_seed)(columns, pressures, elements))
    solo = np.stack([
        np.asarray(ini_abun.eq_seed(columns[k], pressures[k], elements[k]))
        for k in range(2)
    ])
    np.testing.assert_allclose(batched, solo, rtol=1e-12, atol=0.0)

    def total_water(scale):
        y = ini_abun.eq_seed(jnp.asarray(Tco) * scale, jnp.asarray(p_bar),
                             jnp.asarray(b))
        return jnp.sum(y)

    primal, tangent = jax.jvp(total_water, (1.0,), (1.0,))
    assert float(primal) == float(total_water(1.0))
    assert float(tangent) == 0.0


def _hd189_column() -> tuple[np.ndarray, np.ndarray]:
    """The shipped HD189 T-P column as (Tco [K], p [bar])."""
    from vulcan_jax.atm_setup import Atm
    from vulcan_jax.state import _AtmData

    make_atm = Atm()
    data_atm = make_atm.load_TPK(make_atm.f_pico(_AtmData()))
    return (
        np.asarray(data_atm.Tco, dtype=np.float64),
        np.asarray(data_atm.pco, dtype=np.float64) / 1.0e6,
    )
