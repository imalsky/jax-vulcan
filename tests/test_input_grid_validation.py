"""Input-grid contracts: every vendored stellar spectrum loads (upstream accepts
duplicate wavelengths -- sflux-epseri.txt keeps 20, C4), and malformed inputs
fail loud instead of interpolating garbage."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vulcan_jax._paths import PACKAGE_ROOT
from vulcan_jax.atm_setup import read_sflux_binned
from vulcan_jax.photo_setup import _make_bins
from vulcan_jax.state import load_stellar_flux

_SFLUX_FILES = sorted(
    p.name for p in (PACKAGE_ROOT / "atm/stellar_flux").glob("sflux-*.txt")
)


@pytest.mark.parametrize("name", _SFLUX_FILES)
def test_every_vendored_stellar_spectrum_loads(name):
    cfg = SimpleNamespace(
        use_photo=True, sflux_file=f"atm/stellar_flux/{name}", dbin_12trans=240.0
    )
    sf = load_stellar_flux(cfg)
    assert np.all(np.diff(sf.wavelength_nm) >= 0.0) and np.all(np.isfinite(sf.flux))


def _sflux(*rows):
    return np.array(list(rows), dtype=[("lambda", float), ("flux", float)])


_CFG = SimpleNamespace(
    dbin1=1.0, dbin2=2.0, dbin_12trans=4.0, r_star=1.0, orbit_radius=1.0,
    sflux_file="synthetic",
)


@pytest.mark.parametrize(
    "call",
    [
        # transition outside / on the edge of the bin interval
        lambda: _make_bins(1.0, 10.0, 0.1, 1.0, 1.0),
        lambda: _make_bins(1.0, 10.0, 0.1, 1.0, 11.0),
        # decreasing wavelength column
        lambda: read_sflux_binned(
            _CFG, np.array([2.0, 3.0, 4.0, 6.0, 8.0]),
            _sflux((1.0, 1.0), (5.0, 1.0), (4.0, 1.0), (9.0, 1.0)),
        ),
        # grid without the dbin_12trans node
        lambda: read_sflux_binned(
            _CFG, np.array([2.0, 3.0, 5.0, 7.0, 8.0]),
            _sflux((1.0, 1.0), (4.0, 1.0), (9.0, 1.0)),
        ),
    ],
)
def test_malformed_grids_fail_loud(call):
    with pytest.raises(ValueError):
        call()


def test_duplicate_wavelengths_are_accepted():
    bins = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    raw = _sflux((1.0, 1.0), (4.0, 1.0), (4.0, 1.0), (9.0, 1.0))
    out = read_sflux_binned(_CFG, bins, raw)
    assert np.all(np.isfinite(out["sflux_top"]))
