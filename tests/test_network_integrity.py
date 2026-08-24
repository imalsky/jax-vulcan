"""Network-file integrity: no silent duplicates, honest T-range accounting.

The parser is positional and appends every reaction row, so a reaction
duplicated within one section is double-counted in both directions with no
error. The vendored TiSNCHO network ships three such duplicates (an upstream
data bug; no shipped config selects it) — `parse_network` must REFUSE them
(warn under `duplicates_ok=True`), and the networks the shipped configs DO
select must stay clean.

The trailing `Temp` annotation column documents each thermal rate's fitted
range. Nothing enforces it at runtime (matching VULCAN-master), so
`runtime_validation.report_rate_temp_ranges` reports the exposure once per
run. The pinned counts here are measured on the shipped atm-file profiles;
they move only when the network files or the range parser change.

Run from VULCAN-JAX/:
    pytest tests/test_network_integrity.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def _parse_quietly(net_rel_path: str):
    from vulcan_jax import network as netmod

    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        net = netmod.parse_network(net_rel_path, duplicates_ok=True)
    dup_msgs = [str(w.message) for w in log if "duplicated reaction" in str(w.message)]
    return net, dup_msgs


def _configured_networks() -> list[str]:
    nets = set()
    for cfg_path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        with open(cfg_path) as fh:
            nets.add(yaml.safe_load(fh)["network"])
    return sorted(nets)


@pytest.mark.parametrize("net_rel", _configured_networks(), ids=os.path.basename)
def test_configured_networks_have_no_duplicate_reactions(net_rel):
    """No network a shipped config selects may contain a double-counted row."""
    _net, dup_msgs = _parse_quietly(net_rel)
    assert not dup_msgs, dup_msgs


def test_duplicate_reactions_refuse_naming_the_equations():
    """TiSNCHO's three upstream duplicates must refuse by default, warn on opt-in."""
    from vulcan_jax import network as netmod

    with pytest.raises(ValueError, match="3 duplicated reaction"):
        netmod.parse_network("thermo/TiSNCHO_photo_network.txt")

    _net, dup_msgs = _parse_quietly("thermo/TiSNCHO_photo_network.txt")
    assert len(dup_msgs) == 1
    msg = dup_msgs[0]
    for eq in ("TiO2 + M -> Ti + O2 + M", "TiO + M -> Ti + O + M", "VO + M -> V + O + M"):
        assert eq in msg


@pytest.mark.parametrize(
    "annotation, expected",
    [
        ("1992OLD/LOG8426-8430 250-2580", ((250.0, 2580.0),)),
        ("Estimate 300-2.50E4", ((300.0, 25000.0),)),  # sci-notation upper bound
        ("2007GAN/GLO6679-6692 298", ((298.0, 298.0),)),  # bare measurement T
        ("1981PET/SAP771 1560-2270, 1500-1900", ((1560.0, 2270.0), (1500.0, 1900.0))),
        ("1986TSA/HAM1087300-2500", ()),  # reference fused to range: refuse
        ("1991DEA/DAV183-1911500-4200", ()),  # fused, inverted: refuse
        ("--", ()),
        ("KIDA", ()),
    ],
)
def test_temp_range_annotation_parser(annotation, expected):
    """Strict end-of-row scan: real ranges parse, garbled ones refuse."""
    from vulcan_jax.network import _parse_temp_ranges

    assert _parse_temp_ranges(annotation.split()) == expected


# Measured 2026-08-21 on the shipped atm-file profiles (see module docstring).
# any/all = rows outside the union of their documented ranges in >=1 / all
# profile layers; no_range = rows whose annotation has no parseable range.
_EXPOSURE = {
    "HD189.yaml": {"thermal_rows": 390, "no_range": 95, "any_outside": 291, "all_outside": 43},
    "HD209.yaml": {"thermal_rows": 390, "no_range": 95, "any_outside": 291, "all_outside": 58},
    "W39b.yaml": {"thermal_rows": 513, "no_range": 143, "any_outside": 194, "all_outside": 63},
}


def test_advisory_fires_per_profile_not_per_network(capsys, monkeypatch):
    """A second profile on the same network must report; the same one must not.

    The exposure depends on Tco, so caching by network path alone silently
    suppressed every case after the first (e.g. HD209 after HD189).
    """
    from vulcan_jax import runtime_validation as rv

    monkeypatch.setattr(rv, "_TEMP_RANGE_REPORTED", set())
    net, _ = _parse_quietly("thermo/NCHO_photo_network.txt")
    hot = np.linspace(800.0, 6000.0, 150)
    cold = np.linspace(300.0, 1200.0, 100)
    rv.report_rate_temp_ranges(net, hot)
    rv.report_rate_temp_ranges(net, hot)  # identical case: stays quiet
    rv.report_rate_temp_ranges(net, cold)  # new profile, same network: reports
    assert capsys.readouterr().out.count("rate T-range advisory") == 2


@pytest.mark.parametrize("cfg_name", sorted(_EXPOSURE), ids=str)
def test_shipped_profile_temp_range_exposure_is_pinned(cfg_name):
    """The out-of-range exposure of each shipped case is known and stays known."""
    from vulcan_jax import runtime_validation as rv

    with open(f"src/vulcan_jax/configs/{cfg_name}") as fh:
        cfg = yaml.safe_load(fh)
    net, _ = _parse_quietly(cfg["network"])
    prof = np.genfromtxt(f"src/vulcan_jax/{cfg['atm_file']}", names=True, skip_header=1)
    temp_col = next(n for n in prof.dtype.names if n.lower().startswith("temp"))
    exposure = rv.rate_temp_range_exposure(net, np.asarray(prof[temp_col], float))
    assert exposure == _EXPOSURE[cfg_name]


# A network whose species lack a composition row cannot be integrated: atom
# counts and mass drive elemental bookkeeping, mean molecular weight, and
# molecular diffusion. The shipped C3 network uses C3, which upstream's
# all_compose.txt never defines. The composition table is import-frozen, so
# selecting that network needs a fresh process.
def test_network_species_without_composition_row_refuse():
    child = (
        "import os, sys, warnings; warnings.filterwarnings('ignore');"
        "os.environ['JAX_PLATFORM_NAME']='cpu';"
        "os.environ['VULCAN_JAX_NETWORK']='thermo/SNCHO_photo_network_C3.txt';"
        "os.environ['VULCAN_JAX_ATOM_LIST']='H,O,C,N,S';"
        f"sys.path.insert(0, {str(ROOT / 'src')!r});"
        "from vulcan_jax import composition"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True,
        timeout=600, check=False,
    )
    assert proc.returncode != 0, "C3 network loaded despite having no C3 composition row"
    assert "C3" in proc.stderr, proc.stderr
