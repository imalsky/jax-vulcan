"""Network-file integrity: no silent duplicates, positional photo indices,
pinned T-range exposure.

The parser is positional and appends every reaction row, so a reaction
duplicated within one section is double-counted in both directions with no
error. The vendored TiSNCHO network ships three such duplicates (an upstream
data bug; no shipped config selects it): `parse_network` must REFUSE them
(warn under `duplicates_ok=True`), and the networks the shipped configs DO
select must stay clean.

The trailing `Temp` annotation column documents each thermal rate's fitted
range. Nothing enforces it at runtime (matching VULCAN-master), so
`runtime_validation.report_rate_temp_ranges` reports the exposure once per
run. The pinned counts are for the shipped atm-file profiles and move only
with the network files or the range parser.

Photolysis rates are indexed by parser position, never by the id column.
VULCAN's `make_chem_funs.py` renumbers a network file in place the first time
it runs, but a fetched, hand-edited or never-run file keeps stale ids, and
keying photolysis by them puts a rate in the wrong k_arr slot or out of range.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


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
    """No network a shipped config selects may contain a double-counted row
    or a stale photo/ion id column. Stale ids do not change the parse, but a
    cfg.remove_list entry read off the file would select the wrong reaction."""
    net, dup_msgs = _parse_quietly(net_rel)
    assert not dup_msgs, dup_msgs
    assert not net.stale_ids, net.stale_ids


_SHIPPED_NETWORKS = sorted(glob.glob("src/vulcan_jax/thermo/*network*.txt"))


@pytest.mark.parametrize("net_path", _SHIPPED_NETWORKS, ids=os.path.basename)
def test_no_reaction_row_is_dropped(net_path):
    """Every non-comment bracketed row must reach the parsed network.

    The id column is optional upstream (make_chem_funs.py:65-73 renumbers),
    so a row without one must still parse. Each row occupies a forward and a
    reverse slot.
    """
    with open(net_path) as fh:
        rows = sum(
            1 for ln in fh if ln.strip() and not ln.lstrip().startswith("#") and "[" in ln
        )
    net, _ = _parse_quietly(net_path)
    assert net.nr == 2 * rows, f"{net.nr // 2} rows parsed, {rows} in the file"


# The id column is optional (upstream never reads it); a blank id is a
# real reaction row and occupies a position like any other.
_SECTION_RE = re.compile(r"^(\d*)\s*\[")


def _file_ids_by_position(path: str) -> list[tuple[int, int, str]]:
    """Return (position, written_id, section) for every reaction row in `path`.

    Mirrors the section tracking in `network.parse_network` independently:
    forward reactions occupy the odd positions 1, 3, 5, ... and each row
    consumes two slots (forward + reverse).
    """
    out: list[tuple[int, int, str]] = []
    section = "thermal"
    pos = 1
    for line in open(path, errors="replace"):
        s = line.strip()
        if s.startswith("#"):
            low = s.lower()
            if "photo disscoiation" in low or "photo dissociation" in low:
                section = "photo"
            elif "ionization" in low or "ionisation" in low:
                section = "ion"
            continue
        m = _SECTION_RE.match(s)
        if not m:
            continue
        out.append((pos, int(m.group(1) or 0), section))
        pos += 2
    return out


@pytest.mark.parametrize("net_path", _SHIPPED_NETWORKS, ids=os.path.basename)
def test_photo_rate_index_is_positional_and_in_range(net_path):
    """The photo/ion indices must be exactly the parser positions of the
    photo/ion rows, each an in-range odd (forward) slot of k_arr.

    `k_arr` has `nr + 1` rows with row 0 unused, so a valid forward slot is
    odd and `<= nr`.
    """
    # duplicates_ok: the positional-index invariant holds regardless of
    # duplicated rows (TiSNCHO carries three;
    # test_duplicate_reactions_refuse_naming_the_equations pins the refusal).
    net, _ = _parse_quietly(net_path)
    indices = list(net.pho_rate_index.values()) + list(net.ion_rate_index.values())

    for idx in indices:
        assert 1 <= idx <= net.nr, (
            f"{os.path.basename(net_path)}: photo/ion index {idx} outside "
            f"[1, nr={net.nr}] -- this is the IndexError into k_arr"
        )
        assert idx % 2 == 1, (
            f"{os.path.basename(net_path)}: photo/ion index {idx} is even, so it "
            "names a REVERSE slot; photolysis has no reverse"
        )

    positions = sorted(
        pos
        for pos, _fid, sec in _file_ids_by_position(net_path)
        if sec in ("photo", "ion")
    )
    assert sorted(indices) == positions, (
        f"{os.path.basename(net_path)}: photo/ion indices are not parser positions. "
        "They must come from the position, not the file's id column."
    )


def test_stale_id_warning_fires_only_on_stale_rows():
    """A stale file must announce itself; a renumbered one must stay quiet."""
    from vulcan_jax.legacy_io import _warn_stale_reaction_ids

    with pytest.warns(RuntimeWarning, match="remove_list"):
        _warn_stale_reaction_ids("net.txt", [(781, 783, "H2O -> H + OH")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_stale_reaction_ids("net.txt", [])


def test_a_radiative_section_is_refused(tmp_path):
    """Every rate path zeroes a radiative slot, so the section must refuse."""
    from vulcan_jax import network as netmod

    f = tmp_path / "net.txt"
    f.write_text("# two-body\n1  [ H + H -> H2 ]  1.0E-10  0.0  0.0\n# radiative\n")
    with pytest.raises(ValueError, match="radiative-recombination"):
        netmod.parse_network(f)


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


# any/all = rows outside the union of their documented ranges in >=1 / all
# profile layers; no_range = rows whose annotation has no parseable range.
_EXPOSURE = {
    "HD189.yaml": {"thermal_rows": 390, "no_range": 95, "any_outside": 291, "all_outside": 43},
    "HD209.yaml": {"thermal_rows": 390, "no_range": 95, "any_outside": 291, "all_outside": 58},
    "W39b.yaml": {"thermal_rows": 513, "no_range": 143, "any_outside": 194, "all_outside": 63},
}


def test_advisory_fires_per_profile_not_per_network(caplog, monkeypatch):
    """A second profile on the same network must report; the same one must not.

    The exposure depends on Tco, so the report is cached per profile, not per network.
    """
    from vulcan_jax import runtime_validation as rv

    monkeypatch.setattr(rv, "_TEMP_RANGE_REPORTED", set())
    net, _ = _parse_quietly("thermo/NCHO_photo_network.txt")
    hot = np.linspace(800.0, 6000.0, 150)
    cold = np.linspace(300.0, 1200.0, 100)
    rv.report_rate_temp_ranges(net, hot)
    rv.report_rate_temp_ranges(net, hot)  # identical case: stays quiet
    rv.report_rate_temp_ranges(net, cold)  # new profile, same network: reports
    assert sum("rate T-range advisory" in r.getMessage() for r in caplog.records) == 2


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
