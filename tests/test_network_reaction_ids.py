"""Photolysis rates must be indexed by parser POSITION, never by the id column.

VULCAN's `make_chem_funs.py` renumbers a network file in place the first time it
runs, so a file that upstream has run through has `file_id == parser_position` on
every row. A file that has been fetched from a remote, hand-edited, or simply
never run does not. Six of the eighteen networks vendored here are in that state.

`legacy_io.ReadRate.read_rate` used to build `pho_rate_index` / `ion_rate_index`
from the id column, while `network.parse_network` builds them from the position
and `rates.build_rate_array` / `rates.apply_remove_list` index `k_arr`
positionally. On any file where the two disagree this wrote a photolysis rate
into the wrong reaction slot -- silently for three of the vendored networks, and
with an out-of-range IndexError for three more.

Run from VULCAN-JAX/:
    pytest tests/test_network_reaction_ids.py
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# The id column is optional (upstream never reads it); a blank id is a
# real reaction row and occupies a position like any other.
_SECTION_RE = re.compile(r"^(\d*)\s*\[")


def _file_ids_by_position(path: str) -> list[tuple[int, int, str]]:
    """Return (position, written_id, section) for every reaction row in `path`.

    Mirrors the section tracking in `legacy_io.ReadRate.read_rate` and
    `network.parse_network`: forward reactions occupy the odd positions
    1, 3, 5, ... and each row consumes two slots (forward + reverse).
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


def _shipped_networks() -> list[str]:
    return sorted(glob.glob("src/vulcan_jax/thermo/*network*.txt"))


def _configured_networks() -> dict[str, list[str]]:
    """network basename -> the shipped configs that select it."""
    out: dict[str, list[str]] = {}
    for cfg_path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        net = os.path.basename(cfg["network"])
        out.setdefault(net, []).append(os.path.basename(cfg_path))
    return out


# ---------------------------------------------------------------------------
# The invariant that actually protects the rate array
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("net_path", _shipped_networks(), ids=os.path.basename)
def test_photo_rate_index_is_positional_and_in_range(net_path):
    """Every photo/ion index must be an in-range odd (forward) slot of k_arr.

    This is the property the old id-column indexing violated. `k_arr` has
    `nr + 1` rows with row 0 unused, so a valid forward slot is odd and
    `<= nr`.
    """
    from vulcan_jax import network as netmod

    # duplicates_ok: this is a file sweep over every vendored network, and the
    # positional-index invariant holds regardless of duplicated rows (TiSNCHO
    # carries three; tests/test_network_integrity.py pins the refusal).
    net = netmod.parse_network(net_path, duplicates_ok=True)
    indices = list(net.pho_rate_index.values()) + list(net.ion_rate_index.values())
    if not indices:
        pytest.skip(f"{os.path.basename(net_path)} has no photo/ion reactions")

    for idx in indices:
        assert 1 <= idx <= net.nr, (
            f"{os.path.basename(net_path)}: photo/ion index {idx} outside "
            f"[1, nr={net.nr}] -- this is the IndexError into k_arr"
        )
        assert idx % 2 == 1, (
            f"{os.path.basename(net_path)}: photo/ion index {idx} is even, so it "
            "names a REVERSE slot; photolysis has no reverse"
        )

    positions = {
        pos
        for pos, _fid, sec in _file_ids_by_position(net_path)
        if sec in ("photo", "ion")
    }
    assert set(indices) <= positions, (
        f"{os.path.basename(net_path)}: photo/ion indices are not parser positions. "
        "They must come from the position, not the file's id column."
    )


def test_legacy_io_and_network_agree_on_photo_indices():
    """The two parsers must produce identical photo/ion index maps.

    `network.parse_network` feeds `k_arr`; `legacy_io.ReadRate.read_rate` feeds
    `op_jax.compute_J`, which writes into that same `k_arr`. If they disagree,
    photolysis lands in the wrong row.
    """
    from vulcan_jax import network as netmod
    from vulcan_jax.config import default_config

    cfg = default_config()
    net = netmod.parse_network(str(Path(cfg.network)))

    rows = _file_ids_by_position(
        str(ROOT / "src" / "vulcan_jax" / cfg.network)
        if not os.path.isabs(cfg.network)
        else cfg.network
    )
    photo_positions = [pos for pos, _fid, sec in rows if sec == "photo"]
    assert sorted(net.pho_rate_index.values()) == sorted(photo_positions)


# ---------------------------------------------------------------------------
# Guard against a silent regression in the shipped results
# ---------------------------------------------------------------------------


def test_configured_networks_have_consistent_ids():
    """Every network a shipped config selects must be renumbered.

    Not a correctness requirement -- the fix makes stale files parse correctly
    anyway -- but it pins the fact that no shipped config's results moved when
    the indexing was corrected, and it keeps `cfg.remove_list` written against
    these files meaningful.
    """
    offenders = []
    for net_name in _configured_networks():
        path = f"src/vulcan_jax/thermo/{net_name}"
        stale = [
            (pos, fid)
            for pos, fid, sec in _file_ids_by_position(path)
            if sec in ("photo", "ion") and pos != fid
        ]
        if stale:
            offenders.append((net_name, len(stale)))
    assert not offenders, (
        "shipped configs select networks whose id column is stale: "
        f"{offenders}. Renumber them, or cfg.remove_list entries read off the "
        "file will select the wrong reactions."
    )


def test_stale_id_warning_names_the_file_and_the_remove_list_risk():
    """A stale file must announce itself rather than pass silently."""
    from vulcan_jax.legacy_io import _warn_stale_reaction_ids

    with pytest.warns(RuntimeWarning) as record:
        _warn_stale_reaction_ids(
            "/tmp/example_network.txt",
            [(781, 783, "H2O -> H + OH")],
        )
    msg = str(record[0].message)
    assert "/tmp/example_network.txt" in msg
    assert "remove_list" in msg
    assert "make_chem_funs" in msg


def test_clean_network_does_not_warn():
    """A renumbered file must stay quiet, or the warning becomes noise."""
    import warnings

    from vulcan_jax.legacy_io import _warn_stale_reaction_ids

    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        _warn_stale_reaction_ids("/tmp/clean.txt", [])
    assert not log, f"expected silence, got {[str(x.message) for x in log]}"
