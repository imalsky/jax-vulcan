"""Two elemental-abundance presets ship, and a config may select either.

The single most common way to get a wrong "VULCAN-JAX vs VULCAN 2.0" number is
to leave the two codes on different composition files. VULCAN-JAX defaults to
Lodders 2019 with the rocky elements suppressed (correct for networks that have
no Mg/Si/Fe species); upstream VULCAN ships Lodders 2009 with everything at
solar. On HD 189733 b that is a ~20% median disagreement out of the box and
3.8e-06 once matched.

So both files ship, `fastchem_solar_abundance_file` picks between them, and a
file that is neither is rejected rather than silently accepted.

Run from VULCAN-JAX/:
    pytest tests/test_fastchem_abundance_presets.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PKG = ROOT / "src" / "vulcan_jax"
DEFAULT_REL = "fastchem_vulcan/input/solar_element_abundances.dat"
UPSTREAM_REL = "fastchem_vulcan/input/solar_element_abundances_lodders2009.dat"


def _parse(rel: str) -> tuple[dict[str, float], list[str]]:
    """Return (element -> dex, row order) for an abundance file."""
    values: dict[str, float] = {}
    order: list[str] = []
    for line in open(PKG / rel):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            continue
        order.append(parts[0])
    return values, order


def _validate(rel: str) -> list[str]:
    from vulcan_jax.runtime_validation import _validate_fastchem_input_vs_network

    cfg = SimpleNamespace(ini_mix="EQ", fastchem_solar_abundance_file=rel)
    return _validate_fastchem_input_vs_network(cfg, PKG)


# Both presets exist and are accepted


@pytest.mark.parametrize("rel", [DEFAULT_REL, UPSTREAM_REL])
def test_shipped_preset_is_accepted_in_fastchem_row_order(rel):
    """Both presets validate clean, and FastChem hard-codes element slots so
    P must precede S in BOTH files (mass_action_constant.cpp index_P=5,
    index_S=6; upstream's own file has them the other way round, C12)."""
    from vulcan_jax.runtime_validation import _FASTCHEM_ELEMENT_ORDER

    assert (PKG / rel).exists(), f"{rel} is not shipped"
    assert _validate(rel) == [], f"{rel} should validate clean"
    _values, order = _parse(rel)
    assert order in (_FASTCHEM_ELEMENT_ORDER, _FASTCHEM_ELEMENT_ORDER[:-1]), (
        f"{rel} row order is {order}, must be {_FASTCHEM_ELEMENT_ORDER}"
    )
    assert order.index("P") < order.index("S")


def test_upstream_preset_carries_upstream_values():
    """Values must be upstream's verbatim, or it is not a matched comparison.

    Only the P/S ROW ORDER is corrected; no dex is changed.
    """
    values, _order = _parse(UPSTREAM_REL)
    upstream = {
        "C": 8.4434,
        "H": 12.00,
        "He": 10.9864,
        "N": 7.9130,
        "O": 8.7826,
        "P": 5.5058,
        "S": 7.12,
        "Si": 7.5867,
        "Ti": 4.9794,
        "V": 4.0437,
        "Cl": 5.3002,
        "K": 5.1619,
        "Na": 6.3479,
        "Mg": 7.5995,
        "F": 4.49196,
        "Ca": 6.3677,
        "Fe": 7.5151,
    }
    for elem, dex in upstream.items():
        assert values[elem] == dex, f"{elem}: {values[elem]} != upstream {dex}"


def test_the_two_presets_differ_only_where_expected():
    """For a C-H-N-O network, helium is the ONLY value that changes.

    This is why the disagreement hid for so long: helium has no chemistry, so
    it showed up as a flat offset in the deep column rather than as anything
    that looked like a chemistry bug.
    """
    a, _ = _parse(DEFAULT_REL)
    b, _ = _parse(UPSTREAM_REL)
    for elem in ("C", "H", "N", "O"):
        assert a[elem] == b[elem], f"{elem} should be identical across presets"
    assert a["He"] != b["He"], "helium is the C-H-N-O difference; it must differ"
    assert a["Mg"] == -3.0 and b["Mg"] > 0, "rocky suppression is the other axis"


# Anything else is rejected


def test_a_hand_edited_file_is_rejected_naming_both_presets(tmp_path):
    """A file matching neither preset must fail loudly, not pass quietly."""
    values, order = _parse(DEFAULT_REL)
    values["O"] = 8.9  # a plausible-looking edit
    rel = "fastchem_vulcan/input/_test_edited_abundances.dat"
    target = PKG / rel
    target.write_text(
        "# test fixture\n" + "".join(f"{e}  {values[e]}\n" for e in order)
    )
    try:
        errors = _validate(rel)
        assert len(errors) == 1, "an unrecognised composition must be an error"
        msg = errors[0]
        assert "solar_element_abundances.dat" in msg
        assert "solar_element_abundances_lodders2009.dat" in msg
        assert "O=+8.9000" in msg
    finally:
        target.unlink()


# Every config states its choice


def test_every_config_declares_an_existing_abundance_file():
    """No config may inherit the composition implicitly (an omitted key falls
    back to the code default, making the composition invisible in the file
    that defines the run), and each declared file must exist."""
    for cfg_path in sorted(glob.glob("src/vulcan_jax/configs/*.yaml")):
        cfg = yaml.safe_load(open(cfg_path))
        assert "fastchem_solar_abundance_file" in cfg, (
            f"{cfg_path} does not declare fastchem_solar_abundance_file")
        rel = cfg["fastchem_solar_abundance_file"]
        assert (PKG / rel).exists(), f"{cfg_path} points at missing {rel}"
