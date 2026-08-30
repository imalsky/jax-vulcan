"""Parse a VULCAN reaction-network text file into a typed Network.

Forward reactions sit at odd parser-indices (1, 3, 5, ...) with a reverse
slot at i+1; reverse rates are filled in later from NASA-9 Gibbs energies.
Reverse computation stops at `stop_rev_indx`, so condensation, photo,
ion, and radiative-recombination reactions never get a reverse slot.

Sections in file order: two-body, 3-body w/ k_inf (Lindemann), 3-body
w/o k_inf, special (hardcoded), condensation, radiative recombination,
photo, ionisation.

Species ordering follows first appearance in the network file. Species
that appear only in the config (e.g. an inert `const_mix` gas like Ar)
are NOT appended — `runtime_validation` rejects such configs upfront.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SECTION_TWO_BODY = "two_body"
_SECTION_THREE_BODY_KINF = "three_body_kinf"
_SECTION_THREE_BODY_NO_KINF = "three_body_no_kinf"
_SECTION_SPECIAL = "special"
_SECTION_CONDEN = "conden"
_SECTION_RADIATIVE = "radiative"
_SECTION_PHOTO = "photo"
_SECTION_ION = "ion"

# Sections whose rows carry Arrhenius fits read from the file; only these
# have a meaningful documented temperature range in the trailing annotation.
_THERMAL_SECTIONS = (
    _SECTION_TWO_BODY,
    _SECTION_THREE_BODY_KINF,
    _SECTION_THREE_BODY_NO_KINF,
)


@dataclass(frozen=True)
class Network:
    """Parsed reaction network.

    All arrays use 1-based reaction indexing (entries [0] are unused).
    `nr` is the total number of slots including reverses; arrays have
    leading shape [nr+1, ...].
    """

    species: tuple[str, ...]
    species_idx: dict  # species -> int (0-based species index in `species`)
    ni: int  # number of species
    nr: int  # number of reaction slots (forward + reverse)

    # Stoichiometry, padded with `ni` (no-op slot pointing to y[ni]=1.0).
    # Shape [nr+1, max_reac/max_prod]. Row 0 is unused (1-based indexing).
    reactant_idx: np.ndarray  # int64
    product_idx: np.ndarray  # int64
    reactant_stoich: np.ndarray  # float64
    product_stoich: np.ndarray  # float64

    # Per-reaction Arrhenius params (k = a * T^n * exp(-E/T)).
    # Shape [nr+1]. Set to 0 for slots with no Arrhenius (photo, conden, etc.)
    a: np.ndarray
    n: np.ndarray
    E: np.ndarray
    a_inf: np.ndarray
    n_inf: np.ndarray
    E_inf: np.ndarray

    # Reaction-type masks, shape [nr+1]
    is_forward: np.ndarray  # bool: True at odd indices
    is_three_body: np.ndarray  # bool: rate gets multiplied by M
    has_kinf: np.ndarray  # bool: uses Lindemann falloff
    is_special: np.ndarray  # bool: hardcoded rate (OH+CH3+M)
    is_conden: np.ndarray  # bool
    is_photo: np.ndarray  # bool: photodissociation
    is_ion: np.ndarray  # bool: photoionization

    # Section delimiters (parser-i values; 1-based)
    stop_rev_indx: int  # reverse slots filled for even i in 2..stop_rev_indx-1
    conden_indx: int  # parser-i of first condensation reaction
    photo_indx: int  # parser-i of first photo reaction

    # Photo metadata
    photo_sp: tuple[str, ...]  # species that photodissociate
    pho_rate_index: dict  # (species, branch) -> reaction parser-i
    n_branch: dict  # species -> num branches
    ion_sp: tuple[str, ...]  # species that photoionize
    ion_rate_index: dict  # (species, branch) -> reaction parser-i
    ion_branch: dict  # species -> num branches

    # Reaction text, indexed by parser-i (1-based)
    Rf: dict  # parser-i -> "A + B -> C + D"

    # Original file path for debugging
    network_path: str

    # Documented temperature validity, parser-i -> ((lo, hi), ...) in K, for
    # thermal (Arrhenius) forward rows only. An empty tuple means the row's
    # trailing annotation carries no parseable range. Advisory: nothing in
    # either VULCAN implementation enforces these at runtime.
    temp_ranges: dict | None = None


_RE_LINE = re.compile(r"^\s*(\d*)\s*\[\s*([^\]]+)\s*\]\s*(.*)$")

# Temperature annotations in the trailing `Temp` column: '250-2580',
# '300-2.50E4', a bare measurement temperature '298', or comma-separated
# lists of these. No sign and no negative exponent — temperatures are
# positive Kelvin, and allowing 'E-' would make the range split ambiguous.
_NUM = r"\d+(?:\.\d+)?(?:[Ee]\+?\d+)?"
_RE_TEMP_RANGE = re.compile(rf"^({_NUM})-({_NUM})$")
_RE_TEMP_SINGLE = re.compile(rf"^{_NUM}$")


def _parse_temp_ranges(tokens: list[str]) -> tuple[tuple[float, float], ...]:
    """Parse the documented temperature range(s) off a row's annotation tokens.

    Scans from the END of the annotation (the Temp column is last) and stops
    at the first token that is not a range or a bare temperature. Strict on
    purpose: a reference fused to a range (e.g. '1986TSA/HAM1087300-2500')
    or an inverted 'lo-hi' yields NO ranges rather than a garbage window.
    A bare temperature T becomes the degenerate range (T, T).
    """
    out: list[tuple[float, float]] = []
    for tok in reversed(tokens):
        t = tok.strip(",")
        if t == "/":  # k0 / k_inf window separator on three-body rows
            continue
        m = _RE_TEMP_RANGE.match(t)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo > hi:
                return ()
            out.append((lo, hi))
            continue
        if _RE_TEMP_SINGLE.match(t):
            v = float(t)
            out.append((v, v))
            continue
        break
    return tuple(reversed(out))


def _announce_duplicate_reactions(
    network_path: str, dups: list[tuple[str, int, int]], duplicates_ok: bool
) -> None:
    """Refuse (or, with `duplicates_ok`, warn about) repeated reactions.

    The parser is positional and appends every row, so both copies of a
    duplicated reaction become independent slots and their contributions are
    double-counted (forward AND reverse) — silent scientific corruption, so
    the default is to raise. Upstream's `make_chem_funs.py` only prints
    (`check_duplicate()`). The vendored TiSNCHO network ships three such
    duplicates (an upstream data bug); no shipped config selects it.
    `duplicates_ok=True` downgrades to a RuntimeWarning for inspection and
    file-sweep uses that must parse the file anyway.
    """
    if not dups:
        return
    shown = "; ".join(
        f"{eq!r} at positions {first} and {second}"
        for eq, first, second in dups[:4]
    )
    more = f" (+{len(dups) - 4} more)" if len(dups) > 4 else ""
    msg = (
        f"Network {network_path} contains {len(dups)} duplicated reaction(s) "
        f"within the same section: {shown}{more}. Both copies are parsed as "
        "independent reactions, so their rate contributions are DOUBLE-COUNTED "
        "in both directions. Fix the network file before using it for science, "
        "or pass duplicates_ok=True to parse it anyway for inspection."
    )
    if not duplicates_ok:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _parse_term(term: str) -> tuple[int, str]:
    """Parse a reactant/product term like 'H', '2*H', '3*OH'.

    Returns (stoichiometric coefficient, species name).
    """
    parts = term.split("*")
    if len(parts) == 1:
        return 1, parts[0]
    return int(parts[0]), parts[1]


def _parse_eq(eq: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Parse 'A + B + M -> C + D + M' into ([(stoich, species), ...], [(stoich, species), ...]).

    M is preserved in the parsed lists -- the caller decides whether to
    include or exclude it.
    """
    if "->" not in eq:
        raise ValueError(f"Reaction equation missing '->': {eq!r}")
    lhs, rhs = eq.split("->", 1)
    reactants = [_parse_term(t.strip()) for t in lhs.split("+") if t.strip()]
    products = [_parse_term(t.strip()) for t in rhs.split("+") if t.strip()]
    return reactants, products


def _detect_section(line: str, _current: str) -> str | None:
    """Return a new section name if `line` is a section marker, else None.

    Order matters: '# 3-body reactions without high-pressure rates' must
    match before '# 3-body'.
    """
    s = line.lstrip()
    if s.startswith("# 3-body reactions without high-pressure rates"):
        return _SECTION_THREE_BODY_NO_KINF
    if s.startswith("# 3-body"):
        return _SECTION_THREE_BODY_KINF
    if s.startswith("# special"):
        return _SECTION_SPECIAL
    if s.startswith("# condensation"):
        return _SECTION_CONDEN
    if s.startswith("# radiative"):
        return _SECTION_RADIATIVE
    if s.startswith("# photo"):
        return _SECTION_PHOTO
    if s.startswith("# ionisation") or s.startswith("# ionization"):
        return _SECTION_ION
    return None


def parse_network(network_path: str | Path, *, duplicates_ok: bool = False) -> Network:
    """Parse a VULCAN network file. See module docstring.

    Raises ValueError if a reaction is duplicated within one section (its
    contribution would be silently double-counted); `duplicates_ok=True`
    downgrades that to a RuntimeWarning for inspection/file-sweep uses.
    """
    from ._paths import resolve_data_path

    network_path = str(resolve_data_path(str(network_path)))

    species_order: list[str] = []
    species_idx: dict[str, int] = {}

    section: str = _SECTION_TWO_BODY
    parser_i = 1  # forwards: 1, 3, 5, ...
    stop_rev_indx: int | None = None
    conden_indx: int | None = None
    photo_indx: int | None = None

    photo_sp: list[str] = []
    pho_rate_index: dict[tuple[str, int], int] = {}
    n_branch: dict[str, int] = {}
    ion_sp: list[str] = []
    ion_rate_index: dict[tuple[str, int], int] = {}
    ion_branch: dict[str, int] = {}

    forward_records: list[dict] = []
    Rf_text: dict[int, str] = {}
    temp_ranges: dict[int, tuple[tuple[float, float], ...]] = {}

    def _intern_species(sp: str) -> int:
        if sp == "M":
            return -1
        if sp not in species_idx:
            species_idx[sp] = len(species_order)
            species_order.append(sp)
        return species_idx[sp]

    with open(network_path) as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # Section markers must be checked before the # reverse-stops marker.
            new_sec = _detect_section(line, section)
            if new_sec is not None:
                section = new_sec
                if section == _SECTION_CONDEN and conden_indx is None:
                    conden_indx = parser_i
                if section == _SECTION_RADIATIVE:
                    raise ValueError(
                        f"{network_path}:{lineno}: radiative-recombination networks are "
                        "not supported -- rates.py, rates_jax.py and gibbs.py all "
                        "exclude those slots, so their reactions would be silently "
                        "dropped."
                    )
                if section == _SECTION_PHOTO and photo_indx is None:
                    photo_indx = parser_i
                continue

            stripped = line.lstrip()
            if stripped.startswith("# reverse stops"):
                stop_rev_indx = parser_i
                continue
            if stripped.startswith("#"):
                continue

            m = _RE_LINE.match(line)
            if m is None:
                raise ValueError(
                    f"{network_path}:{lineno}: not a reaction row and not a comment: "
                    f"{line.rstrip()!r}"
                )

            file_id_str, eq, tail = m.group(1), m.group(2), m.group(3)
            # A blank id column is legitimate: upstream never reads it and
            # renumbers from its own counter, so such rows are real reactions
            # (NCHO_full_photo_network.txt, SNCHO_photo_network_C3.txt).
            file_id = int(file_id_str) if file_id_str else 0
            eq = eq.strip()
            reactants, products = _parse_eq(eq)

            for _stoich, sp in reactants + products:
                _intern_species(sp)

            # Collapse species → stoich (M dropped here; tracked via has_M_*).
            r_collapsed: dict[int, float] = {}
            for stoich, sp in reactants:
                if sp == "M":
                    continue
                r_collapsed[species_idx[sp]] = r_collapsed.get(
                    species_idx[sp], 0.0
                ) + float(stoich)
            p_collapsed: dict[int, float] = {}
            for stoich, sp in products:
                if sp == "M":
                    continue
                p_collapsed[species_idx[sp]] = p_collapsed.get(
                    species_idx[sp], 0.0
                ) + float(stoich)

            # Asymmetric dissociation (e.g. HNCO + M → H + NCO) has M only
            # on the reactant side; forward and reverse get different
            # M-factors. Track each side independently.
            has_M_reac = any(sp == "M" for _, sp in reactants)
            has_M_prod = any(sp == "M" for _, sp in products)
            has_M = has_M_reac or has_M_prod

            # Parse the numeric prefix of `tail` (Arrhenius / aux); the
            # rest of `tail` is reference / temperature commentary.
            cols = tail.split()
            num_cols: list[float] = []
            for c in cols:
                try:
                    num_cols.append(float(c))
                except ValueError:
                    break

            rec = {
                "parser_i": parser_i,
                "file_id": file_id,
                "section": section,
                "Rf": eq,
                "reactants_collapsed": r_collapsed,
                "products_collapsed": p_collapsed,
                "has_M": has_M,
                "has_M_reac": has_M_reac,
                "has_M_prod": has_M_prod,
                "num_cols": num_cols,
                # Photo/ion-specific metadata: columns[0]=species, columns[1]=branch
                "photo_meta": (cols[0], int(cols[1]))
                if (
                    section in (_SECTION_PHOTO, _SECTION_ION)
                    and len(cols) >= 2
                    and cols[1].isdigit()
                )
                else None,
            }
            forward_records.append(rec)
            Rf_text[parser_i] = eq
            if section in _THERMAL_SECTIONS:
                temp_ranges[parser_i] = _parse_temp_ranges(cols[len(num_cols):])

            if section == _SECTION_PHOTO:
                target_sp = cols[0] if cols else eq.split()[0]
                if target_sp not in photo_sp:
                    photo_sp.append(target_sp)
                if rec["photo_meta"] is not None:
                    pho_rate_index[rec["photo_meta"]] = parser_i
                    n_branch[target_sp] = max(
                        n_branch.get(target_sp, 0), rec["photo_meta"][1]
                    )
            elif section == _SECTION_ION:
                target_sp = cols[0] if cols else eq.split()[0]
                if target_sp not in ion_sp:
                    ion_sp.append(target_sp)
                if rec["photo_meta"] is not None:
                    ion_rate_index[rec["photo_meta"]] = parser_i
                    ion_branch[target_sp] = max(
                        ion_branch.get(target_sp, 0), rec["photo_meta"][1]
                    )

            parser_i += 2

    # parser_i has been bumped past the last reverse → nr = parser_i - 1.
    nr = parser_i - 1
    ni = len(species_order)

    # A reaction repeated within one section is double-counted by this
    # positional parser: refuse (upstream's check_duplicate() only prints).
    seen_eq: dict[tuple, int] = {}
    dups: list[tuple[str, int, int]] = []
    for rec in forward_records:
        key = (
            rec["section"],
            tuple(sorted(rec["reactants_collapsed"].items())),
            tuple(sorted(rec["products_collapsed"].items())),
            rec["has_M_reac"],
            rec["has_M_prod"],
            rec["photo_meta"],
        )
        first_i = seen_eq.setdefault(key, rec["parser_i"])
        if first_i != rec["parser_i"]:
            dups.append((rec["Rf"], first_i, rec["parser_i"]))
    _announce_duplicate_reactions(network_path, dups, duplicates_ok)

    if stop_rev_indx is None:
        # Older networks may omit `# reverse stops`; default: reverses are
        # computed up to the photo section.
        stop_rev_indx = photo_indx if photo_indx is not None else nr + 1
    if conden_indx is None:
        conden_indx = nr + 1
    if photo_indx is None:
        photo_indx = nr + 1

    max_reac = max(
        (len(rec["reactants_collapsed"]) for rec in forward_records),
        default=1,
    )
    max_prod = max(
        (len(rec["products_collapsed"]) for rec in forward_records),
        default=1,
    )
    max_reac = max(max_reac, 1)
    max_prod = max(max_prod, 1)

    PAD = ni  # pad → y[ni]=1; no-op multiplier and segment_sum drops the slot.

    # Reverse reactions store the forward's products in their reactant slot,
    # so both directions need to fit in the same `max_terms` width.
    max_terms = max(max_reac, max_prod)
    reactant_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    product_idx = np.full((nr + 1, max_terms), PAD, dtype=np.int64)
    reactant_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)
    product_stoich = np.zeros((nr + 1, max_terms), dtype=np.float64)

    a = np.zeros(nr + 1, dtype=np.float64)
    n = np.zeros(nr + 1, dtype=np.float64)
    E = np.zeros(nr + 1, dtype=np.float64)
    a_inf = np.zeros(nr + 1, dtype=np.float64)
    n_inf = np.zeros(nr + 1, dtype=np.float64)
    E_inf = np.zeros(nr + 1, dtype=np.float64)

    is_forward = np.zeros(nr + 1, dtype=bool)
    is_three_body = np.zeros(nr + 1, dtype=bool)
    has_kinf = np.zeros(nr + 1, dtype=bool)
    is_special = np.zeros(nr + 1, dtype=bool)
    is_conden = np.zeros(nr + 1, dtype=bool)
    is_photo = np.zeros(nr + 1, dtype=bool)
    is_ion = np.zeros(nr + 1, dtype=bool)

    for rec in forward_records:
        i = rec["parser_i"]
        is_forward[i] = True
        for k_slot, (sp_idx, stoich) in enumerate(rec["reactants_collapsed"].items()):
            reactant_idx[i, k_slot] = sp_idx
            reactant_stoich[i, k_slot] = stoich
        for k_slot, (sp_idx, stoich) in enumerate(rec["products_collapsed"].items()):
            product_idx[i, k_slot] = sp_idx
            product_stoich[i, k_slot] = stoich
        ir = i + 1
        is_forward[ir] = False
        for k_slot, (sp_idx, stoich) in enumerate(rec["products_collapsed"].items()):
            reactant_idx[ir, k_slot] = sp_idx
            reactant_stoich[ir, k_slot] = stoich
        for k_slot, (sp_idx, stoich) in enumerate(rec["reactants_collapsed"].items()):
            product_idx[ir, k_slot] = sp_idx
            product_stoich[ir, k_slot] = stoich

        # Asymmetric dissociation: forward and reverse can differ in M
        # (e.g. `HNCO + M → H + NCO` has M on the LHS only, so the reverse
        # is bimolecular without M).
        is_three_body[i] = rec["has_M_reac"]
        is_three_body[ir] = rec["has_M_prod"]

        sec = rec["section"]
        if sec == _SECTION_THREE_BODY_KINF:
            has_kinf[i] = True
            has_kinf[ir] = True
        if sec == _SECTION_SPECIAL:
            is_special[i] = True
            is_special[ir] = True
        if sec == _SECTION_CONDEN:
            is_conden[i] = True
            is_conden[ir] = True
        if sec == _SECTION_PHOTO:
            is_photo[i] = True
            # Photo has no reverse — guarded via stop_rev_indx.
        if sec == _SECTION_ION:
            is_ion[i] = True

        cols = rec["num_cols"]
        if sec in (
            _SECTION_TWO_BODY,
            _SECTION_THREE_BODY_KINF,
            _SECTION_THREE_BODY_NO_KINF,
        ):
            if len(cols) >= 3:
                a[i], n[i], E[i] = cols[0], cols[1], cols[2]
            if sec == _SECTION_THREE_BODY_KINF and len(cols) >= 6:
                a_inf[i], n_inf[i], E_inf[i] = cols[3], cols[4], cols[5]

    species = tuple(species_order)
    return Network(
        species=species,
        species_idx=dict(species_idx),
        ni=ni,
        nr=nr,
        reactant_idx=reactant_idx,
        product_idx=product_idx,
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        a=a,
        n=n,
        E=E,
        a_inf=a_inf,
        n_inf=n_inf,
        E_inf=E_inf,
        is_forward=is_forward,
        is_three_body=is_three_body,
        has_kinf=has_kinf,
        is_special=is_special,
        is_conden=is_conden,
        is_photo=is_photo,
        is_ion=is_ion,
        stop_rev_indx=stop_rev_indx,
        conden_indx=conden_indx,
        photo_indx=photo_indx,
        photo_sp=tuple(photo_sp),
        pho_rate_index=dict(pho_rate_index),
        n_branch=dict(n_branch),
        ion_sp=tuple(ion_sp),
        ion_rate_index=dict(ion_rate_index),
        ion_branch=dict(ion_branch),
        Rf=dict(Rf_text),
        network_path=network_path,
        temp_ranges=dict(temp_ranges),
    )
