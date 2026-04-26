"""Parse a VULCAN reaction-network text file into a typed, JAX-friendly Network.

The network format mirrors VULCAN-master (see make_chem_funs.read_network and
op.ReadRate.read_rate). Forward reactions are at odd parser-indices 1, 3, 5,...
and each has a reverse slot at i+1 -- reverse rates are filled in later from
Gibbs energies. Reverse computation stops at `stop_rev_indx` (the file's
`# reverse stops` marker), so condensation and photo reactions never get
reverses.

Sections recognized (in the order they appear in the file):
  Two-body                        -> Arrhenius A T^B exp(-C/T)
  3-body w/ k_inf  (after `# 3-body`)
                                  -> Lindemann-Hinshelwood falloff,
                                     k = k0 / (1 + k0*M/k_inf)
  3-body w/o k_inf                -> k = A T^B exp(-C/T)
  special                         -> hardcoded rate (only OH+CH3+M -> CH3OH+M)
  condensation                    -> k = 0 (driven by saturation logic later)
  radiative                       -> radiative recombination (rare)
  photo                           -> k = 0 initially; J-rates set every photo update
  ionisation                      -> like photo but for ions

Species ordering follows the order species are first encountered while parsing
the network -- this exactly matches `chem_funs.spec_list` from VULCAN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Network:
    """Parsed reaction network. All arrays use 1-based reaction indexing
    (entries [0] are unused) to match VULCAN conventions; i.e. `nr` is the
    number of reaction *slots* including reverses, and arrays have shape
    [nr+1, ...].
    """

    species: tuple[str, ...]
    species_idx: dict  # species -> int (0-based species index in `species`)
    ni: int            # number of species
    nr: int            # number of reaction slots (forward + reverse)

    # Stoichiometry, padded with `ni` (no-op slot pointing to y[ni]=1.0).
    # Shape [nr+1, max_reac/max_prod]. Row 0 is unused (1-based indexing).
    reactant_idx: np.ndarray      # int64
    product_idx: np.ndarray       # int64
    reactant_stoich: np.ndarray   # float64
    product_stoich: np.ndarray    # float64

    # Per-reaction Arrhenius params (k = a * T^n * exp(-E/T)).
    # Shape [nr+1]. Set to 0 for slots with no Arrhenius (photo, conden, etc.)
    a: np.ndarray
    n: np.ndarray
    E: np.ndarray
    a_inf: np.ndarray
    n_inf: np.ndarray
    E_inf: np.ndarray

    # Reaction-type masks, shape [nr+1]
    is_forward: np.ndarray         # bool: True at odd indices
    is_three_body: np.ndarray      # bool: rate gets multiplied by M
    has_kinf: np.ndarray           # bool: uses Lindemann falloff
    is_special: np.ndarray         # bool: hardcoded rate (OH+CH3+M)
    is_conden: np.ndarray          # bool
    is_radiative: np.ndarray       # bool: radiative recombination
    is_photo: np.ndarray           # bool: photodissociation
    is_ion: np.ndarray             # bool: photoionization

    # Section delimiters (parser-i values; 1-based)
    stop_rev_indx: int             # reverses computed for i=2..stop_rev_indx-2
    conden_indx: int               # parser-i of first condensation reaction
    radiative_indx: int            # parser-i of first radiative reaction
    photo_indx: int                # parser-i of first photo reaction
    ion_indx: int                  # parser-i of first ion reaction

    # Photo metadata
    photo_sp: tuple[str, ...]              # species that photodissociate
    pho_rate_index: dict                   # (species, branch) -> reaction parser-i
    n_branch: dict                         # species -> num branches
    ion_sp: tuple[str, ...]                # species that photoionize
    ion_rate_index: dict                   # (species, branch) -> reaction parser-i
    ion_branch: dict                       # species -> num branches

    # Reaction text, indexed by parser-i (1-based)
    Rf: dict                                # parser-i -> "A + B -> C + D"
    Rindx: dict                             # parser-i -> file-ID

    # Original file path for debugging
    network_path: str

    def species_index(self, sp: str) -> int:
        return self.species_idx[sp]


_RE_LINE = re.compile(
    r"^\s*(\d+)\s*\[\s*([^\]]+)\s*\]\s*(.*)$"
)


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


def _detect_section(line: str, current: str) -> str | None:
    """Return new section name if `line` is a section marker, else None.

    Order matters: '# 3-body reactions without high-pressure rates' must be
    matched before '# 3-body'.
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


def parse_network(network_path: str | Path) -> Network:
    """Parse a VULCAN network file. See module docstring."""
    network_path = str(network_path)

    # Pass 1: walk the file in order, collect reactions with their section.
    # Build species ordering as encountered (mirrors make_chem_funs).
    species_order: list[str] = []
    species_idx: dict[str, int] = {}

    section: str = _SECTION_TWO_BODY
    parser_i = 1                       # 1, 3, 5, ... for forward reactions
    stop_rev_indx: int | None = None
    conden_indx: int | None = None
    radiative_indx: int | None = None
    photo_indx: int | None = None
    ion_indx: int | None = None

    photo_sp: list[str] = []
    pho_rate_index: dict[tuple[str, int], int] = {}
    n_branch: dict[str, int] = {}
    ion_sp: list[str] = []
    ion_rate_index: dict[tuple[str, int], int] = {}
    ion_branch: dict[str, int] = {}

    # Per-reaction collected data. Use lists indexed by parser-i//2.
    forward_records: list[dict] = []  # one entry per forward reaction
    Rf_text: dict[int, str] = {}
    Rindx_map: dict[int, int] = {}

    def _intern_species(sp: str) -> int:
        if sp == "M":
            return -1
        if sp not in species_idx:
            species_idx[sp] = len(species_order)
            species_order.append(sp)
        return species_idx[sp]

    with open(network_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # Section markers (order matters; check before reverse-stops since
            # they all start with '#').
            new_sec = _detect_section(line, section)
            if new_sec is not None:
                section = new_sec
                if section == _SECTION_CONDEN and conden_indx is None:
                    conden_indx = parser_i
                if section == _SECTION_RADIATIVE and radiative_indx is None:
                    radiative_indx = parser_i
                if section == _SECTION_PHOTO and photo_indx is None:
                    photo_indx = parser_i
                if section == _SECTION_ION and ion_indx is None:
                    ion_indx = parser_i
                continue

            stripped = line.lstrip()
            if stripped.startswith("# reverse stops"):
                stop_rev_indx = parser_i
                continue
            if stripped.startswith("#"):
                continue  # ordinary comment

            m = _RE_LINE.match(line)
            if m is None:
                # Not a reaction line and not a recognized section marker.
                # Most likely a stray header row -- skip silently.
                continue

            file_id_str, eq, tail = m.group(1), m.group(2), m.group(3)
            file_id = int(file_id_str)
            eq = eq.strip()
            reactants, products = _parse_eq(eq)

            # Intern species (preserves first-seen order, skips M)
            for _stoich, sp in reactants + products:
                _intern_species(sp)

            # Build stoichiometry (collapsed: collect species -> stoich, dropping M)
            r_collapsed: dict[int, float] = {}
            for stoich, sp in reactants:
                if sp == "M":
                    continue
                r_collapsed[species_idx[sp]] = (
                    r_collapsed.get(species_idx[sp], 0.0) + float(stoich)
                )
            p_collapsed: dict[int, float] = {}
            for stoich, sp in products:
                if sp == "M":
                    continue
                p_collapsed[species_idx[sp]] = (
                    p_collapsed.get(species_idx[sp], 0.0) + float(stoich)
                )

            # Track whether M appears in reactants and products separately;
            # asymmetric dissociation reactions (e.g. HNCO + M -> H + NCO) have
            # M only on one side, so forward and reverse get different M-factors.
            has_M_reac = any(sp == "M" for _, sp in reactants)
            has_M_prod = any(sp == "M" for _, sp in products)
            has_M = has_M_reac or has_M_prod   # kept for legacy compatibility

            # Parse Arrhenius / aux columns from `tail`. We only need the
            # numeric prefix; trailing text is references / temperature info.
            cols = tail.split()
            # Find longest prefix of cols that parses as floats
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
                "photo_meta": (cols[0], int(cols[1])) if (
                    section in (_SECTION_PHOTO, _SECTION_ION)
                    and len(cols) >= 2
                    and cols[1].isdigit()
                ) else None,
            }
            forward_records.append(rec)
            Rf_text[parser_i] = eq
            Rindx_map[parser_i] = file_id

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

    # Total reaction slots = parser_i - 1 (last forward) + 1 (its reverse) = parser_i.
    # Since parser_i has been bumped past the last one, the last forward is at
    # parser_i - 2 and its reverse is at parser_i - 1. So nr = parser_i - 1.
    nr = parser_i - 1
    ni = len(species_order)

    if stop_rev_indx is None:
        # Older networks may omit `# reverse stops`; assume reverses are
        # computed for everything before the photo section.
        stop_rev_indx = photo_indx if photo_indx is not None else nr + 1
    if conden_indx is None:
        conden_indx = nr + 1   # sentinel: no condensation reactions
    if radiative_indx is None:
        radiative_indx = nr + 1
    if photo_indx is None:
        photo_indx = nr + 1
    if ion_indx is None:
        ion_indx = nr + 1

    # Build padded stoichiometry tables.
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

    PAD = ni  # pad index points at y[ni]=1 (no-op multiplier / no-op segment)

    # Both arrays must accommodate forward AND reverse reactions, so the
    # reactant slot of a reverse reaction holds the forward's products.
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
    is_radiative = np.zeros(nr + 1, dtype=bool)
    is_photo = np.zeros(nr + 1, dtype=bool)
    is_ion = np.zeros(nr + 1, dtype=bool)

    for rec in forward_records:
        i = rec["parser_i"]
        is_forward[i] = True
        # Stoichiometry (forward)
        for k_slot, (sp_idx, stoich) in enumerate(rec["reactants_collapsed"].items()):
            reactant_idx[i, k_slot] = sp_idx
            reactant_stoich[i, k_slot] = stoich
        for k_slot, (sp_idx, stoich) in enumerate(rec["products_collapsed"].items()):
            product_idx[i, k_slot] = sp_idx
            product_stoich[i, k_slot] = stoich
        # Reverse: reactants <-> products
        ir = i + 1
        is_forward[ir] = False
        for k_slot, (sp_idx, stoich) in enumerate(rec["products_collapsed"].items()):
            reactant_idx[ir, k_slot] = sp_idx
            reactant_stoich[ir, k_slot] = stoich
        for k_slot, (sp_idx, stoich) in enumerate(rec["reactants_collapsed"].items()):
            product_idx[ir, k_slot] = sp_idx
            product_stoich[ir, k_slot] = stoich

        # Forward gets M if reactants had M; reverse gets M if forward
        # products had M (= reverse reactants now have M). Asymmetric
        # dissociations like `HNCO + M -> H + NCO` have M-only-on-LHS so the
        # reverse rate is bimolecular without M.
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
        if sec == _SECTION_RADIATIVE:
            is_radiative[i] = True
            is_radiative[ir] = True
        if sec == _SECTION_PHOTO:
            is_photo[i] = True
            # No reverse for photo (handled by stop_rev_indx)
        if sec == _SECTION_ION:
            is_ion[i] = True

        # Arrhenius parameters
        cols = rec["num_cols"]
        if sec in (_SECTION_TWO_BODY, _SECTION_THREE_BODY_KINF, _SECTION_THREE_BODY_NO_KINF):
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
        is_radiative=is_radiative,
        is_photo=is_photo,
        is_ion=is_ion,
        stop_rev_indx=stop_rev_indx,
        conden_indx=conden_indx,
        radiative_indx=radiative_indx,
        photo_indx=photo_indx,
        ion_indx=ion_indx,
        photo_sp=tuple(photo_sp),
        pho_rate_index=dict(pho_rate_index),
        n_branch=dict(n_branch),
        ion_sp=tuple(ion_sp),
        ion_rate_index=dict(ion_rate_index),
        ion_branch=dict(ion_branch),
        Rf=dict(Rf_text),
        Rindx=dict(Rindx_map),
        network_path=network_path,
    )


def summarize(net: Network) -> str:
    """Quick human-readable summary."""
    lines = [
        f"Network: {net.network_path}",
        f"  ni = {net.ni} species",
        f"  nr = {net.nr} reaction slots ({net.nr // 2} forward + reverses)",
        f"  three-body reactions: {int(net.is_three_body[1::2].sum())}",
        f"  with k_inf falloff:   {int(net.has_kinf[1::2].sum())}",
        f"  special (hardcoded):  {int(net.is_special[1::2].sum())}",
        f"  condensation:         {int(net.is_conden[1::2].sum())}",
        f"  radiative recomb.:    {int(net.is_radiative[1::2].sum())}",
        f"  photo dissociation:   {int(net.is_photo[1::2].sum())}",
        f"  photo ionization:     {int(net.is_ion[1::2].sum())}",
        f"  stop_rev_indx = {net.stop_rev_indx}",
        f"  photo_indx    = {net.photo_indx}",
        f"  first 10 species: {net.species[:10]}",
        f"  last 5 species:   {net.species[-5:]}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "thermo/SNCHO_photo_network_2025.txt"
    net = parse_network(path)
    print(summarize(net))
