"""Pre-run configuration validation for the standalone VULCAN-JAX runtime."""

from __future__ import annotations

from pathlib import Path

from . import chem_funs


# Canonical FastChem element abundances (Lodders 2019 / Wogan & Tsai 2023).
# Rocky elements are pinned to -3.0 because no shipped network contains
# Mg / Si / Fe / Ti / V / Cl / K / Na / F / Ca / P species — leaving them at
# solar dex would silently sequester ~19% of O into MgO / SiO2 / FeO that
# `_load_eq_y` cannot read back. S is kept at solar (matched to master) because
# the SNCHO network for W39b includes S species. The NCHO networks (HD189 /
# HD209 / Earth) accept the resulting ~1% S→O sequestration as a known small
# bias matching master.
_CANONICAL_FASTCHEM_ABUNDANCES = {
    "H": 12.00,
    "He": 10.9232,
    "C": 8.4434,
    "N": 7.9130,
    "O": 8.7826,
    "S": 7.1492,
    "P": -3.0,
    "Si": -3.0,
    "Ti": -3.0,
    "V": -3.0,
    "Cl": -3.0,
    "K": -3.0,
    "Na": -3.0,
    "Mg": -3.0,
    "F": -3.0,
    "Ca": -3.0,
    "Fe": -3.0,
}


# Element row order REQUIRED by the vendored FastChem. Its
# fastchem_src/mass_action_constant.cpp subtracts per-element NASA9 reference
# polynomials by HARD-CODED slot index (index_C=0, index_H=1, index_He=2,
# index_N=3, index_O=4, index_P=5, index_S=6, index_Si=7, index_Ti=8,
# index_V=9, index_Cl=10, index_Km=11 [=K], index_Na=12, index_Mg=13,
# index_F=14, index_Ca=15, index_Fe=16, index_e=17). The element vector is built
# in abundance-file row order, so the file MUST list elements in exactly this
# order. A reorder (e.g. H,He,C,...) makes carbon take helium's reference and CO
# never forms — silent, no crash. `tests/test_fastchem_element_order.py` parses
# the C++ and asserts this list still matches the hard-coded indices.
_FASTCHEM_ELEMENT_ORDER = [
    "C",
    "H",
    "He",
    "N",
    "O",
    "P",
    "S",
    "Si",
    "Ti",
    "V",
    "Cl",
    "K",
    "Na",
    "Mg",
    "F",
    "Ca",
    "Fe",
    "e-",
]


def _validate_fastchem_input_vs_network(cfg, root: Path) -> list[str]:
    """Pin `fastchem_solar_abundance_file` content AND element row order.

    Two independent failure modes, both silent (no crash, plausible-looking
    output):

    1. Values: someone (re-)installs the historical Lodders-2009 file with
       Mg / Si / Fe / etc. at solar dex. That sequesters ~19% of O into
       silicate / metal-oxide species the NCHO / SNCHO networks cannot
       represent, biasing initial conditions against `_load_eq_y`.
    2. Order: someone reorders the rows (e.g. H,He,C,... instead of
       C,H,He,...). FastChem's mass_action_constant.cpp subtracts per-element
       NASA9 references by hard-coded slot, so a reorder makes carbon take
       helium's reference -> CO/CH4/CO2 never form, all carbon stays atomic.
       See `_FASTCHEM_ELEMENT_ORDER`.

    Both checks are content-based (parsed values / parsed symbol order), not a
    byte hash, so whitespace / EOL / decimal-format differences don't trip
    false positives.
    """
    errors: list[str] = []
    if getattr(cfg, "ini_mix", None) != "EQ":
        return errors

    abun_rel = getattr(
        cfg,
        "fastchem_solar_abundance_file",
        "fastchem_vulcan/input/solar_element_abundances.dat",
    )
    abun_path = root / abun_rel
    if not abun_path.exists():
        return errors  # missing-file error captured by caller

    parsed: dict[str, float] = {}
    order: list[str] = []
    with open(abun_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                parsed[parts[0]] = float(parts[1])
            except ValueError:
                continue
            order.append(parts[0])

    # Element row order is load-bearing (FastChem hard-codes element slots) AND
    # must be COMPLETE. A truncated file (missing trailing rows) would satisfy a
    # prefix check yet leave later C++ slots (P, S, the rocky elements) pointing
    # at the wrong reference polynomial. Require the exact canonical order; the
    # trailing electron row is optional (absent when use_ion is False).
    if order not in (_FASTCHEM_ELEMENT_ORDER, _FASTCHEM_ELEMENT_ORDER[:-1]):
        errors.append(
            f"FastChem input {abun_rel!r} element ROW ORDER is wrong: got "
            f"{order} but FastChem's mass_action_constant.cpp hard-codes the "
            f"full order {_FASTCHEM_ELEMENT_ORDER}. A reorder OR a missing "
            "element makes carbon (or P/S/rocky) take the wrong reference "
            "polynomial -> the affected molecules never form (e.g. carbon stays "
            "atomic, no CO). Restore the canonical C,H,He,N,O,P,S,... order in "
            "full."
        )

    mismatches: list[str] = []
    for elem, expected in _CANONICAL_FASTCHEM_ABUNDANCES.items():
        actual = parsed.get(elem)
        if actual is None:
            mismatches.append(f"{elem} missing (expected {expected:+.4f})")
        elif actual != expected:
            mismatches.append(f"{elem}={actual:+.4f} (expected {expected:+.4f})")

    if mismatches:
        errors.append(
            f"FastChem input {abun_rel!r} deviates from the canonical "
            f"rocky-suppressed abundances: "
            + "; ".join(mismatches)
            + ". Restore the file to its shipped contents — leaving Mg/Si/Fe "
            "at solar dex sequesters O into species the network cannot "
            "represent (the original HD209 atom_loss anomaly)."
        )
    return errors


def _validate_network_assets(cfg, root: Path) -> list[str]:
    """Return a list of human-readable errors for missing network-linked assets.

    Checks three things that produce cryptic downstream errors if wrong:
    1. Every network species appears in the composition table (all_compose.txt).
    2. Every photodissociation species has a cross-section directory.
    3. cfg.atom_list entries are recognised column headers in all_compose.txt.
    """
    errors: list[str] = []

    com_path = root / getattr(cfg, "com_file", "")
    if not com_path.exists():
        return errors  # file-existence error already captured by caller

    # Read species column from composition table.
    with open(com_path) as f:
        header = f.readline().split()
        compose_species = {line.split()[0] for line in f if line.strip()}
    compose_atoms = set(header[1:])  # all element column names (excluding 'species')

    # 1. Every network species must be in the composition table.
    net_name = getattr(cfg, "network", "")
    for sp in chem_funs.spec_list:
        if sp not in compose_species:
            errors.append(
                f"Species {sp!r} from network {net_name!r} is missing from "
                f"{cfg.com_file}. Add a row for {sp!r} to the composition table."
            )

    # 2. Photo species need a cross-section directory.
    if bool(getattr(cfg, "use_photo", False)):
        cross_folder = root / getattr(cfg, "cross_folder", "")
        for sp in chem_funs._NETWORK.photo_sp:
            cross_csv = cross_folder / sp / f"{sp}_cross.csv"
            if not cross_csv.exists():
                errors.append(
                    f"Photo species {sp!r} has no cross-section file at "
                    f"{cfg.cross_folder}{sp}/{sp}_cross.csv. "
                    f"Add the file or remove {sp!r} from photo reactions."
                )

    # 3. Warn about atom_list entries not in the composition table header.
    for atom in getattr(cfg, "atom_list", []):
        if atom not in compose_atoms:
            errors.append(
                f"atom_list entry {atom!r} is not a column in {cfg.com_file} "
                f"(available: {sorted(compose_atoms)}). "
                f"Update atom_list or add a column to the composition table."
            )

    return errors


def _validate_numerical_bounds(cfg) -> list[str]:
    """Return a list of human-readable errors for out-of-range numerical knobs.

    Catches typos like `adapt_rtol_dec=1.25` (would diverge) or
    `batch_max_retries=0` (would deadlock the JIT'd loop) at validation
    time instead of letting the runner silently misbehave.
    """
    errors: list[str] = []

    # Adaptive rtol controller
    use_adapt = bool(getattr(cfg, "use_adapt_rtol", False))
    adapt_rtol_dec = float(getattr(cfg, "adapt_rtol_dec", 0.75))
    adapt_rtol_inc = float(getattr(cfg, "adapt_rtol_inc", 1.25))
    if not (0.0 < adapt_rtol_dec < 1.0):
        errors.append(
            f"adapt_rtol_dec={adapt_rtol_dec} must satisfy 0 < adapt_rtol_dec < 1."
        )
    if not (adapt_rtol_inc > 1.0):
        errors.append(
            f"adapt_rtol_inc={adapt_rtol_inc} must satisfy adapt_rtol_inc > 1."
        )
    for key in ("adapt_rtol_dec_period", "adapt_rtol_inc_period"):
        v = int(getattr(cfg, key, 1))
        if v < 1:
            errors.append(f"{key}={v} must be >= 1.")
    adapt_loss_mul = float(getattr(cfg, "adapt_rtol_loss_mul", 2.0))
    if adapt_loss_mul <= 1.0:
        errors.append(
            f"adapt_rtol_loss_mul={adapt_loss_mul} must be > 1 (it relaxes the loss criterion)."
        )
    inc_loss_thresh = float(getattr(cfg, "adapt_rtol_inc_loss_thresh", 2e-4))
    if inc_loss_thresh <= 0.0:
        errors.append(f"adapt_rtol_inc_loss_thresh={inc_loss_thresh} must be > 0.")
    loss_eps = float(getattr(cfg, "loss_eps", 1e-1))
    if use_adapt and inc_loss_thresh >= loss_eps:
        errors.append(
            f"adapt_rtol_inc_loss_thresh={inc_loss_thresh} must be < loss_eps={loss_eps} "
            f"(otherwise rtol increases never gate on loss)."
        )

    # rtol bounds
    rtol = float(getattr(cfg, "rtol", 0.2))
    rtol_min = float(getattr(cfg, "rtol_min", 0.0))
    rtol_max = float(getattr(cfg, "rtol_max", 1.0))
    if not (rtol_min <= rtol_max):
        errors.append(f"rtol_min={rtol_min} must be <= rtol_max={rtol_max}.")
    if use_adapt and not (rtol_min <= rtol <= rtol_max):
        errors.append(
            f"rtol={rtol} must lie in [rtol_min={rtol_min}, rtol_max={rtol_max}] "
            f"when use_adapt_rtol=True."
        )

    # Per-step retry cap
    bmr = int(getattr(cfg, "batch_max_retries", 64))
    if bmr < 1:
        errors.append(f"batch_max_retries={bmr} must be >= 1.")

    # Step-size knobs
    safety = float(getattr(cfg, "step_size_safety", 0.9))
    if not (0.0 < safety < 1.0):
        errors.append(
            f"step_size_safety={safety} must satisfy 0 < step_size_safety < 1 "
            f"(safety factor on the Ros2 step prediction)."
        )
    zdf = float(getattr(cfg, "step_size_zero_delta_frac", 0.01))
    if zdf <= 0.0:
        errors.append(f"step_size_zero_delta_frac={zdf} must be > 0.")

    # Photo-frequency switch
    for key in ("photo_switch_longdy_thresh", "photo_switch_longdydt_thresh"):
        v = float(getattr(cfg, key, 1.0))
        if v <= 0.0:
            errors.append(f"{key}={v} must be > 0.")

    # Hycean pin time
    hpt = float(getattr(cfg, "hycean_pin_time", 1e6))
    if hpt <= 0.0:
        errors.append(f"hycean_pin_time={hpt} must be > 0.")

    # FastChem Newton solver (only checked when ini_mix in {EQ, const_lowT})
    ini_mix = getattr(cfg, "ini_mix", None)
    if ini_mix in ("EQ", "const_lowT"):
        max_iter = int(getattr(cfg, "fastchem_newton_max_iter", 50))
        if max_iter < 1:
            errors.append(f"fastchem_newton_max_iter={max_iter} must be >= 1.")
        tol = float(getattr(cfg, "fastchem_newton_tol", 1e-12))
        if tol <= 0.0:
            errors.append(f"fastchem_newton_tol={tol} must be > 0.")

    # loss_ex must be a subset of atom_list
    atom_list = list(getattr(cfg, "atom_list", []))
    loss_ex = list(getattr(cfg, "loss_ex", []))
    extras = [a for a in loss_ex if a not in atom_list]
    if extras:
        errors.append(
            f"loss_ex={loss_ex} contains entries not in atom_list={atom_list}: {extras}."
        )

    return errors


def validate_runtime_config(cfg, root: Path | None = None) -> None:
    """Raise RuntimeError if cfg is unsupported or required files are missing.

    Aggregates every configuration error (solver/flag consistency, required
    files, network assets, FastChem input, numerical bounds) and raises once
    so the user sees all problems at once; returns None on success. `root`
    sets where relative asset paths resolve (defaults to the package dir).
    """
    root = Path(__file__).resolve().parent if root is None else Path(root)
    errors: list[str] = []

    if getattr(cfg, "ode_solver", None) != "Ros2":
        errors.append(
            f"ode_solver={cfg.ode_solver!r} is unsupported; VULCAN-JAX only supports 'Ros2'."
        )

    if bool(getattr(cfg, "use_live_flux", False)) and not bool(
        getattr(cfg, "use_photo", False)
    ):
        errors.append(
            "use_live_flux=True requires use_photo=True (no diffuse fluxes without photochemistry)."
        )

    if bool(getattr(cfg, "use_ion", False)) and not bool(
        getattr(cfg, "use_photo", False)
    ):
        errors.append("use_ion=True requires use_photo=True in VULCAN-JAX.")

    if bool(getattr(cfg, "fix_species", [])) and not bool(
        getattr(cfg, "use_condense", False)
    ):
        errors.append(
            "fix_species is set but use_condense=False; this configuration is inconsistent."
        )

    if bool(getattr(cfg, "use_fix_H2He", False)):
        species = list(getattr(chem_funs, "spec_list", []))
        for sp in ("H2", "He"):
            if sp not in species:
                errors.append(
                    f"use_fix_H2He=True requires {sp!r} in the network; "
                    f"got species list without it."
                )

    if getattr(cfg, "ini_mix", None) == "const_mix":
        species = list(getattr(chem_funs, "spec_list", []))
        for sp in getattr(cfg, "const_mix", {}):
            if sp not in species:
                errors.append(
                    f"const_mix key {sp!r} is not a species of the loaded "
                    "network. Inert/background gases without network "
                    "reactions are not supported — VULCAN-master fails on "
                    "the same configuration (build_atm.ini_y calls "
                    "species.index(sp) unconditionally, e.g. its shipped "
                    "Earth example crashes on 'Ar'). Remove the key or use "
                    "a network that contains the species."
                )

    if bool(getattr(cfg, "use_condense", False)):
        # _SUPPORTED_CONDENSABLES is the saturation-formula tier (superset);
        # SUPPORTED_CONDEN_KINETICS is the conden-reaction tier.
        from .atm_setup import _SUPPORTED_CONDENSABLES
        from .conden import SUPPORTED_CONDEN_KINETICS

        sat_only = sorted(set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS))
        for sp in getattr(cfg, "condense_sp", []):
            if sp not in _SUPPORTED_CONDENSABLES:
                errors.append(
                    f"condense_sp entry {sp!r} is not a supported condensate. "
                    f"Condensation kinetics are ported for "
                    f"{sorted(SUPPORTED_CONDEN_KINETICS)}; {sat_only} have "
                    "saturation data only (capping/cold-trap, no conden "
                    "reactions — as in VULCAN-master). New condensates must "
                    "be added explicitly with physical constants and tests."
                )

    required_paths = [
        ("network", getattr(cfg, "network", None)),
        ("gibbs_text", getattr(cfg, "gibbs_text", None)),
        ("com_file", getattr(cfg, "com_file", None)),
        ("atm_file", getattr(cfg, "atm_file", None)),
    ]
    if bool(getattr(cfg, "use_photo", False)):
        required_paths.extend(
            [
                ("cross_folder", getattr(cfg, "cross_folder", None)),
                ("sflux_file", getattr(cfg, "sflux_file", None)),
            ]
        )
    if bool(getattr(cfg, "use_topflux", False)):
        required_paths.append(
            ("top_BC_flux_file", getattr(cfg, "top_BC_flux_file", None))
        )
    if bool(getattr(cfg, "use_botflux", False)):
        required_paths.append(
            ("bot_BC_flux_file", getattr(cfg, "bot_BC_flux_file", None))
        )
    if getattr(cfg, "ini_mix", None) == "EQ":
        required_paths.append(
            (
                "fastchem_solar_abundance_file",
                getattr(
                    cfg,
                    "fastchem_solar_abundance_file",
                    "fastchem_vulcan/input/solar_element_abundances.dat",
                ),
            )
        )

    for label, rel_path in required_paths:
        if not rel_path:
            errors.append(f"{label} is unset.")
            continue
        if not (root / rel_path).exists():
            errors.append(f"{label}={rel_path!r} does not exist under {root}.")

    errors.extend(_validate_network_assets(cfg, root))
    errors.extend(_validate_fastchem_input_vs_network(cfg, root))
    errors.extend(_validate_numerical_bounds(cfg))

    if errors:
        raise RuntimeError(
            "Unsupported or invalid VULCAN-JAX runtime configuration:\n- "
            + "\n- ".join(errors)
        )
