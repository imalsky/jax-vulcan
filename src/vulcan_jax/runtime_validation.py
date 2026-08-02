"""Pre-run configuration validation for the standalone VULCAN-JAX runtime."""

from __future__ import annotations

import math
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

# Upstream VULCAN's own file: Lodders (2009), every element at solar. Selectable
# via `fastchem_solar_abundance_file` so a cross-code comparison can be run on
# matched inputs — the single most common way to get a wrong VULCAN-JAX vs
# VULCAN 2.0 number is to leave the two codes on different composition files
# (~20% median on HD 189733 b; 3.8e-06 once matched). Values are upstream's
# verbatim; only the P/S row ORDER is corrected (see C12).
_UPSTREAM_LODDERS2009_ABUNDANCES = {
    "H": 12.00,
    "He": 10.9864,
    "C": 8.4434,
    "N": 7.9130,
    "O": 8.7826,
    "S": 7.12,
    "P": 5.5058,
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

# The abundance presets a config may select. Anything else is rejected: the
# failure mode is silent (plausible output from a composition nobody chose), so
# an unrecognised file must not pass.
_FASTCHEM_ABUNDANCE_PRESETS = {
    "rocky-suppressed Lodders 2019 (VULCAN-JAX default)": (
        _CANONICAL_FASTCHEM_ABUNDANCES
    ),
    "full-solar Lodders 2009 (upstream VULCAN)": _UPSTREAM_LODDERS2009_ABUNDANCES,
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

    1. Values: the file has been hand-edited into something that is neither
       shipped preset. Both presets are legitimate — rocky-suppressed
       Lodders 2019 (the default) and full-solar Lodders 2009 (upstream's, for
       matched cross-code runs) — but an arbitrary edit is not. Leaving
       Mg / Si / Fe at solar dex sequesters ~19% of O into silicate /
       metal-oxide species the NCHO / SNCHO networks cannot represent, biasing
       initial conditions against `_load_eq_y`; that is a deliberate trade when
       you pick the upstream preset on purpose and a silent bug otherwise.
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

    # The file must be one of the two shipped presets exactly. Both are valid
    # science; a file that is neither is almost always an accidental edit, and
    # its failure mode is silent, so it is rejected rather than warned about.
    per_preset: dict[str, list[str]] = {}
    for preset_name, table in _FASTCHEM_ABUNDANCE_PRESETS.items():
        mismatches: list[str] = []
        for elem, expected in table.items():
            actual = parsed.get(elem)
            if actual is None:
                mismatches.append(f"{elem} missing (expected {expected:+.4f})")
            elif actual != expected:
                mismatches.append(f"{elem}={actual:+.4f} (expected {expected:+.4f})")
        if not mismatches:
            return errors  # exact match against a known preset
        per_preset[preset_name] = mismatches

    closest = min(per_preset, key=lambda k: len(per_preset[k]))
    errors.append(
        f"FastChem input {abun_rel!r} matches neither shipped abundance preset. "
        f"Closest is the {closest} set, which it deviates from in: "
        + "; ".join(per_preset[closest][:8])
        + (
            f" (+{len(per_preset[closest]) - 8} more)"
            if len(per_preset[closest]) > 8
            else ""
        )
        + ". Point `fastchem_solar_abundance_file` at "
        "fastchem_vulcan/input/solar_element_abundances.dat (rocky-suppressed "
        "Lodders 2019, the default) or "
        "fastchem_vulcan/input/solar_element_abundances_lodders2009.dat "
        "(full-solar Lodders 2009, matches upstream VULCAN). A hand-edited file "
        "fails silently: leaving Mg/Si/Fe at solar dex while the network has no "
        "species for them sequesters O into MgO/SiO2/FeO the kinetics cannot "
        "release (the original HD209 atom_loss anomaly)."
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

    # PI step-size controller exponents (only consumed when
    # use_pi_controller=True; alpha=1, beta=0 reproduces I-control).
    pi_a = float(getattr(cfg, "pi_controller_alpha", 0.7))
    if not (0.0 < pi_a <= 1.0):
        errors.append(
            f"pi_controller_alpha={pi_a} must satisfy 0 < pi_controller_alpha <= 1 "
            f"(proportional exponent of the PI step-size controller)."
        )
    pi_b = float(getattr(cfg, "pi_controller_beta", 0.4))
    if not (0.0 <= pi_b <= 1.0):
        errors.append(
            f"pi_controller_beta={pi_b} must satisfy 0 <= pi_controller_beta <= 1 "
            f"(integral/history exponent of the PI step-size controller)."
        )

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

    # Gravity is derived from Mp + Rp (g = G*Mp/Rp^2); both must be positive.
    Mp = getattr(cfg, "Mp", None)
    Rp = float(getattr(cfg, "Rp", 0.0))
    if Mp is None or float(Mp) <= 0.0:
        errors.append(
            f"Mp={Mp} must be set and > 0 (planet mass, g) to derive gs = G*Mp/Rp^2."
        )
    if Rp <= 0.0:
        errors.append(f"Rp={Rp} must be > 0 (planet radius, cm) to derive gs.")

    # High-temperature bottom cut
    if bool(getattr(cfg, "high_temp_cut", False)):
        htc_K = float(getattr(cfg, "high_temp_cut_K", 3500.0))
        htc_P = float(getattr(cfg, "high_temp_cut_P", 1e6))
        if htc_K <= 0.0:
            errors.append(f"high_temp_cut_K={htc_K} must be > 0 (K).")
        if htc_P <= 0.0:
            errors.append(f"high_temp_cut_P={htc_P} must be > 0 (dyne/cm^2).")

    # --- numerical core -----------------------------------------------------
    # These are the knobs a typo makes silently wrong rather than loudly broken.
    # Before this block they were bound-checked by nothing: `nz=1` and an
    # inverted P_b/P_t both ran to completion, the latter producing a
    # negative-thickness atmosphere. Sign/ordering only — no opinion about what
    # a *good* value is, just what cannot be physical.
    nz = int(getattr(cfg, "nz", 150))
    if nz < 3:
        errors.append(
            f"nz={nz} must be >= 3. The containers allocate nz-1 interface and "
            f"nz+1 edge arrays, so nz<=2 is degenerate (nz=1 gives an empty "
            f"interface grid and the run completes on a meaningless column)."
        )
    P_b = float(getattr(cfg, "P_b", 1e9))
    P_t = float(getattr(cfg, "P_t", 1e-2))
    if P_b <= 0.0 or P_t <= 0.0:
        errors.append(f"P_b={P_b:g} and P_t={P_t:g} must both be > 0 (dyne/cm^2).")
    elif P_b <= P_t:
        errors.append(
            f"P_b={P_b:g} must be > P_t={P_t:g}: P_b is the BOTTOM (highest) "
            f"pressure and P_t the top. Inverting them yields a "
            f"negative-thickness atmosphere that still runs to completion."
        )

    # Strictly-positive scalars. Each would produce NaN/garbage rather than an
    # error if left unchecked.
    for key, default, unit in (
        ("atol", 1e-1, "absolute tolerance floor"),
        ("mtol", 1e-22, "mixing-ratio floor"),
        ("mtol_conv", 1e-14, "convergence mixing-ratio floor"),
        ("pos_cut", 0.0, "positive clip threshold"),
        ("dttry", 1e-8, "initial timestep, s"),
        ("dt_min", 1e-14, "minimum timestep, s"),
        ("dt_max", 1e10, "maximum timestep, s"),
        ("yconv_cri", 1e-2, "convergence criterion"),
        ("yconv_min", 1e-1, "loose-branch convergence gate"),
        ("slope_cri", 1e-4, "slope criterion"),
        ("r_star", 1.0, "stellar radius, Rsun"),
        ("orbit_radius", 1.0, "orbital distance, AU"),
    ):
        if not hasattr(cfg, key):
            continue
        v = float(getattr(cfg, key))
        if not math.isfinite(v):
            errors.append(f"{key}={v} must be finite ({unit}).")
        elif v <= 0.0 and key != "pos_cut":
            errors.append(f"{key}={v:g} must be > 0 ({unit}).")
        elif v < 0.0:
            errors.append(f"{key}={v:g} must be >= 0 ({unit}).")

    if hasattr(cfg, "dt_min") and hasattr(cfg, "dt_max"):
        dt_min = float(getattr(cfg, "dt_min"))
        dt_max = float(getattr(cfg, "dt_max"))
        if dt_min > 0 and dt_max > 0 and dt_min >= dt_max:
            errors.append(f"dt_min={dt_min:g} must be < dt_max={dt_max:g}.")

    for key, lo, hi, unit in (
        ("humidity", 0.0, 1.0, "relative humidity fraction"),
        ("nega_cut", -1.0, 0.0, "negative clip threshold (<= 0)"),
    ):
        if not hasattr(cfg, key):
            continue
        v = float(getattr(cfg, key))
        if not (lo <= v <= hi):
            errors.append(f"{key}={v:g} must satisfy {lo} <= {key} <= {hi} ({unit}).")

    if hasattr(cfg, "sl_angle"):
        sl = float(getattr(cfg, "sl_angle"))
        if not (0.0 <= sl < math.pi / 2.0):
            errors.append(
                f"sl_angle={sl:g} must satisfy 0 <= sl_angle < pi/2 (radians); "
                f"at or beyond pi/2 the slant path is infinite."
            )

    # Positive integer counters
    for key, default in (
        ("count_min", 120),
        ("count_max", 10000),
        ("update_frq", 100),
        ("conv_step", 500),
        ("conv_stall_window", 200),
    ):
        if not hasattr(cfg, key):
            continue
        v = int(getattr(cfg, key))
        if v < 1:
            errors.append(f"{key}={v} must be >= 1.")
    # NO count_min-vs-count_max ordering check. `count_min = count_max + 1` is a
    # deliberate, documented idiom for "never satisfy the convergence floor, just
    # run exactly count_max steps" — used by the master-parity harness
    # (tests/test_default_master_parity.py:94,263) and benchmarks/bench_step.py
    # ("deliberately set out of reach so the run terminates on count_max"). An
    # ordering constraint here would reject the repo's own step-matched runs.

    # Wavelength bins for the photolysis grid
    for key in ("dbin1", "dbin2"):
        if not hasattr(cfg, key):
            continue
        v = float(getattr(cfg, key))
        if v <= 0.0:
            errors.append(f"{key}={v:g} must be > 0 (nm).")

    return errors


def _validate_condensation(cfg) -> list[str]:
    """Return errors for an unsupported or silently-inert condensation config.

    Forward-run validation only. These checks were previously only in the
    vulcan-retrieval forward wrapper; lifting them here means a bare VULCAN-JAX
    run (or any consumer) is guarded identically instead of silently producing a
    zero-rate or mid-window-captured state.

    NOTE: the completed *pinned* condensation state is not differentiable-through
    (transient snapshot, discrete phase switches). That contract is enforced at
    the autodiff entry points in `steady_state_grad.py`, not here, and is
    documented in `../../docs/differentiability.md`.
    """
    if not bool(getattr(cfg, "use_condense", False)):
        return []

    from .atm_setup import _SUPPORTED_CONDENSABLES
    from .conden import SUPPORTED_CONDEN_KINETICS

    errors: list[str] = []
    condense_sp = list(getattr(cfg, "condense_sp", []) or [])

    if not condense_sp:
        errors.append(
            "use_condense=True with an empty condense_sp: nothing would "
            "condense. List the condensable gas species, or set "
            "use_condense=False."
        )

    # The condensation growth coefficient Dg IS the species' molecular-diffusion
    # coefficient (Dzz), which atm setup zeros when use_moldiff=False
    # (atm_setup.compute_mol_diff / atm_jax.build_atm_static), so every
    # condensation rate would silently be zero. Refuse rather than run inert.
    if not bool(getattr(cfg, "use_moldiff", True)):
        errors.append(
            "use_condense=True requires use_moldiff=True: the condensation "
            "growth coefficient Dg is the molecular-diffusion coefficient Dzz, "
            "which is zeroed when use_moldiff=False, so every condensation rate "
            "would silently be zero."
        )

    start_ct = float(getattr(cfg, "start_conden_time", 0.0))
    stop_ct = float(getattr(cfg, "stop_conden_time", 0.0))
    if stop_ct < start_ct:
        errors.append(
            f"stop_conden_time={stop_ct} < start_conden_time={start_ct}: the "
            "condensation window would never open."
        )

    # Support tier: kinetics-ported condensates vs saturation-only (H2S).
    sat_only = sorted(set(_SUPPORTED_CONDENSABLES) - set(SUPPORTED_CONDEN_KINETICS))
    for sp in condense_sp:
        if sp not in _SUPPORTED_CONDENSABLES:
            errors.append(
                f"condense_sp entry {sp!r} is not a supported condensate. "
                f"Condensation kinetics are ported for "
                f"{sorted(SUPPORTED_CONDEN_KINETICS)}; {sat_only} have "
                "saturation data only (capping/cold-trap, no conden reactions "
                "— as in VULCAN-master). New condensates must be added "
                "explicitly with physical constants and tests."
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

    ode_solver = getattr(cfg, "ode_solver", None)
    if ode_solver != "Ros2":
        errors.append(
            f"ode_solver={ode_solver!r} is unsupported; VULCAN-JAX only supports 'Ros2'."
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

    fix_sp = list(getattr(cfg, "fix_species", []) or [])
    if fix_sp and not bool(getattr(cfg, "use_condense", False)):
        errors.append(
            "fix_species is set but use_condense=False; this configuration is inconsistent."
        )
    if fix_sp:
        # Without this, an entry absent from the import-locked network reaches
        # state.py's condensation setup and dies on a bare `.index()` ValueError
        # that names neither the offending species nor the remedy.
        net_species = list(getattr(chem_funs, "spec_list", []))
        if net_species:
            missing = [sp for sp in fix_sp if sp not in net_species]
            if missing:
                errors.append(
                    f"fix_species entries not in the loaded network: {missing}. "
                    f"The network is import-locked, so this cannot be fixed after "
                    f"`import vulcan_jax` — either drop these entries or set "
                    f"$VULCAN_JAX_NETWORK to a network containing them before the "
                    f"first import."
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

    errors.extend(_validate_condensation(cfg))

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
