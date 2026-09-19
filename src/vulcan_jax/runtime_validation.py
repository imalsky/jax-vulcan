"""Pre-run configuration validation for the standalone VULCAN-JAX runtime."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import chem_funs


# The validator bound-checks knobs a config DECLARES; it never supplies one.
# A literal default here would be a second copy of a number the YAML owns --
# and outer_loop.py already owns the runtime back-compat fallback -- so the
# two could drift and this module would then validate a value the run never
# uses. `_declared` returns None for an undeclared knob and the caller skips
# it. `tests/test_runtime_validation_knobs.py` pins that every shipped config
# declares every knob checked here, so the skip cannot hide anything in
# practice.
_ABSENT = object()


def _declared(cfg, key, cast=float):
    """Cast a declared knob, or None when ``cfg`` does not declare it."""
    v = getattr(cfg, key, _ABSENT)
    return None if v is _ABSENT else cast(v)


def _validate_abundance_preset(cfg, root: Path) -> list[str]:
    """The EQ seed's abundance preset must parse and cover the network.

    The file is read as `SYMBOL log10(n_X/n_H)+12` rows. The seed needs a
    finite, positive ratio for hydrogen and for every element the loaded
    network contains; a truncated or hand-mangled file otherwise drops an
    element silently (its species then cannot form at all).
    """
    if getattr(cfg, "ini_mix", None) != "EQ":
        return []
    from . import ini_abun

    rel = getattr(
        cfg, "fastchem_solar_abundance_file", ini_abun.DEFAULT_ABUNDANCE_FILE
    )
    path = root / rel
    if not path.exists():
        return []  # missing-file error captured by caller
    try:
        ratios = ini_abun.read_abundances(path)
    except (OSError, ValueError) as exc:
        return [f"abundance preset {rel!r} does not parse: {exc}"]
    bad = [
        sp
        for sp in ini_abun.SEED_ELEMENTS
        if not math.isfinite(ratios.get(sp, math.nan)) or ratios.get(sp, 0.0) <= 0.0
    ]
    if bad:
        message = (
            f"abundance preset {rel!r} has no finite, positive abundance for "
            f"{', '.join(bad)}, which the {getattr(cfg, 'network', '?')!r} "
            "network contains. Every element of the loaded network must have "
            "a row in the preset."
        )
        return [message]
    return []


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
    adapt_rtol_dec = _declared(cfg, "adapt_rtol_dec")
    adapt_rtol_inc = _declared(cfg, "adapt_rtol_inc")
    if adapt_rtol_dec is not None and not (0.0 < adapt_rtol_dec < 1.0):
        errors.append(
            f"adapt_rtol_dec={adapt_rtol_dec} must satisfy 0 < adapt_rtol_dec < 1."
        )
    if adapt_rtol_inc is not None and not (adapt_rtol_inc > 1.0):
        errors.append(
            f"adapt_rtol_inc={adapt_rtol_inc} must satisfy adapt_rtol_inc > 1."
        )
    for key in ("adapt_rtol_dec_period", "adapt_rtol_inc_period"):
        v = _declared(cfg, key, int)
        if v is not None and v < 1:
            errors.append(f"{key}={v} must be >= 1.")
    adapt_loss_mul = _declared(cfg, "adapt_rtol_loss_mul")
    if adapt_loss_mul is not None and adapt_loss_mul <= 1.0:
        errors.append(
            f"adapt_rtol_loss_mul={adapt_loss_mul} must be > 1 (it relaxes the loss criterion)."
        )
    inc_loss_thresh = _declared(cfg, "adapt_rtol_inc_loss_thresh")
    if inc_loss_thresh is not None and inc_loss_thresh <= 0.0:
        errors.append(f"adapt_rtol_inc_loss_thresh={inc_loss_thresh} must be > 0.")
    loss_eps = _declared(cfg, "loss_eps")
    if use_adapt and None not in (inc_loss_thresh, loss_eps) \
            and inc_loss_thresh >= loss_eps:
        errors.append(
            f"adapt_rtol_inc_loss_thresh={inc_loss_thresh} must be < loss_eps={loss_eps} "
            f"(otherwise rtol increases never gate on loss)."
        )

    # rtol bounds
    rtol = _declared(cfg, "rtol")
    rtol_min = _declared(cfg, "rtol_min")
    rtol_max = _declared(cfg, "rtol_max")
    if None not in (rtol_min, rtol_max) and not (rtol_min <= rtol_max):
        errors.append(f"rtol_min={rtol_min} must be <= rtol_max={rtol_max}.")
    if use_adapt and None not in (rtol, rtol_min, rtol_max) \
            and not (rtol_min <= rtol <= rtol_max):
        errors.append(
            f"rtol={rtol} must lie in [rtol_min={rtol_min}, rtol_max={rtol_max}] "
            f"when use_adapt_rtol=True."
        )

    # Per-step retry cap
    bmr = _declared(cfg, "batch_max_retries", int)
    if bmr is not None and bmr < 1:
        errors.append(f"batch_max_retries={bmr} must be >= 1.")

    # Step-size knobs
    safety = _declared(cfg, "step_size_safety")
    if safety is not None and not (0.0 < safety < 1.0):
        errors.append(
            f"step_size_safety={safety} must satisfy 0 < step_size_safety < 1 "
            f"(safety factor on the Ros2 step prediction)."
        )
    zdf = _declared(cfg, "step_size_zero_delta_frac")
    if zdf is not None and zdf <= 0.0:
        errors.append(f"step_size_zero_delta_frac={zdf} must be > 0.")

    # Photo-frequency switch
    for key in ("photo_switch_longdy_thresh", "photo_switch_longdydt_thresh"):
        v = _declared(cfg, key)
        if v is not None and v <= 0.0:
            errors.append(f"{key}={v} must be > 0.")

    # Hycean pin time
    hpt = _declared(cfg, "hycean_pin_time")
    if hpt is not None and hpt <= 0.0:
        errors.append(f"hycean_pin_time={hpt} must be > 0.")

    # Equilibrium-seed / const_lowT Newton controls (shared knob names)
    ini_mix = getattr(cfg, "ini_mix", None)
    if ini_mix in ("EQ", "const_lowT"):
        max_iter = _declared(cfg, "fastchem_newton_max_iter", int)
        if max_iter is not None and max_iter < 1:
            errors.append(f"fastchem_newton_max_iter={max_iter} must be >= 1.")
        tol = _declared(cfg, "fastchem_newton_tol")
        if tol is not None and tol <= 0.0:
            errors.append(f"fastchem_newton_tol={tol} must be > 0.")

    # loss_ex must be a subset of atom_list
    atom_list = list(getattr(cfg, "atom_list", []))
    loss_ex = list(getattr(cfg, "loss_ex", []))
    if "H" in loss_ex:
        errors.append("loss_ex must not contain H: the element budget is measured relative to H.")
    extras = [a for a in loss_ex if a not in atom_list]
    if extras:
        errors.append(
            f"loss_ex={loss_ex} contains entries not in atom_list={atom_list}: {extras}."
        )

    # Gravity is derived from Mp + Rp (g = G*Mp/Rp^2); both must be positive.
    Mp = _declared(cfg, "Mp")
    Rp = _declared(cfg, "Rp")
    if Mp is None or Mp <= 0.0:
        errors.append(
            f"Mp={'not declared' if Mp is None else Mp} must be set and > 0 "
            f"(planet mass, g) to derive gs = G*Mp/Rp^2."
        )
    if Rp is None or Rp <= 0.0:
        errors.append(
            f"Rp={'not declared' if Rp is None else Rp} must be set and > 0 "
            f"(planet radius, cm) to derive gs."
        )

    # High-temperature bottom cut
    if bool(getattr(cfg, "high_temp_cut", False)):
        htc_K = _declared(cfg, "high_temp_cut_K")
        htc_P = _declared(cfg, "high_temp_cut_P")
        if htc_K is not None and htc_K <= 0.0:
            errors.append(f"high_temp_cut_K={htc_K} must be > 0 (K).")
        if htc_P is not None and htc_P <= 0.0:
            errors.append(f"high_temp_cut_P={htc_P} must be > 0 (dyne/cm^2).")

    # --- numerical core -----------------------------------------------------
    # Knobs a typo makes silently wrong rather than loudly broken (nz=1 and an
    # inverted P_b/P_t both ran to completion). Sign/ordering checks only; no
    # opinion about what a good value is.
    nz = _declared(cfg, "nz", int)
    if nz is None:
        errors.append("nz is not declared; every config must set the layer count.")
    elif nz < 3:
        errors.append(
            f"nz={nz} must be >= 3. The containers allocate nz-1 interface and "
            f"nz+1 edge arrays, so nz<=2 is degenerate (nz=1 gives an empty "
            f"interface grid and the run completes on a meaningless column)."
        )
    P_b = _declared(cfg, "P_b")
    P_t = _declared(cfg, "P_t")
    if P_b is None or P_t is None:
        errors.append(
            "P_b and P_t must both be declared (bottom and top pressure, "
            "dyne/cm^2); the grid cannot be built without them."
        )
    elif P_b <= 0.0 or P_t <= 0.0:
        errors.append(f"P_b={P_b:g} and P_t={P_t:g} must both be > 0 (dyne/cm^2).")
    elif P_b <= P_t:
        errors.append(
            f"P_b={P_b:g} must be > P_t={P_t:g}: P_b is the BOTTOM (highest) "
            f"pressure and P_t the top. Inverting them yields a "
            f"negative-thickness atmosphere that still runs to completion."
        )

    # Strictly-positive scalars. Each would produce NaN/garbage rather than an
    # error if left unchecked.
    for key, unit in (
        ("atol", "absolute tolerance floor"),
        ("mtol", "mixing-ratio floor"),
        ("mtol_conv", "convergence mixing-ratio floor"),
        ("pos_cut", "positive clip threshold"),
        ("dttry", "initial timestep, s"),
        ("dt_min", "minimum timestep, s"),
        ("dt_max", "maximum timestep, s"),
        ("yconv_cri", "convergence criterion"),
        ("yconv_min", "loose-branch convergence gate"),
        ("slope_cri", "slope criterion"),
        ("geom_conv_tol", "geometry certificate tolerance"),
        ("element_budget_tol", "column element-budget certificate tolerance"),
        ("r_star", "stellar radius, Rsun"),
        ("orbit_radius", "orbital distance, AU"),
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
    for key in ("count_min", "count_max", "update_frq", "conv_step",
                "conv_stall_window"):
        if not hasattr(cfg, key):
            continue
        v = int(getattr(cfg, key))
        if v < 1:
            errors.append(f"{key}={v} must be >= 1.")
    # NO count_min-vs-count_max ordering check: `count_min = count_max + 1` is
    # a deliberate idiom ("run exactly count_max steps") used by the parity
    # harness and benchmarks; an ordering constraint would reject those runs.

    # The refresh counter is int32 and used as a modulo divisor in the runner.
    for key in ("ini_update_photo_frq", "final_update_photo_frq"):
        if not hasattr(cfg, key):
            continue
        v = float(getattr(cfg, key))
        if not (math.isfinite(v) and v.is_integer() and 1 <= v < 2**31):
            errors.append(f"{key}={v:g} must be a positive int32 integer.")

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

    Forward-run validation only, applied to every consumer identically.

    NOTE: the completed *pinned* condensation state is not
    differentiable-through (transient snapshot, discrete phase switches); that
    contract is enforced at the autodiff entry points in
    `steady_state_grad.py`, not here (see README.md, Limits).
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

    # The condensation growth coefficient Dg IS the molecular-diffusion Dzz,
    # which setup zeros when use_moldiff=False, so every condensation rate
    # would silently be zero. Refuse rather than run inert.
    if not bool(getattr(cfg, "use_moldiff", True)):
        errors.append(
            "use_condense=True requires use_moldiff=True: the condensation "
            "growth coefficient Dg is the molecular-diffusion coefficient Dzz, "
            "which is zeroed when use_moldiff=False, so every condensation rate "
            "would silently be zero."
        )

    start_ct = _declared(cfg, "start_conden_time")
    stop_ct = _declared(cfg, "stop_conden_time")
    if None not in (start_ct, stop_ct) and stop_ct < start_ct:
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
    files, network assets, abundance preset, numerical bounds) and raises once
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
        # Otherwise an entry absent from the import-locked network dies later
        # on a bare `.index()` ValueError naming neither species nor remedy.
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
                    "thermo/solar_element_abundances.dat",
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
    errors.extend(_validate_abundance_preset(cfg, root))
    errors.extend(_validate_numerical_bounds(cfg))

    if errors:
        raise RuntimeError(
            "Unsupported or invalid VULCAN-JAX runtime configuration:\n- "
            + "\n- ".join(errors)
        )


# One report per (network, T-grid signature, exposure) per process: repeated
# rebuilds of the SAME case stay quiet, but a different profile on the same
# network reports again — the exposure depends on Tco, not just the network.
# (vulcan-forward builds the RunState once per model; per-proposal T varies
# on-graph, so this cannot spam a retrieval log.)
_TEMP_RANGE_REPORTED: set[tuple] = set()


def rate_temp_range_exposure(net, Tco) -> dict:
    """Count thermal rows evaluated outside their documented T ranges on `Tco`.

    Coverage per row is the UNION of its documented ranges (lenient: a
    Lindemann row's k0 and k_inf windows are OR'd, so this undercounts).
    A bare measurement temperature parses as a degenerate (T, T) range, so
    room-temperature-only rates count as out-of-range on hot profiles
    rather than hiding in the no-range bucket.
    """
    T = np.asarray(Tco, dtype=np.float64)
    no_range = any_outside = all_outside = 0
    for ranges in (net.temp_ranges or {}).values():
        if not ranges:
            no_range += 1
            continue
        covered = np.zeros(T.shape, dtype=bool)
        for lo, hi in ranges:
            covered |= (T >= lo) & (T <= hi)
        n_out = int(np.count_nonzero(~covered))
        if n_out:
            any_outside += 1
            if n_out == T.size:
                all_outside += 1
    return {
        "thermal_rows": len(net.temp_ranges or {}),
        "no_range": no_range,
        "any_outside": any_outside,
        "all_outside": all_outside,
    }


def report_rate_temp_ranges(net, Tco) -> None:
    """Print a one-time advisory on rate-law T-range exposure for this run.

    The network files carry per-row documented temperature ranges (the free-
    text `Temp` column) that neither VULCAN implementation enforces, so hot
    or cold profiles silently extrapolate many fitted rates. Say so once per
    distinct (network, T grid, exposure), against the ACTUAL model T grid.
    Advisory only — no rate is altered or gated (VULCAN-master evaluates
    every rate everywhere, and parity requires the same here).
    """
    if net.temp_ranges is None:
        return
    T = np.asarray(Tco, dtype=np.float64)
    ex = rate_temp_range_exposure(net, T)
    if not (ex["any_outside"] or ex["no_range"]):
        return
    key = (
        str(net.network_path),
        int(T.size),
        round(float(T.min()), 1),
        round(float(T.max()), 1),
        ex["any_outside"],
        ex["all_outside"],
        ex["no_range"],
    )
    if key in _TEMP_RANGE_REPORTED:
        return
    _TEMP_RANGE_REPORTED.add(key)
    print(
        f"rate T-range advisory ({Path(net.network_path).name}): model grid "
        f"spans {T.min():.0f}-{T.max():.0f} K over {T.size} layers; of "
        f"{ex['thermal_rows']} thermal rows, {ex['any_outside']} are evaluated "
        f"outside their documented T range in >=1 layer ({ex['all_outside']} in "
        f"ALL layers) and {ex['no_range']} carry no parseable range. Advisory "
        "only: no rate is altered (matches VULCAN-master)."
    )
