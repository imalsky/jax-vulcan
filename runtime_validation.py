"""Pre-run configuration validation for the standalone VULCAN-JAX runtime."""

from __future__ import annotations

from pathlib import Path

import chem_funs


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
        errors.append(
            f"adapt_rtol_inc_loss_thresh={inc_loss_thresh} must be > 0."
        )
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
        errors.append(
            f"step_size_zero_delta_frac={zdf} must be > 0."
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

    return errors


def validate_runtime_config(cfg, root: Path | None = None) -> None:
    """Raise RuntimeError if cfg is unsupported or required files are missing."""
    root = Path(__file__).resolve().parent if root is None else Path(root)
    errors: list[str] = []

    if getattr(cfg, "ode_solver", None) != "Ros2":
        errors.append(
            f"ode_solver={cfg.ode_solver!r} is unsupported; VULCAN-JAX only supports 'Ros2'."
        )

    if bool(getattr(cfg, "use_live_flux", False)) and not bool(getattr(cfg, "use_photo", False)):
        errors.append("use_live_flux=True requires use_photo=True (no diffuse fluxes without photochemistry).")

    if bool(getattr(cfg, "use_ion", False)) and not bool(getattr(cfg, "use_photo", False)):
        errors.append("use_ion=True requires use_photo=True in VULCAN-JAX.")

    if bool(getattr(cfg, "fix_species", [])) and not bool(getattr(cfg, "use_condense", False)):
        errors.append("fix_species is set but use_condense=False; this configuration is inconsistent.")

    if bool(getattr(cfg, "use_fix_H2He", False)):
        species = list(getattr(chem_funs, "spec_list", []))
        for sp in ("H2", "He"):
            if sp not in species:
                errors.append(
                    f"use_fix_H2He=True requires {sp!r} in the network; "
                    f"got species list without it."
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
        required_paths.append(("top_BC_flux_file", getattr(cfg, "top_BC_flux_file", None)))
    if bool(getattr(cfg, "use_botflux", False)):
        required_paths.append(("bot_BC_flux_file", getattr(cfg, "bot_BC_flux_file", None)))
    if getattr(cfg, "ini_mix", None) == "EQ":
        required_paths.append((
            "fastchem_solar_abundance_file",
            getattr(
                cfg,
                "fastchem_solar_abundance_file",
                "fastchem_vulcan/input/solar_element_abundances.dat",
            ),
        ))

    for label, rel_path in required_paths:
        if not rel_path:
            errors.append(f"{label} is unset.")
            continue
        if not (root / rel_path).exists():
            errors.append(f"{label}={rel_path!r} does not exist under {root}.")

    errors.extend(_validate_network_assets(cfg, root))
    errors.extend(_validate_numerical_bounds(cfg))

    if errors:
        raise RuntimeError(
            "Unsupported or invalid VULCAN-JAX runtime configuration:\n- "
            + "\n- ".join(errors)
        )
