"""Pre-run configuration validation for the standalone VULCAN-JAX runtime."""

from __future__ import annotations

from pathlib import Path

import chem_funs


# Phase 19: live-output paths (matplotlib live mixing-ratio plot, live flux
# plot, movie frame save) were dropped from the supported runtime surface.
# `save_out` produces the .vul pickle; the post-run scripts under `plot_py/`
# (and the kept `Output.plot_end` / `plot_evo` / `plot_TP`) cover
# visualisation. These flags are rejected if set to True.
_DROPPED_LIVE_OUTPUT_FLAGS: tuple[str, ...] = (
    "use_live_plot",
    "use_live_flux",
    "use_save_movie",
    "use_flux_movie",
)


def validate_runtime_config(cfg, root: Path | None = None) -> None:
    """Reject unsupported or incomplete runtime configs before JIT setup.

    Enforces solver choice, required file paths, and incompatible option
    combinations. Phase 19 also rejects the dropped live-output flags
    (`use_live_plot` / `use_live_flux` / `use_save_movie` / `use_flux_movie`)
    when set to True.
    """
    root = Path(__file__).resolve().parent if root is None else Path(root)
    errors: list[str] = []

    if getattr(cfg, "ode_solver", None) != "Ros2":
        errors.append(
            f"ode_solver={cfg.ode_solver!r} is unsupported; VULCAN-JAX only supports 'Ros2'."
        )

    for flag in _DROPPED_LIVE_OUTPUT_FLAGS:
        if bool(getattr(cfg, flag, False)):
            errors.append(
                f"{flag}=True is no longer supported (Phase 19 dropped live "
                "output); use save_out + post-run scripts (plot_py/) instead."
            )

    if bool(getattr(cfg, "use_ion", False)) and not bool(getattr(cfg, "use_photo", False)):
        errors.append("use_ion=True requires use_photo=True in VULCAN-JAX.")

    if bool(getattr(cfg, "fix_species", [])) and not bool(getattr(cfg, "use_condense", False)):
        errors.append("fix_species is set but use_condense=False; this configuration is inconsistent.")

    if bool(getattr(cfg, "use_fix_H2He", False)):
        # The Hycean bottom-pin (op.py:2938-2944) snapshots H2/He mixing
        # ratios at t > 1e6 and pins the bottom layer thereafter.
        # OuterLoop port lives in outer_loop._make_runner — this validation
        # only requires that H2 and He exist in the network.
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

    for label, rel_path in required_paths:
        if not rel_path:
            errors.append(f"{label} is unset.")
            continue
        if not (root / rel_path).exists():
            errors.append(f"{label}={rel_path!r} does not exist under {root}.")

    if errors:
        raise RuntimeError(
            "Unsupported or invalid VULCAN-JAX runtime configuration:\n- "
            + "\n- ".join(errors)
        )
