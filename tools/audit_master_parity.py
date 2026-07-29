"""Audit VULCAN-JAX HD189 parity against a VULCAN-master checkout.

The JAX side is loaded from its YAML config (``configs/HD189.yaml`` via
``config.load_config``) — the ``vulcan_cfg.py`` / ``cfg_examples/`` Python
configs were removed in the 2026-07 migration. The master side keeps its
``cfg_examples/vulcan_cfg_HD189.py``. The acceptance target is *scientific
parity*, not byte-for-byte config parity: JAX's own solver tuning and the
vm_branch defaults are expected to differ (see ``INTENTIONAL_JAX_DELTAS``), so
the audit reports only unintended drift on the remaining physics keys, the
Mp/Rp-derived gravity, and the vendored runtime data.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import runpy
import sys
from pathlib import Path
from typing import Any

# Gravity is authored as Mp/Rp on the JAX side and reproduced to sub-ULP; compare
# it to master's explicit `gs` with a relative tolerance, not exact equality.
_GRAVITY_RTOL = 1e-9


JAX_ONLY_DEFAULTS: dict[str, Any] = {
    "fastchem_solar_abundance_file": "fastchem_vulcan/input/solar_element_abundances.dat",
    "use_ini_cold_trap": False,
    "use_sat_surfaceH2O": False,
    "rtol_min": 0.0,
    "rtol_max": 1.0,
    "adapt_rtol_dec_period": 10,
    "adapt_rtol_inc_period": 1000,
    "adapt_rtol_dec": 0.75,
    "adapt_rtol_inc": 1.25,
    "adapt_rtol_loss_mul": 2.0,
    "adapt_rtol_inc_loss_thresh": 2e-4,
    "batch_max_retries": 64,
    "step_size_safety": 0.9,
    "step_size_zero_delta_frac": 0.01,
    "photo_switch_longdy_thresh": 1.0,
    "photo_switch_longdydt_thresh": 1e-6,
    "hycean_pin_time": 1e6,
    "loss_ex": [],
    "fastchem_newton_tol": 1e-12,
    "fastchem_newton_max_iter": 50,
    "use_fix_all_bot": False,
    "use_fix_H2He": False,
    "use_chunked_runner": False,
}

# Shared config keys where VULCAN-JAX deliberately differs from master. These are
# intentional solver/physics choices, not drift, so the audit ignores them:
#   use_vm_mol   -- vm_branch upwind molecular diffusion is on by default in JAX.
#   conver_ignore-- JAX ships a leaner convergence-ignore list under its rtol=0.2
#                   tuning (scientific parity, not byte parity).
INTENTIONAL_JAX_DELTAS = {
    "use_vm_mol",
    "conver_ignore",
}

# Vendored network files where JAX intentionally diverges from master by a known,
# documented set of reactions. SNCHO carries master's own 10.2025 CH2CN typo fix
# (k0 1.00E-20 -> 1.00E-29) that master applied to NCHO but left unapplied to
# SNCHO; JAX applies it to both. Any OTHER differing line is real drift and fails.
KNOWN_THERMO_DIVERGENCES: dict[str, tuple[str, ...]] = {
    "SNCHO_photo_network.txt": ("CH2CN + H + M -> CH3CN + M",),
}

# Vendored stellar-flux files where JAX intentionally diverges from master by a
# uniform flux rescale. Master's builder (atm/make_spectra_in_nm.py) multiplied
# by R_star where the surface-flux conversion divides, so the shipped eps Eri
# spectrum is low by exactly R_star^4 = 0.735^4; JAX ships the corrected file
# (corrections_to_original_code.md, C4). Wavelength columns must stay identical
# and every flux ratio must sit at the documented factor (2-sig-fig rounding
# tolerance); any other difference is real drift and fails.
KNOWN_SFLUX_RESCALES: dict[str, float] = {
    "sflux-epseri.txt": 0.735**-4,
}

# Runtime inputs VULCAN-JAX ships that master has no counterpart for -- planet cases
# added here with their own configs. This audit exists to catch DRIFT in the files
# shared with master, not to stop this repo carrying an extra case, but each extra
# must be named here so a stray or accidentally-copied file still fails. Asymmetric
# on purpose: a file present only in MASTER is always an error (that is a vendored
# input we dropped), and an unlisted jax-only file is too.
JAX_ONLY_RUNTIME_FILES: frozenset[str] = frozenset(
    {
        # configs/K2-18b.yaml, a VULCAN-JAX-only ported case (no master counterpart)
        "atm_K2-18b-Nep100X-apo-H2Oclouds.txt",
    }
)

UI_OUTPUT_KEYS = {
    "output_dir",
    "plot_dir",
    "movie_dir",
    "out_name",
    "plot_TP",
    "use_live_plot",
    "use_live_flux",
    "use_plot_end",
    "use_plot_evo",
    "use_save_movie",
    "use_flux_movie",
    "plot_height",
    "use_PIL",
    "live_plot_frq",
    "save_movie_rate",
    "y_time_freq",
    "plot_spec",
    "output_humanread",
    "use_shark",
    "save_evolution",
    "save_evo_frq",
}

IGNORED_RUNTIME_FILENAMES = {".DS_Store"}
IGNORED_RUNTIME_SUFFIXES = {".py", ".pyc"}
STOCK_FASTCHEM = Path("fastchem_vulcan/input/solar_element_abundances.dat")


def _is_data_value(value: Any) -> bool:
    """Return True for cfg values that can be compared directly."""
    return isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))


def _load_cfg(path: Path) -> dict[str, Any]:
    """Execute a VULCAN cfg and return its public literal data attributes."""
    raw = runpy.run_path(str(path))
    return {
        key: value
        for key, value in raw.items()
        if not key.startswith("_") and _is_data_value(value)
    }


def _same_value(lhs: Any, rhs: Any) -> bool:
    """Compare scalar and container cfg values exactly."""
    if isinstance(lhs, float) or isinstance(rhs, float):
        return float(lhs) == float(rhs)
    return lhs == rhs


def _sha256(path: Path) -> str:
    """Return the SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_files(root: Path) -> dict[Path, str]:
    """Return relative non-code runtime-data paths and hashes under root."""
    files: dict[Path, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name in IGNORED_RUNTIME_FILENAMES:
            continue
        if path.suffix in IGNORED_RUNTIME_SUFFIXES:
            continue
        files[path.relative_to(root)] = _sha256(path)
    return files


def _parse_abundances(path: Path) -> dict[str, float]:
    """Parse a FastChem element-abundance file into element -> log abundance."""
    abundances: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                abundances[parts[0]] = float(parts[1])
    return abundances


def _check_stock_fastchem(path: Path, label: str) -> list[str]:
    """Validate the canonical FastChem abundance file (rocky-suppressed).

    Shipped networks contain no Mg / Si / Fe / Ti / V / Cl / K / Na / F / Ca / P
    species, so any positive abundance for those elements in the FastChem input
    silently sequesters O / H / etc. into species `_load_eq_y` cannot read
    back. The canonical file therefore uses Lodders 2019 H/He/C/N/O/S values
    with all rocky elements pinned to -3.0.
    """
    errors: list[str] = []
    if not path.exists():
        return [f"{label}: missing {path}"]
    abundances = _parse_abundances(path)
    expected = {
        "H": 12.0,
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
    for element, expected_value in expected.items():
        actual = abundances.get(element)
        if actual != expected_value:
            errors.append(f"{label}: {element}={actual!r}, expected {expected_value!r}")
    return errors


def _check_fastchem_files_match(master_path: Path, jax_path: Path) -> list[str]:
    """The two repos' FastChem abundance files must be byte-identical.

    A content drift here silently changes initial conditions in either run.
    """
    if not master_path.exists() or not jax_path.exists():
        return []  # missing-file errors are reported by _check_stock_fastchem
    if _sha256(master_path) != _sha256(jax_path):
        return [
            f"FastChem abundance drift: master {master_path} vs jax {jax_path} "
            "differ by SHA-256"
        ]
    return []


def _compare_cfgs(master_cfg: Path) -> list[str]:
    """Compare master HD189 physics config against the JAX HD189 config (YAML)."""
    from vulcan_jax.atm_setup import surface_gravity
    from vulcan_jax.config import load_config

    errors: list[str] = []
    master = _load_cfg(master_cfg)
    jax_cfg = load_config("HD189")
    jax = {
        key: value
        for key, value in vars(jax_cfg).items()
        if not key.startswith("_") and _is_data_value(value)
    }
    ignored = UI_OUTPUT_KEYS | set(JAX_ONLY_DEFAULTS) | INTENTIONAL_JAX_DELTAS

    # Shared keys (physics + numerics) must match, except UI, JAX-only knobs, and
    # the documented intentional deltas.
    for key in sorted(set(master) & set(jax)):
        if key in ignored:
            continue
        if not _same_value(master[key], jax[key]):
            errors.append(
                f"cfg mismatch {key}: master={master[key]!r}, jax={jax[key]!r}"
            )

    # Gravity migration: JAX derives gs from Mp/Rp; verify it reproduces master's
    # explicit gs (authored to sub-ULP, hence a relative tolerance).
    if "gs" in master:
        try:
            jax_gs = surface_gravity(jax_cfg)
        except Exception as exc:  # noqa: BLE001 - surface the failure loudly
            errors.append(f"JAX surface_gravity(HD189) failed: {exc!r}")
        else:
            if not math.isclose(float(master["gs"]), jax_gs, rel_tol=_GRAVITY_RTOL):
                errors.append(
                    f"gravity mismatch: master gs={master['gs']!r}, "
                    f"jax G*Mp/Rp^2={jax_gs!r}"
                )

    # Real gaps: master physics keys absent in JAX. `gs` is the intended Mp/Rp
    # swap; `fastchem_solar_abundance_file` is a JAX-only addition; keys in
    # config._REMOVED_KEYS were deliberately retired with a loud migration
    # message (e.g. fix_species_time -> stop_conden_time, use_print_delta not
    # ported), so their absence is the documented design, not drift.
    from vulcan_jax.config import _REMOVED_KEYS

    missing_in_jax = sorted(
        key
        for key in set(master) - set(jax) - UI_OUTPUT_KEYS - set(_REMOVED_KEYS)
        if key not in {"fastchem_solar_abundance_file", "gs"}
    )
    if missing_in_jax:
        errors.append(f"JAX cfg missing master keys: {missing_in_jax}")
    return errors


def _known_divergence_only(
    master_path: Path, jax_path: Path, reactions: tuple[str, ...]
) -> list[str]:
    """Return errors for any line drift NOT covered by the allowlisted reactions.

    A vendored network file may intentionally diverge from master on a known set
    of reaction lines (e.g. a typo fix master applied unevenly). Every *other*
    differing line is real drift and is reported.
    """
    master_lines = master_path.read_text().splitlines()
    jax_lines = jax_path.read_text().splitlines()
    if len(master_lines) != len(jax_lines):
        return [
            f"line-count drift ({len(master_lines)} master vs {len(jax_lines)} jax)"
        ]
    errors: list[str] = []
    for lineno, (m_line, j_line) in enumerate(zip(master_lines, jax_lines), start=1):
        if m_line == j_line:
            continue
        if any(rx in m_line or rx in j_line for rx in reactions):
            continue  # known, documented divergence
        errors.append(
            f"unexpected drift at line {lineno}: master={m_line!r}, jax={j_line!r}"
        )
    return errors


def _known_sflux_rescale_only(
    master_path: Path, jax_path: Path, factor: float
) -> list[str]:
    """Return errors unless jax differs from master by exactly the flux rescale.

    Wavelength fields must be byte-identical; every flux ratio jax/master must
    match ``factor`` within the 2-significant-figure rounding of the file format
    (rtol 1e-2). Anything else is real drift and is reported.
    """
    master_lines = master_path.read_text().splitlines()
    jax_lines = jax_path.read_text().splitlines()
    if len(master_lines) != len(jax_lines):
        return [
            f"line-count drift ({len(master_lines)} master vs {len(jax_lines)} jax)"
        ]
    errors: list[str] = []
    for lineno, (m_line, j_line) in enumerate(zip(master_lines, jax_lines), start=1):
        if m_line.startswith("#") or j_line.startswith("#"):
            if m_line != j_line:
                errors.append(f"header drift at line {lineno}")
            continue
        m_parts, j_parts = m_line.split(), j_line.split()
        if len(m_parts) != 2 or len(j_parts) != 2 or m_parts[0] != j_parts[0]:
            errors.append(f"wavelength/format drift at line {lineno}")
            continue
        m_flux, j_flux = float(m_parts[1]), float(j_parts[1])
        if m_flux <= 0 or abs(j_flux / m_flux - factor) > 1e-2 * factor:
            errors.append(
                f"flux drift at line {lineno}: ratio {j_flux / m_flux:.6g} "
                f"vs documented rescale {factor:.6g}"
            )
    return errors


def _compare_runtime_data(master_root: Path, jax_root: Path) -> list[str]:
    """Compare non-code atm/ and thermo/ runtime files byte-for-byte.

    Files listed in ``KNOWN_THERMO_DIVERGENCES`` are allowed to differ on their
    documented reaction lines only; files in ``KNOWN_SFLUX_RESCALES`` must differ
    by exactly their documented flux rescale; any other difference still fails.
    """
    errors: list[str] = []
    for rel_dir in ("atm", "thermo"):
        master_files = _runtime_files(master_root / rel_dir)
        jax_files = _runtime_files(jax_root / rel_dir)
        master_keys = set(master_files)
        jax_keys = set(jax_files)
        only_master = master_keys - jax_keys
        only_jax = {
            rel for rel in jax_keys - master_keys
            if rel.name not in JAX_ONLY_RUNTIME_FILES
        }
        if only_master or only_jax:
            errors.append(
                f"{rel_dir}: file set mismatch, only master={sorted(only_master)}, "
                f"only jax={sorted(only_jax)}"
            )
        for rel_path in sorted(master_keys & jax_keys):
            if master_files[rel_path] == jax_files[rel_path]:
                continue
            known = KNOWN_THERMO_DIVERGENCES.get(rel_path.name)
            rescale = KNOWN_SFLUX_RESCALES.get(rel_path.name)
            if known is not None:
                sub = _known_divergence_only(
                    master_root / rel_dir / rel_path,
                    jax_root / rel_dir / rel_path,
                    known,
                )
                errors.extend(f"{rel_dir}/{rel_path}: {msg}" for msg in sub)
            elif rescale is not None:
                sub = _known_sflux_rescale_only(
                    master_root / rel_dir / rel_path,
                    jax_root / rel_dir / rel_path,
                    rescale,
                )
                errors.extend(f"{rel_dir}/{rel_path}: {msg}" for msg in sub)
            else:
                errors.append(f"{rel_dir}: byte drift in {rel_path}")
    return errors


def audit(master_root: Path, jax_root: Path) -> list[str]:
    """Return all HD189 parity audit errors (JAX side loaded from YAML)."""
    master_cfg = master_root / "cfg_examples" / "vulcan_cfg_HD189.py"
    if not master_cfg.exists():
        return [f"missing master HD189 cfg: {master_cfg}"]

    errors: list[str] = []
    errors.extend(_compare_cfgs(master_cfg))
    errors.extend(_compare_runtime_data(master_root, jax_root))
    errors.extend(_check_stock_fastchem(jax_root / STOCK_FASTCHEM, "JAX FastChem"))
    errors.extend(
        _check_stock_fastchem(master_root / STOCK_FASTCHEM, "master FastChem")
    )
    errors.extend(
        _check_fastchem_files_match(
            master_root / STOCK_FASTCHEM,
            jax_root / STOCK_FASTCHEM,
        )
    )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the parity audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent  # VULCAN-JAX/
    # Runtime data (atm/, thermo/, fastchem_vulcan/) lives under the installed
    # package, not the repo root, after the flat->package restructuring; the JAX
    # config is loaded by name from configs/*.yaml. Default to the package dir so
    # the documented invocation `python tools/audit_master_parity.py --master
    # ../VULCAN-master` works without an explicit --jax-root.
    from vulcan_jax._paths import PACKAGE_ROOT

    parser.add_argument(
        "--master",
        type=Path,
        default=repo_root.parent / "VULCAN-master",
        help="Path to the VULCAN-master checkout.",
    )
    parser.add_argument(
        "--jax-root",
        type=Path,
        default=PACKAGE_ROOT,
        help="Path to the VULCAN-JAX package dir (src/vulcan_jax).",
    )
    args = parser.parse_args(argv)

    errors = audit(args.master.resolve(), args.jax_root.resolve())
    if errors:
        print("FAIL: HD189 parity audit found drift:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: VULCAN-JAX HD189 parity audit is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
