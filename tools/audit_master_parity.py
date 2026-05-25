"""Audit VULCAN-JAX default HD189 parity against a VULCAN-master checkout."""

from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path
from typing import Any


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


def _compare_cfgs(master_cfg: Path, jax_cfg: Path) -> list[str]:
    """Compare HD189 physics/numerical config values."""
    errors: list[str] = []
    master = _load_cfg(master_cfg)
    jax = _load_cfg(jax_cfg)
    ignored = UI_OUTPUT_KEYS | set(JAX_ONLY_DEFAULTS)

    for key in sorted(set(master) & set(jax)):
        if key in ignored:
            continue
        if not _same_value(master[key], jax[key]):
            errors.append(
                f"cfg mismatch {key}: master={master[key]!r}, jax={jax[key]!r}"
            )

    for key, expected in JAX_ONLY_DEFAULTS.items():
        actual = jax.get(key)
        if not _same_value(actual, expected):
            errors.append(f"JAX-only default {key}={actual!r}, expected {expected!r}")

    missing_in_jax = sorted(
        key
        for key in set(master) - set(jax) - UI_OUTPUT_KEYS
        if key not in {"fastchem_solar_abundance_file"}
    )
    if missing_in_jax:
        errors.append(f"JAX cfg missing master keys: {missing_in_jax}")
    return errors


def _compare_jax_root_to_example(jax_root: Path) -> list[str]:
    """Ensure the root default config is the canonical HD189 example."""
    root_cfg = _load_cfg(jax_root / "vulcan_cfg.py")
    example_cfg = _load_cfg(jax_root / "cfg_examples" / "vulcan_cfg_HD189.py")
    errors: list[str] = []
    for key in sorted(set(root_cfg) | set(example_cfg)):
        if key in UI_OUTPUT_KEYS:
            continue
        if key not in root_cfg:
            errors.append(f"root vulcan_cfg.py missing HD189 key {key}")
            continue
        if key not in example_cfg:
            errors.append(f"root vulcan_cfg.py has extra key {key}")
            continue
        if not _same_value(root_cfg[key], example_cfg[key]):
            errors.append(
                "root vulcan_cfg.py differs from cfg_examples/vulcan_cfg_HD189.py "
                f"for {key}: root={root_cfg[key]!r}, example={example_cfg[key]!r}"
            )
    return errors


def _compare_runtime_data(master_root: Path, jax_root: Path) -> list[str]:
    """Compare non-code atm/ and thermo/ runtime files byte-for-byte."""
    errors: list[str] = []
    for rel_dir in ("atm", "thermo"):
        master_files = _runtime_files(master_root / rel_dir)
        jax_files = _runtime_files(jax_root / rel_dir)
        master_keys = set(master_files)
        jax_keys = set(jax_files)
        if master_keys != jax_keys:
            errors.append(
                f"{rel_dir}: file set mismatch, only master={sorted(master_keys - jax_keys)}, "
                f"only jax={sorted(jax_keys - master_keys)}"
            )
        for rel_path in sorted(master_keys & jax_keys):
            if master_files[rel_path] != jax_files[rel_path]:
                errors.append(f"{rel_dir}: byte drift in {rel_path}")
    return errors


def audit(master_root: Path, jax_root: Path) -> list[str]:
    """Return all default-parity audit errors."""
    errors: list[str] = []
    master_cfg = master_root / "cfg_examples" / "vulcan_cfg_HD189.py"
    jax_cfg = jax_root / "vulcan_cfg.py"
    if not master_cfg.exists():
        errors.append(f"missing master HD189 cfg: {master_cfg}")
    if not jax_cfg.exists():
        errors.append(f"missing JAX root cfg: {jax_cfg}")
    if errors:
        return errors

    errors.extend(_compare_cfgs(master_cfg, jax_cfg))
    errors.extend(_compare_jax_root_to_example(jax_root))
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
    default_jax = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--master",
        type=Path,
        default=default_jax.parent / "VULCAN-master",
        help="Path to the VULCAN-master checkout.",
    )
    parser.add_argument(
        "--jax-root",
        type=Path,
        default=default_jax,
        help="Path to the VULCAN-JAX checkout.",
    )
    args = parser.parse_args(argv)

    errors = audit(args.master.resolve(), args.jax_root.resolve())
    if errors:
        print("FAIL: default parity audit found drift:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: VULCAN-JAX default HD189 parity audit is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
