"""Audit VULCAN-JAX HD189 parity against a VULCAN-master checkout.

The JAX side is loaded from its YAML config (``configs/HD189.yaml`` via
``config.load_config``); the master side keeps its
``cfg_examples/vulcan_cfg_HD189.py``. The acceptance target is scientific
parity, not byte-for-byte config parity: JAX's own solver tuning and the
vm_branch defaults are expected to differ (see ``INTENTIONAL_JAX_DELTAS``), so
the audit reports only unintended drift on the remaining physics keys, the
Mp/Rp-derived gravity, and the vendored runtime data.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import runpy
import sys
from pathlib import Path
from typing import Any

# Gravity is authored as Mp/Rp on the JAX side and reproduced to sub-ULP; compare
# it to master's explicit `gs` with a relative tolerance, not exact equality.
_GRAVITY_RTOL = 1e-9


# JAX-only knobs (values in default.yaml); the adapt-rtol schedule is
# vm_branch's (op.py:845-848), with the controller off in every parity config.
JAX_ONLY_KEYS: frozenset[str] = frozenset({
    "fastchem_solar_abundance_file",
    "use_ini_cold_trap",
    "use_sat_surfaceH2O",
    "rtol_min",
    "rtol_max",
    "adapt_rtol_dec_period",
    "adapt_rtol_inc_period",
    "adapt_rtol_dec",
    "adapt_rtol_inc",
    "adapt_rtol_loss_mul",
    "adapt_rtol_inc_loss_thresh",
    "batch_max_retries",
    "step_size_safety",
    "step_size_zero_delta_frac",
    "photo_switch_longdy_thresh",
    "photo_switch_longdydt_thresh",
    "hycean_pin_time",
    "loss_ex",
    "fastchem_newton_tol",
    "fastchem_newton_max_iter",
    "use_fix_all_bot",
    "use_fix_H2He",
})

# Shared keys where VULCAN-JAX intentionally differs from master:
#   use_vm_mol    -- on in the VULCAN 3 preset, off in the VULCAN 2 parity configs.
#   conver_ignore -- parity configs ship the pinned HD189 example's []; the V3
#                    preset ships vm_branch's ['HC3N'] (same step count and
#                    longdy on HD189/HD209).
#   top/bot_BC_flux_file -- master's atm/BC_top.txt / BC_bot.txt ship in
#                    neither tree; the flags are off, so JAX carries null.
INTENTIONAL_JAX_DELTAS = {
    "use_vm_mol",
    "conver_ignore",
    "top_BC_flux_file",
    "bot_BC_flux_file",
    "dt_max",  # capped at config.DT_MAX_S = 1e15 s (C19); master derives 1e17
}

# Network files carrying master's CH2CN + H + M correction (k0 1.00E-29,
# k_inf 1.00E-10), which master applied to NCHO only (C1); any other
# differing line fails.
KNOWN_THERMO_DIVERGENCES: dict[str, tuple[str, ...]] = {
    # Upstream lists NH3 condensate as 16.023 g/mol; the corrected value is
    # 17.031 g/mol. Only that species row may differ in all_compose.txt.
    "all_compose.txt": ("NH3_l_s",),
    "SNCHO_photo_network.txt": ("CH2CN + H + M -> CH3CN + M",),
    "SNCHO_photo_network_C3.txt": ("CH2CN + H + M -> CH3CN + M",),
    "SNCHO_DMS_photo_network_Tsai2024.txt": ("CH2CN + H + M -> CH3CN + M",),
}

# Vendored network files where the only allowed drift is the leading reaction
# index. Upstream's 2025 SNCHO file numbers one row 1039, breaking the
# otherwise increasing odd-numbered forward sequence, and then continues at
# 861; JAX renumbers the run so the indices stay monotonic. Every field after
# the index must still match, so a rate change is still real drift.
KNOWN_THERMO_RENUMBERED: frozenset[str] = frozenset(
    {"SNCHO_photo_network_2025.txt"}
)

# eps Eri flux (C4): master's builder (atm/make_spectra_in_nm.py) multiplies
# by R_star where it should divide, so its file is low by R_star^4; JAX ships
# the corrected file. Wavelengths must match and every flux ratio must sit at
# this factor within _SFLUX_RATIO_RTOL (the files' 2 significant figures).
EPS_ERI_RSTAR_RSUN = 0.735  # upstream atm/make_spectra_in_nm.py:7
_SFLUX_RATIO_RTOL = 1e-2
KNOWN_SFLUX_RESCALES: dict[str, float] = {
    "sflux-epseri.txt": EPS_ERI_RSTAR_RSUN**-4,
}

UI_OUTPUT_KEYS = {"output_dir", "out_name", "save_evolution", "save_evo_frq"}

IGNORED_RUNTIME_FILENAMES = {".DS_Store"}
# .md files are docs, not runtime data (master carries two thermo READMEs).
IGNORED_RUNTIME_SUFFIXES = {".py", ".pyc", ".md"}


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
    ignored = UI_OUTPUT_KEYS | JAX_ONLY_KEYS | INTENTIONAL_JAX_DELTAS

    # Shared keys (physics + numerics) must match, except UI, JAX-only knobs, and
    # the documented intentional deltas.
    for key in sorted(set(master) & set(jax)):
        if key in ignored:
            continue
        if not _same_value(master[key], jax[key]):
            errors.append(
                f"cfg mismatch {key}: master={master[key]!r}, jax={jax[key]!r}"
            )

    # JAX derives gs from Mp/Rp; verify it reproduces master's explicit gs
    # (authored to sub-ULP, hence a relative tolerance).
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

    # Master keys absent in JAX are drift, except gs (derived from Mp/Rp),
    # the JAX-only abundance file and config._REMOVED_KEYS.
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


def _index_renumber_only(master_path: Path, jax_path: Path) -> list[str]:
    """Return errors for any drift beyond the leading reaction index."""
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
        strip = re.compile(r"^\s*\d+\s+")
        if strip.sub("", m_line) == strip.sub("", j_line) and strip.match(m_line):
            continue  # leading reaction index only
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
    (``_SFLUX_RATIO_RTOL``). Anything else is real drift and is reported.
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
        if m_flux <= 0 or abs(j_flux / m_flux - factor) > _SFLUX_RATIO_RTOL * factor:
            errors.append(
                f"flux drift at line {lineno}: ratio {j_flux / m_flux:.6g} "
                f"vs documented rescale {factor:.6g}"
            )
    return errors


def _load_supported_inputs() -> dict:
    """The curated file list from tests/science_sources.yaml.

    Returns {} when the manifest is absent so the tool still runs standalone.
    """
    import yaml

    p = Path(__file__).resolve().parent.parent / "tests" / "science_sources.yaml"
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()).get("supported_inputs", {}) or {}


def _check_supported_inputs(jax_root: Path) -> list[str]:
    """Every manifest-listed vendored input must exist with its recorded hash."""
    errors: list[str] = []
    for rel, spec in sorted(_load_supported_inputs().items()):
        p = jax_root / rel
        if not p.is_file():
            errors.append(f"supported input missing from this checkout: {rel}")
            continue
        got = _sha256(p)
        want = spec.get("sha256")
        if want and got != want:
            errors.append(
                f"{rel}: sha256 {got[:16]}... does not match the manifest's "
                f"{want[:16]}... -- a vendored scientific input changed. If "
                "that was deliberate, update tests/science_sources.yaml and "
                "re-measure anything that depends on it.")
    return errors


def _compare_runtime_data(
    master_root: Path,
    jax_root: Path,
    oracle_family: str = "vulcan2_ncho",
) -> list[str]:
    """Compare the vendored files in `supported_inputs` for this oracle family.

    KNOWN_THERMO_DIVERGENCES files may differ on their listed reactions,
    KNOWN_THERMO_RENUMBERED files on the leading index, KNOWN_SFLUX_RESCALES
    files by their factor; anything else fails. A supported file the oracle
    lacks is an error; a file only the oracle has is not.
    """
    errors: list[str] = []
    supported = {
        rel: spec
        for rel, spec in _load_supported_inputs().items()
        if spec.get("oracle") == oracle_family
    }
    if supported:
        by_dir: dict[str, set[Path]] = {}
        for rel in supported:
            parts = Path(rel).parts
            if parts[0] in ("atm", "thermo"):
                by_dir.setdefault(parts[0], set()).add(Path(*parts[1:]))
    else:
        by_dir = {}

    for rel_dir in ("atm", "thermo"):
        master_files = _runtime_files(master_root / rel_dir)
        jax_files = _runtime_files(jax_root / rel_dir)
        master_keys = set(master_files)
        jax_keys = set(jax_files)
        if by_dir:
            scope = by_dir.get(rel_dir, set())
            master_keys &= scope
            jax_keys &= scope
            missing_from_oracle = scope - set(master_files)
            if missing_from_oracle:
                errors.append(
                    f"{rel_dir}: the oracle does not carry supported input(s) "
                    f"{sorted(str(p) for p in missing_from_oracle)} -- this "
                    "port claims to carry them faithfully, so parity cannot be "
                    "confirmed against this revision")
        else:
            # No manifest: whole-tree symmetric check.
            only_master = master_keys - jax_keys
            only_jax = jax_keys - master_keys
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
            if rel_path.name in KNOWN_THERMO_RENUMBERED:
                errors.extend(
                    f"{rel_dir}/{rel_path}: {msg}"
                    for msg in _index_renumber_only(
                        master_root / rel_dir / rel_path,
                        jax_root / rel_dir / rel_path,
                    )
                )
            elif known is not None:
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


# VULCAN-JAX-only identifiers: an oracle containing any of them is not
# pristine upstream, and comparing against it is circular.
_JAX_ONLY_MARKERS = (
    "conv_stall_window",
    "longdy_seen_min",
    "count_since_new_min",
    "wall_clock_max",
)


def _check_oracle_is_pristine(master_root: Path) -> list[str]:
    """Refuse to audit against a checkout carrying VULCAN-JAX's own code.

    Returns error strings (not warnings): a contaminated oracle produces
    silently wrong "parity" results, which is worse than no audit.
    """
    errors: list[str] = []
    for rel in (
        "op.py",
        "store.py",
        "vulcan_cfg.py",
        "cfg_examples/vulcan_cfg_HD189.py",
    ):
        path = master_root / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        hits = sorted({m for m in _JAX_ONLY_MARKERS if m in text})
        if hits:
            errors.append(
                f"{rel}: contains VULCAN-JAX-only identifier(s) {', '.join(hits)} "
                "-- this checkout is not pristine upstream VULCAN, so a parity "
                "result from it would be circular. Use a clean clone at the "
                "commit pinned in tests/science_sources.yaml."
            )
    if not (master_root / ".git").exists():
        errors.append(
            f"{master_root} has no .git: it is an unversioned copy whose "
            "provenance cannot be established. Do not cite it as upstream."
        )
    return errors


def audit(
    master_root: Path,
    jax_root: Path,
    oracle_family: str = "vulcan2_ncho",
) -> list[str]:
    """Return all HD189 parity errors for one pinned oracle family.

    The manifest pins inputs to different upstream commits, so only this
    family's inputs are compared.
    """
    # Vendored inputs must match the manifest; this needs no oracle.
    errors: list[str] = list(_check_supported_inputs(jax_root))

    # Refuse a contaminated oracle before comparing anything.
    provenance = _check_oracle_is_pristine(master_root)
    if provenance:
        return errors + provenance

    master_cfg = master_root / "cfg_examples" / "vulcan_cfg_HD189.py"
    if not master_cfg.exists():
        return errors + [f"missing master HD189 cfg: {master_cfg}"]

    errors.extend(_compare_cfgs(master_cfg))
    errors.extend(_compare_runtime_data(master_root, jax_root, oracle_family))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the parity audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Runtime data (atm/, thermo/) lives under the installed package, not the
    # repo root; the JAX config is loaded by name from configs/*.yaml. Default
    # to the package dir so no explicit --jax-root is needed.
    from vulcan_jax._paths import PACKAGE_ROOT

    parser.add_argument(
        "--master",
        type=Path,
        default=None,
        help="Path to the VULCAN checkout to audit against; falls back to "
        "$VULCAN_MASTER_DIR (a clean pinned clone).",
    )
    parser.add_argument(
        "--oracle-family",
        default="vulcan2_ncho",
        choices=("vulcan2_ncho",),
        help="Pinned oracle family used for the HD189/runtime comparison.",
    )
    parser.add_argument(
        "--jax-root",
        type=Path,
        default=PACKAGE_ROOT,
        help="Path to the VULCAN-JAX package dir (src/vulcan_jax).",
    )
    args = parser.parse_args(argv)

    master = args.master
    if master is None:
        env_master = os.environ.get("VULCAN_MASTER_DIR")
        if not env_master:
            parser.error(
                "--master is required (or set $VULCAN_MASTER_DIR to a clean "
                "pinned clone of exoclime/VULCAN; see tests/science_sources.yaml)."
            )
        master = Path(env_master)

    errors = audit(
        master.resolve(),
        args.jax_root.resolve(),
        args.oracle_family,
    )
    if errors:
        print("FAIL: HD189 parity audit found drift:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: VULCAN-JAX HD189 parity audit is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
