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
    # These must match the shipped PARITY YAMLs (default/HD189/HD209/W39b),
    # NOT the code fallbacks (1000 / 0.75), which no config uses. The parity
    # configs leave the adapt-rtol controller OFF; HD189_vulcan3 carries
    # vm_branch's 1000 / 0.5 / 1.25.
    "adapt_rtol_dec_period": 10,
    "adapt_rtol_inc_period": 500,
    "adapt_rtol_dec": 0.5,
    "adapt_rtol_inc": 1.5,
    "adapt_rtol_loss_mul": 2.0,
    "adapt_rtol_inc_loss_thresh": 2e-4,
    "batch_max_retries": 110,
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
#   use_vm_mol   -- vm_branch upwind molecular diffusion; on in the VULCAN 3
#                   preset, off in the VULCAN 2 parity configs (which match
#                   fetched exoclime master).
#   conver_ignore-- the parity configs ship upstream master's `[]`; the VULCAN 3
#                   preset ships vm_branch's `['HC3N']`. Measured behaviourally
#                   identical on HD189/HD209 (same step count, same longdy,
#                   same controlling cell).
INTENTIONAL_JAX_DELTAS = {
    "use_vm_mol",
    "conver_ignore",
}

# Vendored network files where JAX intentionally diverges from master by a known,
# documented set of reactions. The sulfur networks carry master's own CH2CN
# typo fix (correct k0 = 1.00E-29, k_inf = 1.00E-10) that master applied to
# NCHO but left unapplied to its sulfur files: the oracle's base SNCHO and C3
# files ship k0 = 1.00E-20 and its DMS Tsai2024 file ships k0/k_inf swapped.
# JAX corrects all three (README.md, Parity & bug guide). Any OTHER differing
# line is real drift and fails.
KNOWN_THERMO_DIVERGENCES: dict[str, tuple[str, ...]] = {
    # Upstream lists NH3 condensate as 16.023 g/mol; the corrected value is
    # 17.031 g/mol. Only that species row may differ in all_compose.txt.
    "all_compose.txt": ("NH3_l_s",),
    "SNCHO_photo_network.txt": ("CH2CN + H + M -> CH3CN + M",),
    "SNCHO_photo_network_C3.txt": ("CH2CN + H + M -> CH3CN + M",),
    "SNCHO_DMS_photo_network_Tsai2024.txt": ("CH2CN + H + M -> CH3CN + M",),
}

# Vendored stellar-flux files where JAX intentionally diverges from master by a
# uniform flux rescale. Master's builder (atm/make_spectra_in_nm.py) multiplied
# by R_star where the surface-flux conversion divides, so the shipped eps Eri
# spectrum is low by exactly R_star^4 = 0.735^4; JAX ships the corrected file
# (README.md, Parity & bug guide). Wavelength columns must stay identical and
# every flux ratio must sit at the documented factor (2-sig-fig tolerance); any
# other difference is real drift and fails.
KNOWN_SFLUX_RESCALES: dict[str, float] = {
    "sflux-epseri.txt": 0.735**-4,
}

# Runtime inputs VULCAN-JAX ships that master has no counterpart for -- planet
# cases added here with their own configs. Each extra must be named here so a
# stray or accidentally-copied file still fails. Asymmetric on purpose: a file
# present only in MASTER is always an error (a vendored input we dropped), and
# an unlisted jax-only file is too.
JAX_ONLY_RUNTIME_FILES: frozenset[str] = frozenset(
    {
        # Input for the withdrawn configs/K2-18b.yaml (removed: it converged in
        # neither code). The T-P file stays vendored so the case can be re-run;
        # it has no master counterpart either way.
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
# `.md` is documentation, not runtime data: master's vendored trees carry
# `thermo/README.md` + `thermo/photo_cross/README.md`, whose content moved into
# our README's "Vendored data provenance" section on 2026-08-16 (3-doc policy).
# Without this the whole-tree symmetric fallback below would report those two
# as `only master=` drift, which is a doc-layout difference, not a data one.
IGNORED_RUNTIME_SUFFIXES = {".py", ".pyc", ".md"}
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


def _report_master_fastchem_preset(master_path: Path) -> list[str]:
    """Identify the oracle's composition. Never an error on its own.

    The two codes legitimately ship different elemental abundances: VULCAN-JAX
    defaults to rocky-suppressed Lodders 2019, upstream to full-solar
    Lodders 2009. Which one a comparison should use depends on the question
    being asked, so this reports rather than judges. An unrecognised file IS an
    error, because it means the oracle's composition came from somewhere
    undocumented.
    """
    if not master_path.exists():
        return []  # missing-file errors are reported by _check_stock_fastchem
    abundances = _parse_abundances(master_path)
    he = abundances.get("He")
    mg = abundances.get("Mg")
    if he is None or mg is None:
        return [f"oracle FastChem file {master_path} is unreadable or truncated"]
    if abs(he - 10.9864) < 1e-9 and mg > 0:
        print(f"  oracle composition: full-solar Lodders 2009 ({master_path.name})")
        return []
    if abs(he - 10.9232) < 1e-9 and mg == -3.0:
        print(
            f"  oracle composition: rocky-suppressed Lodders 2019 "
            f"({master_path.name}) -- matches VULCAN-JAX's default"
        )
        return []
    return [
        f"oracle FastChem file {master_path} matches neither known preset "
        f"(He={he}, Mg={mg}). Its composition has an undocumented provenance, "
        "so any cross-code number from it is unattributable."
    ]


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
    """Every manifest-listed vendored input must exist with its recorded hash.

    This is the positive half of the audit: it states what the port PROMISES to
    carry. A silent edit to a vendored network or abundance file changes results
    everywhere and is otherwise invisible.
    """
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
    """Compare the SUPPORTED vendored runtime files against the oracle.

    Scope is the curated `supported_inputs` list in
    ``tests/science_sources.yaml`` -- the files this port promises to carry --
    NOT whole-tree equality. Do not revert to whole-tree equality: no real
    upstream can satisfy it (upstream ships network files this release
    deliberately does not carry), so the audit would fail on file-set noise
    before comparing anything meaningful.

    Files in ``KNOWN_THERMO_DIVERGENCES`` may differ on their documented
    reaction lines only; files in ``KNOWN_SFLUX_RESCALES`` must differ by
    exactly their documented flux rescale; any other difference still fails.
    A supported file the ORACLE lacks is reported (parity cannot be confirmed
    without it), but a file present only in the oracle is NOT an error --
    that is upstream being newer or broader than this release.
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
            # No manifest: fall back to the old whole-tree symmetric check.
            only_master = master_keys - jax_keys
            only_jax = {
                rel
                for rel in jax_keys - master_keys
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


# VULCAN-JAX-only identifiers. If any of these appear in the checkout being used
# as the oracle, that checkout is NOT pristine upstream VULCAN: someone has
# back-ported VULCAN-JAX code into it, and every comparison below becomes
# circular (the audit would be checking VULCAN-JAX against itself). This has
# happened: the sibling `../VULCAN-master/` copy was found carrying VULCAN-JAX's
# stall detector, `conv_stall_window` knob, `wall_clock_max` exit, and
# 13-species `conver_ignore` list, and was cited as "master parity" evidence.
# Keep this guard.
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
                "result from it would be circular. Fetch upstream instead: "
                "raw.githubusercontent.com/exoclime/VULCAN/master/<path> "
                "(or shami-EEG/VULCAN vm_branch for VULCAN 3 features)."
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

    The manifest deliberately assigns different inputs to different upstream
    commits. Comparing every supported file with one checkout would mix
    incompatible oracles and report valid files as missing.
    """
    # Our OWN vendored inputs must match the manifest first. This half needs no
    # oracle and is always meaningful: it states what the port promises to
    # carry, so a silent edit to a network or abundance file is caught even
    # when no upstream checkout is available.
    errors: list[str] = list(_check_supported_inputs(jax_root))

    # Establish that the oracle is actually upstream BEFORE comparing anything.
    provenance = _check_oracle_is_pristine(master_root)
    if provenance:
        return errors + provenance

    master_cfg = master_root / "cfg_examples" / "vulcan_cfg_HD189.py"
    if not master_cfg.exists():
        return errors + [f"missing master HD189 cfg: {master_cfg}"]

    errors.extend(_compare_cfgs(master_cfg))
    errors.extend(_compare_runtime_data(master_root, jax_root, oracle_family))
    # VULCAN-JAX's own file must be one of the two shipped presets. This is a
    # check on OUR tree and is always meaningful.
    errors.extend(_check_stock_fastchem(jax_root / STOCK_FASTCHEM, "JAX FastChem"))

    # The upstream side is NOT checked against our rocky-suppressed abundance
    # file, and the two are NOT required to match. Do not add such a check: a
    # pristine upstream tree ships full-solar Lodders 2009, so requiring our
    # file could only ever pass against a hand-patched oracle -- exactly what
    # the pristine guard rejects. Composition parity is a science-input CONFIG
    # choice: point `fastchem_solar_abundance_file` at
    # solar_element_abundances_lodders2009.dat to reproduce upstream's
    # composition (README.md, Validation, "Elemental abundances are a config
    # choice"). This audit only reports which composition the oracle would use.
    errors.extend(
        _report_master_fastchem_preset(master_root / STOCK_FASTCHEM),
    )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the parity audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent  # VULCAN-JAX/
    # Runtime data (atm/, thermo/, fastchem_vulcan/) lives under the installed
    # package, not the repo root; the JAX config is loaded by name from
    # configs/*.yaml. Default to the package dir so the documented invocation
    # `python tools/audit_master_parity.py --master ../VULCAN-master` works
    # without an explicit --jax-root.
    from vulcan_jax._paths import PACKAGE_ROOT

    parser.add_argument(
        "--master",
        type=Path,
        default=repo_root.parent / "VULCAN-master",
        help="Path to the VULCAN-master checkout.",
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

    errors = audit(
        args.master.resolve(),
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
