"""Apples-to-apples comparison of two VULCAN ``.vul`` outputs.

The previous version of this script just dumped per-species absolute and
relative mixing-ratio diffs and called anything above 20 % a FAIL. That
metric is dominated by trace species (``CH4`` at ymix ~ 5e-7 can show a
1000× relerr that means nothing scientifically) and silently ignores
the bigger problem: if the two runs stopped at *different physical
times*, every diff is contaminated by time evolution on top of any
numerical drift. The headline becomes uninterpretable.

This rewrite makes the comparison strictly apples-to-apples:

1. **Static invariants.** Network identity (``species`` list, ``nr``)
   and initial state (``y_ini``) must match bit-exactly. If a run
   started from a different initial condition or used a different
   network, no downstream comparison is meaningful.
2. **Convergence state alignment.** ``t``, ``dt``, ``longdy`` are
   reported side-by-side, and a relative-time check gates the
   per-species comparison. If the two runs stopped at substantially
   different ``t``, the script fails fast unless
   ``--allow-time-mismatch`` is passed (in which case it still prints
   the diagnostic but flags the result as untrusted).
3. **Abundance-aware ymix metric.** Per-species ``|Δ log10 ymix|`` is
   computed only over species above an abundance floor (default
   ``1e-10``). The headline is the median of that, not the worst-case
   relerr on trace species. This is the standard metric used elsewhere
   in this workspace (see ``vulcan-emulator`` ``mean_abs_log10_error``)
   and matches the per-component agreement table in ``STATUS.md``.
4. **Mass conservation side-by-side.** ``atom_loss`` is normalized
   against the same ``y_ini`` for both runs, so it remains meaningful
   even when ``t`` diverges; reporting it gives a sanity check that's
   robust to integration-stop differences.

Usage::

    python tests/compare_vul.py output/HD189.vul ../VULCAN-master/output/HD189.vul
    python tests/compare_vul.py a.vul b.vul --floor 1e-8 --log10-tol 0.05
    python tests/compare_vul.py a.vul b.vul --allow-time-mismatch     # diagnostic only
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


EPSILON = 1.0e-30


def load_vul(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _check_static_invariants(var_a: dict, var_b: dict) -> tuple[bool, list[str]]:
    """Return (ok, messages). Hard-FAIL items live here."""
    msgs: list[str] = []
    ok = True

    species_a = list(var_a.get("species", []))
    species_b = list(var_b.get("species", []))
    if species_a != species_b:
        ok = False
        msgs.append(
            f"  species list MISMATCH: a has {len(species_a)}, b has {len(species_b)}"
        )
    else:
        msgs.append(f"  species list: identical ({len(species_a)} species)")

    nr_a, nr_b = var_a.get("nr"), var_b.get("nr")
    if nr_a != nr_b:
        ok = False
        msgs.append(f"  nr MISMATCH: a={nr_a} b={nr_b}")
    else:
        msgs.append(f"  nr          : identical ({nr_a})")

    y_ini_a = np.asarray(var_a["y_ini"], dtype=np.float64)
    y_ini_b = np.asarray(var_b["y_ini"], dtype=np.float64)
    if y_ini_a.shape != y_ini_b.shape:
        ok = False
        msgs.append(f"  y_ini shape MISMATCH: a={y_ini_a.shape} b={y_ini_b.shape}")
    else:
        max_abs = float(np.max(np.abs(y_ini_a - y_ini_b)))
        if max_abs == 0.0:
            msgs.append("  y_ini       : bit-exact")
        else:
            denom = np.maximum(np.abs(y_ini_b), EPSILON)
            max_rel = float(np.max(np.abs(y_ini_a - y_ini_b) / denom))
            ok = ok and max_rel < 1e-12
            msgs.append(
                f"  y_ini       : max abs diff {max_abs:.3e}, "
                f"max rel diff {max_rel:.3e}"
                + ("  (NOT bit-exact)" if max_abs != 0.0 else "")
            )

    return ok, msgs


def _check_time_alignment(var_a: dict, var_b: dict, *, time_tol_rel: float) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    t_a = float(var_a["t"])
    t_b = float(var_b["t"])
    dt_a = float(var_a["dt"])
    dt_b = float(var_b["dt"])
    longdy_a = float(var_a["longdy"])
    longdy_b = float(var_b["longdy"])

    if max(abs(t_a), abs(t_b)) == 0.0:
        rel = 0.0
    else:
        rel = abs(t_a - t_b) / max(abs(t_a), abs(t_b))

    msgs.append(f"  t       : a={t_a:.4e}  b={t_b:.4e}  rel diff={rel:.3e}")
    msgs.append(f"  dt      : a={dt_a:.4e}  b={dt_b:.4e}")
    msgs.append(f"  longdy  : a={longdy_a:.4e}  b={longdy_b:.4e}")
    aligned = rel <= time_tol_rel
    if not aligned:
        msgs.append(
            f"  -> t differs by {rel:.3e} > {time_tol_rel:.3e}; "
            "ymix diffs will conflate numerical drift with time evolution."
        )
    else:
        msgs.append(f"  -> t aligned within {time_tol_rel:.3e}")
    return aligned, msgs


def _per_species_log10_diff(
    ymix_a: np.ndarray,
    ymix_b: np.ndarray,
    species: list[str],
    *,
    floor: float,
) -> tuple[list[tuple[str, float, float, float]], dict[str, float]]:
    """Return per-species (name, max ymix, mean dex err, max dex err) and a summary."""
    rows: list[tuple[str, float, float, float]] = []
    log_a = np.log10(np.clip(ymix_a, EPSILON, None))
    log_b = np.log10(np.clip(ymix_b, EPSILON, None))
    n_species = ymix_a.shape[1]
    abundant_mean: list[float] = []
    abundant_max: list[float] = []
    for j in range(n_species):
        max_y = float(max(np.max(np.abs(ymix_a[:, j])), np.max(np.abs(ymix_b[:, j]))))
        if max_y < floor:
            continue
        delta = np.abs(log_a[:, j] - log_b[:, j])
        mean_dex = float(np.mean(delta))
        max_dex = float(np.max(delta))
        rows.append((str(species[j]), max_y, mean_dex, max_dex))
        abundant_mean.append(mean_dex)
        abundant_max.append(max_dex)
    rows.sort(key=lambda r: r[3], reverse=True)
    summary = {
        "n_abundant": float(len(rows)),
        "median_mean_dex": float(np.median(abundant_mean)) if abundant_mean else float("nan"),
        "median_max_dex": float(np.median(abundant_max)) if abundant_max else float("nan"),
        "max_max_dex": float(np.max(abundant_max)) if abundant_max else float("nan"),
        "mean_mean_dex": float(np.mean(abundant_mean)) if abundant_mean else float("nan"),
    }
    return rows, summary


def _atom_loss_table(var_a: dict, var_b: dict) -> list[str]:
    msgs = ["  atom        a              b              |Δ|"]
    atoms = list(var_a.get("atom_loss", {}).keys())
    if not atoms:
        return ["  (atom_loss not present)"]
    for atom in atoms:
        a = float(var_a["atom_loss"][atom])
        b = float(var_b["atom_loss"][atom])
        msgs.append(f"  {atom:>4s}  {a:>+14.4e}  {b:>+14.4e}  {abs(a - b):.3e}")
    return msgs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path_a", type=Path, help="First .vul file (typically the JAX run).")
    parser.add_argument("path_b", type=Path, help="Second .vul file (typically the master run).")
    parser.add_argument(
        "--floor",
        type=float,
        default=1.0e-10,
        help="Abundance floor: per-species ymix metrics are computed only "
             "for species whose max |ymix| in either run exceeds this "
             "(default 1e-10). Trace species below this are excluded so the "
             "headline isn't dominated by physically irrelevant relerrs.",
    )
    parser.add_argument(
        "--time-tol-rel",
        type=float,
        default=1.0e-2,
        help="Relative tolerance on the integration end-time. If "
             "|t_a - t_b| / max(t) exceeds this, the comparison is flagged "
             "as not apples-to-apples (default 1%%). Use --allow-time-mismatch "
             "to bypass.",
    )
    parser.add_argument(
        "--log10-tol",
        type=float,
        default=0.05,
        help="Pass threshold on the median per-species mean |Δ log10 ymix| "
             "(default 0.05 dex ~ 12%%). Mirrors the dex-style tolerance used "
             "in vulcan-emulator's mean_abs_log10_error.",
    )
    parser.add_argument(
        "--allow-time-mismatch",
        action="store_true",
        help="Don't fail on time misalignment; print metrics anyway and "
             "flag the result UNTRUSTED.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many species to show in the per-species ranking (default 15).",
    )
    args = parser.parse_args(argv)

    da = load_vul(args.path_a)
    db = load_vul(args.path_b)

    print(f"A: {args.path_a}")
    print(f"B: {args.path_b}")
    print()

    var_a = da["variable"]
    var_b = db["variable"]

    print("Static invariants:")
    static_ok, static_msgs = _check_static_invariants(var_a, var_b)
    for line in static_msgs:
        print(line)
    if not static_ok:
        print()
        print("HARD FAIL: static invariants don't match — different network or "
              "different initial state. Per-species ymix comparison is meaningless "
              "before this is fixed.")
        return 2

    print()
    print("Convergence state:")
    time_aligned, time_msgs = _check_time_alignment(
        var_a, var_b, time_tol_rel=args.time_tol_rel
    )
    for line in time_msgs:
        print(line)

    print()
    print("Mass conservation (atom_loss):")
    for line in _atom_loss_table(var_a, var_b):
        print(line)

    ymix_a = np.asarray(var_a["ymix"], dtype=np.float64)
    ymix_b = np.asarray(var_b["ymix"], dtype=np.float64)
    if ymix_a.shape != ymix_b.shape:
        print()
        print(f"HARD FAIL: ymix shape mismatch a={ymix_a.shape} b={ymix_b.shape}")
        return 2

    species = list(var_a.get("species", var_b.get("species", range(ymix_a.shape[1]))))
    rows, summary = _per_species_log10_diff(
        ymix_a, ymix_b, [str(s) for s in species], floor=args.floor
    )

    print()
    print(
        f"Per-species |Δ log10 ymix| over species with max ymix > {args.floor:.0e} "
        f"(top {args.top} by max dex error):"
    )
    print(f"  {'species':>10}  {'max ymix':>10}  {'mean dex':>10}  {'max dex':>10}")
    for sp, my, mean_dex, max_dex in rows[: args.top]:
        print(f"  {sp:>10}  {my:.3e}  {mean_dex:>10.4f}  {max_dex:>10.4f}")
    if not rows:
        print("  (no species above floor — adjust --floor)")

    print()
    print("Summary metrics (apples-to-apples headline):")
    print(f"  n_species above floor : {int(summary['n_abundant'])}")
    print(f"  mean   mean |Δlog10|  : {summary['mean_mean_dex']:.4f} dex")
    print(f"  median mean |Δlog10|  : {summary['median_mean_dex']:.4f} dex")
    print(f"  median max  |Δlog10|  : {summary['median_max_dex']:.4f} dex")
    print(f"  max    max  |Δlog10|  : {summary['max_max_dex']:.4f} dex")

    if not time_aligned and not args.allow_time_mismatch:
        print()
        print(
            "FAIL: integration end-times differ by more than "
            f"{args.time_tol_rel:.0%}. Re-run both VULCANs with matching "
            "stop conditions (same `runtime`, same `count_max`, same "
            "convergence tolerances in vulcan_cfg.py), or pass "
            "--allow-time-mismatch to inspect anyway."
        )
        return 1

    median_metric = summary["median_mean_dex"]
    headline = (
        "(UNTRUSTED — t mismatched)" if not time_aligned else "(apples-to-apples)"
    )
    if not np.isfinite(median_metric):
        print()
        print(f"FAIL {headline}: no abundant species — adjust --floor.")
        return 1
    if median_metric < args.log10_tol:
        print()
        print(
            f"PASS {headline}: median mean |Δlog10 ymix| = "
            f"{median_metric:.4f} dex < tol {args.log10_tol:.4f} dex"
        )
        return 0
    print()
    print(
        f"FAIL {headline}: median mean |Δlog10 ymix| = "
        f"{median_metric:.4f} dex >= tol {args.log10_tol:.4f} dex"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
