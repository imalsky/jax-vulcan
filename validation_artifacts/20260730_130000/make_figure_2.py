#!/usr/bin/env python
"""Figure 2: end-to-end HD 189733 b validation of the hybrid molecular diffusion.

Three panels:
  A  longdy / longdydt vs accepted step for the hybrid run, with the
     upwind -> central phase flip marked and the termination reason annotated.
  B  final mixing-ratio profiles for representative species under four schemes.
  C  hybrid-vs-central difference, above a declared abundance floor.

NOTE ON THE "VULCAN 2" CURVE. This figure does NOT contain an upstream
VULCAN 2.0 run. The sibling ../VULCAN-master checkout is not pristine upstream
(it carries VULCAN-JAX's own stall detector and conver_ignore list, see
git_provenance.txt), so a curve from it would not be an independent code. The
"VULCAN 2 parity" curve here is VULCAN-JAX running its VULCAN 2 parity config,
whose convergence knobs match fetched exoclime/VULCAN master exactly. It is
labelled accordingly and must not be described as an upstream result.

Usage:  python make_figure_2.py <outdir>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHUNK = 20  # accepted steps between host syncs when tracing longdy
FLOOR = 1e-12  # declared abundance floor for the difference panel
SPECIES = ["H2O", "CH4", "CO", "HCN", "C2H2", "H"]


def _integ(cfg):
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop
    from vulcan_jax.state import RunState

    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
    rs = RunState.with_pre_loop_setup(cfg)
    return integ, integ.prepare_runstate(rs)


def run_plain(cfg):
    """Run to termination in one shot; return the final carry."""
    integ, (state, atm_static) = _integ(cfg)
    return integ._runner(state, atm_static)


def run_traced(cfg):
    """Run in chunks, recording longdy / longdydt / phase per chunk.

    `termination_reason == 0` is exactly "stopped on the chunk cap, still
    running" (the runner records the code on every return), so it is the
    correct loop-exit test -- no need to re-derive it from the counters.
    """
    import jax.numpy as jnp

    integ, (state, atm_static) = _integ(cfg)
    trace = []
    while True:
        target = min(int(state.accept_count) + CHUNK, int(state.count_max_dyn) + 1)
        state = state._replace(chunk_target=jnp.int32(target))
        state = integ._runner(state, atm_static)
        trace.append(
            (
                int(state.accept_count),
                float(state.longdy),
                float(state.longdydt),
                float(state.hybrid_use_vm),
            )
        )
        if int(state.termination_reason) != 0:
            break
    return state, np.array(trace, dtype=np.float64)


def cfg_for(name, **over):
    from vulcan_jax.config import load_config

    c = load_config(name)
    for k, v in over.items():
        if not hasattr(c, k):
            raise SystemExit(f"unknown key {k}")
        setattr(c, k, v)
    c.use_print_prog = False
    c.use_live_plot = False
    c.use_live_flux = False
    return c


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)

    from vulcan_jax import chem_funs

    spec = list(chem_funs.spec_list)
    idx = {s: spec.index(s) for s in SPECIES if s in spec}

    # -- the four runs ------------------------------------------------------
    # VULCAN 2 parity: central difference, upstream master's conver_ignore,
    # stall fallback off. This is HD189.yaml as shipped after this session.
    parity = run_plain(cfg_for("HD189"))
    upwind = run_plain(cfg_for("HD189", use_vm_mol=True, use_hybrid_vm_mol=False))
    hybrid, trace = run_traced(cfg_for("HD189", use_vm_mol=True, use_hybrid_vm_mol=True))
    v3, _ = run_traced(cfg_for("HD189_vulcan3"))

    runs = {
        "VULCAN 2 parity (central)": parity,
        "pure upwind": upwind,
        "hybrid (upwind -> central)": hybrid,
        "VULCAN 3 preset": v3,
    }
    for label, st in runs.items():
        print(
            f"{label:32s} steps={int(st.accept_count):6d} "
            f"reason={int(st.termination_reason)} longdy={float(st.longdy):.5g} "
            f"hybrid_use_vm={float(st.hybrid_use_vm):.1f}"
        )

    from vulcan_jax.config import load_config
    from vulcan_jax.state import RunState

    rs_ref = RunState.with_pre_loop_setup(load_config("HD189"))
    p_bar = np.asarray(rs_ref.atm.pco, dtype=np.float64) / 1e6

    ymix = {k: np.asarray(v.ymix, dtype=np.float64) for k, v in runs.items()}

    # -- switch step --------------------------------------------------------
    phase = trace[:, 3]
    flip_rows = np.where(phase < 0.5)[0]
    switch_step = float(trace[flip_rows[0], 0]) if flip_rows.size else float("nan")

    # -- figure -------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # Panel A ---------------------------------------------------------------
    axA.semilogy(trace[:, 0], trace[:, 1], color="#1f77b4", lw=1.5, label="longdy")
    axA.semilogy(trace[:, 0], trace[:, 2], color="#d62728", lw=1.2, ls="--",
                 label="longdydt")
    cfg_ref = load_config("HD189")
    axA.axhline(float(cfg_ref.yconv_min), color="0.35", lw=0.9, ls=":")
    axA.set_ylim(1e-9, 1e6)
    if np.isfinite(switch_step):
        axA.axvspan(trace[0, 0], switch_step, color="#d62728", alpha=0.07)
        axA.axvspan(switch_step, trace[-1, 0], color="#1f77b4", alpha=0.07)
        axA.axvline(switch_step, color="k", lw=1.1)
        axA.annotate(
            f"upwind $\\rightarrow$ central\nat step {int(switch_step)}",
            xy=(switch_step, 3e5), xytext=(-8, 0), textcoords="offset points",
            fontsize=7.5, ha="right", va="top", color="k",
        )
        axA.text(switch_step * 0.5, 2e-8, "upwind phase", fontsize=8,
                 ha="center", color="#8b2020")
        axA.text((switch_step + trace[-1, 0]) * 0.5, 2e-8, "central phase",
                 fontsize=8, ha="center", color="#12537a")
    axA.text(trace[-1, 0], float(cfg_ref.yconv_min) * 1.6, "yconv_min",
             fontsize=7, color="0.3", ha="right")
    reason_txt = {1: "converged", 2: "runtime cap", 3: "step cap",
                  4: "stall fallback"}.get(int(hybrid.termination_reason), "?")
    axA.set_xlabel("accepted step")
    axA.set_ylabel("convergence metric")
    axA.set_title(f"A  hybrid run: '{reason_txt}' at {int(hybrid.accept_count)} steps",
                  fontsize=9)
    axA.legend(frameon=False, fontsize=8, loc="center left")

    # Panel B ---------------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0, 0.82, len(idx)))
    styles = {
        "VULCAN 2 parity (central)": dict(ls="-", lw=2.4, alpha=0.35),
        "pure upwind": dict(ls=":", lw=1.5, alpha=1.0),
        "hybrid (upwind -> central)": dict(ls="--", lw=1.2, alpha=1.0),
    }
    for (sp, i), col in zip(idx.items(), colors):
        for label, sty in styles.items():
            axB.loglog(np.maximum(ymix[label][:, i], 1e-20), p_bar, color=col, **sty)
        axB.plot([], [], color=col, lw=2.0, label=sp)
    for label, sty in styles.items():
        axB.plot([], [], color="0.25", label=label, **sty)
    axB.invert_yaxis()
    axB.set_xlim(1e-14, 3e0)
    axB.set_xlabel("mixing ratio")
    axB.set_ylabel("pressure (bar)")
    axB.set_title("B  final profiles: thick = central, dots = upwind, dash = hybrid",
                  fontsize=9)
    axB.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower left")

    # Panel C ---------------------------------------------------------------
    # Both differences on one axis: the whole point is that the hybrid returns
    # the CENTRAL answer (1e-4 level) while pure upwind does not (order unity).
    stats = {}
    styleC = {
        "pure upwind": dict(color="#d62728", ls=":", lw=1.5, alpha=0.85),
        "hybrid (upwind -> central)": dict(color="#1f77b4", ls="-", lw=1.2, alpha=0.85),
    }
    for label, sty in styleC.items():
        allrel = []
        for sp, i in idx.items():
            a = ymix["VULCAN 2 parity (central)"][:, i]
            b = ymix[label][:, i]
            m = (a > FLOOR) & (b > FLOOR)
            rel = np.full_like(a, np.nan)
            rel[m] = np.abs(b[m] - a[m]) / a[m]
            allrel.append(rel[m])
            axC.loglog(rel, p_bar, **sty)
        cat = np.concatenate([r for r in allrel if r.size])
        stats[label] = (float(np.median(cat)), float(np.max(cat)))
    for label, sty in styleC.items():
        med, mx = stats[label]
        axC.plot([], [], label=f"{label}\n  median {med:.1e}, max {mx:.1e}", **sty)
    axC.invert_yaxis()
    axC.set_xlim(1e-12, 1e1)
    axC.set_xlabel(f"|scheme - central| / central   (above {FLOOR:.0e})")
    axC.set_ylabel("pressure (bar)")
    axC.set_title("C  the hybrid returns the central fixed point", fontsize=9)
    axC.legend(frameon=False, fontsize=6.8, loc="lower left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figure_2_hd189_hybrid_validation.{ext}", bbox_inches="tight")

    # -- plotted data as CSV ------------------------------------------------
    csv = outdir / "figure_2_hd189_hybrid_validation_data.csv"
    with csv.open("w") as fh:
        fh.write("# Panel A: convergence trace of the hybrid run\n")
        fh.write("panel,accepted_step,longdy,longdydt,hybrid_use_vm\n")
        for r in trace:
            fh.write(f"A,{int(r[0])},{r[1]:.10e},{r[2]:.10e},{r[3]:.1f}\n")
        fh.write("# Panel B/C: final mixing ratios per scheme\n")
        fh.write("panel,species,p_bar,"
                 + ",".join(k.replace(",", ";") for k in runs) + "\n")
        for sp, i in idx.items():
            for k in range(len(p_bar)):
                vals = ",".join(f"{ymix[lbl][k, i]:.10e}" for lbl in runs)
                fh.write(f"B,{sp},{p_bar[k]:.10e},{vals}\n")

    summary = {
        "runs": {
            lbl: {
                "accept_steps": int(st.accept_count),
                "termination_reason": int(st.termination_reason),
                "longdy": float(st.longdy),
                "longdydt": float(st.longdydt),
                "hybrid_use_vm_final": float(st.hybrid_use_vm),
            }
            for lbl, st in runs.items()
        },
        "switch_step": switch_step,
        "abundance_floor": FLOOR,
        "diff_vs_central": {k: {"median": v[0], "max": v[1]} for k, v in stats.items()},
        "species_plotted": list(idx),
    }
    (outdir / "figure_2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
