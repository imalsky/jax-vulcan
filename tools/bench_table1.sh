#!/usr/bin/env bash
# Table 1 benchmark: VULCAN 2.0 vs VULCAN-JAX 3.0, free convergence, one host.
#
# WHY THIS SCRIPT EXISTS
# A 2026-07 attempt to regenerate Table 1 produced a 16.8x wall-clock speedup that had to be
# retracted: VULCAN 2.0 consumed only 363 s of CPU across 1238 s of wall (ratio 0.29) because the
# machine was throttled/oversubscribed — plausibly a closed laptop lid. Its wall time was therefore
# not a measurement of VULCAN 2.0. Step counts were unaffected and remained valid.
#
# So this script GUARDS the measurement instead of trusting it:
#   - refuses to start if load average is high
#   - runs under `caffeinate` so the machine cannot sleep or downclock mid-run
#   - records user+sys alongside real for every run
#   - REFUSES to report a speedup unless VULCAN 2.0's cpu/wall is close to 1.0
# A run that fails the guard prints step counts (which are load-independent) and no timing.
#
# Usage:  tools/bench_table1.sh [HD189|HD209|W39b] ...     (default: all three)

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JAX_REPO="$PROJECT/VULCAN-JAX"
MASTER="$PROJECT/VULCAN-master"
PY="${BENCH_PYTHON:-python}"
OUT="${BENCH_OUT:-/tmp/bench_table1_$(date +%Y%m%d_%H%M%S)}"
CPU_WALL_MIN="${CPU_WALL_MIN:-0.85}"   # VULCAN 2.0 is single-threaded: cpu/wall must be ~1
# Load threshold scaled to the machine: a bare "2.0" is wrong on a 12-core box,
# where load 6 still leaves half the cores idle. Half the cores is the default
# ceiling. This is only a pre-flight heuristic — the authoritative check is the
# cpu/wall ratio measured per run below, which catches throttling that load
# average cannot see (e.g. a closed laptop lid).
_NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null \
         || nproc 2>/dev/null || echo 4)"
case "$_NCPU" in ''|*[!0-9]*) _NCPU=4 ;; esac
LOAD_MAX="${LOAD_MAX:-$(awk "BEGIN{printf \"%.1f\", $_NCPU/2}")}"

mkdir -p "$OUT"
PLANETS=("${@:-HD189 HD209 W39b}")
read -r -a PLANETS <<< "${PLANETS[*]}"

# --- network / atom_list per planet (import-locked in VULCAN-JAX, must be set BEFORE import) ---
net_for() { case "$1" in
    W39b) echo "thermo/SNCHO_photo_network.txt" ;;
    *)    echo "thermo/NCHO_photo_network.txt" ;;
  esac; }
atoms_for() { case "$1" in
    W39b) echo "H,O,C,N,S" ;;
    *)    echo "H,O,C,N" ;;
  esac; }

# ---------------------------------------------------------------- preflight
# Load average is a pre-flight courtesy check only, and it is not always
# readable (sysctl is blocked in some sandboxes). Never let its absence abort the
# run — the authoritative gate is the per-run cpu/wall ratio below.
load1="$(uptime 2>/dev/null | sed -n 's/.*averages*: *\([0-9.]*\).*/\1/p')"
if [ -z "${load1:-}" ]; then
  load1="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
fi
if [ -z "${load1:-}" ]; then
  echo "load average: unavailable (sysctl/uptime blocked) — skipping the"
  echo "  pre-flight load check; the per-run cpu/wall guard still applies."
else
  echo "load average (1 min): $load1   (threshold $LOAD_MAX on $_NCPU cores)"
  if awk "BEGIN{exit !($load1 > $LOAD_MAX)}"; then
    echo "ABORT: machine is busy. Timing measured now would not be reproducible." >&2
    echo "       Close other work and retry, or raise LOAD_MAX to accept the noise." >&2
    exit 1
  fi
fi
command -v caffeinate >/dev/null && CAF="caffeinate -dims" || CAF=""
[ -n "$CAF" ] || echo "WARNING: caffeinate absent; the machine may sleep or downclock mid-run."

# time a command, emitting "real user sys" to stdout
timed() { local log="$1"; shift
  # Time the child in Python rather than with `/usr/bin/time -p`: getting time's
  # own stderr into a file while ALSO routing the child's stderr to the log needs
  # fd juggling that proved fragile here (the report kept reaching the terminal
  # and .time stayed empty, blanking every summary field). resource.getrusage on
  # RUSAGE_CHILDREN gives the same user/sys numbers with no redirection puzzle.
  # Emits "real user sys" on stdout; child output goes to $log.
  # shellcheck disable=SC2086  # $CAF is an intentional word-split prefix
  "$PY" - "$log" $CAF "$@" <<'TIMEEOF'
import resource, subprocess, sys, time
log, cmd = sys.argv[1], sys.argv[2:]
t0 = time.time()
with open(log, "wb") as fh:
    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"{wall:.2f} {ru.ru_utime:.2f} {ru.ru_stime:.2f}")
TIMEEOF
}

printf '%-8s %-12s %8s %8s %8s %9s %7s\n' PLANET CODE REAL USER SYS CPU/WALL STEPS | tee "$OUT/summary.txt"

for P in "${PLANETS[@]}"; do
  # ---------------- VULCAN 2.0, in an isolated copy so the oracle tree is never written to
  W="$OUT/master_$P"; mkdir -p "$W/output" "$W/plot"
  for d in atm thermo fastchem_vulcan; do ln -sfn "$MASTER/$d" "$W/$d"; done
  cp "$MASTER"/*.py "$W/" 2>/dev/null || true
  "$PY" - "$W/vulcan_cfg.py" "$(net_for "$P")" <<'PYEOF'
import re, sys, pathlib
p, net = pathlib.Path(sys.argv[1]), sys.argv[2]
t = p.read_text()
t = re.sub(r"^network\s*=.*$", f"network = '{net}'", t, flags=re.M)
for k in ("use_print_prog", "use_live_plot", "use_live_flux", "use_plot_end", "use_plot_evo"):
    t = re.sub(rf"^{k}\s*=.*$", f"{k} = False", t, flags=re.M)
# master ships wall_clock_max = 1800 s, which ABORTS a healthy HD189 benchmark
# mid-integration ("Wall-clock budget exceeded") — on this laptop that run needs
# ~2000+ s even at cpu/wall 0.97. A benchmark must be bounded by convergence or
# count_max, never by a wall-clock timer, or the measurement is of the timer.
if re.search(r"^wall_clock_max\s*=", t, flags=re.M):
    t = re.sub(r"^wall_clock_max\s*=.*$", "wall_clock_max = 1.e9", t, flags=re.M)
else:
    t += "\nwall_clock_max = 1.e9\n"
p.write_text(t)
PYEOF
  read -r m_real m_user m_sys < <(cd "$W" && timed "$W/run.log" "$PY" vulcan.py)
  m_steps=$(grep -oE 'successfully run to steady-state with [0-9]+ steps' "$W/run.log" | grep -oE '[0-9]+' | head -1)
  m_ratio=$(awk "BEGIN{printf \"%.2f\", ($m_user+$m_sys)/$m_real}")
  printf '%-8s %-12s %8s %8s %8s %9s %7s\n' "$P" "VULCAN2.0" "$m_real" "$m_user" "$m_sys" "$m_ratio" "${m_steps:-?}" | tee -a "$OUT/summary.txt"

  # ---------------- VULCAN-JAX 3.0, fresh subprocess + empty XLA cache (Table 1's protocol)
  J="$OUT/jax_$P"; mkdir -p "$J"
  read -r j_real j_user j_sys < <(cd "$J" && \
    JAX_COMPILATION_CACHE_DIR="$J/xlacache" \
    VULCAN_JAX_NETWORK="$JAX_REPO/src/vulcan_jax/$(net_for "$P")" \
    VULCAN_JAX_ATOM_LIST="$(atoms_for "$P")" \
    timed "$J/run.log" "$PY" -m vulcan_jax.vulcan_jax_cli --config "$P")
  j_steps=$(grep -oE 'successfully run to steady-state with [0-9]+ steps' "$J/run.log" | grep -oE '[0-9]+' | head -1)
  j_ratio=$(awk "BEGIN{printf \"%.2f\", ($j_user+$j_sys)/$j_real}")
  printf '%-8s %-12s %8s %8s %8s %9s %7s\n' "$P" "VULCAN-JAX" "$j_real" "$j_user" "$j_sys" "$j_ratio" "${j_steps:-?}" | tee -a "$OUT/summary.txt"

  # ---------------- the guard: only report a speedup if master actually got its core
  if awk "BEGIN{exit !($m_ratio < $CPU_WALL_MIN)}"; then
    {
      echo "  !! TIMING REJECTED for $P: VULCAN 2.0 cpu/wall = $m_ratio < $CPU_WALL_MIN."
      echo "     It waited instead of computing (throttling, sleep, or contention), so its"
      echo "     $m_real s wall is not a measurement. Steps are still valid:"
      echo "     VULCAN 2.0 ${m_steps:-?} steps vs VULCAN-JAX ${j_steps:-?} steps."
    } | tee -a "$OUT/summary.txt"
  else
    awk "BEGIN{printf \"  speedup (wall): %.2fx   steps: %s vs %s\n\", $m_real/$j_real, \"${m_steps:-?}\", \"${j_steps:-?}\"}" \
      | tee -a "$OUT/summary.txt"
  fi
done

echo
echo "logs + summary: $OUT"
echo "Step counts are load-independent and always trustworthy; wall times only when the guard passed."
