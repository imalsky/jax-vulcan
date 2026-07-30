#!/bin/bash
# Phase 11 timings. Run on a QUIET machine with the laptop lid OPEN.
#
# Each measurement is a fresh subprocess. The persistent XLA cache directory is
# the only thing that differs between the cold and warm rows:
#   cold  -- empty cache dir, so the run pays the full XLA compile
#   warm  -- same dir, now populated, so compile is served from disk
# Setup (FastChem + network parse + pre-loop) is reported separately by
# run_case.py and is unaffected by the XLA cache.
#
# Three measured warm repeats per configuration, all reported, plus median and
# minimum. Nothing is tuned after seeing a result.
set -u
cd "$(dirname "$0")/../.." || exit 1
unset PYTHONSAFEPATH
PY=/opt/homebrew/Caskroom/miniforge/base/envs/vulcan/bin/python
ART=validation_artifacts/20260730_130000
OUT="$ART/runs/timings"
mkdir -p "$OUT"
CSV="$ART/timings.csv"

echo "config,mode,repeat,wall_total_s,t_import_s,t_setup_s,t_run_s,accept_steps,delta_rejects,termination_reason,longdy" > "$CSV"

measure() {
  local cfg="$1" tag="$2" mode="$3" rep="$4" cache="$5"
  local t0 t1
  t0=$(python3 -c "import time;print(time.time())")
  JAX_COMPILATION_CACHE_DIR="$cache" $PY "$ART/run_case.py" "$cfg" '{}' \
      "$OUT/${tag}_${mode}_${rep}.npz" > "$OUT/${tag}_${mode}_${rep}.log" 2>&1
  local rc=$?
  t1=$(python3 -c "import time;print(time.time())")
  if [ $rc -ne 0 ]; then echo "  FAILED $tag $mode $rep"; return; fi
  grep '^RESULT ' "$OUT/${tag}_${mode}_${rep}.log" | sed 's/^RESULT //' | \
    $PY -c "
import sys, json
d = json.load(sys.stdin)
wall = $t1 - $t0
print(f\"$cfg,$mode,$rep,{wall:.2f},{d['t_import_s']},{d['t_setup_s']},{d['t_run_s']},\"
      f\"{d['accept_steps']},{d['delta_rejects']},{d['termination_reason']},{d['longdy']:.6g}\")
" | tee -a "$CSV"
}

for spec in "HD189:v2parity" "HD189_vulcan3:v3hybrid"; do
  cfg="${spec%%:*}"; tag="${spec##*:}"
  cache=$(mktemp -d "/tmp/jaxcache_${tag}_XXXX")
  echo "=== $cfg ($tag) : COLD (empty XLA cache at $cache) ==="
  measure "$cfg" "$tag" cold 1 "$cache"
  echo "=== $cfg ($tag) : WARM x3 (same cache) ==="
  for r in 1 2 3; do measure "$cfg" "$tag" warm "$r" "$cache"; done
  rm -rf "$cache"
done

echo
echo "=== timings.csv ==="
cat "$CSV"
