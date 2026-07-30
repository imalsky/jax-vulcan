#!/bin/bash
# W39b re-run: the first attempt failed because the reaction network is
# import-locked and W39b uses SNCHO, not the default NCHO. run_case.py now sets
# $VULCAN_JAX_NETWORK before importing vulcan_jax.
set -u
cd "$(dirname "$0")/../.." || exit 1
unset PYTHONSAFEPATH
PY=/opt/homebrew/Caskroom/miniforge/base/envs/vulcan/bin/python
ART=validation_artifacts/20260730_130000
OUT="$ART/runs"
LOG="$OUT/w39b_results.jsonl"
: > "$LOG"
MASTER='[]'
VM='["HC3N"]'
LOCAL13='["C6H6","C2H2","C6H5","C2H","C2H4","C2H5","C2H6","C3H2","C3H3","C4H5","CH2NH","CH3NH2","H2CCO","HC3N"]'
run() {
  local tag="$1" ov="$2"
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  $PY "$ART/run_case.py" W39b "$ov" "$OUT/${tag}.npz" > "$OUT/${tag}.log" 2>&1 || {
    echo "  FAILED (see $OUT/${tag}.log)"; return; }
  grep '^RESULT ' "$OUT/${tag}.log" | sed 's/^RESULT //' | tee -a "$LOG" | $PY -c "
import sys, json
d = json.load(sys.stdin); c = d.get('controlling_cell') or {}
print('  steps=%d end_case=%d reason=%d longdy=%.4g ctrl=%s@%.2ebar loss=%.2e %.1fs'
      % (d['accept_steps'], d['end_case'], d['termination_reason'], d['longdy'],
         c.get('species','?'), c.get('p_bar',float('nan')), d['max_abs_atom_loss'], d['t_run_s']))"
}
run W39b_master_stalloff  "{\"conver_ignore\": $MASTER,  \"use_conv_stall\": false}"
run W39b_vm_stalloff      "{\"conver_ignore\": $VM,      \"use_conv_stall\": false}"
run W39b_local13_stalloff "{\"conver_ignore\": $LOCAL13, \"use_conv_stall\": false}"
run W39b_master_stallon   "{\"conver_ignore\": $MASTER,  \"use_conv_stall\": true}"
echo "=== [$(date +%H:%M:%S)] W39b COMPLETE ==="
