#!/bin/bash
# Phase 3 + Phase 4 controlled comparison matrix.
#
# Each case runs in a fresh subprocess (clean JAX state, per-process timing).
# Physical inputs are identical within a config; only `conver_ignore` and
# `use_conv_stall` vary.
#
# Lists compared:
#   MASTER   []            -- fetched exoclime/VULCAN master + shami-EEG master
#   VM       ['HC3N']      -- fetched shami-EEG vm_branch (VULCAN 3)
#   LOCAL13  13 + HC3N     -- the list currently in the tree (no upstream origin)
set -u
cd "$(dirname "$0")/../.." || exit 1
unset PYTHONSAFEPATH
PY=/opt/homebrew/Caskroom/miniforge/base/envs/vulcan/bin/python
ART=validation_artifacts/20260730_130000
OUT="$ART/runs"
LOG="$ART/runs/phase3_results.jsonl"
mkdir -p "$OUT"
: > "$LOG"

MASTER='[]'
VM='["HC3N"]'
LOCAL13='["C6H6","C2H2","C6H5","C2H","C2H4","C2H5","C2H6","C3H2","C3H3","C4H5","CH2NH","CH3NH2","H2CCO","HC3N"]'

run() {
  local tag="$1" cfg="$2" ov="$3"
  echo "=== [$(date +%H:%M:%S)] $tag  cfg=$cfg  overrides=$ov ==="
  $PY "$ART/run_case.py" "$cfg" "$ov" "$OUT/${tag}.npz" > "$OUT/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc (see $OUT/${tag}.log)"
    echo "{\"tag\":\"$tag\",\"config\":\"$cfg\",\"FAILED\":true,\"rc\":$rc}" >> "$LOG"
    return
  fi
  grep '^RESULT ' "$OUT/${tag}.log" | sed 's/^RESULT //' | \
    $PY -c "import sys,json; d=json.load(sys.stdin); d['tag']='$tag'; print(json.dumps(d))" >> "$LOG"
  grep '^RESULT ' "$OUT/${tag}.log" | sed 's/^RESULT //' | \
    $PY -c "
import sys, json
d = json.load(sys.stdin)
c = d.get('controlling_cell') or {}
print('  steps=%d reject=%d end_case=%d reason=%d longdy=%.4g t=%.3g ctrl=%s@%.2eBar loss=%.2e %.1fs'
      % (d['accept_steps'], d['delta_rejects'], d['end_case'], d['termination_reason'],
         d['longdy'], d['sim_time_s'], c.get('species','?'), c.get('p_bar',float('nan')),
         d['max_abs_atom_loss'], d['t_run_s']))
"
}

for cfg in HD189 W39b HD209 default; do
  run "${cfg}_master_stalloff"  "$cfg" "{\"conver_ignore\": $MASTER,  \"use_conv_stall\": false}"
  run "${cfg}_vm_stalloff"      "$cfg" "{\"conver_ignore\": $VM,      \"use_conv_stall\": false}"
  run "${cfg}_local13_stalloff" "$cfg" "{\"conver_ignore\": $LOCAL13, \"use_conv_stall\": false}"
  run "${cfg}_local13_stallon"  "$cfg" "{\"conver_ignore\": $LOCAL13, \"use_conv_stall\": true}"
done

run "Earth_master_stalloff" Earth "{\"conver_ignore\": $MASTER, \"use_conv_stall\": false}"
run "Earth_vm_stalloff"     Earth "{\"conver_ignore\": $VM,     \"use_conv_stall\": false}"
run "Earth_vm_stallon"      Earth "{\"conver_ignore\": $VM,     \"use_conv_stall\": true}"
run "K2-18b_asshipped"      K2-18b '{}'

echo "=== [$(date +%H:%M:%S)] MATRIX COMPLETE ==="
wc -l "$LOG"
