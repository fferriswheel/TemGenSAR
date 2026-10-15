#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cjk/Search_Reconmendation
OUTPUT=$ROOT/stage3/runs/temporal_rolling_r4a_cross
mkdir -p "$OUTPUT"
cd "$ROOT"

# Update windows W3, W5, and W9 evaluate W4, W6, and W10 respectively.
for window in W3 W5 W9; do
  bash stage3/scripts/run_temporal_rolling_instance_eval.sh \
    4 "$window" Selective "$OUTPUT/${window}_SelectiveCheckpoint_FixedMapping" Fixed \
    >"$OUTPUT/${window}_SF.log" 2>&1 & p1=$!
  bash stage3/scripts/run_temporal_rolling_instance_eval.sh \
    5 "$window" Fixed "$OUTPUT/${window}_FixedCheckpoint_SelectiveMapping" Selective \
    >"$OUTPUT/${window}_FS.log" 2>&1 & p2=$!
  wait "$p1"
  wait "$p2"
done
