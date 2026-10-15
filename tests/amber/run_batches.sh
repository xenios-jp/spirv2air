#!/bin/sh
# Runs every compute-only Amber script (PIPELINE compute scripts and legacy VkScript [compute shader] scripts) of a
# VK-GL-CTS checkout in batches of BATCH_SIZE, taking the GPU lock once per batch so other jobs can interleave.
# usage: run_batches.sh AMBER_RUNNER CTS_AMBER_DIR OUT.csv LOCKFILE [BATCH_SIZE] [TIMEOUT_SEC]
# Extra runner flags (for example --inline) go in $AMBER_RUNNER_ARGS.
set -eu
runner=$1 cts=$2 csv=$3 lock=$4 batch=${5:-60} timeout=${6:-30}
rm -f "$csv"
offset=0
while :; do
  n=$(nice -n 20 taskpolicy -b lockf -t 3600 "$lock" "$runner" --cts "$cts" --all-compute --offset "$offset" \
        --limit "$batch" --timeout "$timeout" --csv "$csv" ${AMBER_RUNNER_ARGS:-} | tee /dev/stderr |
        sed -n 's/^-- \([0-9]*\) scripts.*/\1/p')
  [ "${n:-0}" -ge "$batch" ] || break
  offset=$((offset + batch))
done
