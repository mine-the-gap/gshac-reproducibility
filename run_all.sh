#!/usr/bin/env bash
#
# run_all.sh — full re-run of BOTH tracks (full-scale + fast), sequentially.
#
#   ./run_all.sh                              # foreground, live output + run.log
#   setsid ./run_all.sh >/dev/null 2>&1 &     # detached (survives logout); run.log still written
#
# Reminder: free RAM first  ->  sudo systemctl stop gdm
#   (the 20 km GeoNames graph peaks at ~28 GiB on a 30 GiB box).
#
# NOT included: `make paper` (PDF compile) — the prose still holds the OLD numbers
# until the next Claude session revises them; and `make budget budget-fast`
# (~+4 h, README compute-budget table only). Run those separately if you want them.

set -euo pipefail

# Always operate from the repo root (this script's directory).
cd "$(dirname "$(readlink -f "$0")")"

LOG=run.log
: > "$LOG"                       # start a fresh log

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

log "=== GSHAC full experiment re-run starting ==="
log "Reminder: free RAM first (sudo systemctl stop gdm) — 20 km GeoNames graph needs ~28 GiB."
START=$SECONDS

# Provision only if needed. This repo's .venv is already fully set up (python + all deps
# + the C extension), but it was created WITHOUT a working pip, so `make setup` (which
# calls .venv/bin/pip) fails and is unnecessary. If the venv imports cleanly we skip setup
# via `-o setup` (tell make it is up-to-date so reproduce-fast's prerequisite does not drag
# it in); otherwise we run the full setup to bootstrap a fresh venv.
#
# Pipeline (one make invocation, stops on first error):
#   experiments    -> results/*.csv        (the numbers the paper cites)
#   figures/tables -> paper/fig_*.pdf, paper/tab_*.tex
#   reproduce-fast -> results_fast/*.csv
#   compare        -> fast-vs-full alignment check
run_pipeline() {
    if .venv/bin/python3 -c 'import numpy, scipy, sklearn, fastcluster, gshac' 2>/dev/null; then
        log "venv OK — skipping 'make setup' (this venv has no pip; setup is unneeded)."
        make -o setup experiments figures tables reproduce-fast compare
    else
        log "venv missing/incomplete — running full 'make setup' first."
        make setup experiments figures tables reproduce-fast compare
    fi
}
if run_pipeline 2>&1 | tee -a "$LOG"; then
    STATUS=ok
else
    STATUS=fail
fi

ELAPSED=$(( SECONDS - START ))
log "=== finished in $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m — status: ${STATUS} ==="

if [ "$STATUS" = ok ]; then
    log "Next: run 'claude --continue' in this directory, then say:"
    log "  resumed — verify the manuscript numbers against the new run"
else
    log "A step FAILED — inspect ${LOG} (search for 'Error' or 'Traceback')."
    exit 1
fi
