#!/bin/sh
# Grow the trunk chain one record at a time, keeping only records that pass the gate.
#
# WHY THE REVERT EXISTS, which is the whole point of this script. The miner appends the
# best attempt EVEN WHEN IT FAILS the gate -- deliberately, so a measurement run cannot
# hide its own failures (see the WARNING in train_gpt2_btm_trunk.c). But the harness then
# folds every record in the file regardless of verdict, whereas a real chain folds only
# accepted ones. So one appended failure puts the chain in a state no real chain could
# reach, and every measurement after it is against a base that includes a rejected delta.
# This backs the file up before each attempt and restores it on anything but a PASS.
#
# WHY ONE ATTEMPT PER RECORD, not the miner's best-of-n. Best-of-n appends the max of n
# draws from an estimator noisier than its mean: at the measured sd/mean of 0.61, best-of-2
# overstates a record's improvement by ~23%. That is harmless if you only want the chain
# longer and fatal if you want the improvement-vs-depth curve, which is the reason to run
# this at all.
#
# THE NUMBER TO WATCH is the ratio column: improvement / gate-needs. Holding above ~1.3
# means the exponent is calibrated. Drifting toward 1.0 means improvement is still decaying
# faster than the bar follows it, and the answer is multi-window scoring on a smaller model
# rather than another exponent change (see the calibration block in llmc/btmmoe.h).
#
# usage: dev/btm_grow_chain.sh [target_records] [steps] [max_attempts]
#   target_records is how many to ADD to whatever is already in the chain file.
#   defaults: 8 records, 480 steps, 40 attempts
#   at L=8/E=16/H=192/K=3, ~220 steps is what fits a 16-minute slot -- 480 was ~4x it
# Run from the repo root. Ctrl-C is safe: the chain file is only ever left at a state
# whose records all passed.

set -u

TARGET=${1:-8}
STEPS=${2:-480}
MAX_ATTEMPTS=${3:-40}

CHAIN=btm-cp-trunk.bin
BAK=btm-cp-trunk.bak.$$
MINER=./train_gpt2_btm_trunk

[ -x "$MINER" ] || { echo "no $MINER -- run: make train_gpt2_btm_trunk" >&2; exit 1; }

# A missing chain file is legitimate (the genesis record) but worth saying out loud, since
# the usual cause is having moved it aside and forgotten.
#
# NOTE ON WHY THIS REPORTS BYTES AND NOT A RECORD COUNT. It used to divide by a hardcoded
# 593583, which was one geometry's record size and is now wrong twice over: records shrank
# to 72 delta entries at L=8/E=16, and the size is not even constant within a chain --
# a proof whose leaf sits under a promoted odd-level node is 64 bytes shorter (973 vs 1037),
# which happened to 19.7% of leaves at the 10.7B corpus. Counting records properly means
# walking the chain with btmmoe_record_size_h, i.e. knowing the geometry, and the geometry
# lives in the two .c files rather than a shared header. Rather than keep a third copy of
# it here to drift out of step, this reports the exact byte size and counts only what THIS
# run adds. Absolute depth is whatever was already in the file plus `added`.
if [ -f "$CHAIN" ]; then
    echo "chain: $CHAIN, $(wc -c < "$CHAIN") bytes (pre-existing records not counted; see above)"
else
    echo "chain: $CHAIN absent -- starting from the genesis record"
fi

# Ctrl-C must not leave the backup lying around pretending to be state.
trap 'rm -f "$BAK"; echo; echo "interrupted"; exit 130' INT TERM

echo "growing to $TARGET record(s) at $STEPS steps, 1 attempt each, max $MAX_ATTEMPTS attempts"
echo

attempt=0
added=0
while [ "$added" -lt "$TARGET" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        echo "stopping: $MAX_ATTEMPTS attempts after adding $added of $TARGET record(s)."
        echo "A pass rate this low means the gate is above what the work can reach --"
        echo "see the calibration block in llmc/btmmoe.h before raising MAX_ATTEMPTS."
        break
    fi

    [ -f "$CHAIN" ] && cp "$CHAIN" "$BAK"

    out=$(BTM_TRAIN_STEPS="$STEPS" "$MINER" "" 1 2>&1)
    line=$(printf '%s\n' "$out" | grep -o 'improvement [+-][0-9.]*.*gate needs [+-][0-9.]*: [A-Z]*')

    # Ratio of achieved to required, computed here because the miner prints both but not
    # their quotient, and the quotient is the only number that says whether the exponent
    # is right.
    ratio=$(printf '%s\n' "$line" | awk '{
        imp=""; need="";
        for (i=1;i<=NF;i++) {
            if ($i=="improvement") imp=$(i+1);
            if ($i=="needs")       need=$(i+1);
        }
        gsub(/[:+]/,"",need); gsub(/[+]/,"",imp);
        if (imp!="" && need+0 != 0) printf "%.2f", imp/need; else printf "?";
    }')

    if printf '%s\n' "$out" | grep -q ': PASS'; then
        added=$((added + 1))
        printf 'added %-2d  ratio %-5s  %s  KEPT (chain now %s bytes)\n' \
               "$added" "$ratio" "$line" "$(wc -c < "$CHAIN")"
        rm -f "$BAK"
    else
        # Covers both a scored record that missed the gate and a structural rejection
        # ("could not be scored"). The restore is a no-op in the second case, because the
        # miner appends nothing when it cannot score -- but doing it unconditionally means
        # there is no path where a bad record survives.
        [ -f "$BAK" ] && mv "$BAK" "$CHAIN"
        if [ -n "$line" ]; then
            printf 'attempt %-2d ratio %-5s  %s  reverted\n' "$attempt" "$ratio" "$line"
        else
            # Dump the TAIL rather than grep for known patterns. The first version
            # grepped for 'score:|error|no attempt' and printed nothing at all for 40
            # consecutive failures, because the actual message ("asked for depth 12 but
            # the trunk is depth 8") matched none of them. A diagnostic that only reports
            # failures it already anticipated is worse than none.
            printf 'attempt %-2d NOT SCORED (structural, not a gate miss):\n' "$attempt"
            printf '%s\n' "$out" | tail -6 | sed 's/^/           /'
        fi
    fi
done

rm -f "$BAK"
echo
if [ -f "$CHAIN" ]; then
    echo "done: added $added record(s) in $attempt attempt(s), chain $(wc -c < "$CHAIN") bytes"
else
    echo "done: no chain file produced"
fi
