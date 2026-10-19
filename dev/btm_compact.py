#!/usr/bin/env python3
"""Build a rank-r COMPACTION of a Bitmark LoRA chain, to test whether it keeps the work.

dev/btm_spectrum.py measures how much of the summed delta's squared Frobenius norm a
rank-r approximation keeps -- 0.70..0.88 at K=8 for the dense tensors, against 0.125 for
independent deltas. That rules out the hopeless case, but it does NOT answer the question
that matters: Frobenius norm is not loss. 0.88 of the energy is 0.35 of the norm
discarded, and whether those discarded directions mattered is an empirical question about
the model, not about the matrix.

This writes the files that answer it, to be scored by the real evaluator:

  compacted.bin  ONE record approximating the sum of all K records, at rank r
  zero.bin       ONE record whose every entry has a zero scale, so it applies NOTHING
                 and scores the bare fixed-seed base model

Then, with the val batch pinned so all three are scored on the same 64 tokens:

    make eval_gpt2_btm_trunk_f32
    for f in btm-cp-trunk.bin compacted.bin zero.bin; do
        BTM_VAL_SEED=12345 ./eval_gpt2_btm_trunk_f32 ... # reads ./btm-cp-trunk.bin
    done

(the evaluator hardcodes its input path, so swap the file in or pass --out to write
compacted.bin over a copy -- see the runner printed at the end).

THE HEADLINE NUMBER is not the loss gap but the fraction of the records' useful work that
survives compaction:

    retained = (loss_base - loss_compacted) / (loss_base - loss_full)

~1.0 means compaction is nearly free in the only sense that matters and unbounded chain
history is possible. ~0 means the steep spectrum was a red herring and the design needs
a sliding window instead.

The compaction is QUANTIZED to int8 factors + fp16 scales exactly as a real record is, so
what is measured is an achievable compaction, not an idealised one -- the rank truncation
and the quantization error both land in the result, which is the honest comparison.
"""
import argparse
import hashlib
import struct
import sys
from collections import defaultdict

import numpy as np

BTM2_HEADER_BYTES = 34
DELTA_LORA_INT8 = 1
DELTA_LORA_EXPERT = 2
MAX_PLAUSIBLE_ENTRIES = 8192
E_CANDIDATES = (8, 16, 32, 64, 100, 128, 256)


def half_bits(x):
    """float -> fp16 bits, as btm2_float_to_half produces."""
    return int(np.float16(x).view(np.uint16))


def half_to_float(bits):
    return float(np.frombuffer(bits.to_bytes(2, "little"), dtype="<f2")[0])


def lora_dims(t, C, E):
    return {4: (3 * C, C), 6: (C, C), 16: (E, C)}.get(t)


def expert_dims(t, C, H):
    return (H, C) if t == 10 else (C, H)


def parse_entry(buf, off, C, H, E):
    """-> (size, kind, key, dims, A, B, scale) or None. `kind` is the delta_type."""
    if off + 4 > len(buf):
        return None
    t, layer, dtype, r = buf[off], buf[off + 1], buf[off + 2], buf[off + 3]
    if r <= 0:
        return None
    if dtype == DELTA_LORA_INT8:
        dims = lora_dims(t, C, E)
        if dims is None:
            return None
        out, in_ = dims
        size = 4 + 4 + out * r + r * in_
        if off + size > len(buf):
            return None
        sA = int.from_bytes(buf[off + 4:off + 6], "little")
        sB = int.from_bytes(buf[off + 6:off + 8], "little")
        a_at, key = off + 8, (t, layer, -1)
    elif dtype == DELTA_LORA_EXPERT:
        if t not in (10, 12):
            return None
        out, in_ = expert_dims(t, C, H)
        size = 4 + 8 + out * r + r * in_
        if off + size > len(buf):
            return None
        eid = int.from_bytes(buf[off + 4:off + 6], "little")
        if eid >= E:
            return None
        sA = int.from_bytes(buf[off + 8:off + 10], "little")
        sB = int.from_bytes(buf[off + 10:off + 12], "little")
        a_at, key = off + 12, (t, layer, eid)
    else:
        return None
    if sA == 0 or sB == 0:
        return size, dtype, key, (out, in_), None, None, 0.0
    A = np.frombuffer(buf, np.int8, out * r, a_at).reshape(out, r)
    B = np.frombuffer(buf, np.int8, r * in_, a_at + out * r).reshape(r, in_)
    return size, dtype, key, (out, in_), A, B, half_to_float(sA) * half_to_float(sB)


def walk(buf, C, H, E, first_only=False):
    recs, off = [], 0
    while off < len(buf):
        if off + BTM2_HEADER_BYTES > len(buf):
            return None
        n = int.from_bytes(buf[off + 32:off + 34], "little")
        if n == 0 or n > MAX_PLAUSIBLE_ENTRIES:
            return None
        o, entries = off + BTM2_HEADER_BYTES, []
        for _ in range(n):
            got = parse_entry(buf, o, C, H, E)
            if got is None:
                return None
            entries.append(got)
            o += got[0]
        recs.append(entries)
        off = o
        if first_only:
            return recs
    return recs or None


def detect_E(buf, C, H):
    for E in E_CANDIDATES:
        if walk(buf, C, H, E, first_only=True) is None:
            continue
        recs = walk(buf, C, H, E)
        if recs is not None:
            return E, recs
    return None, None


def quantize_factors(M, r):
    """Best rank-r approximation of M, as int8 factors + fp16 scales.

    The singular values are split evenly between the two factors (sqrt on each side) so
    neither carries the whole dynamic range into an int8 -- splitting unevenly wastes
    precision on whichever side ends up small. Returns (A_int8, B_int8, sA_bits, sB_bits)
    reproducing (A@B) * half(sA) * half(sB) ~= M_r, which is exactly what
    btm2_apply_lora_int8 computes.
    """
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, len(sv))
    root = np.sqrt(sv[:r])
    A_real = U[:, :r] * root            # [out, r]
    B_real = root[:, None] * Vt[:r]     # [r, in]

    def q(X):
        peak = float(np.abs(X).max())
        if peak == 0.0:
            return np.zeros(X.shape, np.int8), 0
        scale = peak / 127.0
        bits = half_bits(scale)
        if bits == 0:               # underflowed fp16: nothing representable
            return np.zeros(X.shape, np.int8), 0
        # Quantize against the ROUNDED scale, not the exact one: that is the value the
        # verifier will multiply by, so rounding to it here is what keeps the miner's
        # reconstruction identical to the verifier's.
        eff = half_to_float(bits)
        return np.clip(np.rint(X / eff), -127, 127).astype(np.int8), bits

    A, sA = q(A_real)
    B, sB = q(B_real)
    return A, B, sA, sB


def make_record(entries):
    """A record whose head is the double-SHA256 of its own body.

    Any distinct value would do -- these files are synthetic and the evaluator only warns
    that the head does not satisfy the real chaining rule. What matters is that the two
    output files get DIFFERENT heads, because the node's state cache
    (btm-trunk-state.bin) is keyed on the head of the last record folded:

        if (memcmp(cp + off, cache_tip, 32) == 0) { need_init = 0; ... }

    Writing both files with the same head (32 zero bytes, as the first version of this
    script did) makes the second run load the FIRST run's cached weights and report its
    loss -- which silently invalidates the whole comparison. Deriving the head from the
    body makes that impossible rather than merely unlikely.
    """
    body = struct.pack("<H", len(entries)) + b"".join(entries)
    head = hashlib.sha256(hashlib.sha256(body).digest()).digest()
    return head + body


def entry_bytes(dtype, key, A, B, sA, sB):
    t, layer, eid = key
    r = A.shape[1]
    head = bytes([t, layer, dtype, r])
    if dtype == DELTA_LORA_INT8:
        head += struct.pack("<HH", sA, sB)
    else:
        head += struct.pack("<HHHH", eid, 0, sA, sB)
    return head + A.tobytes() + B.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", nargs="?", default="btm-cp-trunk.bin")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--channels", type=int, default=768)
    ap.add_argument("--expert-hidden", type=int, default=384)
    ap.add_argument("--max-k", type=int, default=0, help="compact only the first K records")
    ap.add_argument("--types", default="",
                    help="comma-separated tensor types to KEEP (4 qkvw, 6 attprojw, "
                         "10 expert fcw, 12 expert fcprojw, 16 routerw). Everything else "
                         "is DROPPED -- its accumulated delta is simply discarded, which "
                         "is the point: a compaction needs an entry per matrix any record "
                         "in the span touched, so the long tail of once-touched experts "
                         "is what makes it too big for a block. Default: keep all")
    ap.add_argument("--top-experts", type=int, default=-1, metavar="N",
                    help="carry entries for only the N expert matrices with the largest "
                         "accumulated delta energy, dropping the rest. This is the ENTRY "
                         "BUDGET: a compaction's size is set by how many matrices it "
                         "covers, not by how many records it summarises, so fixing N "
                         "fixes the record size for ANY span -- which is what lets a "
                         "chain accumulate unbounded history. At the ~975 kB solution-tx "
                         "cap the budget is 36 dense + ~154 expert entries. Negative = "
                         "no cap")
    ap.add_argument("--out", default="compacted.bin")
    ap.add_argument("--zero-out", default="zero.bin")
    args = ap.parse_args()

    buf = open(args.chain, "rb").read()
    C, H = args.channels, args.expert_hidden
    E, recs = detect_E(buf, C, H)
    if E is None:
        sys.exit(f"{args.chain} does not parse with C={C} H={H}")
    if args.max_k:
        recs = recs[:args.max_k]
    print(f"{args.chain}: {len(recs)} record(s), E={E}, compacting at rank {args.rank}")

    # Sum each matrix's deltas across the records, and remember the shape of the entry
    # that produced it so the compaction can be written back in the same form.
    summed, meta = {}, {}
    for entries in recs:
        for _size, dtype, key, dims, A, B, scale in entries:
            if A is None:
                continue
            d = (A.astype(np.float64) @ B.astype(np.float64)) * scale
            summed[key] = d if key not in summed else summed[key] + d
            meta[key] = (dtype, dims)

    print(f"  {len(summed)} matrices carry a non-zero sum")

    # --- selection: the entry budget -------------------------------------------------
    # Purely miner-side. Nothing in consensus specifies which matrices a compaction
    # covers, because a verifier judges its OUTCOME -- whether it regresses the loss --
    # exactly as it judges any other record. So this is an optimisation, and a better
    # heuristic can replace it later without a fork.
    energy = {k: float((v ** 2).sum()) for k, v in summed.items()}
    all_sq = sum(energy.values())
    selected = set(summed)
    if args.types:
        keep_types = {int(x) for x in args.types.split(",") if x.strip()}
        selected = {k for k in selected if k[0] in keep_types}
        print(f"  --types {sorted(keep_types)}: {len(selected)} matrices survive")
    if args.top_experts >= 0:
        experts = sorted((k for k in selected if k[0] in (10, 12)),
                         key=lambda k: -energy[k])
        cut = set(experts[args.top_experts:])
        selected -= cut
        print(f"  --top-experts {args.top_experts}: kept {len(experts) - len(cut)} of "
              f"{len(experts)} expert matrices by accumulated energy, dropped {len(cut)}")

    dropped = [k for k in summed if k not in selected]
    if dropped:
        drop_sq = sum(energy[k] for k in dropped)
        for k in dropped:
            del summed[k]
            del meta[k]
        print(f"  carrying {len(summed)} entries, dropped {len(dropped)}")
        print(f"  dropped matrices hold {100 * drop_sq / all_sq:.1f}% of the total "
              f"summed-delta energy")

    out_entries, stats = [], []
    for key in sorted(summed):
        dtype, _dims = meta[key]
        M = summed[key]
        A, B, sA, sB = quantize_factors(M, args.rank)
        out_entries.append(entry_bytes(dtype, key, A, B, sA, sB))
        if sA and sB:
            M4 = (A.astype(np.float64) @ B.astype(np.float64)) * half_to_float(sA) * half_to_float(sB)
            denom = np.linalg.norm(M)
            stats.append(np.linalg.norm(M - M4) / denom if denom > 0 else 0.0)

    rec = make_record(out_entries)
    open(args.out, "wb").write(rec)
    print(f"  wrote {args.out}: {len(rec):,} bytes, {len(out_entries)} entries")
    if stats:
        print(f"  realised relative error ||M-M4||/||M||: "
              f"mean {np.mean(stats):.3f}  min {min(stats):.3f}  max {max(stats):.3f}")
        print(f"  (rank truncation AND int8/fp16 quantization, as a real record would be)")

    # A record that applies nothing, to score the bare base model. Every entry carries a
    # zero scale, which btm2_apply_lora_int8 treats as a zero delta -- so this is a
    # well-formed record the evaluator walks happily and that changes no weight.
    zero_entries = []
    for key in sorted(summed):
        dtype, (out, in_) = meta[key]
        A = np.zeros((out, args.rank), np.int8)
        B = np.zeros((args.rank, in_), np.int8)
        zero_entries.append(entry_bytes(dtype, key, A, B, 0, 0))
    zrec = make_record(zero_entries)
    open(args.zero_out, "wb").write(zrec)
    print(f"  wrote {args.zero_out}: {len(zrec):,} bytes (applies nothing; scores the base)")

    print(f"""
Now score all three on the SAME val batch, with BTM_CHAIN so nothing is moved:

  for f in {args.chain} {args.out} {args.zero_out}; do
      rm -f btm-trunk-state.bin
      echo "== $f"
      BTM_CHAIN=$f BTM_VAL_SEED=12345 ./eval_gpt2_btm_trunk_f32 2>&1 \\
          | grep -E "val loss|cache|DIAGNOSTIC"
  done

Then:  retained = (loss_base - loss_compacted) / (loss_base - loss_full)

Two things to verify in the output, or the comparison is void:
  * every run says "state cache miss: full rebuild" -- btm-trunk-state.bin is keyed on
    the last record's head, so a stale one makes a run report another run's loss;
  * every run says "val batch pinned to seed 12345" -- otherwise each file's last
    record picks its own 64 tokens and the losses are not comparable.

A "chained hash does NOT match" warning on the synthetic files is expected and cosmetic:
nothing mined them, so their heads do not satisfy the real chaining rule.""")


if __name__ == "__main__":
    main()
