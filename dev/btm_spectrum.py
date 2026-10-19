#!/usr/bin/env python3
"""Singular-value spectrum of SUMMED Bitmark LoRA chain records.

THE QUESTION THIS ANSWERS. A stateful dynamic algo rebuilds its model by folding every
record its slot's blocks carry, and a Bitmark verifier can only afford to fold a bounded
number of them (4096 chain reads, and in practice a far tighter compute bound). Beyond
that the chain needs either a sliding window, which forgets, or HIERARCHICAL COMPACTION,
which replaces K records with one record approximating their sum.

Compaction is only possible if the sum of K records is still close to rank r. Each record
contributes a rank-r update per matrix, so a sum of K has rank up to K*r: a rank-4
compaction of 64 records keeps 4 of 256 directions, which is hopeless IF the deltas are
mutually independent. But every record is trained on the same task against a nearly
identical base, so they may be strongly aligned -- in which case the sum is nearly rank-4
already and compaction is nearly lossless.

That is an empirical fact about your records, not something derivable. This measures it.

WHAT IS REPORTED. For each weight matrix, the deltas of the first K records touching it
are summed and the sum is decomposed. The headline number is

    retention(K) = sum of the top-r squared singular values / sum of all of them

which by Eckart-Young is exactly the fraction of squared Frobenius norm that the BEST
rank-r approximation keeps -- i.e. 1 - (relative error of the best rank-r compaction)^2.

Compare it against the independent baseline 1/K, which is what mutually orthogonal
equal-magnitude deltas would give. retention(K) near 1/K means the deltas are
independent and compaction is a dead end; retention(K) near 1 means they are aligned and
compaction is close to free.

Note K=1 is degenerate: one record IS rank r, so retention is 1.0 by construction. It is
printed as a self-check on the parser, not as evidence.

USAGE
    python3 dev/btm_spectrum.py btm-cp-trunk.bin [more-records.bin ...]
    python3 dev/btm_spectrum.py --rank 4 --max-k 64 btm-cp-trunk.bin

Records are read in order across all files given, as the verifier folds them.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

BTM2_HEADER_BYTES = 34  # 32B head + u16 n_entries
DELTA_LORA_INT8 = 1
DELTA_LORA_EXPERT = 2

# Candidate expert counts, tried until the record walk consumes the input exactly. E is
# needed to SIZE a router entry (out = E), so it cannot be read out of the file -- and
# getting it wrong silently desyncs the walk. The trunk Makefile defaults to 100 while
# the committed btm-cp-trunk.bin was built with 64, which is exactly the mismatch this
# avoids guessing at.
E_CANDIDATES = (8, 16, 32, 64, 100, 128, 256)


def half_to_float(bits):
    return float(np.frombuffer(bits.to_bytes(2, "little"), dtype="<f2")[0])


def lora_dims(tensor_type, C, E):
    """Dense-LoRA dims, mirroring btmmoe_lora_dims_r."""
    if tensor_type == 4:
        return 3 * C, C
    if tensor_type == 6:
        return C, C
    if tensor_type == 16:
        return E, C
    return None


def expert_dims(tensor_type, C, H):
    """Expert-LoRA dims, mirroring btmmoe_expert_dims_h."""
    return (H, C) if tensor_type == 10 else (C, H)


def parse_entry(buf, off, C, H, E):
    """One entry -> (size, key, A, B, scale) or None if it does not parse.

    `key` identifies the weight matrix the delta lands on, so deltas from different
    records can be summed per matrix.
    """
    if off + 4 > len(buf):
        return None
    tensor_type, layer, dtype, r = buf[off], buf[off + 1], buf[off + 2], buf[off + 3]
    if r <= 0:
        return None

    if dtype == DELTA_LORA_INT8:
        dims = lora_dims(tensor_type, C, E)
        if dims is None:
            return None
        out, in_ = dims
        size = 4 + 4 + out * r + r * in_
        if off + size > len(buf):
            return None
        sA = int.from_bytes(buf[off + 4:off + 6], "little")
        sB = int.from_bytes(buf[off + 6:off + 8], "little")
        a_at = off + 8
        key = (tensor_type, layer, -1)
    elif dtype == DELTA_LORA_EXPERT:
        if tensor_type not in (10, 12):
            return None
        out, in_ = expert_dims(tensor_type, C, H)
        size = 4 + 8 + out * r + r * in_
        if off + size > len(buf):
            return None
        expert_id = int.from_bytes(buf[off + 4:off + 6], "little")
        if expert_id >= E:
            return None
        sA = int.from_bytes(buf[off + 8:off + 10], "little")
        sB = int.from_bytes(buf[off + 10:off + 12], "little")
        a_at = off + 12
        key = (tensor_type, layer, expert_id)
    else:
        return None  # sparse/dense deltas carry no factors to decompose

    # A zero scale is a zero delta (btm2_apply_lora_int8 returns early).
    if sA == 0 or sB == 0:
        return size, key, None, None, 0.0

    A = np.frombuffer(buf, dtype=np.int8, count=out * r, offset=a_at).reshape(out, r)
    B = np.frombuffer(buf, dtype=np.int8, count=r * in_, offset=a_at + out * r).reshape(r, in_)
    return size, key, A, B, half_to_float(sA) * half_to_float(sB)


def squared_singular_values(M):
    """The squared singular values of M, descending.

    Via the Gram matrix rather than an SVD: sigma^2 are the eigenvalues of M^T M (or
    M M^T, whichever is smaller), which for a 2304x768 delta is a 768x768 symmetric
    eigenproblem instead of a full 2304x768 decomposition -- several times faster, and
    this tool runs it once per matrix per K. Squaring doubles the condition number, but
    the metric is an ENERGY FRACTION over sigma^2, so the small eigenvalues it degrades
    are exactly the ones that contribute least to it. Rounding can push them slightly
    negative; clamped, since a squared singular value cannot be.
    """
    G = M.T @ M if M.shape[0] >= M.shape[1] else M @ M.T
    ev = np.linalg.eigvalsh(G)          # ascending, symmetric
    return np.clip(ev[::-1], 0.0, None)  # descending, non-negative


#: A wrong E desyncs the walk onto arbitrary bytes, where the u16 at offset 32 reads as
#: some huge count and the entry loop runs that many times before failing. Capping it
#: makes a bad candidate fail in microseconds instead of seconds -- which matters because
#: detection tries several. The real records carry 108.
MAX_PLAUSIBLE_ENTRIES = 8192


def walk_records(buf, C, H, E, first_only=False):
    """All records in `buf` -> list of [(key, A, B, scale), ...]. None if it desyncs."""
    records = []
    off = 0
    while off < len(buf):
        if off + BTM2_HEADER_BYTES > len(buf):
            return None
        n_entries = int.from_bytes(buf[off + 32:off + 34], "little")
        if n_entries == 0 or n_entries > MAX_PLAUSIBLE_ENTRIES:
            return None
        o = off + BTM2_HEADER_BYTES
        entries = []
        for _ in range(n_entries):
            got = parse_entry(buf, o, C, H, E)
            if got is None:
                return None
            size, key, A, B, scale = got
            entries.append((key, A, B, scale))
            o += size
        records.append(entries)
        off = o
        if first_only:
            return records
    return records if records else None


def detect_E(buf, C, H):
    """The only parameter the framing needs and the file cannot state.

    Candidates are screened against the FIRST record only -- a wrong E almost always
    fails inside it, and screening the whole file per candidate is what made this the
    slowest part of the tool. The winner is then validated over everything.
    """
    for E in E_CANDIDATES:
        if walk_records(buf, C, H, E, first_only=True) is None:
            continue
        recs = walk_records(buf, C, H, E)
        if recs is not None:
            return E, recs
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="chain record file(s), folded in order")
    ap.add_argument("--channels", type=int, default=768, help="C")
    ap.add_argument("--expert-hidden", type=int, default=384, help="H (MOE_H)")
    ap.add_argument("--experts", type=int, default=0, help="E; 0 = auto-detect")
    ap.add_argument("--rank", type=int, default=4, help="r a compaction would re-project to")
    ap.add_argument("--max-k", type=int, default=0, help="largest K to report; 0 = all")
    ap.add_argument("--sample", type=int, default=0,
                    help="analyse at most this many matrices per tensor type (0 = all). "
                         "The answer is a property of the spectrum, not of which matrix, "
                         "so a handful per type gives the same conclusion far faster")
    args = ap.parse_args()

    buf = b"".join(open(f, "rb").read() for f in args.files)
    C, H = args.channels, args.expert_hidden

    if args.experts:
        E = args.experts
        records = walk_records(buf, C, H, E)
        if records is None:
            sys.exit(f"does not parse with C={C} H={H} E={E}; try --experts 0 to auto-detect")
    else:
        E, records = detect_E(buf, C, H)
        if E is None:
            sys.exit(f"does not parse with C={C} H={H} and any E in {E_CANDIDATES}")

    print(f"{len(buf):,} bytes, {len(records)} record(s)   C={C} H={H} E={E} "
          f"(E {'given' if args.experts else 'auto-detected'})")
    print(f"entries per record: {sorted({len(r) for r in records})}")
    if len(records) < 2:
        print("\nOnly one record: a single delta IS rank r, so there is no sum to "
              "decompose.\nMine more records locally and re-run -- K>=8 is where the "
              "answer starts to matter.")

    # Each matrix's delta PER RECORD, reconstructed once. Keyed by record index, not
    # appended in order: the question is what a span of K records accumulates, and a
    # given expert matrix is touched by only a few records in that span. Collapsing that
    # to "the first K contributions" would silently stretch the span for rarely-touched
    # matrices -- 3 contributions to one expert can span 60 records -- and that is the
    # favourable case worth seeing, not hiding.
    #
    # The FACTORS are kept, not the reconstructed deltas: a 2304x768 delta is 14 MB in
    # float64, so retaining one per matrix per record would be tens of GB at the K values
    # that matter. The factors are ~12 KB each, and expanding them inside the loop below
    # is a cheap int32 matmul next to the eigendecomposition it feeds.
    per_matrix = defaultdict(lambda: defaultdict(list))
    zero_deltas = defaultdict(int)
    for r_idx, entries in enumerate(records):
        for key, A, B, scale in entries:
            if A is None:  # zero scale == zero delta (btm2_apply_lora_int8 returns early)
                zero_deltas[key[0]] += 1
                continue
            per_matrix[key][r_idx].append((A, B, scale))

    if zero_deltas:
        total_zero = sum(zero_deltas.values())
        by = ", ".join(f"type {t}: {n}" for t, n in sorted(zero_deltas.items()))
        print(f"\nzero-scale entries (carried but contributing nothing): "
              f"{total_zero} ({by})")

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= len(records)]
    if args.max_k:
        ks = [k for k in ks if k <= args.max_k]
    # The full span is always worth a row, unless --max-k deliberately excludes it.
    if len(records) not in ks and (not args.max_k or len(records) <= args.max_k):
        ks.append(len(records))

    type_names = {4: "qkvw", 6: "attprojw", 10: "expert fcw", 12: "expert fcprojw",
                  16: "routerw"}
    print(f"\nretention(K) = top-{args.rank} squared singular values / all of them")
    print("             = the fraction of squared Frobenius norm the BEST "
          f"rank-{args.rank} compaction keeps")
    print(f"baseline 1/K = what mutually independent deltas would give\n")
    print(f"{'K':>5}  {'baseline':>9}  {'tensor':<16}{'matrices':>9}"
          f"{'touches':>9}{'mean':>9}{'min':>9}")
    print("-" * 71)

    # Sampling picks the SAME matrices at every K, so a row-to-row comparison down the
    # table is of one population rather than of different ones.
    keys_by_type = defaultdict(list)
    for key in sorted(per_matrix):
        keys_by_type[key[0]].append(key)
    chosen = []
    for tensor_type, keys in keys_by_type.items():
        chosen += keys[:args.sample] if args.sample else keys

    for k in ks:
        by_type = defaultdict(list)
        touches = defaultdict(list)
        for key in chosen:
            (tensor_type, _layer, _expert) = key
            per_record = per_matrix[key]
            in_span = [(r_idx, terms) for r_idx, terms in per_record.items() if r_idx < k]
            if not in_span:
                continue  # untouched in this span of records
            # Accumulated in float64. The int32 A@B product is exact, but the sum ACROSS
            # records is float and order-dependent -- the same reason chain order is
            # consensus-critical when the node folds these for real.
            acc = None
            for _r_idx, terms in in_span:
                for A, B, scale in terms:
                    # float64 rather than int32 for the product: NumPy's integer matmul
                    # is not BLAS-backed and dominates the runtime. It stays EXACT --
                    # |A.B| <= r * 127^2 = 64516 for r=4, far inside float64's exact
                    # integer range -- so this is the same number by a faster route.
                    d = (A.astype(np.float64) @ B.astype(np.float64)) * scale
                    acc = d if acc is None else acc + d
            sq = squared_singular_values(acc)
            energy = float(sq.sum())
            if energy <= 0:
                continue
            by_type[tensor_type].append(float(sq[:args.rank].sum()) / energy)
            touches[tensor_type].append(len(in_span))

        if not by_type:
            continue
        for i, tensor_type in enumerate(sorted(by_type)):
            vals = by_type[tensor_type]
            k_col = f"{k}" if i == 0 else ""
            base_col = f"{1.0 / k:.3f}" if i == 0 else ""
            name = type_names.get(tensor_type, str(tensor_type))
            print(f"{k_col:>5}  {base_col:>9}  {name:<16}"
                  f"{len(vals):>9}{np.mean(touches[tensor_type]):>9.1f}"
                  f"{np.mean(vals):>9.3f}{min(vals):>9.3f}")
        print()

    if len(records) >= 8:
        print("Reading the result: retention tracking 1/K means the deltas are "
              "independent and\ncompaction cannot work at this rank -- use a sliding "
              "window. Retention staying\nhigh (say >0.8) as K grows means they are "
              "aligned and a rank-%d compaction is\nnearly lossless, which is what "
              "makes unbounded history possible." % args.rank)


if __name__ == "__main__":
    main()
