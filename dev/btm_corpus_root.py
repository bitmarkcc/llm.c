#!/usr/bin/env python3
"""Build the Merkle commitment over a tokenised Bitmark trunk corpus.

The verifier scores a record on a 64-token window of the training corpus and has no
filesystem, so a 10B-token corpus (20+ GB) cannot live in the module. Instead the module
holds a 32-byte root, and each record carries its scored window plus a proof against that
root. See llmc/btmcorpus.h for the layout and the reasoning -- this file MUST agree with
it bit-for-bit, and dev/btm_corpus_check.c exists to prove that it does.

    python3 dev/btm_corpus_root.py 'dev/data/edu_fineweb10B_16k/edu_fineweb16k_*.bin' \
        --emit-header llmc/btmcorpus_params.h --save-leaves corpus_leaves.bin

SHARD ORDER IS PART OF THE COMMITMENT. The corpus is the concatenation of the shards in
sorted filename order with each 1024-byte header skipped. Globs are sorted here so the
order does not depend on the shell.

SHARDS MUST BE LEAF-ALIGNED: every shard but the last holds a token count divisible by 64.
Otherwise a leaf straddles two files and no miner can read it with one seek, so this is
asserted rather than assumed. dev/data/edu_fineweb10B_16k.py guarantees it.

SCALE. At 10B tokens this is ~187M leaves, so nothing here may hold the corpus or a
per-leaf Python object list in memory: 187M `bytes` objects cost tens of GB in object
overhead alone. Leaves are streamed from disk, level 0 goes into one flat bytearray
(32 B/leaf, ~6 GB), and every level above it is folded IN PLACE into the front of that
same buffer. Peak memory is therefore one level-0 array and nothing else.
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys

LEAF_TOKENS = 64
LEAF_BYTES = LEAF_TOKENS * 2
HEADER_BYTES = 1024
MAGIC = 20240520


def h_leaf(b):
    return hashlib.sha256(b"\x00" + b).digest()


def h_node(l, r):
    return hashlib.sha256(b"\x01" + l + r).digest()


# ----------------------------------------------------------------------------
# reference implementation, small trees only -- dev/btm_corpus_check.py drives this
# against the C folder over many awkward shapes. Kept separate from the streaming path
# below because being obviously correct matters more here than being fast.

def build(tokens_bytes):
    """-> (root, n_leaves, level0_hashes). Odd levels PROMOTE their last node."""
    n_leaves = len(tokens_bytes) // LEAF_BYTES
    if n_leaves == 0:
        sys.exit("corpus is shorter than one leaf")
    level = [h_leaf(tokens_bytes[i * LEAF_BYTES:(i + 1) * LEAF_BYTES])
             for i in range(n_leaves)]
    leaves = list(level)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(h_node(level[i], level[i + 1]))
        if len(level) & 1:
            nxt.append(level[-1])   # promote, never duplicate (CVE-2012-2459)
        level = nxt
    return level[0], n_leaves, leaves


def proof(leaves, index):
    """The sibling list for `index`, in bottom-up order, omitting promoted levels."""
    sibs = []
    level = list(leaves)
    pos = index
    while len(level) > 1:
        if pos + 1 == len(level) and (len(level) & 1):
            pos //= 2
            level = [h_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)] + [level[-1]]
            continue
        sibs.append(level[pos ^ 1])
        nxt = [h_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) & 1:
            nxt.append(level[-1])
        level = nxt
        pos //= 2
    return sibs


# ----------------------------------------------------------------------------
# streaming path, for the real corpus

def shard_info(paths):
    """-> (per_shard_tokens, n_tokens, n_leaves), validating headers and alignment."""
    counts = []
    for p in paths:
        with open(p, "rb") as f:
            hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            sys.exit(f"{p}: shorter than a header")
        magic, ver, ntok = struct.unpack("<iii", hdr[:12])
        if magic != MAGIC:
            sys.exit(f"{p}: bad magic {magic}")
        if ver != 1:
            sys.exit(f"{p}: bad version {ver}")
        want = HEADER_BYTES + ntok * 2
        have = os.path.getsize(p)
        if have != want:
            sys.exit(f"{p}: header says {ntok:,} tokens ({want:,} bytes) but file is {have:,}")
        counts.append(ntok)

    for i, n in enumerate(counts[:-1]):
        if n % LEAF_TOKENS:
            sys.exit(f"{paths[i]}: {n:,} tokens is not a multiple of {LEAF_TOKENS}; "
                     f"only the LAST shard may be, or leaves straddle shard boundaries")

    n_tokens = sum(counts)
    n_leaves = sum(n // LEAF_TOKENS for n in counts)
    if n_leaves != n_tokens // LEAF_TOKENS:
        sys.exit("internal: leaf count disagrees with the token total")
    return counts, n_tokens, n_leaves


def iter_leaves(paths, counts, chunk_leaves=8192):
    """Yield each whole leaf of the concatenated corpus, in order, streaming from disk.

    Safe to treat shard by shard only because of the alignment invariant checked above:
    a leaf never spans two files, so the last shard's trailing partial leaf is simply
    dropped and nothing carries across.
    """
    for p, ntok in zip(paths, counts):
        whole = ntok // LEAF_TOKENS
        with open(p, "rb") as f:
            f.seek(HEADER_BYTES)
            done = 0
            while done < whole:
                take = min(chunk_leaves, whole - done)
                blob = f.read(take * LEAF_BYTES)
                if len(blob) != take * LEAF_BYTES:
                    sys.exit(f"{p}: short read at leaf {done}")
                for k in range(take):
                    yield blob[k * LEAF_BYTES:(k + 1) * LEAF_BYTES]
                done += take


def level_widths(n_leaves):
    """Node count at each level, bottom-up, ending at the 1-node root."""
    w = [n_leaves]
    while w[-1] > 1:
        w.append((w[-1] + 1) // 2)
    return w


def build_streaming(paths, counts, n_leaves, save_leaves="", save_tree=""):
    """-> root. Streams level 0 into one bytearray, then folds in place above it.

    With --save-tree, every level is appended to one file as it is computed. That costs
    ~2x level 0 on disk (sum of widths ~ 2n) and turns a miner's proof from an O(n) fold
    over a 6 GB array -- 240 s at 177M leaves, measured -- into one 32-byte read per
    level. The level offsets are a function of n_leaves alone (level_widths above), so
    there is no index to write and nothing that can fall out of sync with the data.
    """
    sha = hashlib.sha256
    buf = bytearray(n_leaves * 32)
    print(f"hashing {n_leaves:,} leaves into {n_leaves * 32 / 1e9:.2f} GB ...")
    i = 0
    for leaf in iter_leaves(paths, counts):
        buf[i * 32:(i + 1) * 32] = sha(b"\x00" + leaf).digest()
        i += 1
        if (i & 0xFFFFFF) == 0:
            print(f"  {i:,} / {n_leaves:,}")
    if i != n_leaves:
        sys.exit(f"internal: hashed {i} leaves, expected {n_leaves}")

    # Both written BEFORE folding: the fold overwrites level 0 in place.
    if save_leaves:
        print(f"writing {save_leaves} ({n_leaves * 32 / 1e9:.2f} GB) ...")
        with open(save_leaves, "wb") as f:
            f.write(buf)
    tf = None
    if save_tree:
        total = sum(level_widths(n_leaves)) * 32
        print(f"writing {save_tree} ({total / 1e9:.2f} GB, all levels) ...")
        tf = open(save_tree, "wb")
        tf.write(buf)

    width = n_leaves
    depth = 0
    while width > 1:
        nw = 0
        for j in range(0, width - 1, 2):
            # Both children are adjacent, so one slice is the whole 64-byte payload.
            # bytes + bytearray yields bytes, so no extra copy is needed to hash it.
            buf[nw * 32:(nw + 1) * 32] = sha(b"\x01" + buf[j * 32:(j + 2) * 32]).digest()
            nw += 1
        if width & 1:                 # promote the odd tail unchanged
            buf[nw * 32:(nw + 1) * 32] = buf[(width - 1) * 32:width * 32]
            nw += 1
        width = nw
        depth += 1
        if tf:
            # memoryview, not a slice: buf[:width*32] would copy, and at level 1 that is
            # a 2.8 GB transient on top of the 5.7 GB buffer for no reason.
            tf.write(memoryview(buf)[:width * 32])
        print(f"  level {depth}: {width:,}")
    if tf:
        tf.close()
    return bytes(buf[:32])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="tokenised .bin shards, or a glob")
    ap.add_argument("-o", "--out", default="corpus_root.json")
    ap.add_argument("--emit-header", default="", metavar="PATH",
                    help="write llmc/btmcorpus_params.h with these constants")
    ap.add_argument("--save-leaves", default="", metavar="PATH",
                    help="write the raw level-0 leaf hashes here (32 B each) so a miner "
                         "can build proofs without re-reading the corpus")
    ap.add_argument("--save-tree", default="", metavar="PATH",
                    help="write EVERY level (32 B each, bottom-up, levels concatenated) "
                         "so a miner reads one hash per level instead of folding the "
                         "whole array. ~2x the size of --save-leaves and supersedes it")
    args = ap.parse_args()

    paths = []
    for s in args.shards:
        paths += sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
    if not paths:
        sys.exit("no shards matched")
    print(f"reading {len(paths)} shard(s)")
    counts, n_tokens, n_leaves = shard_info(paths)
    for p, n in list(zip(paths, counts))[:3]:
        print(f"  {p}: {n:,} tokens")
    if len(paths) > 3:
        print(f"  ... and {len(paths) - 3} more")

    root = build_streaming(paths, counts, n_leaves, args.save_leaves, args.save_tree)
    dropped = n_tokens - n_leaves * LEAF_TOKENS
    # Every shard but the last is leaf-aligned, so this is constant across all of them.
    leaves_per_shard = counts[0] // LEAF_TOKENS if len(counts) > 1 else n_leaves
    print(f"\nn_tokens  {n_tokens:,}")
    print(f"n_leaves  {n_leaves:,}  ({LEAF_TOKENS} tokens each, {dropped} trailing dropped)")
    print(f"depth     {max(0, (n_leaves - 1).bit_length())}")
    print(f"root      {root.hex()}")

    meta = {"root": root.hex(), "n_leaves": n_leaves, "n_tokens": n_tokens,
            "leaf_tokens": LEAF_TOKENS, "leaves_per_shard": leaves_per_shard,
            "shard_tokens": counts, "shards": paths}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")

    if args.emit_header:
        with open(args.emit_header, "w") as f:
            f.write(f"""/*
Corpus commitment constants. CONSENSUS: changing the root or the leaf count changes what
the work IS, so it is a new algo branch and a new vote. Generated by
dev/btm_corpus_root.py -- do not hand-edit; re-run the tool over the corpus.

Corpus: {len(paths)} shard(s), {n_tokens:,} tokens
First:  {paths[0]}
Last:   {paths[-1]}
*/
#ifndef BTMCORPUS_PARAMS_H
#define BTMCORPUS_PARAMS_H

#define BTM_CORPUS_ROOT_HEX "{root.hex()}"
#define BTM_CORPUS_N_LEAVES {n_leaves}ull

//! The scored sequence length. A leaf is BTMC_LEAF_TOKENS (64) tokens, and a language
//! model consumes T+1 tokens to score T predictions (inputs[0..T-1] against
//! targets[1..T], as llmc/dataloader.h reads B*T+1), so one leaf scores T = 63.
//! Deliberately NOT 64-with-a-second-leaf: crossing a leaf boundary would need two
//! windows and two proofs for one score.
#define BTM_CORPUS_SCORE_T 63

//! MINER-SIDE ONLY -- the verifier never touches the corpus, so this is not consensus.
//! Every shard but the last holds exactly this many leaves, which is what lets a miner
//! find leaf i at shard i/{leaves_per_shard} offset (i%{leaves_per_shard})*{LEAF_BYTES}
//! with a single seek instead of a shard table.
#define BTM_CORPUS_LEAVES_PER_SHARD {leaves_per_shard}ull

#endif // BTMCORPUS_PARAMS_H
""")
        print(f"wrote {args.emit_header}")


if __name__ == "__main__":
    main()
