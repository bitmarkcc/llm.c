/*
Corpus commitment for the Bitmark trunk chain. CONSENSUS-CRITICAL: the miner and the
verifier must agree bit-for-bit on everything here.

WHY THIS EXISTS. The verifier scores a record by its loss on a 64-token window of the
training corpus, and a wasm verifier has no filesystem. A 10B-token corpus is 20 GB, so
it cannot live in the module. Instead the module commits to the corpus by a single
32-byte Merkle root, and each record CARRIES the window it was scored on plus a proof
that the window really came from the committed corpus. Verification then needs no data
availability at all -- everything it checks arrives in the record. Only MINING needs the
corpus, which is the right place for that burden.

LEAVES. The corpus is the uint16 little-endian token stream exactly as dev/data/*.py
writes it, with the 1024-byte header skipped. Leaf i is tokens [64i, 64i+64):

    n_leaves = n_tokens / 64          (integer division; trailing tokens are DROPPED --
                                       the count is a consensus constant and must not
                                       depend on a remainder)
    leaf_i   = 128 bytes, uint16 LE

HASHING.

    H_leaf(i)    = SHA256(0x00 || leaf_i)
    H_node(l, r) = SHA256(0x01 || l || r)

Three details, each a place where independent implementations silently diverge:

  * DOMAIN SEPARATION (the 0x00/0x01 prefixes). Without it a leaf can be reinterpreted
    as an internal node -- the second-preimage weakness in Bitcoin's merkle tree.
  * AN ODD LEVEL PROMOTES ITS LAST NODE UNCHANGED, it does not duplicate it. Duplicating
    (Bitcoin's CVE-2012-2459) lets two distinct trees share a root.
  * NO PADDING LEAVES. A zero-filled leaf is a valid token sequence, so padding the leaf
    count to a power of two would make tokens provable that are not in the corpus.

WHICH WINDOW SCORES A RECORD. The index is derived from the record's DELTA entries only:

    commit = SHA256(SHA256(prev_head || delta_entry_bytes))
    index  = commit mod n_leaves

Deriving it from the deltas rather than from the whole record is what breaks the
circularity -- the proof entry is part of the record, so an index derived from the record
would depend on the proof that depends on the index. Deriving it from the deltas also
preserves commit-then-evaluate exactly as the old in-corpus val batch did: a miner must
fix its deltas before it can learn which tokens will score them, so it cannot train on
the window it will be judged on. The record's chained head is unchanged and still covers
everything, proof included.
*/
#ifndef BTMCORPUS_H
#define BTMCORPUS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "sha256.h"

//! The proof entry's delta_type and fixed-size prefix. These live here rather than with
//! the other delta types because they ARE part of the corpus commitment's wire format:
//! llmc/btmmoe.h includes this header to size the entry, and the miner's build half
//! includes it to write one.
//!   <u8 type=0><u8 layer=0><u8 dtype=4><u8 rank=0>
//!   <u64 leaf_index><64 x u16 tokens><u8 depth><depth x 32B siblings>
#define BTM2_DELTA_CORPUS_PROOF 4
#define BTM2_PROOF_FIXED_BYTES (4 + 8 + 128 + 1)

#define BTMC_LEAF_TOKENS 64
#define BTMC_LEAF_BYTES (BTMC_LEAF_TOKENS * 2)
#define BTMC_MAX_DEPTH 64   /* 2^64 leaves; bounds a hostile proof's declared length */

//! Leaf hash: SHA256(0x00 || 128 bytes of uint16 LE tokens).
static inline void btmc_hash_leaf(const unsigned char* leaf_bytes, unsigned char out[32]) {
    unsigned char buf[1 + BTMC_LEAF_BYTES];
    buf[0] = 0x00;
    memcpy(buf + 1, leaf_bytes, BTMC_LEAF_BYTES);
    sha256_hash(buf, sizeof(buf), out);
}

//! Internal node: SHA256(0x01 || left || right).
static inline void btmc_hash_node(const unsigned char l[32], const unsigned char r[32],
                                  unsigned char out[32]) {
    unsigned char buf[1 + 64];
    buf[0] = 0x01;
    memcpy(buf + 1, l, 32);
    memcpy(buf + 33, r, 32);
    sha256_hash(buf, sizeof(buf), out);
}

//! The depth of the tree over `n_leaves`, i.e. how many sibling hashes a proof carries.
//! 0 for a single leaf (the leaf IS the root). Promotion of odd levels means the depth
//! is exactly ceil(log2(n_leaves)).
static inline int btmc_depth(uint64_t n_leaves) {
    int d = 0;
    uint64_t n = n_leaves;
    while (n > 1) { n = (n + 1) / 2; d++; }
    return d;
}

//! Fold a proof: hash the leaf, then combine with each sibling, choosing the side from
//! the index's bit at that level. Returns 0 and writes `root_out` on success, non-zero if
//! the shape is impossible (depth disagrees with n_leaves, or the index is out of range).
//!
//! The level width is tracked so that promotion is reproduced faithfully: at a level with
//! an odd count, the final node has NO sibling and is carried up untouched. A proof for
//! such a position must not supply a sibling for that level, and `siblings` is indexed
//! only for the levels that actually pair.
static inline int btmc_fold(uint64_t index, uint64_t n_leaves,
                            const unsigned char* leaf_bytes,
                            const unsigned char* siblings, int n_siblings,
                            unsigned char root_out[32]) {
    if (n_leaves == 0 || index >= n_leaves) return -1;
    if (n_siblings < 0 || n_siblings > BTMC_MAX_DEPTH) return -2;

    unsigned char cur[32];
    btmc_hash_leaf(leaf_bytes, cur);

    uint64_t width = n_leaves;
    uint64_t pos = index;
    int used = 0;
    while (width > 1) {
        if (pos + 1 == width && (width & 1u)) {
            // Last node of an odd level: promoted unchanged, consumes no sibling.
            pos /= 2;
            width = (width + 1) / 2;
            continue;
        }
        if (used >= n_siblings) return -3; // proof too short for this position
        const unsigned char* sib = siblings + (size_t)used * 32;
        unsigned char next[32];
        if (pos & 1u) btmc_hash_node(sib, cur, next);
        else          btmc_hash_node(cur, sib, next);
        memcpy(cur, next, 32);
        used++;
        pos /= 2;
        width = (width + 1) / 2;
    }
    if (used != n_siblings) return -4; // proof longer than this position needs
    memcpy(root_out, cur, 32);
    return 0;
}

//! Which leaf scores a record: SHA256d over the record's chaining preimage, truncated
//! just past its DELTA entries.
//!
//! `preimage` is the record with its first 32 bytes replaced by the previous record's
//! head (or the anchor block hash / the 0xff genesis sentinel) -- exactly the buffer the
//! record's own head is computed over. `prefix_len` is 34 + the total size of the delta
//! entries, i.e. everything up to but not including the proof entry.
//!
//! Taking a PREFIX is what makes this allocation-free, which matters because the verifier
//! is freestanding wasm with no malloc. It also pins a canonical ordering: the proof
//! entry must come LAST, after every delta entry, so that the bytes the index depends on
//! are contiguous and the index cannot depend on the proof that depends on it.
static inline uint64_t btmc_leaf_index(const unsigned char* preimage, size_t prefix_len,
                                       uint64_t n_leaves) {
    unsigned char h1[32], h2[32];
    sha256_hash(preimage, prefix_len, h1);
    sha256_hash(h1, 32, h2);
    // Little-endian u64 from the first 8 bytes, matching how the rest of this format
    // reads integers out of hashes.
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | h2[i];
    return n_leaves ? (v % n_leaves) : 0;
}

#endif // BTMCORPUS_H
