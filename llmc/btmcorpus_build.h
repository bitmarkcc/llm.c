/*
Miner-side construction of a corpus-commitment proof. NOT included by the verifier:
this half uses stdio, malloc and glob, while llmc/btmcorpus.h (the verifying half) is
freestanding so it can compile into the wasm module. Keeping them in separate headers
is what stops a host-only dependency leaking into consensus code -- an allocation in the
verify path compiles natively and fails only at the very end of a wasm port.

A miner needs two things the verifier does not: the corpus itself, to read the window's
tokens, and the level-0 leaf hashes, to build the sibling path. The leaf hashes can be
recomputed from the corpus (fine for a small corpus) or loaded from the file
dev/btm_corpus_root.py --save-leaves writes (necessary at 10B tokens, where rehashing
takes minutes and the array is ~6 GB).

SHARDING. A 10B-token corpus cannot be one file: llm.c's .bin header stores the token
count in an int32, capping a shard at 2^31 tokens, and dev/data/edu_fineweb10B_16k.py
therefore writes ~128 shards of 100M tokens. The corpus is their concatenation in sorted
filename order, and every shard but the last holds a token count divisible by 64, so a
leaf never straddles two files and leaf i is found with a single seek:

    shard  = i / leaves_per_shard
    offset = BTMCB_HEADER_BYTES + (i % leaves_per_shard) * BTMC_LEAF_BYTES

btmcb_corpus_open checks that invariant rather than trusting it: a corpus built without
it yields proofs that silently fail to fold, and the error would appear to be in the
Merkle code rather than in the data.
*/
#ifndef BTMCORPUS_BUILD_H
#define BTMCORPUS_BUILD_H

#include "btmcorpus.h"

#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BTMCB_HEADER_BYTES 1024  /* the .bin token-file header dev/data/*.py writes */
#define BTMCB_MAGIC 20240520

//! An opened sharded corpus: the shard paths in commitment order plus the geometry
//! needed to address a leaf.
typedef struct {
    glob_t g;
    uint64_t leaves_per_shard;  /* leaves in every shard but the last */
    uint64_t n_leaves;          /* whole leaves across the whole corpus */
} btmcb_corpus;

static inline void btmcb_corpus_close(btmcb_corpus* c);

static int btmcb_strcmp_(const void* a, const void* b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

//! Read a shard's token count, or 0 if the file is not a valid shard.
static inline uint64_t btmcb_shard_tokens(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int hdr[256];
    if (fread(hdr, sizeof(int), 256, f) != 256) { fclose(f); return 0; }
    fclose(f);
    if (hdr[0] != BTMCB_MAGIC || hdr[1] != 1 || hdr[2] <= 0) return 0;
    return (uint64_t)hdr[2];
}

//! Open every shard matching `pattern` (a glob, or a single path). Returns 0 on success.
static inline int btmcb_corpus_open(btmcb_corpus* c, const char* pattern) {
    memset(c, 0, sizeof(*c));
    if (glob(pattern, 0, NULL, &c->g) != 0 || c->g.gl_pathc == 0) {
        fprintf(stderr, "corpus: nothing matches %s\n", pattern);
        return -1;
    }
    // Sort with strcmp explicitly. glob() orders by the locale's collating sequence,
    // which is not byte order everywhere, and the shard order IS the corpus order --
    // a locale-dependent corpus would hash to a locale-dependent root.
    qsort(c->g.gl_pathv, c->g.gl_pathc, sizeof(char*), btmcb_strcmp_);

    size_t n = c->g.gl_pathc;
    uint64_t first = btmcb_shard_tokens(c->g.gl_pathv[0]);
    if (first == 0) {
        fprintf(stderr, "corpus: %s is not a valid shard\n", c->g.gl_pathv[0]);
        btmcb_corpus_close(c);
        return -2;
    }
    c->leaves_per_shard = first / BTMC_LEAF_TOKENS;
    c->n_leaves = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t t = btmcb_shard_tokens(c->g.gl_pathv[i]);
        if (t == 0) {
            fprintf(stderr, "corpus: %s is not a valid shard\n", c->g.gl_pathv[i]);
            globfree(&c->g);
            return -2;
        }
        if (i + 1 < n) {
            // Not the last shard: must be leaf-aligned AND the same size as the first,
            // or the constant-divisor addressing below is wrong.
            if (t % BTMC_LEAF_TOKENS) {
                fprintf(stderr, "corpus: %s has %llu tokens, not a multiple of %d -- "
                        "leaves would straddle shards\n", c->g.gl_pathv[i],
                        (unsigned long long)t, BTMC_LEAF_TOKENS);
                globfree(&c->g);
                return -3;
            }
            if (t != first) {
                fprintf(stderr, "corpus: %s has %llu tokens, shard 0 has %llu -- "
                        "only the LAST shard may differ\n", c->g.gl_pathv[i],
                        (unsigned long long)t, (unsigned long long)first);
                globfree(&c->g);
                return -4;
            }
        }
        c->n_leaves += t / BTMC_LEAF_TOKENS;
    }
    return 0;
}

static inline void btmcb_corpus_close(btmcb_corpus* c) {
    if (c->g.gl_pathv) globfree(&c->g);
    memset(c, 0, sizeof(*c));
}

//! Read leaf `index`'s 64 tokens (128 bytes). Returns 0 on success.
static inline int btmcb_corpus_read_leaf(const btmcb_corpus* c, uint64_t index,
                                         unsigned char out[BTMC_LEAF_BYTES]) {
    if (index >= c->n_leaves) return -1;
    uint64_t s = index / c->leaves_per_shard;
    uint64_t k = index % c->leaves_per_shard;
    if (s >= (uint64_t)c->g.gl_pathc) return -2;
    FILE* f = fopen(c->g.gl_pathv[s], "rb");
    if (!f) return -3;
    long off = (long)BTMCB_HEADER_BYTES + (long)(k * BTMC_LEAF_BYTES);
    if (fseek(f, off, SEEK_SET) != 0) { fclose(f); return -4; }
    size_t got = fread(out, 1, BTMC_LEAF_BYTES, f);
    fclose(f);
    return got == BTMC_LEAF_BYTES ? 0 : -5;
}

//! All level-0 leaf hashes, freshly hashed from the corpus. Caller frees.
//! Practical only for a small corpus: at 10B tokens this is ~6 GB and minutes of
//! hashing, which is what the --save-leaves cache exists to avoid.
//!
//! Streams each shard once rather than calling btmcb_corpus_read_leaf per leaf: that
//! helper reopens the file on every call, which is the right trade for the ONE leaf an
//! attempt actually scores and completely wrong for 177M of them.
static inline unsigned char* btmcb_build_leaves(const btmcb_corpus* c,
                                                uint64_t* n_leaves_out) {
    unsigned char* out = (unsigned char*)malloc((size_t)c->n_leaves * 32);
    if (!out) return NULL;
    enum { CHUNK = 4096 };                      /* leaves per read: 512 KB */
    unsigned char* blob = (unsigned char*)malloc((size_t)CHUNK * BTMC_LEAF_BYTES);
    if (!blob) { free(out); return NULL; }

    uint64_t done = 0;
    for (size_t s = 0; s < c->g.gl_pathc; s++) {
        uint64_t ntok = btmcb_shard_tokens(c->g.gl_pathv[s]);
        if (ntok == 0) { free(blob); free(out); return NULL; }
        uint64_t whole = ntok / BTMC_LEAF_TOKENS;
        FILE* f = fopen(c->g.gl_pathv[s], "rb");
        if (!f) { free(blob); free(out); return NULL; }
        if (fseek(f, BTMCB_HEADER_BYTES, SEEK_SET) != 0) {
            fclose(f); free(blob); free(out); return NULL;
        }
        uint64_t left = whole;
        while (left) {
            size_t take = left < (uint64_t)CHUNK ? (size_t)left : (size_t)CHUNK;
            if (fread(blob, 1, take * BTMC_LEAF_BYTES, f) != take * BTMC_LEAF_BYTES) {
                fclose(f); free(blob); free(out); return NULL;
            }
            for (size_t k = 0; k < take; k++) {
                if (done >= c->n_leaves) { fclose(f); free(blob); free(out); return NULL; }
                btmc_hash_leaf(blob + k * BTMC_LEAF_BYTES, out + (size_t)done * 32);
                done++;
            }
            left -= take;
        }
        fclose(f);
    }
    free(blob);
    if (done != c->n_leaves) { free(out); return NULL; }
    *n_leaves_out = c->n_leaves;
    return out;
}

//! Level-0 hashes from the file written by `btm_corpus_root.py --save-leaves`, which is
//! how a miner avoids re-hashing the corpus on every attempt. Caller frees.
static inline unsigned char* btmcb_load_leaves(const char* leaves_path,
                                               uint64_t* n_leaves_out) {
    FILE* f = fopen(leaves_path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || (sz % 32)) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    unsigned char* out = (unsigned char*)malloc((size_t)sz);
    if (!out) { fclose(f); return NULL; }
    if (fread(out, 1, (size_t)sz, f) != (size_t)sz) { free(out); fclose(f); return NULL; }
    fclose(f);
    *n_leaves_out = (uint64_t)sz / 32;
    return out;
}

//! Build the sibling path for `index`, bottom-up, omitting the levels where this
//! position is the promoted last node of an odd level -- exactly the levels btmc_fold
//! skips. Writes at most BTMC_MAX_DEPTH hashes to `sibs_out` and the count to
//! `*n_sibs_out`. Returns 0 on success.
//!
//! DESTRUCTIVE: folds in place and leaves `leaves` holding garbage. That is deliberate.
//! The obvious version copies the array first, and at 10B tokens the array is ~6 GB, so
//! the copy doubles a miner's peak to 12 GB on top of a 2 GB model. The caller reloads
//! or rebuilds the leaves for the next attempt instead.
//!
//! O(n_leaves) work: ~190 s at 187M leaves, the price of not storing the whole tree.
//! A miner doing this at scale should keep the upper levels cached and fold only the
//! 16k-leaf subtree under the cut -- milliseconds instead -- but that is an
//! optimisation, not a change to what is computed.
static inline int btmcb_path_destructive(unsigned char* leaves, uint64_t n_leaves,
                                         uint64_t index, unsigned char* sibs_out,
                                         int* n_sibs_out) {
    if (!leaves || n_leaves == 0 || index >= n_leaves) return -1;
    uint64_t width = n_leaves, pos = index;
    int used = 0;
    while (width > 1) {
        int promoted = (pos + 1 == width) && (width & 1u);
        if (!promoted) {
            if (used >= BTMC_MAX_DEPTH) return -3;
            uint64_t sib = (pos & 1u) ? pos - 1 : pos + 1;
            memcpy(sibs_out + (size_t)used * 32, leaves + (size_t)sib * 32, 32);
            used++;
        }
        // Collapse this level into the next, promoting an odd tail unchanged.
        uint64_t nw = 0;
        for (uint64_t i = 0; i + 1 < width; i += 2) {
            btmc_hash_node(leaves + (size_t)i * 32, leaves + (size_t)(i + 1) * 32,
                           leaves + (size_t)nw * 32);
            nw++;
        }
        if (width & 1u) {
            memmove(leaves + (size_t)nw * 32, leaves + (size_t)(width - 1) * 32, 32);
            nw++;
        }
        pos /= 2;
        width = nw;
    }
    *n_sibs_out = used;
    return 0;
}

//! Build the sibling path for `index` by reading the stored tree written by
//! `btm_corpus_root.py --save-tree`: every level, bottom-up, levels concatenated,
//! 32 bytes per node. Returns 0 on success.
//!
//! This is what btmcb_path_destructive does, without the fold. It touches one 32-byte
//! sibling per level instead of rehashing the whole tree -- 28 reads against 240 s and
//! 6 GB of residency at 177M leaves -- and needs no array in memory at all. The widths
//! are recomputed here rather than stored alongside the data, so the file carries no
//! header that could disagree with it: given n_leaves, the layout is determined.
//!
//! The file is NOT consensus data. A corrupt or stale tree yields a proof that fails to
//! fold, which the miner's own self-check catches before publishing.
static inline int btmcb_path_from_tree(const char* tree_path, uint64_t n_leaves,
                                       uint64_t index, unsigned char* sibs_out,
                                       int* n_sibs_out) {
    // The tree is ~11 GB at 177M leaves, so its offsets exceed 2^31. fseek takes a long,
    // which is 64-bit on LP64 but 32-bit elsewhere -- where this would silently seek to
    // the wrong node and produce a proof that merely fails to fold. Fail at build time
    // instead of debugging the Merkle code.
    typedef char btmcb_tree_offsets_fit_[(sizeof(long) >= 8) ? 1 : -1];
    (void)sizeof(btmcb_tree_offsets_fit_);

    if (n_leaves == 0 || index >= n_leaves) return -1;
    FILE* f = fopen(tree_path, "rb");
    if (!f) return -2;

    uint64_t off = 0;        /* node offset of the current level, in nodes */
    uint64_t width = n_leaves, pos = index;
    int used = 0;
    while (width > 1) {
        int promoted = (pos + 1 == width) && (width & 1u);
        if (!promoted) {
            if (used >= BTMC_MAX_DEPTH) { fclose(f); return -3; }
            uint64_t sib = (pos & 1u) ? pos - 1 : pos + 1;
            if (fseek(f, (long)((off + sib) * 32), SEEK_SET) != 0) { fclose(f); return -4; }
            if (fread(sibs_out + (size_t)used * 32, 1, 32, f) != 32) { fclose(f); return -5; }
            used++;
        }
        off += width;
        pos /= 2;
        width = (width + 1) / 2;
    }
    fclose(f);
    *n_sibs_out = used;
    return 0;
}

//! Total size in bytes of the file --save-tree writes, for a sanity check before use.
static inline uint64_t btmcb_tree_bytes(uint64_t n_leaves) {
    uint64_t total = 0, w = n_leaves;
    if (w == 0) return 0;
    for (;;) {
        total += w;
        if (w == 1) break;
        w = (w + 1) / 2;
    }
    return total * 32;
}

//! Serialize a BTM2_DELTA_CORPUS_PROOF entry into `out` (which must hold
//! BTM2_PROOF_FIXED_BYTES + 32*n_sibs). Returns the byte count written.
static inline size_t btmcb_write_proof_entry(unsigned char* out, uint64_t index,
                                             const unsigned char leaf[BTMC_LEAF_BYTES],
                                             const unsigned char* sibs, int n_sibs) {
    out[0] = 0;                        /* tensor type: unused by a proof entry */
    out[1] = 0;                        /* layer:       unused */
    out[2] = BTM2_DELTA_CORPUS_PROOF;  /* delta_type */
    out[3] = 0;                        /* rank:        unused */
    for (int i = 0; i < 8; i++) out[4 + i] = (unsigned char)(index >> (8 * i));
    memcpy(out + 12, leaf, BTMC_LEAF_BYTES);
    out[12 + BTMC_LEAF_BYTES] = (unsigned char)n_sibs;
    memcpy(out + 13 + BTMC_LEAF_BYTES, sibs, (size_t)n_sibs * 32);
    return (size_t)13 + BTMC_LEAF_BYTES + (size_t)n_sibs * 32;
}

#endif // BTMCORPUS_BUILD_H
