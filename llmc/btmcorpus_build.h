/*
Miner-side construction of a corpus-commitment proof. NOT included by the verifier:
this half uses stdio and malloc, while llmc/btmcorpus.h (the verifying half) is
freestanding so it can compile into the wasm module. Keeping them in separate headers
is what stops a host-only dependency leaking into consensus code -- an allocation in the
verify path compiles natively and fails only at the very end of a wasm port.

A miner needs two things the verifier does not: the corpus itself, to read the window's
tokens, and the level-0 leaf hashes, to build the sibling path. The leaf hashes can be
recomputed from the corpus (fine for a small corpus) or loaded from the file
dev/btm_corpus_root.py --save-leaves writes (necessary for 10B tokens, where hashing the
corpus takes minutes and the cache is 5 GB).
*/
#ifndef BTMCORPUS_BUILD_H
#define BTMCORPUS_BUILD_H

#include "btmcorpus.h"

#include <stdio.h>
#include <stdlib.h>

#define BTMCB_HEADER_BYTES 1024  /* the .bin token-file header dev/data/*.py writes */

//! Read leaf `index`'s 64 tokens (128 bytes) out of a tokenised corpus shard.
//! Returns 0 on success.
static inline int btmcb_read_leaf(const char* corpus_path, uint64_t index,
                                  unsigned char out[BTMC_LEAF_BYTES]) {
    FILE* f = fopen(corpus_path, "rb");
    if (!f) return -1;
    long off = (long)BTMCB_HEADER_BYTES + (long)(index * BTMC_LEAF_BYTES);
    if (fseek(f, off, SEEK_SET) != 0) { fclose(f); return -2; }
    size_t got = fread(out, 1, BTMC_LEAF_BYTES, f);
    fclose(f);
    return got == BTMC_LEAF_BYTES ? 0 : -3;
}

//! How many whole leaves a corpus shard holds. 0 on failure.
static inline uint64_t btmcb_n_leaves(const char* corpus_path) {
    FILE* f = fopen(corpus_path, "rb");
    if (!f) return 0;
    int hdr[256];
    if (fread(hdr, sizeof(int), 256, f) != 256) { fclose(f); return 0; }
    fclose(f);
    if (hdr[0] != 20240520) return 0;
    return (uint64_t)hdr[2] / BTMC_LEAF_TOKENS;
}

//! All level-0 leaf hashes, freshly hashed from the corpus. Caller frees.
//! `*n_leaves_out` receives the count.
static inline unsigned char* btmcb_build_leaves(const char* corpus_path,
                                                uint64_t* n_leaves_out) {
    uint64_t n = btmcb_n_leaves(corpus_path);
    if (!n) return NULL;
    FILE* f = fopen(corpus_path, "rb");
    if (!f) return NULL;
    if (fseek(f, BTMCB_HEADER_BYTES, SEEK_SET) != 0) { fclose(f); return NULL; }
    unsigned char* out = (unsigned char*)malloc((size_t)n * 32);
    if (!out) { fclose(f); return NULL; }
    unsigned char leaf[BTMC_LEAF_BYTES];
    for (uint64_t i = 0; i < n; i++) {
        if (fread(leaf, 1, BTMC_LEAF_BYTES, f) != BTMC_LEAF_BYTES) {
            free(out); fclose(f); return NULL;
        }
        btmc_hash_leaf(leaf, out + (size_t)i * 32);
    }
    fclose(f);
    *n_leaves_out = n;
    return out;
}

//! Level-0 hashes from the file written by `btm_corpus_root.py --save-leaves`, which is
//! how a miner avoids re-hashing 20 GB on every attempt. Caller frees.
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
//! Walks level by level, rebuilding each from the one below. O(n_leaves) work and
//! O(n_leaves) scratch, which is the price of not storing the whole tree: at 156M
//! leaves that is ~5 GB of scratch and a few seconds, paid once per attempt. A miner
//! doing this at scale would keep the upper levels cached instead.
static inline int btmcb_path(const unsigned char* leaves, uint64_t n_leaves,
                             uint64_t index, unsigned char* sibs_out, int* n_sibs_out) {
    if (!leaves || n_leaves == 0 || index >= n_leaves) return -1;
    unsigned char* cur = (unsigned char*)malloc((size_t)n_leaves * 32);
    if (!cur) return -2;
    memcpy(cur, leaves, (size_t)n_leaves * 32);

    uint64_t width = n_leaves, pos = index;
    int used = 0;
    while (width > 1) {
        int promoted = (pos + 1 == width) && (width & 1u);
        if (!promoted) {
            if (used >= BTMC_MAX_DEPTH) { free(cur); return -3; }
            uint64_t sib = (pos & 1u) ? pos - 1 : pos + 1;
            memcpy(sibs_out + (size_t)used * 32, cur + (size_t)sib * 32, 32);
            used++;
        }
        // Collapse this level into the next, promoting an odd tail unchanged.
        uint64_t nw = 0;
        for (uint64_t i = 0; i + 1 < width; i += 2) {
            btmc_hash_node(cur + (size_t)i * 32, cur + (size_t)(i + 1) * 32,
                           cur + (size_t)nw * 32);
            nw++;
        }
        if (width & 1u) {
            memmove(cur + (size_t)nw * 32, cur + (size_t)(width - 1) * 32, 32);
            nw++;
        }
        pos /= 2;
        width = nw;
    }
    free(cur);
    *n_sibs_out = used;
    return 0;
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
