/*
Where the trunk's record chain comes from.

Two backends behind one interface. On wasm it is the node's `chain` imports; natively it is
a local chain file, so the same fold path stays testable without a node. The split exists
because the shapes differ in a way that matters: the module gets records ONE AT A TIME and
must fold and discard, while a local test can hold the file.

WHY THE MODULE DOES NOT PARSE BLOCKS. chain.slot_block() hands back a whole serialized
block, and recovering a solution from one means reimplementing the native header formats:
80 bytes for SCRYPT/SHA256D/YESCRYPT/ARGON2/X17/LYRA2REv2, but EQUIHASH (slot 6) adds
hashReserved, swaps nNonce for nNonce256 and appends a ~1344-byte nSolution, CRYPTONIGHT
(slot 7) uses vector_format rather than the field layout at all, and auxpow on any of them
embeds a parent header that may itself be either. A dynamic algo can be voted onto ANY
slot, so a module-side parser needs all of them.

And a module is IMMUTABLE once pushed and voted in. If a ninth algo is added, a header
changes at a fork, or the algo is activated on a slot its author did not anticipate, a
module-side parser breaks and cannot be updated -- while the node's parser evolves with
the node. So block parsing is structurally the host's job, and this header is written
against one additional import that does it:

    chain.slot_block_solution(i32 index, i32 ptr, i32 cap) -> i32

      Writes the solution BYTES of the slot's block at `index` -- seq=1..N-1 concatenated,
      exactly what src/dynamicalgo/solution.cpp's ExtractBlockSolution yields -- and
      returns the size. Returns 0 when that block carries no solution OR its verdict was
      not valid, and the usual SLOT_BLOCK_* codes otherwise. Charged like slot_block: one
      I/O call plus the bytes written.

Gating on the verdict inside it is what collapses two charged calls into one. A stateful
algo must check the verdict before folding anyway -- fall-through means a record sitting in
a block is no evidence about itself, so folding an unapproved one would corrupt the state
identically on every node -- and it can never legitimately want an unapproved record.

The bigger gain is that it makes the epoch DETERMINISTIC. slot_block charges io_bytes on
the whole serialized block (wasmexec.cpp: io_bytes += blk.size()), so what a fold costs
depends on how full the slot's blocks happen to be -- which is outside this algo's control.
MAX_BLOCK_WEIGHT caps non-witness data at 1 MB (weight is 4x base size with no witness)
and MAX_BLOCK_SERIALIZED_SIZE allows 4 MB once witness data is counted, so at E=32 the
checkpoint epoch would swing with other people's transaction volume:

    block is just the record (~600 KB)   1,508 records   16.8 days   (calls-bound)
    block carries 1 MB                   1,072 records   11.9 days   (bytes-bound)
    block near the 4 MB ceiling            268 records    3.0 days   (bytes-bound)

Charging on the solution instead pins it at 593 KB per record: 1,808 records, 20.1 days,
regardless of what else is in the block. An algo whose mining period shortens by 5x
because the chain got busy is not one whose parameters can be reasoned about.

NOT YET AVAILABLE: that import does not exist in the node yet. The wasm path below declares
it and will link against it when it does; until then only the native backend runs.

INDEX ORDER, the one easy thing to get backwards: index 0 is the NEWEST slot block below
the one being validated, so folding in chain order means counting DOWN from count-1 to 0.
*/
#ifndef BTMCHAIN_H
#define BTMCHAIN_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

//! Mirrors src/dynamicalgo/wasmexec.h. Negative means nothing was written.
#define BTMCHAIN_NO_BLOCK   (-1)
#define BTMCHAIN_BAD_ADDR   (-2)
#define BTMCHAIN_TOO_SMALL  (-3)
#define BTMCHAIN_UNAVAILABLE (-4)   /*!< local fault; the node traps rather than returning */

#ifdef BTM_ONCHAIN

__attribute__((import_module("chain"), import_name("slot_block_count")))
extern int chain_slot_block_count(void);

__attribute__((import_module("chain"), import_name("slot_block_solution_valid")))
extern int chain_slot_block_solution_valid(int index);

#ifdef BTMCHAIN_STUB_SOLUTION
/* MEASUREMENT BUILDS ONLY. The node does not register chain.slot_block_solution yet, so a
 * module importing it cannot be instantiated -- which blocks measuring gas under the
 * node's own instrumentation (contrib/dynamicalgo/gasinstrument + aot_harness). Defining
 * it locally as "no solution here" reports an empty chain, which exercises the genesis
 * path: proof check, model build, two forwards, gate. That is the whole instruction-gas
 * cost; folding contributes I/O gas only, and with an empty chain there is none to fold.
 * NEVER build the deployed module with this. */
static int chain_slot_block_solution(int index, void* ptr, int cap) {
    (void)index; (void)ptr; (void)cap;
    return 0;
}
#else
__attribute__((import_module("chain"), import_name("slot_block_solution")))
extern int chain_slot_block_solution(int index, void* ptr, int cap);
#endif

typedef struct { int n; } btmchain;

static inline int btmchain_open(btmchain* ch) {
    ch->n = chain_slot_block_count();
    return ch->n < 0 ? -1 : 0;
}
static inline void btmchain_close(btmchain* ch) { (void)ch; }
static inline int btmchain_count(const btmchain* ch) { return ch->n; }

static inline int btmchain_valid(const btmchain* ch, int index) {
    (void)ch;
    return chain_slot_block_solution_valid(index);
}
static inline int btmchain_solution(const btmchain* ch, int index,
                                    unsigned char* dst, size_t cap) {
    (void)ch;
    if (cap > (size_t)0x7fffffff) cap = 0x7fffffff;
    return chain_slot_block_solution(index, dst, (int)cap);
}

#else  /* ---- native test backend: a local chain file stands in for the slot ---- */

#include <stdio.h>
#include <stdlib.h>

//! A local chain file read whole. Record k (0 = genesis) corresponds to slot block index
//! n_records-1-k, so the index order matches the host's. Every record in the file counts
//! as valid: the file only ever holds records a verifier accepted, which is the same
//! invariant the verdict gate enforces on-chain.
typedef struct {
    unsigned char* cp;
    size_t cp_bytes;
    size_t* off;        /*!< byte offset of each record, genesis first */
    size_t* len;
    int n_records;
} btmchain;

//! `record_size` walks one record given its start and the bytes remaining, returning 0 on
//! malformation -- i.e. btmmoe_record_size_h bound to this chain's C/H/E by the caller, so
//! this header needs no knowledge of the record format.
typedef size_t (*btmchain_sizer)(const unsigned char* rec, size_t max_bytes, void* ctx);

static inline int btmchain_open_file(btmchain* ch, const char* path,
                                     btmchain_sizer sizer, void* sctx) {
    memset(ch, 0, sizeof(*ch));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;                       /* absent == an empty chain, not an error */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    ch->cp_bytes = (size_t)sz;
    ch->cp = (unsigned char*)malloc(ch->cp_bytes ? ch->cp_bytes : 1);
    if (!ch->cp) { fclose(f); return -1; }
    if (ch->cp_bytes && fread(ch->cp, 1, ch->cp_bytes, f) != ch->cp_bytes) {
        fclose(f); free(ch->cp); ch->cp = NULL; return -1;
    }
    fclose(f);

    size_t cap = 16;
    ch->off = (size_t*)malloc(cap * sizeof(size_t));
    ch->len = (size_t*)malloc(cap * sizeof(size_t));
    if (!ch->off || !ch->len) return -1;
    size_t o = 0;
    while (o < ch->cp_bytes) {
        size_t rsz = sizer(ch->cp + o, ch->cp_bytes - o, sctx);
        if (rsz == 0) return -1;            /* malformed chain file */
        if ((size_t)ch->n_records == cap) {
            cap *= 2;
            size_t* a = (size_t*)realloc(ch->off, cap * sizeof(size_t));
            size_t* b = (size_t*)realloc(ch->len, cap * sizeof(size_t));
            if (!a || !b) return -1;
            ch->off = a; ch->len = b;
        }
        ch->off[ch->n_records] = o;
        ch->len[ch->n_records] = rsz;
        ch->n_records++;
        o += rsz;
    }
    return 0;
}

static inline void btmchain_close(btmchain* ch) {
    free(ch->cp); free(ch->off); free(ch->len);
    memset(ch, 0, sizeof(*ch));
}
static inline int btmchain_count(const btmchain* ch) { return ch->n_records; }

static inline int btmchain_valid(const btmchain* ch, int index) {
    if (index < 0 || index >= ch->n_records) return BTMCHAIN_NO_BLOCK;
    return 1;
}
static inline int btmchain_solution(const btmchain* ch, int index,
                                    unsigned char* dst, size_t cap) {
    if (index < 0 || index >= ch->n_records) return BTMCHAIN_NO_BLOCK;
    int k = ch->n_records - 1 - index;      /* index 0 == newest == last record */
    if (ch->len[k] > cap) return BTMCHAIN_TOO_SMALL;
    memcpy(dst, ch->cp + ch->off[k], ch->len[k]);
    return (int)ch->len[k];
}

#endif /* BTM_ONCHAIN */

/* ---- backend-independent -------------------------------------------------------- */

//! Fold every approved record, OLDEST FIRST, through `fn`. One buffer is reused for all
//! of them, which is the fold-and-discard pattern gasclasses.h describes: the state
//! exceeds linear memory in the general case, so a record is applied and dropped rather
//! than accumulated.
//!
//! `max_records` caps how far back to go (0 = the whole addressable window). Blocks with
//! no valid solution are SKIPPED, not folded: a record in a block is no evidence about
//! itself, so only verdict-approved records may touch the weights.
//!
//! Returns the number folded, or negative on error. `fn` returning non-zero aborts and
//! that value is returned.
typedef int (*btmchain_record_fn)(const unsigned char* rec, size_t len, void* ctx);

static inline int btmchain_fold(const btmchain* ch, int max_records,
                                unsigned char* buf, size_t bufcap,
                                btmchain_record_fn fn, void* ctx) {
    int n = btmchain_count(ch);
    if (n < 0) return n;
    /* `max_records` keeps the NEWEST that many, so it caps where the walk STARTS, not
     * where it ends. Capping the end instead folds the OLDEST records and discards the
     * recent ones -- which is silently wrong rather than visibly broken, because the
     * fold still succeeds and still produces a state. Index 0 is the newest, so the
     * oldest index to fold is max_records-1 and the walk counts down to 0. */
    int start = n - 1;                         /* oldest addressable index */
    if (max_records > 0 && max_records < n) start = max_records - 1;
    int folded = 0;
    for (int i = start; i >= 0; i--) {         /* count DOWN: index 0 is the newest */
        int sz = btmchain_solution(ch, i, buf, bufcap);
        if (sz == 0) continue;                 /* no solution, or verdict not valid */
        if (sz < 0) {
            if (sz == BTMCHAIN_NO_BLOCK) continue;
            return sz;                         /* TOO_SMALL or BAD_ADDR: a real fault */
        }
        int r = fn(buf, (size_t)sz, ctx);
        if (r != 0) return r;
        folded++;
    }
    return folded;
}

//! V for the retargeting rule: how many of the last `window` slot blocks carried a valid
//! solution. Costs one I/O call per probe and ZERO bytes, which is why the rule counts
//! verdicts rather than timing blocks -- a timestamp would mean fetching a whole block.
//!
//! `*window_out` receives how many blocks were actually examined, which is smaller than
//! `window` on a chain younger than it; btmmoe_gate_fraction takes that as its second
//! argument and returns the base fraction when it is 0.
static inline uint32_t btmchain_count_valid(const btmchain* ch, uint32_t window,
                                            uint32_t* window_out) {
    int n = btmchain_count(ch);
    if (n <= 0) { if (window_out) *window_out = 0; return 0; }
    uint32_t w = (uint32_t)n < window ? (uint32_t)n : window;
    uint32_t v = 0;
    for (uint32_t i = 0; i < w; i++) {
        if (btmchain_valid(ch, (int)i) == 1) v++;
    }
    if (window_out) *window_out = w;
    return v;
}

#endif /* BTMCHAIN_H */
