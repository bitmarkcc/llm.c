/*
The slot's PERSISTENT ALGO STATE, as the verifier sees it.

WHY THIS EXISTS. The trunk's base weights are 1.076 GB. verify() used to rebuild them
every call -- the fixed-seed init, then every approved record folded in -- and that does
not fit the gas budget by an order of magnitude. Measured under the node's own
instrumentation, the init ALONE was 18x the MEM cap; nine classes were over. Out-of-gas
prices the no-solution branch, so the algo would have run, blocks would have been valid,
and not one record would ever have been accepted. Silently, forever.

So the state is held off chain by the node, exactly as the UTXO set is -- derived from the
chain, never carried by it -- and read through these imports. Reading it is a memcpy
against the I/O byte budget and costs no scalar MEM, FADD or CALL at all. See
src/dynamicalgo/statedb.h in the bitmark tree for the store, and sec 8.7 for the budget.

WRITES DO NOT WRITE. btmstate_write only DESCRIBES a byte range; the node collects the
ranges and applies them when the block connects, and only if the solution was accepted.
verify() stays a pure function, which is what makes it safe to call at mempool time -- a
write that reached the store from inside it would let an unconfirmed solution mutate
consensus state. Nothing here can make anything durable.

THREE BUILDS, as everywhere else in this verifier (see btmhost.h). Only the on-chain build
has the imports; native and WASI report no state, which makes them take the init-and-fold
path. That is deliberate and not a stub for convenience: the harness is the only way to
cross-check the SIMD kernels against scalar for bit-exactness, so it has to be able to
reach the same weights by the other route.
*/
#ifndef BTMSTATE_H
#define BTMSTATE_H

#include <stddef.h>
#include <stdint.h>

#include "btmhost.h"

/*! Returned by btmstate_read/_write when the slot has no state, or there is no store.
 *  The two are ONE code on purpose: from here they are the same fact -- there is nothing
 *  to read -- and telling them apart would be learning something about the node rather
 *  than about the chain. */
#define BTMSTATE_NO_STATE     (-1)
#define BTMSTATE_OUT_OF_RANGE (-2)
#define BTMSTATE_TOO_LARGE    (-3)

#ifdef BTM_ONCHAIN

__attribute__((import_module("chain"), import_name("state_size")))
extern int64_t chain_state_size(void);

__attribute__((import_module("chain"), import_name("state_read")))
extern int chain_state_read(int64_t offset, void* dst, int len);

__attribute__((import_module("chain"), import_name("state_write")))
extern int chain_state_write(int64_t offset, const void* src, int len);

static inline uint64_t btmstate_size(void) {
    int64_t n = chain_state_size();
    return n > 0 ? (uint64_t)n : 0;
}

/*! Slice size for streaming. 64 MiB keeps the call count low -- 1.076 GB is 17 calls
 *  against GAS_IO_CALLS_LIMIT of 4096 -- without asking the host for a single
 *  gigabyte-sized read. The value is NOT consensus: calls and bytes are both metered, so
 *  a different slicing only changes how many calls are charged, and any slicing that
 *  stays inside the budgets produces identical bytes. */
#define BTMSTATE_SLICE (64u * 1024u * 1024u)

/*! Read `n` bytes from `offset` into `dst`, slicing as needed. 0 on success. */
static inline int btmstate_read(uint64_t offset, void* dst, uint64_t n) {
    unsigned char* p = (unsigned char*)dst;
    while (n > 0) {
        uint32_t want = n > BTMSTATE_SLICE ? BTMSTATE_SLICE : (uint32_t)n;
        int got = chain_state_read((int64_t)offset, p, (int)want);
        if (got != (int)want) return got < 0 ? got : -1;
        offset += want; p += want; n -= want;
    }
    return 0;
}

/*! Ask for `n` bytes at `src` to be written to `offset`. 0 on success. */
static inline int btmstate_write(uint64_t offset, const void* src, uint64_t n) {
    const unsigned char* p = (const unsigned char*)src;
    while (n > 0) {
        uint32_t want = n > BTMSTATE_SLICE ? BTMSTATE_SLICE : (uint32_t)n;
        int put = chain_state_write((int64_t)offset, p, (int)want);
        if (put != (int)want) return put < 0 ? put : -1;
        offset += want; p += want; n -= want;
    }
    return 0;
}

#else /* native and WASI harness: no store, so no state */

static inline uint64_t btmstate_size(void) { return 0; }
static inline int btmstate_read(uint64_t offset, void* dst, uint64_t n) {
    (void)offset; (void)dst; (void)n; return BTMSTATE_NO_STATE;
}
static inline int btmstate_write(uint64_t offset, const void* src, uint64_t n) {
    (void)offset; (void)src; (void)n; return BTMSTATE_NO_STATE;
}

#endif /* BTM_ONCHAIN */

#endif /* BTMSTATE_H */
