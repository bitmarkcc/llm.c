/*
The host facilities the verifier uses, and what they become on-chain.

A real dynamic-algo module imports NOTHING but `chain.*`. The WASI build of this verifier
imports 18 wasi_snapshot_preview1 functions -- fd_write for printf, path_open/fd_read/
fd_seek/fd_close for the chain file and the state cache, args_* for main's argv,
environ_* for getenv, clock_time_get for timing, proc_exit for exit() -- and a module with
any of those cannot be instantiated by the node, which provides none of them. This header
is the seam: every host facility goes through it, so the on-chain build drops them and the
native build keeps them.

WHAT STAYS. expf, logf, sqrtf, tanhf, powf and coshf are kept, because wasi-libc links
them STATICALLY into the module -- they become code inside the wasm, not imports. That
also settles the consensus question they would otherwise raise: a transcendental's last
bit can differ between libm implementations, but every node runs the one compiled into the
module, so there is nothing to disagree about. memcpy and memset likewise come from
compiler-rt, not from an import.

WHAT A TRAP MEANS. BTM_ABORT traps on wasm rather than exiting. That is not a loss of
information: doc sec 7 step 4 prices a solution that is malformed, misses its target, traps
or runs out of gas all the same way -- the no-solution branch -- so an internal
inconsistency correctly scores as "no valid solution" and leaves the block valid. The one
exception is chain_unavailable, which the node distinguishes by its exception message and
escalates as a local fault; nothing here produces that.
*/
#ifndef BTMHOST_H
#define BTMHOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __wasm__

/* Diagnostics cost an import (fd_write), so on-chain they cost nothing instead. */
#define BTMLOG(...) ((void)0)
#define BTMLOGV(...) ((void)0)
#define BTM_ABORT() __builtin_trap()
#define BTM_ASSERT(x) do { if (!(x)) __builtin_trap(); } while (0)

/* Timing is a diagnostic, and clock_gettime is an import (clock_time_get). It also could
 * not be used for anything consensus-bearing even if it were free: a verdict must be a
 * function of the chain alone, never of how long this node took. */
#define BTM_TIMER(a, b)   ((void)0)
#define BTM_TICK(x)       ((void)0)
#define BTM_ELAPSED(a, b) 0.0

/*! Bump allocator over linear memory.
 *
 *  There is no malloc in a -nostdlib module, and nothing to replace it with: the module
 *  is instantiated per verify() call and torn down after, so memory is reclaimed by the
 *  instance's death and freeing individually buys nothing. memory.grow hands back whole
 *  pages, which suits the allocation pattern here -- about a dozen calls, one of them
 *  1.0 GiB of weights -- so each request takes its own pages and never shares.
 *
 *  Returns NULL if the grow fails, which happens when the request would exceed the
 *  node's WASM_MAX_MEMORY_PAGES (32512, 1.984 GiB). Callers must check: a NULL here means
 *  the model does not fit, not that something went wrong. */
static inline void* btm_alloc(size_t n) {
    if (n == 0) n = 1;
    size_t pages = (n + 65535u) / 65536u;
    size_t prev = (size_t)__builtin_wasm_memory_grow(0, pages);
    if (prev == (size_t)-1) return NULL;
    return (void*)(prev * 65536u);
}
static inline void btm_free(void* p) { (void)p; }   /* the instance is the arena */

#else  /* ---- native ---- */

#include <stdio.h>
#include <stdlib.h>

#define BTMLOG(...) printf(__VA_ARGS__)
#define BTMLOGV(...) printf(__VA_ARGS__)
#define BTM_ABORT() exit(1)
#define BTM_ASSERT(x) do { if (!(x)) { printf("assertion failed: %s\n", #x); exit(1); } } while (0)

#include <time.h>
#define BTM_TIMER(a, b)   struct timespec a, b
#define BTM_TICK(x)       clock_gettime(CLOCK_MONOTONIC, &(x))
#define BTM_ELAPSED(a, b) (((b).tv_sec - (a).tv_sec) + ((b).tv_nsec - (a).tv_nsec) / 1e9)

static inline void* btm_alloc(size_t n) {
    void* p = malloc(n ? n : 1);
    if (!p) { printf("btm_alloc: out of memory requesting %zu bytes\n", n); exit(1); }
    return p;
}
static inline void btm_free(void* p) { free(p); }

#endif /* __wasm__ */

//! Parse `n` bytes of hex into `out`. Returns 0 on success, non-zero on a bad digit.
//! Replaces the sscanf("%2x") the root comparison used: sscanf is a large chunk of libc
//! to drag in for 32 bytes of hex, and on wasm it is one more thing that might reach for
//! an import.
static inline int btm_unhex(const char* hex, unsigned char* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned v = 0;
        for (int k = 0; k < 2; k++) {
            char c = hex[2 * i + k];
            unsigned d;
            if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a') + 10u;
            else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A') + 10u;
            else return -1;
            v = (v << 4) | d;
        }
        out[i] = (unsigned char)v;
    }
    return 0;
}

#endif /* BTMHOST_H */
