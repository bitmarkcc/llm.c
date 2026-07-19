/*
Node-side weight-state cache. NOT CONSENSUS: this is a memoization of what the
canonical model build computes -- a suspicious node can always rebuild from
genesis and must obtain byte-identical state. Used by the trunk miner and
verifier to skip the ~20 s fixed-seed init + full record replay when the chain
has only advanced by a few records since the cached state.

File layout: <32B tip><raw f32 params_memory>
  tip = head (chained hash) of the LAST record folded into the state,
        or 32 x 0xEE (the genesis sentinel) when the state is the bare init.
A size mismatch (different E/H/config) simply misses and triggers a rebuild.
*/
#ifndef BTMCACHE_H
#define BTMCACHE_H

#include <stdio.h>
#include <string.h>

static inline void btmcache_genesis_tip(unsigned char tip[32]) { memset(tip, 0xEE, 32); }

// chunked I/O: wasi (and stdio generally) may return short counts on
// multi-GB reads/writes; loop in bounded chunks and treat 0 progress as failure
#define BTMCACHE_CHUNK (64u * 1024u * 1024u)

static inline int btmcache_read_all(FILE* f, unsigned char* dst, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t want = n - done;
        if (want > BTMCACHE_CHUNK) want = BTMCACHE_CHUNK;
        size_t got = fread(dst + done, 1, want, f);
        if (got == 0) return -1;
        done += got;
    }
    return 0;
}

static inline int btmcache_write_all(FILE* f, const unsigned char* src, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t want = n - done;
        if (want > BTMCACHE_CHUNK) want = BTMCACHE_CHUNK;
        size_t put = fwrite(src + done, 1, want, f);
        if (put == 0) return -1;
        done += put;
    }
    return 0;
}

// load tip+params if the file exists and has exactly the expected size; 0 on success
static inline int btmcache_load(const char* path, unsigned char tip_out[32],
                                float* params, size_t n_params) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0L, SEEK_END);
    long sz = ftell(f);
    if (sz < 0 || (size_t)sz != 32 + n_params * sizeof(float)) { fclose(f); return -1; }
    fseek(f, 0L, SEEK_SET);
    if (fread(tip_out, 1, 32, f) != 32) { fclose(f); return -1; }
    int rc = btmcache_read_all(f, (unsigned char*)params, n_params * sizeof(float));
    fclose(f);
    return rc;
}

// write to a temp file, then rename over the target: a failed save must never
// destroy an existing valid cache
static inline int btmcache_save(const char* path, const unsigned char tip[32],
                                const float* params, size_t n_params) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(tip, 1, 32, f) == 32
          && btmcache_write_all(f, (const unsigned char*)params, n_params * sizeof(float)) == 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

#endif // BTMCACHE_H
