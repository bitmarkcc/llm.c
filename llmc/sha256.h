/*
Self-contained SHA-256 (FIPS 180-4), replacing the OpenSSL dependency for builds
where libcrypto is unavailable (notably wasm32-wasi). Pure portable C99, no
endianness assumptions. Verified against the standard test vectors ("abc", "").
*/
#ifndef LLMC_SHA256_H
#define LLMC_SHA256_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint32_t sha256_K_[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

#define SHA256_ROTR_(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block_(uint32_t h[8], const unsigned char* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16)
             | ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA256_ROTR_(w[i-15], 7) ^ SHA256_ROTR_(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = SHA256_ROTR_(w[i-2], 17) ^ SHA256_ROTR_(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SHA256_ROTR_(e, 6) ^ SHA256_ROTR_(e, 11) ^ SHA256_ROTR_(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + sha256_K_[i] + w[i];
        uint32_t S0 = SHA256_ROTR_(a, 2) ^ SHA256_ROTR_(a, 13) ^ SHA256_ROTR_(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// one-shot SHA-256 of (data, len) into md[32]; returns md (same signature shape as OpenSSL's SHA256)
static unsigned char* sha256_hash(const unsigned char* data, size_t len, unsigned char* md) {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    size_t off = 0;
    while (len - off >= 64) {
        sha256_block_(h, data + off);
        off += 64;
    }
    // final block(s): remaining bytes + 0x80 pad + zeros + 64-bit big-endian bit length
    unsigned char tail[128];
    size_t rem = len - off;
    memcpy(tail, data + off, rem);
    tail[rem] = 0x80;
    size_t tail_len = (rem + 1 + 8 <= 64) ? 64 : 128;
    memset(tail + rem + 1, 0, tail_len - rem - 1 - 8);
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        tail[tail_len - 1 - i] = (unsigned char)(bitlen >> (8 * i));
    }
    sha256_block_(h, tail);
    if (tail_len == 128) sha256_block_(h, tail + 64);
    for (int i = 0; i < 8; i++) {
        md[4*i]   = (unsigned char)(h[i] >> 24);
        md[4*i+1] = (unsigned char)(h[i] >> 16);
        md[4*i+2] = (unsigned char)(h[i] >> 8);
        md[4*i+3] = (unsigned char)(h[i]);
    }
    return md;
}

#endif // LLMC_SHA256_H
