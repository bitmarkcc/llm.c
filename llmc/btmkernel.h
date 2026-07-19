/*
Canonical deterministic compute kernel for the btm trunk chain.
CONSENSUS-CRITICAL: miner and verifier share it; the wasm module pins it.

The dot product accumulates in FOUR partial sums over the input index
(lane j takes elements i+j) with the fixed final reduction (a0+a1)+(a2+a3),
then a serial tail for n % 4 leftovers. The wasm SIMD128 path computes the
same four lanes with per-lane IEEE ops, so it is bit-identical to the portable
path. This 4-accumulator order IS the canonical arithmetic of the trunk chain
(it deliberately differs from a plain serial dot; consensus pins the module,
so cross-version parity with the older chains is not required).

No FMA anywhere: plain x86-64 (SSE2) has no FMA instruction, wasm scalar/simd128
have no fusing, and we never enable -ffast-math or relaxed-simd.
*/
#ifndef BTMKERNEL_H
#define BTMKERNEL_H

#include <stddef.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
static inline float btmk_dot(const float* w, const float* x, int n) {
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(w + i), wasm_v128_load(x + i)));
    }
    float a0 = wasm_f32x4_extract_lane(acc, 0);
    float a1 = wasm_f32x4_extract_lane(acc, 1);
    float a2 = wasm_f32x4_extract_lane(acc, 2);
    float a3 = wasm_f32x4_extract_lane(acc, 3);
    float s = (a0 + a1) + (a2 + a3);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}
#else
static inline float btmk_dot(const float* w, const float* x, int n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += w[i]   * x[i];
        a1 += w[i+1] * x[i+1];
        a2 += w[i+2] * x[i+2];
        a3 += w[i+3] * x[i+3];
    }
    float s = (a0 + a1) + (a2 + a3);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}
#endif

// matmul with the canonical dot; layout as llm.c matmul_forward (weight (OC, Cin))
static inline void matmul_forward_k(float* out, const float* inp, const float* weight, const float* bias,
                                    int B, int T, int Cin, int OC) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + ((size_t)b * T + t) * Cin;
            float* o = out + ((size_t)b * T + t) * OC;
            for (int oc = 0; oc < OC; oc++) {
                float v = bias ? bias[oc] : 0.0f;
                o[oc] = v + btmk_dot(weight + (size_t)oc * Cin, x, Cin);
            }
        }
    }
}

#endif // BTMKERNEL_H
