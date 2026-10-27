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

UNROLLED 4x (16 elements per iteration), and this is BIT-PRESERVING rather than a
numerics change: each of the four accumulators still receives its own products in the
same order, so every IEEE add happens in the sequence it did before. Only the loop
bookkeeping changes.

It is unrolled because of GAS, not speed. The node charges per opcode CLASS, and a
measurement under its own instrumentation (contrib/dynamicalgo/gasinstrument) showed the
arithmetic comfortably inside budget -- SIMD 96%, SIMD_MEM 89% for two forwards -- while
the bookkeeping AROUND it was over: VAR (local.get/set) 131%, INT (index arithmetic)
111%, BRANCH (loop back-edges) 103%. At 4 elements per iteration the loop spent roughly
8 VAR ops to buy 2 SIMD ops. Both trunk dimensions are multiples of 16 (C=768, H=384),
so the 4-element and serial cleanups below do not execute at all in practice; they exist
for correctness on any n.
*/
#ifndef BTMKERNEL_H
#define BTMKERNEL_H

#include <stddef.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
//! always_inline is belt-and-braces, NOT a fix for anything measured. I expected the 4x
//! unroll to have pushed the compiler into emitting a real call per dot -- 13.7M across
//! two forwards, which would have matched the measured CALL of 14.08M -- and expected
//! forcing the inline to collapse scoring's MEM from 227% of its cap. It did neither: the
//! module changed by 16 bytes and every gas class came back byte-identical, so the
//! compiler was already inlining. The 14.08M calls are the transcendentals (expf in
//! softmax, tanhf in gelu, and their internal helpers), which at 4.4% of the CALL cap are
//! not a scoring problem. Kept because pinning the inline costs nothing and removes a
//! compiler judgement call from a consensus path; scoring's MEM remains unexplained.
__attribute__((always_inline)) static inline float btmk_dot(const float* w, const float* x, int n) {
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(w + i),      wasm_v128_load(x + i)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(w + i + 4),  wasm_v128_load(x + i + 4)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(w + i + 8),  wasm_v128_load(x + i + 8)));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(w + i + 12), wasm_v128_load(x + i + 12)));
    }
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
__attribute__((always_inline)) static inline float btmk_dot(const float* w, const float* x, int n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int i = 0;
    // Unrolled to match the SIMD path element-for-element: each accumulator takes its
    // products in the same order, so the two paths stay bit-identical to each other and
    // to the pre-unroll kernel.
    for (; i + 16 <= n; i += 16) {
        a0 += w[i]    * x[i];     a1 += w[i+1]  * x[i+1];
        a2 += w[i+2]  * x[i+2];   a3 += w[i+3]  * x[i+3];
        a0 += w[i+4]  * x[i+4];   a1 += w[i+5]  * x[i+5];
        a2 += w[i+6]  * x[i+6];   a3 += w[i+7]  * x[i+7];
        a0 += w[i+8]  * x[i+8];   a1 += w[i+9]  * x[i+9];
        a2 += w[i+10] * x[i+10];  a3 += w[i+11] * x[i+11];
        a0 += w[i+12] * x[i+12];  a1 += w[i+13] * x[i+13];
        a2 += w[i+14] * x[i+14];  a3 += w[i+15] * x[i+15];
    }
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
