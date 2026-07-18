/*
Bitmark (BTM) v2 typed chain-record format, shared by the miner
(train_gpt2_btm2.c) and the verifier (eval_gpt2_btm2_f32.c).
CONSENSUS-CRITICAL: both sides must agree bit-for-bit on everything here.

Record layout (all integers little-endian, explicit widths -- wasm32 size_t is
32-bit, never read format fields through size_t). No magic/version bytes and no
redundant length fields: how records are framed inside a Bitmark block is the
protocol's concern; record sizes are derived by parsing entry headers.

  <32B head>         on disk: this record's own chained hash (head-replacement
                     lifecycle, same as v1); in the hash preimage: the previous
                     record's head (or the anchor block hash / 0xff*32 genesis)
  <u16 n_entries>
  <entries...>

Entry: 4-byte header, then a payload whose layout is implied by delta_type:
  <u16 tensor_id>    (layer << 8) | tensor_type   (llm.c tensor type 0..15)
  <u8  delta_type>   see below
  <u8  rank>         r for LoRA entries, 0 otherwise

  delta_type 1, LoRA int8 (ADDITIVE; the four matrix kinds 4/6/10/12 only):
      <u16 scaleA fp16 bits><u16 scaleB fp16 bits><int8 A[out*r]><int8 B[r*in]>
      Expansion: W[oc][ic] += (f32)(int32 dot of A row oc with B col ic)
                              * (half2float(sA) * half2float(sB))
      Integer accumulation is exact and order-independent; exactly ONE float
      multiply-add per weight touched.
  delta_type 0, sparse list (OVERWRITE; any tensor, v1-style but slice-relative):
      <u32 count><count x (u32 idx, f32 val)>
      idx is relative to the (tensor_type, layer) slice.

Canonical entry order emitted by the miner: layer 0..L-1, within each layer
tensor types {4 (qkvw), 6 (attprojw), 10 (fcw), 12 (fcprojw)}.

The base model (before any records) is a full GPT-2 random init with the fixed
seed BTM2_BASE_SEED -- deterministic given llmc/rand.h; part of consensus.
*/
#ifndef BTMRECORD_H
#define BTMRECORD_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rand.h"

#define BTM2_DELTA_SPARSE 0
#define BTM2_DELTA_LORA_INT8 1
#define BTM2_RANK 4
#define BTM2_BASE_SEED 20260718u
#define BTM2_HEADER_BYTES 34   /* 32B head + u16 n_entries */
#define BTM2_LORA_TYPES_N 4
static const int btm2_lora_types[BTM2_LORA_TYPES_N] = {4, 6, 10, 12};

// ----------------------------------------------------------------------------
// fp16 (IEEE binary16) <-> f32, portable bit manipulation, round-to-nearest-even

static inline uint16_t btm2_float_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7fffffu;
    uint32_t exp8 = (x >> 23) & 0xffu;
    if (exp8 == 0xffu) { // inf / nan
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    int32_t e = (int32_t)exp8 - 112; // rebias 127 -> 15
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u); // overflow -> inf
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign; // underflow -> signed zero
        mant |= 0x800000u; // make implicit bit explicit, build subnormal
        uint32_t shift = (uint32_t)(14 - e); // 14..24
        uint32_t half = 1u << (shift - 1);
        uint32_t r = (mant + half - 1u + ((mant >> shift) & 1u)) >> shift; // RNE
        return (uint16_t)(sign | r);
    }
    uint32_t r = mant + 0xfffu + ((mant >> 13) & 1u); // RNE on 23->10 bits
    if (r & 0x800000u) { // mantissa overflow carries into exponent
        r = 0;
        if (++e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (r >> 13));
}

static inline float btm2_half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp5 = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t man = (uint32_t)h & 0x3ffu;
    uint32_t x;
    if (exp5 == 0) {
        if (man == 0) {
            x = sign;
        } else { // subnormal: renormalize
            int e = 0;
            while (!(man & 0x400u)) { man <<= 1; e++; }
            x = sign | ((uint32_t)(113 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp5 == 0x1fu) {
        x = sign | 0x7f800000u | (man << 13);
    } else {
        x = sign | ((exp5 + 112u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &x, 4);
    return f;
}

// ----------------------------------------------------------------------------
// tensor geometry (mirrors llm.c fill_in_parameter_sizes ordering)

// cumulative start offset of each of the 16 parameter tensors in params_memory
static inline void btm2_tensor_offsets(const size_t* param_sizes, size_t out_offsets[16]) {
    size_t off = 0;
    for (int i = 0; i < 16; i++) { out_offsets[i] = off; off += param_sizes[i]; }
}

// per-(type,layer) slice element count; layered tensors are types 2..13
static inline size_t btm2_slice_numel(int type, int L, const size_t* param_sizes) {
    if (type >= 2 && type <= 13) return param_sizes[type] / (size_t)L;
    return param_sizes[type];
}

// (out, in) dims of the four LoRA-target matrix types
static inline void btm2_lora_dims(int type, int C, int* out, int* in) {
    if      (type == 4)  { *out = 3*C; *in = C;   } // qkvw
    else if (type == 6)  { *out = C;   *in = C;   } // attprojw
    else if (type == 10) { *out = 4*C; *in = C;   } // fcw
    else                 { *out = C;   *in = 4*C; } // fcprojw (type 12)
}

// start of matrix (type, layer) within params_memory
static inline size_t btm2_matrix_offset(int type, int layer, int C, const size_t* tensor_offsets) {
    int out, in;
    btm2_lora_dims(type, C, &out, &in);
    return tensor_offsets[type] + (size_t)layer * (size_t)out * (size_t)in;
}

// ----------------------------------------------------------------------------
// int8 symmetric quantization with fp16 scale

// q = round(x/scale) clamped to [-127,127], scale = max|x|/127 stored as fp16.
// Dequantization uses the fp16-ROUNDED scale, so quantization divides by that
// same rounded value -- keeps the committed error minimal and deterministic.
static inline void btm2_quantize(const float* x, size_t n, int8_t* q, uint16_t* scale_bits) {
    float maxabs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > maxabs) maxabs = a;
    }
    if (maxabs == 0.0f) {
        *scale_bits = 0;
        memset(q, 0, n);
        return;
    }
    uint16_t sb = btm2_float_to_half(maxabs / 127.0f);
    float s = btm2_half_to_float(sb);
    if (s <= 0.0f) { // fp16 underflow of a tiny scale: commit zeros
        *scale_bits = 0;
        memset(q, 0, n);
        return;
    }
    *scale_bits = sb;
    for (size_t i = 0; i < n; i++) {
        float v = x[i] / s;
        int32_t iv = (int32_t)lroundf(v);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        q[i] = (int8_t)iv;
    }
}

// ----------------------------------------------------------------------------
// deterministic LoRA expansion: W += dequant(A) . dequant(B)
// int32 accumulation (exact), then a single f32 multiply-add per element.

static inline void btm2_apply_lora_int8(float* W, int out, int in, int r,
                                        const int8_t* A, const int8_t* B,
                                        uint16_t sA_bits, uint16_t sB_bits) {
    if (sA_bits == 0 || sB_bits == 0) return; // zero delta
    float s = btm2_half_to_float(sA_bits) * btm2_half_to_float(sB_bits);
    for (int oc = 0; oc < out; oc++) {
        const int8_t* Arow = A + (size_t)oc * r;
        float* Wrow = W + (size_t)oc * in;
        for (int ic = 0; ic < in; ic++) {
            int32_t acc = 0;
            for (int j = 0; j < r; j++) {
                acc += (int32_t)Arow[j] * (int32_t)B[(size_t)j * in + ic];
            }
            Wrow[ic] += (float)acc * s;
        }
    }
}

// ----------------------------------------------------------------------------
// record walking / application

// entry size (header + payload) from its 4-byte header; 0 if malformed/unknown
static inline size_t btm2_entry_size(const unsigned char* e, size_t max_bytes, int C) {
    if (max_bytes < 4) return 0;
    int type = e[0]; // low byte of tensor_id
    int dtype = e[2];
    int r = e[3];
    if (dtype == BTM2_DELTA_LORA_INT8) {
        int out, in;
        if (!(type == 4 || type == 6 || type == 10 || type == 12)) return 0;
        if (r <= 0) return 0;
        btm2_lora_dims(type, C, &out, &in);
        return 4 + 4 + (size_t)out * r + (size_t)r * in;
    }
    if (dtype == BTM2_DELTA_SPARSE) {
        if (max_bytes < 8) return 0;
        uint32_t count;
        memcpy(&count, e + 4, 4);
        return 8 + (size_t)count * 8;
    }
    return 0;
}

// total record size (34-byte header + parsed entries); 0 if malformed/truncated
static inline size_t btm2_record_size(const unsigned char* rec, size_t max_bytes, int C) {
    if (max_bytes < BTM2_HEADER_BYTES) return 0;
    uint16_t n_entries;
    memcpy(&n_entries, rec + 32, 2);
    size_t off = BTM2_HEADER_BYTES;
    for (uint16_t i = 0; i < n_entries; i++) {
        size_t esz = btm2_entry_size(rec + off, max_bytes - off, C);
        if (esz == 0 || off + esz > max_bytes) return 0;
        off += esz;
    }
    return off;
}

// apply one record's entries to params_memory; returns 0 ok, -1 malformed
static inline int btm2_apply_record(float* params, const unsigned char* rec, size_t rec_bytes,
                                    int L, int C, const size_t* param_sizes, const size_t* tensor_offsets) {
    uint16_t n_entries;
    memcpy(&n_entries, rec + 32, 2);
    const unsigned char* e = rec + BTM2_HEADER_BYTES;
    size_t left = rec_bytes - BTM2_HEADER_BYTES;
    for (uint16_t i = 0; i < n_entries; i++) {
        size_t esz = btm2_entry_size(e, left, C);
        if (esz == 0 || esz > left) return -1;
        int type = e[0];
        int layer = e[1];
        int dtype = e[2];
        int r = e[3];
        if (dtype == BTM2_DELTA_LORA_INT8) {
            int out, in;
            btm2_lora_dims(type, C, &out, &in);
            uint16_t sA, sB;
            memcpy(&sA, e + 4, 2);
            memcpy(&sB, e + 6, 2);
            const int8_t* A = (const int8_t*)(e + 8);
            const int8_t* B = A + (size_t)out * r;
            btm2_apply_lora_int8(params + btm2_matrix_offset(type, layer, C, tensor_offsets),
                                 out, in, r, A, B, sA, sB);
        } else { // sparse overwrite
            uint32_t count;
            memcpy(&count, e + 4, 4);
            size_t slice = btm2_slice_numel(type, L, param_sizes);
            size_t base = tensor_offsets[type] + (size_t)layer * slice;
            for (uint32_t j = 0; j < count; j++) {
                uint32_t idx; float val;
                memcpy(&idx, e + 8 + (size_t)j * 8, 4);
                memcpy(&val, e + 8 + (size_t)j * 8 + 4, 4);
                if (idx < slice) params[base + idx] = val;
            }
        }
        e += esz;
        left -= esz;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// deterministic full base init (CONSENSUS): GPT-2 init, all weights, fixed seed.
// Mirrors the v1 gpt2_build_from_random init loop with every weight active.
// params must be zeroed by the caller before this call.

static inline void btm2_init_base_params(float* params, const size_t* param_sizes,
                                         int num_layers, int channels, int vocab_size,
                                         unsigned int seed) {
    size_t L = (size_t)num_layers;
    float residual_scale = 1.0f / sqrtf(2.0f * (float)num_layers);
    mt19937_state init_rng;
    manual_seed(&init_rng, seed);
    size_t offset;
    for (size_t l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < 16; i++) {
            // layernorm scale tensors to 1.0 (once, at l == 0)
            if (l == 0 && (i == 2 || i == 8 || i == 14)) {
                for (size_t j = 0; j < param_sizes[i]; j++) params[offset + j] = 1.0f;
            }
            if ((l == 0 && (i == 0 || i == 1)) || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = param_sizes[i];
                size_t layer_offset = 0;
                if (i == 0) n = (size_t)vocab_size * (size_t)channels; // init V rows, not Vp
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    n = n / L;
                    layer_offset = l * n;
                }
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                float* buf = (float*)malloc(n * sizeof(float));
                normal_(buf, (unsigned int)n, 0.0f, scale, &init_rng);
                memcpy(params + offset + layer_offset, buf, n * sizeof(float));
                free(buf);
            }
            offset += param_sizes[i];
        }
    }
}

#endif // BTMRECORD_H
