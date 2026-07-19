/*
MoE extension of the Bitmark v2 chain-record format (llmc/btmrecord.h).
Shared by train_gpt2_btm_moe.c and eval_gpt2_btm_moe_f32.c.
CONSENSUS-CRITICAL: both sides must agree bit-for-bit on everything here.

Model: GPT-2 skeleton with the FFN replaced by E expert FFNs + a router per
layer. Per token: router logits over E experts; select top-k by logit with the
tie-break "strictly greater wins, ascending index scan" (equal logits keep the
LOWER expert index); softmax over the k SELECTED logits (renormalized,
Mixtral-style); output = sum of gate_s * expert_s(x).

Parameter layout: 18 tensors. 0..15 as in llm.c, except the FFN tensors gain an
expert dimension (layer-major, then expert):
  10 fcw     [L][E][4C][C]      11 fcb     [L][E][4C]
  12 fcprojw [L][E][C][4C]      13 fcprojb [L][E][C]
appended:
  16 routerw [L][E][C]          17 routerb [L][E]

Upcycling (dense -> MoE, deterministic, exactly function-preserving at k>=2):
every expert of layer l is a verbatim copy of the dense layer-l FFN; router
weights and biases are all zero. With all logits equal, the tie-break selects
experts 0..k-1 and the renormalized gates are 1/k each; identical experts then
sum to exactly the dense FFN output.

Record entry types added to btmrecord.h's 0 (sparse overwrite) and 1 (LoRA):
  delta_type 2: LoRA int8 on ONE EXPERT's matrix.
      tensor_id = (layer<<8)|type, type in {10 (expert fcw), 12 (expert fcprojw)}
      payload: <u16 expert_id><u16 reserved=0><u16 scaleA fp16><u16 scaleB fp16>
               <int8 A[out*r]><int8 B[r*in]>
      (expert_id as u16 scales to the trunk's hundreds of experts per layer)
  delta_type 3: dense f32 slice OVERWRITE (used for router slices; rank=0).
      tensor_id types 16 (routerw layer slice, numel=E*C) / 17 (routerb, numel=E)
      payload: <u32 numel><f32 values[numel]>
*/
#ifndef BTMMOE_H
#define BTMMOE_H

#include "btmrecord.h"

#define BTM2_DELTA_LORA_EXPERT 2
#define BTM2_DELTA_DENSE 3
#define BTMMOE_NUM_TENSORS 18

// fill the 18 MoE parameter tensor sizes (Vp = padded vocab)
static inline void btmmoe_fill_param_sizes(size_t* param_sizes, int Vp, int maxT, int L, int C, int E) {
    param_sizes[0]  = (size_t)Vp * C;              // wte
    param_sizes[1]  = (size_t)maxT * C;            // wpe
    param_sizes[2]  = (size_t)L * C;               // ln1w
    param_sizes[3]  = (size_t)L * C;               // ln1b
    param_sizes[4]  = (size_t)L * 3*C * C;         // qkvw
    param_sizes[5]  = (size_t)L * 3*C;             // qkvb
    param_sizes[6]  = (size_t)L * C * C;           // attprojw
    param_sizes[7]  = (size_t)L * C;               // attprojb
    param_sizes[8]  = (size_t)L * C;               // ln2w
    param_sizes[9]  = (size_t)L * C;               // ln2b
    param_sizes[10] = (size_t)L * E * 4*C * C;     // expert fcw
    param_sizes[11] = (size_t)L * E * 4*C;         // expert fcb
    param_sizes[12] = (size_t)L * E * C * 4*C;     // expert fcprojw
    param_sizes[13] = (size_t)L * E * C;           // expert fcprojb
    param_sizes[14] = (size_t)C;                   // lnfw
    param_sizes[15] = (size_t)C;                   // lnfb
    param_sizes[16] = (size_t)L * E * C;           // routerw
    param_sizes[17] = (size_t)L * E;               // routerb
}

static inline void btmmoe_tensor_offsets(const size_t* param_sizes, size_t out_offsets[BTMMOE_NUM_TENSORS]) {
    size_t off = 0;
    for (int i = 0; i < BTMMOE_NUM_TENSORS; i++) { out_offsets[i] = off; off += param_sizes[i]; }
}

// start of expert e's matrix (type 10 or 12) of layer l
static inline size_t btmmoe_expert_matrix_offset(int type, int layer, int expert, int C, int E,
                                                 const size_t* tensor_offsets) {
    size_t mat = (size_t)4*C * C; // both expert matrix kinds have 4C*C elements
    return tensor_offsets[type] + ((size_t)layer * E + expert) * mat;
}

// (out, in) dims of an expert matrix kind
static inline void btmmoe_expert_dims(int type, int C, int* out, int* in) {
    if (type == 10) { *out = 4*C; *in = C;   } // expert fcw
    else            { *out = C;   *in = 4*C; } // expert fcprojw (type 12)
}

// ----------------------------------------------------------------------------
// deterministic upcycle: dense 16-tensor params -> MoE 18-tensor params.
// dense_sizes/dense_offsets describe the depth-matched DENSE model.
// Exactly function-preserving under k>=2 renormalized top-k gating (see banner).
static inline void btmmoe_upcycle(float* moe_params, const size_t* moe_sizes, const size_t* moe_offsets,
                                  const float* dense_params, const size_t* dense_sizes, const size_t* dense_offsets,
                                  int L, int C, int E) {
    // tensors 0..9, 14, 15: verbatim copies
    for (int t = 0; t < 16; t++) {
        if (t >= 10 && t <= 13) continue;
        memcpy(moe_params + moe_offsets[t], dense_params + dense_offsets[t],
               dense_sizes[t] * sizeof(float));
    }
    // FFN tensors 10..13: every expert = copy of the dense layer FFN
    for (int t = 10; t <= 13; t++) {
        size_t slice = dense_sizes[t] / (size_t)L; // dense per-layer slice
        for (int l = 0; l < L; l++) {
            const float* src = dense_params + dense_offsets[t] + (size_t)l * slice;
            for (int e = 0; e < E; e++) {
                memcpy(moe_params + moe_offsets[t] + ((size_t)l * E + e) * slice,
                       src, slice * sizeof(float));
            }
        }
    }
    // router: zeros (moe_params assumed zero-initialized by the caller)
    (void)moe_sizes; (void)C;
}

// ----------------------------------------------------------------------------
// consensus top-k selection: scan experts in ascending index order, strictly
// greater logit wins => equal logits keep the lower index. Returns indices in
// ASCENDING expert order in sel[], gates renormalized by softmax over selected.
static inline void btmmoe_topk_gates(const float* logits, int E, int k, int* sel, float* gates) {
    // selection: repeatedly take the max not yet selected (stable by index)
    for (int s = 0; s < k; s++) {
        int best = -1;
        float bestv = 0.0f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int t = 0; t < s; t++) if (sel[t] == e) { taken = 1; break; }
            if (taken) continue;
            if (best == -1 || logits[e] > bestv) { best = e; bestv = logits[e]; }
        }
        sel[s] = best;
    }
    // sort selected ascending (deterministic canonical order; k is tiny)
    for (int a = 0; a < k; a++) {
        for (int b = a + 1; b < k; b++) {
            if (sel[b] < sel[a]) { int tmp = sel[a]; sel[a] = sel[b]; sel[b] = tmp; }
        }
    }
    // softmax over the selected logits (max-subtracted, in ascending sel order)
    float maxl = logits[sel[0]];
    for (int s = 1; s < k; s++) if (logits[sel[s]] > maxl) maxl = logits[sel[s]];
    float sum = 0.0f;
    for (int s = 0; s < k; s++) {
        gates[s] = expf(logits[sel[s]] - maxl);
        sum += gates[s];
    }
    for (int s = 0; s < k; s++) gates[s] /= sum;
}

// ----------------------------------------------------------------------------
// MoE-aware record walking/application (handles delta types 0..3)

static inline size_t btmmoe_entry_size(const unsigned char* e, size_t max_bytes, int C) {
    if (max_bytes < 4) return 0;
    int type = e[0];
    int dtype = e[2];
    int r = e[3];
    if (dtype == BTM2_DELTA_LORA_INT8) {
        int out, in;
        if (!(type == 4 || type == 6)) return 0; // dense LoRA: attention matrices only in MoE models
        btm2_lora_dims(type, C, &out, &in);
        return 4 + 4 + (size_t)out * r + (size_t)r * in;
    }
    if (dtype == BTM2_DELTA_LORA_EXPERT) {
        int out, in;
        if (!(type == 10 || type == 12)) return 0;
        if (r <= 0) return 0;
        btmmoe_expert_dims(type, C, &out, &in);
        return 4 + 8 + (size_t)out * r + (size_t)r * in; // + expert_id/reserved/scales
    }
    if (dtype == BTM2_DELTA_DENSE) {
        if (max_bytes < 8) return 0;
        uint32_t numel;
        memcpy(&numel, e + 4, 4);
        return 8 + (size_t)numel * 4;
    }
    if (dtype == BTM2_DELTA_SPARSE) {
        if (max_bytes < 8) return 0;
        uint32_t count;
        memcpy(&count, e + 4, 4);
        return 8 + (size_t)count * 8;
    }
    return 0;
}

static inline size_t btmmoe_record_size(const unsigned char* rec, size_t max_bytes, int C) {
    if (max_bytes < BTM2_HEADER_BYTES) return 0;
    uint16_t n_entries;
    memcpy(&n_entries, rec + 32, 2);
    size_t off = BTM2_HEADER_BYTES;
    for (uint16_t i = 0; i < n_entries; i++) {
        size_t esz = btmmoe_entry_size(rec + off, max_bytes - off, C);
        if (esz == 0 || off + esz > max_bytes) return 0;
        off += esz;
    }
    return off;
}

static inline int btmmoe_apply_record(float* params, const unsigned char* rec, size_t rec_bytes,
                                      int L, int C, int E,
                                      const size_t* param_sizes, const size_t* tensor_offsets) {
    uint16_t n_entries;
    memcpy(&n_entries, rec + 32, 2);
    const unsigned char* e = rec + BTM2_HEADER_BYTES;
    size_t left = rec_bytes - BTM2_HEADER_BYTES;
    for (uint16_t i = 0; i < n_entries; i++) {
        size_t esz = btmmoe_entry_size(e, left, C);
        if (esz == 0 || esz > left) return -1;
        int type = e[0];
        int layer = e[1];
        int dtype = e[2];
        int r = e[3];
        if (dtype == BTM2_DELTA_LORA_INT8) { // dense attention matrices
            int out, in;
            btm2_lora_dims(type, C, &out, &in);
            uint16_t sA, sB;
            memcpy(&sA, e + 4, 2);
            memcpy(&sB, e + 6, 2);
            const int8_t* A = (const int8_t*)(e + 8);
            const int8_t* B = A + (size_t)out * r;
            btm2_apply_lora_int8(params + btm2_matrix_offset(type, layer, C, tensor_offsets),
                                 out, in, r, A, B, sA, sB);
        } else if (dtype == BTM2_DELTA_LORA_EXPERT) {
            int out, in;
            btmmoe_expert_dims(type, C, &out, &in);
            uint16_t expert_id, sA, sB;
            memcpy(&expert_id, e + 4, 2);
            memcpy(&sA, e + 8, 2);
            memcpy(&sB, e + 10, 2);
            if ((int)expert_id >= E) return -1;
            const int8_t* A = (const int8_t*)(e + 12);
            const int8_t* B = A + (size_t)out * r;
            btm2_apply_lora_int8(params + btmmoe_expert_matrix_offset(type, layer, expert_id, C, E, tensor_offsets),
                                 out, in, r, A, B, sA, sB);
        } else if (dtype == BTM2_DELTA_DENSE) {
            uint32_t numel;
            memcpy(&numel, e + 4, 4);
            size_t slice = param_sizes[type] / (size_t)L; // types 16/17 are layered
            if (type < 2 || type == 14 || type == 15) slice = param_sizes[type];
            if ((size_t)numel != slice) return -1;
            memcpy(params + tensor_offsets[type] + (size_t)layer * slice, e + 8, (size_t)numel * 4);
        } else { // sparse overwrite
            uint32_t count;
            memcpy(&count, e + 4, 4);
            size_t slice = param_sizes[type] / (size_t)L;
            if (type < 2 || type == 14 || type == 15) slice = param_sizes[type];
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

#endif // BTMMOE_H
