/*
Bitmark v2 miner: LoRA structured-delta proof-of-useful-work (see
doc/btm-proof-of-useful-work.md and llmc/btmrecord.h for the record format).

Per mining attempt:
- build the base model: fixed-seed full GPT-2 init (BTM2_BASE_SEED) + apply all
  chain records from btm-cp2.bin
- train rank-r LoRA factors (A ~ N(0,0.02) from the attempt seed, B = 0) on the
  4*L weight matrices, base frozen: 30 AdamW steps on the factors only
- commit: quantize factors to int8 + fp16 scales, materialize the QUANTIZED
  delta (so the scored model is exactly what a verifier reconstructs), build the
  typed record, double-SHA256 it, use the hash to select the val batch
  (commit-then-evaluate), score, and hand back the record with head <- hash
- best of n_attempts is appended to btm-cp2.bin

The model/layer code below (through gpt2_free) is shared verbatim with the v1
miner; the v2 logic starts at "v2 LoRA miner" near the end of the file.
*/

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#ifndef BTM_ONCHAIN
// Host-only: an on-chain module imports nothing but chain.*, and every one of these
// reaches for WASI -- stdio for fd_write, time for clock_time_get, stdlib for proc_exit.
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#endif
// defines: sha256_hash (self-contained SHA-256; no OpenSSL)
// defines: BTMLOG / BTM_ABORT / btm_alloc -- the seam between the native
// harness and an on-chain module, which may import nothing but chain.*
#include "llmc/btmhost.h"
#include "llmc/sha256.h"
// defines: the v2 typed record format, fp16, int8 quant, LoRA expansion, base init
#include "llmc/btmrecord.h"
// defines: MoE geometry, top-k gating, expert-aware records, trunk extensions
#include "llmc/btmmoe.h"
// defines: btmk_dot / matmul_forward_k -- the canonical 4-accumulator kernel
#include "llmc/btmkernel.h"
#ifndef BTM_ONCHAIN
// defines: node-side weight-state cache (NOT consensus; needs stdio)
#include "llmc/btmcache.h"
#endif

// defines: corpus-commitment VERIFICATION (freestanding half) + the committed root
#include "llmc/btmcorpus.h"
#include "llmc/btmcorpus_params.h"

// trunk configuration (consensus parameters of this chain)
//
// E=32 => 269,428,992 total params = 1.004 GiB in f32. An expert is 2 * C*H * L =
// 2 * 768*384 * 12 = 7,077,888 params, so halving E from 64 frees 226M params.
//
// NOT chosen to fit the verifier's 1.984 GiB linear-memory cap -- E=64 (1.849 GiB) fit
// that. Chosen for the CHECKPOINT BUDGET.
//
// verify() must stream the weight checkpoint AND fold every delta published since it,
// inside GAS_IO_BYTES_LIMIT (2 GiB) and GAS_IO_CALLS_LIMIT (4096). A smaller model leaves
// room for more deltas per epoch, and at 16 min per slot block (NUM_ALGOS=8 x 2 min) that
// is the length of the mining period between checkpoints:
//
//     E=64:  1.8485 GiB checkpoint -> 272 deltas fit ->  3.0 days of mining per epoch
//     E=32:  1.0021 GiB checkpoint -> 1508 deltas fit -> 16.8 days of mining per epoch
//
// E=64 also made the checkpoint unpublishable in practice: 1.85 GiB takes ~2,036 slot
// blocks (22.6 days) to transmit against a 3.0-day staleness bound, so it could never be
// kept current, forcing dedicated publish phases and a ~12% useful-work duty cycle.
//
// A secondary gain: the model build streams half as much, ~19.6 s -> ~10 s measured.
//
// The freed linear memory (139 MiB -> 1006 MiB under the cap, against 52.7 MiB per
// scoring window) would also permit scoring a record on ~19 windows instead of 1, which
// would cut the gate's estimator noise as sqrt(W), sigma 0.067 -> ~0.015. That is NOT
// done, and the blocker is gas, not memory: the forward is compute-bound (~4.4 FLOP per
// weight byte), so W windows cost ~W times the instruction count, and the existing two
// forwards already sit near GAS_LIMIT[GC_SIMD]. See the note at the scoring site in
// eval_gpt2_btm_trunk_f32.c. E=32 keeps that option open for the day those limits are
// revisited; it does not take it.
//
// The cost is half the trunk's expert capacity, which is a real loss for something whose
// purpose is to BE a knowledge base. The settled 32:1 target (E~300, 2.2B, 8.8 GB) needs
// memory64 or lazy expert loading -- see doc. All machinery below is E-agnostic.
//
// CONSENSUS-CRITICAL and must match the other file: E sizes the router entry (out = E)
// and therefore the FRAMING of every record, and it sizes the base init and the expert
// rotation, so a miner and a verifier that disagree cannot even walk the same chain.
#ifndef MOE_E
#define MOE_E 32
#endif
//! Vocabulary. CONSENSUS: sizes wte and the lm_head, so it frames the record's
//! dense entries exactly as E frames the router's.
#define TRUNK_VOCAB 16384
#define MOE_K 4   // active experts per token
#define MOE_H 384 // expert hidden width (C/2): fine-grained experts
#include <float.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ifndef BTM_ONCHAIN
#include "llmc/utils.h"   /* fopenCheck/hex_to_uchar: the native harness only */
#endif
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#ifndef BTM_ONCHAIN
#include "llmc/tokenizer.h"   /* unused by the verifier */
#endif
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#ifndef BTM_ONCHAIN
#include "llmc/dataloader.h"   /* the scored window comes from the record, not a loader */
#endif

typedef unsigned char uchar;

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
#ifdef BTMV_MEASURE_NO_ATTENTION
    // MEASUREMENT ONLY: zero the output and return, to price attention's share of the
    // gas classes by difference. Produces wrong losses by construction.
    (void)preatt; (void)att; (void)NH;
    for (int i = 0; i < B * T * C; i++) out[i] = 0.0f;
    return;
#endif

    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model (MoE: 18 tensors, see llmc/btmmoe.h)
#define NUM_PARAMETER_TENSORS 18
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, E, 4*C, C) expert fc weights
    float* fcb; // (L, E, 4*C)
    float* fcprojw; // (L, E, C, 4*C) expert fcproj weights
    float* fcprojb; // (L, E, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
    float* routerw; // (L, E, C)
    float* routerb; // (L, E)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    btmmoe_fill_param_sizes_h(param_sizes, config.padded_vocab_size, config.max_seq_len,
                              config.num_layers, config.channels, MOE_E, MOE_H);
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)btm_alloc(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb, &params->routerw, &params->routerb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 27
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, K, 4*C) per selected expert
    float* fch_gelu; // (L, B, T, K, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
    float* router_logits; // (L, B, T, E)
    float* gates; // (L, B, T, K) renormalized softmax over the selected experts
    float* expert_sel; // (L, B, T, K) selected expert indices, float-encoded
    float* expert_out; // (L, B, T, K, C) pre-gate output of each selected expert
} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)btm_alloc(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses,
        &acts->router_logits, &acts->gates, &acts->expert_sel, &acts->expert_out
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    // the currently active weights/parameters
    uint32_t* active_weights;
    size_t n_active_weights;
    bool* params_memory_active;
} GPT2;

// ---- host-only: a model checkpoint file and the v1 random init. Neither is reachable
// from gpt2_eval_trunk (the trunk builds from the chain), and both need stdio.
#ifndef BTM_ONCHAIN
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    if (model_file == NULL) { BTMLOG("Error opening model file\n"); BTM_ABORT(); }
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { BTMLOG("Bad magic model file\n"); BTM_ABORT(); }
    if (model_header[1] != 3) {
        BTMLOG("Bad version in model file\n");
        BTMLOG("---> HINT: try to re-run `python train_gpt2.py`\n");
        BTM_ABORT();
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    BTMLOG("[GPT-2]\n");
    BTMLOG("max_seq_len: %zu\n", maxT);
    BTMLOG("vocab_size: %zu\n", V);
    BTMLOG("padded_vocab_size: %zu\n", Vp);
    BTMLOG("num_layers: %zu\n", L);
    BTMLOG("num_heads: %zu\n", NH);
    BTMLOG("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    BTMLOG("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

// Build the miner's starting model. Two independent seeds with distinct roles:
//   rng_seed_1 picks WHICH n_active_weights parameter indices this attempt will train;
//   rng_seed_2 draws their random initial values (GPT-2 style normal init).
// All other weights start at 0 (layernorm params at 1), then the chain records in cp are
// applied on top -- each attempt trains on top of the network accumulated so far.
// The seeds are the miner's free choice and are NOT consensus-relevant: the published
// record stores explicit (index,value) pairs, so verifiers never replay this init.
void gpt2_build_from_random(GPT2 *model, int depth, size_t n_active_weights, unsigned int rng_seed_1, unsigned int rng_seed_2, uchar* cp, size_t cp_bytes) {
    // init random (training from scratch)

    // parameterize the size of gpt2 based only on the depth of the model (num_layers)
    model->config.num_layers = depth;
    // follows GPT-2 sizes
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; } // gpt2-tiny (30M)
    else if (depth == 12) { channels = 768; num_heads = 12; } // gpt2 (124M)
    else if (depth == 24) { channels = 1024; num_heads = 16; } // gpt2-medium (350M)
    else if (depth == 36) { channels = 1280; num_heads = 20; } // gpt2-large (774M)
    else if (depth == 48) { channels = 1600; num_heads = 25; } // gpt2-xl (1558M)
    else { BTMLOG("Unsupported depth for now\n"); BTM_ABORT(); }
    model->config.channels = channels;
    model->config.num_heads = num_heads;
    model->config.max_seq_len = 1024;
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128

    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_sizes, model->config);
    model->num_parameters = 0;
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_sizes[i];
        num_parameters_bytes += model->param_sizes[i] * sizeof(float);
    }
    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);

    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss

    //size_t n_active_weights = 4000; // number of weights that will be adjusted (change to 4000)
    BTMLOG("n_active_weights = %lu\n",n_active_weights);
    mt19937_state init_rng_1;
    manual_seed(&init_rng_1, rng_seed_1);

    // get a random subset of weights
    uint32_t* active_weights = btm_alloc(n_active_weights*sizeof(uint32_t));
    size_t n_weights = model->num_parameters;
    BTMLOG("n_weights = %lu\n",n_weights);
    for (size_t i=0; i<n_active_weights; i++)
	active_weights[i] = n_weights; // dummy value
    bool* params_memory_active = btm_alloc(sizeof(bool)*n_weights);
    for (size_t i=0; i<n_weights; i++)
	params_memory_active[i] = false;
    int i_active = 0;
    BTMLOG("rng_seed_1 = %u\n",rng_seed_1);
    for (size_t i=0; i<n_active_weights; i++) {
	// get random number from 0 to n_weights-1, not chosen before
	while (true) {
	    uint32_t rand_weight = (uint32_t)(randfloat32(&init_rng_1)*n_weights);
	    if (!params_memory_active[rand_weight]) { // bitmap check, same result as scanning active_weights but O(1)
		active_weights[i] = rand_weight;
		params_memory_active[rand_weight] = true;
		if (i%10000==0) BTMLOG("active_weights[%lu] = %u\n",i,active_weights[i]);
		break;
	    }
	}
    }
    
    model->active_weights = active_weights;
    model->n_active_weights = n_active_weights;
    model->params_memory_active = params_memory_active;

    mt19937_state init_rng_2;
    manual_seed(&init_rng_2, rng_seed_2);
    // allocate and random init the memory for all the parameters with GPT-2 schema
    // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
    // NOTE: assuming all parameters are of the type floatX, could be relaxed later
    float* params_memory_cpu = (float*)btm_alloc(num_parameters_bytes);
    memset(params_memory_cpu, 0, num_parameters_bytes);
    // fill in all the weights with random values
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    // we have to init all these tensors exactly in the order that PyTorch initializes them
    // so that we can match them up and get correctness and exactly the same initial conditions
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (int l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            // the layernorm parameters are all initialized to 1
            if (l == 0 && (i == 2 || i == 8 || i == 14)) { // only at l = 0 to init these just once
                for (size_t j = 0; j < model->param_sizes[i]; j++) {
                    params_memory_cpu[offset + j] = 1.0f;
                }
            }
            // weights tensors are handled here
            if ((l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
              || i == 4 || i == 6 || i == 10 || i == 12) {
                int n = model->param_sizes[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    // for wte tensor (padded vocab) override to init V instead of Vp rows
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    // weight tensors, we are only initializing layer l
                    BTM_ASSERT(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                // in GPT-2, the projections back into the residual stream are additionally
                // scaled by 1/sqrt(2*L) for training stability
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                // okay let's draw the random numbers and write them
                float *fp32_buffer = (float*)btm_alloc(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng_2);
                for (size_t j = 0; j < n; j++) {
		    size_t i_pmc = offset+layer_offset+j;
		    if (params_memory_active[i_pmc]) {
			params_memory_cpu[i_pmc] = fp32_buffer[j];
		    }
		    /*else { // set inactive weights to 0
			params_memory_cpu[i_pmc] = 0.0f;
			}*/
                }
                btm_free(fp32_buffer);
            }
            offset += model->param_sizes[i];
        }
    }

    // apply checkpoint data
    if (cp_bytes % 8 != 0) {
	BTMLOG("Warning: The size of the checkpoint data is not a multiple of 8. Ignoring it.\n");
    }
    else {
	size_t bytes_scanned = 0;
	while (bytes_scanned<cp_bytes) {
	    size_t n_cp_weights = *((size_t*)(cp+bytes_scanned+32));
	    for (int i=0; i<n_cp_weights; i++) {
		uint32_t weight_index = ((uint32_t*)(cp+bytes_scanned+40))[2*i];
		float weight_value = ((float*)(cp+bytes_scanned+40))[2*i+1];
		params_memory_cpu[weight_index] = weight_value;
	    }
	    bytes_scanned += 40+n_cp_weights*8;
	}
    }
    
    // copy them to the model
    memcpy(model->params_memory, params_memory_cpu, num_parameters_bytes);
    btm_free(params_memory_cpu);
    //model->params_memory = params_memory_cpu;
}

// MoE FFN forward for one layer. Per token: router logits over E experts,
// consensus top-k selection (btmmoe_topk_gates: ties keep the lower index),
// renormalized softmax gates, out = sum_s gate_s * expert_s(x). Caches logits,
// selection, gates, per-slot hidden activations, and per-slot pre-gate outputs
// for the backward pass. Parallel over tokens (disjoint writes; deterministic).
#endif /* !BTM_ONCHAIN */
void moe_ffn_forward(float* out, float* fch, float* fch_gelu,
                     float* router_logits, float* gates, float* expert_sel, float* expert_out,
                     const float* inp,
                     const float* fcw, const float* fcb,
                     const float* fcprojw, const float* fcprojb,
                     const float* routerw, const float* routerb,
                     int B, int T, int C, int E, int K) {
#ifdef BTMV_MEASURE_NO_MOE
    // MEASUREMENT ONLY: zero the output and return, to price the MoE FFN's share of the
    // gas classes by difference. Produces wrong losses by construction.
    (void)fch; (void)fch_gelu; (void)router_logits; (void)gates; (void)expert_sel;
    (void)expert_out; (void)inp; (void)fcw; (void)fcb; (void)fcprojw; (void)fcprojb;
    (void)routerw; (void)routerb; (void)E; (void)K;
    for (int i = 0; i < B * T * C; i++) out[i] = 0.0f;
    return;
#endif

    int H = MOE_H; // expert hidden width (fine-grained)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            size_t bt = (size_t)b * T + t;
            const float* x = inp + bt * C;
            float* logits = router_logits + bt * E;
            for (int e = 0; e < E; e++) {
                logits[e] = routerb[e] + btmk_dot(routerw + (size_t)e * C, x, C);
            }
            int sel[8]; // K <= 8
            float g[8];
            btmmoe_topk_gates(logits, E, K, sel, g);
            float* out_bt = out + bt * C;
            for (int c = 0; c < C; c++) out_bt[c] = 0.0f;
            for (int s = 0; s < K; s++) {
                int e = sel[s];
                expert_sel[bt * K + s] = (float)e;
                gates[bt * K + s] = g[s];
                float* h = fch + (bt * K + s) * H;
                const float* efcw = fcw + (size_t)e * H * C;
                const float* efcb = fcb + (size_t)e * H;
                for (int i = 0; i < H; i++) {
                    h[i] = efcb[i] + btmk_dot(efcw + (size_t)i * C, x, C);
                }
                float* hg = fch_gelu + (bt * K + s) * H;
                gelu_forward(hg, h, H);
                float* o = expert_out + (bt * K + s) * C;
                const float* epw = fcprojw + (size_t)e * C * H;
                const float* epb = fcprojb + (size_t)e * C;
                for (int c = 0; c < C; c++) {
                    o[c] = epb[c] + btmk_dot(epw + (size_t)c * H, hg, H);
                }
                for (int c = 0; c < C; c++) out_bt[c] += g[s] * o[c];
            }
        }
    }
}

// MoE FFN backward for one layer. Serial over tokens: different tokens may
// select the same expert, so weight-gradient accumulation would race under OMP.
// dinp accumulates (+=); weight grads accumulate into the zeroed grads buffers.
// ---- host-only: the backward pass. A verifier only ever runs forward.
#ifndef BTM_ONCHAIN
void moe_ffn_backward(float* dinp,
                      float* dfcw, float* dfcb, float* dfcprojw, float* dfcprojb,
                      float* drouterw, float* drouterb,
                      float* dfch, float* dfch_gelu,
                      const float* dout,
                      const float* fch, const float* fch_gelu,
                      const float* gates, const float* expert_sel, const float* expert_out,
                      const float* inp,
                      const float* fcw, const float* fcprojw, const float* routerw,
                      int B, int T, int C, int E, int K) {
    int H = MOE_H;
    for (size_t bt = 0; bt < (size_t)B * T; bt++) {
        const float* x = inp + bt * C;
        const float* dout_bt = dout + bt * C;
        float* dx = dinp + bt * C;
        // gate gradients: dgate_s = <dout, expert_out_s>
        float dgate[8], gsum = 0.0f;
        for (int s = 0; s < K; s++) {
            const float* o = expert_out + (bt * K + s) * C;
            float acc = 0.0f;
            for (int c = 0; c < C; c++) acc += dout_bt[c] * o[c];
            dgate[s] = acc;
            gsum += gates[bt * K + s] * acc;
        }
        for (int s = 0; s < K; s++) {
            int e = (int)expert_sel[bt * K + s];
            float g = gates[bt * K + s];
            // softmax-over-selected backward -> selected logits, then router
            float dlogit = g * (dgate[s] - gsum);
            drouterb[e] += dlogit;
            float* drw = drouterw + (size_t)e * C;
            const float* rw = routerw + (size_t)e * C;
            for (int c = 0; c < C; c++) {
                drw[c] += dlogit * x[c];
                dx[c] += dlogit * rw[c];
            }
            // expert path, scaled by the gate
            const float* hg = fch_gelu + (bt * K + s) * H;
            float* dhg = dfch_gelu + (bt * K + s) * H;
            const float* epw = fcprojw + (size_t)e * C * H;
            float* depw = dfcprojw + (size_t)e * C * H;
            float* depb = dfcprojb + (size_t)e * C;
            for (int i = 0; i < H; i++) dhg[i] = 0.0f;
            for (int c = 0; c < C; c++) {
                float d = g * dout_bt[c];
                const float* wrow = epw + (size_t)c * H;
                float* dwrow = depw + (size_t)c * H;
                for (int i = 0; i < H; i++) {
                    dhg[i] += wrow[i] * d;
                    dwrow[i] += hg[i] * d;
                }
                depb[c] += d;
            }
            const float* h = fch + (bt * K + s) * H;
            float* dh = dfch + (bt * K + s) * H;
            for (int i = 0; i < H; i++) dh[i] = 0.0f;
            gelu_backward(dh, (float*)h, dhg, H);
            const float* efcw = fcw + (size_t)e * H * C;
            float* defcw = dfcw + (size_t)e * H * C;
            float* defcb = dfcb + (size_t)e * H;
            for (int i = 0; i < H; i++) {
                float d = dh[i];
                if (d == 0.0f) continue;
                const float* wrow = efcw + (size_t)i * C;
                float* dwrow = defcw + (size_t)i * C;
                for (int c = 0; c < C; c++) {
                    dx[c] += wrow[c] * d;
                    dwrow[c] += x[c] * d;
                }
                defcb[i] += d;
            }
        }
    }
}

#endif /* !BTM_ONCHAIN */
void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        BTMLOG("Error: model was not initialized properly.\n");
        BTM_ABORT();
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        BTM_ASSERT(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            BTM_ASSERT(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C; // encoded
        model->act_sizes[1] = L * B * T * C; // ln1
        model->act_sizes[2] = L * B * T;  // ln1_mean
        model->act_sizes[3] = L * B * T;  // ln1_rstd
        model->act_sizes[4] = L * B * T * 3*C; // qkv
        model->act_sizes[5] = L * B * T * C;  // atty
        model->act_sizes[6] = L * B * NH * T * T;  // preatt
        model->act_sizes[7] = L * B * NH * T * T;  // att
        model->act_sizes[8] = L * B * T * C; // attproj
        model->act_sizes[9] = L * B * T * C; // residual2
        model->act_sizes[10] = L * B * T * C; // ln2
        model->act_sizes[11] = L * B * T; // ln2_mean
        model->act_sizes[12] = L * B * T; // ln2_rstd
        model->act_sizes[13] = L * B * T * MOE_K * MOE_H; // fch (per selected expert)
        model->act_sizes[14] = L * B * T * MOE_K * MOE_H; // fch_gelu
        model->act_sizes[15] = L * B * T * C; // fcproj
        model->act_sizes[16] = L * B * T * C; // residual3
        model->act_sizes[17] = B * T * C; // lnf
        model->act_sizes[18] = B * T; // lnf_mean
        model->act_sizes[19] = B * T; // lnf_rstd
        model->act_sizes[20] = B * T * Vp; // logits
        model->act_sizes[21] = B * T * Vp; // probs
        model->act_sizes[22] = B * T; // losses
        model->act_sizes[23] = L * B * T * MOE_E; // router_logits
        model->act_sizes[24] = L * B * T * MOE_K; // gates
        model->act_sizes[25] = L * B * T * MOE_K; // expert_sel
        model->act_sizes[26] = L * B * T * MOE_K * C; // expert_out
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        BTMLOG("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)btm_alloc(B * T * sizeof(int));
        model->targets = (int*)btm_alloc(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            BTMLOG("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            BTM_ABORT();
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + (size_t)l * MOE_E * MOE_H * C;     // (E, H, C)
        float* l_fcb = params.fcb + (size_t)l * MOE_E * MOE_H;         // (E, H)
        float* l_fcprojw = params.fcprojw + (size_t)l * MOE_E * C * MOE_H; // (E, C, H)
        float* l_fcprojb = params.fcprojb + (size_t)l * MOE_E * C;     // (E, C)
        float* l_routerw = params.routerw + (size_t)l * MOE_E * C;     // (E, C)
        float* l_routerb = params.routerb + (size_t)l * MOE_E;         // (E)

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + (size_t)l * B * T * MOE_K * MOE_H;
        float* l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * MOE_K * MOE_H;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        float* l_router_logits = acts.router_logits + (size_t)l * B * T * MOE_E;
        float* l_gates = acts.gates + (size_t)l * B * T * MOE_K;
        float* l_expert_sel = acts.expert_sel + (size_t)l * B * T * MOE_K;
        float* l_expert_out = acts.expert_out + (size_t)l * B * T * MOE_K * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward_k(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward_k(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        moe_ffn_forward(l_fcproj, l_fch, l_fch_gelu, l_router_logits, l_gates, l_expert_sel,
                        l_expert_out, l_ln2, l_fcw, l_fcb, l_fcprojw, l_fcprojb,
                        l_routerw, l_routerb, B, T, C, MOE_E, MOE_K);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward_k(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) {
	    mean_loss += model->acts.losses[i];
	}
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

// ---- host-only: zero_grad, backward and the AdamW update. Mining needs them; a
// verifier does not, and dropping them removes the only float_control-sensitive
// code a verifier would otherwise carry.
#ifndef BTM_ONCHAIN
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        BTMLOG("Error: must forward with targets before backward\n");
        BTM_ABORT();
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + (size_t)l * MOE_E * MOE_H * C;
        float* l_fcprojw = params.fcprojw + (size_t)l * MOE_E * C * MOE_H;
        float* l_routerw = params.routerw + (size_t)l * MOE_E * C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + (size_t)l * MOE_E * MOE_H * C;
        float* dl_fcb = grads.fcb + (size_t)l * MOE_E * MOE_H;
        float* dl_fcprojw = grads.fcprojw + (size_t)l * MOE_E * C * MOE_H;
        float* dl_fcprojb = grads.fcprojb + (size_t)l * MOE_E * C;
        float* dl_routerw = grads.routerw + (size_t)l * MOE_E * C;
        float* dl_routerb = grads.routerb + (size_t)l * MOE_E;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + (size_t)l * B * T * MOE_K * MOE_H;
        float* l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * MOE_K * MOE_H;
        float* l_gates = acts.gates + (size_t)l * B * T * MOE_K;
        float* l_expert_sel = acts.expert_sel + (size_t)l * B * T * MOE_K;
        float* l_expert_out = acts.expert_out + (size_t)l * B * T * MOE_K * C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + (size_t)l * B * T * MOE_K * MOE_H;
        float* dl_fch_gelu = grads_acts.fch_gelu + (size_t)l * B * T * MOE_K * MOE_H;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        moe_ffn_backward(dl_ln2, dl_fcw, dl_fcb, dl_fcprojw, dl_fcprojb,
                         dl_routerw, dl_routerb, dl_fch, dl_fch_gelu,
                         dl_fcproj, l_fch, l_fch_gelu, l_gates, l_expert_sel, l_expert_out,
                         l_ln2, l_fcw, l_fcprojw, l_routerw, B, T, C, MOE_E, MOE_K);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    //BTMLOG("gpt2_update(): beta1=%.2e, beta2=%.2e, weight_decay=%.2e\n",beta1,beta2,weight_decay);
    
    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
	if (!model->params_memory_active[i])
	    continue;
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

	//BTMLOG("update param %lu (%.2e) with m_hat = %.2e (grad %.2e)\n",i,param,m_hat,grad);

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

#endif /* !BTM_ONCHAIN */
void gpt2_free(GPT2 *model) {
    btm_free(model->params_memory);
    btm_free(model->grads_memory);
    btm_free(model->m_memory);
    btm_free(model->v_memory);
    btm_free(model->acts_memory);
    btm_free(model->grads_acts_memory);
    btm_free(model->inputs);
    btm_free(model->targets);
    btm_free(model->active_weights);
    btm_free(model->params_memory_active);
}





// ----------------------------------------------------------------------------
// Trunk chain verifier (wasm32-wasi compatible; canonical btmk_dot kernel;
// build with -msimd128 for the bit-identical SIMD128 fast path)

//! Build the trunk weights, applying only the records that start below `cp_limit`.
//!
//! The caller passes the offset of the record being SCORED, so what comes back is the
//! state that record is an improvement OVER -- see gpt2_eval_trunk for why the score is
//! a difference. `cp_limit == cp_bytes` applies the whole chain.
//! The trunk's geometry and a zeroed parameter block. CONSENSUS: these six numbers plus
//! MOE_E/MOE_K/MOE_H are what a record's framing is computed from, so a verifier that
//! disagrees here cannot walk the chain at all.
//!
//! Returns the byte size of params_memory, or 0 if it could not be allocated -- which on
//! wasm means the model does not fit under the node's memory cap, not that anything went
//! wrong. Shared by the native file path and the on-chain fold path so the two cannot
//! drift apart in geometry.
static size_t gpt2_init_trunk_params(GPT2 *model) {
    model->config.num_layers = 12;
    model->config.channels = 768;
    model->config.num_heads = 12;
    model->config.max_seq_len = 1024;
    model->config.vocab_size = TRUNK_VOCAB;
    model->config.padded_vocab_size = TRUNK_VOCAB;

    fill_in_parameter_sizes(model->param_sizes, model->config);
    model->num_parameters = 0;
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_sizes[i];
        num_parameters_bytes += model->param_sizes[i] * sizeof(float);
    }
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    if (!model->params_memory) return 0;
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    model->active_weights = NULL;
    model->n_active_weights = 0;
    model->params_memory_active = NULL;
    return num_parameters_bytes;
}

void gpt2_build_from_chain_trunk(GPT2 *model, uchar* cp, size_t cp_bytes, size_t cp_limit) {
    if (cp_limit > cp_bytes) cp_limit = cp_bytes;
    size_t num_parameters_bytes = gpt2_init_trunk_params(model);
    if (num_parameters_bytes == 0) { BTMLOG("cannot allocate trunk parameters\n"); BTM_ABORT(); }

    size_t moe_offsets[BTMMOE_NUM_TENSORS];
    btmmoe_tensor_offsets(model->param_sizes, moe_offsets);

    // node-side weight-state cache (llmc/btmcache.h; NOT consensus): if the
    // cached tip matches a record head in the chain, resume from there instead
    // of re-deriving init + full replay.
    //
    // Absent on-chain, and not merely because there is no filesystem: a verifier must be
    // a pure function of the chain, so it may not carry state between calls. The cache is
    // the NATIVE harness's shortcut for re-scoring the same chain repeatedly, and the
    // on-chain build simply rebuilds. What makes that affordable is the checkpoint, which
    // bounds a rebuild to one epoch rather than the whole history.
    size_t start_off = 0;
    int need_init = 1;
#ifndef BTM_ONCHAIN
    const char* cache_path = "btm-trunk-state.bin";
    unsigned char cache_tip[32];
    if (btmcache_load(cache_path, cache_tip, model->params_memory, model->num_parameters) == 0) {
        unsigned char gen[32];
        btmcache_genesis_tip(gen);
        if (memcmp(cache_tip, gen, 32) == 0) {
            need_init = 0; // cached bare init; all records still to apply
        } else {
            // Only records BELOW the limit may be resumed from: a cached tip at or past
            // it describes a state that already includes the record we are about to
            // score, which would make the improvement come out as zero.
            size_t off = 0;
            while (off < cp_limit) {
                size_t rsz = btmmoe_record_size_h(cp + off, cp_limit - off, 768, MOE_H, MOE_E);
                if (rsz == 0) break;
                if (memcmp(cp + off, cache_tip, 32) == 0) { need_init = 0; start_off = off + rsz; break; }
                off += rsz;
            }
        }
    }
#endif /* !BTM_ONCHAIN */
    if (need_init) {
        BTMLOG("state cache miss: full rebuild\n");
        memset(model->params_memory, 0, num_parameters_bytes);
        btmmoe_init_base_params(model->params_memory, model->param_sizes, moe_offsets,
                                12, 768, model->config.vocab_size, MOE_E, MOE_H, BTM2_BASE_SEED);
        start_off = 0;
    } else {
        BTMLOG("state cache hit (resuming at chain offset %zu)\n", start_off);
    }

    int applied_any = 0;
#ifndef BTM_ONCHAIN
    unsigned char newtip[32];
    btmcache_genesis_tip(newtip);
#endif
    size_t off = start_off;
    size_t last_head_off = (size_t)-1;
    {
        // The tip must name the last record this build APPLIES, not the last in the
        // file: the cache stores the state at cp_limit.
        size_t o2 = 0;
        while (o2 < cp_limit) {
            size_t rsz = btmmoe_record_size_h(cp + o2, cp_limit - o2, 768, MOE_H, MOE_E);
            if (rsz == 0) break;
            last_head_off = o2;
            o2 += rsz;
        }
    }
    while (off < cp_limit) {
        size_t rsz = btmmoe_record_size_h(cp + off, cp_limit - off, 768, MOE_H, MOE_E);
        if (rsz == 0) { BTMLOG("error: malformed trunk chain record at offset %zu\n", off); BTM_ABORT(); }
        if (btmmoe_apply_record_h(model->params_memory, cp + off, rsz,
                                  12, 768, MOE_E, MOE_H, model->param_sizes, moe_offsets) != 0) {
            BTMLOG("error: failed to apply trunk chain record at offset %zu\n", off); BTM_ABORT();
        }
        applied_any = 1;
        off += rsz;
    }
#ifndef BTM_ONCHAIN
    if (need_init || applied_any) {
        if (last_head_off != (size_t)-1) memcpy(newtip, cp + last_head_off, 32);
        // Never cache the state of a chain supplied by BTM_CHAIN. A candidate record's
        // folded state is not the real chain's, and the cache is keyed on the last
        // record's head -- writing it here poisons the next run, which is exactly the
        // aliasing that made the first compaction measurement report one file's loss for
        // another. Also keeps candidate scoring cheap: the base stays a cache hit.
        if (getenv("BTM_CHAIN")) {
            BTMLOG("not caching state: scoring a BTM_CHAIN candidate\n");
        } else if (btmcache_save(cache_path, newtip, model->params_memory,
                                 model->num_parameters) != 0) {
            BTMLOG("warning: could not write state cache\n");
        }
    }
#endif /* !BTM_ONCHAIN */
}

//! Score the LAST record of `cp` by how much it IMPROVES the loss of the chain before it,
//! on the window that record commits to.
//!
//! WHY A DIFFERENCE AND NOT A LOSS. The window is 63 token predictions chosen by the
//! record's own deltas, and window difficulty varies by ~0.75 nats across the corpus --
//! far more than 30 LoRA steps move the loss. An absolute loss therefore reports which
//! window a record drew, not what it learned, and best-of-n over absolute loss is
//! rejection sampling on the draw: measured directly, two attempts whose training
//! trajectories agreed to four decimals scored 8.342280 and 7.592348 purely because they
//! landed on leaf 2926 versus leaf 3287.
//!
//! Differencing against the SAME window removes it exactly. Both forwards below run on
//! identical tokens, so whatever that window costs cancels, and what is left is the
//! record's own contribution. Subtracting two f32 losses is exact in IEEE, so the
//! improvement is as reproducible as the losses are.
//!
//! `*ploss` gets the loss with the record applied, `*pbase_loss` the loss without it;
//! the improvement is their difference, in that order.
//! Locate the candidate's corpus proof, check that it names the window its own deltas
//! commit to and that the window really is in the committed corpus, and decode the
//! window into next-token pairs. Returns 0 and fills vin/vtar; 1 means NO VALID SOLUTION.
//!
//! `prev_head` is the previous record's head, or the anchor hash for a chain's first
//! record (0xff.. when there is no anchor either). Takes NO model on purpose: a bad proof
//! should cost a verifier a few hashes rather than a 1.0 GiB build.
//!
//! Shared by the native harness and the on-chain verify(), so the two cannot drift on any
//! rule that decides whether a record is a solution.
static int btmv_proof_window(const uchar* rec, size_t rec_len, const uchar prev_head[32],
                             int* vin, int* vtar) {
    if (rec_len < BTM2_HEADER_BYTES) {
        BTMLOG("no valid solution: record is shorter than its own header\n");
        return 1;
    }

    // The chaining preimage: this record with prev_head in place of its own head. The
    // head must be the double-SHA256 of it, or the record is not bound to its content and
    // the next record's prev_head is merely whatever this one asserts.
    uchar* preimage = (uchar*)btm_alloc(rec_len);
    if (!preimage) { BTMLOG("no valid solution: cannot allocate the preimage\n"); return 1; }
    memcpy(preimage, rec, rec_len);
    memcpy(preimage, prev_head, 32);
    {
        uchar h1[32], h2[32];
        sha256_hash(preimage, rec_len, h1);
        sha256_hash(h1, 32, h2);
        if (memcmp(h2, rec, 32) != 0) {
            BTMLOG("no valid solution: record head does not chain over prev_head\n");
            btm_free(preimage);
            return 1;
        }
    }

    // --- the scored window comes from the record, not from a corpus on disk ----------
    //
    // A wasm verifier has no filesystem, so it cannot hold a 10B-token corpus. The record
    // carries the window it was scored on plus a Merkle path to the root this module
    // commits to; everything checked here arrives in the record, so verification needs no
    // data availability at all (llmc/btmcorpus.h).
    const unsigned char* proof = NULL;
    size_t delta_len = 0;
    {
        uint16_t n_ent;
        memcpy(&n_ent, rec + 32, 2);
        const unsigned char* e = rec + BTM2_HEADER_BYTES;
        size_t left = rec_len - BTM2_HEADER_BYTES;
        for (uint16_t i = 0; i < n_ent; i++) {
            size_t esz = btmmoe_entry_size_h(e, left, 768, MOE_H, MOE_E);
            if (esz == 0 || esz > left) {
                BTMLOG("no valid solution: malformed entry %u\n", i);
                btm_free(preimage);
                return 1;
            }
            if (e[2] == BTM2_DELTA_CORPUS_PROOF) {
                // Must be the LAST entry: the index is derived from the prefix ending
                // here, so a proof anywhere else would be covered by its own input.
                if (i + 1 != n_ent) {
                    BTMLOG("no valid solution: corpus proof is entry %u of %u, not last\n",
                           i, n_ent);
                    btm_free(preimage);
                    return 1;
                }
                proof = e;
            } else {
                delta_len += esz;
            }
            e += esz;
            left -= esz;
        }
    }
    if (!proof) {
        BTMLOG("no valid solution: record carries no corpus proof\n");
        btm_free(preimage);
        return 1;
    }

    uint64_t claimed = 0;
    for (int i = 7; i >= 0; i--) claimed = (claimed << 8) | proof[4 + i];
    const unsigned char* leaf = proof + 12;
    int n_sibs = proof[12 + BTMC_LEAF_BYTES];
    const unsigned char* sibs = proof + 13 + BTMC_LEAF_BYTES;

    // 1. the window must be the one this record's deltas commit to. `preimage` already
    //    holds prev_head in its first 32 bytes, so the prefix is contiguous.
    uint64_t expect = btmc_leaf_index(preimage, BTM2_HEADER_BYTES + delta_len,
                                      BTM_CORPUS_N_LEAVES);
    btm_free(preimage);
    if (claimed != expect) {
        BTMLOG("no valid solution: proof names leaf %llu, deltas commit to %llu\n",
               (unsigned long long)claimed, (unsigned long long)expect);
        return 1;
    }
    // 2. the window must really be in the committed corpus.
    {
        unsigned char want[32], got[32];
        if (btm_unhex(BTM_CORPUS_ROOT_HEX, want, 32) != 0) {
            BTMLOG("no valid solution: BTM_CORPUS_ROOT_HEX is not 64 hex digits\n");
            return 1;
        }
        int rc = btmc_fold(claimed, BTM_CORPUS_N_LEAVES, leaf, sibs, n_sibs, got);
        if (rc != 0 || memcmp(got, want, 32) != 0) {
            BTMLOG("no valid solution: corpus proof does not fold to the root (rc=%d)\n", rc);
            return 1;
        }
    }
    BTMLOG("corpus proof ok: leaf %llu of %llu, depth %d\n",
           (unsigned long long)claimed, (unsigned long long)BTM_CORPUS_N_LEAVES, n_sibs);

    // The window, as next-token pairs: a 64-token leaf scores 63 predictions.
    for (int j = 0; j < BTM_CORPUS_SCORE_T; j++) {
        uint16_t a, b;
        memcpy(&a, leaf + 2 * j, 2);
        memcpy(&b, leaf + 2 * (j + 1), 2);
        vin[j] = (int)a;
        vtar[j] = (int)b;
    }
    // The tokens are uint16 out of the record, so they can name an id the model has no
    // embedding for -- 65535 against a vocab of 16384. The proof must fold to the
    // committed root, so in practice they are the real corpus's and all in range; this is
    // the defence in depth, and it turns what would otherwise be a trap deep in
    // gpt2_forward's index check into an ordinary "no valid solution".
    for (int j = 0; j < BTM_CORPUS_SCORE_T; j++) {
        if (vin[j] >= TRUNK_VOCAB || vtar[j] >= TRUNK_VOCAB) {
            BTMLOG("no valid solution: window token %d out of vocab range (%d >= %d)\n",
                   j, vin[j] > vtar[j] ? vin[j] : vtar[j], TRUNK_VOCAB);
            return 1;
        }
    }
    BTMLOG("inputs:");
    for (int j = 0; j < BTM_CORPUS_SCORE_T; j++) BTMLOG(" %d", vin[j]);
    BTMLOG("\n");
    return 0;
}

//! Score `rec` against the base state already held in `model`: one forward on the window,
//! the record applied, a second forward on the SAME window, then the gate.
//! Returns 0 if the record is a valid solution, 1 if it is not. `*pbase` and `*pval` are
//! written before the gate runs, so a rejected record still yields its numbers.
static int btmv_score(GPT2* model, const uchar* rec, size_t rec_len,
                      const int* vin, const int* vtar, uint32_t nbits, float gate_g,
                      float* pbase, float* pval) {
    gpt2_forward(model, (int*)vin, (int*)vtar, 1, BTM_CORPUS_SCORE_T);
    float base_loss = model->mean_loss;
    BTMLOG("base loss %f\n", base_loss);

    {
        size_t moe_offsets[BTMMOE_NUM_TENSORS];
        btmmoe_tensor_offsets(model->param_sizes, moe_offsets);
#ifdef BTMV_MEASURE_NO_APPLY
        // MEASUREMENT ONLY: skip applying the record, to price it by difference. Makes
        // both forwards identical, so improvement comes out as exactly zero.
        if (0) {
#else
        if (btmmoe_apply_record_h(model->params_memory, rec, rec_len,
                                  12, 768, MOE_E, MOE_H, model->param_sizes,
                                  moe_offsets) != 0) {
#endif
            BTMLOG("no valid solution: cannot apply the scored record\n");
            return 1;
        }
    }
    // Same B,T as the baseline forward, so the activation buffers are reused rather than
    // reallocated -- and, more to the point, the two losses come from the same code path
    // over the same tokens, differing only in the weights.
    //
    // GAS: THIS SECOND FORWARD COSTS THE WHOLE MARGIN. src/dynamicalgo/gasclasses.h sets
    // each per-class limit as max(2 x trunk-count, count for a 0.5 s slice) -- calibrated
    // against THIS algo, at 2x. One forward over 63 tokens is ~4.38e9 multiply-accumulates
    // (69.5M active params per token: dense + top-4 experts + lm_head), so with 4 f32
    // lanes and separate mul/add that is ~2.2e9 SIMD ops against GAS_LIMIT[GC_SIMD] =
    // 5e9. One forward is ~44% of it; two are ~88%.
    //
    // So improvement scoring, which is what makes the gate measurable at all, leaves
    // almost no headroom. Anything that multiplies the forward work -- more scored
    // windows, a longer T, a larger top-k -- trips the budget, and tripping gas is not a
    // slow verifier but a solution scored INVALID. Those counts are estimated from FLOPs,
    // not measured; contrib/dynamicalgo/aot_harness runs the module under the node's own
    // instrumentation and is what should settle them before this ships.
    gpt2_forward(model, (int*)vin, (int*)vtar, 1, BTM_CORPUS_SCORE_T);
    float val_loss = model->mean_loss;
    BTMLOG("val loss %f\n", val_loss);
    uint32_t loss_bits;
    memcpy(&loss_bits, &val_loss, 4);
    BTMLOG("val loss bits = %08x\n", loss_bits);

    // THE SCORE. Exact in IEEE: both operands are f32 and the subtraction is a single
    // rounded operation, so every verifier that agrees on the two losses agrees on this.
    float improvement = base_loss - val_loss;
    uint32_t imp_bits;
    memcpy(&imp_bits, &improvement, 4);
    BTMLOG("improvement %+.8f\n", improvement);
    BTMLOG("improvement bits = %08x\n", imp_bits);

    if (pval) *pval = val_loss;
    if (pbase) *pbase = base_loss;

    // --- THE GATE (llmc/btmmoe.h) ----------------------------------------------------
    // Failing it means NO VALID SOLUTION -- the same fall-through as a malformed record
    // or a proof that does not fold. It never makes the containing block invalid.
    {
        float scale = btmmoe_difficulty_scale(BTM_GATE_NBITS_0, nbits);
        float required = btmmoe_gate_required_g(base_loss, nbits, gate_g);
        uint32_t req_bits;
        memcpy(&req_bits, &required, 4);
        BTMLOG("gate requires %+.8f (base %f * g %.9g, difficulty scale %g from nbits"
               " %08x vs D0 %08x), bits = %08x\n",
               required, base_loss, (double)gate_g, (double)scale,
               nbits, (unsigned)BTM_GATE_NBITS_0, req_bits);
        if (!btmmoe_gate_passes_g(base_loss, val_loss, nbits, gate_g)) {
            BTMLOG("no valid solution: improvement %+.8f is below the required %+.8f\n",
                   improvement, required);
            return 1;
        }
        BTMLOG("gate passed: improvement %+.8f >= %+.8f\n", improvement, required);
    }
    return 0;
}

//! Native harness: score the LAST record of a flat chain buffer. The on-chain path is
//! verify(), which takes the candidate from the block's solution and the base state from
//! the chain imports; both share btmv_proof_window and btmv_score, so neither can drift
//! on a rule that decides validity.
int gpt2_eval_trunk(float* ploss, float* pbase_loss, uchar* block_hash,
                    uchar* cp, size_t cp_bytes, uint32_t nbits) {
    GPT2 model;
    BTM_TIMER(t0, t1);
    BTM_TICK(t0);

    size_t off = 0, last_off = 0, last_size = 0, prev_off = (size_t)-1;
    while (off < cp_bytes) {
        size_t rsz = btmmoe_record_size_h(cp + off, cp_bytes - off, 768, MOE_H, MOE_E);
        if (rsz == 0) { BTMLOG("error: malformed trunk chain record at offset %zu\n", off); return 1; }
        if (off + rsz >= cp_bytes) { last_off = off; last_size = rsz; }
        else { prev_off = off; }
        off += rsz;
    }
    if (last_size == 0) { BTMLOG("error: no complete record in chain data\n"); return 1; }

    uchar prev_head[32];
    if (prev_off != (size_t)-1) memcpy(prev_head, cp + prev_off, 32);
    else if (block_hash) memcpy(prev_head, block_hash, 32);
    else memset(prev_head, 255, 32);

    int vin[BTM_CORPUS_SCORE_T], vtar[BTM_CORPUS_SCORE_T];
    if (btmv_proof_window(cp + last_off, last_size, prev_head, vin, vtar) != 0) return 1;

    // The baseline is the chain WITHOUT this record: `last_off` is where it begins, so the
    // build stops just short of it. For a chain's first record that is 0 and the baseline
    // is the bare fixed-seed init, which is the correct thing to improve on.
    gpt2_build_from_chain_trunk(&model, cp, cp_bytes, last_off);
    BTM_TICK(t1);
    BTMLOG("model build took %.3f s\n", BTM_ELAPSED(t0, t1));

    // BTM_GATE_FRACTION, not a retargeted g: a flat chain file carries no verdict
    // history. Every record in it is one a verifier accepted, so counting would give
    // V == W and the HARDEST possible gate, which is the wrong answer for a dev harness.
    // The on-chain path reads real verdicts and retargets; see verify().
    int r = btmv_score(&model, cp + last_off, last_size, vin, vtar, nbits,
                       BTM_GATE_FRACTION, pbase_loss, ploss);
    BTM_TICK(t1);
    BTMLOG("two forwards took %.3f s\n", BTM_ELAPSED(t0, t1));
    gpt2_free(&model);
    return r;
}

// ---- host-only: the native harness. On-chain the entry point is verify(), below.
#ifndef BTM_ONCHAIN
int main(int argc, char** argv) {

    BTM_ASSERT(CHAR_BIT == 8);
    BTM_ASSERT(sizeof(float) == 4);
    BTM_ASSERT(sizeof(unsigned int) == 4);
    const int one = 1;
    BTM_ASSERT( (*(const char*)&one) != 0 ); // little endian

    uchar* block_hash = 0;
    if (argc > 1) {
        size_t len = hex_to_uchar(&block_hash, argv[1]);
        if (len > 0) {
            if (len != 32) {
                BTMLOG("block hash length must be 32 bytes (length 64 hex string)\n");
                BTM_ABORT();
            }
            BTMLOG("have block_hash =");
            for (int i = 0; i < 32; i++) BTMLOG(" %02x", block_hash[i]);
            BTMLOG("\n");
        }
    }

    const char* cpfname = "btm-cp-trunk.bin";
#ifndef BTM_ONCHAIN
    // DIAGNOSTIC ONLY, absent from every consensus path (the wasm verifier has no
    // environment): score a DIFFERENT chain file without moving anything.
    //
    // Scoring a compaction against the chain it compacts otherwise means copying each
    // candidate over btm-cp-trunk.bin in turn, which puts the real chain one skipped
    // restore away from being overwritten by a synthetic file -- and the synthetic ones
    // parse perfectly, so nothing complains. Mining state should not be reachable by a
    // read-only experiment at all.
    {
        const char* alt = getenv("BTM_CHAIN");
        if (alt && *alt) {
            cpfname = alt;
            BTMLOG("DIAGNOSTIC: reading chain from %s (BTM_CHAIN)\n", cpfname);
        }
    }
#endif
    FILE* cpf = fopen(cpfname, "rb");
    if (!cpf) { BTMLOG("error: cannot open %s\n", cpfname); BTM_ABORT(); }
    fseekCheck(cpf, 0L, SEEK_END);
    size_t cp_bytes = (size_t)ftell(cpf);
    fseekCheck(cpf, 0L, SEEK_SET);
    uchar* cp = (uchar*)btm_alloc(cp_bytes);
    freadCheck(cp, 1, cp_bytes, cpf);
    fcloseCheck(cpf);
    BTMLOG("cp_bytes = %zu\n", cp_bytes);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    float loss = -1.0f, base_loss = -1.0f;
    // On-chain, nbits is a verify() parameter from the block being validated. Here it
    // comes from the environment so the harness can exercise the scaling; the default
    // reproduces D/D_0 = 1, i.e. the unscaled 1/64 rule.
    uint32_t nbits = BTM_GATE_NBITS_0;
    {
        const char* s = getenv("BTM_NBITS");
        if (s && *s) nbits = (uint32_t)strtoul(s, NULL, 0);
    }
    int ret = gpt2_eval_trunk(&loss, &base_loss, block_hash, cp, cp_bytes, nbits);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    BTMLOG("eval took %.3f s (base %f, loss %f, improvement %+.8f)\n",
           time_elapsed_s, base_loss, loss, base_loss - loss);

    btm_free(cp);
    if (block_hash) btm_free(block_hash);
    return ret;
}

#endif /* !BTM_ONCHAIN */
#ifdef BTM_ONCHAIN
// ----------------------------------------------------------------------------
// On-chain entry point.
//
// The node calls verify() with the block's solution bytes as `nonce`; the base state
// comes from the slot's earlier approved records via the chain imports. Nothing here
// touches a filesystem, and the module imports nothing but chain.*.
//
// defines: the chain source and its fold-and-discard iterator
#include "llmc/btmchain.h"

//! Largest solution this verifier will accept. A record at E=32 is 593,583 bytes, and
//! MAX_SOLUTION_TX_WEIGHT caps a block's solution payload near 975 KB regardless, so a
//! megabyte is comfortable while still refusing anything absurd before allocating.
#define BTMV_RECORD_MAX (1u << 20)

static void btmv_put_u32le(unsigned char* p, uint32_t v) {
    p[0] = (unsigned char)v;         p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

//! alpha and beta as Q32 fixed point, both 0.5, matching contrib/dynamicalgo's reference
//! algos. CONSTANT, and written on EVERY path -- the early returns and the node's
//! empty-solution probe for the no-solution branch included. A solution-dependent alpha
//! would nullify the inclusion incentive: the native miner would be paid the same whether
//! or not it included a solution that arrived in its mempool.
static void btmv_put_ab(unsigned char* out_ab) {
    btmv_put_u32le(out_ab,     0x80000000u);
    btmv_put_u32le(out_ab + 4, 0x80000000u);
}

typedef struct {
    GPT2* model;
    size_t* moe;
    int err;
} btmv_fold_ctx;

static int btmv_fold_one(const unsigned char* rec, size_t len, void* c) {
    btmv_fold_ctx* f = (btmv_fold_ctx*)c;
    if (btmmoe_apply_record_h(f->model->params_memory, rec, len, 12, 768, MOE_E, MOE_H,
                              f->model->param_sizes, f->moe) != 0) {
        f->err = 1;
        return -1;
    }
    return 0;
}

//! The newest APPROVED record's head, which is the candidate's prev_head. Scans down from
//! index 0 -- the newest slot block -- until a block yields a solution; blocks whose
//! verdict was not valid return 0 and are skipped, exactly as the fold skips them.
//!
//! Fetched before the fold on purpose. It costs one extra I/O call and one record's bytes,
//! and it buys back the property that a record with a bad proof is rejected for a few
//! hashes instead of a 1.0 GiB build plus a whole-chain fold.
static int btmv_prev_head(const btmchain* ch, unsigned char* buf, size_t cap,
                          unsigned char out[32]) {
    int n = btmchain_count(ch);
    for (int i = 0; i < n; i++) {
        int sz = btmchain_solution(ch, i, buf, cap);
        if (sz > 0) {
            if (sz < 32) return 0;      /* too short to carry a head: not a record */
            memcpy(out, buf, 32);
            return 1;
        }
        if (sz < 0 && sz != BTMCHAIN_NO_BLOCK) return -1;
    }
    return 0;                            /* no approved record yet: a genesis candidate */
}

__attribute__((export_name("verify")))
int verify(const unsigned char* anchor_hash,
           const unsigned char* payout, uint32_t payout_len,
           uint32_t nbits,
           const unsigned char* nonce, uint32_t nonce_len,
           unsigned char* out_ab)
{
    btmv_put_ab(out_ab);                 /* on every path, before anything can fail */

    // The payout is bound by the solution's own seq=0 chunk and by the seed the node
    // derives from it; this algo does not hash it, so the record is the same whoever is
    // being paid. Named rather than silently ignored.
    (void)payout; (void)payout_len;

    if (nonce_len == 0 || nonce_len > BTMV_RECORD_MAX) return 1;

    btmchain ch;
    if (btmchain_open(&ch) != 0) return 1;

    unsigned char* buf = (unsigned char*)btm_alloc(BTMV_RECORD_MAX);
    if (!buf) return 1;

    unsigned char prev_head[32];
    int ph = btmv_prev_head(&ch, buf, BTMV_RECORD_MAX, prev_head);
    if (ph < 0) return 1;
    if (ph == 0) memcpy(prev_head, anchor_hash, 32);   /* first record of the chain */

    // Proof first, model second: everything btmv_proof_window checks is in the record.
    int vin[BTM_CORPUS_SCORE_T], vtar[BTM_CORPUS_SCORE_T];
    if (btmv_proof_window(nonce, nonce_len, prev_head, vin, vtar) != 0) return 1;

    // The base state: the fixed-seed init with every approved record folded in. Records
    // arrive one at a time and are discarded after being applied, because the state
    // (1.0 GiB) dwarfs anything a block can carry.
    GPT2 model;
    size_t nb = gpt2_init_trunk_params(&model);
    if (nb == 0) return 1;               /* does not fit under WASM_MAX_MEMORY_PAGES */
    size_t moe_offsets[BTMMOE_NUM_TENSORS];
    btmmoe_tensor_offsets(model.param_sizes, moe_offsets);
    memset(model.params_memory, 0, nb);
    btmmoe_init_base_params(model.params_memory, model.param_sizes, moe_offsets,
                            12, 768, model.config.vocab_size, MOE_E, MOE_H, BTM2_BASE_SEED);

#ifdef BTMV_MEASURE_INIT_ONLY
    // MEASUREMENT BUILDS ONLY: stop after the base init so its gas cost can be separated
    // from the fold and the two forwards. Never build the deployed module with this.
    return 1;
#endif
    btmv_fold_ctx fc;
    fc.model = &model; fc.moe = moe_offsets; fc.err = 0;
    if (btmchain_fold(&ch, 0, buf, BTMV_RECORD_MAX, btmv_fold_one, &fc) < 0 || fc.err)
        return 1;

    // The gate's difficulty, from how many of the last BTM_GATE_WINDOW slot blocks
    // carried a valid solution. One I/O call per probe and zero bytes, which is why the
    // rule counts verdicts rather than timing blocks.
    uint32_t window = 0;
    uint32_t valid = btmchain_count_valid(&ch, BTM_GATE_WINDOW, &window);
    float gate_g = btmmoe_gate_fraction(valid, window);

    float base_loss = 0.0f, val_loss = 0.0f;
    return btmv_score(&model, nonce, nonce_len, vin, vtar, nbits, gate_g,
                      &base_loss, &val_loss);
}
#endif /* BTM_ONCHAIN */
