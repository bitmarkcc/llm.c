/*
TRUNK GEOMETRY. CONSENSUS-CRITICAL, and the single source of truth for it.

WHY THIS FILE EXISTS. These constants used to be duplicated verbatim in
eval_gpt2_btm_trunk_f32.c and train_gpt2_btm_trunk.c -- 51 near-identical lines, two
copies that had to agree exactly. They frame every record: E sizes the router entry
(out = E), V sizes the dense entries, and L and C size all of them, so a miner and a
verifier that disagree on any of it cannot walk the same chain. The failure is silent in
the worst way: records come out well-formed and simply never validate, because the two
sides hash different preimages and derive different scoring windows.

That was not theoretical. Going from L=12 to L=8 had to touch ELEVEN hardcoded `12`s
across the two files, and the miner separately carried `config.vocab_size = 16384` as a
literal while the verifier used the define. A geometry half-applied is not a smaller
model, it is a buffer overrun with a plausible-looking log.

So: change the numbers HERE, nowhere else, and regenerate BTM_GENESIS_STATE_ID
(llmc/btmmoe.h) in the same commit -- every one of these is upstream of the base weights,
so the pinned digest is the check that the change was applied consistently.
*/
#ifndef BTMTRUNK_PARAMS_H
#define BTMTRUNK_PARAMS_H

//! Experts per layer (CAPACITY: parameters, not compute).
#ifndef MOE_E
#define MOE_E 16
#endif

//! Active experts per token (COMPUTE: the forward cost scales with K, not E).
#define MOE_K 3

//! Expert hidden width.
#define MOE_H 192

//! Depth and width.
#ifndef TRUNK_LAYERS
#define TRUNK_LAYERS 8
#endif
#define TRUNK_CHANNELS 768

//! Attention heads. Independent of depth, but NOT of width: head dim is
//! TRUNK_CHANNELS / TRUNK_HEADS = 64, and a C change has to keep that whole.
#define TRUNK_HEADS 12

//! Vocabulary. Also the PADDED vocab -- the trunk has no padding slack, unlike upstream
//! llm.c's 50257 -> 50304.
//!
//! THE MOST EXPENSIVE CONSTANT HERE, by a wide margin. It is not just a model dimension:
//! the 10.7B-token corpus was tokenized with a 16384-entry ByteLevelBPE trained for it
//! (dev/data/edu_fineweb10B_16k.py), so changing V means retokenizing all 108 shards,
//! regenerating BTM_CORPUS_ROOT_HEX and BTM_CORPUS_N_LEAVES, rebuilding a 10.7 GB
//! corpus_tree.bin, and invalidating every record's Merkle proof. L/E/H/K are a
//! recompile; V is a day.
#define TRUNK_VOCAB 16384

/*
WHAT THIS GEOMETRY COSTS, measured at L=8 C=768 E=16 H=192 K=3 V=16384:

    total params        70,264,448      state 281 MB in f32
    active per token    38,633,472      of which lm_head (C*V) is 12,582,912 = 33%
    record              72 delta entries (L * (3 + 2 * TRUNK_EXPERTS_PER_BLOCK))

Against the superseded L=12 E=32 H=384 K=4: 3.83x fewer parameters but only 1.80x cheaper
per forward, because lm_head does not shrink with L, E, H or K. That asymmetry is the
thing to remember when sizing the next one -- the body responds to these knobs and the
head only responds to C and V.

WHY IT WAS SHRUNK, which is a measurement and not a preference. At L=12/E=32 the gate
could not work: at chain depth 4, 480 training steps produced a mean improvement of
0.0105 with sd 0.0269 over 19 attempts -- sd/mean = 2.57, with 5 of 19 improvements
NEGATIVE. The window noise was 2.6x the signal, so no gate threshold could separate work
from luck, and the chain stalled completely (0 of 19 passed). Worse, 480 steps took ~63
minutes against a 16-minute slot, so even that figure was measured on a miner spending
4x its budget.

Two levers follow from this geometry, and they draw on DIFFERENT budgets:
  * training steps per slot (miner wall-clock): ~1.8x more, so ~220 steps fits 16 min
  * scoring windows (verifier GAS): two forwards drop from 94.3% to ~52% of
    GAS_LIMIT[GC_SIMD], which affords about one extra window -- not the 16 that would
    cut sd by 4x. Cutting sd by f costs f^2, so multi-window scoring needs a far cheaper
    forward than this, and the lm_head floor above says that means C or V.

Whether this is ENOUGH is open. The unmeasured term is how improvement scales with model
size; small models often learn much faster per token, which would help more than the
1.8x suggests. See the calibration block in llmc/btmmoe.h for what to measure.
*/

#endif // BTMTRUNK_PARAMS_H
