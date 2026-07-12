# Bitmark Proof-of-Useful-Work: LLM Training (btm branch)

Status notes for the `btm` branch of this fork (bitmarkcc/llm.c). Last worked on
August 2024; this document reconstructs the design so the project can be picked
up again. Written 2026-07-10 by Claude-Fable, last updated 2026-07-12.

## Goal

Prototype a "proof of useful work" mining algorithm for the Bitmark protocol:
in addition to regular hash-based mining, miners solve *dynamic* PoW algorithms
whose task is finding neural-network weights that lower a training loss. The
dynamic algorithms are delivered on-chain as wasm code via `OP_PUSHCODE`
(see the companion repo `~/git/bitmark-dev`, branch `dev2024`), voted on, and
then executed by all nodes.

`OP_PUSHCODE` is opcode `0xb2`, aliased to `OP_NOP3` — the standard soft-fork
deployment trick (same as CHECKLOCKTIMEVERIFY) — gated behind a
`SCRIPT_VERIFY_PUSHCODE` flag. Fork activation and unit tests exist on
`dev2024`.

## The two programs

### `train_gpt2_btm.c` — the miner

- Trains a GPT-2 in ordinary fast `float` arithmetic (CPU, forked from
  `train_gpt2.c`). Model size is selected by depth: 6 → 30M "tiny",
  12 → 124M, 24 → 350M, 36 → 774M, 48 → 1558M.
- Does **not** train all weights. Each round, a random subset of
  `n_active_weights` (default 62,000) is selected by `rng_seed_1` and
  initialized by `rng_seed_2` (both default to unix time + sweep index).
  All other weights come from the base init plus the checkpoint chain.
- Runs an 8×8 grid of `(seed_1, seed_2)` pairs = 64 training attempts. Each
  attempt does 30 AdamW steps at B=4, T=64 on tiny_shakespeare, then computes
  a validation loss. The attempt with the lowest val loss wins.
- Appends the winning `weight_state` record to `btm-cp.bin` (the prototype
  "blockchain" file). Subsequent runs build on top of it.

Usage: `./train_gpt2_btm [block_hash_hex] [depth] [n_active_weights] [rng_seed_offset]`

### `eval_gpt2_btm.c` — the verifier

- Same GPT-2 forward pass, but every `float` replaced by `pfloat` (see below),
  so any node computes the exact same loss bit-for-bit.
- Rebuilds the full model from the base init plus *all* records in
  `btm-cp.bin` applied in order, then scores the **last** record: reproduces
  the hash → validation-batch derivation and computes the canonical loss for
  the committed weights.
- Params live in a `std::vector<pfloat>` (each pfloat is a heap-allocated MPFR
  object), so the memory/speed cost of multiprecision is paid only during
  verification, and only for one forward pass on a tiny batch.

## Chain structure: `btm-cp.bin`

Append-only file of `weight_state` records, each ~500 KB for 62,000 weights:

```
<32-byte block hash (prev record's hash, or 0xff*32 for genesis)>
<8-byte little-endian weight count N>
<N × (uint32 weight index, float32 weight value)>
```

Hashing is Bitcoin-style double SHA-256 of the entire record. Each record's
first 32 bytes are the *previous* record's hash, so records form a hash chain.
Weights accumulate across records: the network's shared model improves round
over round while each block carries only a sparse delta.

## Commit-then-evaluate (anti-overfitting)

The validation batch is not known in advance. After training, the full
`weight_state` (including the exact float bit patterns of the trained weights)
is double-SHA-256 hashed, and that hash seeds the val loader's shuffle RNG
(`manual_seed(&val_loader.shuffle_rng, hash)`), which selects the validation
batch. A miner must therefore commit to weights *before* learning which data
it will be scored on — a Fiat–Shamir-style construction that prevents
overfitting to a known validation set, while keeping batch selection
deterministic and reproducible by any verifier.

Currently only the first 4 bytes of the hash are used as the seed (marked
`todo: use full 32 bytes` in the code).

## The determinism problem and its solution: `pfloat`

The blocker this project existed to solve: consensus requires every node to
agree on the exact loss for a given set of weights, and hardware floating
point is not bit-reproducible across platforms (compiler optimizations, FMA
contraction, x87 vs SSE, libm differences).

The solution splits the problem:

- **Training doesn't need determinism.** Miners train in fast native `float`;
  nobody has to agree on how the weights were found. The weights are committed
  as exact 32-bit patterns in the weight_state.
- **Only evaluation needs determinism.** The verifier's single forward pass
  runs entirely in `pfloat`, defined in `llmc/pfloat.h` as
  `boost::multiprecision::number<mpfr_float_backend<8>>` — an MPFR float with
  8 decimal digits of precision (`pdouble` = 17 digits). MPFR guarantees
  correctly-rounded, bit-identical results on any hardware/compiler/libm, so
  the `pfloat` loss is the canonical, consensus-grade value.

The miner's own `float` val loss is only a guide for picking its best attempt;
it is close to but not identical to the canonical `pfloat` loss.

`test2.c` is a small sanity check for pfloat arithmetic/sizes. Makefile
targets: `train_gpt2_btm` (links `-lcrypto`), `eval_gpt2_btm` (C++, links
`-lcrypto -lmpfr`, defines `LLMC_PFLOAT`), `test2` (links `-lmpfr`).
`llmc/rand.h` gained `LLMC_PFLOAT`-guarded pfloat variants of the RNG helpers
(`randpfloat32`, `pnormal_` etc.) so init can also be reproduced in
multiprecision.

## State as of August 2024

The mine → append record → verify-deterministically loop works end to end.
Last substantive commits (2024-07-31): use the hash of the weight state as
the "block hash" for the next iteration; use the previous record's hash when
evaluating a checkpoint record; fix the offset when copying checkpoint data
into the weight_state. Then cleanup and an upstream merge (2024-08-02).

## Updates 2026-07

- **Verified end to end against a real Bitmark block hash.** Mined a record
  anchored to block hash `b127cb60...` (fresh chain, GPT-2 124M, 62,000
  active weights), then independently reproduced its chained hash and
  hash-selected validation batch in the pfloat verifier:
  `./eval_gpt2_btm <block_hash_hex>`.
- **Fixed O(n²) active-weight selection** (`train_gpt2_btm.c`): the
  uniqueness check during selection now uses the `params_memory_active`
  bitmap instead of linearly scanning the `active_weights` array. The RNG
  draw sequence is untouched, so selection is bit-identical for any seed;
  startup drops from ~2 billion comparisons per attempt to ~62k lookups.
- **Fixed genesis prev-hash in the verifier** (`eval_gpt2_btm.c`):
  `gpt2_eval` accepted a `block_hash` argument but never used it, assuming a
  `0xff*32` prev-hash for the first record. That mismatched any training run
  anchored to a real block hash, producing a different val batch than the
  miner committed to. The first record now uses the block hash passed on the
  command line (later records were already correct — their prev-hash comes
  from the preceding record in the file).
- **MPFR verification is allocation-bound, not arithmetic-bound.** On musl
  (mallocng allocator), a 124M eval took ~50 min using only ~2.6 of 8 cores —
  the OpenMP threads were serialized on allocator locks by per-operation MPFR
  temporaries. Preloading mimalloc cut it to ~5 min (~10x); the Makefile now
  links `-lmimalloc` into `eval_gpt2_btm` permanently. Peak RAM ~12 GB
  (every pfloat is a heap-allocated MPFR object). Implication: a
  buffer-reusing or fixed-point evaluator should get most of the remaining
  headroom, and wasm runtimes (where the allocator is ours to choose) keep
  this win.

## Open ends / next steps

- **No difficulty or acceptance rule.** The miner just appends its
  best-of-64 result; there is no "loss must beat X" threshold tied to block
  validity. That logic belongs on the Bitmark side, defined against the
  canonical `pfloat` loss.
- **Seed the val RNG with the full 32-byte hash**, not just 4 bytes.
- **Wasm packaging.** The `pfloat` evaluator needs to compile to wasm so
  `OP_PUSHCODE` can carry it; no wasm exists yet in either repo. Wasm would
  also strengthen determinism (its integer ops are fully deterministic).
- **Scaling.** MPFR evaluation of a 124M model is slow and memory-heavy
  (every `pfloat` is a heap allocation), and ~500 KB of weight data per block
  is a lot of chain space. Largely mitigated for speed by mimalloc (~5 min,
  see Updates 2026-07), but RAM is still ~12 GB. Consider fixed-point/integer
  evaluation, fewer active weights, quantized deltas, or committing only a
  hash of weights on-chain with data availability handled elsewhere.
- **Val set size.** Verification currently scores a single B=4, T=64 batch
  (`val_num_batches = 1`); one tiny batch is a noisy measure of loss and
  may need to grow for the score to be meaningful.
