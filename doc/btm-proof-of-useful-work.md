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
- **wasm float verifier built and validated (2026-07-13).** `eval_gpt2_btm_f32.c` is a
  plain-C float32 port of the verifier: the layer/model code is taken verbatim from
  `train_gpt2_btm.c` (so its arithmetic is exactly the miner's `sqrtf/expf/tanhf` float
  math), the verifier logic is a float port of `eval_gpt2_btm.c`, and SHA-256 is the new
  self-contained `llmc/sha256.h` (no OpenSSL; verified against standard vectors). It
  compiles both natively and to wasm32-wasip1 (`make eval_gpt2_btm_f32` /
  `make eval_gpt2_btm_f32.wasm`; toolchain paths overridable via `WASI_CLANG`,
  `WASI_SYSROOT`, `WASI_BUILTINS`). Determinism comes from the wasm spec's exact IEEE 754
  semantics + every node running the identical module, not from the numeric type. Strict
  `-O2` only, never `-Ofast`/`-ffast-math`. Chain-format fields are read with explicit
  `uint64_t` widths because wasm32 `size_t` is 32-bit. The verifier also self-checks that
  the recomputed hash equals the record head stored in the chain file.
  Results on the machine-1 chain (record anchored to block hash `b127cb60...`):
  native and wasm outputs are byte-identical except timing -- `hash2 = 140 118 65 195`
  (same as miner and MPFR verifier), val loss `10.824298`, bits `412d3053`;
  ~17-25 s native (gcc -O2, 1 thread), ~19-20 s wasm (wasmtime 46, Cranelift), vs ~5 min
  for MPFR+mimalloc. **Cross-machine determinism CONFIRMED (2026-07-17): the same
  `.wasm` on the second machine (glibc) reproduced `val loss bits = 412d3053`
  bit-for-bit.** Different hardware, OS, and host libc; identical consensus value.
  This closes the deterministic-numerics problem that stalled the project in 2024.
  Toolchain build steps are recorded in Claude's memory (wasi-toolchain-source-build):
  clang/lld via portage with `LLVM_TARGETS="X86 WebAssembly"`, wasi-libc source-built at
  tag wasi-sdk-32, compiler-rt builtins source-built from the portage llvm distfile,
  wasmtime musl binary.
- **MPFR verification is allocation-bound, not arithmetic-bound.** On musl
  (mallocng allocator), a 124M eval took ~50 min using only ~2.6 of 8 cores —
  the OpenMP threads were serialized on allocator locks by per-operation MPFR
  temporaries. Preloading mimalloc cut it to ~5 min (~10x); the Makefile now
  links `-lmimalloc` into `eval_gpt2_btm` permanently. Peak RAM ~12 GB
  (every pfloat is a heap-allocated MPFR object). Implication: a
  buffer-reusing or fixed-point evaluator should get most of the remaining
  headroom, and wasm runtimes (where the allocator is ours to choose) keep
  this win.

## Scaling roadmap (added 2026-07-17)

Goal: grow from the current fixed 124M demonstrator toward a competitive
(billion+ param) model *incrementally on-chain*. Key facts shaping the plan:

- **Merging independently trained models does not work.** Neural nets have
  permutation symmetry (neuron k in one net has no correspondence to neuron k
  in another), so averaging/concatenating separately trained models fails.
  Merging works only between models sharing a trained common ancestor -- which
  the chain naturally provides, since all work descends from the accumulated
  weights.
- **Progressive model growth works.** Function-preserving transformations
  (Net2Net-style widening, layer stacking for depth, bert2BERT/LiGO growth
  operators) turn a trained small model into a larger one computing the same
  function, which then continues training. On-chain this is an "epoch": an
  OP_PUSHCODE vote installs a new wasm module whose model-build step
  deterministically grows the accumulated weights (e.g. depth 12 -> 24), then
  applies post-epoch records in the new index space. Determinism is preserved
  because the growth operator is ordinary arithmetic inside the module.
- **Mixture-of-Experts (MoE) is the scale endgame.** Replace each FFN with E
  expert FFNs + a router; only k (1-2) experts run per token. Total capacity
  grows with E while compute -- and therefore *verification cost* -- scales
  with ACTIVE params only. Illustrative: depth 12, C=768, E=64, k=2 gives
  ~3.7B total params but ~181M active per token, i.e. ~1.5x the current 20 s
  eval for a 30x bigger model. Established recipes: sparse upcycling (dense
  checkpoint -> MoE by replicating the FFN as experts), B-T-M (Branch-Train-Merge)
  (parallel expert training from a shared ancestor; could map to specialized
  mining or parallel sub-chains merged deterministically).
- **Honest throughput caveat.** A winning attempt currently processes ~7,680
  tokens; competitive 124M training needs ~10B tokens, i.e. >1M blocks at the
  demonstrator's per-block work. Per-block work is a protocol parameter:
  miners can apply arbitrary compute per block (the record commits results,
  not work), with difficulty expressed as required loss improvement. The open
  research problem is keeping per-block work coherent as optimization
  (block-coordinate descent plateaus if deltas are too small/greedy) and the
  inherent PoW redundancy (competing miners' discarded runs are the security
  budget, and they divide useful-work efficiency). Precedent that
  decentralized ~10B training is feasible: INTELLECT-1, DisTrO-style
  low-communication optimizers (2024-25) -- though without consensus
  verification, which is exactly what this project adds.
- **Format note.** Chain records index weights with uint32; that caps the
  parameter space at ~4.3B. Growing past it needs a record-format epoch
  (uint64 or (tensor_id, offset) indexing).

**Verification budget math (2026-07-17).** Eval cost = 2 x active_params x
val_tokens FLOPs. Wasm throughput ladder: ~3 GFLOP/s scalar (today), ~12-15
with a hand SIMD128 kernel, ~80-100 with wasip1-threads on 8 cores. So <1 s
verification caps active params at ~100-200M (256-token val batch) to
~400-800M (64 tokens; per-block score noise averages out across blocks). LM
head is a budget-eater (2 x tokens x C x V): keep C at 768-1024 and V ~32k.
A 100B-capacity/<1 s config is fine-grained MoE: small experts (hidden = C,
1-2M params), k=4-8 active, E~2000-5000/layer -- but that is ~500:1 sparsity,
beyond proven practice (~10-30:1, e.g. DeepSeek-V3 671B/37B); ultra-sparse
quality (PEER-style) is an open research bet. Fallback preserving everything
else: 5-20 s verification with conventional sparsity + the parallel/optimistic
verification machinery.

**Timeline to a trained 100B.** Training compute ~6 x active x tokens (e.g.
10B active x 2T tokens = 1.2e23 FLOP). Winner-takes-all mining makes chain
progress SERIAL (one surviving delta per block regardless of miner count):
~1+ year at a single 8-GPU miner, order 1e5-1e6 blocks. Two protocol levers:
(a) expert-sharded mining (B-T-M-style): different miners train different
(disjoint) experts in the same block, merged deterministically -- throughput
scales with lanes; MoE is what makes this possible; (b) structured deltas:
raw (index,value) records cannot move a 100B model (1.6M blocks to touch each
weight once) -- the chain should store a compressed DESCRIPTION of the change,
expanded deterministically in-module. Flagship: LoRA-style low-rank deltas --
miner trains factors A (CxR), B (RxC') directly (frozen W, delta exactly
low-rank by construction, replaces the random-active-subset scheme), verifier
applies W += A*B (trivial matmul). Numbers at 124M, r=8, all layer matrices:
~85M weights moved per ~4.7 MB record (fp32 factors; ~2.4 MB fp16) vs 62k
weights per 496 KB today -- ~1000x more weights moved per chain byte, and no
indices (structure implies position). Composable structures: quantized factors
(int8+scale, deterministic dequant), tile/range updates, sparse lists (keep
for LN/bias edits). Rejected: seed-replay deltas (verification would cost as
much as training). Record format becomes typed: <prev-hash><tensor id><delta
type><shape><payload>; hashing/commit-then-evaluate unchanged (hash the
description). Bonus: tensor-name addressing survives architecture-growth
epochs better than flat parameter indices. Concretely for GPT-2 124M: 48
(A,B) pairs (4 matrices x 12 layers; skip embeddings, keep sparse lists for
biases/LN), 147456*r factor params total; recommend r=8, fp16 factors, ~2.4 MB
records. Low per-block rank does not cap learning: rank ACCUMULATES across
blocks (sum of k rank-r deltas reaches rank k*r; full rank 768 in ~100 blocks
at r=8), with published precedent that pretraining works this way (ReLoRA,
GaLore). r, alpha, precision are per-epoch consensus parameters.
  **Bitmark chain budget (2026-07-18):** 1 MB max block, ~2 min blocks, 8 PoW
  algos + dynamic algo slot => ~1 MB per 16 min per LLM chain (shared with
  transactions, so realistically ~600-700 KB). r=8 fp16 (2.4 MB) does NOT
  fit; r=4 int8 (~590 KB) does, with full-rank accumulation in ~192 dynamic
  blocks (~2 days). Preferred structural answer: hash-only on-chain (32 B),
  delta bytes on the same off-chain data-availability layer as datasets --
  on-chain storage otherwise grows ~33 GB/year per LLM chain (vs ~50 GB/yr
  for all of Bitcoin). Add checkpoint epochs (commit full weight-state hash
  every N blocks; prune older deltas, assumeUTXO-style) to bound storage.
  Cadence math: verification duty cycle ~2% (20 s per 16 min); miner work
  quantum ~16 min/block => 10B-active/2T-token training ~= 39k dynamic blocks
  ~= 14 months serial, reinforcing domain-federation/expert-sharding as the
  parallelism levers (multiple dynamic algo slots = multiple concurrent LLM
  chains). Two distinct per-block constraints -- compute (16 min of miner GPU)
  and DELTA BANDWIDTH (one rank-r delta per block cannot express unlimited
  training progress; full-rank accumulation ~192 blocks at r=4): small models
  are bandwidth-bound (124M/10B-token: FLOP floor ~2-3 blocks but realistic
  timescale hundreds of blocks; surplus miner compute goes into best-of-many
  attempt search, buying block quality), large models are compute-bound (the
  39k-block figure). Bandwidth levers: higher r, multiple deltas per block,
  shorter dynamic cadence.
  **Transport floor (2026-07-18):** the chain/DA layer must carry at least the
  description length of the final model (every trained param transported >=
  once). 1 MB/16 min = 33 GB/yr on-chain: a 1T-total model (~1-2 TB) would
  take 30-60 years through the block pipe, and in-block r=4 deltas over 39k
  blocks carry only ~23 GB (~2% of 1T) -- in-block deltas + 1T total is
  arithmetically impossible, independent of rank. Consistent design for a
  10B-active/1T-total/2T-token trunk: hash-only on-chain + ~50 MB DA-layer
  deltas, where compute (39k blocks) and transport (2 TB / 50 MB) converge;
  realistic estimate with low-rank training tax ~1.5-2.5 years. The 14-month
  figure is the pure compute floor. In-block r=4/int8 remains correct for the
  124M demonstrator (~250 MB description vs 33 GB/yr pipe) and as the
  format-validation step. Strategic alternative: size the trunk to one pipe
  (10-30B total, months-to-a-year of transport; its job is shared substrate,
  not capacity) and reach 1T AGGREGATE via the domain-chain federation --
  parallel chains = parallel pipes is the native bandwidth-scaling mechanism.
  **Target trunk config (settled 2026-07-18):** C=768, L=12, NH=12, maxT=1024,
  V=16384 byte-level BPE; fine-grained MoE: experts H=C/2 (0.59M params), k=4,
  E~300/layer => ~69M active, ~2.2B total (32:1 sparsity, deliberately within
  proven MoE practice). Val batch 64 tokens. Verification ~2.7 s scalar wasm,
  ~0.7 s with the SIMD128 kernel (kernel required for the <1 s target).
  Delta scheme: r=4 int8 factors + per-factor fp16 scales, typed records,
  rotating per-block coverage (~25-30 experts' factors per block). Fully
  in-block trainable: transport floor 2-4 GB / 590 KB-per-16-min ~= 4k-7.5k
  dynamic blocks (~6-12 weeks), compute floor ~270 block-windows (~3 days,
  bandwidth-bound => surplus miner compute buys best-of-N block quality);
  no relay-payload/DA prerequisite -- those become upgrades for later growth
  epochs (adding experts raises total without touching verification cost).
  2T-token target is heavy overtraining for 2.2B (Chinchilla ~50-100B tokens);
  dataset epochs can extend as needed. Implementation ladder: (a) v2 typed
  records + LoRA r=4/int8 on the 124M demonstrator; (b) toy MoE (depth 6,
  E=4, k=1: upcycling epoch, router tie-break determinism, expert-aware
  records); (c) this trunk + SIMD kernel.
  **DA layer design (2026-07-18):** not a separate storage system -- the same
  node network relays delta bytes as a companion payload outside the 1 MB
  hashed block body. On-chain hash = content address: verification integrity
  is location-independent (fetch bytes from anyone, hash-check, expand, eval).
  The DA problem is WITHHOLDING, not trust; rule: nodes do not
  accept/relay/mine-on a block until its delta is fetched and hash-verified,
  so unavailable blocks orphan and withholding only hurts the withholder
  (Bitcoin's own availability model). Storage stays light: miners inherently
  hold the current weight state (needed to train); historical deltas prunable
  via on-chain weight-state checkpoints (new nodes sync from checkpoint);
  erasure coding / availability sampling only if light clients are wanted
  later.
  **Val-batch evidence (2026-07-18):** batch = B*T+1 contiguous uint16 tokens
  = 514 bytes; Merkle-per-sample-slot proof (512 B leaves = dataloader sample
  grid) = 352 B (tiny_shakespeare, depth 11) to ~832 B (10B-token dataset,
  depth 26); union epoch commits per-dataset (root, num_samples, weight) once
  per epoch. Selection index derivable from the record hash alone (spec: index
  directly from hash mod num_samples instead of materializing the mt19937
  permutation at scale). INCLUDE batch+proof in the block payload (~1.5 KB per
  batch, ~25 KB even at 16 val batches): verification then needs NO dataset
  access at all -- datasets are a mining-side requirement only, and historical
  blocks stay self-verifiable even if old dataset availability decays.

**Domain-expert federation (proposed 2026-07-17).** Alternative/complementary
to token-level MoE, and the current preferred vision: each PoUW chain trains a
complete small LM specialized to one data domain (its dataset commitment IS
its domain label), each with its own vocabulary and channel width. A separate
small router model -- large input vocab (cheap: input embeddings are lookups),
small C, output head = E domain scores instead of V logits -- maps an input to
the top-scoring domain expert. This is the B-T-M (Branch-Train-Merge) / C-BTM /
"MoErging" pattern. Fit with the chain: dissolves winner-takes-all
serialization (domain chains mine concurrently, no weight merging ever, so no
permutation problem); verification stays ~seconds per block forever; capacity
scales horizontally with number of chains; the router itself is trainable and
verifiable as another PoUW chain, with training labels already on-chain.
Costs/decisions: (a) whole-input routing caps per-query capacity at one
expert (federation of specialists != monolithic generalist on cross-domain
reasoning); (b) per-expert vocabs make outputs/losses incomparable across
experts -- routing must be winner-takes-all on raw text; never compare losses
across chains, difficulty is per-chain; (c) STRONGLY consider a shared seed
trunk (B-T-M recipe): genesis foundation chain, domain chains fork from its
committed weights -- buys sample efficiency and keeps sibling-merging possible
later. Fork mechanics: a domain chain's genesis commits to (foundation chain,
block hash X) and its model build = apply foundation records up to X, then own
records -- the trunk is referenced, not copied, using the existing
apply-records-in-order machinery. Constraints: a branch inherits the trunk's C rigidly (every matrix is
C-shaped), but vocab is SOFT: byte-level BPE gives universal coverage (new
words split into pieces -- efficiency cost, not impossibility, so the trunk
vocab need NOT be the union of specialist vocabs), vocab EXTENSION at fork is
function-preserving (append merges + wte rows initialized from constituent
pieces; requires re-tokenized dataset commitment since the merge table changes
tokenization), vocab PRUNING at fork implements limited-vocab specialists and
cuts the C x V LM-head verification cost (~1/3 of eval today), and truly alien
tokenizers can still inherit the vocab-agnostic transformer body with fresh
embeddings/head. Trunk updates after a fork do not propagate (no clean rebase:
version trunk snapshots as fork points, re-fork when worthwhile); (d) router is the quality bottleneck and needs a governance rhythm for
retraining as domains are added (may start off-chain: only its training needs
consensus, not inference-time use). Long-run synthesis: domain-expert chains
for parallelism/capacity, token-MoE inside each domain model for per-query
capacity.

Sequence: (1) prove the consensus loop at small dense scale [done]; (2) scale
per-block work via difficulty-as-loss-threshold; (3) architecture-growth
epochs via OP_PUSHCODE; (4) MoE composition, keeping verification ~constant
as capacity grows; (5) expert-sharded mining + low-rank deltas for
competitive-scale throughput; with the domain-expert federation as the
framing architecture over (3)-(5).

## v2 implementation: LoRA structured deltas (2026-07-18)

Implementation ladder step (a) COMPLETE on the 124M demonstrator:

- `llmc/btmrecord.h` -- the shared consensus-critical format header: v2 typed
  records (34 B header: 32 B chained head + u16 n_entries; entry = 4 B header
  <u16 tensor_id=(layer<<8)|type><u8 delta_type><u8 rank> + payload), fp16
  encode/decode (exhaustively round-trip tested), int8 symmetric quantization
  with fp16 scales, deterministic LoRA expansion (int32 dots, one f32
  multiply-add per weight), fixed-seed full base init (BTM2_BASE_SEED; no
  more random-subset scheme), record walk/apply. Delta type 1 = LoRA int8
  (additive, the 4 matrix kinds); type 0 = sparse overwrite list
  (slice-relative; for 1-D tensors/exact patches; not yet emitted).
- `train_gpt2_btm2.c` -- the LoRA miner: per attempt, base frozen, rank-4
  factors (A seeded per attempt, B=0) on all 4*L matrices, 30 AdamW steps on
  factors only (lr 1e-3), then quantize -> materialize the QUANTIZED delta ->
  hash -> hash-seeded val batch -> score (so the miner's loss is exactly the
  verifier's), best-of-N appended to btm-cp2.bin.
- `eval_gpt2_btm2_f32.c` (+ .wasm) -- the v2 verifier: base init + apply all
  records, recompute last record's hash from its preimage, self-check vs
  stored head, score the committed val batch.
- First v2 block mined and verified (anchored to the same Bitmark block hash
  b127cb60...): record exactly 590,242 bytes (48 entries, r=4, fits the 1 MB
  block budget), train loss 10.8 -> ~9.2 in 30 rank-4 steps (LoRA learning
  confirmed), miner best_loss 9.224760 == native verifier loss == wasm
  verifier loss, bits 4113989e, native and wasm outputs byte-identical except
  timing (~33 s native / ~37 s wasm; slower than v1's ~20 s because the model
  build now does the full fixed-seed init + record expansion).
- Misc: `llmc/rand.h` inline fns made static inline (C99 linkage fix for the
  wasm/clang build); miner grad-projection loops are single-threaded (known
  speedup TODO: OMP pragmas).

Next ladder steps: (b) toy MoE [DONE, below]; (c) the settled trunk config
(C=768, V=16384, fine experts H=C/2, k=4, E~300/layer) + SIMD128 kernel.

## Toy MoE: ladder step (b) COMPLETE (2026-07-19)

Depth 6 (C=384), E=4 experts/layer, k=2, jumbo experts (H=4C) via upcycling.
k=2 (not 1) deliberately: renormalized top-k softmax gating keeps the upcycle
EXACTLY function-preserving (identical expert copies at gates 1/2+1/2 sum to
the dense FFN) while still giving the router gradients (k=1 renormalized would
freeze it at gate=1.0).

- `llmc/btmmoe.h`: MoE 18-tensor layout (expert FFNs L*E*..., routerw L*E*C,
  routerb L*E); deterministic upcycle (replicate dense FFN into E experts,
  zero router); CONSENSUS top-k rule: ascending-index scan, strictly-greater
  wins => equal logits keep the LOWER expert index (unit-tested); gates =
  softmax over the selected k only. New record entry types: delta_type 2 =
  LoRA int8 on ONE expert's matrix (u16 expert_id in payload -- scales to the
  trunk's hundreds of experts), delta_type 3 = dense f32 slice overwrite
  (routers). Walk/apply handles all 4 types.
- `train_gpt2_btm_moe.c`: MoE forward (per-token routing, gate-weighted
  expert FFNs, OMP over tokens) + backward (serial over tokens -- expert
  weight-grad accumulation would race; through gates, softmax, router, and
  both selected experts). Build = dense fixed-seed base -> upcycle -> apply
  btm-cp-moe.bin. LoRA factors on attention + all E experts' matrices;
  routers trained directly, committed dense. Chain file btm-cp-moe.bin.
- `eval_gpt2_btm_moe_f32.c` (+ .wasm): same build path, forward-only.
- Results: upcycle preservation VERIFIED (dense and upcycled-MoE step-0 loss
  both 10.842146 on the identical batch); mined block anchored to the same
  Bitmark hash: record 461,698 bytes (72 LoRA + 12 router entries), train
  loss 10.84 -> 10.03 through routed experts, miner best_loss 10.032036 ==
  native == wasm verifier loss, bits 41208338, outputs byte-identical except
  timing (~9.6 s native / ~13.2 s wasm at depth 6).

All three chains (v1, v2, MoE) now coexist: btm-cp.bin / btm-cp2.bin /
btm-cp-moe.bin with their own program pairs.

## Open ends / next steps

- **No difficulty or acceptance rule.** The miner just appends its
  best-of-64 result; there is no "loss must beat X" threshold tied to block
  validity. That logic belongs on the Bitmark side, defined against the
  canonical loss (now: the wasm f32 verifier's output).
- **Verification stays on CPU (decided 2026-07).** GPUs are for mining only:
  GPU float results are not bit-reproducible across vendors/drivers
  (reduction order, vendor transcendentals), so they cannot define consensus.
  Verification cost is handled by parallelism instead: each record's eval is
  an independent pure function of the chain prefix, so blocks verify in
  parallel (~800 MB per concurrent eval; one wasmtime module, N instances).
  Design notes for the acceptance rule: (a) validate the cheap sequential
  hash chain first, then farm loss evals to a worker pool (or sample during
  IBD); (b) the binding constraint is latency at the TIP, not IBD throughput
  -- decide relay-before-loss-verification vs verify-then-relay, since 20-25 s
  per hop would slow block propagation and raise orphan rates.
- **Seed the val RNG with the full 32-byte hash**, not just 4 bytes.
- **Wasm packaging.** Resolved for determinism (see Updates 2026-07: verified
  bit-reproducible across two machines). Remaining: deciding the module's exact
  on-chain form for `OP_PUSHCODE` (wasip1 file I/O vs passing chain data in
  memory), and embedding a runtime (e.g. wasmtime's C API) in the Bitmark node.
- **Scaling.** MPFR evaluation of a 124M model is slow and memory-heavy
  (every `pfloat` is a heap allocation), and ~500 KB of weight data per block
  is a lot of chain space. Largely mitigated for speed by mimalloc (~5 min,
  see Updates 2026-07), but RAM is still ~12 GB. Consider fixed-point/integer
  evaluation, fewer active weights, quantized deltas, or committing only a
  hash of weights on-chain with data availability handled elsewhere.
- **Val set size.** Verification currently scores a single B=4, T=64 batch
  (`val_num_batches = 1`); one tiny batch is a noisy measure of loss and
  may need to grow for the score to be meaningful.
- **Dataset epochs and the anti-forgetting rule (2026-07-17).** When the chain
  commits an additional dataset for the same model, weights carry forward
  unchanged (new blocks keep appending deltas; knowledge grows, capacity
  doesn't). CONSENSUS RULE: the val-batch pool must be the UNION of all
  committed datasets (governance-weighted) -- if val batches came only from
  the new data, miners would be rewarded for catastrophic forgetting of old
  domains. Union-scoring makes replay-style mixed training the rational miner
  strategy (the protocol cannot see training data, only score results).
  Details to spec: mixing weights as a governance lever (possibly annealed
  after each epoch); eternal data availability incl. for historical
  re-verification (Merkle-per-batch commitments); difficulty re-anchoring at
  epoch boundaries (loss jumps when unfamiliar data enters the pool);
  capacity saturation (loss floor stalling on the growing union) as the
  trigger signal for an architecture-growth epoch.
