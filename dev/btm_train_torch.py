"""GPU trainer for one trunk mining attempt: PyTorch trains, the C miner assembles.

WHY THIS EXISTS. train_gpt2_btm_trunk.c is plain C with OpenMP and has no GPU path --
neither does llm.c's CUDA, which implements the DENSE model and has no MoE router, experts
or LoRA. Measured on a 2013 Xeon, a record needs ~4M tokens to clear the gate reliably,
which is ~14 hours there and 1..7 minutes at 10 TFLOP/s. The design works on a GPU and
cannot work on a CPU, so the trainer has to move and the kernels are not worth porting.

WHAT MAKES THIS SAFE, and it is the whole reason a Python trainer is admissible at all: a
record carries int8 LoRA FACTORS, not a computation trace. Training determinism is
irrelevant -- quantization washes it out, and the verifier applies fixed integers in f32 on
CPU. So this may use any framework, any precision, any hardware, and need not agree with
the C code bit for bit. It only has to find factors that IMPROVE the loss the verifier
measures, which means the model it trains must be the same model STRUCTURALLY.

WHAT THIS DELIBERATELY DOES NOT DO. It does not quantize, assemble a record, compute a
corpus proof or chain a head. Every one of those is consensus format, lives in C, and has
exactly one implementation; a second copy here is the drift hazard llmc/btmtrunk_params.h
was created to remove. This writes raw f32 factors and stops.

    ./train_gpt2_btm_trunk "" 1                 # once, to produce btm-trunk-state.bin
    python dev/btm_train_torch.py --steps 2000 --batch 256
    BTM_FACTORS=btm-factors.bin ./train_gpt2_btm_trunk "" 1

ROCm NOTE (MI50 / gfx906): Vega 20 went to maintenance in ROCm 5.x and was dropped in
6.0, so stock wheels may not target it. Check `torch.cuda.is_available()` and
`torch.cuda.get_device_name(0)` before assuming; a gfx906 build may need
PYTORCH_ROCM_ARCH=gfx906 from source. --device cpu works for correctness testing.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------
# Geometry. MUST match llmc/btmtrunk_params.h -- parsed from it rather than restated,
# because two copies of the geometry is precisely the bug that header exists to prevent.

def read_geometry(path):
    want = {"MOE_E": None, "MOE_K": None, "MOE_H": None,
            "TRUNK_LAYERS": None, "TRUNK_CHANNELS": None,
            "TRUNK_HEADS": None, "TRUNK_VOCAB": None}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1] in want:
            if want[parts[1]] is None:          # first definition wins, as cpp does
                want[parts[1]] = int(parts[2])
    missing = [k for k, v in want.items() if v is None]
    if missing:
        sys.exit(f"{path}: could not find {missing}")
    return want

RANK = 4            # BTM2_RANK
MAX_SEQ_LEN = 1024  # the trunk's max_seq_len; wpe is sized for it
BASE_SEED = 20260718  # BTM2_BASE_SEED
EXPERTS_PER_BLOCK = 3  # TRUNK_EXPERTS_PER_BLOCK

# Tensor order of btmmoe_fill_param_sizes_h. The state blob is these 18 tensors
# concatenated in this order, which is what btmmoe_tensor_offsets computes as a prefix sum.
def param_sizes(g):
    L, C, E, H, V = (g["TRUNK_LAYERS"], g["TRUNK_CHANNELS"], g["MOE_E"],
                     g["MOE_H"], g["TRUNK_VOCAB"])
    return [
        V * C,            # 0  wte          (also the lm_head: weight-tied, no bias)
        MAX_SEQ_LEN * C,  # 1  wpe
        L * C,            # 2  ln1w
        L * C,            # 3  ln1b
        L * 3 * C * C,    # 4  qkvw
        L * 3 * C,        # 5  qkvb
        L * C * C,        # 6  attprojw
        L * C,            # 7  attprojb
        L * C,            # 8  ln2w
        L * C,            # 9  ln2b
        L * E * H * C,    # 10 fcw      [L][E][H][C]
        L * E * H,        # 11 fcb
        L * E * C * H,    # 12 fcprojw  [L][E][C][H]
        L * E * C,        # 13 fcprojb
        C,                # 14 lnfw
        C,                # 15 lnfb
        L * E * C,        # 16 routerw  [L][E][C]
        L * E,            # 17 routerb
    ]


def load_state(path, g, device):
    """btm-trunk-state.bin is a 32-byte tip hash followed by the raw f32 parameters."""
    sizes = param_sizes(g)
    total = sum(sizes)
    want = 32 + total * 4
    have = os.path.getsize(path)
    if have != want:
        sys.exit(f"{path}: {have} bytes, expected {want} for this geometry "
                 f"({total:,} params). Wrong geometry, or a stale cache from another one "
                 f"-- delete it and re-run the C miner once.")
    with open(path, "rb") as f:
        tip = f.read(32)
        flat = np.frombuffer(f.read(), dtype=np.float32)
    out, off = [], 0
    for n in sizes:
        out.append(torch.from_numpy(flat[off:off + n].copy()).to(device))
        off += n
    return out, tip


# ---------------------------------------------------------------------------------
# The model. Structurally identical to gpt2_forward in eval_gpt2_btm_trunk_f32.c:
# pre-LN blocks, causal attention, the FFN replaced by a top-k MoE, final LN, and an
# lm_head weight-tied to wte with no bias.

class LoRA(nn.Module):
    """W_eff = W_frozen + A @ B, with A (out, r) and B (r, in).

    That orientation is not a choice: btm2_apply_lora_int8 computes
    W[oc][ic] += sum_j A[oc*r + j] * B[j*in + ic], so A is row-major (out, r) and B is
    row-major (r, in). The C miner inits A ~ N(0, 0.02) and B to ZERO, which is what makes
    the delta start at exactly zero -- the first step cannot perturb the base model.
    """

    def __init__(self, w_frozen, r, gen):
        super().__init__()
        out_dim, in_dim = w_frozen.shape
        self.register_buffer("w0", w_frozen, persistent=False)
        self.A = nn.Parameter(torch.empty(out_dim, r, device=w_frozen.device)
                              .normal_(0.0, 0.02, generator=gen))
        self.B = nn.Parameter(torch.zeros(r, in_dim, device=w_frozen.device))

    def weight(self):
        return self.w0 + self.A @ self.B


class Frozen(nn.Module):
    """A weight with no delta. Held as a buffer so it never reaches the optimizer."""

    def __init__(self, w):
        super().__init__()
        self.register_buffer("w0", w, persistent=False)

    def weight(self):
        return self.w0


class MoEBlock(nn.Module):
    def __init__(self, g, layer, params, offs, experts, gen):
        super().__init__()
        L, C, E, H = g["TRUNK_LAYERS"], g["TRUNK_CHANNELS"], g["MOE_E"], g["MOE_H"]
        self.C, self.E, self.H, self.K = C, E, H, g["MOE_K"]
        self.NH = g["TRUNK_HEADS"]

        def slice_of(idx, shape):
            per = params[idx].numel() // L
            return params[idx][layer * per:(layer + 1) * per].view(*shape)

        self.ln1w, self.ln1b = slice_of(2, (C,)), slice_of(3, (C,))
        self.ln2w, self.ln2b = slice_of(8, (C,)), slice_of(9, (C,))
        self.qkvb, self.attprojb = slice_of(5, (3 * C,)), slice_of(7, (C,))
        self.fcb = slice_of(11, (E, H))
        self.fcprojb = slice_of(13, (E, C))
        self.routerb = slice_of(17, (E,))

        # The three dense matrices always carry a delta (types 4, 6, 16).
        self.qkvw = LoRA(slice_of(4, (3 * C, C)), RANK, gen)
        self.attprojw = LoRA(slice_of(6, (C, C)), RANK, gen)
        self.routerw = LoRA(slice_of(16, (E, C)), RANK, gen)

        # Experts: only the committed subset carries a delta (types 10, 12).
        fcw = slice_of(10, (E, H, C))
        fcprojw = slice_of(12, (E, C, H))
        self.fcw = nn.ModuleList(
            [LoRA(fcw[e], RANK, gen) if e in experts else Frozen(fcw[e]) for e in range(E)])
        self.fcprojw = nn.ModuleList(
            [LoRA(fcprojw[e], RANK, gen) if e in experts else Frozen(fcprojw[e])
             for e in range(E)])

    def forward(self, x):
        B, T, C = x.shape
        h = F.layer_norm(x, (C,), self.ln1w, self.ln1b)
        qkv = F.linear(h, self.qkvw.weight(), self.qkvb)
        q, k, v = qkv.split(C, dim=2)
        hd = C // self.NH
        q, k, v = (t.view(B, T, self.NH, hd).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + F.linear(a, self.attprojw.weight(), self.attprojb)

        h = F.layer_norm(x, (C,), self.ln2w, self.ln2b)
        logits = F.linear(h, self.routerw.weight(), self.routerb)
        # btmmoe_topk_gates: take the top K, then softmax over ONLY those. The C code
        # sorts the selected ascending before the softmax; softmax is permutation
        # invariant over the gate-expert pairing, so the set is all that matters.
        gv, gi = torch.topk(logits, self.K, dim=-1)
        gates = torch.softmax(gv, dim=-1)

        out = torch.zeros_like(h)
        # Dense-per-expert dispatch: every expert processes every token and the gate is
        # zero where it was not selected. Wasteful at E=16 but it keeps the whole batch in
        # one matmul per expert, which is what a GPU wants; a gather/scatter dispatch is
        # the optimisation if E grows.
        for e in range(self.E):
            w = (gates * (gi == e)).sum(-1, keepdim=True)
            if not bool((w > 0).any()):
                continue
            fh = F.gelu(F.linear(h, self.fcw[e].weight(), self.fcb[e]), approximate="tanh")
            out = out + w * F.linear(fh, self.fcprojw[e].weight(), self.fcprojb[e])
        return x + out


class Trunk(nn.Module):
    def __init__(self, g, params, offs, sub, gen):
        super().__init__()
        self.g = g
        C, V = g["TRUNK_CHANNELS"], g["TRUNK_VOCAB"]
        self.register_buffer("wte", params[0].view(V, C), persistent=False)
        self.register_buffer("wpe", params[1].view(MAX_SEQ_LEN, C), persistent=False)
        self.register_buffer("lnfw", params[14].view(C), persistent=False)
        self.register_buffer("lnfb", params[15].view(C), persistent=False)
        self.blocks = nn.ModuleList(
            [MoEBlock(g, l, params, offs, sub[l], gen) for l in range(g["TRUNK_LAYERS"])])

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.wte[idx] + self.wpe[:T]
        for blk in self.blocks:
            x = blk(x)
        x = F.layer_norm(x, (x.shape[-1],), self.lnfw, self.lnfb)
        logits = F.linear(x, self.wte)        # weight-tied, no bias
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


# ---------------------------------------------------------------------------------
# Data. The shards are llm.c .bin: 256 int32 header (magic 20240520, version 1, ntok)
# then uint16 tokens.

class Shards:
    def __init__(self, pattern, device):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            sys.exit(f"no corpus shards match {pattern!r}")
        self.device = device

    def batches(self, B, T, steps, seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.files))
        fi = 0
        toks, pos = None, 0
        for _ in range(steps):
            need = B * T + 1
            while toks is None or pos + need > len(toks):
                path = self.files[order[fi % len(order)]]
                fi += 1
                header = np.fromfile(path, dtype=np.int32, count=256)
                if header[0] != 20240520 or header[1] != 1:
                    sys.exit(f"{path}: not an llm.c .bin shard")
                toks = np.fromfile(path, dtype=np.uint16, offset=1024)
                pos = int(rng.integers(0, max(1, len(toks) - need)))
            buf = torch.from_numpy(toks[pos:pos + need].astype(np.int64))
            pos += B * T
            x = buf[:-1].view(B, T).to(self.device, non_blocking=True)
            y = buf[1:].view(B, T).to(self.device, non_blocking=True)
            yield x, y


# ---------------------------------------------------------------------------------
# Factor file. Carries the expert SELECTION as well as the factors, because which experts
# a record commits is the miner's free choice (the record commits values, not the
# selection procedure) -- so the choice has to travel from whoever made it to whoever
# writes the entries. The C side validates every dimension against its own geometry
# before trusting a byte of this.
#
#   magic "BTMF" | version u32 | L u32 | n_mat u32 | r u32
#   n_mat x { type u8, layer u8, expert i16, out u32, in u32 }
#   f32 A concatenated in entry order
#   f32 B concatenated in entry order

FACTOR_MAGIC = b"BTMF"
FACTOR_VERSION = 1


def entry_list(g, sub):
    """Canonical entry order, matching gpt2_train_trunk: per layer qkvw(4), attprojw(6),
    routerw(16), then each committed expert in ascending slot order as (fcw 10, fcprojw 12).
    """
    L, C, E, H = g["TRUNK_LAYERS"], g["TRUNK_CHANNELS"], g["MOE_E"], g["MOE_H"]
    out = []
    for l in range(L):
        out.append((4, l, -1, 3 * C, C))
        out.append((6, l, -1, C, C))
        out.append((16, l, -1, E, C))
        for e in sub[l]:
            out.append((10, l, e, H, C))
            out.append((12, l, e, C, H))
    return out


def write_factors(path, g, sub, model):
    entries = entry_list(g, sub)
    blocks = {(4, None): "qkvw", (6, None): "attprojw", (16, None): "routerw"}
    A_parts, B_parts = [], []
    for (t, l, e, out_dim, in_dim) in entries:
        blk = model.blocks[l]
        mod = (blk.fcw[e] if t == 10 else blk.fcprojw[e] if t == 12
               else getattr(blk, blocks[(t, None)]))
        assert tuple(mod.A.shape) == (out_dim, RANK), (t, l, e, mod.A.shape)
        assert tuple(mod.B.shape) == (RANK, in_dim), (t, l, e, mod.B.shape)
        A_parts.append(mod.A.detach().float().cpu().numpy().ravel())
        B_parts.append(mod.B.detach().float().cpu().numpy().ravel())
    with open(path, "wb") as f:
        f.write(FACTOR_MAGIC)
        f.write(struct.pack("<IIII", FACTOR_VERSION, g["TRUNK_LAYERS"], len(entries), RANK))
        for (t, l, e, out_dim, in_dim) in entries:
            f.write(struct.pack("<BBhII", t, l, e, out_dim, in_dim))
        np.concatenate(A_parts).astype(np.float32).tofile(f)
        np.concatenate(B_parts).astype(np.float32).tofile(f)
    return len(entries), sum(a.size for a in A_parts), sum(b.size for b in B_parts)


# ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="llmc/btmtrunk_params.h")
    ap.add_argument("--state", default="btm-trunk-state.bin")
    ap.add_argument("--corpus", default=os.environ.get(
        "BTM_CORPUS", "dev/data/edu_fineweb10B_16k/edu_fineweb16k_*.bin"))
    ap.add_argument("--out", default="btm-factors.bin")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64, help="B; the GPU lever, see --help")
    ap.add_argument("--seqlen", type=int, default=64, help="T")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1, help="attempt seed: LoRA A init + experts")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="defaults to BTM2_BASE_SEED; the C miner derives it from "
                         "block_hash so every miner at a height trains on the same tokens")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.startswith("cuda") else ""))

    g = read_geometry(args.params)
    print("geometry: " + "  ".join(f"{k}={v}" for k, v in g.items()))

    params, tip = load_state(args.state, g, dev)
    n = sum(p.numel() for p in params)
    print(f"state: {args.state}, {n:,} params, tip {tip[:8].hex()}...")

    # Expert selection. Free choice, so this just needs to be reproducible and spread --
    # it does NOT have to match the C miner's mt19937 walk, because the selection travels
    # in the factor file rather than being recomputed on the other side.
    rng = np.random.default_rng(args.seed)
    sub = [sorted(rng.choice(g["MOE_E"], EXPERTS_PER_BLOCK, replace=False).tolist())
           for _ in range(g["TRUNK_LAYERS"])]
    print(f"experts per layer ({EXPERTS_PER_BLOCK} of {g['MOE_E']}): {sub}")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    model = Trunk(g, params, None, sub, gen).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in trainable):,} "
          f"of {n:,} ({100 * sum(p.numel() for p in trainable) / n:.3f}%)")

    opt = torch.optim.Adam(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    amp = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    data = Shards(args.corpus, dev)
    data_seed = args.data_seed if args.data_seed is not None else BASE_SEED
    tok = args.steps * args.batch * args.seqlen
    print(f"train: B={args.batch} T={args.seqlen} steps={args.steps} -> {tok:,} tokens"
          f"   lr={args.lr}  dtype={args.dtype}  data seed {data_seed}")

    model.train()
    for step, (x, y) in enumerate(
            data.batches(args.batch, args.seqlen, args.steps, data_seed)):
        opt.zero_grad(set_to_none=True)
        if amp is not None:
            with torch.autocast(device_type=dev.split(":")[0], dtype=amp):
                loss = model(x, y)
        else:
            loss = model(x, y)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.6f}", flush=True)

    n_ent, nA, nB = write_factors(args.out, g, sub, model)
    print(f"\nwrote {args.out}: {n_ent} entries, {nA:,} A floats + {nB:,} B floats "
          f"({os.path.getsize(args.out):,} bytes)")
    print(f"next: BTM_FACTORS={args.out} ./train_gpt2_btm_trunk \"\" 1")
    print("  (the C miner quantizes these, assembles the record, attaches the corpus "
          "proof, and scores it with the real verifier -- this script's loss above is "
          "NOT the gate's number)")


if __name__ == "__main__":
    main()
