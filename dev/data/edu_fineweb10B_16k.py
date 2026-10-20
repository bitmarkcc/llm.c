"""
FineWeb-Edu sample-10BT tokenized for the btm trunk chain (V=16384).

dev/data/fineweb.py cannot be used for the trunk: it tokenizes with GPT-2's tiktoken,
whose 50257 ids do not fit a V=16384 model. So this trains a 16384-entry byte-level BPE
on the corpus itself and tokenizes the corpus with it.

BOTH OUTPUTS ARE CONSENSUS ARTIFACTS of the trunk chain. The merge table decides what
every token id MEANS, and the shard set is what dev/btm_corpus_root.py commits to, so
changing either changes what the work IS -- a new algo branch and a new vote. The
tokenizer that produced a chain must be kept with it; a chain cannot be verified without
the root, and the root cannot be reproduced without the tokenizer.

    python3 dev/data/edu_fineweb10B_16k.py              # both phases, resumable
    python3 dev/data/edu_fineweb10B_16k.py --phase bpe  # just train the tokenizer

WHY PER-PARQUET-FILE. sample-10BT is 14 parquet files. Streaming the whole split in one
pass cannot be resumed -- a network failure five hours in means starting over, because an
IterableDataset cannot seek. Processing one file at a time gives 14 checkpoints and bounds
disk to one file (~1.4 GB) plus the output, instead of ~19 GB of parquet cache.

LEAF ALIGNMENT, which is consensus-relevant. The corpus commitment treats the corpus as
the concatenation of these shards in filename order, cut into 64-token leaves. If a shard
held a token count that were not a multiple of 64, leaves would straddle shard boundaries
and a miner could not read a leaf with one seek. SHARD_TOKENS is therefore a multiple of
64, and every shard but the last holds exactly that many -- asserted on write. The
in-progress shard is carried across files in partial.bin precisely so this holds; writing
each file's leftovers as a short shard would break it.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import ByteLevelBPETokenizer

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT"
VOCAB_SIZE = 16384
# 100M tokens/shard: 200 MB each, under the 2**31 token cap write_datafile enforces, and
# 100,000,000 / 64 = 1,562,500 leaves exactly, so leaf i lives wholly in one shard.
SHARD_TOKENS = 100_000_000
LEAF_TOKENS = 64

OUT_DIR = os.path.join(os.path.dirname(__file__), "edu_fineweb10B_16k")
TOK_PATH = os.path.join(OUT_DIR, "tokenizer16k.json")
PROGRESS = os.path.join(OUT_DIR, "progress.json")
PARTIAL = os.path.join(OUT_DIR, "partial.bin")
PARQUET_DIR = os.path.join(OUT_DIR, "parquet")

assert SHARD_TOKENS % LEAF_TOKENS == 0, "shard size must be a whole number of leaves"


def shard_path(i):
    return os.path.join(OUT_DIR, f"edu_fineweb16k_{i:06d}.bin")


def write_shard(path, toks, is_last):
    """llm.c .bin: 256 int32 header then uint16 tokens. Matches data_common.write_datafile
    and what llmc/dataloader.h validates (magic 20240520, version 1)."""
    assert len(toks) < 2 ** 31, "token count overflows the int32 header field"
    assert is_last or len(toks) == SHARD_TOKENS, "only the last shard may be short"
    assert is_last or len(toks) % LEAF_TOKENS == 0, "shard must be a whole number of leaves"
    assert toks.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())
    os.replace(tmp, path)   # atomic: a killed run never leaves a half shard behind
    print(f"  wrote {path} ({len(toks):,} tokens)")


def parquet_files():
    fs = [f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(PREFIX)]
    return sorted(fs)


def texts_from(path, batch_size=512):
    """Yield lists of document texts, one row-group batch at a time (memory-bounded)."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        yield batch.column("text").to_pylist()


def fetch(fname):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    return hf_hub_download(REPO, fname, repo_type="dataset", local_dir=PARQUET_DIR)


# ----------------------------------------------------------------------------
# phase 1: train the 16k byte-level BPE

def train_tokenizer(max_chars):
    files = parquet_files()
    if not files:
        raise SystemExit(f"no files under {PREFIX} in {REPO}")
    # One file is 1/14 of the corpus; a 16k merge table converges on far less than that,
    # and this file is needed for phase 2 anyway so the download is not wasted.
    path = fetch(files[0])

    def it():
        seen = 0
        for texts in texts_from(path):
            for t in texts:
                yield t
                seen += len(t)
            if seen >= max_chars:
                print(f"  BPE sample: {seen:,} chars")
                return

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(it(), vocab_size=VOCAB_SIZE, min_frequency=2,
                            special_tokens=["<|endoftext|>"])
    os.makedirs(OUT_DIR, exist_ok=True)
    tok.save(TOK_PATH)
    print(f"saved {TOK_PATH} (vocab {tok.get_vocab_size()})")
    return tok


# ----------------------------------------------------------------------------
# phase 2: tokenize every file into fixed-size shards
#
# WHY PROCESSES AND NOT encode_batch. tokenizers parallelises encode_batch internally with
# rayon, but its threads contend with the GIL that the surrounding Python loop holds, and
# the result is pathological: 521% CPU for 1.28 MB/s, barely more than the 1.28 MB/s one
# process manages alone. Eight fork()ed workers with TOKENIZERS_PARALLELISM=false reach
# 5.67 MB/s on the same machine and the same input, with byte-identical output -- 2 hours
# for the corpus instead of 20. Measured, both ways, before choosing.

_TOK = None
_EOT = None


def _worker_init(tok_path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # rayon off: this process IS the unit
    global _TOK, _EOT
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tok_path)
    _EOT = _TOK.token_to_id("<|endoftext|>")


def _encode_docs(texts):
    """-> uint16 array holding <eot> doc <eot> doc ... for this batch of documents."""
    encs = _TOK.encode_batch(texts)
    total = sum(len(e.ids) + 1 for e in encs)
    a = np.empty(total, dtype=np.int32)
    i = 0
    for e in encs:
        a[i] = _EOT
        i += 1
        ids = e.ids
        a[i:i + len(ids)] = ids
        i += len(ids)
    # Range-checked as int32 BEFORE the cast: an id of 65536+ would wrap silently in
    # uint16 and write a plausible-looking wrong token into a consensus artifact.
    if total and not (a.min() >= 0 and a.max() < VOCAB_SIZE):
        raise ValueError(f"token id outside the {VOCAB_SIZE} vocab")
    return a.astype(np.uint16)


def _grouped(it, n):
    """Yield lists of up to n items. Bounds how far the pool runs ahead of the writer."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {"files_done": 0, "shard_index": 0, "n_tokens_total": 0, "n_chars_total": 0}


def save_progress(p, buf, count):
    with open(PARTIAL, "wb") as f:
        f.write(buf[:count].tobytes())
    p["partial"] = int(count)
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS)


def tokenize_all(nprocs):
    if not os.path.exists(TOK_PATH):
        raise SystemExit(f"missing {TOK_PATH}: run --phase bpe first")

    files = parquet_files()
    p = load_progress()
    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    count = 0
    if p.get("partial"):
        prev = np.fromfile(PARTIAL, dtype=np.uint16)
        assert len(prev) == p["partial"], f"{PARTIAL} disagrees with progress.json"
        buf[:len(prev)] = prev
        count = len(prev)
        print(f"resuming: {p['files_done']}/{len(files)} files, shard {p['shard_index']}, "
              f"{count:,} tokens carried")

    t_start = time.time()
    chars_start = p["n_chars_total"]

    def absorb(arr):
        """Append a worker's tokens, flushing whole shards as they fill."""
        nonlocal count
        n = len(arr)
        off = 0
        while off < n:
            take = min(n - off, SHARD_TOKENS - count)
            buf[count:count + take] = arr[off:off + take]
            count += take
            off += take
            if count == SHARD_TOKENS:
                write_shard(shard_path(p["shard_index"]), buf, is_last=False)
                p["shard_index"] += 1
                count = 0
                el = time.time() - t_start
                mbs = (p["n_chars_total"] - chars_start) / max(1e-9, el) / 1e6
                print(f"    {p['n_tokens_total']:,} tokens, {mbs:.2f} MB/s, "
                      f"{el / 60:.1f} min elapsed")

    # fork: workers inherit the parent, so the initializer only has to load the tokenizer.
    # Python 3.14 defaults to forkserver, which re-imports __main__ -- harmless here but
    # fork is cheaper and this module has no import-time side effects to repeat.
    ctx = mp.get_context("fork")
    with ctx.Pool(nprocs, initializer=_worker_init, initargs=(TOK_PATH,)) as pool:
        for fi in range(p["files_done"], len(files)):
            fname = files[fi]
            print(f"[{fi + 1}/{len(files)}] {fname}")
            path = fetch(fname)
            for group in _grouped(texts_from(path, batch_size=256), nprocs * 2):
                p["n_chars_total"] += sum(len(t) for b in group for t in b)
                for arr in pool.map(_encode_docs, group, chunksize=1):
                    p["n_tokens_total"] += len(arr)
                    absorb(arr)
            os.remove(path)                  # parquet is never kept once consumed
            p["files_done"] = fi + 1
            save_progress(p, buf, count)
            cpt = p["n_chars_total"] / max(1, p["n_tokens_total"])
            print(f"  {p['n_tokens_total']:,} tokens so far ({cpt:.2f} chars/token)")

    # The leftover buffer becomes the last shard ONLY once every file is consumed. Flushing
    # it at the end of a partial run would write a short shard in the MIDDLE of the corpus,
    # and then leaves straddle that boundary -- which silently breaks the one alignment
    # invariant the commitment depends on, in a consensus artifact, with every shard around
    # it looking perfectly well-formed.
    if p["files_done"] < len(files):
        save_progress(p, buf, count)
        print(f"stopped after {p['files_done']}/{len(files)} files, "
              f"{count:,} tokens carried in partial.bin -- run again to continue")
        return
    if count:
        write_shard(shard_path(p["shard_index"]), buf[:count], is_last=True)
        p["shard_index"] += 1
        count = 0
    save_progress(p, buf, 0)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)

    n = p["n_tokens_total"]
    print(f"\n{n:,} tokens in {p['shard_index']} shards "
          f"({p['n_chars_total'] / max(1, n):.2f} chars/token)")
    print(f"whole leaves: {n // LEAF_TOKENS:,}  ({n % LEAF_TOKENS} trailing tokens dropped)")
    print(f"\nNext:\n  python3 dev/btm_corpus_root.py '{OUT_DIR}/edu_fineweb16k_*.bin' \\\n"
          f"      --emit-header llmc/btmcorpus_params.h --save-leaves corpus_leaves.bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["bpe", "tokenize", "both"], default="both")
    ap.add_argument("--bpe-chars", type=int, default=300_000_000,
                    help="characters of text to train the BPE on (default 300M)")
    ap.add_argument("--procs", type=int, default=os.cpu_count(),
                    help="worker processes for tokenizing (default: all cores)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.phase in ("bpe", "both"):
        if os.path.exists(TOK_PATH):
            print(f"{TOK_PATH} exists; skipping BPE training")
        else:
            train_tokenizer(args.bpe_chars)
    if args.phase in ("tokenize", "both"):
        tokenize_all(max(1, args.procs))
