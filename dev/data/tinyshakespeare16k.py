"""
Trains a 16384-entry byte-level BPE tokenizer on TinyShakespeare and tokenizes
the corpus into llm.c-format .bin files for the btm trunk chain (V=16384).

The tokenizer JSON is a CONSENSUS ARTIFACT of the trunk chain: its merge table
determines how every committed dataset tokenizes. Its file hash would be
committed on-chain alongside the dataset commitment.

Outputs (in dev/data/tinyshakespeare16k/):
  tokenizer16k.json                    the trained tokenizer (consensus artifact)
  tiny_shakespeare16k_train.bin        llm.c format: 1024B header + uint16 tokens
  tiny_shakespeare16k_val.bin          first 32768 tokens held out, as upstream does
"""
import os
import numpy as np
from tokenizers import ByteLevelBPETokenizer

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare16k")
INPUT_TXT = os.path.join(os.path.dirname(__file__), "tinyshakespeare", "tiny_shakespeare.txt")
VOCAB_SIZE = 16384

def write_datafile(filename, toks):
    # llm.c .bin format (matches dev/data/data_common.py for uint16 tokens)
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    assert (0 <= toks_np).all() and (toks_np < VOCAB_SIZE).all(), "token id out of range"
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    print(f"Saved {len(toks)} tokens to {filename}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_TXT):
        raise SystemExit(f"missing {INPUT_TXT}: run dev/data/tinyshakespeare.py first")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    tok = ByteLevelBPETokenizer()
    tok.train([INPUT_TXT], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<|endoftext|>"])
    tok_path = os.path.join(DATA_CACHE_DIR, "tokenizer16k.json")
    tok.save(tok_path)
    print(f"Saved tokenizer to {tok_path} (vocab {tok.get_vocab_size()})")

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read()
    eot = tok.token_to_id("<|endoftext|>")
    toks = [eot] + tok.encode(text).ids
    print(f"{len(toks)} tokens total")
    # same split convention as upstream tinyshakespeare.py: first 32768 tokens val
    write_datafile(os.path.join(DATA_CACHE_DIR, "tiny_shakespeare16k_val.bin"), toks[:32768])
    write_datafile(os.path.join(DATA_CACHE_DIR, "tiny_shakespeare16k_train.bin"), toks[32768:])
