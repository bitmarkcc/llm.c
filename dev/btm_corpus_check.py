#!/usr/bin/env python3
"""Drive dev/btm_corpus_check.c over many corpus shapes and compare with the Python build.

The shapes that matter are the ODD ones -- a level with an odd node count promotes its
last node, and the position of that promoted node differs per level. An implementation
that duplicates instead of promoting, or that mis-tracks which levels pair, agrees with
a correct one on every power-of-two leaf count and disagrees on most others. So the leaf
counts below deliberately include odd numbers, powers of two, and one-off-from-powers.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from btm_corpus_root import LEAF_BYTES, build, h_leaf, proof  # noqa: E402

BIN = os.path.join(os.path.dirname(__file__), "btm_corpus_check")


def run(*args):
    r = subprocess.run([BIN, *[str(a) for a in args]], capture_output=True, text=True)
    if r.returncode != 0:
        return f"FAIL({r.returncode}): {r.stdout.strip()}{r.stderr.strip()}"
    return r.stdout.strip()


def corpus(n_leaves, seed=1):
    """Deterministic pseudo-random token bytes for n_leaves whole leaves."""
    import hashlib
    out = bytearray()
    h = hashlib.sha256(bytes([seed])).digest()
    while len(out) < n_leaves * LEAF_BYTES:
        h = hashlib.sha256(h).digest()
        out += h
    return bytes(out[:n_leaves * LEAF_BYTES])


def main():
    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not built -- cc -O2 -I. -o dev/btm_corpus_check dev/btm_corpus_check.c")

    ok = fail = 0

    def check(label, got, want):
        nonlocal ok, fail
        if got == want:
            ok += 1
        else:
            fail += 1
            print(f"  MISMATCH {label}\n    python: {want}\n    c:      {got}")

    # depth
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 1000, 156250000):
        want = str(max(0, (n - 1).bit_length()))
        check(f"depth({n})", run("depth", n), want)

    # fold: every leaf of several awkward shapes, plus spot checks on large ones
    for n_leaves in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 31, 32, 33):
        toks = corpus(n_leaves)
        root, nl, leaves = build(toks)
        assert nl == n_leaves
        for idx in range(n_leaves):
            leaf = toks[idx * LEAF_BYTES:(idx + 1) * LEAF_BYTES]
            sibs = proof(leaves, idx)
            got = run("fold", idx, n_leaves, leaf.hex(), *[s.hex() for s in sibs])
            check(f"fold(n={n_leaves}, i={idx})", got, root.hex())

    # index derivation, including the empty-delta edge
    for prev in (b"\xff" * 32, bytes(range(32))):
        for delta in (b"", b"\x01", bytes(range(100))):
            for n in (1, 7, 156250000):
                import hashlib
                h2 = hashlib.sha256(hashlib.sha256(prev + delta).digest()).digest()
                want = str(int.from_bytes(h2[:8], "little") % n)
                got = run("index", prev.hex(), delta.hex(), n)
                check(f"index(prev={prev[0]:02x}.., |d|={len(delta)}, n={n})", got, want)

    print(f"\n{ok} agreed, {fail} mismatched")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
