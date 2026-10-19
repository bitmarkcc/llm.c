/*
Cross-check llmc/btmcorpus.h against dev/btm_corpus_root.py.

The corpus commitment is consensus-critical and has two independent implementations: the
Python builder that produces the root, and the C folder the miner and verifier run. If
they disagree on anything -- domain separation, how an odd level promotes, the bit order
of the index -- then records a miner believes are valid are rejected by every verifier,
and nothing in either implementation alone would reveal it. So they are compared here.

Build and run:
    cc -O2 -I. -o dev/btm_corpus_check dev/btm_corpus_check.c
    python3 dev/btm_corpus_check.py            # drives it over many shapes
*/
#include "llmc/btmcorpus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int unhex(const char* s, unsigned char* out, size_t cap) {
    size_t n = strlen(s);
    if (n % 2 || n / 2 > cap) return -1;
    for (size_t i = 0; i < n / 2; i++) {
        unsigned v;
        if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (unsigned char)v;
    }
    return (int)(n / 2);
}

static void puthex(const unsigned char* b, int n) {
    for (int i = 0; i < n; i++) printf("%02x", b[i]);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  %s fold <index> <n_leaves> <leaf_hex> [sibling_hex ...]\n"
                "  %s index <prev_head_hex> <delta_hex> <n_leaves>   (prefix = prev||delta)\n"
                "  %s depth <n_leaves>\n", argv[0], argv[0], argv[0]);
        return 2;
    }

    if (!strcmp(argv[1], "fold")) {
        if (argc < 5) return 2;
        uint64_t index = strtoull(argv[2], NULL, 10);
        uint64_t n_leaves = strtoull(argv[3], NULL, 10);
        unsigned char leaf[BTMC_LEAF_BYTES];
        if (unhex(argv[4], leaf, sizeof(leaf)) != BTMC_LEAF_BYTES) {
            fprintf(stderr, "leaf must be %d bytes of hex\n", BTMC_LEAF_BYTES);
            return 2;
        }
        int n_sib = argc - 5;
        unsigned char* sibs = (unsigned char*)malloc((size_t)(n_sib ? n_sib : 1) * 32);
        for (int i = 0; i < n_sib; i++) {
            if (unhex(argv[5 + i], sibs + (size_t)i * 32, 32) != 32) return 2;
        }
        unsigned char root[32];
        int rc = btmc_fold(index, n_leaves, leaf, sibs, n_sib, root);
        if (rc != 0) { printf("ERR %d\n", rc); return 1; }
        puthex(root, 32);
        printf("\n");
        free(sibs);
        return 0;
    }

    if (!strcmp(argv[1], "index")) {
        if (argc < 5) return 2;
        // The caller hands btmc_leaf_index a PREFIX of the chaining preimage, so build
        // one here: prev_head, then the delta bytes. The real miner/verifier already has
        // this laid out contiguously and passes a pointer into the record itself.
        size_t dlen = strlen(argv[3]) / 2;
        unsigned char* buf = (unsigned char*)malloc(32 + (dlen ? dlen : 1));
        if (unhex(argv[2], buf, 32) != 32) return 2;
        if (dlen && unhex(argv[3], buf + 32, dlen) < 0) return 2;
        uint64_t n_leaves = strtoull(argv[4], NULL, 10);
        printf("%llu\n",
               (unsigned long long)btmc_leaf_index(buf, 32 + dlen, n_leaves));
        free(buf);
        return 0;
    }

    if (!strcmp(argv[1], "depth")) {
        if (argc < 3) return 2;
        printf("%d\n", btmc_depth(strtoull(argv[2], NULL, 10)));
        return 0;
    }

    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
