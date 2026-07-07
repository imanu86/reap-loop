/* ds4_spex_predict — implementazione loader/scorer del probe SPEX D2. Vedi .h per il formato. */
#include "ds4_spex_predict.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* fp16 -> fp32 (IEEE half, no denormal-flush; sufficiente per lo scoring). */
static inline float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) {
        if (m == 0) out = s;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
               out = s | (e << 23) | (m << 13); }
    } else if (e == 31) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; memcpy(&f, &out, 4); return f;
}

int ds4_spex_load(const char *path, ds4_spex_model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    uint32_t hdr[7];  /* version,predictor,n_layer,n_embd,n_expert,ridge(bits),reserved */
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SPX1", 4) != 0) { fclose(f); return -2; }
    if (fread(hdr, 4, 7, f) != 7) { fclose(f); return -3; }
    m->version = hdr[0]; m->predictor = hdr[1];
    m->n_layer = hdr[2]; m->n_embd = hdr[3]; m->n_expert = hdr[4];
    memcpy(&m->ridge, &hdr[5], 4); m->reserved = hdr[6];
    if (m->version != 1 || m->predictor != 2) { fclose(f); return -4; }
    size_t n = (size_t)m->n_layer * m->n_embd * m->n_expert;
    uint16_t *W = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!W) { fclose(f); return -5; }
    if (fread(W, sizeof(uint16_t), n, f) != n) { free(W); fclose(f); return -6; }
    fclose(f);
    m->W = W; m->_owned = W;
    return 0;
}

void ds4_spex_free(ds4_spex_model *m) {
    if (m && m->_owned) { free(m->_owned); m->_owned = NULL; m->W = NULL; }
}

int ds4_spex_predict_topk(const ds4_spex_model *m, uint32_t src_layer,
                          const float *h, int K, int32_t *out_ids, float *scratch) {
    if (!m || !m->W || src_layer + 1 >= m->n_layer) return 0;  /* niente L+1 */
    const uint32_t NE = m->n_embd, E = m->n_expert;
    const uint16_t *Wl = m->W + (size_t)src_layer * NE * E;
    float *sc = scratch;                 /* deve avere E elementi */
    for (uint32_t e = 0; e < E; e++) sc[e] = 0.0f;
    /* score[e] = sum_i h[i] * W[i*E + e]  (W row-major su n_embd) */
    for (uint32_t i = 0; i < NE; i++) {
        float hi = h[i];
        if (hi == 0.0f) continue;
        const uint16_t *row = Wl + (size_t)i * E;
        for (uint32_t e = 0; e < E; e++) sc[e] += hi * h2f(row[e]);
    }
    /* top-K per selezione parziale (K piccolo: selezione lineare) */
    if (K > (int)E) K = (int)E;
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = -1e30f;
        for (uint32_t e = 0; e < E; e++) if (sc[e] > bv) { bv = sc[e]; best = (int)e; }
        out_ids[k] = best; sc[best] = -1e30f;   /* rimuovi il preso */
    }
    return K;
}
