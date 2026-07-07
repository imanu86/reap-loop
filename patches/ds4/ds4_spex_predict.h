/* ds4_spex_predict — loader del predittore SPEX hidden->next-layer-experts (D2).
 *
 * Formato .spex (little-endian), prodotto da scripts/spex_fit_predictor.py --export-spex:
 *   Header 32B: char magic[4]="SPX1"; u32 version=1; u32 predictor=2(hidden-next-layer);
 *               u32 n_layer; u32 n_embd; u32 n_expert; f32 ridge; u32 reserved;
 *   Poi, per SOURCE layer l in [0..n_layer-1]: fp16 W[l][n_embd*n_expert] (row-major).
 *
 * Uso a inference-time (input router = ffn_norm del layer L, catturato come nella trace 0007):
 *   score(expert e per il layer L+1) = sum_i h_L[i] * W[L][i*n_expert + e]
 *   -> prendi i top-K score -> quelli sono gli expert da PREFETCHARE per L+1 (mai gating: solo loading).
 *
 * Integrazione: chiamare ds4_spex_predict_topk() nell'hook a ds4.c:19415 (dopo lo swap hidden,
 * dove g->cur_hc = hidden(L)), passare gli ID a ds4_gpu_stream_expert_cache_seed_experts_async
 * (variante async 0002/A3). Env DS4_SPEX_FILE=/path/model.spex (default OFF => nessun overhead).
 */
#ifndef DS4_SPEX_PREDICT_H
#define DS4_SPEX_PREDICT_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t version, predictor, n_layer, n_embd, n_expert, reserved;
    float ridge;
    const uint16_t *W;   /* [n_layer * n_embd * n_expert] fp16, mmap o malloc */
    void *_owned;        /* buffer da free (NULL se mmap) */
} ds4_spex_model;

/* Carica un .spex. Ritorna 0 ok, <0 errore (magic/versione/short-read). */
int  ds4_spex_load(const char *path, ds4_spex_model *m);
void ds4_spex_free(ds4_spex_model *m);

/* Predice i top-K expert del layer src_layer+1 dal hidden h (n_embd float32).
 * Scrive fino a K id in out_ids (ordinati per score desc); ritorna il numero scritto.
 * h_len deve == m->n_embd. Nessuna allocazione se scratch!=NULL (n_expert float). */
int  ds4_spex_predict_topk(const ds4_spex_model *m, uint32_t src_layer,
                           const float *h, int K, int32_t *out_ids, float *scratch);

#endif
