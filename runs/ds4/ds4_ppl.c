/* ds4_ppl — teacher-forced perplexity over a raw text file.
 *
 * Loads the model once (sidecar via DS4_Q1_0_EXPERT_SIDECAR + friends, exactly
 * like ds4_server), then for each token reads the logprob the model assigns to
 * the ACTUAL next token and advances.  This is the correct base-vs-Q1 quality
 * metric: butterfly-free, unlike greedy string comparison.
 *
 * Usage: ds4_ppl -m MODEL.gguf [-c CTX] [-t THREADS] TEXTFILE
 */
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    if (out_len) *out_len = (long)rd;
    return buf;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *text_path = NULL;
    int ctx_size = 4096;
    int n_threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(a, "-c") && i + 1 < argc) ctx_size = atoi(argv[++i]);
        else if (!strcmp(a, "-t") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else text_path = a;
    }
    if (!model_path || !text_path) {
        fprintf(stderr, "usage: ds4_ppl -m MODEL.gguf [-c CTX] [-t THREADS] TEXTFILE\n");
        return 2;
    }

    ds4_engine_options opt = {0};
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = n_threads;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "ds4_ppl: failed to open model %s\n", model_path);
        return 1;
    }

    long len = 0;
    char *text = read_file(text_path, &len);
    if (!text) {
        fprintf(stderr, "ds4_ppl: cannot read text file %s\n", text_path);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_tokens toks = {0};
    ds4_tokenize_text(engine, text, &toks);
    free(text);
    if (toks.len < 2) {
        fprintf(stderr, "ds4_ppl: need >=2 tokens (got %d)\n", toks.len);
        ds4_tokens_free(&toks);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4_ppl: session create failed\n");
        ds4_tokens_free(&toks);
        ds4_engine_close(engine);
        return 1;
    }

    char err[160];
    /* Seed the session with a small prefix (single-token prefill is an
     * untested edge case); score from the seed boundary onward, which is the
     * standard perplexity context-warmup convention. */
    int seed = 8;
    if (seed > toks.len - 1) seed = toks.len - 1;
    if (seed < 1) seed = 1;
    ds4_tokens prefix = {0};
    for (int j = 0; j < seed; j++) ds4_tokens_push(&prefix, toks.v[j]);
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4_ppl: sync failed: %s\n", err);
        ds4_tokens_free(&prefix);
        ds4_tokens_free(&toks);
        ds4_session_free(session);
        ds4_engine_close(engine);
        return 1;
    }
    ds4_tokens_free(&prefix);

    int limit = toks.len;
    if (ctx_size > 0 && limit > ctx_size) limit = ctx_size;
    double nll = 0.0;
    int counted = 0;
    for (int i = seed; i < limit; i++) {
        ds4_token_score sc = {0};
        /* NB: token_logprob returns 1 on success, 0 on failure (opposite of
         * sync/eval which are 0=ok). */
        if (ds4_session_token_logprob(session, toks.v[i], &sc) == 0) {
            fprintf(stderr, "ds4_ppl: token_logprob failed at %d\n", i);
            break;
        }
        nll += -(double)sc.logprob;
        counted++;
        if (ds4_session_eval(session, toks.v[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4_ppl: eval failed at %d: %s\n", i, err);
            break;
        }
        if ((i % 64) == 0) {
            fprintf(stderr, "\r  ppl progress %d/%d nll/tok=%.4f", i, limit - 1, nll / counted);
        }
    }
    fprintf(stderr, "\n");
    double avg = counted > 0 ? nll / (double)counted : 0.0;
    printf("perplexity: %.6f  avg_nll: %.6f  tokens: %d\n", exp(avg), avg, counted);

    ds4_tokens_free(&toks);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
