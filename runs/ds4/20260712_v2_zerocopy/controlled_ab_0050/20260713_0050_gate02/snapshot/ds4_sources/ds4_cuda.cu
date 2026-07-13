#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    DS4_CUDA_ROUTED_EXPERTS_PER_TOKEN = 6u,
    DS4_CUDA_STREAM_EXPERT_DEFAULT = 8u * 64u,
    DS4_CUDA_STREAM_EXPERT_MAX = 61u * 384u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

struct ds4_gpu_async_read {
    void *host;
    uint64_t bytes;
    uint64_t pending_bytes;
    cudaStream_t stream;
    cudaEvent_t ready;
    cudaEvent_t done;
    int pending;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static int g_model_mapping_failure_notice_printed;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static int g_quality_mode;
static int g_ssd_streaming_mode;

struct cuda_spex_stats {
    uint64_t expert_cache_hits;
    uint64_t expert_cache_misses;
    uint64_t selected_load_batches;
    uint64_t selected_expert_requests;
    uint64_t selected_direct_loads;
    uint64_t selected_copy_calls;
    uint64_t selected_copy_bytes;
    uint64_t selected_sync_calls;
    double selected_copy_seconds;
    double selected_sync_seconds;
};

static cuda_spex_stats g_spex_stats;
/* 0036 in-place resident counters (defined here so cuda_spex_stats_print can
 * report them; incremented by the in-place decode path far below). */
static uint64_t g_inplace_reuse_hits;
static uint64_t g_inplace_admissions;
static uint64_t g_inplace_slot_uploads;
static uint64_t g_inplace_pin_admits;   /* experts pinned at full cap (~= resident working set) */
static uint64_t g_inplace_pin_rejects;  /* admits that hit the pin-budget wall (residual churn) */
static uint64_t g_spex_pin_promotions;  /* 0044: provisional SPEX residency priority */
static uint64_t g_spex_pin_releases;
static uint64_t g_spex_pin_displacements;
static int g_spex_stats_enabled = -1;
static int g_spex_stats_atexit_registered;
static int g_spex_stats_printed;

enum {
    DS4_CUDA_TIERING_MAX_LAYER = 256u,
    DS4_CUDA_TIERING_MAX_EXPERT = 512u
};

struct cuda_expert_tiering_layer_stats {
    uint64_t batches;
    uint64_t slot_requests;
    uint64_t compact_experts;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t direct_loads;
    uint64_t evictions;
    uint64_t direct_bytes;
    uint64_t compact_bytes;
    uint32_t max_total_expert;
    uint32_t max_cache_capacity;
};

struct cuda_expert_tiering_observe_stats {
    uint64_t batches;
    uint64_t slot_requests;
    uint64_t compact_experts;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t direct_loads;
    uint64_t evictions;
    uint64_t direct_bytes;
    uint64_t compact_bytes;
    cuda_expert_tiering_layer_stats layers[DS4_CUDA_TIERING_MAX_LAYER];
    uint64_t selected_hist[DS4_CUDA_TIERING_MAX_LAYER][DS4_CUDA_TIERING_MAX_EXPERT];
};

static cuda_expert_tiering_observe_stats g_expert_tiering_observe;
static int g_expert_tiering_observe_enabled = -1;
static int g_expert_tiering_observe_atexit_registered;
static int g_expert_tiering_observe_notice_printed;
static int g_expert_tiering_observe_log_failed;
static FILE *g_expert_tiering_observe_log;

struct cuda_expert_tier_promote_stats {
    uint64_t prompt_batches;
    uint64_t decode_batches;
    uint64_t prompt_selected;
    uint64_t requests;
    uint64_t hot_hits;
    uint64_t promotions;
    uint64_t evictions;
    uint64_t resets;
    uint64_t finalized;
    uint64_t total_preloaded;
    uint64_t tick;
    uint32_t cap;
    uint32_t hot_count;
    uint32_t preloaded;
    int enabled;
    int initialized;
    int collecting_prompt;
    int preload_ready;
    int seen_any;
    int notice_printed;
    int verbose;
    int last_layer;
    uint32_t prompt_counts[DS4_CUDA_TIERING_MAX_LAYER][DS4_CUDA_TIERING_MAX_EXPERT];
    uint8_t hot[DS4_CUDA_TIERING_MAX_LAYER][DS4_CUDA_TIERING_MAX_EXPERT];
    uint64_t last_used[DS4_CUDA_TIERING_MAX_LAYER][DS4_CUDA_TIERING_MAX_EXPERT];
};

struct cuda_expert_tier_promote_batch {
    int enabled;
    int reset;
    int finalized;
    const char *phase;
    uint32_t hot_hits;
    uint32_t promotions;
    uint32_t evictions;
    uint32_t preloaded;
    uint32_t hot_count;
};

static cuda_expert_tier_promote_stats g_expert_tier_promote;

enum cuda_expert_cold_format {
    CUDA_EXPERT_COLD_FORMAT_NONE = 0,
    CUDA_EXPERT_COLD_FORMAT_LOSSLESS = 1,
    CUDA_EXPERT_COLD_FORMAT_CQ1 = 2,
};

struct cuda_expert_cold_lossless_entry {
    uint64_t key;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t blob_bytes;
    uint64_t checksum;
    uint32_t format;
    uint32_t cq1_group;
    char *blob;
};

struct cuda_expert_cold_lossless_stats {
    int initialized;
    int enabled;
    int allow_prefill;
    int allow_ungated;
    int verify;
    int verbose;
    int notice_printed;
    int atexit_registered;
    uint32_t format;
    uint32_t cq1_group;
    uint32_t native_decode_tokens;
    uint32_t native_decode_layers;
    uint64_t materialized;
    uint64_t reuses;
    uint64_t copies;
    uint64_t copy_failures;
    uint64_t materialize_failures;
    uint64_t verify_failures;
    uint64_t policy_allowed;
    uint64_t policy_skip_prompt;
    uint64_t policy_skip_no_policy;
    uint64_t policy_skip_no_hotset;
    uint64_t policy_skip_warmup;
    uint64_t policy_skip_hot;
    uint64_t policy_skip_invalid;
    uint64_t bytes_materialized;
    uint64_t bytes_repacked;
    uint64_t repack_failures;
    char *scratch;
    uint64_t scratch_bytes;
    std::vector<cuda_expert_cold_lossless_entry> entries;
    std::unordered_map<uint64_t, uint32_t> index;
};

static cuda_expert_cold_lossless_stats g_expert_cold_lossless;

static int cuda_env_truthy(const char *v) {
    if (!v || !v[0] || strcmp(v, "0") == 0 ||
        strcmp(v, "false") == 0 || strcmp(v, "off") == 0 ||
        strcmp(v, "no") == 0) {
        return 0;
    }
    return 1;
}

static uint32_t cuda_env_u32(const char *name, uint32_t dflt, uint32_t max_value) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 10);
    if (end == v || *end != '\0') return dflt;
    if (parsed > (unsigned long)max_value) return max_value;
    return (uint32_t)parsed;
}

static int cuda_expert_tiering_observe_is_enabled(void) {
    if (g_expert_tiering_observe_enabled < 0) {
        const char *mode = getenv("DS4_EXPERT_TIERING");
        g_expert_tiering_observe_enabled =
            (mode && strcmp(mode, "observe") == 0) ? 1 : 0;
        if (!g_expert_tiering_observe_enabled && cuda_env_truthy(mode)) {
            fprintf(stderr,
                    "ds4: expert tiering mode '%s' ignored; only "
                    "DS4_EXPERT_TIERING=observe is implemented\n",
                    mode);
        }
    }
    return g_expert_tiering_observe_enabled;
}

static int cuda_expert_tier_promote_is_enabled(void) {
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    if (!p->initialized) {
        memset(p, 0, sizeof(*p));
        p->initialized = 1;
        p->enabled = 0;
        p->last_layer = -1;
        const char *policy = getenv("DS4_EXPERT_TIER_POLICY");
        if (policy && (!strcmp(policy, "observe_promote") ||
                       !strcmp(policy, "promote_observe"))) {
            p->enabled = 1;
        } else if (cuda_env_truthy(policy) && policy &&
                   strcmp(policy, "observe") != 0) {
            fprintf(stderr,
                    "ds4: expert tier policy '%s' ignored; only "
                    "DS4_EXPERT_TIER_POLICY=observe_promote is implemented\n",
                    policy);
        }
        p->cap = cuda_env_u32("DS4_EXPERT_TIER_PROMOTE_CAP",
                              1024u,
                              DS4_CUDA_TIERING_MAX_LAYER *
                                  DS4_CUDA_TIERING_MAX_EXPERT);
        p->verbose = cuda_env_truthy(getenv("DS4_EXPERT_TIER_PROMOTE_VERBOSE"));
    }
    return p->enabled;
}

static void cuda_expert_tiering_observe_print_summary(void);

static void cuda_expert_tiering_observe_atexit(void) {
    cuda_expert_tiering_observe_print_summary();
    if (g_expert_tiering_observe_log) {
        fclose(g_expert_tiering_observe_log);
        g_expert_tiering_observe_log = NULL;
    }
}

static FILE *cuda_expert_tiering_observe_log_file(void) {
    if (!cuda_expert_tiering_observe_is_enabled() ||
        g_expert_tiering_observe_log_failed) {
        return NULL;
    }
    if (g_expert_tiering_observe_log) return g_expert_tiering_observe_log;
    const char *path = getenv("DS4_EXPERT_TIERING_LOG");
    if (!path || !path[0]) return NULL;
    g_expert_tiering_observe_log = fopen(path, "a");
    if (!g_expert_tiering_observe_log) {
        fprintf(stderr,
                "ds4: expert tiering observe failed to open %s: %s\n",
                path,
                strerror(errno));
        g_expert_tiering_observe_log_failed = 1;
        return NULL;
    }
    return g_expert_tiering_observe_log;
}

static int cuda_expert_tiering_observe_log_ids_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = cuda_env_truthy(getenv("DS4_EXPERT_TIERING_LOG_IDS"));
    }
    return cached;
}

static void cuda_expert_tier_promote_reset_prompt(void) {
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    memset(p->prompt_counts, 0, sizeof(p->prompt_counts));
    memset(p->hot, 0, sizeof(p->hot));
    memset(p->last_used, 0, sizeof(p->last_used));
    p->collecting_prompt = 1;
    p->preload_ready = 0;
    p->seen_any = 0;
    p->last_layer = -1;
    p->hot_count = 0;
    p->preloaded = 0;
    p->tick = 0;
    p->resets++;
}

static void cuda_expert_tier_promote_evict_one(
        cuda_expert_tier_promote_batch *batch) {
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    uint64_t best_age = UINT64_MAX;
    uint32_t best_layer = UINT32_MAX;
    uint32_t best_expert = UINT32_MAX;
    for (uint32_t layer = 0; layer < DS4_CUDA_TIERING_MAX_LAYER; layer++) {
        for (uint32_t expert = 0; expert < DS4_CUDA_TIERING_MAX_EXPERT; expert++) {
            if (!p->hot[layer][expert]) continue;
            const uint64_t age = p->last_used[layer][expert];
            if (age < best_age) {
                best_age = age;
                best_layer = layer;
                best_expert = expert;
            }
        }
    }
    if (best_layer == UINT32_MAX) return;
    p->hot[best_layer][best_expert] = 0;
    p->last_used[best_layer][best_expert] = 0;
    if (p->hot_count > 0) p->hot_count--;
    p->evictions++;
    if (batch) batch->evictions++;
}

static void cuda_expert_tier_promote_finalize_prompt(
        cuda_expert_tier_promote_batch *batch) {
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    if (p->preload_ready) return;
    p->collecting_prompt = 0;
    p->preload_ready = 1;
    p->finalized++;
    if (batch) batch->finalized = 1;
    const uint32_t cap = p->cap;
    for (uint32_t rank = 0; rank < cap; rank++) {
        uint32_t best_count = 0;
        uint32_t best_layer = UINT32_MAX;
        uint32_t best_expert = UINT32_MAX;
        for (uint32_t layer = 0; layer < DS4_CUDA_TIERING_MAX_LAYER; layer++) {
            for (uint32_t expert = 0; expert < DS4_CUDA_TIERING_MAX_EXPERT; expert++) {
                if (p->hot[layer][expert]) continue;
                const uint32_t count = p->prompt_counts[layer][expert];
                if (count > best_count) {
                    best_count = count;
                    best_layer = layer;
                    best_expert = expert;
                }
            }
        }
        if (best_layer == UINT32_MAX || best_count == 0) break;
        p->hot[best_layer][best_expert] = 1;
        p->last_used[best_layer][best_expert] = 0;
        p->hot_count++;
        p->preloaded++;
        p->total_preloaded++;
    }
    if (batch) {
        batch->preloaded = p->preloaded;
        batch->hot_count = p->hot_count;
    }
    if (p->verbose) {
        fprintf(stderr,
                "ds4: expert tier promote preloaded=%u cap=%u prompt_batches=%llu\n",
                p->preloaded,
                p->cap,
                (unsigned long long)p->prompt_batches);
    }
}

static cuda_expert_tier_promote_batch cuda_expert_tier_promote_note_batch(
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       compact_count,
        uint32_t       slot_count) {
    cuda_expert_tier_promote_batch batch = {0};
    batch.phase = "off";
    if (!cuda_expert_tier_promote_is_enabled()) return batch;
    batch.enabled = 1;
    batch.phase = "idle";
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    if (!p->notice_printed) {
        fprintf(stderr,
                "ds4: expert tier policy observe_promote enabled cap=%u "
                "(metadata plus optional cold-sidecar gate)\n",
                p->cap);
        p->notice_printed = 1;
    }
    if (!compact_ids || compact_count == 0 ||
        layer >= DS4_CUDA_TIERING_MAX_LAYER) {
        return batch;
    }

    const int prompt_like_batch = slot_count > DS4_CUDA_ROUTED_EXPERTS_PER_TOKEN;
    if (prompt_like_batch && layer == 0 &&
        (!p->collecting_prompt || p->preload_ready || !p->seen_any)) {
        cuda_expert_tier_promote_reset_prompt();
        batch.reset = 1;
    }
    if (!p->collecting_prompt && !p->preload_ready) {
        cuda_expert_tier_promote_reset_prompt();
        batch.reset = 1;
    }
    if (p->collecting_prompt && p->seen_any &&
        (!prompt_like_batch || (int)layer <= p->last_layer)) {
        cuda_expert_tier_promote_finalize_prompt(&batch);
    }

    if (p->collecting_prompt) {
        batch.phase = "prompt";
        p->prompt_batches++;
        if (slot_ids) {
            for (uint32_t i = 0; i < slot_count; i++) {
                const int32_t slot = slot_ids[i];
                if (slot < 0 || (uint32_t)slot >= compact_count) continue;
                const int32_t expert = compact_ids[(uint32_t)slot];
                if (expert < 0 ||
                    (uint32_t)expert >= DS4_CUDA_TIERING_MAX_EXPERT) {
                    continue;
                }
                p->prompt_counts[layer][(uint32_t)expert]++;
                p->prompt_selected++;
            }
        }
        p->seen_any = 1;
        p->last_layer = (int)layer;
        batch.hot_count = p->hot_count;
        return batch;
    }

    if (!p->preload_ready) {
        cuda_expert_tier_promote_finalize_prompt(&batch);
    }

    batch.phase = "decode";
    p->decode_batches++;
    for (uint32_t i = 0; i < compact_count; i++) {
        const int32_t expert_i = compact_ids[i];
        if (expert_i < 0 ||
            (uint32_t)expert_i >= DS4_CUDA_TIERING_MAX_EXPERT) {
            continue;
        }
        const uint32_t expert = (uint32_t)expert_i;
        p->requests++;
        if (p->hot[layer][expert]) {
            p->hot_hits++;
            batch.hot_hits++;
        } else {
            p->promotions++;
            batch.promotions++;
            if (p->cap != 0) {
                p->hot[layer][expert] = 1;
                p->hot_count++;
            }
        }
        if (p->cap != 0) {
            p->last_used[layer][expert] = ++p->tick;
            while (p->hot_count > p->cap) {
                cuda_expert_tier_promote_evict_one(&batch);
            }
        }
    }
    batch.hot_count = p->hot_count;
    return batch;
}

static uint64_t cuda_expert_tiering_observe_summary_every(void) {
    static int initialized;
    static uint64_t value;
    if (!initialized) {
        initialized = 1;
        const char *env = getenv("DS4_EXPERT_TIERING_SUMMARY_EVERY");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long long parsed = strtoull(env, &end, 10);
            if (end && *end == '\0') value = (uint64_t)parsed;
        }
    }
    return value;
}

static void cuda_expert_tiering_observe_append_top_experts(
        char    *buf,
        size_t   cap,
        uint32_t layer,
        uint32_t n_total_expert) {
    if (!buf || cap == 0) return;
    buf[0] = '\0';
    if (layer >= DS4_CUDA_TIERING_MAX_LAYER) return;
    if (n_total_expert > DS4_CUDA_TIERING_MAX_EXPERT) {
        n_total_expert = DS4_CUDA_TIERING_MAX_EXPERT;
    }
    uint32_t chosen[6] = {0};
    uint32_t chosen_count = 0;
    size_t used = 0;
    for (uint32_t rank = 0; rank < 6; rank++) {
        uint64_t best = 0;
        uint32_t best_expert = UINT32_MAX;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            int already = 0;
            for (uint32_t i = 0; i < chosen_count; i++) {
                if (chosen[i] == e) {
                    already = 1;
                    break;
                }
            }
            if (already) continue;
            const uint64_t count =
                g_expert_tiering_observe.selected_hist[layer][e];
            if (count > best) {
                best = count;
                best_expert = e;
            }
        }
        if (best_expert == UINT32_MAX || best == 0) break;
        chosen[chosen_count++] = best_expert;
        int n = snprintf(buf + used,
                         used < cap ? cap - used : 0,
                         "%s%u:%llu",
                         used ? "," : "",
                         best_expert,
                         (unsigned long long)best);
        if (n < 0) break;
        used += (size_t)n;
        if (used >= cap) {
            buf[cap - 1] = '\0';
            break;
        }
    }
}

static void cuda_expert_tiering_observe_print_summary(void) {
    if (!cuda_expert_tiering_observe_is_enabled()) return;
    const cuda_expert_tiering_observe_stats *s = &g_expert_tiering_observe;
    if (s->batches == 0) return;
    const uint64_t lookups = s->cache_hits + s->cache_misses;
    const double hit_rate =
        lookups ? (double)s->cache_hits / (double)lookups : 0.0;
    fprintf(stderr,
            "ds4: expert tiering observe summary batches=%llu slots=%llu "
            "compact=%llu hit_rate=%.4f hits=%llu misses=%llu direct=%llu "
            "evictions=%llu direct=%.2f MiB compact=%.2f MiB\n",
            (unsigned long long)s->batches,
            (unsigned long long)s->slot_requests,
            (unsigned long long)s->compact_experts,
            hit_rate,
            (unsigned long long)s->cache_hits,
            (unsigned long long)s->cache_misses,
            (unsigned long long)s->direct_loads,
            (unsigned long long)s->evictions,
            (double)s->direct_bytes / 1048576.0,
            (double)s->compact_bytes / 1048576.0);

    uint32_t printed[5] = {0};
    uint32_t printed_count = 0;
    for (uint32_t rank = 0; rank < 5; rank++) {
        uint64_t best_score = 0;
        uint32_t best_layer = UINT32_MAX;
        for (uint32_t layer = 0; layer < DS4_CUDA_TIERING_MAX_LAYER; layer++) {
            const cuda_expert_tiering_layer_stats *ls = &s->layers[layer];
            if (ls->batches == 0) continue;
            int already = 0;
            for (uint32_t i = 0; i < printed_count; i++) {
                if (printed[i] == layer) {
                    already = 1;
                    break;
                }
            }
            if (already) continue;
            const uint64_t score =
                ls->cache_misses + ls->direct_loads + ls->evictions;
            if (score > best_score) {
                best_score = score;
                best_layer = layer;
            }
        }
        if (best_layer == UINT32_MAX || best_score == 0) break;
        printed[printed_count++] = best_layer;
        const cuda_expert_tiering_layer_stats *ls = &s->layers[best_layer];
        const uint64_t layer_lookups = ls->cache_hits + ls->cache_misses;
        const double layer_hit_rate =
            layer_lookups ?
            (double)ls->cache_hits / (double)layer_lookups : 0.0;
        char top[160];
        cuda_expert_tiering_observe_append_top_experts(
                top,
                sizeof(top),
                best_layer,
                ls->max_total_expert);
        fprintf(stderr,
                "ds4: expert tiering observe layer=%u batches=%llu slots=%llu "
                "compact=%llu hit_rate=%.4f misses=%llu direct=%llu "
                "evictions=%llu cap=%u top=%s\n",
                best_layer,
                (unsigned long long)ls->batches,
                (unsigned long long)ls->slot_requests,
                (unsigned long long)ls->compact_experts,
                layer_hit_rate,
                (unsigned long long)ls->cache_misses,
                (unsigned long long)ls->direct_loads,
                (unsigned long long)ls->evictions,
                ls->max_cache_capacity,
                top[0] ? top : "-");
    }

    if (cuda_expert_tier_promote_is_enabled()) {
        const cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
        const double hot_rate =
            p->requests ? (double)p->hot_hits / (double)p->requests : 0.0;
        const double promotion_rate =
            p->requests ? (double)p->promotions / (double)p->requests : 0.0;
        fprintf(stderr,
                "ds4: expert tier promote summary cap=%u prompt_batches=%llu "
                "decode_batches=%llu preloaded=%u hot_count=%u requests=%llu "
                "hot_hit_rate=%.4f promotions=%llu promotion_rate=%.4f "
                "evictions=%llu resets=%llu finalized=%llu\n",
                p->cap,
                (unsigned long long)p->prompt_batches,
                (unsigned long long)p->decode_batches,
                p->preloaded,
                p->hot_count,
                (unsigned long long)p->requests,
                hot_rate,
                (unsigned long long)p->promotions,
                promotion_rate,
                (unsigned long long)p->evictions,
                (unsigned long long)p->resets,
                (unsigned long long)p->finalized);
    }
}

static void cuda_expert_tiering_observe_note_batch(
        uint32_t       layer,
        uint32_t       n_total_expert,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint32_t       cache_capacity,
        uint32_t       cache_count_before,
        uint32_t       cache_count_after,
        uint32_t       cache_hits,
        uint32_t       cache_misses,
        uint32_t       direct_loads,
        uint32_t       evictions,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        const cuda_expert_tier_promote_batch *promote_batch_in) {
    if (!cuda_expert_tiering_observe_is_enabled()) return;
    if (!g_expert_tiering_observe_atexit_registered) {
        atexit(cuda_expert_tiering_observe_atexit);
        g_expert_tiering_observe_atexit_registered = 1;
    }
    if (!g_expert_tiering_observe_notice_printed) {
        fprintf(stderr,
                "ds4: expert tiering observe enabled "
                "(DS4_EXPERT_TIERING=observe; log=%s)\n",
                getenv("DS4_EXPERT_TIERING_LOG") ?
                    getenv("DS4_EXPERT_TIERING_LOG") : "stderr-summary");
        g_expert_tiering_observe_notice_printed = 1;
    }

    const uint64_t expert_bytes =
        gate_expert_bytes > UINT64_MAX - gate_expert_bytes ||
        gate_expert_bytes + gate_expert_bytes > UINT64_MAX - down_expert_bytes ?
        0 : gate_expert_bytes + gate_expert_bytes + down_expert_bytes;
    const uint64_t compact_bytes =
        expert_bytes != 0 && compact_count <= UINT64_MAX / expert_bytes ?
        (uint64_t)compact_count * expert_bytes : 0;
    const uint64_t direct_bytes =
        expert_bytes != 0 && direct_loads <= UINT64_MAX / expert_bytes ?
        (uint64_t)direct_loads * expert_bytes : 0;
    cuda_expert_tier_promote_batch promote_batch_fallback = {0};
    const cuda_expert_tier_promote_batch *promote_batch = promote_batch_in;
    if (!promote_batch) {
        promote_batch_fallback =
            cuda_expert_tier_promote_note_batch(layer,
                                                compact_ids,
                                                slot_ids,
                                                compact_count,
                                                slot_count);
        promote_batch = &promote_batch_fallback;
    }

    cuda_expert_tiering_observe_stats *s = &g_expert_tiering_observe;
    s->batches++;
    s->slot_requests += slot_count;
    s->compact_experts += compact_count;
    s->cache_hits += cache_hits;
    s->cache_misses += cache_misses;
    s->direct_loads += direct_loads;
    s->evictions += evictions;
    s->direct_bytes += direct_bytes;
    s->compact_bytes += compact_bytes;

    if (layer < DS4_CUDA_TIERING_MAX_LAYER) {
        cuda_expert_tiering_layer_stats *ls = &s->layers[layer];
        ls->batches++;
        ls->slot_requests += slot_count;
        ls->compact_experts += compact_count;
        ls->cache_hits += cache_hits;
        ls->cache_misses += cache_misses;
        ls->direct_loads += direct_loads;
        ls->evictions += evictions;
        ls->direct_bytes += direct_bytes;
        ls->compact_bytes += compact_bytes;
        if (n_total_expert > ls->max_total_expert) {
            ls->max_total_expert = n_total_expert;
        }
        if (cache_capacity > ls->max_cache_capacity) {
            ls->max_cache_capacity = cache_capacity;
        }
        if (compact_ids && slot_ids &&
            n_total_expert <= DS4_CUDA_TIERING_MAX_EXPERT) {
            for (uint32_t i = 0; i < slot_count; i++) {
                const int32_t slot = slot_ids[i];
                if (slot < 0 || (uint32_t)slot >= compact_count) continue;
                const int32_t expert = compact_ids[(uint32_t)slot];
                if (expert < 0 ||
                    (uint32_t)expert >= DS4_CUDA_TIERING_MAX_EXPERT) {
                    continue;
                }
                s->selected_hist[layer][(uint32_t)expert]++;
            }
        }
    }

    FILE *fp = cuda_expert_tiering_observe_log_file();
    if (fp) {
        const char *path = cache_capacity != 0 ? "resident" : "selected_direct";
        fprintf(fp,
                "{\"event\":\"tiering_observe\",\"time_unix\":%lld,"
                "\"path\":\"%s\",\"layer\":%u,"
                "\"slots\":%u,\"compact\":%u,\"n_total_expert\":%u,"
                "\"cache_capacity\":%u,\"cache_before\":%u,"
                "\"cache_after\":%u,\"hits\":%u,\"misses\":%u,"
                "\"direct\":%u,\"evictions\":%u,"
                "\"direct_bytes\":%llu,\"compact_bytes\":%llu",
                (long long)time(NULL),
                path,
                layer,
                slot_count,
                compact_count,
                n_total_expert,
                cache_capacity,
                cache_count_before,
                cache_count_after,
                cache_hits,
                cache_misses,
                direct_loads,
                evictions,
                (unsigned long long)direct_bytes,
                (unsigned long long)compact_bytes);
        if (promote_batch->enabled) {
            fprintf(fp,
                    ",\"tier_policy\":\"observe_promote\","
                    "\"tier_policy_phase\":\"%s\","
                    "\"tier_policy_reset\":%d,"
                    "\"tier_policy_finalized\":%d,"
                    "\"tier_policy_hot_hits\":%u,"
                    "\"tier_policy_promotions\":%u,"
                    "\"tier_policy_evictions\":%u,"
                    "\"tier_policy_preloaded\":%u,"
                    "\"tier_policy_hot_count\":%u",
                    promote_batch->phase ? promote_batch->phase : "unknown",
                    promote_batch->reset,
                    promote_batch->finalized,
                    promote_batch->hot_hits,
                    promote_batch->promotions,
                    promote_batch->evictions,
                    promote_batch->preloaded,
                    promote_batch->hot_count);
        }
        if (cuda_expert_tiering_observe_log_ids_enabled()) {
            fputs(",\"selected\":[", fp);
            for (uint32_t i = 0; i < slot_count; i++) {
                int32_t expert = -1;
                if (slot_ids && compact_ids) {
                    const int32_t slot = slot_ids[i];
                    if (slot >= 0 && (uint32_t)slot < compact_count) {
                        expert = compact_ids[(uint32_t)slot];
                    }
                }
                fprintf(fp, "%s%d", i ? "," : "", expert);
            }
            fputs("],\"compact_ids\":[", fp);
            for (uint32_t i = 0; i < compact_count; i++) {
                fprintf(fp, "%s%d", i ? "," : "",
                        compact_ids ? compact_ids[i] : -1);
            }
            fputc(']', fp);
        }
        fputs("}\n", fp);
        fflush(fp);
    }

    const uint64_t every = cuda_expert_tiering_observe_summary_every();
    if (every != 0 && s->batches % every == 0) {
        cuda_expert_tiering_observe_print_summary();
    }
}

static int cuda_spex_stats_is_enabled(void) {
    if (g_spex_stats_enabled < 0) {
        const char *v = getenv("DS4_SPEX_STATS");
        g_spex_stats_enabled = (v && v[0] && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return g_spex_stats_enabled;
}

static double cuda_spex_now_sec(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int cuda_spex_stats_finish_copy(int ok, uint64_t bytes, double t0) {
    if (t0 > 0.0) {
        const double t1 = cuda_spex_now_sec();
        g_spex_stats.selected_copy_calls++;
        if (ok) g_spex_stats.selected_copy_bytes += bytes;
        if (t1 >= t0) g_spex_stats.selected_copy_seconds += t1 - t0;
    }
    return ok;
}

static void cuda_spex_stats_print(void) {
    if (g_spex_stats_printed || !cuda_spex_stats_is_enabled()) return;
    const uint64_t lookups =
        g_spex_stats.expert_cache_hits + g_spex_stats.expert_cache_misses;
    if (lookups == 0 &&
        g_spex_stats.selected_load_batches == 0 &&
        g_spex_stats.selected_copy_calls == 0) {
        return;
    }
    g_spex_stats_printed = 1;

    const double hit_rate = lookups ?
        (double)g_spex_stats.expert_cache_hits / (double)lookups : 0.0;
    const double miss_per_expert = g_spex_stats.selected_expert_requests ?
        (double)g_spex_stats.expert_cache_misses /
            (double)g_spex_stats.selected_expert_requests : 0.0;
    const double copy_ms = g_spex_stats.selected_copy_seconds * 1000.0;
    const double sync_ms = g_spex_stats.selected_sync_seconds * 1000.0;
    const double copy_ms_per_batch = g_spex_stats.selected_load_batches ?
        copy_ms / (double)g_spex_stats.selected_load_batches : 0.0;
    const double sync_ms_per_batch = g_spex_stats.selected_load_batches ?
        sync_ms / (double)g_spex_stats.selected_load_batches : 0.0;
    const double copied_mib =
        (double)g_spex_stats.selected_copy_bytes / 1048576.0;

    fprintf(stderr,
            "ds4: SPEX stats: selected_batches=%llu selected_experts=%llu "
            "cache_hits=%llu cache_misses=%llu hit_rate=%.4f "
            "miss_per_expert=%.4f direct_loads=%llu copy_calls=%llu "
            "copied=%.2f MiB copy_ms=%.3f copy_ms_per_batch=%.3f "
            "sync_calls=%llu sync_ms=%.3f sync_ms_per_batch=%.3f\n",
            (unsigned long long)g_spex_stats.selected_load_batches,
            (unsigned long long)g_spex_stats.selected_expert_requests,
            (unsigned long long)g_spex_stats.expert_cache_hits,
            (unsigned long long)g_spex_stats.expert_cache_misses,
            hit_rate,
            miss_per_expert,
            (unsigned long long)g_spex_stats.selected_direct_loads,
            (unsigned long long)g_spex_stats.selected_copy_calls,
            copied_mib,
            copy_ms,
            copy_ms_per_batch,
            (unsigned long long)g_spex_stats.selected_sync_calls,
            sync_ms,
            sync_ms_per_batch);
    if (g_inplace_reuse_hits || g_inplace_admissions || g_inplace_slot_uploads) {
        fprintf(stderr,
                "ds4: SPEX inplace: reuse_hits=%llu admissions=%llu "
                "slot_uploads=%llu pin_admits=%llu pin_rejects=%llu "
                "spex_pin_promotions=%llu spex_pin_releases=%llu "
                "spex_pin_displacements=%llu "
                "(compact re-gather bypassed)\n",
                (unsigned long long)g_inplace_reuse_hits,
                (unsigned long long)g_inplace_admissions,
                (unsigned long long)g_inplace_slot_uploads,
                (unsigned long long)g_inplace_pin_admits,
                (unsigned long long)g_inplace_pin_rejects,
                (unsigned long long)g_spex_pin_promotions,
                (unsigned long long)g_spex_pin_releases,
                (unsigned long long)g_spex_pin_displacements);
    }
}

static void cuda_spex_stats_print_atexit(void) {
    cuda_spex_stats_print();
}

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

struct cuda_stream_selected_cache {
    int valid;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
};

struct cuda_stream_expert_cache_slot {
    int valid;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t age;
    int pinned;      /* 0031: eviction-immune residency flag (LRU skips it) */
    int reap_spex_pinned; /* 0044: pin ownership by the provisional SPEX lane */
    double demand;   /* 0031: EWMA request frequency (cache-local rmass) */
    double cusum;    /* 0031: CUSUM accumulator for demand-driven rotation */
    int tier;        /* 0033: residency tier 0=SSD-blocked 1=RAM 2=VRAM-pinned */
    double knock;    /* 0033: decayed knock (request) count, sustained-demand proxy */
};

struct cuda_stream_expert_cache {
    int valid;
    uint32_t capacity;
    uint32_t count;
    uint64_t tick;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    /* 0031 pin-keep / residency-rotation runtime state (reset on realloc). */
    int pin_frozen;
    uint32_t pin_budget_slots;
    uint32_t pinned_count;
    uint32_t reap_pin_epoch_seen; /* 0044: producer overlay generation */
    uint64_t pin_calls;
    uint64_t pin_last_rotate;
    uint64_t pin_rotations;
    int pin_min_valid;
    uint32_t pin_min_slot;
    double pin_min_demand;
    /* 0033 tiered-hysteresis dynamic residency runtime state (reset on realloc). */
    int tier_seeded;
    uint32_t tier_vram_count;
    uint64_t tier_calls;
    uint64_t tier_last_swap;
    uint64_t tier_promotions;
    uint64_t tier_evictions;
    std::vector<cuda_stream_expert_cache_slot> slots;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;

/* patch 0050: PRIVATE index of masked-window ranges pinned via cudaHostRegister
 * (ds4_gpu_register_masked_ranges). Deliberately NOT stored in
 * g_model_ranges/g_model_range_by_offset above: those serve
 * cuda_model_range_ptr()'s "hand a UVA/device pointer straight to a compute
 * kernel" contract, and a MoE GEMM kernel reading its expert matrix
 * element-by-element from host RAM over PCIe is a ~25x decode collapse
 * (measured 0.10 t/s on the 2026-07-12 dev box when these ranges were visible
 * to cuda_model_range_ptr). This index instead answers a different, narrower
 * question for cuda_model_copy_to_device_streamed's hot MoE-expert-load path:
 * "is [offset,offset+bytes) already pinned host memory?" so that path can
 * DMA-copy straight from the pinned mmap into the VRAM expert-cache slot
 * (cudaMemcpyAsync at real PCIe bandwidth) instead of pread()-ing into a
 * staging buffer first -- eliminating the pread+staging toll without changing
 * the "compute reads VRAM" contract at all. reg_base/reg_bytes remember each
 * page-aligned cudaHostRegister block for the matching cudaHostUnregister.
 * Sorted by offset after the batch insert; binary search at lookup time so it
 * stays cheap with several thousand coalesced ranges. */
struct cuda_masked_pin_range {
    uint64_t offset;      /* model-map byte offset of the payload */
    uint64_t bytes;       /* payload length */
    void    *reg_base;    /* page-aligned cudaHostRegister address */
    uint64_t reg_bytes;   /* page-aligned registered length */
};
static std::vector<cuda_masked_pin_range> g_masked_pin_ranges;
static const void *g_masked_pin_host_base;

enum cuda_masked_pin_cover_result {
    CUDA_MASKED_PIN_COVER_HIT = 0,
    CUDA_MASKED_PIN_COVER_EMPTY,
    CUDA_MASKED_PIN_COVER_BASE_MISMATCH,
    CUDA_MASKED_PIN_COVER_BEFORE_FIRST,
    CUDA_MASKED_PIN_COVER_RANGE_MISS,
};

struct cuda_masked_pin_diag_stats {
    uint64_t queries;
    uint64_t query_bytes;
    uint64_t hits;
    uint64_t hit_bytes;
    uint64_t dma_ok;
    uint64_t dma_ok_bytes;
    uint64_t dma_failed;
    uint64_t miss_empty;
    uint64_t miss_base;
    uint64_t miss_before_first;
    uint64_t miss_range;
    int first_miss_logged[5];
};
static cuda_masked_pin_diag_stats g_masked_pin_diag;

static bool cuda_masked_pin_diag_enabled(void) {
    static const bool enabled = []() {
        const char *v = getenv("DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG");
        return v && *v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

static void cuda_masked_pin_diag_report(const char *why) {
    if (!cuda_masked_pin_diag_enabled()) return;
    fprintf(stderr,
            "ds4: CUDA masked zero-copy diag %s: queries=%llu/%.2f MiB "
            "covered=%llu/%.2f MiB dma_ok=%llu/%.2f MiB dma_failed=%llu "
            "miss_empty=%llu miss_base=%llu miss_before=%llu miss_range=%llu\n",
            why ? why : "snapshot",
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.queries, 0),
            (double)__sync_fetch_and_add(&g_masked_pin_diag.query_bytes, 0) / 1048576.0,
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.hits, 0),
            (double)__sync_fetch_and_add(&g_masked_pin_diag.hit_bytes, 0) / 1048576.0,
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok, 0),
            (double)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok_bytes, 0) / 1048576.0,
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.dma_failed, 0),
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.miss_empty, 0),
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.miss_base, 0),
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.miss_before_first, 0),
            (unsigned long long)__sync_fetch_and_add(&g_masked_pin_diag.miss_range, 0));
}

static void cuda_masked_pin_finalize(void) {
    std::sort(g_masked_pin_ranges.begin(), g_masked_pin_ranges.end(),
              [](const cuda_masked_pin_range &a, const cuda_masked_pin_range &b) {
                  return a.offset < b.offset;
              });
}

static void cuda_masked_pin_release_all(void) {
    cuda_masked_pin_diag_report("final");
    for (const cuda_masked_pin_range &r : g_masked_pin_ranges) {
        if (r.reg_base) {
            cudaError_t err = cudaHostUnregister(r.reg_base);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA masked zero-copy unregister failed for %p "
                        "(%.2f MiB): %s\n",
                        r.reg_base, (double)r.reg_bytes / 1048576.0,
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
            }
        }
    }
    g_masked_pin_ranges.clear();
    g_masked_pin_host_base = NULL;
    memset(&g_masked_pin_diag, 0, sizeof(g_masked_pin_diag));
}

/* Pin one [offset,bytes) span of the model map with page-aligned
 * cudaHostRegister (ReadOnly first, plain retry on the same error class the
 * per-range register above handles) and remember it in the private index.
 * No cudaHostGetDevicePointer, no g_model_ranges entry: compute-kernel
 * lookups must never see this memory. Returns 1 on success. */
static int cuda_masked_pin_register(const void *model_map, uint64_t offset,
                                    uint64_t bytes, int *unsupported) {
    if (bytes == 0) return 0;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
    const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);

    /* Same flag combo the per-range mapped register uses (known-good on this
     * WSL2 box for thousands of ranges over a PROT_READ mmap; ReadOnly is
     * what makes registering a read-only mapping legal). We keep Mapped for
     * flag parity but never call cudaHostGetDevicePointer -- pinning for DMA
     * is all this path wants. */
    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) flags = cudaHostRegisterMapped;
    cudaError_t err = cudaHostRegister((void *)reg_addr, (size_t)reg_bytes, flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)reg_addr, (size_t)reg_bytes,
                               cudaHostRegisterMapped);
    }
    if (err != cudaSuccess) {
        if (unsupported &&
            (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
            *unsupported = 1;
        }
        (void)cudaGetLastError();
        return 0;
    }
    g_masked_pin_host_base = model_map;
    g_masked_pin_ranges.push_back({offset, bytes, (void *)reg_addr, reg_bytes});
    return 1;
}

/* True iff the ENTIRE [offset, offset+bytes) span is covered by one pinned
 * range (ranges are never adjacent-merged across separate cudaHostRegister
 * calls, so a query spanning two ranges intentionally reports false -- the
 * caller falls back to the normal pread path for that chunk, which is always
 * correct, just not accelerated). */
static bool cuda_masked_pin_covers(const void *model_map, uint64_t offset,
                                   uint64_t bytes, int *out_reason,
                                   uint64_t *out_range_start,
                                   uint64_t *out_range_end) {
    if (out_reason) *out_reason = CUDA_MASKED_PIN_COVER_RANGE_MISS;
    if (out_range_start) *out_range_start = 0;
    if (out_range_end) *out_range_end = 0;
    if (g_masked_pin_ranges.empty()) {
        if (out_reason) *out_reason = CUDA_MASKED_PIN_COVER_EMPTY;
        return false;
    }
    if (model_map != g_masked_pin_host_base) {
        if (out_reason) *out_reason = CUDA_MASKED_PIN_COVER_BASE_MISMATCH;
        return false;
    }
    auto it = std::upper_bound(
        g_masked_pin_ranges.begin(), g_masked_pin_ranges.end(), offset,
        [](uint64_t val, const cuda_masked_pin_range &r) { return val < r.offset; });
    if (it == g_masked_pin_ranges.begin()) {
        if (out_reason) *out_reason = CUDA_MASKED_PIN_COVER_BEFORE_FIRST;
        if (out_range_start) *out_range_start = it->offset;
        if (out_range_end) *out_range_end = it->offset + it->bytes;
        return false;
    }
    --it;
    if (out_range_start) *out_range_start = it->offset;
    if (out_range_end) *out_range_end = it->offset + it->bytes;
    const uint64_t end = offset + bytes;
    const bool covered = end >= offset && offset >= it->offset &&
                         end <= it->offset + it->bytes;
    if (out_reason) *out_reason = covered ? CUDA_MASKED_PIN_COVER_HIT
                                          : CUDA_MASKED_PIN_COVER_RANGE_MISS;
    return covered;
}
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static cuda_stream_selected_cache g_stream_selected_cache;
static cuda_stream_expert_cache g_stream_expert_cache;
static uint32_t g_stream_expert_budget_override;
static uint32_t g_stream_expert_runtime_cap;
static uint32_t g_stream_expert_memory_cap_notice;
static uint64_t g_stream_expert_runtime_gate_bytes;
static uint64_t g_stream_expert_runtime_down_bytes;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static uint64_t g_model_load_progress_last_bytes = UINT64_MAX;
static uint64_t g_model_load_progress_last_cgib = UINT64_MAX;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static void *g_stream_selected_stage_raw[4];
static void *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_bytes;
static cudaStream_t g_stream_selected_upload_stream;
static cudaEvent_t g_stream_selected_upload_done_event;
static int g_stream_selected_upload_done_recorded;
static int g_stream_selected_upload_event_enabled = -1;

static int cuda_ok(cudaError_t err, const char *what);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset);
static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what,
        int defer_upload_sync);
static uint64_t cuda_model_cache_limit_bytes(void);
static uint64_t cuda_model_local_model_limit_bytes(void);
static int cuda_model_cache_limit_explicit(void);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_register_mapped(const void *model_map,
                                                    uint64_t offset,
                                                    uint64_t bytes,
                                                    const char *what) {
    if (!g_model_range_mapping_supported || bytes == 0) return NULL;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
    uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
    if (model_map == g_model_host_base &&
        g_model_registered_size >= 88ull * 1073741824ull &&
        g_model_registered_size <= 96ull * 1073741824ull &&
        g_model_range_bytes >= 80ull * 1073741824ull) {
        const uintptr_t model_base = (uintptr_t)model_map;
        const uintptr_t model_end = model_base + (uintptr_t)g_model_registered_size;
        if (model_end > model_base && model_end > reg_addr) {
            const uint64_t tail_bytes = (uint64_t)(model_end - reg_addr);
            reg_bytes = (tail_bytes + page_sz - 1u) & ~(page_sz - 1u);
        }
    }
    void *reg_dev = NULL;

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }

    cudaError_t err = cudaHostRegister((void *)reg_addr,
                                       (size_t)reg_bytes,
                                       flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped);
    }
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
        if (err == cudaSuccess && reg_dev) {
            char *dev_ptr = (char *)reg_dev + reg_delta;
            g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
            g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
            if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                        what ? what : "weights",
                        (double)bytes / 1048576.0);
            }
            return dev_ptr;
        }
        fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaHostUnregister((void *)reg_addr);
        (void)cudaGetLastError();
        return NULL;
    }

    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        g_model_range_mapping_supported = 0;
    }
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA model range map skipped for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
    }
    (void)cudaGetLastError();
    return NULL;
}

/* Allocate a device-resident copy of [offset, offset+bytes) from model_map and
 * push it into g_model_ranges so future cuda_model_range_ptr lookups hit it.
 * Returns the device pointer on success, NULL on cudaMalloc/cudaMemcpy failure.
 * Caller is responsible for any policy gating (budget cap, env opt-out, etc.) */
static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                          uint64_t offset,
                                                          uint64_t bytes,
                                                          const char *what) {
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA skipped device copy for %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return NULL;
    }

    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Device-resident HBM cache hits win over UVA-mapped registered pointers:
     * direct HBM reads are ~10% faster than mapped reads through host page
     * tables (measured on plain decode at GB10).  Cache lookup runs first; the
     * registered-mapped shortcut below is the cold fallback when an allocation
     * hasn't been pre-populated. */
    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        getenv("DS4_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("DS4_CUDA_WEIGHT_PRELOAD") == NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = getenv("DS4_CUDA_DIRECT_MODEL");
    if (direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    const char *mapped = cuda_model_range_register_mapped(model_map, offset, bytes, what);
    if (mapped) return mapped;

    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    /* On 96/128 GB UMA Spark-class systems the expanded Q8->F16 cache can
     * pass a simple free-memory reserve check but still leave too little room
     * for long-prefill cuBLAS execution.  Keep the startup cache useful but
     * bounded unless the caller explicitly sets DS4_CUDA_Q8_F16_CACHE_MB. */
    if (limit == UINT64_MAX &&
        total_bytes <= 128ull * 1073741824ull &&
        (g_model_range_bytes >= 64ull * 1073741824ull ||
         g_model_registered_size >= 64ull * 1073741824ull)) {
        if (g_model_registered_size >= 112ull * 1073741824ull) {
            limit = 4ull * 1073741824ull;
        } else if (g_model_registered_size >= 88ull * 1073741824ull ||
                   g_model_range_bytes >= 88ull * 1073741824ull) {
            limit = 16ull * 1073741824ull;
        } else if (g_model_range_bytes >= 64ull * 1073741824ull) {
            limit = 12ull * 1073741824ull;
        } else {
            limit = 8ull * 1073741824ull;
        }
        if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
            cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
            return 0;
        }
    }
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (getenv("DS4_CUDA_Q8_F16_ALL") != NULL) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE") == NULL;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_use_dp4a(void) {
    return getenv("DS4_CUDA_NO_Q8_DP4A") == NULL;
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (getenv("DS4_CUDA_NO_Q8_F32_CACHE") != NULL) return 0;
    if (getenv("DS4_CUDA_Q8_F32_ALL") != NULL) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") != NULL;
    }
    return getenv("DS4_CUDA_Q8_F32_LARGE") != NULL &&
           in_dim == 1024u && out_dim == 32768u;
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f32_by_offset.find(offset);
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static uint16_t cuda_expert_cold_f32_to_f16(float v) {
    __half h = __float2half(v);
    uint16_t bits = 0;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static float cuda_expert_cold_f16_to_f32(uint16_t bits) {
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

static uint64_t cuda_expert_cq1_block_bytes(uint32_t group) {
    if (group != 32u && group != 64u && group != 256u) return 0;
    return 32u + (uint64_t)(CUDA_QK_K / group) * sizeof(uint16_t);
}

static int cuda_expert_cq1_parse_format(const char *fmt, uint32_t *group_out) {
    if (!fmt || !fmt[0]) return 0;
    if (strcmp(fmt, "cq1") == 0 || strcmp(fmt, "cq1g32") == 0) {
        if (group_out) *group_out = 32u;
        return 1;
    }
    if (strcmp(fmt, "cq1g64") == 0) {
        if (group_out) *group_out = 64u;
        return 1;
    }
    if (strcmp(fmt, "cq1g256") == 0) {
        if (group_out) *group_out = 256u;
        return 1;
    }
    return 0;
}

static int cuda_expert_cq1_fetch_iq2_tables(uint64_t grid[256], uint8_t signs[128]) {
    static uint64_t host_grid[256];
    static uint8_t host_signs[128];
    static int ready;
    static int failed;
    if (!ready && !failed) {
        cudaError_t err = cudaMemcpyFromSymbol(host_grid,
                                               cuda_iq2xxs_grid,
                                               sizeof(host_grid));
        if (err == cudaSuccess) {
            err = cudaMemcpyFromSymbol(host_signs,
                                       cuda_ksigns_iq2xs,
                                       sizeof(host_signs));
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: cold CQ1 failed to read IQ2 tables: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            failed = 1;
        } else {
            ready = 1;
        }
    }
    if (!ready) return 0;
    memcpy(grid, host_grid, sizeof(host_grid));
    memcpy(signs, host_signs, sizeof(host_signs));
    return 1;
}

static void cuda_expert_cq1_write_sign_bit(uint8_t *signs, uint32_t idx, uint32_t negative) {
    if (negative) signs[idx >> 3u] |= (uint8_t)(1u << (idx & 7u));
}

static uint32_t cuda_expert_cq1_read_sign_bit(const uint8_t *signs, uint32_t idx) {
    return (uint32_t)((signs[idx >> 3u] >> (idx & 7u)) & 1u);
}

static void cuda_expert_cq1_write_f16_scale(char *dst, uint32_t group, float scale) {
    const uint16_t h = cuda_expert_cold_f32_to_f16(scale);
    memcpy(dst + (uint64_t)group * sizeof(uint16_t), &h, sizeof(h));
}

static float cuda_expert_cq1_read_f16_scale(const char *src, uint32_t group) {
    uint16_t h = 0;
    memcpy(&h, src + (uint64_t)group * sizeof(uint16_t), sizeof(h));
    return cuda_expert_cold_f16_to_f32(h);
}

static int cuda_expert_cq1_encode_iq2_block(
        const char *src,
        char *dst,
        uint32_t group,
        const uint64_t grid[256],
        const uint8_t signs_lut[128]) {
    const uint64_t block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!block_bytes) return 0;
    const uint32_t groups = CUDA_QK_K / group;
    float sum_abs[8] = {0};
    uint32_t count[8] = {0};
    uint8_t sign_bytes[32];
    memset(sign_bytes, 0, sizeof(sign_bytes));

    uint16_t hd = 0;
    memcpy(&hd, src, sizeof(hd));
    const float d = cuda_expert_cold_f16_to_f32(hd);
    const uint16_t *q2 = (const uint16_t *)(src + sizeof(uint16_t));
    for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32u; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16u);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16u);
        q2 += 4;
        const float ls = (float)(2u * (aux1 >> 28u) + 1u);
        const uint8_t grid_idx[4] = {
            (uint8_t)(aux0 & 0xffu),
            (uint8_t)((aux0 >> 8u) & 0xffu),
            (uint8_t)((aux0 >> 16u) & 0xffu),
            (uint8_t)((aux0 >> 24u) & 0xffu),
        };
        const uint32_t sign_idx[4] = {
            (aux1 >> 0u) & 127u,
            (aux1 >> 7u) & 127u,
            (aux1 >> 14u) & 127u,
            (aux1 >> 21u) & 127u,
        };
        for (uint32_t sg = 0; sg < 4u; sg++) {
            const uint64_t g = grid[grid_idx[sg]];
            const uint8_t sign_mask = signs_lut[sign_idx[sg]];
            for (uint32_t i = 0; i < 8u; i++) {
                const uint32_t idx = ib32 * 32u + sg * 8u + i;
                const uint32_t cqg = idx / group;
                const uint8_t mag = (uint8_t)((g >> (8u * i)) & 0xffu);
                sum_abs[cqg] += 0.125f * d * ls * (float)mag;
                count[cqg]++;
                cuda_expert_cq1_write_sign_bit(sign_bytes, idx, (sign_mask >> i) & 1u);
            }
        }
    }
    memset(dst, 0, (size_t)block_bytes);
    for (uint32_t g = 0; g < groups; g++) {
        const float scale = count[g] ? sum_abs[g] / (float)count[g] : 0.0f;
        cuda_expert_cq1_write_f16_scale(dst, g, scale);
    }
    memcpy(dst + groups * sizeof(uint16_t), sign_bytes, sizeof(sign_bytes));
    return 1;
}

static int cuda_expert_cq1_encode_q2_block(const char *src, char *dst, uint32_t group) {
    const uint64_t block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!block_bytes) return 0;
    const uint32_t groups = CUDA_QK_K / group;
    const cuda_block_q2_K *x = (const cuda_block_q2_K *)src;
    float sum_abs[8] = {0};
    uint32_t count[8] = {0};
    uint8_t sign_bytes[32];
    memset(sign_bytes, 0, sizeof(sign_bytes));
    const float d = cuda_expert_cold_f16_to_f32(x->d);
    const float dmin = cuda_expert_cold_f16_to_f32(x->dmin);
    for (uint32_t qg = 0; qg < 16u; qg++) {
        const uint32_t half = qg / 8u;
        const uint32_t within = qg % 8u;
        const uint32_t qbase = half * 32u + ((within & 1u) ? 16u : 0u);
        const uint32_t shift = (within / 2u) * 2u;
        const uint32_t sc = x->scales[qg] & 0x0fu;
        const uint32_t mn = x->scales[qg] >> 4u;
        for (uint32_t i = 0; i < 16u; i++) {
            const uint32_t idx = qg * 16u + i;
            const uint32_t q = (x->qs[qbase + i] >> shift) & 0x03u;
            const float v = d * (float)sc * (float)q - dmin * (float)mn;
            sum_abs[idx / group] += fabsf(v);
            count[idx / group]++;
            cuda_expert_cq1_write_sign_bit(sign_bytes, idx, v < 0.0f);
        }
    }
    memset(dst, 0, (size_t)block_bytes);
    for (uint32_t g = 0; g < groups; g++) {
        const float scale = count[g] ? sum_abs[g] / (float)count[g] : 0.0f;
        cuda_expert_cq1_write_f16_scale(dst, g, scale);
    }
    memcpy(dst + groups * sizeof(uint16_t), sign_bytes, sizeof(sign_bytes));
    return 1;
}

static int cuda_expert_cq1_decode_iq2_block(const char *src, char *dst, uint32_t group) {
    const uint64_t block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!block_bytes) return 0;
    const uint32_t groups = CUDA_QK_K / group;
    const uint8_t *sign_bytes = (const uint8_t *)(src + groups * sizeof(uint16_t));
    float scale32[CUDA_QK_K / 32u];
    float max_scale = 0.0f;
    for (uint32_t g32 = 0; g32 < CUDA_QK_K / 32u; g32++) {
        const uint32_t cqg = (g32 * 32u) / group;
        scale32[g32] = cuda_expert_cq1_read_f16_scale(src, cqg);
        max_scale = fmaxf(max_scale, scale32[g32]);
    }
    cuda_block_iq2_xxs *out = (cuda_block_iq2_xxs *)dst;
    memset(out, 0, sizeof(*out));
    const float d = max_scale > 0.0f ? max_scale / 31.0f : 0.0f;
    out->d = cuda_expert_cold_f32_to_f16(d);
    uint16_t *q2 = out->qs;
    for (uint32_t g32 = 0; g32 < CUDA_QK_K / 32u; g32++) {
        uint32_t l = 0;
        if (d > 0.0f) {
            const float target = scale32[g32] / d;
            int il = (int)lrintf((target - 1.0f) * 0.5f);
            if (il < 0) il = 0;
            if (il > 15) il = 15;
            l = (uint32_t)il;
        }
        uint32_t aux1 = l << 28u;
        for (uint32_t sg = 0; sg < 4u; sg++) {
            uint32_t sign_idx = 0;
            for (uint32_t i = 0; i < 7u; i++) {
                const uint32_t idx = g32 * 32u + sg * 8u + i;
                sign_idx |= cuda_expert_cq1_read_sign_bit(sign_bytes, idx) << i;
            }
            aux1 |= sign_idx << (sg * 7u);
        }
        const uint32_t aux0 = 0u; /* grid 0 = all unit magnitudes. */
        q2[0] = (uint16_t)(aux0 & 0xffffu);
        q2[1] = (uint16_t)(aux0 >> 16u);
        q2[2] = (uint16_t)(aux1 & 0xffffu);
        q2[3] = (uint16_t)(aux1 >> 16u);
        q2 += 4;
    }
    return 1;
}

static int cuda_expert_cq1_decode_q2_block(const char *src, char *dst, uint32_t group) {
    const uint64_t block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!block_bytes) return 0;
    const uint32_t groups = CUDA_QK_K / group;
    const uint8_t *sign_bytes = (const uint8_t *)(src + groups * sizeof(uint16_t));
    float scale16[16];
    float max_scale = 0.0f;
    for (uint32_t g16 = 0; g16 < 16u; g16++) {
        const uint32_t cqg = (g16 * 16u) / group;
        scale16[g16] = cuda_expert_cq1_read_f16_scale(src, cqg);
        max_scale = fmaxf(max_scale, scale16[g16]);
    }
    cuda_block_q2_K *out = (cuda_block_q2_K *)dst;
    memset(out, 0, sizeof(*out));
    const float unit = max_scale > 0.0f ? max_scale / 15.0f : 0.0f;
    out->d = cuda_expert_cold_f32_to_f16(unit);
    out->dmin = cuda_expert_cold_f32_to_f16(unit);
    for (uint32_t g16 = 0; g16 < 16u; g16++) {
        uint32_t sc = 0;
        if (unit > 0.0f) {
            int is = (int)lrintf(scale16[g16] / unit);
            if (is < 0) is = 0;
            if (is > 15) is = 15;
            sc = (uint32_t)is;
        }
        out->scales[g16] = (uint8_t)(sc | (sc << 4u));
        const uint32_t half = g16 / 8u;
        const uint32_t within = g16 % 8u;
        const uint32_t qbase = half * 32u + ((within & 1u) ? 16u : 0u);
        const uint32_t shift = (within / 2u) * 2u;
        for (uint32_t i = 0; i < 16u; i++) {
            const uint32_t idx = g16 * 16u + i;
            const uint32_t q = cuda_expert_cq1_read_sign_bit(sign_bytes, idx) ? 0u : 2u;
            out->qs[qbase + i] |= (uint8_t)(q << shift);
        }
    }
    return 1;
}

static int cuda_expert_cq1_encode_blocks(
        const char *src,
        uint64_t native_block_bytes,
        uint64_t native_blocks,
        char *dst,
        uint32_t group,
        int is_iq2,
        const uint64_t grid[256],
        const uint8_t signs[128]) {
    const uint64_t cq1_block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!cq1_block_bytes || !native_block_bytes) return 0;
    for (uint64_t i = 0; i < native_blocks; i++) {
        const char *block_src = src + i * native_block_bytes;
        char *block_dst = dst + i * cq1_block_bytes;
        int ok = is_iq2 ?
            cuda_expert_cq1_encode_iq2_block(block_src, block_dst, group, grid, signs) :
            cuda_expert_cq1_encode_q2_block(block_src, block_dst, group);
        if (!ok) return 0;
    }
    return 1;
}

static int cuda_expert_cq1_decode_blocks(
        const char *src,
        uint64_t native_block_bytes,
        uint64_t native_blocks,
        char *dst,
        uint32_t group,
        int is_iq2) {
    const uint64_t cq1_block_bytes = cuda_expert_cq1_block_bytes(group);
    if (!cq1_block_bytes || !native_block_bytes) return 0;
    for (uint64_t i = 0; i < native_blocks; i++) {
        const char *block_src = src + i * cq1_block_bytes;
        char *block_dst = dst + i * native_block_bytes;
        int ok = is_iq2 ?
            cuda_expert_cq1_decode_iq2_block(block_src, block_dst, group) :
            cuda_expert_cq1_decode_q2_block(block_src, block_dst, group);
        if (!ok) return 0;
    }
    return 1;
}

static uint64_t cuda_expert_cold_lossless_mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
    return h;
}

static uint64_t cuda_expert_cold_lossless_hash_bytes(
        const char *ptr,
        uint64_t bytes,
        uint64_t seed) {
    uint64_t h = seed ? seed : 1469598103934665603ull;
    for (uint64_t i = 0; i < bytes; i++) {
        h ^= (uint8_t)ptr[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t cuda_expert_cold_lossless_key(
        const void *model_map,
        uint32_t layer,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t format,
        uint32_t cq1_group) {
    uint64_t h = 1469598103934665603ull;
    h = cuda_expert_cold_lossless_mix(h, (uint64_t)(uintptr_t)model_map);
    h = cuda_expert_cold_lossless_mix(h, layer);
    h = cuda_expert_cold_lossless_mix(h, expert);
    h = cuda_expert_cold_lossless_mix(h, gate_offset);
    h = cuda_expert_cold_lossless_mix(h, up_offset);
    h = cuda_expert_cold_lossless_mix(h, down_offset);
    h = cuda_expert_cold_lossless_mix(h, gate_expert_bytes);
    h = cuda_expert_cold_lossless_mix(h, down_expert_bytes);
    h = cuda_expert_cold_lossless_mix(h, format);
    h = cuda_expert_cold_lossless_mix(h, cq1_group);
    return h;
}

static void cuda_expert_cold_lossless_print_summary(void) {
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if (!s->enabled) return;
    const char *fmt = s->format == CUDA_EXPERT_COLD_FORMAT_CQ1 ? "cq1" : "lossless";
    fprintf(stderr,
            "ds4: cold RAM sidecar summary format=%s group=%u entries=%zu materialized=%llu "
            "reuses=%llu copies=%llu failures(mat=%llu copy=%llu verify=%llu) "
            "bytes=%.2f MiB repacked=%.2f MiB repack_failures=%llu "
            "prefill=%d ungated=%d native_tokens=%u verify=%d "
            "policy(allow=%llu prompt=%llu no_policy=%llu no_hotset=%llu "
            "warmup=%llu hot=%llu invalid=%llu)\n",
            fmt,
            s->cq1_group,
            s->entries.size(),
            (unsigned long long)s->materialized,
            (unsigned long long)s->reuses,
            (unsigned long long)s->copies,
            (unsigned long long)s->materialize_failures,
            (unsigned long long)s->copy_failures,
            (unsigned long long)s->verify_failures,
            (double)s->bytes_materialized / 1048576.0,
            (double)s->bytes_repacked / 1048576.0,
            (unsigned long long)s->repack_failures,
            s->allow_prefill,
            s->allow_ungated,
            s->native_decode_tokens,
            s->verify,
            (unsigned long long)s->policy_allowed,
            (unsigned long long)s->policy_skip_prompt,
            (unsigned long long)s->policy_skip_no_policy,
            (unsigned long long)s->policy_skip_no_hotset,
            (unsigned long long)s->policy_skip_warmup,
            (unsigned long long)s->policy_skip_hot,
            (unsigned long long)s->policy_skip_invalid);
}

enum cuda_expert_cold_policy_reason {
    CUDA_EXPERT_COLD_POLICY_ALLOW = 0,
    CUDA_EXPERT_COLD_POLICY_PROMPT,
    CUDA_EXPERT_COLD_POLICY_NO_POLICY,
    CUDA_EXPERT_COLD_POLICY_NO_HOTSET,
    CUDA_EXPERT_COLD_POLICY_WARMUP,
    CUDA_EXPERT_COLD_POLICY_HOT,
    CUDA_EXPERT_COLD_POLICY_INVALID,
};

static void cuda_expert_cold_policy_note(
        cuda_expert_cold_lossless_stats *s,
        int reason) {
    switch (reason) {
    case CUDA_EXPERT_COLD_POLICY_ALLOW:
        s->policy_allowed++;
        break;
    case CUDA_EXPERT_COLD_POLICY_PROMPT:
        s->policy_skip_prompt++;
        break;
    case CUDA_EXPERT_COLD_POLICY_NO_POLICY:
        s->policy_skip_no_policy++;
        break;
    case CUDA_EXPERT_COLD_POLICY_NO_HOTSET:
        s->policy_skip_no_hotset++;
        break;
    case CUDA_EXPERT_COLD_POLICY_WARMUP:
        s->policy_skip_warmup++;
        break;
    case CUDA_EXPERT_COLD_POLICY_HOT:
        s->policy_skip_hot++;
        break;
    default:
        s->policy_skip_invalid++;
        break;
    }
}

static int cuda_expert_cold_cq1_policy_allows(
        uint32_t layer,
        uint32_t expert,
        int prompt_like_batch,
        uint32_t native_decode_tokens,
        uint32_t native_decode_layers,
        int *reason_out) {
    if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_ALLOW;
    if (prompt_like_batch) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_PROMPT;
        return 0;
    }
    if (!cuda_expert_tier_promote_is_enabled()) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_NO_POLICY;
        return 0;
    }
    cuda_expert_tier_promote_stats *p = &g_expert_tier_promote;
    if (!p->preload_ready) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_NO_HOTSET;
        return 0;
    }
    const uint32_t layers =
        native_decode_layers != 0 ? native_decode_layers : 43u;
    uint64_t warmup_batches = UINT64_MAX;
    if (native_decode_tokens <= UINT64_MAX / (uint64_t)layers) {
        warmup_batches = (uint64_t)native_decode_tokens * (uint64_t)layers;
    }
    if (warmup_batches != 0 && p->decode_batches < warmup_batches) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_WARMUP;
        return 0;
    }
    if (layer >= DS4_CUDA_TIERING_MAX_LAYER ||
        expert >= DS4_CUDA_TIERING_MAX_EXPERT) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_INVALID;
        return 0;
    }
    if (p->hot[layer][expert]) {
        if (reason_out) *reason_out = CUDA_EXPERT_COLD_POLICY_HOT;
        return 0;
    }
    return 1;
}

static void cuda_expert_cold_lossless_atexit(void) {
    cuda_expert_cold_lossless_print_summary();
}

static int cuda_expert_cold_lossless_is_enabled(void) {
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if (!s->initialized) {
        s->initialized = 1;
        const char *fmt = getenv("DS4_EXPERT_COLD_FORMAT");
        uint32_t cq1_group = 0;
        if (cuda_expert_cq1_parse_format(fmt, &cq1_group)) {
            s->enabled = 1;
            s->format = CUDA_EXPERT_COLD_FORMAT_CQ1;
            s->cq1_group = cq1_group;
        } else if (fmt && (strcmp(fmt, "lossless") == 0 || strcmp(fmt, "native") == 0)) {
            s->enabled = 1;
            s->format = CUDA_EXPERT_COLD_FORMAT_LOSSLESS;
        } else {
            s->enabled = cuda_env_truthy(getenv("DS4_EXPERT_COLD_RAM_LOSSLESS"));
            s->format = s->enabled ? CUDA_EXPERT_COLD_FORMAT_LOSSLESS : CUDA_EXPERT_COLD_FORMAT_NONE;
        }
        s->allow_prefill = cuda_env_truthy(getenv("DS4_EXPERT_COLD_RAM_PREFILL"));
        s->allow_ungated = cuda_env_truthy(getenv("DS4_EXPERT_COLD_ALLOW_UNGATED"));
        s->native_decode_tokens = cuda_env_u32("DS4_EXPERT_COLD_NATIVE_TOKENS",
                                               50u,
                                               1000000u);
        s->native_decode_layers = cuda_env_u32("DS4_EXPERT_COLD_NATIVE_LAYERS",
                                               43u,
                                               DS4_CUDA_TIERING_MAX_LAYER);
        if (s->native_decode_layers == 0) s->native_decode_layers = 43u;
        s->verify = cuda_env_truthy(getenv("DS4_EXPERT_COLD_RAM_VERIFY"));
        s->verbose = cuda_env_truthy(getenv("DS4_EXPERT_COLD_RAM_VERBOSE"));
        if (s->enabled && !s->atexit_registered) {
            atexit(cuda_expert_cold_lossless_atexit);
            s->atexit_registered = 1;
        }
    }
    if (s->enabled && !s->notice_printed) {
        const char *fmt = s->format == CUDA_EXPERT_COLD_FORMAT_CQ1 ? "cq1" : "lossless";
        fprintf(stderr,
                "ds4: cold RAM sidecar enabled "
                "(format=%s group=%u prefill=%d ungated=%d native_tokens=%u "
                "native_layers=%u verify=%d)\n",
                fmt,
                s->cq1_group,
                s->allow_prefill,
                s->allow_ungated,
                s->native_decode_tokens,
                s->native_decode_layers,
                s->verify);
        s->notice_printed = 1;
    }
    return s->enabled;
}

static uint64_t cuda_expert_cold_lossless_checksum(
        const cuda_expert_cold_lossless_entry *entry) {
    if (!entry || !entry->blob) return 0;
    uint64_t h = cuda_expert_cold_lossless_hash_bytes(entry->blob,
                                                      entry->blob_bytes,
                                                      1469598103934665603ull);
    h = cuda_expert_cold_lossless_mix(h, entry->format);
    h = cuda_expert_cold_lossless_mix(h, entry->cq1_group);
    return h;
}

static cuda_expert_cold_lossless_entry *cuda_expert_cold_lossless_find(
        uint64_t key,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t format,
        uint32_t cq1_group) {
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    std::unordered_map<uint64_t, uint32_t>::iterator it = s->index.find(key);
    if (it == s->index.end()) return NULL;
    if ((uint64_t)it->second >= s->entries.size()) return NULL;
    cuda_expert_cold_lossless_entry *entry = &s->entries[it->second];
    if (entry->key == key &&
        entry->model_map == model_map &&
        entry->model_size == model_size &&
        entry->layer == layer &&
        entry->n_total_expert == n_total_expert &&
        entry->expert == expert &&
        entry->gate_offset == gate_offset &&
        entry->up_offset == up_offset &&
        entry->down_offset == down_offset &&
        entry->gate_expert_bytes == gate_expert_bytes &&
        entry->down_expert_bytes == down_expert_bytes &&
        entry->format == format &&
        entry->cq1_group == cq1_group) {
        return entry;
    }
    return NULL;
}

static cuda_expert_cold_lossless_entry *cuda_expert_cold_lossless_materialize(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if (!model_map || expert >= n_total_expert ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_expert_bytes > UINT64_MAX / 2u ||
        gate_expert_bytes * 2u > UINT64_MAX - down_expert_bytes) {
        s->materialize_failures++;
        return NULL;
    }
    uint64_t blob_bytes = gate_expert_bytes * 2u + down_expert_bytes;
    uint64_t gate_cq1_bytes = 0;
    uint64_t down_cq1_bytes = 0;
    uint64_t gate_blocks = 0;
    uint64_t down_blocks = 0;
    uint64_t cq1_block_bytes = 0;
    if (s->format == CUDA_EXPERT_COLD_FORMAT_CQ1) {
        cq1_block_bytes = cuda_expert_cq1_block_bytes(s->cq1_group);
        if (!cq1_block_bytes ||
            (gate_expert_bytes % sizeof(cuda_block_iq2_xxs)) != 0 ||
            (down_expert_bytes % sizeof(cuda_block_q2_K)) != 0) {
            s->materialize_failures++;
            return NULL;
        }
        gate_blocks = gate_expert_bytes / sizeof(cuda_block_iq2_xxs);
        down_blocks = down_expert_bytes / sizeof(cuda_block_q2_K);
        if (gate_blocks > UINT64_MAX / cq1_block_bytes ||
            down_blocks > UINT64_MAX / cq1_block_bytes) {
            s->materialize_failures++;
            return NULL;
        }
        gate_cq1_bytes = gate_blocks * cq1_block_bytes;
        down_cq1_bytes = down_blocks * cq1_block_bytes;
        if (gate_cq1_bytes > UINT64_MAX / 2u ||
            gate_cq1_bytes * 2u > UINT64_MAX - down_cq1_bytes) {
            s->materialize_failures++;
            return NULL;
        }
        blob_bytes = gate_cq1_bytes * 2u + down_cq1_bytes;
    }
    if (blob_bytes > (uint64_t)SIZE_MAX) {
        s->materialize_failures++;
        return NULL;
    }
    const uint64_t gate_src = gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src = up_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src = down_offset + (uint64_t)expert * down_expert_bytes;
    if (gate_src > model_size || up_src > model_size || down_src > model_size ||
        gate_expert_bytes > model_size - gate_src ||
        gate_expert_bytes > model_size - up_src ||
        down_expert_bytes > model_size - down_src) {
        s->materialize_failures++;
        return NULL;
    }
    char *blob = (char *)malloc((size_t)blob_bytes);
    if (!blob) {
        s->materialize_failures++;
        return NULL;
    }
    if (s->format == CUDA_EXPERT_COLD_FORMAT_CQ1) {
        uint64_t iq2_grid[256];
        uint8_t iq2_signs[128];
        if (!cuda_expert_cq1_fetch_iq2_tables(iq2_grid, iq2_signs) ||
            !cuda_expert_cq1_encode_blocks((const char *)model_map + gate_src,
                                           sizeof(cuda_block_iq2_xxs),
                                           gate_blocks,
                                           blob,
                                           s->cq1_group,
                                           1,
                                           iq2_grid,
                                           iq2_signs) ||
            !cuda_expert_cq1_encode_blocks((const char *)model_map + up_src,
                                           sizeof(cuda_block_iq2_xxs),
                                           gate_blocks,
                                           blob + gate_cq1_bytes,
                                           s->cq1_group,
                                           1,
                                           iq2_grid,
                                           iq2_signs) ||
            !cuda_expert_cq1_encode_blocks((const char *)model_map + down_src,
                                           sizeof(cuda_block_q2_K),
                                           down_blocks,
                                           blob + gate_cq1_bytes * 2u,
                                           s->cq1_group,
                                           0,
                                           iq2_grid,
                                           iq2_signs)) {
            free(blob);
            s->materialize_failures++;
            return NULL;
        }
    } else {
        memcpy(blob, (const char *)model_map + gate_src, (size_t)gate_expert_bytes);
        memcpy(blob + gate_expert_bytes,
               (const char *)model_map + up_src,
               (size_t)gate_expert_bytes);
        memcpy(blob + gate_expert_bytes * 2u,
               (const char *)model_map + down_src,
               (size_t)down_expert_bytes);
    }

    cuda_expert_cold_lossless_entry entry;
    memset(&entry, 0, sizeof(entry));
    entry.key = cuda_expert_cold_lossless_key(model_map,
                                              layer,
                                              expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              s->format,
                                              s->cq1_group);
    entry.model_map = model_map;
    entry.model_size = model_size;
    entry.layer = layer;
    entry.n_total_expert = n_total_expert;
    entry.expert = expert;
    entry.gate_offset = gate_offset;
    entry.up_offset = up_offset;
    entry.down_offset = down_offset;
    entry.gate_expert_bytes = gate_expert_bytes;
    entry.down_expert_bytes = down_expert_bytes;
    entry.blob_bytes = blob_bytes;
    entry.format = s->format;
    entry.cq1_group = s->cq1_group;
    entry.blob = blob;
    entry.checksum = cuda_expert_cold_lossless_checksum(&entry);
    const uint32_t index = (uint32_t)s->entries.size();
    s->entries.push_back(entry);
    s->index[entry.key] = index;
    s->materialized++;
    s->bytes_materialized += blob_bytes;
    return &s->entries[index];
}

static char *cuda_expert_cold_lossless_scratch(uint64_t bytes) {
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) return NULL;
    if (s->scratch && s->scratch_bytes >= bytes) return s->scratch;
    char *next = (char *)realloc(s->scratch, (size_t)bytes);
    if (!next) {
        s->repack_failures++;
        return NULL;
    }
    s->scratch = next;
    s->scratch_bytes = bytes;
    return s->scratch;
}

static int cuda_expert_cold_lossless_repack_cq1(
        const cuda_expert_cold_lossless_entry *entry,
        char **gate_src,
        char **up_src,
        char **down_src) {
    if (!entry || !gate_src || !up_src || !down_src ||
        entry->format != CUDA_EXPERT_COLD_FORMAT_CQ1) {
        return 0;
    }
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if ((entry->gate_expert_bytes % sizeof(cuda_block_iq2_xxs)) != 0 ||
        (entry->down_expert_bytes % sizeof(cuda_block_q2_K)) != 0 ||
        entry->gate_expert_bytes > UINT64_MAX / 2u ||
        entry->gate_expert_bytes * 2u > UINT64_MAX - entry->down_expert_bytes) {
        s->repack_failures++;
        return 0;
    }
    const uint64_t native_bytes =
        entry->gate_expert_bytes * 2u + entry->down_expert_bytes;
    char *scratch = cuda_expert_cold_lossless_scratch(native_bytes);
    if (!scratch) return 0;

    const uint64_t cq1_block_bytes = cuda_expert_cq1_block_bytes(entry->cq1_group);
    const uint64_t gate_blocks = entry->gate_expert_bytes / sizeof(cuda_block_iq2_xxs);
    const uint64_t down_blocks = entry->down_expert_bytes / sizeof(cuda_block_q2_K);
    const uint64_t gate_cq1_bytes = gate_blocks * cq1_block_bytes;
    const uint64_t down_cq1_bytes = down_blocks * cq1_block_bytes;
    if (!cq1_block_bytes ||
        gate_cq1_bytes > entry->blob_bytes ||
        gate_cq1_bytes * 2u > entry->blob_bytes ||
        gate_cq1_bytes * 2u + down_cq1_bytes > entry->blob_bytes) {
        s->repack_failures++;
        return 0;
    }

    char *gate = scratch;
    char *up = scratch + entry->gate_expert_bytes;
    char *down = scratch + entry->gate_expert_bytes * 2u;
    if (!cuda_expert_cq1_decode_blocks(entry->blob,
                                       sizeof(cuda_block_iq2_xxs),
                                       gate_blocks,
                                       gate,
                                       entry->cq1_group,
                                       1) ||
        !cuda_expert_cq1_decode_blocks(entry->blob + gate_cq1_bytes,
                                       sizeof(cuda_block_iq2_xxs),
                                       gate_blocks,
                                       up,
                                       entry->cq1_group,
                                       1) ||
        !cuda_expert_cq1_decode_blocks(entry->blob + gate_cq1_bytes * 2u,
                                       sizeof(cuda_block_q2_K),
                                       down_blocks,
                                       down,
                                       entry->cq1_group,
                                       0)) {
        s->repack_failures++;
        return 0;
    }
    s->bytes_repacked += native_bytes;
    *gate_src = gate;
    *up_src = up;
    *down_src = down;
    return 1;
}

static int cuda_expert_cold_lossless_copy_to_device(
        char *gate_dst,
        char *up_dst,
        char *down_dst,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        int prompt_like_batch) {
    if (!cuda_expert_cold_lossless_is_enabled()) return 0;
    cuda_expert_cold_lossless_stats *s = &g_expert_cold_lossless;
    if (prompt_like_batch && !s->allow_prefill) return 0;
    if (s->format == CUDA_EXPERT_COLD_FORMAT_CQ1 &&
        !s->allow_prefill &&
        !s->allow_ungated) {
        int reason = CUDA_EXPERT_COLD_POLICY_ALLOW;
        if (!cuda_expert_cold_cq1_policy_allows(layer,
                                                expert,
                                                prompt_like_batch,
                                                s->native_decode_tokens,
                                                s->native_decode_layers,
                                                &reason)) {
            cuda_expert_cold_policy_note(s, reason);
            return 0;
        }
        cuda_expert_cold_policy_note(s, CUDA_EXPERT_COLD_POLICY_ALLOW);
    }
    const uint64_t key = cuda_expert_cold_lossless_key(model_map,
                                                       layer,
                                                       expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       s->format,
                                                       s->cq1_group);
    cuda_expert_cold_lossless_entry *entry =
        cuda_expert_cold_lossless_find(key,
                                       model_map,
                                       model_size,
                                       layer,
                                       n_total_expert,
                                       expert,
                                       gate_offset,
                                       up_offset,
                                       down_offset,
                                       gate_expert_bytes,
                                       down_expert_bytes,
                                       s->format,
                                       s->cq1_group);
    if (entry) {
        s->reuses++;
    } else {
        entry = cuda_expert_cold_lossless_materialize(model_map,
                                                      model_size,
                                                      layer,
                                                      n_total_expert,
                                                      expert,
                                                      gate_offset,
                                                      up_offset,
                                                      down_offset,
                                                      gate_expert_bytes,
                                                      down_expert_bytes);
        if (!entry) return 0;
    }
    if (s->verify &&
        cuda_expert_cold_lossless_checksum(entry) != entry->checksum) {
        s->verify_failures++;
        return 0;
    }
    char *repacked_gate = NULL;
    char *repacked_up = NULL;
    char *repacked_down = NULL;
    const char *gate_src = entry->blob;
    const char *up_src = entry->blob + entry->gate_expert_bytes;
    const char *down_src = entry->blob + entry->gate_expert_bytes * 2u;
    if (entry->format == CUDA_EXPERT_COLD_FORMAT_CQ1) {
        if (!cuda_expert_cold_lossless_repack_cq1(entry,
                                                  &repacked_gate,
                                                  &repacked_up,
                                                  &repacked_down)) {
            return 0;
        }
        gate_src = repacked_gate;
        up_src = repacked_up;
        down_src = repacked_down;
    }
    if (!cuda_ok(cudaMemcpy(gate_dst,
                            gate_src,
                            (size_t)entry->gate_expert_bytes,
                            cudaMemcpyHostToDevice),
                 "cold lossless gate copy") ||
        !cuda_ok(cudaMemcpy(up_dst,
                            up_src,
                            (size_t)entry->gate_expert_bytes,
                            cudaMemcpyHostToDevice),
                 "cold lossless up copy") ||
        !cuda_ok(cudaMemcpy(down_dst,
                            down_src,
                            (size_t)entry->down_expert_bytes,
                            cudaMemcpyHostToDevice),
                 "cold lossless down copy")) {
        s->copy_failures++;
        return 0;
    }
    s->copies++;
    return 1;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last_bytes = UINT64_MAX;
    g_model_load_progress_last_cgib = UINT64_MAX;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_finish(void) {
    if (!g_model_load_progress_started) return;
    if (g_model_load_progress_tty) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    g_model_load_progress_started = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    const int tty = isatty(STDERR_FILENO) != 0;
    const uint64_t step = (tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t display_cgib =
        cached_bytes > (UINT64_MAX - gib / 2ull) / 100ull ?
        UINT64_MAX : (cached_bytes * 100ull + gib / 2ull) / gib;
    if (g_model_load_progress_next == 0) {
        g_model_load_progress_next = step;
    }
    if (g_model_load_progress_last != 0.0 &&
        (cached_bytes == g_model_load_progress_last_bytes ||
         display_cgib == g_model_load_progress_last_cgib)) {
        return;
    }
    if (g_model_load_progress_last != 0.0 &&
        cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (tty ? 2.0 : 10.0)) {
        return;
    }

    g_model_load_progress_started = 1;
    g_model_load_progress_tty = tty;
    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r\033[Kds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        if (g_model_load_progress_last == 0.0) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                    (double)cached_bytes / 1073741824.0);
        }
    }
    fflush(stderr);
    g_model_load_progress_last_bytes = cached_bytes;
    g_model_load_progress_last_cgib = display_cgib;
    g_model_load_progress_last = now;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("DS4_CUDA_COPY_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("DS4_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
        return gb * 1073741824ull;
    }
    /* One Spark can run the IQ2 model (~81 GiB) and the mixed q2/q4 model
     * (~91 GiB) via the old startup tensor cache.  Keep enough headroom for
     * scratch, KV, and optional Q8->F16 buffers, and make the full-Q4 model
     * use distributed layer loading unless the operator opts into a larger
     * cache budget explicitly. */
    return 96ull * 1073741824ull;
}

static uint64_t cuda_model_local_model_limit_bytes(void) {
    const uint64_t default_limit = 96ull * 1073741824ull;
    if (!cuda_model_cache_limit_explicit()) return default_limit;
    const uint64_t explicit_limit = cuda_model_cache_limit_bytes();
    return explicit_limit > default_limit ? explicit_limit : default_limit;
}

static int cuda_model_cache_limit_explicit(void) {
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    return env && env[0];
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (need > bytes / 2u) {
        const uint64_t align = 64ull * 1048576ull;
        return (need + align - 1u) & ~(align - 1u);
    }
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

/* A raw host pointer is safe for kernels only after CUDA owns, registered, or
 * HMM-prefetched the mapping.  Otherwise let the caller try per-range mapping
 * or a device copy instead of surfacing an async illegal access later. */
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    cuda_model_load_progress_finish();
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
}

static void cuda_stream_selected_cache_invalidate(void) {
    g_stream_selected_cache.valid = 0;
}

static void cuda_stream_selected_cache_release(void) {
    if (g_stream_selected_cache.gate_ptr) {
        (void)cudaFree(g_stream_selected_cache.gate_ptr);
    }
    if (g_stream_selected_cache.up_ptr) {
        (void)cudaFree(g_stream_selected_cache.up_ptr);
    }
    if (g_stream_selected_cache.down_ptr) {
        (void)cudaFree(g_stream_selected_cache.down_ptr);
    }
    if (g_stream_selected_cache.slot_selected_ptr) {
        (void)cudaFree(g_stream_selected_cache.slot_selected_ptr);
    }
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
}

static void cuda_inplace_reset_all(void); /* 0036 fwd decl (defined below) */

static void cuda_stream_expert_cache_release_all(void) {
    /* 0036: the resident slab is about to be freed/reallocated -- drop every
     * in-place slot map so a stale index can never point into a moved slab. */
    cuda_inplace_reset_all();
    if (g_stream_expert_cache.gate_ptr) {
        (void)cudaFree(g_stream_expert_cache.gate_ptr);
    }
    if (g_stream_expert_cache.up_ptr) {
        (void)cudaFree(g_stream_expert_cache.up_ptr);
    }
    if (g_stream_expert_cache.down_ptr) {
        (void)cudaFree(g_stream_expert_cache.down_ptr);
    }
    g_stream_expert_cache.slots.clear();
    memset(&g_stream_expert_cache, 0, sizeof(g_stream_expert_cache));
}

static void cuda_stream_expert_cache_invalidate(void) {
    for (cuda_stream_expert_cache_slot &slot : g_stream_expert_cache.slots) {
        slot.valid = 0;
        slot.pinned = 0;
        slot.reap_spex_pinned = 0;
    }
    g_stream_expert_cache.valid = 0;
    g_stream_expert_cache.count = 0;
    g_stream_expert_cache.tick = 0;
    g_stream_expert_cache.pinned_count = 0;
    g_stream_expert_cache.reap_pin_epoch_seen = 0;
}

static uint32_t cuda_stream_expert_cache_requested_budget(void) {
    uint32_t cap = g_stream_expert_budget_override != 0 ?
        g_stream_expert_budget_override : DS4_CUDA_STREAM_EXPERT_DEFAULT;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            cap = v > DS4_CUDA_STREAM_EXPERT_MAX ?
                DS4_CUDA_STREAM_EXPERT_MAX : (uint32_t)v;
        }
    }
    if (cap > DS4_CUDA_STREAM_EXPERT_MAX) {
        cap = DS4_CUDA_STREAM_EXPERT_MAX;
    }
    return cap;
}

static uint32_t cuda_stream_expert_cache_configured_budget(void) {
    uint32_t cap = cuda_stream_expert_cache_requested_budget();
    if (g_stream_expert_runtime_cap != 0 && cap > g_stream_expert_runtime_cap) {
        cap = g_stream_expert_runtime_cap;
    }
    return cap;
}

static int cuda_stream_expert_cache_budget_visible_to_shared(void) {
    if (!g_ssd_streaming_mode) return 0;
    if (g_stream_expert_budget_override != 0) return 1;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) return 1;
    env = getenv("DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST");
    if (!env || !env[0]) {
        env = getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST");
    }
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t cuda_stream_expert_cache_reserve_bytes(void) {
    /* 0024 resident-hit fix: the old default was 16 GiB parsed with strtoull
     * (integer only).  On a 12 GiB card 16 GiB is clamped to VRAM/2 = 6 GiB by
     * the caller, which exceeds free VRAM after the resident model tensors ->
     * the streaming expert cache is never allocated and every selected expert
     * falls to a direct H2D copy (resident hit ~= 0).  Worse, a fractional
     * override such as "0.25" was silently ignored by strtoull (it stops at the
     * '.', the *end=='\0' guard fails, gb stays 16), so operators who lowered
     * the reserve still hit the 6 GiB clamp.  Parse the override as a float and
     * default to a small floor so the cache actually fits on small cards. */
    double gb = 0.5;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        double v = strtod(env, &end);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0' && v >= 0.0) {
            gb = v;
        }
    }
    if (gb < 0.0) gb = 0.0;
    const double bytes = gb * 1073741824.0;
    if (bytes >= (double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)bytes;
}

static uint32_t cuda_stream_expert_cache_live_budget(
        uint32_t requested,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint64_t reclaim_bytes,
        int report) {
    if (requested == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    const uint64_t per_expert_bytes =
        gate_expert_bytes * 2ull + down_expert_bytes;
    if (per_expert_bytes == 0) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming expert cache memory query failed: %s; "
                "using direct selected loads\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    uint64_t free_bytes = (uint64_t)free_b;
    if (reclaim_bytes > UINT64_MAX - free_bytes) {
        free_bytes = UINT64_MAX;
    } else {
        free_bytes += reclaim_bytes;
    }
    uint64_t reserve = cuda_stream_expert_cache_reserve_bytes();
    const uint64_t total_bytes = (uint64_t)total_b;
    if (total_bytes != 0 && reserve > total_bytes / 2ull) {
        reserve = total_bytes / 2ull;
    }
    if (free_bytes <= reserve) {
        if (report && g_stream_expert_memory_cap_notice != requested) {
            cuda_model_load_progress_finish();
            fprintf(stderr,
                    "ds4: CUDA streaming expert cache disabled: available %.2f GiB <= reserve %.2f GiB\n",
                    (double)free_bytes / 1073741824.0,
                    (double)reserve / 1073741824.0);
            g_stream_expert_memory_cap_notice = requested;
        }
        return 0;
    }

    uint64_t usable = free_bytes - reserve;
    uint64_t max_slots64 = usable / per_expert_bytes;
    if (max_slots64 > UINT32_MAX) max_slots64 = UINT32_MAX;
    uint32_t capped = requested;
    if ((uint64_t)capped > max_slots64) capped = (uint32_t)max_slots64;
    if (report && capped != requested && g_stream_expert_memory_cap_notice != capped) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming expert cache capped from %u to %u experts "
                "(available %.2f GiB, reserve %.2f GiB, %.2f MiB/expert)\n",
                requested,
                capped,
                (double)free_bytes / 1073741824.0,
                (double)reserve / 1073741824.0,
                (double)per_expert_bytes / 1048576.0);
        g_stream_expert_memory_cap_notice = capped;
    }
    return capped;
}

static uint64_t cuda_stream_expert_cache_expert_bytes(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    return gate_expert_bytes * 2ull + down_expert_bytes;
}

static void cuda_stream_expert_cache_note_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (g_stream_expert_runtime_gate_bytes == gate_expert_bytes &&
        g_stream_expert_runtime_down_bytes == down_expert_bytes) {
        return;
    }
    g_stream_expert_runtime_gate_bytes = gate_expert_bytes;
    g_stream_expert_runtime_down_bytes = down_expert_bytes;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_memory_cap_notice = 0;
}

static uint32_t cuda_stream_expert_cache_shrunken_cap(uint32_t cap) {
    if (cap == 0) return 0;
    const uint32_t release = (cap + 9u) / 10u;
    return cap > release ? cap - release : 0;
}

static void cuda_stream_expert_cache_note_oom_cap(
        uint32_t failed_cap,
        uint32_t new_cap,
        uint64_t expert_bytes,
        const char *errstr) {
    if (g_stream_expert_runtime_cap != 0 &&
        g_stream_expert_runtime_cap <= new_cap) {
        return;
    }
    g_stream_expert_runtime_cap = new_cap;
    const uint32_t released =
        failed_cap > new_cap ? failed_cap - new_cap : 0;
    cuda_model_load_progress_finish();
    fprintf(stderr,
            "ds4: CUDA streaming expert cache allocation failed at %u experts "
            "/ %.2f GiB%s%s\n",
            failed_cap,
            expert_bytes != 0 ?
                (double)((uint64_t)failed_cap * expert_bytes) / 1073741824.0 :
                0.0,
            errstr && errstr[0] ? ": " : "",
            errstr && errstr[0] ? errstr : "");
    if (new_cap != 0) {
        fprintf(stderr,
                "ds4:   shrinking resident cache margin by %u experts / %.2f GiB; "
                "runtime cache cap now %u experts\n",
                released,
                expert_bytes != 0 ?
                    (double)((uint64_t)released * expert_bytes) / 1073741824.0 :
                    0.0,
                new_cap);
    } else {
        fprintf(stderr,
                "ds4:   disabling resident expert cache after OOM; using direct selected loads\n");
    }
}

static int cuda_stream_expert_cache_try_alloc(
        uint32_t cap,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        char **gate_ptr,
        char **up_ptr,
        char **down_ptr,
        const char **errstr) {
    *gate_ptr = NULL;
    *up_ptr = NULL;
    *down_ptr = NULL;
    if (errstr) *errstr = NULL;
    if (cap == 0 ||
        (uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = (uint64_t)cap * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)cap * down_expert_bytes;

    void *gate = NULL;
    void *up = NULL;
    void *down = NULL;
    cudaError_t err = cudaMalloc(&gate, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&up, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&down, (size_t)down_bytes);
    if (err != cudaSuccess) goto fail;

    *gate_ptr = (char *)gate;
    *up_ptr = (char *)up;
    *down_ptr = (char *)down;
    return 1;

fail:
    if (errstr) *errstr = cudaGetErrorString(err);
    (void)cudaGetLastError();
    if (gate) (void)cudaFree(gate);
    if (up) (void)cudaFree(up);
    if (down) (void)cudaFree(down);
    return 0;
}

static void cuda_stream_selected_stage_release(void) {
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (g_stream_selected_upload_stream) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream);
        g_stream_selected_upload_stream = NULL;
    }
    if (g_stream_selected_upload_done_event) {
        (void)cudaEventDestroy(g_stream_selected_upload_done_event);
        g_stream_selected_upload_done_event = NULL;
    }
    g_stream_selected_upload_done_recorded = 0;
}

static int cuda_stream_selected_stage_pool_alloc(uint64_t bytes) {
    if (g_stream_selected_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (!g_stream_selected_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_stream_selected_upload_stream,
                                                    cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_stream_selected_stage_raw[i],
                                         (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_stage[i] =
            cuda_align_ptr(g_stream_selected_stage_raw[i],
                           g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_stream_selected_stage_event[i],
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_stream_selected_stage_bytes = bytes;
    return 1;
}

static int cuda_stream_selected_upload_event_enabled(void) {
    if (g_stream_selected_upload_event_enabled < 0) {
        const char *v = getenv("DS4_SELECTED_UPLOAD_EVENT");
        if (!v) v = getenv("DS4_CUDA_SELECTED_UPLOAD_EVENT");
        g_stream_selected_upload_event_enabled =
            (v && v[0] && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return g_stream_selected_upload_event_enabled;
}

static int cuda_stream_selected_upload_done_event_ensure(void) {
    if (g_stream_selected_upload_done_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_selected_upload_done_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_upload_wait_if_recorded(const char *what) {
    if (!cuda_stream_selected_upload_event_enabled()) return 1;
    if (!g_stream_selected_upload_done_recorded ||
        !g_stream_selected_upload_done_event) {
        return 1;
    }
    cudaError_t err =
        cudaStreamWaitEvent((cudaStream_t)0,
                            g_stream_selected_upload_done_event,
                            0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload wait failed for %s: %s\n",
                what ? what : "selected upload",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_upload_done_recorded = 0;
    return 1;
}

static int cuda_stream_selected_ensure_bytes(
        char **ptr,
        uint64_t *capacity,
        uint64_t bytes,
        const char *what) {
    if (bytes == 0) return 1;
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%.2f MiB): %s\n",
                what ? what : "experts",
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (char *)dev;
    *capacity = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_i32(
        int32_t **ptr,
        uint64_t *capacity,
        uint64_t count,
        const char *what) {
    if (count == 0 || count > UINT64_MAX / sizeof(int32_t)) return 0;
    const uint64_t bytes = count * sizeof(int32_t);
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%u entries): %s\n",
                what ? what : "selected slots",
                (unsigned)count,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (int32_t *)dev;
    *capacity = bytes;
    return 1;
}

static cuda_stream_expert_cache *cuda_stream_expert_cache_prepare(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t target_cap) {
    const uint64_t expert_bytes =
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes);
    if (expert_bytes == 0) return NULL;
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);

    const uint32_t requested_cap = cuda_stream_expert_cache_configured_budget();
    if (requested_cap == 0) return NULL;
    if (target_cap == 0 || target_cap > requested_cap) target_cap = requested_cap;
    if (target_cap == 0) return NULL;
    const int same_dims =
        g_stream_expert_cache.valid &&
        g_stream_expert_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_expert_cache.down_expert_bytes == down_expert_bytes;
    if (!same_dims && g_stream_expert_cache.valid) {
        cuda_stream_expert_cache_release_all();
    }
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= target_cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity) {
        return &g_stream_expert_cache;
    }

    uint64_t reclaim_bytes = 0;
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        (uint64_t)g_stream_expert_cache.capacity <= UINT64_MAX / expert_bytes) {
        reclaim_bytes = (uint64_t)g_stream_expert_cache.capacity * expert_bytes;
    }
    uint32_t cap =
        cuda_stream_expert_cache_live_budget(target_cap,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             reclaim_bytes,
                                             reclaim_bytes == 0);
    if (cap == 0) return NULL;
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity) {
        return &g_stream_expert_cache;
    }

    cuda_stream_expert_cache_release_all();
    while (cap != 0) {
        if ((uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
            (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: CUDA streaming expert cache size overflow\n");
            return NULL;
        }

        char *gate_ptr = NULL;
        char *up_ptr = NULL;
        char *down_ptr = NULL;
        const char *alloc_error = NULL;
        if (!cuda_stream_expert_cache_try_alloc(cap,
                                                gate_expert_bytes,
                                                down_expert_bytes,
                                                &gate_ptr,
                                                &up_ptr,
                                                &down_ptr,
                                                &alloc_error)) {
            const uint32_t new_cap =
                cuda_stream_expert_cache_shrunken_cap(cap);
            cuda_stream_expert_cache_note_oom_cap(cap,
                                                  new_cap,
                                                  expert_bytes,
                                                  alloc_error);
            cap = new_cap;
            if (cap != 0) {
                cap = cuda_stream_expert_cache_live_budget(cap,
                                                           gate_expert_bytes,
                                                           down_expert_bytes,
                                                           0,
                                                           1);
            }
            continue;
        }

        try {
            g_stream_expert_cache.slots.resize(cap);
        } catch (...) {
            fprintf(stderr, "ds4: CUDA streaming expert cache metadata allocation failed\n");
            (void)cudaFree(gate_ptr);
            (void)cudaFree(up_ptr);
            (void)cudaFree(down_ptr);
            cuda_stream_expert_cache_release_all();
            return NULL;
        }

        g_stream_expert_cache.valid = 1;
        g_stream_expert_cache.capacity = cap;
        if (getenv("DS4_CACHEFIX_TRACE"))
            fprintf(stderr, "ds4: cachefix cap=%u requested=%u\n", cap, requested_cap);
        g_stream_expert_cache.count = 0;
        g_stream_expert_cache.tick = 0;
        g_stream_expert_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_expert_cache.down_expert_bytes = down_expert_bytes;
        g_stream_expert_cache.gate_ptr = gate_ptr;
        g_stream_expert_cache.up_ptr = up_ptr;
        g_stream_expert_cache.down_ptr = down_ptr;
        g_stream_expert_cache.gate_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.up_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.down_capacity =
            (uint64_t)cap * down_expert_bytes;
        return &g_stream_expert_cache;
    }
    return NULL;
}

static int cuda_stream_expert_cache_find(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cache || !cache->valid) return -1;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        const cuda_stream_expert_cache_slot &slot = cache->slots[i];
        if (slot.valid &&
            slot.model_map == model_map &&
            slot.model_size == model_size &&
            slot.layer == layer &&
            slot.n_total_expert == n_total_expert &&
            slot.expert == expert &&
            slot.gate_offset == gate_offset &&
            slot.up_offset == up_offset &&
            slot.down_offset == down_offset &&
            slot.gate_expert_bytes == gate_expert_bytes &&
            slot.down_expert_bytes == down_expert_bytes) {
            if (cuda_spex_stats_is_enabled()) g_spex_stats.expert_cache_hits++;
            return (int)i;
        }
    }
    if (cuda_spex_stats_is_enabled()) g_spex_stats.expert_cache_misses++;
    return -1;
}

/* ---- 0031 pin-keep / residency-rotation ---------------------------------
 * Binds the (post-mask) demand stream to VRAM residency WITHOUT touching the
 * router mask or selection.  A per-slot `pinned` flag makes the LRU victim
 * selector skip a hot subset of the streaming expert cache, so those experts
 * stay resident across tokens instead of being re-copied H2D every step.  The
 * pinned set is chosen at a warmup "freeze" from a cache-local demand EWMA
 * (an rmass-0020 analog computed from the compact_ids the router actually
 * requests -- under a frozen static mask that stream IS the keep set) and,
 * optionally, rotated on demand via a 0026-style CUSUM.
 *
 * Correctness invariants (declared, load-bearing):
 *  - Pinning is RESIDENCY, not SELECTION: it never writes g_reap_mask_pruned
 *    nor the router bias, so top-k selection is unchanged -> output is
 *    bit-identical to the non-pinned streaming path.  This is what separates
 *    0031 from rotate32/0015 (which re-ranked the mask and collapsed).
 *  - Pinning is REPRESENTATION-neutral: a pinned slot serves the exact bytes
 *    load_slot produced for that expert (native 2-bit via the cold-lossless
 *    path when active).  0031 never promotes an expert to a q8/f16 form.  To
 *    avoid the q8/f16 serving crack (BITEXACT.md: ~2.6x diff vs native) run
 *    with DS4_CUDA_NO_Q8_F16_CACHE=1; 0031 inherits, never overrides, that.
 *  - Composition with SPEX: SPEX predicts WHICH experts to promote (topK ->
 *    ds4_gpu_stream_expert_cache_seed_experts_async); 0031 makes that
 *    residency eviction-immune.  Seeds credit the same demand EWMA, so a
 *    predicted-hot expert gets pinned and the SPEX "skip all resident" filter
 *    then stops re-seeding it (kills the per-token seed H2D, the J30 cost).
 * All hooks are no-ops unless DS4_PACE_PIN=1, so the default engine is
 * byte-identical to post-0024. */
struct cuda_pin_config {
    int ready;
    int enabled;      /* DS4_PACE_PIN            master gate (default off) */
    int rotate;       /* DS4_PACE_PIN_ROTATE     demand rotation (default off) */
    double budget_mb; /* DS4_PACE_PIN_BUDGET_MB  pinned sub-budget of the cache */
    uint64_t warmup;  /* DS4_PACE_PIN_WARMUP     decode calls before freeze */
    double alpha;     /* DS4_PACE_PIN_EWMA       demand EWMA rate */
    double cusum_k;   /* DS4_PACE_PIN_CUSUM_K    rotation slack */
    double cusum_h;   /* DS4_PACE_PIN_CUSUM_H    rotation threshold */
    uint64_t cooldown;/* DS4_PACE_PIN_COOLDOWN   anti-thrash calls */
    FILE *log;        /* DS4_PACE_PIN_LOG        JSONL events */
};
static cuda_pin_config g_pin_cfg;

static int cuda_pin_env_flag(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}
static double cuda_pin_env_double(const char *name, double dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    char *end = NULL;
    errno = 0;
    const double x = strtod(v, &end);
    while (end && (*end == ' ' || *end == '\t')) end++;
    if (end != v && errno == 0 && end && *end == '\0') return x;
    return dflt;
}
static uint64_t cuda_pin_env_u64(const char *name, uint64_t dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    char *end = NULL;
    errno = 0;
    const unsigned long long x = strtoull(v, &end, 10);
    while (end && (*end == ' ' || *end == '\t')) end++;
    if (end != v && errno == 0 && end && *end == '\0') return (uint64_t)x;
    return dflt;
}
static const cuda_pin_config *cuda_pin_get_config(void) {
    if (g_pin_cfg.ready) return &g_pin_cfg;
    g_pin_cfg.enabled = cuda_pin_env_flag("DS4_PACE_PIN");
    g_pin_cfg.rotate = cuda_pin_env_flag("DS4_PACE_PIN_ROTATE");
    g_pin_cfg.budget_mb = cuda_pin_env_double("DS4_PACE_PIN_BUDGET_MB", 3500.0);
    if (g_pin_cfg.budget_mb < 0.0) g_pin_cfg.budget_mb = 0.0;
    g_pin_cfg.warmup = cuda_pin_env_u64("DS4_PACE_PIN_WARMUP", 512ull);
    g_pin_cfg.alpha = cuda_pin_env_double("DS4_PACE_PIN_EWMA", 0.05);
    if (g_pin_cfg.alpha <= 0.0 || g_pin_cfg.alpha > 1.0) g_pin_cfg.alpha = 0.05;
    g_pin_cfg.cusum_k = cuda_pin_env_double("DS4_PACE_PIN_CUSUM_K", 0.05);
    g_pin_cfg.cusum_h = cuda_pin_env_double("DS4_PACE_PIN_CUSUM_H", 1.0);
    if (g_pin_cfg.cusum_h <= 0.0) g_pin_cfg.cusum_h = 1.0;
    g_pin_cfg.cooldown = cuda_pin_env_u64("DS4_PACE_PIN_COOLDOWN", 128ull);
    g_pin_cfg.log = NULL;
    const char *lp = getenv("DS4_PACE_PIN_LOG");
    if (lp && lp[0]) g_pin_cfg.log = fopen(lp, "a");
    if (g_pin_cfg.enabled && getenv("DS4_CUDA_NO_Q8_F16_CACHE") == NULL) {
        fprintf(stderr,
                "ds4: DS4_PACE_PIN active but DS4_CUDA_NO_Q8_F16_CACHE is unset; "
                "0031 pins RESIDENCY only and serves each expert in its existing "
                "representation. For bit-exact 2-bit-native serving (no q8/f16 "
                "precision crack) set DS4_CUDA_NO_Q8_F16_CACHE=1.\n");
    }
    g_pin_cfg.ready = 1;
    return &g_pin_cfg;
}

/* ---- 0036 in-place resident (no-copy) decode path ----------------------
 * When DS4_CUDA_INPLACE_RESIDENT=1 the routed-MoE decode GEMM reads each
 * resident expert directly from its VRAM cache slot in g_stream_expert_cache
 * (the uniform-stride slab the kernel already indexes as base +
 * slot*expert_bytes) instead of re-gathering the routed subset into a
 * per-token compact buffer.  A per-layer resident-slot map + a per-layer dirty
 * epoch make staging INCREMENTAL: a layer whose routed expert set is unchanged
 * and still fully resident reuses the prior binding with ZERO copies, ZERO
 * finds and ZERO uploads.  All hooks are no-ops unless the env is set, so the
 * default engine is byte-identical to post-0033. */
struct cuda_inplace_config {
    int ready;
    int enabled;   /* DS4_CUDA_INPLACE_RESIDENT master gate (default off) */
    int verify;    /* DS4_CUDA_INPLACE_VERIFY    debug: re-find + assert  */
};
static cuda_inplace_config g_inplace_cfg;
static const cuda_inplace_config *cuda_inplace_get_config(void) {
    if (g_inplace_cfg.ready) return &g_inplace_cfg;
    g_inplace_cfg.enabled = cuda_pin_env_flag("DS4_CUDA_INPLACE_RESIDENT");
    g_inplace_cfg.verify = cuda_pin_env_flag("DS4_CUDA_INPLACE_VERIFY");
    g_inplace_cfg.ready = 1;
    return &g_inplace_cfg;
}
static int cuda_inplace_resident_enabled(void) {
    return cuda_inplace_get_config()->enabled;
}

/* Per-layer persistent staging state for the in-place decode path.  Keyed by
 * layer index: the expert cache is keyed by (layer,expert), so residency of
 * layer L is independent of admissions in any other layer. */
struct cuda_inplace_layer_state {
    std::vector<int32_t> prev_ids;       /* compact_ids snapshot (order-sensitive) */
    std::vector<int32_t> prev_slot_ids;  /* slot_ids snapshot (routed order)       */
    std::vector<int32_t> prev_slot_map;  /* last uploaded resident-slot map         */
    int32_t *slot_dev = NULL;            /* persistent device int32[slot_count]     */
    uint64_t slot_dev_capacity = 0;      /* element capacity of slot_dev            */
    ds4_gpu_tensor slot_dev_tensor{};    /* tensor view the GEMM reads              */
    uint32_t prev_count = 0;             /* compact_count                           */
    uint32_t slot_count = 0;             /* routed slot count (n_tokens*n_expert)   */
    uint64_t epoch_seen = 0;             /* g_inplace_epoch[layer] at last build    */
    const void *model_map = NULL;
    uint32_t n_total_expert = 0;
    uint64_t gate_offset = 0;
    uint64_t up_offset = 0;
    uint64_t down_offset = 0;
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    uint8_t valid = 0;
};
static std::vector<cuda_inplace_layer_state> g_inplace_layer;
static std::vector<uint64_t> g_inplace_epoch; /* per-layer (layer,expert)->slot dirty ctr */
/* g_inplace_reuse_hits / g_inplace_admissions / g_inplace_slot_uploads are
 * declared near g_spex_stats (used by cuda_spex_stats_print, defined earlier). */

static uint64_t cuda_inplace_epoch_get(uint32_t layer) {
    return (uint64_t)layer < g_inplace_epoch.size() ? g_inplace_epoch[layer] : 0ull;
}
static void cuda_inplace_epoch_bump(uint32_t layer) {
    if ((uint64_t)layer >= g_inplace_epoch.size()) {
        g_inplace_epoch.resize((size_t)layer + 1u, 0ull);
    }
    g_inplace_epoch[layer]++;
}
/* Invalidate ALL in-place bindings + epochs.  Called whenever the resident
 * expert-cache slab is (re)allocated -- every such path funnels through
 * cuda_stream_expert_cache_release_all -- so a stale slot map can never survive
 * a slab move. */
static void cuda_inplace_reset_all(void) {
    for (cuda_inplace_layer_state &st : g_inplace_layer) {
        if (st.slot_dev) { (void)cudaFree(st.slot_dev); }
        st.slot_dev = NULL;
        st.slot_dev_capacity = 0;
        st.valid = 0;
        st.prev_ids.clear();
        st.prev_slot_ids.clear();
        st.prev_slot_map.clear();
        memset(&st.slot_dev_tensor, 0, sizeof(st.slot_dev_tensor));
    }
    g_inplace_epoch.clear();
}

/* Freeze: pin the `budget` highest-demand resident slots (the hot keeps).  The
 * budget is a sub-budget of the ALREADY-allocated cache (pinning a slot adds no
 * VRAM -- it only forbids its eviction), and we always keep >= capacity/8 + 1
 * non-pinned slots so the LRU always has a victim to rotate through. */
static void cuda_pin_freeze(cuda_stream_expert_cache *cache) {
    const cuda_pin_config *cfg = cuda_pin_get_config();
    if (!cache || !cache->valid) return;
    const uint64_t per_expert =
        cache->gate_expert_bytes * 2ull + cache->down_expert_bytes;
    uint32_t budget = 0;
    if (per_expert != 0 && cfg->budget_mb > 0.0) {
        const double b = (cfg->budget_mb * 1048576.0) / (double)per_expert;
        if (b >= (double)cache->capacity) budget = cache->capacity;
        else if (b > 0.0) budget = (uint32_t)b;
    }
    const uint32_t reserve_free = cache->capacity / 8u + 1u;
    if (budget + reserve_free > cache->capacity) {
        budget = cache->capacity > reserve_free ?
            cache->capacity - reserve_free : 0u;
    }
    cache->pin_budget_slots = budget;
    cache->pinned_count = 0;
    cache->pin_frozen = 1;
    for (uint32_t k = 0; k < budget; k++) {
        int best = -1;
        double best_demand = 0.0;
        for (uint32_t i = 0; i < cache->capacity; i++) {
            cuda_stream_expert_cache_slot &s = cache->slots[i];
            if (!s.valid || s.pinned) continue;
            if (best < 0 || s.demand > best_demand) {
                best_demand = s.demand;
                best = (int)i;
            }
        }
        if (best < 0 || best_demand <= 0.0) break; /* no more warm keeps */
        cache->slots[(uint32_t)best].pinned = 1;
        cache->slots[(uint32_t)best].cusum = 0.0;
        cache->pinned_count++;
    }
    if (cfg->log) {
        fprintf(cfg->log,
                "{\"event\":\"pin_freeze\",\"pinned\":%u,\"budget_slots\":%u,"
                "\"capacity\":%u,\"per_expert_mib\":%.2f}\n",
                cache->pinned_count, cache->pin_budget_slots, cache->capacity,
                (double)per_expert / 1048576.0);
        fflush(cfg->log);
    }
}

/* Once per decode expert-load call: advance the demand clock (decay this
 * layer's EWMA), refresh the cached coldest-pinned slot (rotation floor +
 * victim), and freeze the pinned set when warmup elapses. */
static void cuda_pin_begin_call(cuda_stream_expert_cache *cache, uint32_t layer) {
    const cuda_pin_config *cfg = cuda_pin_get_config();
    if (!cfg->enabled || !cache || !cache->valid) return;
    const double keep = 1.0 - cfg->alpha;
    cache->pin_min_valid = 0;
    cache->pin_min_demand = 0.0;
    cache->pin_min_slot = 0;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        if (!s.valid) continue;
        if (s.layer == layer) s.demand *= keep;
        if (s.pinned) {
            if (!cache->pin_min_valid || s.demand < cache->pin_min_demand) {
                cache->pin_min_demand = s.demand;
                cache->pin_min_slot = i;
                cache->pin_min_valid = 1;
            }
        }
    }
    cache->pin_calls++;
    if (!cache->pin_frozen && cache->pin_calls >= cfg->warmup &&
        !cuda_inplace_resident_enabled()) {
        /* 0036: under in-place residency, pin-on-admit (load_slot) owns the
         * pinned set from first touch; the warmup EWMA freeze would fight it
         * (stale snapshot + reserve clamp), so it is disabled. */
        cuda_pin_freeze(cache);
    }
}

/* Credit demand for a resident expert (decode HIT/LOAD or SPEX seed) and, when
 * rotation is enabled and the mask is frozen, run the CUSUM that swaps the
 * residency pin from the coldest pinned expert to a persistently hotter
 * non-pinned one.  Both are already resident -> the swap is a flag toggle, no
 * copy; the mask/selection is untouched -> output stays bit-identical. */
static void cuda_pin_note_resident(cuda_stream_expert_cache *cache, uint32_t slot) {
    const cuda_pin_config *cfg = cuda_pin_get_config();
    if (!cfg->enabled || !cache || !cache->valid || slot >= cache->capacity) {
        return;
    }
    cuda_stream_expert_cache_slot &s = cache->slots[slot];
    s.demand += cfg->alpha;
    if (s.demand > 1.0) s.demand = 1.0;
    if (!cfg->rotate || !cache->pin_frozen || s.pinned) return;
    if (!cache->pin_min_valid) return;
    s.cusum += (s.demand - cache->pin_min_demand) - cfg->cusum_k;
    if (s.cusum < 0.0) s.cusum = 0.0;
    if (s.cusum < cfg->cusum_h) return;
    if (cache->pin_calls - cache->pin_last_rotate < cfg->cooldown) return;
    const uint32_t victim = cache->pin_min_slot;
    if (victim >= cache->capacity || !cache->slots[victim].pinned) return;
    if (s.demand <= cache->slots[victim].demand) { s.cusum = 0.0; return; }
    const uint32_t in_expert = s.expert;
    const uint32_t in_layer = s.layer;
    const uint32_t out_expert = cache->slots[victim].expert;
    const uint32_t out_layer = cache->slots[victim].layer;
    cache->slots[victim].pinned = 0;
    cache->slots[victim].cusum = 0.0;
    s.pinned = 1;
    s.cusum = 0.0;
    cache->pin_last_rotate = cache->pin_calls;
    cache->pin_rotations++;
    cache->pin_min_valid = 0; /* recompute the floor on the next call */
    if (cfg->log) {
        fprintf(cfg->log,
                "{\"event\":\"pin_rotate\",\"pinned_in\":%u,\"in_layer\":%u,"
                "\"evicted_out\":%u,\"out_layer\":%u,\"call\":%llu,"
                "\"rotations\":%llu}\n",
                in_expert, in_layer, out_expert, out_layer,
                (unsigned long long)cache->pin_calls,
                (unsigned long long)cache->pin_rotations);
        fflush(cfg->log);
    }
}

/* ---- 0033 tiered-hysteresis dynamic residency (K12 dynamic) ---------------
 * Extends 0031's single warmup "freeze" into a DYNAMIC seed + a CONTINUOUS
 * promote/evict loop with HYSTERESIS over three residency tiers:
 *   tier 2 = VRAM  (hot, pinned, eviction-immune)  -- reuses 0031's `pinned`
 *   tier 1 = RAM   (warm, resident, LRU-evictable)
 *   tier 0 = SSD   (cold, BLOCKED, preferred LRU victim -> direct-load)
 * The hot set is NOT a static mask: after the first DS4_PACE_TIER_WARMUP decode
 * expert-load calls (~the first 10-20 tokens over the layer stack) the observed
 * routing SEEDS the initial hot set; thereafter each slot climbs/descends the
 * ladder by its decayed knock count.  Promotion is GATED by SUSTAINED demand:
 *   SSD -> RAM   only after knock >= X            (DS4_PACE_TIER_X)
 *   RAM -> VRAM  only after knock >= X + Y         (+ DS4_PACE_TIER_Y)
 * so a single fluke request never promotes a cold expert -- the pathology of a
 * naive LRU/frequency cache, which promotes on every miss and thrashes (a big
 * cache measured WORSE for exactly this reason).  Only sustained knocking wins
 * VRAM residency, so the VRAM set stays small, stable and genuinely hot.
 * Demotion (VRAM->RAM->SSD) uses a hysteresis margin (DS4_PACE_TIER_HYST) so a
 * slot at the boundary does not oscillate.  The VRAM tier is capped at
 * DS4_PACE_TIER_VRAM_SLOTS (~394, the real 12 GB fit) and holds the top-by-knock
 * experts within budget: once budget is full a hotter RAM slot may DISPLACE the
 * coldest VRAM slot (only when HYST hotter and past a cooldown) -- both are
 * already resident, so the swap is a flag toggle, no copy.  Re-entry: knock
 * DECAYS (DS4_PACE_TIER_DECAY) rather than resetting on demotion, so a blocked
 * (tier-0) slot that resumes knocking climbs back up the ladder.
 *
 * Correctness invariants (same load-bearing distinction as 0031):
 *  - RESIDENCY, not SELECTION: touches only the per-slot `tier`/`pinned`/`knock`
 *    management state, never g_reap_mask_pruned nor the router bias -> top-k
 *    selection is unchanged -> output is BIT-IDENTICAL whether tiering is on,
 *    off or actively promoting.  Which slot is resident/evicted only decides
 *    HIT vs H2D re-copy of the SAME bytes, never WHICH expert is selected.
 *  - REPRESENTATION-neutral: every tier serves the exact bytes load_slot
 *    produced (native 2-bit via cold-lossless); 0033 never promotes to q8/f16.
 *    For the bit-exact 2-bit-native path run DS4_CUDA_NO_Q8_F16_CACHE=1 (0033
 *    inherits, never overrides; one-shot stderr reminder if unset).
 *  - OFF by default: every hook is a no-op unless DS4_PACE_TIER=1, so the
 *    default engine is byte-identical to post-0031.
 * DS4_PACE_TIER supersedes 0031's static freeze -- both drive the same `pinned`
 * flag, so enable one (a one-shot stderr note fires if both are set).  Known
 * approximation (shared with 0031): knock/tier live on the SLOT, not the
 * expert, so a physically evicted+reloaded expert loses its history; the seed +
 * decay self-correct and the mask/selection is untouched, so this is a residency
 * efficiency wart, never a correctness one. */
struct cuda_tier_config {
    int ready;
    int enabled;        /* DS4_PACE_TIER            master gate (default off) */
    uint64_t warmup;    /* DS4_PACE_TIER_WARMUP     calls before the dynamic seed */
    double x;           /* DS4_PACE_TIER_X          knocks for SSD->RAM */
    double y;           /* DS4_PACE_TIER_Y          extra knocks for RAM->VRAM */
    double hyst;        /* DS4_PACE_TIER_HYST       demotion hysteresis margin */
    uint32_t vram_slots;/* DS4_PACE_TIER_VRAM_SLOTS VRAM tier budget (slots) */
    double decay;       /* DS4_PACE_TIER_DECAY      per-call knock decay */
    double knock_inc;   /* DS4_PACE_TIER_KNOCK      knock increment per request */
    uint64_t cooldown;  /* DS4_PACE_TIER_COOLDOWN   anti-thrash calls between swaps */
    FILE *log;          /* DS4_PACE_TIER_LOG        JSONL events */
};
static cuda_tier_config g_tier_cfg;

static const cuda_tier_config *cuda_tier_get_config(void) {
    if (g_tier_cfg.ready) return &g_tier_cfg;
    g_tier_cfg.enabled = cuda_pin_env_flag("DS4_PACE_TIER");
    g_tier_cfg.warmup = cuda_pin_env_u64("DS4_PACE_TIER_WARMUP", 512ull);
    g_tier_cfg.x = cuda_pin_env_double("DS4_PACE_TIER_X", 3.0);
    if (g_tier_cfg.x < 0.0) g_tier_cfg.x = 0.0;
    g_tier_cfg.y = cuda_pin_env_double("DS4_PACE_TIER_Y", 5.0);
    if (g_tier_cfg.y < 0.0) g_tier_cfg.y = 0.0;
    g_tier_cfg.hyst = cuda_pin_env_double("DS4_PACE_TIER_HYST", 1.0);
    if (g_tier_cfg.hyst < 0.0) g_tier_cfg.hyst = 0.0;
    {
        const uint64_t vs = cuda_pin_env_u64("DS4_PACE_TIER_VRAM_SLOTS", 394ull);
        g_tier_cfg.vram_slots = vs > 0xffffffffull ? 0xffffffffu : (uint32_t)vs;
    }
    g_tier_cfg.decay = cuda_pin_env_double("DS4_PACE_TIER_DECAY", 0.98);
    if (g_tier_cfg.decay <= 0.0 || g_tier_cfg.decay > 1.0) g_tier_cfg.decay = 0.98;
    g_tier_cfg.knock_inc = cuda_pin_env_double("DS4_PACE_TIER_KNOCK", 1.0);
    if (g_tier_cfg.knock_inc <= 0.0) g_tier_cfg.knock_inc = 1.0;
    g_tier_cfg.cooldown = cuda_pin_env_u64("DS4_PACE_TIER_COOLDOWN", 64ull);
    g_tier_cfg.log = NULL;
    const char *lp = getenv("DS4_PACE_TIER_LOG");
    if (lp && lp[0]) g_tier_cfg.log = fopen(lp, "a");
    if (g_tier_cfg.enabled && cuda_pin_env_flag("DS4_PACE_PIN")) {
        fprintf(stderr,
                "ds4: DS4_PACE_TIER and DS4_PACE_PIN are both set; the tiered "
                "controller (0033) supersedes the 0031 static freeze and both "
                "drive the same `pinned` residency flag -- enable one.\n");
    }
    if (g_tier_cfg.enabled && getenv("DS4_CUDA_NO_Q8_F16_CACHE") == NULL) {
        fprintf(stderr,
                "ds4: DS4_PACE_TIER active but DS4_CUDA_NO_Q8_F16_CACHE is unset; "
                "0033 manages RESIDENCY tiers only and serves each expert in its "
                "existing representation. For bit-exact 2-bit-native serving set "
                "DS4_CUDA_NO_Q8_F16_CACHE=1.\n");
    }
    g_tier_cfg.ready = 1;
    return &g_tier_cfg;
}

/* Effective VRAM budget: the configured slot cap, clamped to leave >= a rotation
 * reserve of non-VRAM slots so the LRU always has a victim (a load never
 * deadlocks) -- mirrors 0031's reserve_free. */
static uint32_t cuda_tier_vram_budget(const cuda_stream_expert_cache *cache,
                                      const cuda_tier_config *cfg) {
    uint32_t budget = cfg->vram_slots;
    if (budget > cache->capacity) budget = cache->capacity;
    const uint32_t reserve = cache->capacity / 8u + 1u;
    if (budget + reserve > cache->capacity) {
        budget = cache->capacity > reserve ? cache->capacity - reserve : 0u;
    }
    return budget;
}

/* Dynamic seed (replaces 0031's one-shot freeze): after warmup, classify every
 * resident slot from the observed knock -- knock>=X -> RAM, else SSD -- then
 * promote the hottest RAM slots (knock>=X+Y) to VRAM up to budget.  The
 * continuous loop takes over afterwards. */
static void cuda_tier_seed(cuda_stream_expert_cache *cache) {
    const cuda_tier_config *cfg = cuda_tier_get_config();
    if (!cache || !cache->valid) return;
    const uint32_t budget = cuda_tier_vram_budget(cache, cfg);
    const double vram_gate = cfg->x + cfg->y;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        s.pinned = 0;
        if (!s.valid) { s.tier = 0; continue; }
        s.tier = (s.knock >= cfg->x) ? 1 : 0;
    }
    uint32_t vram = 0;
    for (uint32_t k = 0; k < budget; k++) {
        int best = -1;
        double best_knock = 0.0;
        for (uint32_t i = 0; i < cache->capacity; i++) {
            cuda_stream_expert_cache_slot &s = cache->slots[i];
            if (!s.valid || s.tier == 2 || s.knock < vram_gate) continue;
            if (best < 0 || s.knock > best_knock) {
                best_knock = s.knock;
                best = (int)i;
            }
        }
        if (best < 0) break;
        cache->slots[(uint32_t)best].tier = 2;
        cache->slots[(uint32_t)best].pinned = 1;
        vram++;
    }
    cache->tier_vram_count = vram;
    cache->tier_seeded = 1;
    if (cfg->log) {
        fprintf(cfg->log,
                "{\"event\":\"tier_seed\",\"vram\":%u,\"budget\":%u,"
                "\"capacity\":%u,\"call\":%llu}\n",
                vram, budget, cache->capacity,
                (unsigned long long)cache->tier_calls);
        fflush(cfg->log);
    }
}

/* Once per decode expert-load call: decay this layer's knock, run the DEMOTION
 * half of the loop (cooled slots descend one tier per call, with hysteresis),
 * and fire the dynamic seed when warmup elapses. */
static void cuda_tier_begin_call(cuda_stream_expert_cache *cache, uint32_t layer) {
    const cuda_tier_config *cfg = cuda_tier_get_config();
    if (!cfg->enabled || !cache || !cache->valid) return;
    cache->tier_calls++;
    if (!cache->tier_seeded) {
        for (uint32_t i = 0; i < cache->capacity; i++) {
            cuda_stream_expert_cache_slot &s = cache->slots[i];
            if (s.valid && s.layer == layer) s.knock *= cfg->decay;
        }
        /* Strategy 2 (seed-after-grow): defer the 0033 dynamic tier seed until
         * the cache has fully grown to the configured budget, so tier==2/pinned
         * classification (and the cuda_tier_note_resident promotions gated on
         * tier_seeded) never fire on a tiny prefill-phase capacity and lock in
         * prefill-transient experts.  Flag toggles only -> bit-exact. */
        if (cache->tier_calls >= cfg->warmup &&
            cache->capacity >= cuda_stream_expert_cache_configured_budget()) {
            cuda_tier_seed(cache);
        }
        return;
    }
    const double vram_keep = (cfg->x + cfg->y) - cfg->hyst;
    const double ram_keep = cfg->x - cfg->hyst;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        if (!s.valid || s.layer != layer) continue;
        s.knock *= cfg->decay;
        if (s.tier == 2) {
            if (s.knock < vram_keep) {
                s.tier = 1;
                s.pinned = 0;
                if (cache->tier_vram_count) cache->tier_vram_count--;
                cache->tier_evictions++;
                if (cfg->log) {
                    fprintf(cfg->log,
                            "{\"event\":\"tier_demote\",\"from\":2,\"to\":1,"
                            "\"expert\":%u,\"layer\":%u,\"knock\":%.4f,"
                            "\"call\":%llu}\n",
                            s.expert, s.layer, s.knock,
                            (unsigned long long)cache->tier_calls);
                    fflush(cfg->log);
                }
            }
        } else if (s.tier == 1) {
            if (s.knock < ram_keep) s.tier = 0; /* blocked back to SSD */
        }
    }
}

/* Credit a knock for a resident expert (decode HIT/LOAD or SPEX seed) and run
 * the PROMOTION half of the loop: sustained knocking climbs the ladder one step
 * per call (SSD->RAM at X, RAM->VRAM at X+Y).  When the VRAM budget is full a
 * hotter RAM slot displaces the coldest VRAM slot (top-by-knock, HYST margin,
 * cooldown-gated) -- both are already resident so the pin swap is a flag toggle,
 * no copy, and the mask/selection is untouched -> output stays bit-identical. */
static void cuda_tier_note_resident(cuda_stream_expert_cache *cache, uint32_t slot) {
    const cuda_tier_config *cfg = cuda_tier_get_config();
    if (!cfg->enabled || !cache || !cache->valid || slot >= cache->capacity) {
        return;
    }
    cuda_stream_expert_cache_slot &s = cache->slots[slot];
    s.knock += cfg->knock_inc;
    if (!cache->tier_seeded) return;          /* accumulate only until the seed */
    if (s.tier == 0) {
        if (s.knock >= cfg->x) s.tier = 1;    /* SSD -> RAM after X knocks */
        return;
    }
    if (s.tier != 1) return;                  /* already VRAM -> nothing to do */
    if (s.knock < cfg->x + cfg->y) return;    /* RAM -> VRAM needs X+Y knocks */
    const uint32_t budget = cuda_tier_vram_budget(cache, cfg);
    if (cache->tier_vram_count < budget) {
        s.tier = 2;
        s.pinned = 1;
        cache->tier_vram_count++;
        cache->tier_promotions++;
        if (cfg->log) {
            fprintf(cfg->log,
                    "{\"event\":\"tier_promote\",\"from\":1,\"to\":2,"
                    "\"expert\":%u,\"layer\":%u,\"knock\":%.4f,\"vram\":%u,"
                    "\"call\":%llu}\n",
                    s.expert, s.layer, s.knock, cache->tier_vram_count,
                    (unsigned long long)cache->tier_calls);
            fflush(cfg->log);
        }
        return;
    }
    /* Budget full: displace the coldest VRAM slot iff this one is HYST hotter. */
    if (cache->tier_calls - cache->tier_last_swap < cfg->cooldown) return;
    int victim = -1;
    double victim_knock = 0.0;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &v = cache->slots[i];
        if (!v.valid || v.tier != 2) continue;
        if (victim < 0 || v.knock < victim_knock) {
            victim_knock = v.knock;
            victim = (int)i;
        }
    }
    if (victim < 0 || s.knock <= victim_knock + cfg->hyst) return;
    const uint32_t out_expert = cache->slots[(uint32_t)victim].expert;
    const uint32_t out_layer = cache->slots[(uint32_t)victim].layer;
    cache->slots[(uint32_t)victim].tier = 1;
    cache->slots[(uint32_t)victim].pinned = 0;
    s.tier = 2;
    s.pinned = 1;
    cache->tier_last_swap = cache->tier_calls;
    cache->tier_promotions++;
    if (cfg->log) {
        fprintf(cfg->log,
                "{\"event\":\"tier_swap\",\"pinned_in\":%u,\"in_layer\":%u,"
                "\"evicted_out\":%u,\"out_layer\":%u,\"in_knock\":%.4f,"
                "\"out_knock\":%.4f,\"call\":%llu}\n",
                s.expert, s.layer, out_expert, out_layer, s.knock, victim_knock,
                (unsigned long long)cache->tier_calls);
        fflush(cfg->log);
    }
}

/* ===================================================================== *
 * FROZEN §7 SEAM CONSUMER (DS4_REAP_PIN_BY_MASS) -- livemask pin-by-mass.
 * Producer (ds4.c) publishes, once per decode token after the mask settles:
 *   g_reap_pin_mass[L][e] = lm_wshare[L][e] if e eligible (mask non-pruned)
 *                           else 0.0f ;   g_reap_pin_epoch++ on every change.
 * Both are C-linkage globals OWNED by ds4.c per §7.  ds4.c has NOT defined
 * them yet, so we provide WEAK zero definitions here: the consumer builds/
 * links/runs standalone (gate OFF => all-zero mass, no behaviour change) and
 * the producer's future STRONG ds4.c definition wins at link with no
 * duplicate-symbol collision.  Do NOT redefine the seam name/semantics/env.
 *
 * DIMS: ds4.c declares [DS4_MAX_LAYER][DS4_MAX_EXPERT] with DS4_MAX_LAYER=61
 * (ds4.c:111) / DS4_MAX_EXPERT=384 (ds4.c:122) living in an ANONYMOUS enum
 * inside ds4.c -- absent from this TU (not exported via ds4_gpu.h; grep of
 * this file for the token = 0 hits).  Cross-TU array externs match by NUMERIC
 * extent, not macro name, so we mirror the literals and static_assert their
 * product against this file's existing DS4_CUDA_STREAM_EXPERT_MAX (== 61u*384u,
 * line 40) so any future dim drift on either side is a COMPILE error, never a
 * silent mis-index. */
enum {
    DS4_REAP_PIN_MAX_LAYER  = 61,   /* == ds4.c DS4_MAX_LAYER  */
    DS4_REAP_PIN_MAX_EXPERT = 384   /* == ds4.c DS4_MAX_EXPERT */
};
static_assert((unsigned)DS4_REAP_PIN_MAX_LAYER * (unsigned)DS4_REAP_PIN_MAX_EXPERT
                  == (unsigned)DS4_CUDA_STREAM_EXPERT_MAX,
              "g_reap_pin_mass dims must equal ds4.c DS4_MAX_LAYER*DS4_MAX_EXPERT");

extern "C" {
/* Weak zero defs -- ds4.c's future strong definition overrides at link. */
__attribute__((weak)) float g_reap_pin_mass[DS4_REAP_PIN_MAX_LAYER]
                                           [DS4_REAP_PIN_MAX_EXPERT];
__attribute__((weak)) uint8_t g_reap_spex_pin[DS4_REAP_PIN_MAX_LAYER]
                                             [DS4_REAP_PIN_MAX_EXPERT];
__attribute__((weak)) volatile uint32_t g_reap_pin_epoch;
}

/* Gate: OFF (unset or "0") => every hook below is byte-for-byte 0040. */
static int cuda_reap_pin_by_mass_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = cuda_pin_env_flag("DS4_REAP_PIN_BY_MASS");
    return cached;
}

/* Bounds-safe read: slot/param layer,expert are model-routed ids already
 * < 61 / < n_total_expert(<=384), but clamp defensively so an out-of-range id
 * can never index past the seam arrays. */
static inline float cuda_reap_pin_mass_lookup(uint32_t layer, uint32_t expert) {
    if (layer  >= (uint32_t)DS4_REAP_PIN_MAX_LAYER ||
        expert >= (uint32_t)DS4_REAP_PIN_MAX_EXPERT) return 0.0f;
    return g_reap_pin_mass[layer][expert];
}

static inline int cuda_reap_spex_pin_lookup(uint32_t layer, uint32_t expert) {
    if (layer  >= (uint32_t)DS4_REAP_PIN_MAX_LAYER ||
        expert >= (uint32_t)DS4_REAP_PIN_MAX_EXPERT) return 0;
    return g_reap_spex_pin[layer][expert] != 0;
}

static int cuda_reap_spex_pin_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_PACE_LIVEMASK_SPEX_ADD");
        cached = v && v[0] && strcmp(v, "0") != 0;
    }
    return cached;
}

static FILE *cuda_reap_spex_pin_log(void) {
    static FILE *fh;
    static int checked;
    if (!checked) {
        checked = 1;
        const char *path = getenv("DS4_PACE_LIVEMASK_SPEX_PIN_LOG");
        if (path && path[0]) fh = fopen(path, "w");
    }
    return fh;
}

static void cuda_reap_spex_log_event(const char *ev,
                                     uint32_t layer,
                                     uint32_t expert,
                                     float mass) {
    FILE *fh = cuda_reap_spex_pin_log();
    if (!fh) return;
    fprintf(fh,
            "{\"ev\":\"%s\",\"epoch\":%u,\"layer\":%u,"
            "\"expert\":%u,\"mass\":%.9g}\n",
            ev, (uint32_t)g_reap_pin_epoch, layer, expert, mass);
    fflush(fh);
}

static uint32_t cuda_reap_sticky_budget(const cuda_stream_expert_cache *cache) {
    if (!cache || cache->capacity == 0) return 0;
    uint32_t reserve = cache->capacity / 8u + 1u;
    const char *rs = getenv("DS4_CUDA_INPLACE_RESERVE_SLOTS");
    if (rs && rs[0]) {
        char *end = NULL;
        const unsigned long v = strtoul(rs, &end, 10);
        if (end && *end == '\0' && v < cache->capacity) reserve = (uint32_t)v;
    }
    return cache->capacity > reserve ? cache->capacity - reserve : 0u;
}

/* Free one residency pin for a provisional SPEX expert. Prefer the lowest-mass
 * non-SPEX pin; selection membership is untouched. */
static int cuda_reap_spex_make_room(cuda_stream_expert_cache *cache,
                                    uint32_t budget) {
    if (cache->pinned_count < budget) return 1;
    int victim = -1;
    float victim_mass = 0.0f;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        if (!s.valid || !s.pinned || s.reap_spex_pinned) continue;
        const float mass = cuda_reap_pin_mass_lookup(s.layer, s.expert);
        if (victim < 0 || mass < victim_mass) {
            victim = (int)i;
            victim_mass = mass;
        }
    }
    if (victim < 0) return 0;
    cuda_reap_spex_log_event(
        "spex_pin_displace_core",
        cache->slots[(uint32_t)victim].layer,
        cache->slots[(uint32_t)victim].expert,
        victim_mass);
    cache->slots[(uint32_t)victim].pinned = 0;
    if (cache->pinned_count > 0) cache->pinned_count--;
    g_spex_pin_displacements++;
    return 1;
}

/* Reconcile only SPEX-owned pins when the producer overlay changes. Core pins
 * stay owned by pin-by-mass; an expired provisional pin graduates to core when
 * its observed mass is positive, otherwise it is released. */
static void cuda_reap_spex_refresh(cuda_stream_expert_cache *cache) {
    if (!cuda_reap_spex_pin_enabled() || !cache || !cache->valid) return;
    const uint32_t epoch = g_reap_pin_epoch;
    if (cache->reap_pin_epoch_seen == epoch) return;
    cache->reap_pin_epoch_seen = epoch;
    const uint32_t budget = cuda_reap_sticky_budget(cache);

    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        if (!s.valid || !s.reap_spex_pinned) continue;
        if (cuda_reap_spex_pin_lookup(s.layer, s.expert)) continue;
        s.reap_spex_pinned = 0;
        if (s.pinned && cuda_reap_pin_mass_lookup(s.layer, s.expert) <= 0.0f) {
            cuda_reap_spex_log_event("spex_pin_release", s.layer, s.expert, 0.0f);
            s.pinned = 0;
            if (cache->pinned_count > 0) cache->pinned_count--;
            g_spex_pin_releases++;
        }
    }

    for (uint32_t i = 0; i < cache->capacity; i++) {
        cuda_stream_expert_cache_slot &s = cache->slots[i];
        if (!s.valid || !cuda_reap_spex_pin_lookup(s.layer, s.expert)) continue;
        if (!s.pinned) {
            if (!cuda_reap_spex_make_room(cache, budget)) continue;
            s.pinned = 1;
            cache->pinned_count++;
            g_spex_pin_promotions++;
            cuda_reap_spex_log_event(
                "spex_pin_promote_resident", s.layer, s.expert,
                cuda_reap_pin_mass_lookup(s.layer, s.expert));
        }
        s.reap_spex_pinned = 1;
    }
}

static uint32_t cuda_stream_expert_cache_lru_slot(
        cuda_stream_expert_cache *cache) {
    cuda_reap_spex_refresh(cache);
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (!cache->slots[i].valid) return i; /* empty slots are never pinned */
    }
    /* 0031: skip pinned (eviction-immune) slots when choosing an LRU victim.
     * 0033: among the non-pinned candidates, prefer the LOWER tier first
     * (SSD-blocked tier 0 before RAM tier 1), then the lower age.  With tiering
     * off every slot has tier 0, so this reduces to the pure min-age selection
     * of 0031 -> byte-identical when DS4_PACE_TIER is unset. */
    int have = 0;
    uint32_t slot = 0;
    int best_tier = 0;
    uint64_t best_age = 0;
    float best_mass = 0.0f;
    /* §7 mass-weighted eviction: with DS4_REAP_PIN_BY_MASS, the victim is the
     * non-pinned resident with the LOWEST producer pin-mass (ineligible / low
     * share evicted before the marked working set); the 0033 (tier,age) order
     * stays the tiebreak when masses are equal.  Gate OFF => by_mass==0, the
     * per-slot lookup is NOT called (m is the literal 0.0f), the test takes the
     * else-branch (t<best_tier || (t==best_tier && a<best_age)) which for
     * !have|other reproduces the original expression exactly and best_mass is
     * dead => byte-identical victim to 0040.  Reads cache->slots[i].layer/
     * .expert = LIVE resident identity (correct at this point). */
    const int by_mass = cuda_reap_pin_by_mass_enabled();
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (cache->slots[i].pinned) continue;
        const int t = cache->slots[i].tier;
        const uint64_t a = cache->slots[i].age;
        const float m = by_mass
            ? cuda_reap_pin_mass_lookup(cache->slots[i].layer,
                                        cache->slots[i].expert)
            : 0.0f;
        int better;
        if (!have) {
            better = 1;
        } else if (by_mass) {
            better = (m < best_mass) ||
                     (m == best_mass &&
                      (t < best_tier || (t == best_tier && a < best_age)));
        } else {
            better = (t < best_tier || (t == best_tier && a < best_age));
        }
        if (better) {
            best_tier = t;
            best_age  = a;
            best_mass = m;
            slot = i;
            have = 1;
        }
    }
    if (have) return slot;
    /* Safety net: if every valid slot is pinned (budget is capped below
     * capacity so this should not happen), fall back to global min-age so a
     * load never deadlocks. */
    slot = 0;
    best_age = cache->slots[0].age;
    for (uint32_t i = 1; i < cache->capacity; i++) {
        if (cache->slots[i].age < best_age) {
            best_age = cache->slots[i].age;
            slot = i;
        }
    }
    return slot;
}

static int cuda_stream_expert_cache_copy_to_compact(
        cuda_stream_expert_cache *cache,
        uint32_t cache_slot,
        uint32_t compact_slot,
        char *compact_gate,
        char *compact_up,
        char *compact_down) {
    const uint64_t gate_src = (uint64_t)cache_slot * cache->gate_expert_bytes;
    const uint64_t down_src = (uint64_t)cache_slot * cache->down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)compact_slot * cache->gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)compact_slot * cache->down_expert_bytes;
    return cuda_ok(cudaMemcpy(compact_gate + gate_dst,
                              cache->gate_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected gate cache copy") &&
           cuda_ok(cudaMemcpy(compact_up + gate_dst,
                              cache->up_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected up cache copy") &&
           cuda_ok(cudaMemcpy(compact_down + down_dst,
                              cache->down_ptr + down_src,
                              (size_t)cache->down_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected down cache copy");
}

static int cuda_stream_expert_cache_load_slot(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t slot,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        int defer_upload_sync,
        int prompt_like_batch) {
    /* 0036a epoch-fix: capture this slot's PRE-repurpose (valid,layer)
     * identity BEFORE 0034's `cache->slots[slot].valid = 0` stamp (just
     * below) zeroes it.  0036's cross-layer epoch bump further down must
     * test the identity the slot had on entry: reading entry.valid AFTER 0034
     * always sees 0, so the bump for a genuine cross-layer eviction would never
     * fire and that other layer's stale in-place slot map would keep pointing
     * at this repurposed slot -- its decode GEMM would then read the WRONG
     * expert's weights (silent, non-bit-exact, counted as 0 misses). */
    const int      inplace_prev_valid = cache->slots[slot].valid;
    const uint32_t inplace_prev_layer = cache->slots[slot].layer;
    /* 0034 stale-slot fix: stamp this slot invalid before the first buffer
     * write below.  If any of the copies that follow fail partway through,
     * the slot is left holding a torn buffer; clearing valid here (not just
     * in the two call-site failure branches downstream) means a partial
     * failure can never be observed as a hit by find(), even via a future
     * call path that doesn't go through those two guarded sites.  On
     * success this function re-sets valid=1 with the new expert's identity
     * below. */
    cache->slots[slot].valid = 0;
    cache->slots[slot].reap_spex_pinned = 0;
    const uint64_t gate_src =
        gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src =
        up_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src =
        down_offset + (uint64_t)expert * down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)slot * gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)slot * down_expert_bytes;
    const int copied_from_cold =
        cuda_expert_cold_lossless_copy_to_device(cache->gate_ptr + gate_dst,
                                                 cache->up_ptr + gate_dst,
                                                 cache->down_ptr + down_dst,
                                                 model_map,
                                                 model_size,
                                                 layer,
                                                 n_total_expert,
                                                 expert,
                                                 gate_offset,
                                                 up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes,
                                                 prompt_like_batch);
    if (!copied_from_cold &&
        (!cuda_model_copy_to_device_streamed(cache->gate_ptr + gate_dst,
                                             model_map,
                                             model_size,
                                             gate_src,
                                             gate_expert_bytes,
                                             "cached moe_gate",
                                             defer_upload_sync) ||
         !cuda_model_copy_to_device_streamed(cache->up_ptr + gate_dst,
                                             model_map,
                                             model_size,
                                             up_src,
                                             gate_expert_bytes,
                                             "cached moe_up",
                                             defer_upload_sync) ||
         !cuda_model_copy_to_device_streamed(cache->down_ptr + down_dst,
                                             model_map,
                                             model_size,
                                             down_src,
                                             down_expert_bytes,
                                             "cached moe_down",
                                             defer_upload_sync))) {
        return 0;
    }
    cuda_stream_expert_cache_slot &entry = cache->slots[slot];
    const int spex_wanted = cuda_reap_spex_pin_enabled() &&
        cuda_reap_spex_pin_lookup(layer, expert);
    if (cuda_inplace_resident_enabled()) {
        /* 0036: a (layer,expert)->slot rebinding invalidates the in-place fast
         * path for BOTH the evicted victim's layer and the admitted expert's
         * layer.  Bump both epochs so a stale fast-path reuse can never read a
         * repurposed slot. */
        if (inplace_prev_valid && inplace_prev_layer != layer) {
            /* 0036a: test the PRE-0034 identity captured at function entry, not
             * entry.valid -- 0034 zeroed it at the top, so the original test
             * never fired for a real cross-layer eviction and left that other
             * layer's stale in-place binding live over a now-repurposed slot. */
            cuda_inplace_epoch_bump(inplace_prev_layer);
        }
        cuda_inplace_epoch_bump(layer);
        /* 0036 cache-first: pin the admitted expert on first touch (sticky) so
         * a resident routed expert is never evicted and always HITs on later
         * tokens.  Bounded by capacity - reserve so the LRU keeps a victim;
         * beyond the budget new admissions stay LRU-evictable (honest degrade
         * to the per-token-copy behaviour, warned once). */
        if (!entry.pinned) {
            /* Strategy 2 (pin-after-grow): only pin once the cache has fully
             * grown to the configured budget (decode-stable / post-prefill),
             * mirroring 0031's freeze-after-warmup.  During the prefill growth
             * climb (capacity 10 -> ... -> budget) the sticky budget derived
             * from the TINY live capacity caps pins at ~8, and every subsequent
             * release_all()+realloc on growth wipes them -> the working set is
             * re-admitted (re-copied H2D) every token.  Pinning only at full
             * capacity makes the pinned set the decode-routed working set, and
             * sticky resolves to capacity-(capacity/8+1) against the true
             * ceiling (400 -> 349).  This changes only WHEN a slot becomes
             * eviction-immune, never which weights the GEMM reads -> bit-exact. */
            const uint32_t full_cap =
                cuda_stream_expert_cache_configured_budget();
            if (full_cap != 0 && cache->capacity >= full_cap) {
                /* Reserve slots kept LRU-evictable so the working set that
                 * exceeds the pin budget still has victims.  Default cap/8+1
                 * (12.5%); tune down via DS4_CUDA_INPLACE_RESERVE_SLOTS when the
                 * routed working set is close to capacity (e.g. K8 ~= 350 vs
                 * cap 400) so the whole set pins and admissions plateau. */
                uint32_t reserve = cache->capacity / 8u + 1u;
                { const char *rs = getenv("DS4_CUDA_INPLACE_RESERVE_SLOTS");
                  if (rs && rs[0]) {
                      char *e = NULL; unsigned long v = strtoul(rs, &e, 10);
                      if (e && *e == '\0' && v < cache->capacity) reserve = (uint32_t)v;
                  } }
                const uint32_t sticky_budget =
                    cache->capacity > reserve ? cache->capacity - reserve : 0u;
                if (spex_wanted && cache->pinned_count >= sticky_budget)
                    (void)cuda_reap_spex_make_room(cache, sticky_budget);
                if (cache->pinned_count < sticky_budget) {
                    /* §7 pin-by-mass: with DS4_REAP_PIN_BY_MASS, pin only the
                     * eligible set (producer mass>0).  Ineligible admits stay
                     * LRU-evictable and are NOT counted as a pin-budget wall
                     * (that reject path is a distinct, capacity-driven event).
                     * The 0040 full-cap+sticky gate still WINS as the outer
                     * necessary condition; by-mass only filters WHICH budgeted
                     * slots pin.  IDENTITY: key the lookup on the incoming
                     * function params (layer,expert) -- the expert being
                     * ADMITTED -- NOT entry.layer/entry.expert, which at this
                     * point still hold the evicted victim's STALE ids (entry is
                     * bound to cache->slots[slot] @4577; its identity is only
                     * assigned below at ~4647-4649).  Gate OFF => pin_ok==1 =>
                     * the three statements run exactly as in 0040 (byte-id). */
                    int pin_ok = 1;
                    if (cuda_reap_pin_by_mass_enabled()) {
                        pin_ok = cuda_reap_pin_mass_lookup(layer, expert) > 0.0f ||
                                 spex_wanted;
                    }
                    if (pin_ok) {
                        entry.pinned = 1;
                        entry.reap_spex_pinned = spex_wanted;
                        cache->pinned_count++;
                        g_inplace_pin_admits++;
                        if (spex_wanted) {
                            g_spex_pin_promotions++;
                            cuda_reap_spex_log_event(
                                "spex_pin_promote_admit", layer, expert,
                                cuda_reap_pin_mass_lookup(layer, expert));
                        }
                    }
                } else {
                    g_inplace_pin_rejects++;
                    static int warned_capacity_wall = 0;
                    if (!warned_capacity_wall) {
                        warned_capacity_wall = 1;
                        fprintf(stderr,
                                "ds4: DS4_CUDA_INPLACE_RESIDENT capacity wall: working "
                                "set exceeds pin budget (%u slots); new admissions stay "
                                "LRU-evictable and may re-copy per token. Raise "
                                "DS4_CUDA_STREAMING_EXPERT_CACHE_N.\n",
                                sticky_budget);
                    }
                }
            }
        }
    }
    /* SPEX mask residency is independent of the no-copy in-place path: even
     * when DS4_CUDA_INPLACE_RESIDENT=0, a provisional expert must survive until
     * the target router consumes it. */
    if (spex_wanted && !entry.pinned) {
        const uint32_t sticky_budget = cuda_reap_sticky_budget(cache);
        if (cuda_reap_spex_make_room(cache, sticky_budget) &&
            cache->pinned_count < sticky_budget) {
            entry.pinned = 1;
            entry.reap_spex_pinned = 1;
            cache->pinned_count++;
            g_spex_pin_promotions++;
            cuda_reap_spex_log_event(
                "spex_pin_promote_admit", layer, expert,
                cuda_reap_pin_mass_lookup(layer, expert));
        }
    }
    entry.valid = 1;
    entry.model_map = model_map;
    entry.model_size = model_size;
    entry.layer = layer;
    entry.n_total_expert = n_total_expert;
    entry.expert = expert;
    entry.gate_offset = gate_offset;
    entry.up_offset = up_offset;
    entry.down_offset = down_offset;
    entry.gate_expert_bytes = gate_expert_bytes;
    entry.down_expert_bytes = down_expert_bytes;
    entry.age = ++cache->tick;
    return 1;
}

static int cuda_stream_expert_cache_seed_one(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        int defer_upload_sync) {
    int cache_slot = cuda_stream_expert_cache_find(cache,
                                                   model_map,
                                                   model_size,
                                                   layer,
                                                   n_total_expert,
                                                   expert,
                                                   gate_offset,
                                                   up_offset,
                                                   down_offset,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    if (cache_slot >= 0) {
        cache->slots[(uint32_t)cache_slot].age = ++cache->tick;
        cuda_pin_note_resident(cache, (uint32_t)cache_slot); /* 0031: SPEX seed */
        cuda_tier_note_resident(cache, (uint32_t)cache_slot); /* 0033: SPEX seed */
        return 1;
    }

    const uint32_t load_slot = cuda_stream_expert_cache_lru_slot(cache);
    const int append = !cache->slots[load_slot].valid;
    if (!cuda_stream_expert_cache_load_slot(cache,
                                            model_map,
                                            model_size,
                                            load_slot,
                                            layer,
                                            n_total_expert,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes,
                                            defer_upload_sync,
                                            0)) {
        return 0;
    }
    if (append && cache->count < cache->capacity) cache->count++;
    cuda_pin_note_resident(cache, load_slot); /* 0031: SPEX seed -> pin candidate */
    cuda_tier_note_resident(cache, load_slot); /* 0033: SPEX seed -> tier candidate */
    return 1;
}

static int cuda_stream_layer_expert_ranges_valid(
        uint64_t model_size,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char *what) {
    if (n_total_expert == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert size overflow\n",
                what ? what : "selected");
        return 0;
    }
    const uint64_t full_gate_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size ||
        down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert range outside model map\n",
                what ? what : "selected");
        return 0;
    }
    return 1;
}

static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what,
        int defer_upload_sync) {
    if (!dst || !model_map || offset > model_size || bytes > model_size - offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    const double spex_t0 =
        cuda_spex_stats_is_enabled() ? cuda_spex_now_sec() : 0.0;

    /* patch 0050 fast path: this exact [offset,bytes) span is already pinned
     * host memory (DS4_CUDA_STREAM_FROM_RAM_MASKED registration) -- DMA
     * straight from the mmap into the VRAM destination `dst`, skipping
     * pread()+staging-buffer-copy entirely. This is the toll elimination the
     * patch is for; note it is a DIFFERENT consumer than cuda_model_range_ptr
     * (which hands a UVA pointer straight to compute kernels -- fine for
     * small non-expert tensors, a fine-grained-PCIe-access catastrophe for
     * MoE expert GEMM inputs). On any failure here we fall through to the
     * normal path below unconditionally, so correctness never depends on
     * this branch succeeding. */
    int masked_cover_reason = CUDA_MASKED_PIN_COVER_RANGE_MISS;
    uint64_t masked_range_start = 0;
    uint64_t masked_range_end = 0;
    const bool masked_covered = cuda_masked_pin_covers(
            model_map, offset, bytes, &masked_cover_reason,
            &masked_range_start, &masked_range_end);
    if (cuda_masked_pin_diag_enabled()) {
        const uint64_t q = __sync_add_and_fetch(&g_masked_pin_diag.queries, 1);
        (void)__sync_fetch_and_add(&g_masked_pin_diag.query_bytes, bytes);
        if (masked_covered) {
            (void)__sync_fetch_and_add(&g_masked_pin_diag.hits, 1);
            (void)__sync_fetch_and_add(&g_masked_pin_diag.hit_bytes, bytes);
        } else {
            uint64_t *miss_counter = &g_masked_pin_diag.miss_range;
            if (masked_cover_reason == CUDA_MASKED_PIN_COVER_EMPTY) {
                miss_counter = &g_masked_pin_diag.miss_empty;
            } else if (masked_cover_reason == CUDA_MASKED_PIN_COVER_BASE_MISMATCH) {
                miss_counter = &g_masked_pin_diag.miss_base;
            } else if (masked_cover_reason == CUDA_MASKED_PIN_COVER_BEFORE_FIRST) {
                miss_counter = &g_masked_pin_diag.miss_before_first;
            }
            (void)__sync_fetch_and_add(miss_counter, 1);
            if (masked_cover_reason >= CUDA_MASKED_PIN_COVER_EMPTY &&
                masked_cover_reason <= CUDA_MASKED_PIN_COVER_RANGE_MISS &&
                __sync_bool_compare_and_swap(
                    &g_masked_pin_diag.first_miss_logged[masked_cover_reason], 0, 1)) {
                fprintf(stderr,
                        "ds4: CUDA masked zero-copy first miss: reason=%d "
                        "registered_base=%p copy_base=%p query=[%llu,%llu) "
                        "nearest=[%llu,%llu)\n",
                        masked_cover_reason, g_masked_pin_host_base, model_map,
                        (unsigned long long)offset,
                        (unsigned long long)(offset + bytes),
                        (unsigned long long)masked_range_start,
                        (unsigned long long)masked_range_end);
            }
        }
        if ((q & 511u) == 0) cuda_masked_pin_diag_report("periodic");
    }
    if (masked_covered) {
        static int pin_path_logged;
        if (__sync_bool_compare_and_swap(&pin_path_logged, 0, 1)) {
            fprintf(stderr,
                    "ds4: CUDA masked zero-copy DMA path ACTIVE (first hit: "
                    "%s, %.2f MiB)\n",
                    what ? what : "expert", (double)bytes / 1048576.0);
        }
        cudaError_t pin_err = cudaMemcpyAsync(dst,
                                              (const char *)model_map + offset,
                                              (size_t)bytes,
                                              cudaMemcpyHostToDevice,
                                              g_stream_selected_upload_stream);
        const bool pin_copy_enqueued = pin_err == cudaSuccess;
        if (pin_copy_enqueued) {
            if (defer_upload_sync && cuda_stream_selected_upload_event_enabled()) {
                if (cuda_stream_selected_upload_done_event_ensure()) {
                    pin_err = cudaEventRecord(g_stream_selected_upload_done_event,
                                              g_stream_selected_upload_stream);
                    if (pin_err == cudaSuccess) {
                        g_stream_selected_upload_done_recorded = 1;
                        if (cuda_masked_pin_diag_enabled()) {
                            (void)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok, 1);
                            (void)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok_bytes, bytes);
                        }
                        return cuda_spex_stats_finish_copy(1, bytes, spex_t0);
                    }
                }
                /* The copy is already queued. If event bookkeeping fails,
                 * drain the stream and accept the completed copy; never start
                 * a competing pread write into the same destination slot. */
                pin_err = cudaStreamSynchronize(g_stream_selected_upload_stream);
            } else {
                pin_err = cudaStreamSynchronize(g_stream_selected_upload_stream);
            }
            if (pin_err == cudaSuccess) {
                if (cuda_masked_pin_diag_enabled()) {
                    (void)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok, 1);
                    (void)__sync_fetch_and_add(&g_masked_pin_diag.dma_ok_bytes, bytes);
                }
                return cuda_spex_stats_finish_copy(1, bytes, spex_t0);
            }
        }
        if (cuda_masked_pin_diag_enabled()) {
            (void)__sync_fetch_and_add(&g_masked_pin_diag.dma_failed, 1);
        }
        fprintf(stderr,
                "ds4: CUDA masked zero-copy DMA failed for %s%s: %s\n",
                what ? what : "expert",
                pin_copy_enqueued ? " after enqueue; refusing overlapping fallback"
                                  : "; falling back to pread",
                cudaGetErrorString(pin_err));
        (void)cudaGetLastError();
        if (pin_copy_enqueued) return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        /* fall through to the normal pread+staging path below */
    }

    if (g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return cuda_spex_stats_finish_copy(
                cuda_ok(cudaMemcpy(dst,
                                   (const char *)model_map + offset,
                                   (size_t)bytes,
                                   cudaMemcpyHostToDevice),
                        what ? what : "stream selected expert copy"),
                bytes,
                spex_t0);
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_stream_selected_stage_pool_alloc(stage_bytes)) {
        return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
    }

    cudaError_t err = cudaSuccess;
    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_stream_selected_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA streaming selected staging wait failed for %s: %s\n",
                        what ? what : "expert",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
            }
        }

        const char *payload = NULL;
        if (!cuda_model_stage_read(g_stream_selected_stage[bi],
                                   g_stream_selected_stage_bytes,
                                   offset + copied,
                                   n,
                                   &payload)) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected read failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
        err = cudaMemcpyAsync(dst + copied,
                              payload,
                              (size_t)n,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
        err = cudaEventRecord(g_stream_selected_stage_event[bi],
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging record failed for %s: %s\n",
                    what ? what : "expert",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size, offset + copied, n);
        copied += n;
        chunk_idx++;
    }

    if (defer_upload_sync && cuda_stream_selected_upload_event_enabled()) {
        if (!cuda_stream_selected_upload_done_event_ensure()) {
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
        err = cudaEventRecord(g_stream_selected_upload_done_event,
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected upload event record failed for %s: %s\n",
                    what ? what : "expert",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
        g_stream_selected_upload_done_recorded = 1;
    } else {
        const double spex_sync_t0 = spex_t0 > 0.0 ? cuda_spex_now_sec() : 0.0;
        err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (spex_sync_t0 > 0.0) {
            const double spex_sync_t1 = cuda_spex_now_sec();
            g_spex_stats.selected_sync_calls++;
            if (spex_sync_t1 >= spex_sync_t0) {
                g_spex_stats.selected_sync_seconds += spex_sync_t1 - spex_sync_t0;
            }
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected upload sync failed for %s: %s\n",
                    what ? what : "expert",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return cuda_spex_stats_finish_copy(0, bytes, spex_t0);
        }
    }
    return cuda_spex_stats_finish_copy(1, bytes, spex_t0);
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    if (cuda_spex_stats_is_enabled() && !g_spex_stats_atexit_registered) {
        atexit(cuda_spex_stats_print_atexit);
        g_spex_stats_atexit_registered = 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    cuda_spex_stats_print();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
    cuda_stream_selected_cache_release();
    cuda_stream_expert_cache_release_all();
    cuda_stream_selected_stage_release();
    cuda_model_range_release_all();
    /* patch 0050: unlike the ordinary tensor cache ranges, masked DMA pins
     * survive same-map cache rebuilds. Cleanup is a true map-lifetime end. */
    cuda_masked_pin_release_all();
    cuda_model_load_progress_reset();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    g_ssd_streaming_mode = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" ds4_gpu_async_read *ds4_gpu_async_read_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_async_read *r = (ds4_gpu_async_read *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->bytes = bytes;
    if (!cuda_ok(cudaHostAlloc(&r->host, (size_t)bytes, cudaHostAllocDefault),
                 "async read host alloc") ||
        !cuda_ok(cudaStreamCreateWithFlags(&r->stream, cudaStreamNonBlocking),
                 "async read stream create") ||
        !cuda_ok(cudaEventCreateWithFlags(&r->ready, cudaEventDisableTiming),
                 "async read ready event create") ||
        !cuda_ok(cudaEventCreateWithFlags(&r->done, cudaEventDisableTiming),
                 "async read done event create")) {
        ds4_gpu_async_read_free(r);
        return NULL;
    }
    return r;
}

extern "C" void ds4_gpu_async_read_free(ds4_gpu_async_read *readback) {
    if (!readback) return;
    if (readback->done) (void)cudaEventDestroy(readback->done);
    if (readback->ready) (void)cudaEventDestroy(readback->ready);
    if (readback->stream) (void)cudaStreamDestroy(readback->stream);
    if (readback->host) (void)cudaFreeHost(readback->host);
    free(readback);
}

extern "C" void *ds4_gpu_async_read_host(ds4_gpu_async_read *readback) {
    return readback ? readback->host : NULL;
}

extern "C" int ds4_gpu_async_read_ready(ds4_gpu_async_read *readback) {
    if (!readback || !readback->pending) return 0;
    cudaError_t err = cudaEventQuery(readback->done);
    if (err == cudaSuccess) {
        readback->pending = 0;
        return 1;
    }
    if (err == cudaErrorNotReady) return 0;
    readback->pending = 0;
    return cuda_ok(err, "async read query");
}

extern "C" int ds4_gpu_tensor_read_async(ds4_gpu_async_read *readback,
                                          const ds4_gpu_tensor *tensor,
                                          uint64_t offset,
                                          uint64_t bytes) {
    if (!readback || !tensor || offset > tensor->bytes ||
        bytes > tensor->bytes - offset || bytes > readback->bytes) {
        return 0;
    }
    if (readback->pending && !ds4_gpu_async_read_ready(readback)) return 0;
    readback->pending = 0;
    readback->pending_bytes = bytes;
    if (bytes == 0) return 1;
    if (!cuda_ok(cudaEventRecord(readback->ready, 0),
                 "async read dependency event record") ||
        !cuda_ok(cudaStreamWaitEvent(readback->stream, readback->ready, 0),
                 "async read stream wait") ||
        !cuda_ok(cudaMemcpyAsync(readback->host,
                                 (const char *)tensor->ptr + offset,
                                 (size_t)bytes,
                                 cudaMemcpyDeviceToHost,
                                 readback->stream),
                 "async tensor read copy") ||
        !cuda_ok(cudaEventRecord(readback->done, readback->stream),
                 "async read done event record")) {
        return 0;
    }
    readback->pending = 1;
    return 1;
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                                         uint64_t offset,
                                                         void *data,
                                                         uint64_t bytes,
                                                         uint64_t event_value,
                                                         const char *label) {
    (void)event_value;
    (void)label;
    return ds4_gpu_tensor_read(tensor, offset, data, bytes);
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }

extern "C" int ds4_gpu_wait_selected_upload(void) {
    return cuda_stream_selected_upload_wait_if_recorded("selected compact load");
}
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 1;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback signal");
}
extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(), label ? label : "selected readback wait");
}
extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    (void)event_value;
    (void)label;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback wait");
}
extern "C" int ds4_gpu_end_commands(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "synchronize");
}

static int cuda_model_set_host_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    const int same_backing_model =
        g_model_host_base == model_map &&
        g_model_registered_size == model_size;
    if (!same_backing_model && g_stream_selected_upload_stream) {
        /* A true model-map switch invalidates the source addresses owned by
         * patch 0050. Drain every queued H2D copy before unregistering those
         * pages or publishing a different host base. */
        cudaError_t drain_err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (drain_err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA cannot switch model map: masked upload drain "
                    "failed: %s\n",
                    cudaGetErrorString(drain_err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_upload_done_recorded = 0;
    }
    cuda_stream_selected_cache_invalidate();
    if (!same_backing_model) {
        cuda_stream_expert_cache_release_all();
        /* patch 0050: release private DMA pins only when their backing mmap
         * identity really changes. cuda_model_set_host_map is also called for
         * same-map tensor-cache rebuilds during session preparation; tying the
         * pins to cuda_model_range_release_all made every registered range
         * disappear before the first expert copy. */
        cuda_masked_pin_release_all();
    }
    cuda_model_range_release_all();
    if (!same_backing_model) {
        cuda_model_load_progress_reset();
    }
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (!same_backing_model) {
        if (g_model_device_owned && g_model_device_base) {
            (void)cudaFree((void *)g_model_device_base);
            g_model_device_owned = 0;
        }
        if (g_model_registered && g_model_host_base) {
            (void)cudaHostUnregister((void *)g_model_host_base);
            g_model_registered = 0;
        }
        g_model_host_base = model_map;
        g_model_device_base = (const char *)model_map;
        g_model_registered_size = model_size;
    } else if (!g_model_device_owned && !g_model_registered) {
        g_model_device_base = (const char *)model_map;
    }
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       flags);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        const uint64_t limit = cuda_model_local_model_limit_bytes();
        if (!cuda_model_cache_limit_explicit() && model_size > limit) {
            fprintf(stderr,
                    "ds4: CUDA model %.2f GiB exceeds the default single-GPU "
                    "startup cache budget %.2f GiB; use distributed layer "
                    "loading or set DS4_CUDA_WEIGHT_CACHE_LIMIT_GB explicitly\n",
                    (double)model_size / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                                uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                                uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                                uint32_t n_total_expert) {
    (void)model_map;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL) {
        for (uint32_t i = 0; i < count; i++) {
            (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
        }
    }
    return 1;
}

/* patch 0050: zero-copy masked-window streaming. See ds4_gpu.h. Each range is
 * registered independently via the existing bounded cuda_model_range_register_mapped
 * (same primitive the cold on-demand path in cuda_model_range_ptr falls back
 * to for a single tensor slice) -- this call just does it eagerly, up front,
 * for the whole REAP keep-window, so every subsequent expert fetch in that
 * window is a UVA-mapped host-RAM read from the GPU instead of a pread +
 * pinned-stage + cudaMemcpyAsync + sync round trip. Ranges are expected
 * disjoint and already coalesced by the caller (adjacent kept experts merged
 * into one range per gate/up/down tensor run); this function does not itself
 * merge or sort them. If the driver reports range mapping unsupported
 * (cudaErrorNotSupported/cudaErrorInvalidValue -- see
 * cuda_model_range_register_mapped, which latches g_model_range_mapping_supported
 * to 0 on that class of failure) further attempts in this call are skipped
 * since they would fail identically; ranges already registered before that
 * point remain registered and usable. A partial result (some ranges mapped,
 * some not) is intentional degraded-mode behavior, not an error: unmapped
 * bytes simply keep paying the normal pread toll. */
extern "C" int ds4_gpu_register_masked_ranges(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t *out_bytes) {
    if (out_bytes) *out_bytes = 0;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] == 0 || sizes[i] > model_size - offsets[i]) {
            fprintf(stderr,
                    "ds4: CUDA masked zero-copy register: range %u out of bounds "
                    "(offset=%llu bytes=%llu model_size=%llu), aborting\n",
                    i, (unsigned long long)offsets[i], (unsigned long long)sizes[i],
                    (unsigned long long)model_size);
            return 0;
        }
    }
    /* The ordinary tensor/Q8 caches were prepared immediately before this
     * call. Rebinding an already-current map through cuda_model_set_host_map
     * would release all of them and force a lazy cold rebuild in the first
     * request. Bind only when the backing map identity actually differs. */
    if ((g_model_host_base != model_map ||
         g_model_registered_size != model_size) &&
        !cuda_model_set_host_map(model_map, model_size)) {
        return 0;
    }

    const double t0 = clock() / (double)CLOCKS_PER_SEC;
    int registered = 0;
    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        /* PRIVATE pin (see cuda_masked_pin_register): pinned for DMA in
         * cuda_model_copy_to_device_streamed only, intentionally invisible to
         * cuda_model_range_ptr/g_model_ranges -- letting compute kernels see
         * a UVA host pointer for expert tensors turns MoE GEMM into
         * per-element PCIe reads (~0.10 t/s decode, measured). */
        int unsupported = 0;
        if (cuda_masked_pin_register(model_map, offsets[i], sizes[i], &unsupported)) {
            registered++;
            total_bytes += sizes[i];
        } else if (unsupported) {
            fprintf(stderr,
                    "ds4: CUDA masked zero-copy register: driver reports host "
                    "registration unsupported after %d/%u range%s (%.2f GiB); "
                    "stopping, remaining ranges keep the normal pread path\n",
                    registered, count, count == 1 ? "" : "s",
                    (double)total_bytes / 1073741824.0);
            break;
        }
    }
    cuda_masked_pin_finalize();
    const double t1 = clock() / (double)CLOCKS_PER_SEC;
    fprintf(stderr,
            "ds4: CUDA masked zero-copy register: %d/%u range%s mapped, "
            "%.2f GiB pinned in %.3fs\n",
            registered, count, count == 1 ? "" : "s",
            (double)total_bytes / 1073741824.0, t1 - t0);
    if (out_bytes) *out_bytes = total_bytes;
    return registered;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_model_fd = fd;
    g_model_fd_host_base = model_map;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    return ds4_gpu_set_model_fd_for_map(fd, g_model_host_base);
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (cuda_model_range_is_cached(model_map, offset, bytes)) return 1;

    const char *ptr = cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor");
    if (!ptr || !cuda_model_range_is_cached(model_map, offset, bytes)) {
        if (!g_model_mapping_failure_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA failed to prepare model tensor spans for device access\n");
            g_model_mapping_failure_notice_printed = 1;
        }
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    const char *cache_label = label ? label : "q8_0";
    if (getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
        return 1;
    }
    if (!cuda_q8_f16_cache_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    if (!g_ssd_streaming_mode) {
        cuda_stream_selected_cache_release();
        cuda_stream_expert_cache_release_all();
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_budget_override = experts;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    cuda_stream_selected_cache_invalidate();
    cuda_stream_expert_cache_release_all();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared()) return 0;
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache.count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_expert_cache_release_all();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared() ||
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes) == 0) {
        return 0;
    }
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    return cuda_stream_expert_cache_configured_budget();
}

static int cuda_stream_expert_cache_contains_table(
        const cuda_stream_expert_cache       *cache,
        const ds4_gpu_stream_expert_table    *table,
        uint32_t                              expert) {
    if (!cache || !table || !cache->valid || cache->capacity == 0) return 0;
    for (uint32_t slot_i = 0; slot_i < cache->capacity; slot_i++) {
        const cuda_stream_expert_cache_slot &slot = cache->slots[slot_i];
        if (slot.valid &&
            slot.model_map == table->model_map &&
            slot.model_size == table->model_size &&
            slot.layer == table->layer &&
            slot.n_total_expert == table->n_total_expert &&
            slot.expert == expert &&
            slot.gate_offset == table->gate_offset &&
            slot.up_offset == table->up_offset &&
            slot.down_offset == table->down_offset &&
            slot.gate_expert_bytes == table->gate_expert_bytes &&
            slot.down_expert_bytes == table->down_expert_bytes) {
            return 1;
        }
    }
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_count_resident(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        uint32_t                           n_experts) {
    if (!table || !expert_ids || n_experts == 0) return 0;
    const cuda_stream_expert_cache *cache = &g_stream_expert_cache;
    if (!cache->valid || cache->capacity == 0) return 0;
    uint32_t resident = 0;
    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert) continue;
        if (cuda_stream_expert_cache_contains_table(cache,
                                                    table,
                                                    (uint32_t)expert)) {
            resident++;
        }
    }
    return resident;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_filter_missing(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        uint32_t                           n_experts,
        int32_t                           *missing_ids,
        uint32_t                           missing_cap,
        uint32_t                          *resident_count) {
    if (resident_count) *resident_count = 0;
    if (!table || !expert_ids || !missing_ids || n_experts == 0 ||
        missing_cap == 0) {
        return 0;
    }
    cuda_reap_spex_refresh(&g_stream_expert_cache);
    const cuda_stream_expert_cache *cache = &g_stream_expert_cache;
    uint32_t missing = 0;
    uint32_t resident = 0;
    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert) continue;
        if (cuda_stream_expert_cache_contains_table(cache,
                                                    table,
                                                    (uint32_t)expert)) {
            resident++;
            continue;
        }
        if (missing < missing_cap) {
            missing_ids[missing++] = expert;
        }
    }
    if (resident_count) *resident_count = resident;
    return missing;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids || n_selected == 0 ||
        n_selected > n_total_expert ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed selected")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_selected);
    if (!cache) return 1;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming seed selected expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)selected_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               0)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    return 1;
}

/* 0036: build or reuse the per-layer in-place binding for the decode GEMM.
 * On success g_inplace_layer[layer] holds a resident-slot map the caller may
 * return on immediately (routed_moe_launch reads the resident slab + this map,
 * incurring ZERO weight copies).  Returns 1 when the binding is ready (fast
 * reuse OR incremental rebuild), -1 on a hard failure (caller falls back to the
 * legacy compact path so the token still produces correct output). */
static int cuda_inplace_begin_layer(
        cuda_stream_expert_cache *cache,
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            prompt_like_batch) {
    if (!cache || !cache->valid) return -1;
    if ((uint64_t)layer >= g_inplace_layer.size()) {
        g_inplace_layer.resize((size_t)layer + 1u);
    }
    cuda_inplace_layer_state &st = g_inplace_layer[layer];
    const cuda_inplace_config *cfg = cuda_inplace_get_config();
    const uint64_t epoch_now = cuda_inplace_epoch_get(layer);

    /* ---- FAST PATH: routed set unchanged and still resident -> reuse ---- */
    if (st.valid && st.slot_dev &&
        st.prev_count == compact_count &&
        st.slot_count == slot_count &&
        st.epoch_seen == epoch_now &&
        st.model_map == model_map &&
        st.n_total_expert == n_total_expert &&
        st.gate_offset == gate_offset &&
        st.up_offset == up_offset &&
        st.down_offset == down_offset &&
        st.gate_expert_bytes == gate_expert_bytes &&
        st.down_expert_bytes == down_expert_bytes &&
        st.prev_ids.size() == (size_t)compact_count &&
        st.prev_slot_ids.size() == (size_t)slot_count &&
        memcmp(st.prev_ids.data(), compact_ids,
               (size_t)compact_count * sizeof(int32_t)) == 0 &&
        memcmp(st.prev_slot_ids.data(), slot_ids,
               (size_t)slot_count * sizeof(int32_t)) == 0) {
        if (cfg->verify) {
            for (uint32_t c = 0; c < compact_count; c++) {
                const int vslot = cuda_stream_expert_cache_find(
                        cache, model_map, model_size, layer, n_total_expert,
                        (uint32_t)compact_ids[c], gate_offset, up_offset,
                        down_offset, gate_expert_bytes, down_expert_bytes);
                if (vslot < 0) {
                    fprintf(stderr,
                            "ds4: INPLACE_VERIFY layer=%u idx=%u expert=%d "
                            "fast-path reuse but expert is NOT resident\n",
                            layer, c, compact_ids[c]);
                }
            }
        }
        g_inplace_reuse_hits++;
        return 1;
    }

    /* ---- REBUILD (incremental): find residents, admit misses exactly once ---- */
    std::vector<int32_t> resident_slot(compact_count);
    for (uint32_t c = 0; c < compact_count; c++) {
        if (compact_ids[c] < 0 || (uint32_t)compact_ids[c] >= n_total_expert) {
            st.valid = 0;
            return -1;
        }
        const uint32_t expert = (uint32_t)compact_ids[c];
        int slot = cuda_stream_expert_cache_find(
                cache, model_map, model_size, layer, n_total_expert, expert,
                gate_offset, up_offset, down_offset,
                gate_expert_bytes, down_expert_bytes);
        if (slot < 0) {
            const uint32_t load_slot =
                cuda_stream_expert_cache_lru_slot(cache);
            const int append = !cache->slots[load_slot].valid;
            if (!cuda_stream_expert_cache_load_slot(
                        cache, model_map, model_size, load_slot, layer,
                        n_total_expert, expert, gate_offset, up_offset,
                        down_offset, gate_expert_bytes, down_expert_bytes,
                        0, prompt_like_batch)) {
                st.valid = 0;
                return -1;
            }
            if (append && cache->count < cache->capacity) cache->count++;
            slot = (int)load_slot;
            g_inplace_admissions++;
        }
        cache->slots[(uint32_t)slot].age = ++cache->tick;
        cuda_pin_note_resident(cache, (uint32_t)slot);
        cuda_tier_note_resident(cache, (uint32_t)slot);
        resident_slot[c] = slot;
    }

    /* Build the routed-position -> resident-slot map the kernel indexes. */
    std::vector<int32_t> slot_map(slot_count);
    for (uint32_t j = 0; j < slot_count; j++) {
        const int32_t ci = slot_ids[j];
        if (ci < 0 || (uint32_t)ci >= compact_count) {
            st.valid = 0;
            return -1;
        }
        slot_map[j] = resident_slot[(uint32_t)ci];
    }

    if (st.slot_dev_capacity < (uint64_t)slot_count) {
        if (st.slot_dev) { (void)cudaFree(st.slot_dev); st.slot_dev = NULL; }
        if (!cuda_ok(cudaMalloc((void **)&st.slot_dev,
                                (size_t)slot_count * sizeof(int32_t)),
                     "inplace slot map alloc")) {
            st.slot_dev_capacity = 0;
            st.valid = 0;
            return -1;
        }
        st.slot_dev_capacity = slot_count;
    }

    /* Upload the slot map only when it changed (a stable-order routed set that
     * happens to re-derive identical residency skips even this metadata H2D). */
    const int need_upload =
        st.prev_slot_map.size() != (size_t)slot_count ||
        memcmp(st.prev_slot_map.data(), slot_map.data(),
               (size_t)slot_count * sizeof(int32_t)) != 0;
    if (need_upload) {
        if (!cuda_ok(cudaMemcpy(st.slot_dev, slot_map.data(),
                                (size_t)slot_count * sizeof(int32_t),
                                cudaMemcpyHostToDevice),
                     "inplace slot map upload")) {
            st.valid = 0;
            return -1;
        }
        st.prev_slot_map.assign(slot_map.begin(), slot_map.end());
        g_inplace_slot_uploads++;
    }

    st.prev_ids.assign(compact_ids, compact_ids + compact_count);
    st.prev_slot_ids.assign(slot_ids, slot_ids + slot_count);
    st.prev_count = compact_count;
    st.slot_count = slot_count;
    st.model_map = model_map;
    st.n_total_expert = n_total_expert;
    st.gate_offset = gate_offset;
    st.up_offset = up_offset;
    st.down_offset = down_offset;
    st.gate_expert_bytes = gate_expert_bytes;
    st.down_expert_bytes = down_expert_bytes;
    /* Re-read AFTER admissions: this rebuild's own admits bumped epoch[layer];
     * capturing the post-admit value lets the next token's fast path match. */
    st.epoch_seen = cuda_inplace_epoch_get(layer);
    st.slot_dev_tensor.ptr = st.slot_dev;
    st.slot_dev_tensor.bytes = (uint64_t)slot_count * sizeof(int32_t);
    st.slot_dev_tensor.owner = 0;
    st.valid = 1;
    return 1;
}

static int cuda_stream_selected_cache_begin_compact_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            strict_failure,
        int            allow_global_cache) {
    cuda_stream_selected_cache_invalidate();
    cuda_model_load_progress_finish();

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !compact_ids || !slot_ids ||
        n_total_expert == 0 ||
        compact_count == 0 || compact_count > n_total_expert ||
        slot_count == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: CUDA streaming selected expert size overflow\n");
        return 0;
    }

    const uint64_t full_gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t compact_gate_bytes = (uint64_t)compact_count * gate_expert_bytes;
    const uint64_t compact_down_bytes = (uint64_t)compact_count * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: CUDA streaming selected expert range outside model map\n");
        return 0;
    }

    if (!allow_global_cache) {
        cuda_stream_expert_cache_release_all();
    }

    if (!cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.gate_ptr,
                                           &g_stream_selected_cache.gate_capacity,
                                           compact_gate_bytes,
                                           "selected gate experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.up_ptr,
                                           &g_stream_selected_cache.up_capacity,
                                           compact_gate_bytes,
                                           "selected up experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.down_ptr,
                                           &g_stream_selected_cache.down_capacity,
                                           compact_down_bytes,
                                           "selected down experts") ||
        !cuda_stream_selected_ensure_i32(&g_stream_selected_cache.slot_selected_ptr,
                                         &g_stream_selected_cache.slot_selected_capacity,
                                         slot_count,
                                         "selected expert slots")) {
        return strict_failure ? 0 : 1;
    }

    if (allow_global_cache) {
        cuda_stream_expert_cache_note_size(gate_expert_bytes,
                                           down_expert_bytes);
    }
    const uint32_t configured_cache_budget =
        cuda_stream_expert_cache_configured_budget();
    const int use_global_cache =
        allow_global_cache &&
        configured_cache_budget != 0;
    cuda_stream_expert_cache *expert_cache = use_global_cache ?
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         configured_cache_budget) :
        NULL;
    int expert_cache_disabled = expert_cache == NULL;
    const uint32_t cache_count_before =
        expert_cache && expert_cache->valid ? expert_cache->count : 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t direct_loads = 0;
    uint32_t evictions = 0;
    const int prompt_like_batch =
        slot_count > DS4_CUDA_ROUTED_EXPERTS_PER_TOKEN;
    if (cuda_spex_stats_is_enabled()) {
        g_spex_stats.selected_load_batches++;
        g_spex_stats.selected_expert_requests += compact_count;
    }
    if (!cuda_stream_selected_upload_wait_if_recorded("global expert cache")) {
        return strict_failure ? 0 : 1;
    }

    cuda_pin_begin_call(expert_cache, layer); /* 0031: demand clock + freeze */
    cuda_tier_begin_call(expert_cache, layer); /* 0033: knock clock + seed/demote */

    /* 0036: in-place resident decode path.  When enabled and the global expert
     * cache is live, stage INCREMENTALLY into the resident slab (find resident /
     * admit-once) and bind the GEMM directly to those slots -- no per-token
     * D2D re-gather into the compact buffer, no slot_ids upload on reuse.  On a
     * hard failure we fall through to the legacy compact path below. */
    const int inplace_enabled = cuda_inplace_resident_enabled();
    if (inplace_enabled && !expert_cache_disabled &&
        expert_cache && expert_cache->valid) {
        const int rc = cuda_inplace_begin_layer(
                expert_cache, model_map, model_size, layer,
                compact_ids, slot_ids, n_total_expert, compact_count,
                slot_count, gate_offset, up_offset, down_offset,
                gate_expert_bytes, down_expert_bytes, prompt_like_batch);
        if (rc >= 0) {
            return 1;
        }
        if ((uint64_t)layer < g_inplace_layer.size()) {
            g_inplace_layer[layer].valid = 0;
        }
    }

    for (uint32_t i = 0; i < compact_count; i++) {
        if (compact_ids[i] < 0 || (uint32_t)compact_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    compact_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }

        const uint64_t expert = (uint64_t)(uint32_t)compact_ids[i];
        const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
        int copied_from_global_cache = 0;

        if (!expert_cache_disabled) {
            int cache_slot =
                cuda_stream_expert_cache_find(expert_cache,
                                              model_map,
                                              model_size,
                                              layer,
                                              n_total_expert,
                                              (uint32_t)expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes);
            if (cache_slot >= 0) {
                cache_hits++;
                expert_cache->slots[(uint32_t)cache_slot].age =
                    ++expert_cache->tick;
            } else {
                cache_misses++;
                const uint32_t load_slot =
                    cuda_stream_expert_cache_lru_slot(expert_cache);
                const int evicting = expert_cache->slots[load_slot].valid;
                const int append = !expert_cache->slots[load_slot].valid;
                if (cuda_stream_expert_cache_load_slot(expert_cache,
                                                       model_map,
                                                       model_size,
                                                       load_slot,
                                                       layer,
                                                       n_total_expert,
                                                       (uint32_t)expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       0,
                                                       prompt_like_batch)) {
                    if (append && expert_cache->count < expert_cache->capacity) {
                        expert_cache->count++;
                    }
                    if (evicting) evictions++;
                    cache_slot = (int)load_slot;
                } else {
                    /* 0024 resident-hit fix: a single failed slot load must not
                     * invalidate the whole cache and disable it for the rest of
                     * the session (the old behaviour collapsed all 258
                     * experts/token to direct loads after one transient copy
                     * failure, and cuda_stream_expert_cache_invalidate() left
                     * the ~GiB slot pool orphaned so the cache could never
                     * re-allocate).  Degrade only THIS expert to a direct load
                     * and keep the resident cache alive. */
                    /* 0034 stale-slot fix: 0024 stopped calling invalidate()
                     * here but left this slot's OLD identity (expert X,
                     * valid=1) mapped onto a buffer that load_slot may have
                     * partially overwritten with the NEW expert's weights.  A
                     * later find() for expert X would match this slot and
                     * serve corrupted, non-bit-exact weights for the rest of
                     * the session.  Mark it invalid so find() can't match it;
                     * the slot remains eviction-eligible (age untouched) and
                     * gets a clean load_slot() on its next use. */
                    expert_cache->slots[load_slot].valid = 0;
                    cache_slot = -1;
                }
            }

            if (cache_slot >= 0) {
                cuda_pin_note_resident(expert_cache, (uint32_t)cache_slot); /* 0031 */
                cuda_tier_note_resident(expert_cache, (uint32_t)cache_slot); /* 0033 */
                copied_from_global_cache =
                    cuda_stream_expert_cache_copy_to_compact(
                            expert_cache,
                            (uint32_t)cache_slot,
                            i,
                            g_stream_selected_cache.gate_ptr,
                            g_stream_selected_cache.up_ptr,
                            g_stream_selected_cache.down_ptr);
                if (!copied_from_global_cache) {
                    /* 0024 resident-hit fix: keep the cache resident on a
                     * transient device-to-device copy failure; this expert
                     * falls through to a direct load below. */
                    /* 0034 stale-slot fix: the D2D readout out of this slot
                     * just failed, so its buffer contents can no longer be
                     * trusted.  Invalidate it before falling through to the
                     * direct load so a later find() can't match a slot whose
                     * copy-out we just proved is bad. */
                    expert_cache->slots[(uint32_t)cache_slot].valid = 0;
                }
            }
        }

        if (!copied_from_global_cache) {
            const uint64_t gate_src = gate_offset + expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + expert * down_expert_bytes;
            direct_loads++;
            if (cuda_spex_stats_is_enabled()) g_spex_stats.selected_direct_loads++;
            const int copied_from_cold_lossless =
                cuda_expert_cold_lossless_copy_to_device(
                        g_stream_selected_cache.gate_ptr + gate_dst,
                        g_stream_selected_cache.up_ptr + gate_dst,
                        g_stream_selected_cache.down_ptr + down_dst,
                        model_map,
                        model_size,
                        layer,
                        n_total_expert,
                        (uint32_t)expert,
                        gate_offset,
                        up_offset,
                        down_offset,
                        gate_expert_bytes,
                        down_expert_bytes,
                        prompt_like_batch);
            if (!copied_from_cold_lossless &&
                (!cuda_model_copy_to_device_streamed(g_stream_selected_cache.gate_ptr + gate_dst,
                                                     model_map,
                                                     model_size,
                                                     gate_src,
                                                     gate_expert_bytes,
                                                     "selected moe_gate",
                                                     1) ||
                 !cuda_model_copy_to_device_streamed(g_stream_selected_cache.up_ptr + gate_dst,
                                                     model_map,
                                                     model_size,
                                                     up_src,
                                                     gate_expert_bytes,
                                                     "selected moe_up",
                                                     1) ||
                 !cuda_model_copy_to_device_streamed(g_stream_selected_cache.down_ptr + down_dst,
                                                     model_map,
                                                     model_size,
                                                     down_src,
                                                     down_expert_bytes,
                                                     "selected moe_down",
                                                     1))) {
                cuda_stream_selected_cache_invalidate();
                return strict_failure ? 0 : 1;
            }
        }
    }

    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache.slot_selected_ptr,
                            slot_ids,
                            (size_t)slot_count * sizeof(slot_ids[0]),
                            cudaMemcpyHostToDevice),
                 "streaming selected slot upload")) {
        cuda_stream_selected_cache_invalidate();
        return strict_failure ? 0 : 1;
    }

    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = slot_count;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = compact_count;
    g_stream_selected_cache.gate_offset = gate_offset;
    g_stream_selected_cache.up_offset = up_offset;
    g_stream_selected_cache.down_offset = down_offset;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.valid = 1;

    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming selected layer=%u slots=%u compact=%u global_budget=%u before=%u after=%u hits=%u misses=%u direct=%u evictions=%u gate/up %.2f MiB down %.2f MiB\n",
                layer,
                slot_count,
                compact_count,
                expert_cache && expert_cache->valid ? expert_cache->capacity : 0,
                cache_count_before,
                expert_cache && expert_cache->valid ? expert_cache->count : 0,
                cache_hits,
                cache_misses,
                direct_loads,
                evictions,
                (double)compact_gate_bytes / 1048576.0,
                (double)compact_down_bytes / 1048576.0);
    }
    cuda_expert_tier_promote_batch promote_batch =
        cuda_expert_tier_promote_note_batch(layer,
                                            compact_ids,
                                            slot_ids,
                                            compact_count,
                                            slot_count);
    cuda_expert_tiering_observe_note_batch(
            layer,
            n_total_expert,
            compact_ids,
            slot_ids,
            compact_count,
            slot_count,
            expert_cache && expert_cache->valid ? expert_cache->capacity : 0,
            cache_count_before,
            expert_cache && expert_cache->valid ? expert_cache->count : 0,
            cache_hits,
            cache_misses,
            direct_loads,
            evictions,
            gate_expert_bytes,
            down_expert_bytes,
            &promote_batch);
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected_ids || n_selected == 0) return 0;
    cuda_reap_spex_refresh(&g_stream_expert_cache);
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids(n_selected);
    compact_ids.reserve(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            n_selected,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            0,
            1);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table ||
        !selected_ids ||
        table->n_total_expert == 0 ||
        n_selected == 0 ||
        n_tokens == 0 ||
        (uint64_t)n_tokens > UINT32_MAX / (uint64_t)n_selected) {
        return 0;
    }
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    const uint32_t slot_count = n_tokens * n_selected;
    std::vector<int32_t> slot_ids(slot_count);
    compact_ids.reserve(slot_count < n_total_expert ? slot_count : n_total_expert);

    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming batch selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }

    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            slot_count,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            1,
            0);
}

static int cuda_stream_expert_cache_seed_experts_impl(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts,
        int                                defer_upload_sync,
        const char                        *label) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !expert_ids || n_experts == 0 ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               label ? label : "seed hotlist")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_experts);
    if (!cache || cache->capacity == 0) return 1;
    cuda_reap_spex_refresh(cache);

    const uint32_t layer_seed_cap =
        n_experts < cache->capacity ? n_experts : cache->capacity;
    std::vector<uint32_t> chosen;
    try {
        chosen.reserve(layer_seed_cap);
    } catch (...) {
        return 1;
    }

    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming %s expert id %d is outside 0..%u at layer %u\n",
                    label ? label : "seed",
                    expert,
                    n_total_expert,
                    layer);
            return 0;
        }
        const uint32_t priority =
            expert_priorities ? expert_priorities[i] : (n_experts - i);
        uint32_t pos = 0;
        while (pos < chosen.size()) {
            const uint32_t other = chosen[pos];
            const uint32_t other_priority =
                expert_priorities ? expert_priorities[other] :
                                    (n_experts - other);
            if (priority > other_priority) break;
            pos++;
        }
        if (chosen.size() < layer_seed_cap) {
            chosen.insert(chosen.begin() + pos, i);
        } else if (pos < chosen.size()) {
            chosen.insert(chosen.begin() + pos, i);
            chosen.pop_back();
        }
    }

    const uint32_t n = (uint32_t)chosen.size();
    for (uint32_t ri = 0; ri < n; ri++) {
        const uint32_t i = chosen[n - 1u - ri];
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)expert_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               defer_upload_sync)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "ds4: CUDA streaming %s layer=%u requested=%u cached=%u cap=%u\n",
                label ? label : "seeded",
                layer,
                n_experts,
                n,
                cache->capacity);
    }
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    return cuda_stream_expert_cache_seed_experts_impl(table,
                                                      expert_ids,
                                                      expert_priorities,
                                                      n_experts,
                                                      0,
                                                      "hotlist seed");
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts_async(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    return cuda_stream_expert_cache_seed_experts_impl(table,
                                                      expert_ids,
                                                      expert_priorities,
                                                      n_experts,
                                                      1,
                                                      "async prefetch seed");
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void spex_hidden_score_kernel(
        float *__restrict__ scores,
        const __half *__restrict__ weights,
        const float *__restrict__ hidden,
        uint32_t n_embd,
        uint32_t n_expert) {
    uint32_t expert = (uint32_t)blockIdx.x;
    if (expert >= n_expert) return;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        sum += __half2float(weights[(uint64_t)i * n_expert + expert]) * hidden[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[expert] = partial[0];
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_cpu_equiv_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static DS4_CUDA_UNUSED void rope_tail_one_dev(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(0.0f, floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t i = 0; i < n_rot; i += 2) {
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float mix = rope_yarn_ramp_cpu_equiv_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = x[n_nope + i];
        float x1 = x[n_nope + i + 1];
        x[n_nope + i] = x0 * c - x1 * s;
        x[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible_comp) {
            uint32_t slot = atomicAdd(&comp_count, 1u);
            if (slot < 512u) comp_rows[slot] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (comp_count > 512u) comp_count = 512u;
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_idx * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}

__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ static void indexer_scores_wmma_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    if (tid >= 32u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
                const uint32_t r = i >> 4u;
                const uint32_t c = i & 15u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float acc_sh[16 * 16];

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
            const uint32_t r = i >> 4u;
            const uint32_t token = tile_t + r;
            if (token < n_tokens) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
        const uint32_t r = i >> 4u;
        const uint32_t c = i & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 64u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 32u; i += 64u) {
                const uint32_t r = i >> 5u;
                const uint32_t c = i & 31u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[32 * 128];
    __shared__ float c_sh[2 * 16 * 16];
    __shared__ float acc_sh[2 * 16 * 16];

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 32u * 128u; i += 64u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 64u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma64_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 64u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 64u; i += 128u) {
                const uint32_t r = i >> 6u;
                const uint32_t c = i & 63u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[64 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float acc_sh[4 * 16 * 16];

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 64u * 128u; i += 128u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 128u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

/* Single-block argmax over n_vocab F32 logits. One block of 1024 threads
 * cooperatively scans the vocab, tracking a (best_v, best_idx) pair per
 * thread, then reduces in shared memory with value-keyed comparison.
 *
 * Tie-breaking: lower index wins, matching the host sample_argmax used by
 * the CPU reference path. Replaces the indexer-as-argmax workaround used
 * in the MTP top-id sites, which fell through to the legacy single-thread
 * indexer_topk_kernel at top_k=1, costing ~17.5 ms per call on n_vocab=129280. */
__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            /* Larger value wins; on exact ties prefer the lower index. */
            const bool take_right = (vr > vl) || (vr == vl && ir < il);
            if (take_right) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = sm_idx[0];
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE") == NULL) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_WMMA") == NULL) {
        if (getenv("DS4_CUDA_NO_INDEXER_WMMA128") == NULL) {
            dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, n_tokens, pos0, n_head,
                                                         head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA64") == NULL) {
            dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma64_kernel<<<grid, 128>>>((float *)scores->ptr,
                                                        (const float *)q->ptr,
                                                        (const float *)weights->ptr,
                                                        (const float *)index_comp->ptr,
                                                        n_comp, n_tokens, pos0, n_head,
                                                        head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma64 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA32") == NULL) {
            dim3 grid((n_comp + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma32_kernel<<<grid, 64>>>((float *)scores->ptr,
                                                       (const float *)q->ptr,
                                                       (const float *)weights->ptr,
                                                       (const float *)index_comp->ptr,
                                                       n_comp, n_tokens, pos0, n_head,
                                                       head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma32 launch");
        } else {
            dim3 grid((n_comp + 15u) / 16u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma_kernel<<<grid, 32>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma launch");
        }
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u &&
        getenv("DS4_CUDA_NO_TOPK1024") == NULL) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK8192") == NULL) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u && getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK_CHUNKED") == NULL) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}
static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL && blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    if (!serial_f16 && g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *weights, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !weights || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(uint16_t)) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (weights->bytes < weight_bytes ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const __half *w = (const __half *)weights->ptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    if (!serial_f16 && g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 tensor-weight gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 tensor-weight activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 tensor-weight matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_tensor_weight_router_serial launch" : "matmul_f16_tensor_weight_serial launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_tensor_weight_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16_tensor_weight launch");
}

extern "C" int ds4_gpu_spex_hidden_score_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_expert,
        const ds4_gpu_tensor *hidden) {
    if (!scores || !weights || !hidden || n_embd == 0 || n_expert == 0) return 0;
    const uint64_t weight_elems = (uint64_t)n_embd * n_expert;
    if (weight_elems > UINT64_MAX / sizeof(uint16_t)) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (weights->bytes < weight_bytes ||
        hidden->bytes < (uint64_t)n_embd * sizeof(float) ||
        scores->bytes < (uint64_t)n_expert * sizeof(float)) {
        return 0;
    }
    spex_hidden_score_kernel<<<n_expert, 256>>>((float *)scores->ptr,
                                                (const __half *)weights->ptr,
                                                (const float *)hidden->ptr,
                                                n_embd,
                                                n_expert);
    return cuda_ok(cudaGetLastError(), "spex hidden score launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1 ||
        getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL ||
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != NULL) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") == NULL) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *q_half,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        bool inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (n_tokens > 1u && top_k == 512u &&
        getenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") == NULL) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        getenv("DS4_CUDA_NO_INDEXED_HEADS8") == NULL) {
        if (getenv("DS4_CUDA_INDEXED_TWOPASS") == NULL) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A") == NULL) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    (void)out_b;
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h,
        ds4_gpu_tensor *low,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                            xscale,
                                            (const float *)heads->ptr,
                                            group_dim,
                                            blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q8 prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, 1, 1);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      1,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(), "attention_output_low_q8 launch");
}
extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") == NULL) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}
extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
            getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            router_select_parallel_kernel<<<1, 256>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
        getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8(
        const uint8_t *qs,
        const cuda_block_q8_K *const ys[8],
        uint32_t n,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) sums[p] = __dp4a(v, *(const int32_t *)(ys[p]->qs + y_off + i), sums[p]);
        }
    }
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8_full(
        const uint8_t *qs,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sums[0] = __dp4a(v, *(const int32_t *)(y0->qs + y_off + i), sums[0]);
        sums[1] = __dp4a(v, *(const int32_t *)(y1->qs + y_off + i), sums[1]);
        sums[2] = __dp4a(v, *(const int32_t *)(y2->qs + y_off + i), sums[2]);
        sums[3] = __dp4a(v, *(const int32_t *)(y3->qs + y_off + i), sums[3]);
        sums[4] = __dp4a(v, *(const int32_t *)(y4->qs + y_off + i), sums[4]);
        sums[5] = __dp4a(v, *(const int32_t *)(y5->qs + y_off + i), sums[5]);
        sums[6] = __dp4a(v, *(const int32_t *)(y6->qs + y_off + i), sums[6]);
        sums[7] = __dp4a(v, *(const int32_t *)(y7->qs + y_off + i), sums[7]);
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8(x->qs + byte_off, ys, n, y_off, shift, dots);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
                isum[p] += (int)sc * dots[p];
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
        }
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8_full(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8_full(
            x->qs + byte_off,
            y0, y1, y2, y3, y4, y5, y6, y7,
            y_off,
            shift,
            dots);
        const int ms0 = (int)m * (int)(y0->bsums[2u * j] + y0->bsums[2u * j + 1u]);
        const int ms1 = (int)m * (int)(y1->bsums[2u * j] + y1->bsums[2u * j + 1u]);
        const int ms2 = (int)m * (int)(y2->bsums[2u * j] + y2->bsums[2u * j + 1u]);
        const int ms3 = (int)m * (int)(y3->bsums[2u * j] + y3->bsums[2u * j + 1u]);
        const int ms4 = (int)m * (int)(y4->bsums[2u * j] + y4->bsums[2u * j + 1u]);
        const int ms5 = (int)m * (int)(y5->bsums[2u * j] + y5->bsums[2u * j + 1u]);
        const int ms6 = (int)m * (int)(y6->bsums[2u * j] + y6->bsums[2u * j + 1u]);
        const int ms7 = (int)m * (int)(y7->bsums[2u * j] + y7->bsums[2u * j + 1u]);
        summs[0] += ms0;
        summs[1] += ms1;
        summs[2] += ms2;
        summs[3] += ms3;
        summs[4] += ms4;
        summs[5] += ms5;
        summs[6] += ms6;
        summs[7] += ms7;
        isum[0] += (int)sc * dots[0];
        isum[1] += (int)sc * dots[1];
        isum[2] += (int)sc * dots[2];
        isum[3] += (int)sc * dots[3];
        isum[4] += (int)sc * dots[4];
        isum[5] += (int)sc * dots[5];
        isum[6] += (int)sc * dots[6];
        isum[7] += (int)sc * dots[7];
    }
    acc[0] += y0->d * xd * (float)isum[0] - y0->d * xmin * (float)summs[0];
    acc[1] += y1->d * xd * (float)isum[1] - y1->d * xmin * (float)summs[1];
    acc[2] += y2->d * xd * (float)isum[2] - y2->d * xmin * (float)summs[2];
    acc[3] += y3->d * xd * (float)isum[3] - y3->d * xmin * (float)summs[3];
    acc[4] += y4->d * xd * (float)isum[4] - y4->d * xmin * (float)summs[4];
    acc[5] += y5->d * xd * (float)isum[5] - y5->d * xmin * (float)summs[5];
    acc[6] += y6->d * xd * (float)isum[6] - y6->d * xmin * (float)summs[6];
    acc[7] += y7->d * xd * (float)isum[7] - y7->d * xmin * (float)summs[7];
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[expert_count] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[expert_count] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= expert_count) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(gr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, gate);
                dev_dot_q4_K_q8_K_block8_full(ur + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, up);
            } else {
                dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
                dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np >= 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
            if (np > 8u) {
                if (np == 16u) {
                    dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[8] + b, xqb[9] + b,
                                                  xqb[10] + b, xqb[11] + b,
                                                  xqb[12] + b, xqb[13] + b,
                                                  xqb[14] + b, xqb[15] + b, acc + 8);
                } else {
                    dev_dot_q4_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                             xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                             xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                             xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
                }
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const uint64_t required_slot_count = (uint64_t)n_tokens * n_expert;
    /* 0036: in-place resident decode binding.  When enabled and this layer has a
     * live in-place slot map (built by cuda_inplace_begin_layer), the GEMM reads
     * each routed expert straight from its resident slab slot -- selected[] holds
     * resident cache-slot indices and the slab stride equals gate/down_expert_bytes
     * (already the kernel's stride arg), so no kernel change is needed.  Decode
     * only (n_tokens==1); the sorted prefill path is left on the compact path. */
    const cuda_inplace_layer_state *inplace_state = NULL;
    if (cuda_inplace_resident_enabled() && n_tokens == 1u &&
        g_ssd_streaming_mode &&
        g_stream_expert_cache.valid &&
        g_stream_expert_cache.gate_ptr &&
        g_stream_expert_cache.up_ptr &&
        g_stream_expert_cache.down_ptr &&
        g_stream_expert_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_expert_cache.down_expert_bytes == down_expert_bytes &&
        (uint64_t)layer_index < g_inplace_layer.size()) {
        const cuda_inplace_layer_state &s = g_inplace_layer[layer_index];
        if (s.valid && s.slot_dev && s.slot_dev_tensor.ptr &&
            s.model_map == model_map &&
            s.n_total_expert == n_total_expert &&
            (uint64_t)s.slot_count >= required_slot_count &&
            s.gate_offset == gate_offset &&
            s.up_offset == up_offset &&
            s.down_offset == down_offset &&
            s.gate_expert_bytes == gate_expert_bytes &&
            s.down_expert_bytes == down_expert_bytes &&
            s.slot_dev_tensor.bytes >= required_slot_count * sizeof(int32_t)) {
            inplace_state = &s;
        }
    }
    const int use_inplace = (inplace_state != NULL);
    const int use_stream_selected_cache =
        !use_inplace &&
        g_ssd_streaming_mode &&
        g_stream_selected_cache.valid &&
        g_stream_selected_cache.model_map == model_map &&
        g_stream_selected_cache.layer == layer_index &&
        g_stream_selected_cache.n_total_expert == n_total_expert &&
        g_stream_selected_cache.slot_count >= required_slot_count &&
        g_stream_selected_cache.gate_offset == gate_offset &&
        g_stream_selected_cache.up_offset == up_offset &&
        g_stream_selected_cache.down_offset == down_offset &&
        g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
        g_stream_selected_cache.gate_ptr &&
        g_stream_selected_cache.up_ptr &&
        g_stream_selected_cache.down_ptr &&
        g_stream_selected_cache.slot_selected_tensor.ptr &&
        g_stream_selected_cache.slot_selected_tensor.bytes >=
            required_slot_count * sizeof(int32_t);
    const ds4_gpu_tensor *selected_tensor =
        use_inplace ? &inplace_state->slot_dev_tensor :
        (use_stream_selected_cache ? &g_stream_selected_cache.slot_selected_tensor : selected);
    const int32_t *selected_ptr = (const int32_t *)selected_tensor->ptr;
    const char *gate_w = use_inplace
        ? g_stream_expert_cache.gate_ptr
        : (use_stream_selected_cache
        ? g_stream_selected_cache.gate_ptr
        : cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate"));
    const char *up_w = use_inplace
        ? g_stream_expert_cache.up_ptr
        : (use_stream_selected_cache
        ? g_stream_selected_cache.up_ptr
        : cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up"));
    const char *down_w = use_inplace
        ? g_stream_expert_cache.down_ptr
        : (use_stream_selected_cache
        ? g_stream_selected_cache.down_ptr
        : cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down"));
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_q4_expert_tiles =
            q4k_path && getenv("DS4_CUDA_MOE_NO_Q4_EXPERT_TILES") == NULL;
        const uint32_t use_sorted_pairs = n_tokens > 1u && (!q4k_path || use_q4_expert_tiles);
        const uint32_t use_expert_tiles = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL;
        const uint32_t expert_tile_m = (!q4k_path && getenv("DS4_CUDA_MOE_TILE4")) ? 4u : 8u;
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted = use_sorted_pairs && !q4k_path && getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = use_expert_tiles &&
            getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (!q4k_path && n_tokens >= 128u));
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_down_tile16 = expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL &&
            (use_atomic_down || q4k_path);
        const uint32_t use_decode_lut_gate =
            !q4k_path && n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u : 1024u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint32_t sort_expert_count =
                use_stream_selected_cache ? g_stream_selected_cache.compact_count :
                n_total_expert;
            if (sort_expert_count == 0) ok = 0;
            const uint64_t counts_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + sort_expert_count;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + sort_expert_count) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, sort_expert_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile_experts, tile_starts, tile_offsets, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    if (use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else {
                        dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                        moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && !q4k_path && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {
                    moe_gate_up_mid_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    if (use_down_row2048) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<2048><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        }
                    } else if (use_down_tile16) {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile16_rowspan_kernel<32><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (!q4k_path && sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            selected_ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            selected_ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 1;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens);
}
extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (getenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED") == NULL) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        if (n_rows == 1) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                    3ull * sizeof(float), "hc_scale");
            const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                    mix_bytes, "hc_base");
            const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
                    (float *)out->ptr,
                    (float *)norm_out->ptr,
                    (float *)split->ptr,
                    (const float *)mix->ptr,
                    (const float *)residual_hc->ptr,
                    scale,
                    base,
                    norm_w,
                    n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h; (void)residual_hc;
    (void)split; (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}


/* patch 0011 v2: REAP runtime mask - upsert di range device (vince sul fallback). */
extern "C" int ds4_gpu_model_range_update(const void *model_map, uint64_t offset, const void *data, uint64_t bytes) {
    int updated = 0;
    for (size_t i = 0; i < g_model_ranges.size(); i++) {
        cuda_model_range &r = g_model_ranges[i];
        if (r.host_base != model_map) continue;
        if (offset < r.offset || bytes > r.bytes || offset - r.offset > r.bytes - bytes) continue;
        if (r.device_ptr && !r.host_registered) {
            if (cudaMemcpy(r.device_ptr + (offset - r.offset), data, bytes, cudaMemcpyHostToDevice) == cudaSuccess) updated++;
            else (void)cudaGetLastError();
        }
    }
    if (updated) return updated;
    void *dev = NULL;
    if (cudaMalloc(&dev, (size_t)bytes) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaMemcpy(dev, data, (size_t)bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError(); (void)cudaFree(dev); return 0;
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    return 1;
}
