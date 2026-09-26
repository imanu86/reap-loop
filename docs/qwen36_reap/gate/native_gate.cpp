// Source-only numerical probe. Build proposal and exact API references: README.md.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;
static constexpr const char * MODEL_SHA = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7";
static void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}
static std::string read_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "cannot open input: " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    require(!f.bad(), "input read failed: " + path);
    return s;
}
// No exists-then-open race; never truncate an existing artifact, including empty ones.
class exclusive_file {
    FILE * f = nullptr;
public:
    explicit exclusive_file(const std::string & path) {
#ifdef _WIN32
        const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
        if (fd >= 0) { f = _fdopen(fd, "wb"); if (!f) _close(fd); }
#else
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) { f = fdopen(fd, "wb"); if (!f) close(fd); }
#endif
        require(f != nullptr, "exclusive create failed (existing file or I/O error): " + path);
    }
    exclusive_file(const exclusive_file &) = delete;
    exclusive_file & operator=(const exclusive_file &) = delete;
    ~exclusive_file() { if (f) std::fclose(f); }
    void write(const void * p, size_t n) { require(std::fwrite(p, 1, n, f) == n, "artifact write failed"); }
    void finish() {
        FILE * old = f; f = nullptr;
        require(std::fclose(old) == 0, "artifact close failed");
    }
};
static std::string meta(llama_model * model, const char * key) {
    char value[256];
    int n = llama_model_meta_val_str(model, key, value, sizeof(value));
    require(n >= 0 && size_t(n) < sizeof(value), std::string("missing/oversized model metadata: ") + key);
    return std::string(value, size_t(n));
}
struct batch_owner {
    llama_batch value;
    explicit batch_owner(int n) : value(llama_batch_init(n, 0, 1)) {
        require(value.token && value.pos && value.n_seq_id && value.seq_id && value.logits, "batch allocation failed");
    }
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};
struct backend_owner {
    backend_owner() { llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
};
static void decode(llama_context * ctx, batch_owner & storage, const llama_token * ids,
                   int n, int pos, bool output_last) {
    auto & b = storage.value;
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i] = ids[i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = output_last && i == n - 1;
    }
    require(llama_decode(ctx, b) == 0, "llama_decode failed (including nonzero warnings)");
}
static json env_snapshot() {
    json j = json::object();
    for (const char * name : {"QWEN36_REAP_MASK", "QWEN36_REAP_TRACE", "QWEN36_REAP_MODEL_SHA256",
                              "LLAMA_MOE_CACHE_BATCH", "LLAMA_MOE_ELASTIC", "LLAMA_MOE_DEMAND_GPU",
                              "LLAMA_MOE_PHASE_CACHE", "LLAMA_MOE_MASSA", "LLAMA_MOE_MASSA_EMIVITA", "LLAMA_MOE_CACHE_CONGELA"}) {
        const char * v = std::getenv(name); j[name] = v ? json(v) : json(nullptr);
    }
    return j;
}
static int run(int argc, char ** argv) {
    common_params p;
    p.n_ctx = 4096; p.n_predict = 8; p.n_batch = 512; p.n_ubatch = 128;
    p.n_parallel = 1; p.n_sequences = 1; p.escape = false; p.offline = true; p.fit_params = false;
    std::map<std::string, std::string> own;
    std::vector<char *> args{argv[0]};
    const std::set<std::string> own_names = {"--gate-out", "--gate-run-id", "--gate-model-sha256", "--gate-teacher", "--gate-token-id"};
    const std::set<std::string> unary = {"--no-warmup", "--no-escape", "--offline", "--cpu-moe", "-cmoe", "--no-mmap"};
    const std::set<std::string> binary = {"-m", "--model", "-f", "--file", "-n", "--predict", "--n-predict",
        "-c", "--ctx-size", "-b", "--batch-size", "-ub", "--ubatch-size", "-ngl", "--gpu-layers", "--n-gpu-layers",
        "--moe-expert-cache", "--moe-expert-cache-inserts", "-t", "--threads", "-tb", "--threads-batch",
        "-ctk", "--cache-type-k", "-ctv", "--cache-type-v"};
    bool explicit_no_warmup = false, explicit_file = false, explicit_model = false;
    json invocation = json::array();
    for (int i = 0; i < argc; ++i) invocation.push_back(argv[i]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (own_names.count(a)) {
            require(i + 1 < argc && !own.count(a), "missing/duplicate helper option: " + a);
            own[a] = argv[++i];
        } else {
            require(unary.count(a) || binary.count(a), "unsupported gate option: " + a);
            explicit_no_warmup |= a == "--no-warmup";
            explicit_file |= a == "-f" || a == "--file";
            explicit_model |= a == "-m" || a == "--model";
            args.push_back(argv[i]);
            if (binary.count(a)) { require(i + 1 < argc, "missing value: " + a); args.push_back(argv[++i]); }
        }
    }
    const bool token_prompt = own.count("--gate-token-id") != 0;
    require(explicit_no_warmup && explicit_model && (explicit_file != token_prompt),
            "required: -m LOCAL.gguf --no-warmup and exactly one of -f RAW_UTF8.txt / --gate-token-id ID");
    require(!own["--gate-out"].empty() && !own["--gate-run-id"].empty(), "required: --gate-out PREFIX --gate-run-id LABEL");
    require(own["--gate-model-sha256"] == MODEL_SHA, "launcher must assert the expected model SHA256");
    common_init();
    const int common_argc = int(args.size()); args.push_back(nullptr);
    // COMPLETION inherits COMMON flags and exposes --no-warmup (COMMON alone does not).
    require(common_params_parse(common_argc, args.data(), p, LLAMA_EXAMPLE_COMPLETION), "common argument parse failed");
    require(!p.warmup && !p.escape && p.offline && !p.fit_params, "require no warmup/escape, offline, fit disabled; sanitize LLAMA_ARG_* environment");
    require(p.n_predict >= 1 && p.n_predict <= 16, "fixed token count must be 1..16 (default 8)");
    require(p.n_ctx > 0 && p.n_batch > 0 && p.n_ubatch > 0 && p.n_parallel == 1 && p.n_sequences == 1,
            "invalid context/batch or multiple sequences");
    require(!p.embedding && !p.sampling.backend_sampling, "require host-visible full logits, no embeddings/backend sampler");
    require(p.cb_eval == nullptr && p.cb_eval_user_data == nullptr, "unexpected common callback; helper does not replace runtime trace callback");
    require(p.speculative.types == std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE}, "no speculative/MTP mode");
    require(p.lora_adapters.empty() && p.control_vectors.empty() && p.kv_overrides.empty() && p.mmproj.path.empty(), "no adapters, model metadata overrides or multimodal inputs");
    require(!p.model.path.empty() && std::ifstream(p.model.path, std::ios::binary).good(), "model must be an existing local file");
    // The common -f implementation strips LF/processes escapes; restore exact raw bytes.
    p.prompt = token_prompt ? "" : read_bytes(p.prompt_file);
    require(token_prompt || !p.prompt.empty(), "empty raw prompt");
    (void) json(p.prompt).dump(); // nlohmann strict UTF-8 validation, no replacement/normalization
    json teacher;
    if (own.count("--gate-teacher")) teacher = json::parse(read_bytes(own.at("--gate-teacher")));
    exclusive_file logits_out(own["--gate-out"] + ".logits.f32");
    exclusive_file metadata_out(own["--gate-out"] + ".json");
    backend_owner backend;
    auto init = common_init_from_params(p);
    require(init && init->model() && init->context(), "common initialization failed");
    auto * model = init->model(); auto * ctx = init->context();
    require(meta(model, "general.architecture") == "qwen35moe" && llama_model_n_layer(model) == 40,
            "gate requires qwen35moe, exactly 40 layers");
    require(meta(model, "qwen35moe.expert_count") == "256" && meta(model, "qwen35moe.expert_used_count") == "8",
            "gate requires 256 experts, top-8");
    require(llama_model_n_layer_nextn(model) == 0, "MTP layers are forbidden");
    // The target model identity is pinned by the launcher hash; no draft context is created.
    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 0, "invalid vocabulary");
    std::vector<llama_token> prompt_ids;
    if (token_prompt) {
        const auto & text = own.at("--gate-token-id");
        require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos, "single token ID must be decimal nonnegative");
        const auto id = std::stoll(text);
        require(id >= 0 && id < n_vocab, "single token ID outside vocabulary");
        prompt_ids.push_back(llama_token(id)); // exactly one token: deliberately NO implicit BOS
    } else {
        prompt_ids = common_tokenize(vocab, p.prompt, true, false);
    }
    require(!prompt_ids.empty() && prompt_ids.size() + size_t(p.n_predict) <= llama_n_ctx(ctx), "prompt + fixed tokens exceed context");
    std::vector<llama_token> forced;
    if (!teacher.is_null()) {
        require(teacher.at("schema_version") == 1 && teacher.at("complete") == true && teacher.at("model_sha256") == MODEL_SHA,
                "teacher must be a complete gate metadata artifact for this model");
        require(teacher.at("prompt_token_ids") == json(prompt_ids) && teacher.at("prompt_utf8") == p.prompt, "teacher prompt mismatch");
        const auto & ids = teacher.at("token_ids");
        require(ids.is_array() && ids.size() == size_t(p.n_predict), "teacher length must equal fixed token count");
        for (const auto & id : ids) {
            require(id.is_number_integer() && id.get<int64_t>() >= 0 && id.get<int64_t>() < n_vocab, "teacher token out of range");
            forced.push_back(id.get<llama_token>());
        }
    }
    batch_owner batch(p.n_batch);
    json decode_calls = json::array();
    for (size_t pos = 0; pos < prompt_ids.size();) {
        const int n = int(std::min(size_t(p.n_batch), prompt_ids.size() - pos));
        decode(ctx, batch, prompt_ids.data() + pos, n, int(pos), pos + size_t(n) == prompt_ids.size());
        decode_calls.push_back({{"helper_call_ordinal_1based", decode_calls.size() + 1}, {"phase", "prefill"},
                                {"position_start", pos}, {"n_tokens", n}, {"seq_id", 0}});
        pos += size_t(n);
    }
    std::vector<llama_token> token_ids, greedy_ids;
    json steps = json::array();
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE754 float32 required");
    std::vector<unsigned char> row(size_t(n_vocab) * 4);
    for (int step = 0; step < p.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "missing complete last-token logits");
        llama_token top = 0;
        for (int id = 0; id < n_vocab; ++id) {
            require(std::isfinite(logits[id]), "nonfinite model logit: gate fails closed");
            if (logits[id] > logits[top]) top = id; // ties: lowest token ID
            uint32_t bits; std::memcpy(&bits, logits + id, 4);
            for (int byte = 0; byte < 4; ++byte) row[size_t(id) * 4 + byte] = (bits >> (8 * byte)) & 0xff;
        }
        logits_out.write(row.data(), row.size());
        const llama_token chosen = forced.empty() ? top : forced[size_t(step)];
        greedy_ids.push_back(top); token_ids.push_back(chosen);
        steps.push_back({{"step", step}, {"phase", step == 0 ? "last_prefill" : "incremental_decode"},
                         {"prefix_length", prompt_ids.size() + size_t(step)},
                         {"helper_call_ordinal_1based", decode_calls.size()}, 
                         {"logits_offset_bytes", uint64_t(step) * uint64_t(n_vocab) * 4},
                         {"greedy_token_id", top}, {"token_id", chosen}, {"is_eog", llama_vocab_is_eog(vocab, chosen)}});
        // Fixed-count numerical probe: EOG is recorded but deliberately does not stop generation.
        if (step + 1 < p.n_predict) {
            const size_t pos = prompt_ids.size() + size_t(step);
            decode(ctx, batch, &chosen, 1, int(pos), true);
            decode_calls.push_back({{"helper_call_ordinal_1based", decode_calls.size() + 1}, {"phase", "decode"},
                                    {"position_start", pos}, {"n_tokens", 1}, {"seq_id", 0}, {"token_id", chosen}});
        }
    }
    logits_out.finish(); // never publish complete metadata before successful logits close
    json result = {{"schema_version", 1}, {"complete", true}, {"run_id", own["--gate-run-id"]},
        {"model_sha256", MODEL_SHA}, {"model_hash_provenance", "launcher_asserted_not_hashed_by_helper"},
        {"model_path", p.model.path}, {"architecture", "qwen35moe"}, {"layer_count", 40}, {"expert_count", 256}, {"top_k", 8},
        {"dtype", "float32"}, {"endianness", "little"}, {"shape", {p.n_predict, n_vocab}},
        {"prompt_utf8", p.prompt}, {"prompt_bytes", p.prompt.size()}, {"prompt_token_ids", prompt_ids},
        {"token_ids", token_ids}, {"greedy_token_ids", greedy_ids}, {"steps", steps}, {"decode_calls", decode_calls},
        {"mode", forced.empty() ? "greedy" : "teacher_forcing"}, {"prompt_mode", token_prompt ? "single_token" : "raw_utf8"},
        {"add_special", !token_prompt}, {"parse_special", false},
        {"fixed_count_including_eog", true}, {"warmup", false}, {"target_only", true},
        {"settings", {{"n_ctx_requested", p.n_ctx}, {"n_ctx_actual", llama_n_ctx(ctx)}, {"n_batch", p.n_batch},
            {"n_ubatch", p.n_ubatch}, {"n_gpu_layers", p.n_gpu_layers}, {"moe_cache_slots", p.n_moe_cache_slots},
            {"moe_cache_inserts", p.n_moe_cache_inserts}, {"cache_type_k", int(p.cache_type_k)}, {"cache_type_v", int(p.cache_type_v)},
            {"threads", p.cpuparams.n_threads}, {"threads_batch", p.cpuparams_batch.n_threads}, {"flash_attn", int(p.flash_attn_type)}}},
        {"argv", invocation}, {"environment", env_snapshot()}};
    const std::string encoded = result.dump(2) + "\n";
    metadata_out.write(encoded.data(), encoded.size()); metadata_out.finish();
    return 0;
}
int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "native-gate: %s\n", e.what()); return 2; }
}
