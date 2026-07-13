#include <cuda_runtime.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kKiB = 1024ULL;
constexpr std::uint64_t kMiB = 1024ULL * kKiB;
constexpr std::uint64_t kGiB = 1024ULL * kMiB;
constexpr std::uint64_t kDefaultAvailableReserveBytes = 8ULL * kGiB;
constexpr std::uint64_t kMaximumBlockCount = 65536ULL;
constexpr std::uint64_t kMaximumStagingBytes = 1ULL * kGiB;
constexpr std::uint64_t kFnv1a64OffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;

static_assert(sizeof(std::size_t) >= 8,
              "This probe requires a 64-bit userspace for multi-GiB allocations.");

enum class Mode { Blocks, Single };
enum class ArenaApi { HostAlloc, MallocHost };

struct Options {
  Mode mode = Mode::Blocks;
  ArenaApi arena_api = ArenaApi::HostAlloc;
  std::uint64_t target_bytes = kGiB / 4;  // Deliberately small: 256 MiB.
  std::uint64_t step_bytes = 5ULL * kGiB;
  std::uint64_t touch_stride_bytes = 4ULL * kKiB;
  std::uint64_t staging_bytes = 16ULL * kMiB;
  std::uint64_t reserve_bytes = kDefaultAvailableReserveBytes;
  int device = 0;
  bool target_explicit = false;
};

const char* mode_name(Mode mode) {
  return mode == Mode::Blocks ? "blocks" : "single";
}

const char* api_name(ArenaApi api) {
  return api == ArenaApi::HostAlloc ? "cudaHostAlloc" : "cudaMallocHost";
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::optional<std::uint64_t> read_mem_available_bytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(status.ullAvailPhys);
#else
  std::ifstream input("/proc/meminfo");
  if (!input) {
    return std::nullopt;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("MemAvailable:", 0) != 0) {
      continue;
    }

    std::istringstream fields(line);
    std::string label;
    std::string unit;
    unsigned long long value_kib = 0;
    if (!(fields >> label >> value_kib >> unit) || label != "MemAvailable:" ||
        unit != "kB" ||
        value_kib > std::numeric_limits<std::uint64_t>::max() / kKiB) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(value_kib) * kKiB;
  }

  return std::nullopt;
#endif
}

bool preserves_host_headroom(std::uint64_t mem_available,
                             std::uint64_t bytes_to_allocate,
                             std::uint64_t reserve_bytes) {
  if (mem_available < reserve_bytes) {
    return false;
  }
  return bytes_to_allocate <= mem_available - reserve_bytes;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t* result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

void write_json_string(std::ostream& output, const std::string& value) {
  output << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (c < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(c) << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(c);
        }
    }
  }
  output << '"';
}

void write_optional_gib(std::ostream& output,
                        const std::optional<std::uint64_t>& bytes) {
  if (!bytes) {
    output << "null";
    return;
  }
  output << std::fixed << std::setprecision(6)
         << static_cast<long double>(*bytes) / static_cast<long double>(kGiB);
}

std::string checksum_hex(std::uint64_t checksum) {
  std::ostringstream output;
  output << "0x" << std::hex << std::setw(16) << std::setfill('0') << checksum;
  return output.str();
}

struct Record {
  std::string event;
  std::string status;
  std::string post_allocation_status;
  std::string detail;
  std::uint64_t attempted_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  double elapsed_seconds = 0.0;
  std::optional<double> operation_elapsed_seconds;
  std::optional<std::uint64_t> mem_available_before;
  std::optional<std::uint64_t> mem_available_after;
  std::optional<double> copy_bandwidth_gib_s;
  std::optional<std::uint64_t> expected_checksum;
  std::optional<std::uint64_t> actual_checksum;
  bool cuda_called = false;
  std::string cuda_call;
  cudaError_t cuda_error = cudaSuccess;
};

void emit_record(const Options& options, const Record& record) {
  std::ostream& output = std::cout;
  output << '{';
  output << "\"event\":";
  write_json_string(output, record.event);
  output << ",\"mode\":";
  write_json_string(output, mode_name(options.mode));
  output << ",\"allocator\":";
  write_json_string(output, api_name(options.arena_api));
  output << ",\"target_explicit\":"
         << (options.target_explicit ? "true" : "false");
  output << ",\"requested_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.target_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"step_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.step_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"reserve_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.reserve_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"attempted_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(record.attempted_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"allocated_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(record.allocated_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"elapsed_seconds\":" << std::fixed << std::setprecision(6)
         << record.elapsed_seconds;
  output << ",\"operation_elapsed_seconds\":";
  if (record.operation_elapsed_seconds) {
    output << std::fixed << std::setprecision(6)
           << *record.operation_elapsed_seconds;
  } else {
    output << "null";
  }
  output << ",\"cuda_call\":";
  if (record.cuda_called) {
    write_json_string(output, record.cuda_call);
  } else {
    output << "null";
  }
  output << ",\"cuda_code\":";
  if (record.cuda_called) {
    output << static_cast<int>(record.cuda_error);
  } else {
    output << "null";
  }
  output << ",\"cuda_name\":";
  if (record.cuda_called) {
    const char* name = cudaGetErrorName(record.cuda_error);
    write_json_string(output, name != nullptr ? name : "unknown");
  } else {
    output << "null";
  }
  output << ",\"cuda_error\":";
  if (record.cuda_called) {
    const char* message = cudaGetErrorString(record.cuda_error);
    write_json_string(output, message != nullptr ? message : "unknown");
  } else {
    output << "null";
  }
  output << ",\"mem_available_before_gib\":";
  write_optional_gib(output, record.mem_available_before);
  output << ",\"mem_available_after_gib\":";
  write_optional_gib(output, record.mem_available_after);
  output << ",\"copy_bandwidth_gib_s\":";
  if (record.copy_bandwidth_gib_s) {
    output << std::fixed << std::setprecision(6)
           << *record.copy_bandwidth_gib_s;
  } else {
    output << "null";
  }
  output << ",\"expected_checksum\":";
  if (record.expected_checksum) {
    write_json_string(output, checksum_hex(*record.expected_checksum));
  } else {
    output << "null";
  }
  output << ",\"actual_checksum\":";
  if (record.actual_checksum) {
    write_json_string(output, checksum_hex(*record.actual_checksum));
  } else {
    output << "null";
  }
  output << ",\"status\":";
  write_json_string(output, record.status);
  output << ",\"post_allocation_status\":";
  write_json_string(output, record.post_allocation_status);
  output << ",\"detail\":";
  write_json_string(output, record.detail);
  output << "}\n";
  output.flush();
}

bool parse_scaled_bytes(const std::string& text, std::uint64_t scale,
                        std::uint64_t* bytes, std::string* error) {
  errno = 0;
  char* end = nullptr;
  const long double value = std::strtold(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(value) || value <= 0.0L) {
    *error = "invalid positive number: " + text;
    return false;
  }

  const long double scaled = value * static_cast<long double>(scale);
  const long double maximum = static_cast<long double>(
      std::min<std::uint64_t>(std::numeric_limits<std::uint64_t>::max(),
                              std::numeric_limits<std::size_t>::max()));
  if (!std::isfinite(scaled) || scaled > maximum || scaled < 1.0L) {
    *error = "byte count is outside the supported range: " + text;
    return false;
  }

  *bytes = static_cast<std::uint64_t>(scaled);
  return true;
}

bool parse_device(const std::string& text, int* device, std::string* error) {
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    *error = "invalid CUDA device index: " + text;
    return false;
  }
  *device = static_cast<int>(value);
  return true;
}

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n"
      << "  --target-gib N       Total pinned arena target (default: 0.25)\n"
      << "  --step-gib N         Retained block size in blocks mode (default: 5)\n"
      << "  --mode blocks|single Separate retained blocks or one target-sized block\n"
      << "  --api hostalloc|mallochost  Arena allocation API (default: hostalloc)\n"
      << "  --touch-stride-mib N Touch one byte in a page per interval (default: 0.00390625, every 4 KiB page)\n"
      << "  --staging-mib N      Pinned copy-verification buffer, max 1024 (default: 16)\n"
      << "  --reserve-gib N      Minimum host MemAvailable reserve (default: 8)\n"
      << "  --device N           CUDA device index (default: 0)\n"
      << "  --help               Show this text\n";
}

bool take_value(int argc, char** argv, int* index, std::string* value,
                std::string* error) {
  if (*index + 1 >= argc) {
    *error = std::string("missing value for ") + argv[*index];
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

bool parse_options(int argc, char** argv, Options* options, bool* show_help,
                   std::string* error) {
  *show_help = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    std::string value;
    if (argument == "--help" || argument == "-h") {
      *show_help = true;
      return true;
    }
    if (argument == "--target-gib") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_scaled_bytes(value, kGiB, &options->target_bytes, error)) {
        return false;
      }
      options->target_explicit = true;
    } else if (argument == "--step-gib") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_scaled_bytes(value, kGiB, &options->step_bytes, error)) {
        return false;
      }
    } else if (argument == "--touch-stride-mib") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_scaled_bytes(value, kMiB, &options->touch_stride_bytes,
                              error)) {
        return false;
      }
    } else if (argument == "--staging-mib") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_scaled_bytes(value, kMiB, &options->staging_bytes, error)) {
        return false;
      }
    } else if (argument == "--reserve-gib") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_scaled_bytes(value, kGiB, &options->reserve_bytes, error)) {
        return false;
      }
    } else if (argument == "--device") {
      if (!take_value(argc, argv, &i, &value, error) ||
          !parse_device(value, &options->device, error)) {
        return false;
      }
    } else if (argument == "--mode") {
      if (!take_value(argc, argv, &i, &value, error)) {
        return false;
      }
      if (value == "blocks") {
        options->mode = Mode::Blocks;
      } else if (value == "single") {
        options->mode = Mode::Single;
      } else {
        *error = "--mode must be blocks or single";
        return false;
      }
    } else if (argument == "--api") {
      if (!take_value(argc, argv, &i, &value, error)) {
        return false;
      }
      if (value == "hostalloc") {
        options->arena_api = ArenaApi::HostAlloc;
      } else if (value == "mallochost") {
        options->arena_api = ArenaApi::MallocHost;
      } else {
        *error = "--api must be hostalloc or mallochost";
        return false;
      }
    } else {
      *error = "unknown option: " + argument;
      return false;
    }
  }

  if (options->staging_bytes > kMaximumStagingBytes) {
    *error = "--staging-mib must not exceed 1024 MiB";
    return false;
  }

  const std::uint64_t block_count =
      options->mode == Mode::Single
          ? 1ULL
          : options->target_bytes / options->step_bytes +
                (options->target_bytes % options->step_bytes != 0 ? 1ULL : 0ULL);
  if (block_count > kMaximumBlockCount) {
    *error = "requested target/step would create more than 65536 blocks";
    return false;
  }
  return true;
}

struct CleanupState {
  bool failed = false;
  const char* first_call = nullptr;
  cudaError_t first_error = cudaSuccess;
  std::uint64_t calls = 0;

  void record(const char* call, cudaError_t error) noexcept {
    ++calls;
    if (error != cudaSuccess && !failed) {
      failed = true;
      first_call = call;
      first_error = error;
    }
  }
};

struct HostBlock {
  void* pointer = nullptr;
  std::size_t bytes = 0;
};

class Resources {
 public:
  Resources() = default;
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  ~Resources() {
    CleanupState ignored;
    release(&ignored);
  }

  void reserve_arena(std::size_t count) { arena_.reserve(count); }

  void retain(void* pointer, std::size_t bytes) {
    arena_.push_back(HostBlock{pointer, bytes});
  }

  std::size_t arena_count() const { return arena_.size(); }

  std::uint8_t* arena_source() const {
    return arena_.empty()
               ? nullptr
               : static_cast<std::uint8_t*>(arena_.front().pointer);
  }

  std::size_t arena_source_bytes() const {
    return arena_.empty() ? 0 : arena_.front().bytes;
  }

  void release(CleanupState* cleanup) noexcept {
    if (released_) {
      return;
    }
    released_ = true;

    if (stream != nullptr) {
      cleanup->record("cudaStreamSynchronize(cleanup)",
                      cudaStreamSynchronize(stream));
    }
    if (stop_event != nullptr) {
      cleanup->record("cudaEventDestroy(stop)", cudaEventDestroy(stop_event));
      stop_event = nullptr;
    }
    if (start_event != nullptr) {
      cleanup->record("cudaEventDestroy(start)", cudaEventDestroy(start_event));
      start_event = nullptr;
    }
    if (stream != nullptr) {
      cleanup->record("cudaStreamDestroy", cudaStreamDestroy(stream));
      stream = nullptr;
    }
    if (device_buffer != nullptr) {
      cleanup->record("cudaFree(device)", cudaFree(device_buffer));
      device_buffer = nullptr;
    }
    if (staging_buffer != nullptr) {
      cleanup->record("cudaFreeHost(staging)",
                      cudaFreeHost(staging_buffer));
      staging_buffer = nullptr;
    }
    for (auto block = arena_.rbegin(); block != arena_.rend(); ++block) {
      if (block->pointer != nullptr) {
        cleanup->record("cudaFreeHost(arena)", cudaFreeHost(block->pointer));
        block->pointer = nullptr;
      }
    }
    arena_.clear();
  }

  void* device_buffer = nullptr;
  std::uint8_t* staging_buffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

 private:
  std::vector<HostBlock> arena_;
  bool released_ = false;
};

cudaError_t allocate_arena_block(ArenaApi api, void** pointer,
                                 std::size_t bytes) {
  if (api == ArenaApi::HostAlloc) {
    return cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
  }
  return cudaMallocHost(pointer, bytes);
}

void touch_intervals(void* pointer, std::uint64_t bytes,
                     std::uint64_t requested_stride, std::uint8_t seed) {
#if defined(_WIN32)
  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  const std::uint64_t page_size = system_info.dwPageSize > 0
                                      ? system_info.dwPageSize
                                      : 4096ULL;
#else
  const long page_size_result = sysconf(_SC_PAGESIZE);
  const std::uint64_t page_size =
      page_size_result > 0 ? static_cast<std::uint64_t>(page_size_result) : 4096ULL;
#endif
  const std::uint64_t stride = std::max(requested_stride, page_size);
  volatile std::uint8_t* data = static_cast<volatile std::uint8_t*>(pointer);

  std::uint64_t interval_start = 0;
  while (interval_start < bytes) {
    const std::uint64_t page_start = (interval_start / page_size) * page_size;
    data[page_start] = static_cast<std::uint8_t>(seed ^ (page_start >> 12));
    if (stride > bytes - interval_start) {
      break;
    }
    interval_start += stride;
  }

  const std::uint64_t last_page = ((bytes - 1) / page_size) * page_size;
  data[last_page] = static_cast<std::uint8_t>(seed ^ (last_page >> 12) ^ 0xA5U);
}

std::uint8_t pattern_byte(std::uint64_t index) {
  std::uint64_t mixed = index * 0x9E3779B185EBCA87ULL;
  mixed ^= mixed >> 29;
  mixed *= 0xC2B2AE3D27D4EB4FULL;
  mixed ^= mixed >> 32;
  return static_cast<std::uint8_t>(mixed & 0xFFU);
}

std::uint64_t fill_pattern(std::uint8_t* buffer, std::size_t bytes) {
  std::uint64_t checksum = kFnv1a64OffsetBasis;
  for (std::size_t i = 0; i < bytes; ++i) {
    const std::uint8_t value = pattern_byte(i);
    buffer[i] = value;
    checksum ^= value;
    checksum *= kFnv1a64Prime;
  }
  return checksum;
}

struct Verification {
  std::uint64_t checksum = kFnv1a64OffsetBasis;
  bool matches = true;
  std::optional<std::uint64_t> first_mismatch;
};

Verification verify_pattern(const std::uint8_t* buffer, std::size_t bytes) {
  Verification result;
  for (std::size_t i = 0; i < bytes; ++i) {
    const std::uint8_t actual = buffer[i];
    result.checksum ^= actual;
    result.checksum *= kFnv1a64Prime;
    if (result.matches && actual != pattern_byte(i)) {
      result.matches = false;
      result.first_mismatch = i;
    }
  }
  return result;
}

struct CudaFailure {
  bool set = false;
  const char* call = nullptr;
  cudaError_t error = cudaSuccess;

  void capture(const char* failed_call, cudaError_t failed_error) {
    if (!set && failed_error != cudaSuccess) {
      set = true;
      call = failed_call;
      error = failed_error;
    }
  }
};

int run_probe(const Options& options) {
  const Clock::time_point probe_start = Clock::now();
  const std::optional<std::uint64_t> initial_mem = read_mem_available_bytes();
  emit_record(options,
              Record{"start", "ready", "not_started", "JSON Lines output",
                     0, 0, seconds_since(probe_start), std::nullopt, initial_mem,
                     initial_mem});

  const std::uint64_t first_attempt =
      options.mode == Mode::Single
          ? options.target_bytes
          : std::min(options.target_bytes, options.step_bytes);
  std::uint64_t preflight_host_bytes = 0;
  const bool preflight_sum_ok = checked_add(
      first_attempt, options.staging_bytes, &preflight_host_bytes);
  if (!initial_mem || !preflight_sum_ok ||
      !preserves_host_headroom(*initial_mem, preflight_host_bytes,
                               options.reserve_bytes)) {
    const std::string detail =
        !initial_mem
            ? "cannot read host MemAvailable; refusing host allocation"
            : "preflight would violate the configured MemAvailable reserve; no CUDA call made";
    emit_record(options,
                Record{"safety_abort", "aborted", "not_started", detail,
                       first_attempt, 0, seconds_since(probe_start), std::nullopt,
                       initial_mem, read_mem_available_bytes()});
    emit_record(options,
                Record{"summary", "safety_abort", "not_started", detail, 0, 0,
                       seconds_since(probe_start), std::nullopt, initial_mem,
                       read_mem_available_bytes()});
    return 3;
  }

  const std::uint64_t block_count =
      options.mode == Mode::Single
          ? 1ULL
          : options.target_bytes / options.step_bytes +
                (options.target_bytes % options.step_bytes != 0 ? 1ULL : 0ULL);

  Resources resources;
  resources.reserve_arena(static_cast<std::size_t>(block_count));

  cudaError_t error = cudaSetDevice(options.device);
  emit_record(options,
              Record{"cuda_setup", error == cudaSuccess ? "ok" : "cuda_error",
                     "not_started", "select CUDA device", 0, 0,
                     seconds_since(probe_start), std::nullopt,
                     read_mem_available_bytes(), read_mem_available_bytes(),
                     std::nullopt, std::nullopt, std::nullopt, true,
                     "cudaSetDevice", error});
  if (error != cudaSuccess) {
    emit_record(options,
                Record{"summary", "cuda_setup_failed", "not_started",
                       "cudaSetDevice failed before arena allocation", 0, 0,
                       seconds_since(probe_start), std::nullopt,
                       read_mem_available_bytes(), read_mem_available_bytes(),
                       std::nullopt, std::nullopt, std::nullopt, true,
                       "cudaSetDevice", error});
    return 4;
  }

  std::uint64_t allocated_bytes = 0;
  std::string arena_status = "target_reached";
  CudaFailure arena_failure;

  while (allocated_bytes < options.target_bytes) {
    const std::uint64_t remaining = options.target_bytes - allocated_bytes;
    const std::uint64_t attempt =
        options.mode == Mode::Single ? remaining
                                     : std::min(remaining, options.step_bytes);
    const std::optional<std::uint64_t> mem_before =
        read_mem_available_bytes();
    std::uint64_t required_host_bytes = 0;
    const bool required_sum_ok = checked_add(
        attempt, options.staging_bytes, &required_host_bytes);
    if (!mem_before || !required_sum_ok ||
        !preserves_host_headroom(*mem_before, required_host_bytes,
                                 options.reserve_bytes)) {
      arena_status = !mem_before ? "safety_abort_memavailable_unreadable"
                                 : "safety_abort_insufficient_headroom";
      const std::string detail =
          !mem_before
              ? "MemAvailable became unreadable; refusing the next block"
              : "next block plus staging would violate the configured MemAvailable reserve";
      emit_record(options,
                  Record{"allocation", "aborted", "not_started", detail,
                         attempt, allocated_bytes, seconds_since(probe_start),
                         std::nullopt, mem_before, read_mem_available_bytes()});
      break;
    }

    void* pointer = nullptr;
    const Clock::time_point operation_start = Clock::now();
    error = allocate_arena_block(options.arena_api, &pointer,
                                 static_cast<std::size_t>(attempt));
    if (error == cudaSuccess) {
      resources.retain(pointer, static_cast<std::size_t>(attempt));
      touch_intervals(pointer, attempt, options.touch_stride_bytes,
                      static_cast<std::uint8_t>(resources.arena_count()));
      allocated_bytes += attempt;
    } else {
      arena_status = "cuda_allocation_failed";
      arena_failure.capture(api_name(options.arena_api), error);
    }
    const double operation_elapsed = seconds_since(operation_start);
    const std::optional<std::uint64_t> mem_after = read_mem_available_bytes();
    emit_record(options,
                Record{"allocation",
                       error == cudaSuccess ? "retained" : "cuda_error",
                       "not_started",
                       error == cudaSuccess
                           ? "block remains allocated until final cleanup"
                           : "arena allocation call failed; prior blocks remain retained",
                       attempt, allocated_bytes, seconds_since(probe_start),
                       operation_elapsed, mem_before, mem_after, std::nullopt,
                       std::nullopt, std::nullopt, true,
                       api_name(options.arena_api), error});
    if (error != cudaSuccess || options.mode == Mode::Single) {
      break;
    }
  }

  if (allocated_bytes == options.target_bytes) {
    arena_status = "target_reached";
  }
  emit_record(options,
              Record{"arena_result", arena_status, "pending",
                     "arena blocks are still retained", 0, allocated_bytes,
                     seconds_since(probe_start), std::nullopt,
                     initial_mem, read_mem_available_bytes(), std::nullopt,
                     std::nullopt, std::nullopt, arena_failure.set,
                     arena_failure.set ? arena_failure.call : "",
                     arena_failure.set ? arena_failure.error : cudaSuccess});

  std::string post_status = "not_started";
  CudaFailure post_failure;
  std::optional<double> h2d_bandwidth;
  std::optional<std::uint64_t> expected_checksum;
  std::optional<std::uint64_t> actual_checksum;
  const std::size_t verification_bytes = std::min<std::size_t>(
      static_cast<std::size_t>(options.staging_bytes),
      resources.arena_source_bytes());

  const std::optional<std::uint64_t> post_mem_before =
      read_mem_available_bytes();
  if (verification_bytes == 0) {
    post_status = "no_retained_arena_source";
    emit_record(options,
                Record{"post_allocation", "aborted", post_status,
                       "no retained arena bytes are available for direct DMA verification",
                       0, allocated_bytes, seconds_since(probe_start),
                       std::nullopt, post_mem_before,
                       read_mem_available_bytes()});
  } else if (!post_mem_before ||
      !preserves_host_headroom(*post_mem_before, options.staging_bytes,
                               options.reserve_bytes)) {
    post_status = !post_mem_before ? "safety_abort_memavailable_unreadable"
                                   : "safety_abort_insufficient_headroom";
    emit_record(options,
                Record{"post_allocation", "aborted", post_status,
                       "refusing staging allocation because the configured reserve cannot be guaranteed",
                       options.staging_bytes, allocated_bytes,
                       seconds_since(probe_start), std::nullopt, post_mem_before,
                       read_mem_available_bytes()});
  } else {
    error = cudaMalloc(&resources.device_buffer,
                       verification_bytes);
    post_failure.capture("cudaMalloc(device)", error);
    emit_record(options,
                Record{"device_buffer",
                       error == cudaSuccess ? "allocated" : "cuda_error",
                       error == cudaSuccess ? "device_allocated" : "cuda_error",
                       "device buffer has staging-buffer size",
                       verification_bytes, allocated_bytes,
                       seconds_since(probe_start), std::nullopt, post_mem_before,
                       read_mem_available_bytes(), std::nullopt, std::nullopt,
                       std::nullopt, true, "cudaMalloc(device)", error});

    if (error == cudaSuccess) {
      const std::optional<std::uint64_t> staging_mem_before =
          read_mem_available_bytes();
      if (!staging_mem_before ||
          !preserves_host_headroom(*staging_mem_before,
                                   options.staging_bytes,
                                   options.reserve_bytes)) {
        post_status = !staging_mem_before
                          ? "safety_abort_memavailable_unreadable"
                          : "safety_abort_insufficient_headroom";
        emit_record(options,
                    Record{"staging_buffer", "aborted", post_status,
                           "MemAvailable guard rejected the pinned staging allocation",
                           options.staging_bytes, allocated_bytes,
                           seconds_since(probe_start), std::nullopt,
                           staging_mem_before, read_mem_available_bytes()});
      } else {
        error = cudaMallocHost(
            reinterpret_cast<void**>(&resources.staging_buffer),
            verification_bytes);
        post_failure.capture("cudaMallocHost(staging)", error);
        emit_record(options,
                    Record{"staging_buffer",
                           error == cudaSuccess ? "allocated" : "cuda_error",
                           error == cudaSuccess ? "staging_allocated"
                                                : "cuda_error",
                           "additional pinned staging buffer",
                           verification_bytes, allocated_bytes,
                           seconds_since(probe_start), std::nullopt,
                           staging_mem_before, read_mem_available_bytes(),
                           std::nullopt, std::nullopt, std::nullopt, true,
                           "cudaMallocHost(staging)", error});
      }
    }

    const bool post_sequence_allowed = post_status == "not_started";
    if (post_sequence_allowed && !post_failure.set &&
        resources.staging_buffer != nullptr) {
      error = cudaStreamCreateWithFlags(&resources.stream,
                                        cudaStreamNonBlocking);
      post_failure.capture("cudaStreamCreateWithFlags", error);
    }
    if (post_sequence_allowed && !post_failure.set &&
        resources.stream != nullptr) {
      error = cudaEventCreate(&resources.start_event);
      post_failure.capture("cudaEventCreate(start)", error);
    }
    if (post_sequence_allowed && !post_failure.set &&
        resources.start_event != nullptr) {
      error = cudaEventCreate(&resources.stop_event);
      post_failure.capture("cudaEventCreate(stop)", error);
    }

    if (post_sequence_allowed && !post_failure.set &&
        resources.stop_event != nullptr) {
      expected_checksum = fill_pattern(resources.arena_source(),
                                       verification_bytes);
      error = cudaEventRecord(resources.start_event, resources.stream);
      post_failure.capture("cudaEventRecord(start)", error);
    }
    if (post_sequence_allowed && !post_failure.set) {
      error = cudaMemcpyAsync(resources.device_buffer,
                              resources.arena_source(), verification_bytes,
                              cudaMemcpyHostToDevice, resources.stream);
      post_failure.capture("cudaMemcpyAsync(H2D)", error);
    }
    if (post_sequence_allowed && !post_failure.set) {
      error = cudaEventRecord(resources.stop_event, resources.stream);
      post_failure.capture("cudaEventRecord(stop)", error);
    }
    if (post_sequence_allowed && !post_failure.set) {
      error = cudaEventSynchronize(resources.stop_event);
      post_failure.capture("cudaEventSynchronize(stop)", error);
    }

    float h2d_milliseconds = 0.0F;
    if (post_sequence_allowed && !post_failure.set) {
      error = cudaEventElapsedTime(&h2d_milliseconds, resources.start_event,
                                   resources.stop_event);
      post_failure.capture("cudaEventElapsedTime", error);
      if (error == cudaSuccess && h2d_milliseconds > 0.0F) {
        h2d_bandwidth =
            (static_cast<double>(verification_bytes) /
             static_cast<double>(kGiB)) /
            (static_cast<double>(h2d_milliseconds) / 1000.0);
      }
    }

    if (post_sequence_allowed) {
      emit_record(options,
                  Record{"h2d_copy",
                         post_failure.set ? "cuda_error" : "synchronized",
                         post_failure.set ? "cuda_error" : "h2d_complete",
                         post_failure.set
                             ? "H2D setup, copy, timing, or synchronization failed"
                             : "timed cudaMemcpyAsync H2D directly from retained arena completed",
                         verification_bytes, allocated_bytes,
                         seconds_since(probe_start), std::nullopt,
                         post_mem_before, read_mem_available_bytes(),
                         h2d_bandwidth, expected_checksum, std::nullopt, true,
                         post_failure.set ? post_failure.call
                                          : "cudaMemcpyAsync(H2D)",
                         post_failure.set ? post_failure.error : cudaSuccess});

      if (!post_failure.set) {
        std::memset(resources.staging_buffer, 0,
                    verification_bytes);
        error = cudaMemcpyAsync(resources.staging_buffer,
                                resources.device_buffer, verification_bytes,
                                cudaMemcpyDeviceToHost, resources.stream);
        post_failure.capture("cudaMemcpyAsync(D2H verify)", error);
      }
      if (!post_failure.set) {
        error = cudaStreamSynchronize(resources.stream);
        post_failure.capture("cudaStreamSynchronize(D2H verify)", error);
      }

      if (!post_failure.set) {
        const Verification verification = verify_pattern(
            resources.staging_buffer, verification_bytes);
        actual_checksum = verification.checksum;
        const bool verified = verification.matches && expected_checksum &&
                              actual_checksum == expected_checksum;
        post_status = verified ? "copy_verified" : "verification_failed";
        std::string detail = "round-trip pattern and checksum match";
        if (!verified) {
          detail = "round-trip data mismatch";
          if (verification.first_mismatch) {
            detail += " at byte " +
                      std::to_string(*verification.first_mismatch);
          }
        }
        emit_record(options,
                    Record{"copy_verify", verified ? "verified" : "mismatch",
                           post_status, detail, verification_bytes,
                           allocated_bytes, seconds_since(probe_start),
                           std::nullopt, post_mem_before,
                           read_mem_available_bytes(), h2d_bandwidth,
                           expected_checksum, actual_checksum, true,
                           "cudaStreamSynchronize(D2H verify)", cudaSuccess});
      } else {
        post_status = "cuda_error";
      }
    }
  }

  CleanupState cleanup;
  resources.release(&cleanup);
  const std::optional<std::uint64_t> final_mem = read_mem_available_bytes();
  emit_record(options,
              Record{"cleanup", cleanup.failed ? "cuda_error" : "complete",
                     post_status,
                     "attempted cleanup calls: " + std::to_string(cleanup.calls),
                     0, allocated_bytes, seconds_since(probe_start), std::nullopt,
                     final_mem, final_mem, h2d_bandwidth, expected_checksum,
                     actual_checksum, cleanup.calls != 0,
                     cleanup.failed ? cleanup.first_call : "cleanup_sequence",
                     cleanup.failed ? cleanup.first_error : cudaSuccess});

  int exit_code = 0;
  if (arena_status.rfind("safety_abort", 0) == 0) {
    exit_code = 3;
  } else if (arena_status == "cuda_allocation_failed") {
    exit_code = 4;
  }
  if (post_status == "verification_failed" && exit_code == 0) {
    exit_code = 6;
  } else if (post_status != "copy_verified" && exit_code == 0) {
    exit_code = 5;
  }
  if (cleanup.failed && exit_code == 0) {
    exit_code = 7;
  }

  CudaFailure summary_failure;
  if (arena_failure.set) {
    summary_failure = arena_failure;
  } else if (post_failure.set) {
    summary_failure = post_failure;
  } else if (cleanup.failed) {
    summary_failure.capture(cleanup.first_call, cleanup.first_error);
  }

  const std::string summary_status = exit_code == 0 ? "success" : "failed";
  const std::string summary_detail =
      "arena_status=" + arena_status + ", post_status=" + post_status;
  emit_record(options,
              Record{"summary", summary_status, post_status, summary_detail, 0,
                     allocated_bytes, seconds_since(probe_start), std::nullopt,
                     initial_mem, final_mem, h2d_bandwidth, expected_checksum,
                     actual_checksum, summary_failure.set,
                     summary_failure.set ? summary_failure.call : "",
                     summary_failure.set ? summary_failure.error : cudaSuccess});
  return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  bool show_help = false;
  std::string error;
  if (!parse_options(argc, argv, &options, &show_help, &error)) {
    std::cerr << "error: " << error << "\n";
    print_usage(argv[0]);
    return 2;
  }
  if (show_help) {
    print_usage(argv[0]);
    return 0;
  }

  try {
    return run_probe(options);
  } catch (const std::exception& exception) {
    std::cerr << "fatal host exception: " << exception.what() << "\n";
    return 10;
  }
}
