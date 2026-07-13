#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN64)
#error This probe intentionally supports only the Windows x64 CUDA Driver ABI.
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kKiB = 1024ULL;
constexpr std::uint64_t kMiB = 1024ULL * kKiB;
constexpr std::uint64_t kGiB = 1024ULL * kMiB;
constexpr std::uint64_t kMaximumBlockCount = 65536ULL;
constexpr std::uint64_t kMaximumStagingBytes = 1ULL * kGiB;
constexpr std::uint64_t kFnv1a64OffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;

// Minimal CUDA Driver API declarations. These deliberately do not depend on
// cuda.h or cuda.lib. CUDAAPI is __stdcall on Windows; x64 has one native
// calling convention, but retaining the annotation documents the ABI contract.
#define CUDAAPI __stdcall

using CUresult = int;
using CUdevice = int;
using CUdeviceptr_v1 = unsigned int;
using CUdeviceptr_v2 = unsigned long long;
struct CUctx_st;
struct CUstream_st;
using CUcontext = CUctx_st*;
using CUstream = CUstream_st*;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr unsigned int CU_MEMHOSTALLOC_DEFAULT = 0;
constexpr unsigned int CU_STREAM_NON_BLOCKING = 1;

static_assert(sizeof(void*) == 8, "Windows x64 pointer size is required");
static_assert(sizeof(std::size_t) == 8, "Windows x64 size_t is required");
static_assert(sizeof(CUresult) == 4, "CUresult must have a 32-bit ABI");
static_assert(sizeof(CUdevice) == 4, "CUdevice must have a 32-bit ABI");
static_assert(sizeof(CUdeviceptr_v1) == 4,
              "legacy CUdeviceptr must have a 32-bit ABI");
static_assert(sizeof(CUdeviceptr_v2) == 8,
              "CUdeviceptr_v2 must have a 64-bit ABI on x64");
static_assert(sizeof(CUcontext) == 8, "CUcontext must be pointer-sized");
static_assert(sizeof(CUstream) == 8, "CUstream must be pointer-sized");

using PfnCuInit = CUresult(CUDAAPI*)(unsigned int);
using PfnCuGetErrorName = CUresult(CUDAAPI*)(CUresult, const char**);
using PfnCuGetErrorString = CUresult(CUDAAPI*)(CUresult, const char**);
using PfnCuDeviceGet = CUresult(CUDAAPI*)(CUdevice*, int);
using PfnCuCtxCreate =
    CUresult(CUDAAPI*)(CUcontext*, unsigned int, CUdevice);
using PfnCuCtxDestroy = CUresult(CUDAAPI*)(CUcontext);
using PfnCuMemHostAlloc =
    CUresult(CUDAAPI*)(void**, std::size_t, unsigned int);
using PfnCuMemFreeHost = CUresult(CUDAAPI*)(void*);
using PfnCuMemAllocV2 = CUresult(CUDAAPI*)(CUdeviceptr_v2*, std::size_t);
using PfnCuMemAllocV1 = CUresult(CUDAAPI*)(CUdeviceptr_v1*, unsigned int);
using PfnCuMemFreeV2 = CUresult(CUDAAPI*)(CUdeviceptr_v2);
using PfnCuMemFreeV1 = CUresult(CUDAAPI*)(CUdeviceptr_v1);
using PfnCuStreamCreate = CUresult(CUDAAPI*)(CUstream*, unsigned int);
using PfnCuStreamSynchronize = CUresult(CUDAAPI*)(CUstream);
using PfnCuStreamDestroy = CUresult(CUDAAPI*)(CUstream);
using PfnCuMemcpyHtoDAsyncV2 = CUresult(CUDAAPI*)(
    CUdeviceptr_v2, const void*, std::size_t, CUstream);
using PfnCuMemcpyHtoDAsyncV1 = CUresult(CUDAAPI*)(
    CUdeviceptr_v1, const void*, unsigned int, CUstream);
using PfnCuMemcpyDtoHV2 =
    CUresult(CUDAAPI*)(void*, CUdeviceptr_v2, std::size_t);
using PfnCuMemcpyDtoHV1 =
    CUresult(CUDAAPI*)(void*, CUdeviceptr_v1, unsigned int);

enum class Mode { Blocks, Single };

struct Options {
  Mode mode = Mode::Blocks;
  std::uint64_t target_bytes = kGiB / 4;  // Safe default: 0.25 GiB.
  std::uint64_t step_bytes = 1ULL * kGiB;
  std::uint64_t staging_bytes = 16ULL * kMiB;
  std::uint64_t reserve_bytes = 2ULL * kGiB;
  int device = 0;
  bool target_explicit = false;
};

const char* mode_name(Mode mode) {
  return mode == Mode::Blocks ? "blocks" : "single";
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::optional<std::uint64_t> read_windows_available_bytes() {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(status.ullAvailPhys);
}

bool preserves_host_headroom(std::uint64_t available_bytes,
                             std::uint64_t bytes_to_allocate,
                             std::uint64_t reserve_bytes) {
  return available_bytes >= reserve_bytes &&
         bytes_to_allocate <= available_bytes - reserve_bytes;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t* result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

std::wstring system_cuda_dll_path(std::string* error) {
  wchar_t system_directory[MAX_PATH]{};
  const UINT length =
      GetSystemDirectoryW(system_directory, static_cast<UINT>(MAX_PATH));
  if (length == 0 || length >= MAX_PATH) {
    *error = "GetSystemDirectoryW failed, win32_code=" +
             std::to_string(GetLastError());
    return {};
  }

  std::wstring path(system_directory, length);
  if (!path.empty() && path.back() != L'\\') {
    path.push_back(L'\\');
  }
  path += L"nvcuda.dll";
  return path;
}

class CudaDriver {
 public:
  CudaDriver() = default;
  CudaDriver(const CudaDriver&) = delete;
  CudaDriver& operator=(const CudaDriver&) = delete;

  ~CudaDriver() {
    if (module_ != nullptr) {
      FreeLibrary(module_);
    }
  }

  bool load(std::string* error) {
    const std::wstring path = system_cuda_dll_path(error);
    if (path.empty()) {
      return false;
    }

    module_ = LoadLibraryW(path.c_str());
    if (module_ == nullptr) {
      *error = "LoadLibraryW(System32\\nvcuda.dll) failed, win32_code=" +
               std::to_string(GetLastError());
      return false;
    }

    init_ = resolve<PfnCuInit>("cuInit");
    get_error_name_ = resolve<PfnCuGetErrorName>("cuGetErrorName");
    get_error_string_ = resolve<PfnCuGetErrorString>("cuGetErrorString");
    device_get_ = resolve<PfnCuDeviceGet>("cuDeviceGet");
    mem_host_alloc_ = resolve<PfnCuMemHostAlloc>("cuMemHostAlloc");
    mem_free_host_ = resolve<PfnCuMemFreeHost>("cuMemFreeHost");
    stream_create_ = resolve<PfnCuStreamCreate>("cuStreamCreate");
    stream_synchronize_ =
        resolve<PfnCuStreamSynchronize>("cuStreamSynchronize");

    std::vector<std::string> missing;
    require(init_, "cuInit", &missing);
    require(device_get_, "cuDeviceGet", &missing);
    require(mem_host_alloc_, "cuMemHostAlloc", &missing);
    require(mem_free_host_, "cuMemFreeHost", &missing);
    require(stream_create_, "cuStreamCreate", &missing);
    require(stream_synchronize_, "cuStreamSynchronize", &missing);

    const PfnCuCtxCreate context_create_v2 =
        resolve<PfnCuCtxCreate>("cuCtxCreate_v2");
    const PfnCuCtxDestroy context_destroy_v2 =
        resolve<PfnCuCtxDestroy>("cuCtxDestroy_v2");
    if (context_create_v2 != nullptr && context_destroy_v2 != nullptr) {
      context_create_ = context_create_v2;
      context_destroy_ = context_destroy_v2;
      context_abi_ = "v2";
    } else {
      context_create_ = resolve<PfnCuCtxCreate>("cuCtxCreate");
      context_destroy_ = resolve<PfnCuCtxDestroy>("cuCtxDestroy");
      context_abi_ = "legacy";
      require(context_create_, "cuCtxCreate_v2 or cuCtxCreate", &missing);
      require(context_destroy_, "cuCtxDestroy_v2 or cuCtxDestroy", &missing);
    }

    const PfnCuStreamDestroy stream_destroy_v2 =
        resolve<PfnCuStreamDestroy>("cuStreamDestroy_v2");
    if (stream_destroy_v2 != nullptr) {
      stream_destroy_ = stream_destroy_v2;
      stream_destroy_abi_ = "v2";
    } else {
      stream_destroy_ = resolve<PfnCuStreamDestroy>("cuStreamDestroy");
      stream_destroy_abi_ = "legacy";
      require(stream_destroy_, "cuStreamDestroy_v2 or cuStreamDestroy",
              &missing);
    }

    const PfnCuMemAllocV2 mem_alloc_v2 =
        resolve<PfnCuMemAllocV2>("cuMemAlloc_v2");
    const PfnCuMemFreeV2 mem_free_v2 =
        resolve<PfnCuMemFreeV2>("cuMemFree_v2");
    const PfnCuMemcpyHtoDAsyncV2 h2d_v2 =
        resolve<PfnCuMemcpyHtoDAsyncV2>("cuMemcpyHtoDAsync_v2");
    const PfnCuMemcpyDtoHV2 dtoh_v2 =
        resolve<PfnCuMemcpyDtoHV2>("cuMemcpyDtoH_v2");

    if (mem_alloc_v2 != nullptr && mem_free_v2 != nullptr &&
        h2d_v2 != nullptr && dtoh_v2 != nullptr) {
      mem_alloc_v2_ = mem_alloc_v2;
      mem_free_v2_ = mem_free_v2;
      h2d_v2_ = h2d_v2;
      dtoh_v2_ = dtoh_v2;
      device_memory_v2_ = true;
      device_memory_abi_ = "v2";
    } else {
      // The CUDA 2.x exports are a genuinely different ABI: 32-bit
      // CUdeviceptr_v1 and unsigned-int byte counts. Resolve all four legacy
      // functions as one family rather than casting them to v2 prototypes or
      // mixing allocation/copy/free versions for the same resource.
      mem_alloc_v1_ = resolve<PfnCuMemAllocV1>("cuMemAlloc");
      mem_free_v1_ = resolve<PfnCuMemFreeV1>("cuMemFree");
      h2d_v1_ = resolve<PfnCuMemcpyHtoDAsyncV1>("cuMemcpyHtoDAsync");
      dtoh_v1_ = resolve<PfnCuMemcpyDtoHV1>("cuMemcpyDtoH");
      device_memory_v2_ = false;
      device_memory_abi_ = "legacy_v1_32bit";
      require(mem_alloc_v1_, "complete cuMemAlloc legacy family", &missing);
      require(mem_free_v1_, "complete cuMemFree legacy family", &missing);
      require(h2d_v1_, "complete cuMemcpyHtoDAsync legacy family", &missing);
      require(dtoh_v1_, "complete cuMemcpyDtoH legacy family", &missing);
    }

    if (!missing.empty()) {
      std::ostringstream detail;
      detail << "nvcuda.dll is missing required Driver API symbol(s): ";
      for (std::size_t i = 0; i < missing.size(); ++i) {
        if (i != 0) {
          detail << ", ";
        }
        detail << missing[i];
      }
      *error = detail.str();
      return false;
    }

    ready_ = true;
    abi_summary_ = "context_" + context_abi_ + ";device_memory_" +
                   device_memory_abi_ + ";stream_destroy_" +
                   stream_destroy_abi_;
    return true;
  }

  bool ready() const { return ready_; }
  const std::string& abi_summary() const { return abi_summary_; }

  CUresult init() const { return init_(0); }
  CUresult device_get(CUdevice* device, int ordinal) const {
    return device_get_(device, ordinal);
  }
  CUresult context_create(CUcontext* context, CUdevice device) const {
    return context_create_(context, 0, device);
  }
  CUresult context_destroy(CUcontext context) const {
    return context_destroy_(context);
  }
  CUresult mem_host_alloc(void** pointer, std::size_t bytes) const {
    return mem_host_alloc_(pointer, bytes, CU_MEMHOSTALLOC_DEFAULT);
  }
  CUresult mem_free_host(void* pointer) const {
    return mem_free_host_(pointer);
  }
  CUresult stream_create(CUstream* stream) const {
    return stream_create_(stream, CU_STREAM_NON_BLOCKING);
  }
  CUresult stream_synchronize(CUstream stream) const {
    return stream_synchronize_(stream);
  }
  CUresult stream_destroy(CUstream stream) const {
    return stream_destroy_(stream);
  }

  CUresult mem_alloc(std::uint64_t* pointer, std::size_t bytes) const {
    if (device_memory_v2_) {
      CUdeviceptr_v2 result = 0;
      const CUresult error = mem_alloc_v2_(&result, bytes);
      if (error == CUDA_SUCCESS) {
        *pointer = result;
      }
      return error;
    }
    if (bytes > std::numeric_limits<unsigned int>::max()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    CUdeviceptr_v1 result = 0;
    const CUresult error =
        mem_alloc_v1_(&result, static_cast<unsigned int>(bytes));
    if (error == CUDA_SUCCESS) {
      *pointer = result;
    }
    return error;
  }

  CUresult mem_free(std::uint64_t pointer) const {
    return device_memory_v2_
               ? mem_free_v2_(static_cast<CUdeviceptr_v2>(pointer))
               : mem_free_v1_(static_cast<CUdeviceptr_v1>(pointer));
  }

  CUresult memcpy_h2d_async(std::uint64_t destination, const void* source,
                            std::size_t bytes, CUstream stream) const {
    if (device_memory_v2_) {
      return h2d_v2_(static_cast<CUdeviceptr_v2>(destination), source, bytes,
                     stream);
    }
    if (bytes > std::numeric_limits<unsigned int>::max()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return h2d_v1_(static_cast<CUdeviceptr_v1>(destination), source,
                   static_cast<unsigned int>(bytes), stream);
  }

  CUresult memcpy_dtoh(void* destination, std::uint64_t source,
                       std::size_t bytes) const {
    if (device_memory_v2_) {
      return dtoh_v2_(destination, static_cast<CUdeviceptr_v2>(source), bytes);
    }
    if (bytes > std::numeric_limits<unsigned int>::max()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return dtoh_v1_(destination, static_cast<CUdeviceptr_v1>(source),
                    static_cast<unsigned int>(bytes));
  }

  const char* context_create_name() const {
    return context_abi_ == "v2" ? "cuCtxCreate_v2" : "cuCtxCreate";
  }
  const char* context_destroy_name() const {
    return context_abi_ == "v2" ? "cuCtxDestroy_v2" : "cuCtxDestroy";
  }
  const char* stream_destroy_name() const {
    return stream_destroy_abi_ == "v2" ? "cuStreamDestroy_v2"
                                        : "cuStreamDestroy";
  }
  const char* mem_alloc_name() const {
    return device_memory_v2_ ? "cuMemAlloc_v2" : "cuMemAlloc";
  }
  const char* mem_free_name() const {
    return device_memory_v2_ ? "cuMemFree_v2" : "cuMemFree";
  }
  const char* h2d_name() const {
    return device_memory_v2_ ? "cuMemcpyHtoDAsync_v2"
                             : "cuMemcpyHtoDAsync";
  }
  const char* dtoh_name() const {
    return device_memory_v2_ ? "cuMemcpyDtoH_v2" : "cuMemcpyDtoH";
  }

  std::string error_name(CUresult error) const {
    const char* text = nullptr;
    if (get_error_name_ != nullptr &&
        get_error_name_(error, &text) == CUDA_SUCCESS && text != nullptr) {
      return text;
    }
    return "CUresult_" + std::to_string(error);
  }

  std::string error_string(CUresult error) const {
    const char* text = nullptr;
    if (get_error_string_ != nullptr &&
        get_error_string_(error, &text) == CUDA_SUCCESS && text != nullptr) {
      return text;
    }
    return "CUDA Driver error code " + std::to_string(error);
  }

 private:
  template <typename Function>
  Function resolve(const char* name) const {
    return reinterpret_cast<Function>(GetProcAddress(module_, name));
  }

  template <typename Function>
  static void require(Function function, const char* name,
                      std::vector<std::string>* missing) {
    if (function == nullptr) {
      missing->emplace_back(name);
    }
  }

  HMODULE module_ = nullptr;
  bool ready_ = false;
  bool device_memory_v2_ = false;
  std::string context_abi_;
  std::string device_memory_abi_;
  std::string stream_destroy_abi_;
  std::string abi_summary_;

  PfnCuInit init_ = nullptr;
  PfnCuGetErrorName get_error_name_ = nullptr;
  PfnCuGetErrorString get_error_string_ = nullptr;
  PfnCuDeviceGet device_get_ = nullptr;
  PfnCuCtxCreate context_create_ = nullptr;
  PfnCuCtxDestroy context_destroy_ = nullptr;
  PfnCuMemHostAlloc mem_host_alloc_ = nullptr;
  PfnCuMemFreeHost mem_free_host_ = nullptr;
  PfnCuStreamCreate stream_create_ = nullptr;
  PfnCuStreamSynchronize stream_synchronize_ = nullptr;
  PfnCuStreamDestroy stream_destroy_ = nullptr;
  PfnCuMemAllocV2 mem_alloc_v2_ = nullptr;
  PfnCuMemFreeV2 mem_free_v2_ = nullptr;
  PfnCuMemcpyHtoDAsyncV2 h2d_v2_ = nullptr;
  PfnCuMemcpyDtoHV2 dtoh_v2_ = nullptr;
  PfnCuMemAllocV1 mem_alloc_v1_ = nullptr;
  PfnCuMemFreeV1 mem_free_v1_ = nullptr;
  PfnCuMemcpyHtoDAsyncV1 h2d_v1_ = nullptr;
  PfnCuMemcpyDtoHV1 dtoh_v1_ = nullptr;
};

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
                 << static_cast<unsigned int>(c) << std::dec
                 << std::setfill(' ');
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

void write_optional_double(std::ostream& output,
                           const std::optional<double>& value) {
  if (!value) {
    output << "null";
    return;
  }
  output << std::fixed << std::setprecision(6) << *value;
}

void write_optional_u64(std::ostream& output,
                        const std::optional<std::uint64_t>& value) {
  if (value) {
    output << *value;
  } else {
    output << "null";
  }
}

std::string checksum_hex(std::uint64_t checksum) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "0x" << std::hex << std::setw(16) << std::setfill('0') << checksum;
  return output.str();
}

struct Record {
  Record(std::string event_value, std::string status_value,
         std::string detail_value, double elapsed_value)
      : event(std::move(event_value)),
        status(std::move(status_value)),
        detail(std::move(detail_value)),
        elapsed_seconds(elapsed_value) {}

  std::string event;
  std::string status;
  std::string detail;
  std::uint64_t attempted_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  std::uint64_t verification_bytes = 0;
  double elapsed_seconds = 0.0;
  std::optional<double> operation_elapsed_seconds;
  std::optional<double> copy_bandwidth_gib_s;
  std::optional<std::uint64_t> windows_available_before;
  std::optional<std::uint64_t> windows_available_after;
  std::optional<std::uint64_t> materialized_pages;
  std::optional<std::uint64_t> first_mismatch;
  std::optional<std::uint64_t> expected_checksum;
  std::optional<std::uint64_t> actual_checksum;
  bool cuda_called = false;
  std::string cuda_call;
  CUresult cuda_error = CUDA_SUCCESS;
};

void set_cuda_result(Record* record, const char* call, CUresult error) {
  record->cuda_called = true;
  record->cuda_call = call;
  record->cuda_error = error;
}

void emit_record(const Options& options, const Record& record,
                 const CudaDriver* driver) {
  std::ostream& output = std::cout;
  output << '{';
  output << "\"schema_version\":1";
  output << ",\"pid\":" << GetCurrentProcessId();
  output << ",\"event\":";
  write_json_string(output, record.event);
  output << ",\"mode\":";
  write_json_string(output, mode_name(options.mode));
  output << ",\"allocator\":\"cuMemHostAlloc\"";
  output << ",\"target_explicit\":"
         << (options.target_explicit ? "true" : "false");
  output << ",\"requested_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.target_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"step_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.step_bytes) /
                static_cast<long double>(kGiB);
  output << ",\"staging_gib\":" << std::fixed << std::setprecision(6)
         << static_cast<long double>(options.staging_bytes) /
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
  output << ",\"verification_bytes\":" << record.verification_bytes;
  output << ",\"elapsed_seconds\":" << std::fixed << std::setprecision(6)
         << record.elapsed_seconds;
  output << ",\"operation_elapsed_seconds\":";
  write_optional_double(output, record.operation_elapsed_seconds);
  output << ",\"copy_bandwidth_gib_s\":";
  write_optional_double(output, record.copy_bandwidth_gib_s);
  output << ",\"windows_available_before_gib\":";
  write_optional_gib(output, record.windows_available_before);
  output << ",\"windows_available_after_gib\":";
  write_optional_gib(output, record.windows_available_after);
  output << ",\"materialized_pages\":";
  write_optional_u64(output, record.materialized_pages);
  output << ",\"first_mismatch_byte\":";
  write_optional_u64(output, record.first_mismatch);
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
  output << ",\"driver_abi\":";
  if (driver != nullptr && driver->ready()) {
    write_json_string(output, driver->abi_summary());
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
    output << record.cuda_error;
  } else {
    output << "null";
  }
  output << ",\"cuda_name\":";
  if (record.cuda_called) {
    write_json_string(output, driver != nullptr
                                  ? driver->error_name(record.cuda_error)
                                  : "unavailable");
  } else {
    output << "null";
  }
  output << ",\"cuda_error\":";
  if (record.cuda_called) {
    write_json_string(output, driver != nullptr
                                  ? driver->error_string(record.cuda_error)
                                  : "unavailable");
  } else {
    output << "null";
  }
  output << ",\"status\":";
  write_json_string(output, record.status);
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
  if (!std::isfinite(scaled) || scaled < 1.0L || scaled > maximum) {
    *error = "byte count is outside the supported x64 range: " + text;
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
      << "  --target-gib N   Total pinned arena target (default: 0.25); 50 must be explicit\n"
      << "  --step-gib N     Retained block size in blocks mode (default: 1)\n"
      << "  --mode blocks|single  Separate blocks or one target-sized allocation\n"
      << "  --staging-mib N  Extra pinned round-trip buffer, max 1024 (default: 16)\n"
      << "  --reserve-gib N  Minimum Windows available-physical reserve (default: 2)\n"
      << "  --device N       CUDA device ordinal (default: 0)\n"
      << "  --help           Show this text\n";
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
                (options->target_bytes % options->step_bytes != 0 ? 1ULL
                                                                  : 0ULL);
  if (block_count > kMaximumBlockCount) {
    *error = "requested target/step would create more than 65536 blocks";
    return false;
  }
  return true;
}

std::uint64_t fnv1a_update(std::uint64_t checksum, std::uint8_t value) {
  checksum ^= value;
  checksum *= kFnv1a64Prime;
  return checksum;
}

std::uint8_t materialized_byte(std::uint64_t block_index,
                               std::uint64_t offset) {
  std::uint64_t mixed = offset ^ (block_index * 0x9E3779B185EBCA87ULL);
  mixed ^= mixed >> 29;
  mixed *= 0xC2B2AE3D27D4EB4FULL;
  mixed ^= mixed >> 32;
  return static_cast<std::uint8_t>((mixed & 0xFFU) ^ 0xA5U);
}

struct Materialization {
  std::uint64_t pages = 0;
  std::uint64_t expected_checksum = kFnv1a64OffsetBasis;
  std::uint64_t actual_checksum = kFnv1a64OffsetBasis;
  std::optional<std::uint64_t> first_mismatch;

  bool valid() const {
    return !first_mismatch && expected_checksum == actual_checksum;
  }
};

Materialization materialize_and_validate(void* pointer, std::uint64_t bytes,
                                         std::uint64_t block_index) {
  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  const std::uint64_t page_size =
      system_info.dwPageSize == 0
          ? 4096ULL
          : static_cast<std::uint64_t>(system_info.dwPageSize);
  volatile std::uint8_t* data =
      static_cast<volatile std::uint8_t*>(pointer);
  Materialization result;

  std::uint64_t offset = 0;
  for (;;) {
    const std::uint8_t expected = materialized_byte(block_index, offset);
    data[offset] = expected;
    result.expected_checksum = fnv1a_update(result.expected_checksum, expected);
    ++result.pages;
    if (page_size > bytes - 1 - offset) {
      break;
    }
    offset += page_size;
  }

  const std::uint64_t last_byte = bytes - 1;
  const bool endpoint_is_extra = last_byte != offset;
  if (endpoint_is_extra) {
    const std::uint8_t expected = materialized_byte(block_index, last_byte);
    data[last_byte] = expected;
    result.expected_checksum = fnv1a_update(result.expected_checksum, expected);
  }

  offset = 0;
  for (;;) {
    const std::uint8_t expected = materialized_byte(block_index, offset);
    const std::uint8_t actual = data[offset];
    result.actual_checksum = fnv1a_update(result.actual_checksum, actual);
    if (!result.first_mismatch && actual != expected) {
      result.first_mismatch = offset;
    }
    if (page_size > bytes - 1 - offset) {
      break;
    }
    offset += page_size;
  }

  if (endpoint_is_extra) {
    const std::uint8_t expected = materialized_byte(block_index, last_byte);
    const std::uint8_t actual = data[last_byte];
    result.actual_checksum = fnv1a_update(result.actual_checksum, actual);
    if (!result.first_mismatch && actual != expected) {
      result.first_mismatch = last_byte;
    }
  }
  return result;
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
    checksum = fnv1a_update(checksum, value);
  }
  return checksum;
}

struct Verification {
  std::uint64_t checksum = kFnv1a64OffsetBasis;
  std::optional<std::uint64_t> first_mismatch;
};

Verification verify_pattern(const std::uint8_t* buffer, std::size_t bytes) {
  Verification result;
  for (std::size_t i = 0; i < bytes; ++i) {
    const std::uint8_t actual = buffer[i];
    result.checksum = fnv1a_update(result.checksum, actual);
    if (!result.first_mismatch && actual != pattern_byte(i)) {
      result.first_mismatch = static_cast<std::uint64_t>(i);
    }
  }
  return result;
}

struct HostBlock {
  void* pointer = nullptr;
  std::size_t bytes = 0;
};

struct CleanupState {
  bool failed = false;
  const char* first_call = nullptr;
  CUresult first_error = CUDA_SUCCESS;
  std::uint64_t calls = 0;

  void record(const char* call, CUresult error) noexcept {
    ++calls;
    if (!failed && error != CUDA_SUCCESS) {
      failed = true;
      first_call = call;
      first_error = error;
    }
  }
};

class Resources {
 public:
  explicit Resources(const CudaDriver* driver) : driver_(driver) {}
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  ~Resources() {
    CleanupState ignored;
    release(&ignored);
  }

  void reserve_arena(std::size_t count) { arena_.reserve(count); }

  void retain_arena(void* pointer, std::size_t bytes) {
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
      cleanup->record("cuStreamSynchronize(cleanup)",
                      driver_->stream_synchronize(stream));
      cleanup->record(driver_->stream_destroy_name(),
                      driver_->stream_destroy(stream));
      stream = nullptr;
    }
    if (device_allocated) {
      cleanup->record(driver_->mem_free_name(),
                      driver_->mem_free(device_pointer));
      device_pointer = 0;
      device_allocated = false;
    }
    if (staging_buffer != nullptr) {
      cleanup->record("cuMemFreeHost(staging)",
                      driver_->mem_free_host(staging_buffer));
      staging_buffer = nullptr;
    }
    for (auto block = arena_.rbegin(); block != arena_.rend(); ++block) {
      if (block->pointer != nullptr) {
        cleanup->record("cuMemFreeHost(arena)",
                        driver_->mem_free_host(block->pointer));
        block->pointer = nullptr;
      }
    }
    arena_.clear();
    if (context != nullptr) {
      cleanup->record(driver_->context_destroy_name(),
                      driver_->context_destroy(context));
      context = nullptr;
    }
  }

  CUcontext context = nullptr;
  CUstream stream = nullptr;
  std::uint64_t device_pointer = 0;
  bool device_allocated = false;
  std::uint8_t* staging_buffer = nullptr;

 private:
  const CudaDriver* driver_;
  std::vector<HostBlock> arena_;
  bool released_ = false;
};

struct CudaFailure {
  bool set = false;
  std::string call;
  CUresult error = CUDA_SUCCESS;

  void capture(const char* failed_call, CUresult failed_error) {
    if (!set && failed_error != CUDA_SUCCESS) {
      set = true;
      call = failed_call;
      error = failed_error;
    }
  }
};

int emit_setup_failure(const Options& options, Clock::time_point probe_start,
                       const CudaDriver& driver, const char* call,
                       CUresult error, const std::string& detail) {
  Record setup("cuda_setup", "cuda_error", detail,
               seconds_since(probe_start));
  setup.windows_available_before = read_windows_available_bytes();
  setup.windows_available_after = setup.windows_available_before;
  set_cuda_result(&setup, call, error);
  emit_record(options, setup, &driver);

  Record summary("summary", "failed", detail, seconds_since(probe_start));
  summary.windows_available_before = setup.windows_available_before;
  summary.windows_available_after = read_windows_available_bytes();
  set_cuda_result(&summary, call, error);
  emit_record(options, summary, &driver);
  return 4;
}

int run_probe(const Options& options) {
  const Clock::time_point probe_start = Clock::now();
  const std::optional<std::uint64_t> initial_mem =
      read_windows_available_bytes();
  Record start("start", "ready",
               "native Windows x64 probe; JSON Lines use invariant culture",
               seconds_since(probe_start));
  start.windows_available_before = initial_mem;
  start.windows_available_after = initial_mem;
  emit_record(options, start, nullptr);

  const std::uint64_t first_attempt =
      options.mode == Mode::Single
          ? options.target_bytes
          : std::min(options.target_bytes, options.step_bytes);
  std::uint64_t preflight_bytes = 0;
  const bool preflight_sum_ok =
      checked_add(first_attempt, options.staging_bytes, &preflight_bytes);
  if (!initial_mem || !preflight_sum_ok ||
      !preserves_host_headroom(*initial_mem, preflight_bytes,
                               options.reserve_bytes)) {
    const std::string detail =
        !initial_mem
            ? "GlobalMemoryStatusEx failed; no CUDA allocation was attempted"
            : "first arena allocation plus staging would violate the Windows available-memory reserve";
    Record abort("safety_abort", "aborted", detail,
                 seconds_since(probe_start));
    abort.attempted_bytes = first_attempt;
    abort.windows_available_before = initial_mem;
    abort.windows_available_after = read_windows_available_bytes();
    emit_record(options, abort, nullptr);
    Record summary("summary", "safety_abort", detail,
                   seconds_since(probe_start));
    summary.windows_available_before = initial_mem;
    summary.windows_available_after = read_windows_available_bytes();
    emit_record(options, summary, nullptr);
    return 3;
  }

  CudaDriver driver;
  std::string load_error;
  if (!driver.load(&load_error)) {
    Record load("driver_load", "failed", load_error,
                seconds_since(probe_start));
    load.windows_available_before = initial_mem;
    load.windows_available_after = read_windows_available_bytes();
    emit_record(options, load, nullptr);
    Record summary("summary", "failed", load_error,
                   seconds_since(probe_start));
    summary.windows_available_before = initial_mem;
    summary.windows_available_after = read_windows_available_bytes();
    emit_record(options, summary, nullptr);
    return 4;
  }

  Record load("driver_load", "loaded",
              "loaded System32\\nvcuda.dll and resolved the minimal Driver API",
              seconds_since(probe_start));
  load.windows_available_before = initial_mem;
  load.windows_available_after = read_windows_available_bytes();
  emit_record(options, load, &driver);

  CUresult error = driver.init();
  if (error != CUDA_SUCCESS) {
    return emit_setup_failure(options, probe_start, driver, "cuInit", error,
                              "cuInit failed");
  }

  CUdevice device = 0;
  error = driver.device_get(&device, options.device);
  if (error != CUDA_SUCCESS) {
    return emit_setup_failure(options, probe_start, driver, "cuDeviceGet",
                              error, "cuDeviceGet failed");
  }

  Resources resources(&driver);
  CUcontext context = nullptr;
  error = driver.context_create(&context, device);
  if (error != CUDA_SUCCESS) {
    return emit_setup_failure(options, probe_start, driver,
                              driver.context_create_name(), error,
                              "CUDA context creation failed");
  }
  resources.context = context;

  Record setup("cuda_setup", "ok", "CUDA device and context are ready",
               seconds_since(probe_start));
  setup.windows_available_before = read_windows_available_bytes();
  setup.windows_available_after = setup.windows_available_before;
  set_cuda_result(&setup, driver.context_create_name(), CUDA_SUCCESS);
  emit_record(options, setup, &driver);

  const std::uint64_t block_count =
      options.mode == Mode::Single
          ? 1ULL
          : options.target_bytes / options.step_bytes +
                (options.target_bytes % options.step_bytes != 0 ? 1ULL
                                                                  : 0ULL);
  resources.reserve_arena(static_cast<std::size_t>(block_count));

  std::uint64_t allocated_bytes = 0;
  std::string arena_status = "target_reached";
  CudaFailure arena_failure;
  std::optional<std::uint64_t> materialization_expected;
  std::optional<std::uint64_t> materialization_actual;

  while (allocated_bytes < options.target_bytes) {
    const std::uint64_t remaining = options.target_bytes - allocated_bytes;
    const std::uint64_t attempt =
        options.mode == Mode::Single ? remaining
                                     : std::min(remaining, options.step_bytes);
    const std::optional<std::uint64_t> mem_before =
        read_windows_available_bytes();
    std::uint64_t required_host_bytes = 0;
    const bool required_sum_ok =
        checked_add(attempt, options.staging_bytes, &required_host_bytes);
    if (!mem_before || !required_sum_ok ||
        !preserves_host_headroom(*mem_before, required_host_bytes,
                                 options.reserve_bytes)) {
      arena_status = !mem_before ? "safety_abort_memavailable_unreadable"
                                 : "safety_abort_insufficient_headroom";
      const std::string detail =
          !mem_before
              ? "GlobalMemoryStatusEx failed before the next arena allocation"
              : "next arena block plus staging would violate the Windows available-memory reserve";
      Record allocation("allocation", "aborted", detail,
                        seconds_since(probe_start));
      allocation.attempted_bytes = attempt;
      allocation.allocated_bytes = allocated_bytes;
      allocation.windows_available_before = mem_before;
      allocation.windows_available_after = read_windows_available_bytes();
      emit_record(options, allocation, &driver);
      break;
    }

    void* pointer = nullptr;
    const Clock::time_point allocation_start = Clock::now();
    error = driver.mem_host_alloc(&pointer, static_cast<std::size_t>(attempt));
    const double allocation_elapsed = seconds_since(allocation_start);
    if (error == CUDA_SUCCESS) {
      resources.retain_arena(pointer, static_cast<std::size_t>(attempt));
      allocated_bytes += attempt;
    } else {
      arena_status = "cuda_allocation_failed";
      arena_failure.capture("cuMemHostAlloc(arena)", error);
    }

    Record allocation("allocation",
                      error == CUDA_SUCCESS ? "retained" : "cuda_error",
                      error == CUDA_SUCCESS
                          ? "block is retained until final cleanup"
                          : "cuMemHostAlloc failed; earlier blocks remain retained",
                      seconds_since(probe_start));
    allocation.attempted_bytes = attempt;
    allocation.allocated_bytes = allocated_bytes;
    allocation.operation_elapsed_seconds = allocation_elapsed;
    allocation.windows_available_before = mem_before;
    allocation.windows_available_after = read_windows_available_bytes();
    set_cuda_result(&allocation, "cuMemHostAlloc(arena)", error);
    emit_record(options, allocation, &driver);

    if (error != CUDA_SUCCESS) {
      break;
    }

    const Clock::time_point materialization_start = Clock::now();
    const Materialization materialization = materialize_and_validate(
        pointer, attempt, resources.arena_count());
    materialization_expected = materialization.expected_checksum;
    materialization_actual = materialization.actual_checksum;
    Record materialized(
        "materialization", materialization.valid() ? "validated" : "mismatch",
        materialization.valid()
            ? "wrote and reread one byte in every Windows page plus the allocation endpoint"
            : "page materialization readback did not match",
        seconds_since(probe_start));
    materialized.attempted_bytes = attempt;
    materialized.allocated_bytes = allocated_bytes;
    materialized.operation_elapsed_seconds =
        seconds_since(materialization_start);
    materialized.windows_available_before = allocation.windows_available_after;
    materialized.windows_available_after = read_windows_available_bytes();
    materialized.materialized_pages = materialization.pages;
    materialized.first_mismatch = materialization.first_mismatch;
    materialized.expected_checksum = materialization.expected_checksum;
    materialized.actual_checksum = materialization.actual_checksum;
    emit_record(options, materialized, &driver);

    if (!materialization.valid()) {
      arena_status = "materialization_failed";
      break;
    }
    if (options.mode == Mode::Single) {
      break;
    }
  }

  if (allocated_bytes == options.target_bytes &&
      arena_status != "materialization_failed") {
    arena_status = "target_reached";
  }

  Record arena_result("arena_result", arena_status,
                      "all successful arena allocations remain retained",
                      seconds_since(probe_start));
  arena_result.allocated_bytes = allocated_bytes;
  arena_result.windows_available_before = initial_mem;
  arena_result.windows_available_after = read_windows_available_bytes();
  arena_result.expected_checksum = materialization_expected;
  arena_result.actual_checksum = materialization_actual;
  if (arena_failure.set) {
    set_cuda_result(&arena_result, arena_failure.call.c_str(),
                    arena_failure.error);
  }
  emit_record(options, arena_result, &driver);

  std::string post_status = "not_started";
  CudaFailure post_failure;
  std::optional<double> h2d_bandwidth;
  std::optional<std::uint64_t> expected_checksum;
  std::optional<std::uint64_t> actual_checksum;
  const std::size_t verification_bytes = std::min<std::size_t>(
      static_cast<std::size_t>(options.staging_bytes),
      resources.arena_source_bytes());
  const std::optional<std::uint64_t> post_mem_before =
      read_windows_available_bytes();

  if (verification_bytes == 0) {
    post_status = "no_retained_arena_source";
    Record post("post_allocation", "aborted",
                "no retained arena bytes are available for the DMA test",
                seconds_since(probe_start));
    post.allocated_bytes = allocated_bytes;
    post.windows_available_before = post_mem_before;
    post.windows_available_after = read_windows_available_bytes();
    emit_record(options, post, &driver);
  } else if (!post_mem_before ||
             !preserves_host_headroom(*post_mem_before,
                                      options.staging_bytes,
                                      options.reserve_bytes)) {
    post_status = !post_mem_before ? "safety_abort_memavailable_unreadable"
                                   : "safety_abort_insufficient_headroom";
    Record post("post_allocation", "aborted",
                "Windows available-memory guard rejected the extra pinned staging allocation",
                seconds_since(probe_start));
    post.attempted_bytes = options.staging_bytes;
    post.allocated_bytes = allocated_bytes;
    post.verification_bytes = verification_bytes;
    post.windows_available_before = post_mem_before;
    post.windows_available_after = read_windows_available_bytes();
    emit_record(options, post, &driver);
  } else {
    void* staging = nullptr;
    error = driver.mem_host_alloc(&staging, verification_bytes);
    post_failure.capture("cuMemHostAlloc(staging)", error);
    if (error == CUDA_SUCCESS) {
      resources.staging_buffer = static_cast<std::uint8_t*>(staging);
    }
    Record staging_record(
        "staging_buffer", error == CUDA_SUCCESS ? "allocated" : "cuda_error",
        "extra page-locked host buffer used only for D2H verification",
        seconds_since(probe_start));
    staging_record.attempted_bytes = verification_bytes;
    staging_record.allocated_bytes = allocated_bytes;
    staging_record.verification_bytes = verification_bytes;
    staging_record.windows_available_before = post_mem_before;
    staging_record.windows_available_after = read_windows_available_bytes();
    set_cuda_result(&staging_record, "cuMemHostAlloc(staging)", error);
    emit_record(options, staging_record, &driver);

    if (!post_failure.set) {
      error = driver.mem_alloc(&resources.device_pointer, verification_bytes);
      post_failure.capture(driver.mem_alloc_name(), error);
      if (error == CUDA_SUCCESS) {
        resources.device_allocated = true;
      }
      Record device_record(
          "device_buffer", error == CUDA_SUCCESS ? "allocated" : "cuda_error",
          "device buffer matches the verification window",
          seconds_since(probe_start));
      device_record.allocated_bytes = allocated_bytes;
      device_record.verification_bytes = verification_bytes;
      device_record.windows_available_before = read_windows_available_bytes();
      device_record.windows_available_after =
          device_record.windows_available_before;
      set_cuda_result(&device_record, driver.mem_alloc_name(), error);
      emit_record(options, device_record, &driver);
    }

    if (!post_failure.set) {
      error = driver.stream_create(&resources.stream);
      post_failure.capture("cuStreamCreate", error);
      Record stream_record(
          "stream", error == CUDA_SUCCESS ? "created" : "cuda_error",
          "non-blocking CUDA stream for the H2D transfer",
          seconds_since(probe_start));
      stream_record.allocated_bytes = allocated_bytes;
      stream_record.verification_bytes = verification_bytes;
      stream_record.windows_available_before = read_windows_available_bytes();
      stream_record.windows_available_after =
          stream_record.windows_available_before;
      set_cuda_result(&stream_record, "cuStreamCreate", error);
      emit_record(options, stream_record, &driver);
    }

    if (!post_failure.set) {
      expected_checksum =
          fill_pattern(resources.arena_source(), verification_bytes);
      std::memset(resources.staging_buffer, 0, verification_bytes);

      const Clock::time_point copy_start = Clock::now();
      error = driver.memcpy_h2d_async(
          resources.device_pointer, resources.arena_source(),
          verification_bytes, resources.stream);
      post_failure.capture(driver.h2d_name(), error);
      if (!post_failure.set) {
        error = driver.stream_synchronize(resources.stream);
        post_failure.capture("cuStreamSynchronize(H2D)", error);
      }
      const double h2d_seconds = seconds_since(copy_start);
      if (!post_failure.set && h2d_seconds > 0.0) {
        h2d_bandwidth =
            (static_cast<double>(verification_bytes) /
             static_cast<double>(kGiB)) /
            h2d_seconds;
      }

      Record h2d("h2d_copy",
                 post_failure.set ? "cuda_error" : "synchronized",
                 post_failure.set
                     ? "cuMemcpyHtoDAsync or stream synchronization failed"
                     : "copied directly from the retained pinned arena and synchronized the stream",
                 seconds_since(probe_start));
      h2d.allocated_bytes = allocated_bytes;
      h2d.verification_bytes = verification_bytes;
      h2d.operation_elapsed_seconds = h2d_seconds;
      h2d.copy_bandwidth_gib_s = h2d_bandwidth;
      h2d.windows_available_before = post_mem_before;
      h2d.windows_available_after = read_windows_available_bytes();
      h2d.expected_checksum = expected_checksum;
      set_cuda_result(&h2d,
                      post_failure.set ? post_failure.call.c_str()
                                       : driver.h2d_name(),
                      post_failure.set ? post_failure.error : CUDA_SUCCESS);
      emit_record(options, h2d, &driver);
    }

    if (!post_failure.set) {
      const Clock::time_point copy_start = Clock::now();
      error = driver.memcpy_dtoh(resources.staging_buffer,
                                 resources.device_pointer,
                                 verification_bytes);
      post_failure.capture(driver.dtoh_name(), error);
      Record dtoh("dtoh_copy",
                  error == CUDA_SUCCESS ? "complete" : "cuda_error",
                  "synchronous device-to-host copy into the extra pinned staging buffer",
                  seconds_since(probe_start));
      dtoh.allocated_bytes = allocated_bytes;
      dtoh.verification_bytes = verification_bytes;
      dtoh.operation_elapsed_seconds = seconds_since(copy_start);
      dtoh.windows_available_before = read_windows_available_bytes();
      dtoh.windows_available_after = dtoh.windows_available_before;
      dtoh.expected_checksum = expected_checksum;
      set_cuda_result(&dtoh, driver.dtoh_name(), error);
      emit_record(options, dtoh, &driver);
    }

    if (!post_failure.set) {
      const Verification verification =
          verify_pattern(resources.staging_buffer, verification_bytes);
      actual_checksum = verification.checksum;
      const bool verified = !verification.first_mismatch &&
                            expected_checksum == actual_checksum;
      post_status = verified ? "copy_verified" : "verification_failed";
      std::string detail = verified ? "round-trip pattern and checksum match"
                                    : "round-trip data mismatch";
      if (verification.first_mismatch) {
        detail += " at byte " +
                  std::to_string(*verification.first_mismatch);
      }
      Record verify("copy_verify", verified ? "verified" : "mismatch",
                    detail, seconds_since(probe_start));
      verify.allocated_bytes = allocated_bytes;
      verify.verification_bytes = verification_bytes;
      verify.copy_bandwidth_gib_s = h2d_bandwidth;
      verify.windows_available_before = post_mem_before;
      verify.windows_available_after = read_windows_available_bytes();
      verify.first_mismatch = verification.first_mismatch;
      verify.expected_checksum = expected_checksum;
      verify.actual_checksum = actual_checksum;
      emit_record(options, verify, &driver);
    } else {
      post_status = "cuda_error";
    }
  }

  CleanupState cleanup;
  resources.release(&cleanup);
  const std::optional<std::uint64_t> final_mem =
      read_windows_available_bytes();
  Record cleanup_record(
      "cleanup", cleanup.failed ? "cuda_error" : "complete",
      "attempted cleanup calls: " + std::to_string(cleanup.calls),
      seconds_since(probe_start));
  cleanup_record.allocated_bytes = allocated_bytes;
  cleanup_record.verification_bytes = verification_bytes;
  cleanup_record.windows_available_before = final_mem;
  cleanup_record.windows_available_after = final_mem;
  cleanup_record.copy_bandwidth_gib_s = h2d_bandwidth;
  cleanup_record.expected_checksum = expected_checksum;
  cleanup_record.actual_checksum = actual_checksum;
  if (cleanup.calls != 0) {
    set_cuda_result(&cleanup_record,
                    cleanup.failed ? cleanup.first_call
                                   : "cleanup_sequence",
                    cleanup.failed ? cleanup.first_error : CUDA_SUCCESS);
  }
  emit_record(options, cleanup_record, &driver);

  int exit_code = 0;
  if (arena_status.rfind("safety_abort", 0) == 0) {
    exit_code = 3;
  } else if (arena_status == "cuda_allocation_failed") {
    exit_code = 4;
  } else if (arena_status == "materialization_failed") {
    exit_code = 6;
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

  const std::string summary_detail =
      "arena_status=" + arena_status + ", post_status=" + post_status;
  Record summary("summary", exit_code == 0 ? "success" : "failed",
                 summary_detail, seconds_since(probe_start));
  summary.allocated_bytes = allocated_bytes;
  summary.verification_bytes = verification_bytes;
  summary.copy_bandwidth_gib_s = h2d_bandwidth;
  summary.windows_available_before = initial_mem;
  summary.windows_available_after = final_mem;
  summary.expected_checksum = expected_checksum;
  summary.actual_checksum = actual_checksum;
  if (summary_failure.set) {
    set_cuda_result(&summary, summary_failure.call.c_str(),
                    summary_failure.error);
  }
  emit_record(options, summary, &driver);
  return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_ALL, "C");
  std::locale::global(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  std::cerr.imbue(std::locale::classic());

  Options options;
  bool show_help = false;
  std::string error;
  if (!parse_options(argc, argv, &options, &show_help, &error)) {
    std::cerr << "error: " << error << '\n';
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
    std::cerr << "fatal host exception: " << exception.what() << '\n';
    return 10;
  }
}
