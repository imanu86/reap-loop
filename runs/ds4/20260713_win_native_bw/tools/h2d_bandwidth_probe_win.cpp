// h2d_bandwidth_probe_win.cpp
//
// Native Windows CUDA Driver API H2D bandwidth probe. Loads nvcuda.dll
// dynamically (same pattern as cuda_pinned_arena_probe_win.cpp), so it needs
// no cuda.lib / CUDA SDK at build time -- only the driver, which ships with
// the GPU driver package. Cross-compiled with MinGW from WSL, executed
// natively on Windows via powershell.exe (NOT wsl.exe).
//
// Measures Host->Device memcpy bandwidth for:
//   - PINNED host memory (cuMemHostAlloc)
//   - PAGEABLE host memory (plain malloc)
// at 1 GiB and 2 GiB, N=12 timed copies per condition, first copy (warmup)
// discarded, mean/median/stddev GiB/s reported over the remaining copies.
//
// Output: JSON Lines to stdout (one record per condition) + a plain-text
// summary table to stderr for quick human reading.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr int kIterations = 12;      // 1 warmup + 11 measured
constexpr int kWarmupCount = 1;

#define CUDAAPI __stdcall
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUstream_st;
using CUcontext = CUctx_st*;
using CUstream = CUstream_st*;
constexpr CUresult CUDA_SUCCESS = 0;

using PfnCuInit = CUresult(CUDAAPI*)(unsigned int);
using PfnCuGetErrorString = CUresult(CUDAAPI*)(CUresult, const char**);
using PfnCuDeviceGet = CUresult(CUDAAPI*)(CUdevice*, int);
using PfnCuCtxCreate = CUresult(CUDAAPI*)(CUcontext*, unsigned int, CUdevice);
using PfnCuCtxDestroy = CUresult(CUDAAPI*)(CUcontext);
using PfnCuMemHostAlloc = CUresult(CUDAAPI*)(void**, std::size_t, unsigned int);
using PfnCuMemFreeHost = CUresult(CUDAAPI*)(void*);
using PfnCuMemAlloc = CUresult(CUDAAPI*)(CUdeviceptr*, std::size_t);
using PfnCuMemFree = CUresult(CUDAAPI*)(CUdeviceptr);
using PfnCuMemcpyHtoD = CUresult(CUDAAPI*)(CUdeviceptr, const void*, std::size_t);
using PfnCuCtxSynchronize = CUresult(CUDAAPI*)(void);

struct Driver {
  HMODULE mod = nullptr;
  PfnCuInit cuInit_ = nullptr;
  PfnCuGetErrorString cuGetErrorString_ = nullptr;
  PfnCuDeviceGet cuDeviceGet_ = nullptr;
  PfnCuCtxCreate cuCtxCreate_ = nullptr;
  PfnCuCtxDestroy cuCtxDestroy_ = nullptr;
  PfnCuMemHostAlloc cuMemHostAlloc_ = nullptr;
  PfnCuMemFreeHost cuMemFreeHost_ = nullptr;
  PfnCuMemAlloc cuMemAlloc_ = nullptr;
  PfnCuMemFree cuMemFree_ = nullptr;
  PfnCuMemcpyHtoD cuMemcpyHtoD_ = nullptr;
  PfnCuCtxSynchronize cuCtxSynchronize_ = nullptr;

  template <typename F>
  F resolve(const char* name) {
    return reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(mod, name)));
  }

  bool load(std::string* err) {
    wchar_t sysdir[MAX_PATH]{};
    UINT n = GetSystemDirectoryW(sysdir, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) { *err = "GetSystemDirectoryW failed"; return false; }
    std::wstring path(sysdir, n);
    if (!path.empty() && path.back() != L'\\') path.push_back(L'\\');
    path += L"nvcuda.dll";
    mod = LoadLibraryW(path.c_str());
    if (!mod) { *err = "LoadLibraryW(nvcuda.dll) failed, code=" + std::to_string(GetLastError()); return false; }

    cuInit_ = resolve<PfnCuInit>("cuInit");
    cuGetErrorString_ = resolve<PfnCuGetErrorString>("cuGetErrorString");
    cuDeviceGet_ = resolve<PfnCuDeviceGet>("cuDeviceGet");
    cuCtxCreate_ = resolve<PfnCuCtxCreate>("cuCtxCreate_v2");
    cuCtxDestroy_ = resolve<PfnCuCtxDestroy>("cuCtxDestroy_v2");
    cuMemHostAlloc_ = resolve<PfnCuMemHostAlloc>("cuMemHostAlloc");
    cuMemFreeHost_ = resolve<PfnCuMemFreeHost>("cuMemFreeHost");
    cuMemAlloc_ = resolve<PfnCuMemAlloc>("cuMemAlloc_v2");
    cuMemFree_ = resolve<PfnCuMemFree>("cuMemFree_v2");
    cuMemcpyHtoD_ = resolve<PfnCuMemcpyHtoD>("cuMemcpyHtoD_v2");
    cuCtxSynchronize_ = resolve<PfnCuCtxSynchronize>("cuCtxSynchronize");

    if (!cuInit_ || !cuDeviceGet_ || !cuCtxCreate_ || !cuCtxDestroy_ ||
        !cuMemHostAlloc_ || !cuMemFreeHost_ || !cuMemAlloc_ || !cuMemFree_ ||
        !cuMemcpyHtoD_ || !cuCtxSynchronize_) {
      *err = "nvcuda.dll is missing one or more required Driver API symbols";
      return false;
    }
    return true;
  }

  std::string errstr(CUresult r) {
    const char* s = nullptr;
    if (cuGetErrorString_ && cuGetErrorString_(r, &s) == CUDA_SUCCESS && s) return s;
    return "CUresult_" + std::to_string(r);
  }
};

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Stats {
  double mean_gib_s = 0.0;
  double median_gib_s = 0.0;
  double stddev_gib_s = 0.0;
  double min_gib_s = 0.0;
  double max_gib_s = 0.0;
};

Stats compute_stats(std::vector<double> samples) {
  Stats s;
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  s.min_gib_s = samples.front();
  s.max_gib_s = samples.back();
  double sum = 0.0;
  for (double v : samples) sum += v;
  s.mean_gib_s = sum / static_cast<double>(samples.size());
  const std::size_t mid = samples.size() / 2;
  s.median_gib_s = (samples.size() % 2 == 0)
                        ? (samples[mid - 1] + samples[mid]) / 2.0
                        : samples[mid];
  double var = 0.0;
  for (double v : samples) var += (v - s.mean_gib_s) * (v - s.mean_gib_s);
  var /= static_cast<double>(samples.size());
  s.stddev_gib_s = std::sqrt(var);
  return s;
}

void json_string(std::ostream& o, const std::string& v) {
  o << '"';
  for (char c : v) {
    if (c == '"' || c == '\\') o << '\\';
    o << c;
  }
  o << '"';
}

// Run kIterations blocking H2D copies of `bytes` from src (host) to dst
// (device). Discards the first kWarmupCount iterations, returns Stats over
// the rest plus the raw per-iteration seconds for transparency.
bool run_condition(Driver& drv, const char* label, const void* src,
                   CUdeviceptr dst, std::size_t bytes,
                   std::vector<double>* raw_gib_s, std::string* error) {
  raw_gib_s->clear();
  for (int i = 0; i < kIterations; ++i) {
    const Clock::time_point t0 = Clock::now();
    CUresult r = drv.cuMemcpyHtoD_(dst, src, bytes);
    if (r != CUDA_SUCCESS) {
      *error = std::string(label) + ": cuMemcpyHtoD_v2 failed: " + drv.errstr(r);
      return false;
    }
    r = drv.cuCtxSynchronize_();
    if (r != CUDA_SUCCESS) {
      *error = std::string(label) + ": cuCtxSynchronize failed: " + drv.errstr(r);
      return false;
    }
    const double elapsed = seconds_since(t0);
    const double gib_s = (static_cast<double>(bytes) / static_cast<double>(kGiB)) / elapsed;
    raw_gib_s->push_back(gib_s);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t size_gib_list_default[] = {1, 2};
  std::vector<std::uint64_t> sizes_bytes;
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      sizes_bytes.push_back(static_cast<std::uint64_t>(std::atof(argv[i]) * static_cast<double>(kGiB)));
    }
  } else {
    for (std::size_t g : size_gib_list_default) sizes_bytes.push_back(g * kGiB);
  }

  Driver drv;
  std::string err;
  if (!drv.load(&err)) {
    std::cerr << "driver_load_failed: " << err << "\n";
    std::cout << "{\"event\":\"driver_load\",\"status\":\"failed\",\"detail\":";
    json_string(std::cout, err);
    std::cout << "}\n";
    return 4;
  }

  CUresult r = drv.cuInit_(0);
  if (r != CUDA_SUCCESS) {
    std::cerr << "cuInit failed: " << drv.errstr(r) << "\n";
    return 4;
  }
  CUdevice device = 0;
  r = drv.cuDeviceGet_(&device, 0);
  if (r != CUDA_SUCCESS) {
    std::cerr << "cuDeviceGet failed: " << drv.errstr(r) << "\n";
    return 4;
  }
  CUcontext ctx = nullptr;
  r = drv.cuCtxCreate_(&ctx, 0, device);
  if (r != CUDA_SUCCESS) {
    std::cerr << "cuCtxCreate failed: " << drv.errstr(r) << "\n";
    return 4;
  }

  std::cerr << "=== h2d_bandwidth_probe_win: native Windows CUDA Driver API ===\n";
  std::cerr << "iterations_per_condition=" << kIterations
            << " warmup_discarded=" << kWarmupCount << "\n";
  std::cerr << std::left << std::setw(10) << "alloc" << std::setw(10) << "size_gib"
            << std::setw(12) << "mean_gib_s" << std::setw(12) << "median_gib_s"
            << std::setw(12) << "min_gib_s" << std::setw(12) << "max_gib_s"
            << std::setw(12) << "stddev" << "\n";

  bool any_failure = false;

  for (std::uint64_t bytes : sizes_bytes) {
    const double size_gib = static_cast<double>(bytes) / static_cast<double>(kGiB);

    // Device buffer, sized for this condition.
    CUdeviceptr device_ptr = 0;
    r = drv.cuMemAlloc_(&device_ptr, static_cast<std::size_t>(bytes));
    if (r != CUDA_SUCCESS) {
      std::cerr << "cuMemAlloc failed for " << size_gib << " GiB: " << drv.errstr(r) << "\n";
      any_failure = true;
      continue;
    }

    // PINNED host buffer via cuMemHostAlloc.
    void* pinned_ptr = nullptr;
    r = drv.cuMemHostAlloc_(&pinned_ptr, static_cast<std::size_t>(bytes), 0);
    if (r != CUDA_SUCCESS) {
      std::cerr << "cuMemHostAlloc failed for " << size_gib << " GiB: " << drv.errstr(r) << "\n";
      drv.cuMemFree_(device_ptr);
      any_failure = true;
      continue;
    }
    std::memset(pinned_ptr, 0xA5, static_cast<std::size_t>(bytes));

    // PAGEABLE host buffer via plain malloc (ordinary, non-pinned memory).
    void* pageable_ptr = std::malloc(static_cast<std::size_t>(bytes));
    if (!pageable_ptr) {
      std::cerr << "malloc failed for " << size_gib << " GiB pageable buffer\n";
      drv.cuMemFreeHost_(pinned_ptr);
      drv.cuMemFree_(device_ptr);
      any_failure = true;
      continue;
    }
    std::memset(pageable_ptr, 0x5A, static_cast<std::size_t>(bytes));

    struct Cond { const char* label; const void* src; };
    Cond conditions[] = {
        {"pinned", pinned_ptr},
        {"pageable", pageable_ptr},
    };

    for (const Cond& c : conditions) {
      std::vector<double> raw;
      std::string cond_err;
      const bool ok = run_condition(drv, c.label, c.src, device_ptr, bytes, &raw, &cond_err);
      if (!ok) {
        std::cerr << "condition failed: " << cond_err << "\n";
        std::cout << "{\"event\":\"condition\",\"alloc\":";
        json_string(std::cout, c.label);
        std::cout << ",\"size_gib\":" << size_gib << ",\"status\":\"failed\",\"detail\":";
        json_string(std::cout, cond_err);
        std::cout << "}\n";
        any_failure = true;
        continue;
      }
      std::vector<double> measured(raw.begin() + kWarmupCount, raw.end());
      const Stats stats = compute_stats(measured);

      std::cerr << std::left << std::setw(10) << c.label << std::setw(10) << size_gib
                << std::setw(12) << std::fixed << std::setprecision(3) << stats.mean_gib_s
                << std::setw(12) << stats.median_gib_s
                << std::setw(12) << stats.min_gib_s
                << std::setw(12) << stats.max_gib_s
                << std::setw(12) << stats.stddev_gib_s << "\n";

      std::cout << "{\"event\":\"condition\",\"alloc\":";
      json_string(std::cout, c.label);
      std::cout << ",\"size_gib\":" << size_gib
                << ",\"iterations\":" << kIterations
                << ",\"warmup_discarded\":" << kWarmupCount
                << ",\"mean_gib_s\":" << std::fixed << std::setprecision(6) << stats.mean_gib_s
                << ",\"median_gib_s\":" << stats.median_gib_s
                << ",\"min_gib_s\":" << stats.min_gib_s
                << ",\"max_gib_s\":" << stats.max_gib_s
                << ",\"stddev_gib_s\":" << stats.stddev_gib_s
                << ",\"raw_gib_s\":[";
      for (std::size_t i = 0; i < raw.size(); ++i) {
        if (i != 0) std::cout << ",";
        std::cout << std::fixed << std::setprecision(6) << raw[i];
      }
      std::cout << "],\"status\":\"ok\"}\n";
    }

    drv.cuMemFreeHost_(pinned_ptr);
    std::free(pageable_ptr);
    drv.cuMemFree_(device_ptr);
  }

  drv.cuCtxDestroy_(ctx);
  return any_failure ? 5 : 0;
}
