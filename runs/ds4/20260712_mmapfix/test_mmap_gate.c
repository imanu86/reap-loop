/* Standalone harness to exercise ds4_gpu_set_model_map() directly (the
 * exact function patched for DS4_CUDA_NO_WHOLE_MMAP_REGISTER), without
 * going through the full ds4.c engine / ssd-streaming pipeline. Deliberately
 * run against a SMALL file (the 3.8 GiB MTP model, not the 81 GiB main
 * model) so a misbehaving whole-mmap cudaHostRegister can't wedge the box.
 *
 * Usage: test_mmap_gate <path-to-gguf>
 * Exercises:
 *   1. ds4_gpu_init()
 *   2. ds4_gpu_set_model_map(map, size)      <- the patched function
 *   3. ds4_gpu_set_model_fd_for_map(fd, map) <- as ds4.c always does
 *   4. ds4_gpu_cache_model_range() at a few offsets (head/mid/tail),
 *      exercising the same cuda_model_range_ptr() fallback chain that
 *      real weight access uses.
 * Prints PASS/FAIL and returns the appropriate exit code so the caller can
 * compare behavior across DS4_CUDA_NO_WHOLE_MMAP_REGISTER on/off.
 */
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to-gguf>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 2; }
    uint64_t size = (uint64_t)st.st_size;
    fprintf(stderr, "[harness] file=%s size=%.3f GiB\n", path, (double)size / 1073741824.0);

    void *map = mmap(NULL, (size_t)size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 2; }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "[harness] FAIL: ds4_gpu_init failed\n");
        return 1;
    }

    int ok = ds4_gpu_set_model_map(map, size);
    fprintf(stderr, "[harness] ds4_gpu_set_model_map -> %d\n", ok);
    if (!ok) {
        fprintf(stderr, "[harness] FAIL: ds4_gpu_set_model_map returned 0\n");
        return 1;
    }

    if (!ds4_gpu_set_model_fd_for_map(fd, map)) {
        fprintf(stderr, "[harness] WARN: ds4_gpu_set_model_fd_for_map returned 0\n");
    }

    /* Exercise the same range-cache pointer path real weight access uses,
     * at head / mid / tail offsets, in 4 MiB chunks. */
    const uint64_t chunk = 4ull * 1024 * 1024;
    uint64_t offs[3];
    offs[0] = 0;
    offs[1] = (size / 2) & ~(chunk - 1);
    offs[2] = size > chunk ? size - chunk : 0;
    const char *labels[3] = {"head", "mid", "tail"};
    int all_ok = 1;
    for (int i = 0; i < 3; i++) {
        uint64_t bytes = (offs[i] + chunk <= size) ? chunk : (size - offs[i]);
        if (bytes == 0) continue;
        int cok = ds4_gpu_cache_model_range(map, size, offs[i], bytes, labels[i]);
        fprintf(stderr, "[harness] ds4_gpu_cache_model_range(%s off=%llu bytes=%llu) -> %d\n",
                labels[i], (unsigned long long)offs[i], (unsigned long long)bytes, cok);
        if (!cok) all_ok = 0;
    }

    ds4_gpu_synchronize();
    ds4_gpu_cleanup();
    munmap(map, (size_t)size);
    close(fd);

    if (!all_ok) {
        fprintf(stderr, "[harness] FAIL: one or more cache_model_range calls failed\n");
        return 1;
    }
    fprintf(stderr, "[harness] PASS\n");
    return 0;
}
