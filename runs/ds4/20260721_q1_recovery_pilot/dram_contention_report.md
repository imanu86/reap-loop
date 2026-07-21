# Two-Lane DRAM Contention Bench: G132 Investigation 2

CPU-only native MSVC/OpenMP bench. Lane A runs the working DS4 IQ2 expert forward on a ring of distinct layer-3 experts resident in RAM. Lane B uses low-priority Windows background threads, direct unbuffered reads of 7,077,888-byte chunks from the GGUF, then memcpy into preallocated RAM buffers.

MB/token pacing is converted to MB/s with 1.650 token/s. Lane-B DRAM estimate counts disk DMA write plus memcpy read plus memcpy write, so `lane_b_dram ~= 3 * payload_MB_s`.

## Baseline

| Lane | Result |
|---|---:|
| A alone, 6 OpenMP threads | 0.988 ms/expert |
| A alone compressed weight read | 7.163 GB/s |
| A forwards timed | 4059 |

## Lane B Alone

| Source | Pace | Target MB/s | Achieved MB/s | Chunks | Errors |
|---|---:|---:|---:|---:|---:|
| C: | 56 MB/token | 92.4 | 92.1 | 34 | 0 |
| C: | 112 MB/token | 184.8 | 184.7 | 106 | 0 |
| C: | 224 MB/token | 369.6 | 368.4 | 210 | 0 |
| C: | unthrottled | 0.0 | 1429.7 | 812 | 0 |
| D: | 56 MB/token | 92.4 | 92.1 | 52 | 0 |
| D: | 112 MB/token | 184.8 | 184.6 | 106 | 0 |
| D: | 224 MB/token | 369.6 | 369.0 | 210 | 0 |
| D: | unthrottled | 0.0 | 504.6 | 288 | 0 |

## Lane B Unthrottled Worker Sweep

| Source | Workers | Achieved MB/s | Chunks | Errors |
|---|---:|---:|---:|---:|
| C: | 1 | 927.6 | 526 | 0 |
| C: | 2 | 1105.1 | 628 | 0 |
| D: | 1 | 527.7 | 299 | 0 |
| D: | 2 | 501.0 | 285 | 0 |

## Concurrent Matrix

| Source | Pace | Target MB/s | A ms/expert | A degradation | B achieved MB/s | Combined DRAM est GB/s | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C: | 56 MB/token | 92.4 | 1.087 | 10.0% | 92.3 | 6.789 | yes |
| C: | 112 MB/token | 184.8 | 1.101 | 11.5% | 184.8 | 6.981 | yes |
| C: | 224 MB/token | 369.6 | 1.096 | 10.9% | 369.0 | 7.568 | yes |
| C: | unthrottled | 0.0 | 1.022 | 3.4% | 1290.0 | 10.794 | yes |
| D: | 56 MB/token | 92.4 | 1.111 | 12.4% | 92.3 | 6.647 | yes |
| D: | 112 MB/token | 184.8 | 1.058 | 7.1% | 184.5 | 7.241 | yes |
| D: | 224 MB/token | 369.6 | 1.041 | 5.4% | 368.9 | 7.905 | yes |
| D: | unthrottled | 0.0 | 1.058 | 7.1% | 503.2 | 8.198 | yes |

## Verdict

PASS criterion for 56-112 MB/token pacing: A degradation <15% and B sustains at least 95% of target pace. Result: **PASS**.

The `unthrottled` rows intentionally stress the disk and memory subsystem beyond the live-mask pacing target; they are diagnostic, not part of the pass criterion.
