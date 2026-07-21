# Working-Set Temporal Coherence Analysis

Parsed three 64-token DS4 decode replays, 16,512 route rows each. Expert identity is `(layer, expert)`.

## Method Notes

- Novelty is `1 - overlap` against the union of prior routed experts in the window; token 0 has an empty prior.
- Warm-set coverage uses the union through token `t` to score token `t+1`.
- Heat persistence uses only tokens with a full future horizon.
- Promotion hits are scored before observing a token. After each token, candidates with at least two touches in the last 8 tokens are ranked by recent touch count, mass, and recency; cooled entries are demoted; up to `P` missing hot entries are promoted.
- Promotion steady state is reported for tokens 16-63. Static top-K uses the same median steady-state warm size as the adaptive run, ranked by gate mass in the first 16 tokens.

## 1. Per-Token Novelty

### replay1_cyberpunk_html

| Window | Count median | Count p90 | Mass median | Mass p90 |
|---:|---:|---:|---:|---:|
| 1 | 54.8% | 81.4% | 47.7% | 78.3% |
| 4 | 40.7% | 61.2% | 34.5% | 58.2% |
| 8 | 32.8% | 52.3% | 27.8% | 44.6% |
| 16 | 26.0% | 40.7% | 20.9% | 41.0% |
| 32 | 21.9% | 36.4% | 17.8% | 32.8% |
| ALL_PRIOR | 19.6% | 36.4% | 15.7% | 32.8% |

### replay2_python_code

| Window | Count median | Count p90 | Mass median | Mass p90 |
|---:|---:|---:|---:|---:|
| 1 | 53.1% | 65.1% | 42.6% | 57.4% |
| 4 | 33.7% | 45.3% | 23.6% | 40.5% |
| 8 | 25.8% | 40.3% | 18.4% | 35.7% |
| 16 | 20.3% | 35.3% | 14.6% | 28.7% |
| 32 | 16.5% | 35.3% | 10.9% | 28.7% |
| ALL_PRIOR | 13.2% | 35.3% | 8.9% | 28.7% |

### replay3_history_essay

| Window | Count median | Count p90 | Mass median | Mass p90 |
|---:|---:|---:|---:|---:|
| 1 | 55.0% | 72.1% | 49.4% | 68.3% |
| 4 | 42.6% | 58.5% | 38.5% | 55.2% |
| 8 | 35.1% | 52.3% | 30.6% | 50.2% |
| 16 | 26.9% | 44.6% | 23.4% | 42.1% |
| 32 | 21.9% | 41.1% | 17.9% | 36.2% |
| ALL_PRIOR | 18.6% | 41.1% | 14.7% | 36.2% |

### pooled

| Window | Count median | Count p90 | Mass median | Mass p90 |
|---:|---:|---:|---:|---:|
| 1 | 54.7% | 72.9% | 45.5% | 68.6% |
| 4 | 37.4% | 55.8% | 30.9% | 51.8% |
| 8 | 31.0% | 47.3% | 24.8% | 44.5% |
| 16 | 24.2% | 41.9% | 19.1% | 37.5% |
| 32 | 20.0% | 37.6% | 15.9% | 33.9% |
| ALL_PRIOR | 18.2% | 37.6% | 13.6% | 33.9% |

## 2. Warm-Set Size and Next-Token Coverage

### replay1_cyberpunk_html

| Window | Size median | Size max | Next mass coverage median | Next mass coverage p10 |
|---:|---:|---:|---:|---:|
| 4 | 628 | 808 | 65.5% | 42.2% |
| 8 | 1027 | 1302 | 72.4% | 55.5% |
| 16 | 1555 | 1752 | 79.4% | 66.4% |
| 32 | 2258 | 2683 | 82.4% | 68.0% |
| 64 | 2310 | 3739 | 84.5% | 69.2% |

### replay2_python_code

| Window | Size median | Size max | Next mass coverage median | Next mass coverage p10 |
|---:|---:|---:|---:|---:|
| 4 | 592 | 701 | 76.5% | 60.7% |
| 8 | 873 | 1081 | 81.7% | 64.4% |
| 16 | 1302 | 1580 | 85.5% | 73.0% |
| 32 | 1888 | 2202 | 89.1% | 74.0% |
| 64 | 2220 | 2992 | 91.1% | 74.0% |

### replay3_history_essay

| Window | Size median | Size max | Next mass coverage median | Next mass coverage p10 |
|---:|---:|---:|---:|---:|
| 4 | 639 | 751 | 61.9% | 46.6% |
| 8 | 1031 | 1204 | 69.4% | 50.5% |
| 16 | 1610 | 1817 | 76.8% | 58.3% |
| 32 | 2422 | 2741 | 82.2% | 63.9% |
| 64 | 2431 | 3789 | 86.4% | 63.9% |

### pooled

| Window | Size median | Size max | Next mass coverage median | Next mass coverage p10 |
|---:|---:|---:|---:|---:|
| 4 | 620 | 808 | 69.2% | 48.7% |
| 8 | 978 | 1302 | 75.3% | 56.3% |
| 16 | 1507 | 1817 | 81.1% | 63.0% |
| 32 | 2018 | 2741 | 84.5% | 66.1% |
| 64 | 2308 | 3789 | 86.4% | 66.4% |

## 3. Heat Persistence

| Replay | Horizon | Count probability | Mass probability |
|---|---:|---:|---:|
| replay1_cyberpunk_html | 1 | 41.8% | 49.1% |
| replay1_cyberpunk_html | 4 | 60.4% | 66.5% |
| replay1_cyberpunk_html | 16 | 76.9% | 80.8% |
| replay2_python_code | 1 | 45.9% | 56.8% |
| replay2_python_code | 4 | 68.5% | 76.5% |
| replay2_python_code | 16 | 80.7% | 85.9% |
| replay3_history_essay | 1 | 44.2% | 51.6% |
| replay3_history_essay | 4 | 58.6% | 64.4% |
| replay3_history_essay | 16 | 74.5% | 78.3% |
| pooled | 1 | 44.0% | 52.5% |
| pooled | 4 | 62.5% | 69.1% |
| pooled | 16 | 77.4% | 81.7% |

## 4. Promotion Feasibility

| Replay | P/token | Adaptive warm median | Adaptive hit mean | Adaptive hit p10 | Static K | Static hit mean | Adaptive - static |
|---|---:|---:|---:|---:|---:|---:|---:|
| replay1_cyberpunk_html | 2 | 49 | 18.2% | 11.7% | 49 | 13.8% | 4.4% |
| replay1_cyberpunk_html | 8 | 159 | 37.4% | 24.8% | 159 | 27.9% | 9.5% |
| replay1_cyberpunk_html | 32 | 364 | 55.5% | 36.4% | 364 | 39.2% | 16.4% |
| replay2_python_code | 2 | 69 | 36.5% | 20.8% | 69 | 21.0% | 15.5% |
| replay2_python_code | 8 | 210 | 60.5% | 42.8% | 210 | 38.8% | 21.7% |
| replay2_python_code | 32 | 382 | 73.8% | 63.9% | 382 | 49.2% | 24.6% |
| replay3_history_essay | 2 | 47 | 16.8% | 10.0% | 47 | 11.1% | 5.7% |
| replay3_history_essay | 8 | 135 | 30.4% | 18.5% | 135 | 21.6% | 8.8% |
| replay3_history_essay | 32 | 343 | 47.0% | 29.5% | 343 | 33.1% | 13.9% |
| pooled | 2 | 51 | 23.8% | 11.7% | 49/69/47 | 15.3% | 8.5% |
| pooled | 8 | 153 | 42.8% | 22.4% | 159/210/135 | 29.5% | 13.3% |
| pooled | 32 | 365 | 58.8% | 36.4% | 364/382/343 | 40.5% | 18.3% |

## 5. Cross-Domain Switch: Replay1 Warm Set -> Replay2

| P/token | Initial warm size | Replay2 in-domain steady hit | First token >=90% of in-domain | First trailing-4 >=90% of in-domain | First token >= absolute 90% |
|---:|---:|---:|---:|---:|---:|
| 2 | 71 | 36.5% | 31 | 32 | not reached |
| 8 | 178 | 60.5% | 27 | 28 | not reached |
| 32 | 350 | 73.8% | 15 | 17 | not reached |

## Verdict

The traces show strong temporal coherence rather than fresh-expert churn. A last-8-token union covers 75.3% median next-token gate mass, and routed heat persists: 69.1% of gate mass reappears within 4 tokens and 81.7% within 16 tokens.

The adaptive hot-set tracker beats the first-16-token static baseline for every tested promotion rate in pooled steady state. `P=8` reaches 42.8% mean gate-mass hit with a median warm size of 153 experts; `P=32` lifts that to 58.8% while increasing promotion bandwidth.

For this reap-loop design, the working set evolves slowly enough to chase. `P=8` promotions/token is the practical sufficiency point in these traces; `P=2` works but lags, while `P=32` is mostly headroom for abrupt switches.
