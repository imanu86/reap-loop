# S1 onset detector tuning — summary (auto-generated)

Slow-erosion regime only. delay_p50/p90 = synthetic ramps; FA_real = K91 healthy plateau + K0 router-libero; FA_pod = aggressive pod r1 (adversarial); lead = 2476 - first collapse-region fire (real K91).

| config | family | d_p50 | d_p90 | miss | FA_real | FA_pod | K91 first fire | K91 lead |
|---|---|---|---|---|---|---|---|---|
| cusum_raw_k1_h16 | cusum_raw | 76 | 147 | 0.050 | 0.363 | 0.000 | 2277 | 199 |
| cusum_raw_k1_h24 | cusum_raw | 88 | 162 | 0.075 | 0.363 | 0.000 | 2283 | 193 |
| vote_16of40_k0.5_h12 | vote_kofN | 98 | 224 | 0.000 | 0.363 | 0.000 | 2276 | 200 |
| cusum_ewma50_k1_h16 | cusum_ewma50 | 58 | 102 | 0.000 | 0.726 | 0.000 | 2270 | 206 |
| cusum_ewma30_fast_k1_h24 | cusum_ewma30_fast | 65 | 136 | 0.025 | 0.726 | 0.000 | 2264 | 212 |
| cusum_raw_k1_h12 | cusum_raw | 70 | 140 | 0.025 | 0.726 | 0.000 | 2274 | 202 |
| cusum_ewma50_k1_h24 | cusum_ewma50 | 71 | 121 | 0.013 | 0.726 | 0.000 | 2275 | 201 |
| cusum_raw_k1_h6 | cusum_raw | 48 | 88 | 0.000 | 1.089 | 0.000 | 2266 | 210 |
| cusum_ewma50_k1_h12 | cusum_ewma50 | 50 | 94 | 0.000 | 1.089 | 0.000 | 2266 | 210 |
| cusum_raw_k1_h8 | cusum_raw | 56 | 98 | 0.013 | 1.089 | 0.000 | 2269 | 207 |
| cusum_ewma30_k1_h24 | cusum_ewma30 | 60 | 102 | 0.000 | 1.089 | 0.000 | 2271 | 205 |
| vote_12of40_k0.5_h12 | vote_kofN | 82 | 156 | 0.000 | 1.089 | 0.408 | 2269 | 207 |
| cusum_raw_k1_h4 | cusum_raw | 40 | 75 | 0.000 | 1.451 | 0.000 | 2262 | 214 |
| cusum_ewma30_k1_h16 | cusum_ewma30 | 51 | 91 | 0.000 | 1.451 | 0.000 | 2266 | 210 |
| cusum_ewma30_fast_k1_h16 | cusum_ewma30_fast | 55 | 99 | 0.013 | 1.451 | 0.000 | 2257 | 219 |
| cusum_raw_k0.5_h24 | cusum_raw | 62 | 99 | 0.000 | 1.451 | 0.000 | 2272 | 204 |
| vote_16of40_k0.5_h8 | vote_kofN | 69 | 118 | 0.000 | 1.452 | 0.408 | 2270 | 206 |
| cusum_ewma50_k1_h8 | cusum_ewma50 | 40 | 80 | 0.000 | 1.814 | 0.000 | 2262 | 214 |
| cusum_ewma30_fast_k1_h12 | cusum_ewma30_fast | 45 | 85 | 0.000 | 1.814 | 0.000 | 2253 | 223 |
| cusum_raw_k0.5_h16 | cusum_raw | 49 | 85 | 0.000 | 1.814 | 0.000 | 2266 | 210 |
| cusum_ewma50_k0.5_h24 | cusum_ewma50 | 50 | 84 | 0.000 | 1.814 | 0.000 | 2267 | 209 |
| cusum_ewma30_k1_h12 | cusum_ewma30 | 42 | 81 | 0.000 | 2.177 | 0.000 | 2263 | 213 |
| cusum_ewma15_k1_h24 | cusum_ewma15 | 54 | 91 | 0.000 | 2.177 | 0.000 | 2269 | 207 |
| slope_base_a10_w64_s16_initx0_thr0.0003 | slope | 78 | 155 | 0.062 | 2.177 | 0.000 | 2281 | 195 |
| cusum_raw_k1_h3 | cusum_raw | 34 | 62 | 0.000 | 2.540 | 0.408 | 2259 | 217 |

## Recommended profiles

{
  "note": "Detector scoped to the SLOW-EROSION regime only (static/wide masks, K91-family). No usable S1 lead in the aggressive W50+K23+rotate32 regime (S1 pinned flat ~0.815); the pod r1 series is used only as an adversarial false-alarm control. delay_p50/p90 are synthetic ramps calibrated on the real K91 noise; real_k91_lead is the single measured lead vs the ground-truth text lock (pos 2476).",
  "recommended_detector": "aggregate EWMA-CUSUM (alpha=0.30, self-calibrated sigma over first 128 tok, lagged-window baseline base_lag=32 base_win=128); two thresholds on the normalized CUSUM g/sigma.",
  "profile_arm_relearn_admit": {
    "name": "cusum_ewma50_k0.5_h4",
    "family": "cusum_ewma50",
    "params": {
      "k_sigma": 0.5,
      "h_sigma": 4,
      "pre_alpha": 0.5,
      "base_lag": 32,
      "base_win": 128,
      "sigma_cal": 0.01407
    },
    "delay_p50": 31,
    "delay_p90": 55,
    "miss_rate": 0.0,
    "fa_per_1k_synth": 9.476824960356774,
    "fa_per_1k_real": 7.982583454281568,
    "fa_per_1k_pod": 2.857142857142857,
    "real_k91_first_fire": 2251,
    "real_k91_lead": 225
  },
  "profile_arm_rewind": {
    "name": "cusum_ewma50_k1_h8",
    "family": "cusum_ewma50",
    "params": {
      "k_sigma": 1.0,
      "h_sigma": 8,
      "pre_alpha": 0.5,
      "base_lag": 32,
      "base_win": 128,
      "sigma_cal": 0.01407
    },
    "delay_p50": 40,
    "delay_p90": 80,
    "miss_rate": 0.0,
    "fa_per_1k_synth": 1.8196890353626236,
    "fa_per_1k_real": 1.8142235123367199,
    "fa_per_1k_pod": 0.0,
    "real_k91_first_fire": 2262,
    "real_k91_lead": 214
  },
  "delay_floor_any_family_admit_cap": {
    "name": "vote_8of40_k0.5_h6",
    "family": "vote_kofN",
    "params": {
      "K": 8,
      "k_sigma": 0.5,
      "h_sigma": 6
    },
    "delay_p50": 30,
    "delay_p90": 58,
    "miss_rate": 0.0,
    "fa_per_1k_synth": 7.848651222306536,
    "fa_per_1k_real": 6.533575317604356,
    "fa_per_1k_pod": 5.3061224489795915,
    "real_k91_first_fire": 2306,
    "real_k91_lead": 170
  },
  "delay_floor_any_family_rewind_cap": {
    "name": "cusum_raw_k1_h4",
    "family": "cusum_raw",
    "params": {
      "k_sigma": 1.0,
      "h_sigma": 4,
      "pre_alpha": null,
      "base_lag": 32,
      "base_win": 128,
      "sigma_cal": 0.0174
    },
    "delay_p50": 40,
    "delay_p90": 75,
    "miss_rate": 0.0,
    "fa_per_1k_synth": 3.018373082466574,
    "fa_per_1k_real": 1.4513788098693758,
    "fa_per_1k_pod": 0.0,
    "real_k91_first_fire": 2262,
    "real_k91_lead": 214
  },
  "baseline_0020_default_thr3e-4": {
    "name": "slope_base_a10_w64_s16_init0_thr0.0003",
    "family": "slope",
    "params": {
      "alpha": 0.1,
      "win": 64,
      "stable": 16,
      "thr": 0.0003,
      "ema_init_first": false
    },
    "delay_p50": 78,
    "delay_p90": 155,
    "miss_rate": 0.0625,
    "fa_per_1k_synth": 0.5863442447279565,
    "fa_per_1k_real": 2.9027576197387517,
    "fa_per_1k_pod": 0.40816326530612246,
    "real_k91_first_fire": 2281,
    "real_k91_lead": 195
  }
}
