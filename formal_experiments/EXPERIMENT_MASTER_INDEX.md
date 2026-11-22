# Stage 5 Experiment Master Index (`CLWB-Dual`)

> **Verification Status**: **Gate 1–6 ALL PASSED** | Upstream (`stage2_model/`, `stage3_algorithm/`, `stage4_audit/`, `复现材料/`): **0 bytes modified** (`SHA-256 verified`) | Policy: single frozen `make_clwb_dual()` across all workloads, seeds, traces, and hardware profiles.

## 1. Automated Gate Summary (Gate 1–6)

| Gate | Check Scope | Verified Result | Status |
| :--- | :--- | :--- | :---: |
| **Gate 1** | Parity vs. Stage-3 Reference | `max_parity_err = 0.00e+00`, Micro-100 Capture = `0.9875`, CEs L1=`2/4`, L2=`3/3`, L3=`3/4`, 2ndOrder=`4/5` | PASS |
| **Gate 2** | Physical Fidelity & Home Sanity | Bit rel err `< 1e-8` (`8.53e-14`), Wallclock `R^2 = 0.9878` (`rho = 1.0`), Azure MAPE mean `1.179%` / max `3.966%`, `illegal_actions = 0` | PASS |
| **Gate 3** | Upstream Read-Only Integrity | Stage 2 (`8`), Stage 3 (`6`), Stage 4 (`5`) SHA-256 unchanged | PASS |
| **Gate 4** | Anti-Pooling Statistical Integrity | `Main_K12` ($K=12$) `max holm_p = 0.00342 < 0.01`, `All_24` ($K=24$) `max holm_p = 0.000108 < 0.001`; Azure ($N=64$) exact match | PASS |
| **Gate 5** | Data Hygiene & Publication Assets | `19` CSVs (0 NaN/Inf), `7` JSONs, `8` PDF+PNG pairs (TrueType font embedded, 300 DPI) | PASS |
| **Gate 6** | Full Transparency Master Index | All 24 workloads, neutral ties, ablation trade-offs, Azure 64-trace, and FAIL-1..3 boundaries indexed | PASS |

## 2. Primary Benchmark Summary (Main K=12 & Combined K=24, Anti-Pooled Wilcoxon)

- **Main K=12** (`CLWB_Dual`): On-Time Rate = **97.94%** (95% CI: `[97.05%, 98.75%]`), Worst Workload = `94.67%`, Grant Miss = `0.40%`, Max P99 = `14.58 ms`, Skipped Ticks = `0`.
- **Combined K=24** (`CLWB_Dual`): On-Time Rate = **96.89%** (95% CI: `[95.01%, 98.30%]`), Grant Miss = `0.40%`.

| Opponent (+CG) | Main K=12 Gain (pp) | K=12 Holm p | K=12 W/T/L | All K=24 Gain (pp) | All K=24 95% CI (pp) | All K=24 Holm p | All K=24 Cliff's delta | All K=24 W/T/L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bambos_ALP+CG` | `+8.11` | `0.00342` | `12/0/0` | `+8.76` | `[6.40, 11.17]` | `1.08e-04` | `0.632` | `23/1/0` |
| `Chen_Game+CG` | `+27.01` | `0.00342` | `12/0/0` | `+25.67` | `[22.09, 28.97]` | `8.34e-07` | `0.934` | `24/0/0` |
| `EDF_PreBatch+CG` | `+63.53` | `0.00342` | `12/0/0` | `+59.88` | `[53.51, 64.18]` | `1.08e-04` | `0.920` | `23/1/0` |
| `LLF_PreBatch+CG` | `+63.65` | `0.00342` | `12/0/0` | `+59.94` | `[53.51, 64.31]` | `1.08e-04` | `0.920` | `23/1/0` |
| `DGOLB+CG` | `+75.04` | `0.00342` | `12/0/0` | `+70.84` | `[63.51, 75.63]` | `8.34e-07` | `0.941` | `24/0/0` |
| `SFA+CG` | `+81.32` | `0.00342` | `12/0/0` | `+75.34` | `[66.75, 81.77]` | `1.08e-04` | `0.920` | `23/1/0` |
| `IBR_5+CG` | `+92.87` | `0.00342` | `12/0/0` | `+87.70` | `[79.75, 92.88]` | `9.10e-05` | `0.997` | `24/0/0` |

## 3. Complete 24-Workload Transparent Breakdown (10 Seeds / Workload)

> **Honest Regime Notes**: `BL08_SPARSE_RELAXED` is an intentionally light neutral tie regime where 7 of 8 methods achieve `100.00%`. `BL04_SUBBAND_REUSE4` is our narrowest positive-margin blind subband regime (`CLWB_Dual = 99.72%` vs. `ALP+CG = 99.22%`, `+0.50 pp`). `BL06_MICRO_EDGE_CELL` (`kappa_UL = 14.582`) is our hardest absolute-rate blind regime (`CLWB_Dual = 80.12%` vs. `ALP+CG = 63.12%`, `+17.00 pp`).

| Suite | Workload | kappa_UL | rho_cmp | CLWB-Dual (%) | ALP+CG (%) | Chen+CG (%) | EDF+CG (%) | DGOLB+CG (%) | SFA+CG (%) | Margin vs Best Base (pp) | CLWB Miss (%) | CLWB P99 Max (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Blind_K8` | `BL01_MIXED_AR_CTRL` | 1.568 | 0.721 | **95.64** | 81.07 | 58.43 | 26.71 | 20.50 | 25.21 | `+14.57` | 0.29 | 6.46 |
| `Blind_K8` | `BL02_NEAR_FAR_HEAVY` | 3.369 | 0.776 | **89.47** | 70.40 | 55.13 | 29.13 | 23.07 | 28.67 | `+19.07` | 0.20 | 5.90 |
| `Blind_K8` | `BL03_CSI_NOISY_2DB` | 0.343 | 0.702 | **97.01** | 84.38 | 64.86 | 29.44 | 19.93 | 20.90 | `+12.64` | 1.70 | 7.06 |
| `Blind_K8` | `BL04_SUBBAND_REUSE4` *(narrowest positive margin)* | 0.074 | 0.753 | **99.72** | 99.22 | 84.78 | 58.56 | 41.50 | 15.94 | `+0.50` | 0.16 | 12.27 |
| `Blind_K8` | `BL05_ULTRA_BURST85` | 1.595 | 0.839 | **99.50** | 98.20 | 90.70 | 41.60 | 20.80 | 6.10 | `+1.30` | 0.00 | 11.43 |
| `Blind_K8` | `BL06_MICRO_EDGE_CELL` *(hardest micro-cell)* | 14.582 | 0.810 | **80.12** | 63.12 | 54.69 | 27.00 | 21.00 | 20.31 | `+17.00` | 0.00 | 4.91 |
| `Blind_K8` | `BL07_TICK_FAST25MS` | 0.444 | 0.706 | **99.93** | 94.38 | 67.15 | 29.79 | 21.81 | 17.57 | `+5.56` | 0.00 | 4.86 |
| `Blind_K8` | `BL08_SPARSE_RELAXED` *(neutral tie)* | 0.039 | 0.349 | **100.00** | 100.00 | 98.56 | 100.00 | 99.22 | 100.00 | `+0.00` | 0.00 | 1.00 |
| `Main_K12` | `WK01_BALANCED_MED` | 0.236 | 0.604 | **100.00** | 98.89 | 80.56 | 40.35 | 32.71 | 26.94 | `+1.11` | 0.00 | 6.51 |
| `Main_K12` | `WK02_DENSE_TIGHT` | 3.421 | 0.735 | **94.67** | 76.53 | 67.13 | 28.60 | 19.00 | 15.87 | `+18.13` | 0.00 | 6.69 |
| `Main_K12` | `WK03_DEADLINE_TIGHT` | 0.426 | 0.702 | **96.39** | 82.29 | 58.54 | 25.97 | 17.08 | 15.00 | `+14.10` | 0.36 | 7.06 |
| `Main_K12` | `WK04_HETERO_DEV` | 0.671 | 0.710 | **97.71** | 89.17 | 62.36 | 34.24 | 25.69 | 18.19 | `+8.54` | 0.10 | 7.32 |
| `Main_K12` | `WK05_HETERO_DL` | 0.992 | 0.700 | **98.93** | 94.50 | 71.29 | 34.21 | 26.64 | 31.29 | `+4.43` | 0.79 | 7.00 |
| `Main_K12` | `WK06_BURST_MODERATE` | 1.189 | 0.854 | **98.31** | 88.06 | 71.75 | 33.12 | 18.31 | 8.38 | `+10.25` | 0.00 | 9.45 |
| `Main_K12` | `WK07_BURST_HEAVY` | 3.076 | 0.901 | **97.39** | 86.00 | 72.06 | 35.52 | 18.06 | 4.67 | `+11.39` | 0.18 | 9.64 |
| `Main_K12` | `WK08_HIGH_PAYLOAD` | 0.770 | 0.641 | **99.33** | 92.58 | 75.67 | 36.92 | 27.58 | 21.75 | `+6.75` | 0.00 | 5.65 |
| `Main_K12` | `WK09_SMALL_CELL_ASYM` | 3.094 | 0.711 | **97.19** | 85.78 | 70.47 | 31.09 | 20.70 | 25.23 | `+11.41` | 0.43 | 5.88 |
| `Main_K12` | `WK10_DUAL_SUBBAND` | 0.297 | 0.757 | **99.17** | 97.33 | 75.72 | 38.17 | 25.44 | 13.50 | `+1.83` | 0.30 | 10.50 |
| `Main_K12` | `WK11_CSI_NOISE_2DB` | 0.466 | 0.708 | **96.67** | 90.83 | 63.47 | 32.64 | 23.26 | 18.54 | `+5.83` | 2.63 | 6.73 |
| `Main_K12` | `WK12_LARGE_SCALE` | 1.620 | 0.811 | **99.50** | 95.94 | 82.11 | 42.06 | 20.28 | 0.00 | `+3.56` | 0.00 | 14.58 |
| `OOD_K4` | `OOD1_ASYM_NEARFAR` | 3.269 | 0.751 | **95.82** | 79.82 | 66.53 | 31.18 | 20.65 | 13.88 | `+16.00` | 0.25 | 9.21 |
| `OOD_K4` | `OOD2_HETDL_CSI2DB` | 1.102 | 0.710 | **98.07** | 96.29 | 67.57 | 33.79 | 23.00 | 27.36 | `+1.79` | 1.48 | 6.93 |
| `OOD_K4` | `OOD3_EXTREME_BURST75` | 6.123 | 0.925 | **98.93** | 89.27 | 79.47 | 37.20 | 15.67 | 5.53 | `+9.67` | 0.41 | 11.81 |
| `OOD_K4` | `OOD4_SPARSE_SRV` | 4.838 | 0.782 | **95.85** | 80.92 | 70.31 | 31.00 | 23.23 | 36.31 | `+14.92` | 0.32 | 4.73 |

## 4. Mechanism Ablation & Honest Trade-Offs (Batch 3B, 20 Workloads x 10 Seeds)

> **Honest Role of 2nd-Order Externality Pricing (`LOO_no_ext`)**: Across the 20 standard+blind workloads, removing externality pricing leaves macro average slightly higher by `+0.07 pp` (`96.89%` vs. `96.82%`, 95% CI `[-0.17, +0.03] pp`, helping 7 / hurting 9 workloads) because conservative two-flow interference pricing trades negligible slack margin in non-adversarial ticks. However, `LOO_no_ext` **drops from `4/5` to `2/5` on `CE_2ndOrder` and from `3/4` to `1/4` on `L3`**, confirming it is an indispensable analytical anti-cascade guard (`2x2` CE_2ndOrder: `S00=1/5 -> S10=1/5 -> S01=2/5 -> S11=4/5`).

| Variant | 20-Wk On-Time (%) | Drop vs Full (pp) | Main-12 Drop (pp) | Blind-8 Drop (pp) | L1 | L2 | L3 | CE_2ndOrder |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Full CLWB-Dual` | **96.82** | `0.00` | `0.00` | `0.00` | 2/4 | 3/3 | 3/4 | 4/5 |
| `LOO_no_ext` | 96.89 | `-0.07` | `-0.12` | `-0.00` | 2/4 | 3/3 | 1/4 | 2/5 |
| `LOO_no_defer` | 69.30 | `+27.52` | `+28.83` | `+25.55` | 2/4 | 1/3 | 3/4 | 1/5 |
| `LOO_no_fluid` | 37.59 | `+59.23` | `+59.75` | `+58.43` | 0/4 | 0/3 | 0/4 | 0/5 |
| `sub_no_backfill` | 90.28 | `+6.53` | `+5.98` | `+7.36` | 2/4 | 3/3 | 3/4 | 4/5 |
| `sub_no_act_half_slack` | 96.48 | `+0.33` | `+0.26` | `+0.44` | 2/4 | 3/3 | 3/4 | 4/5 |
| `sub_no_edge_shield` | 85.24 | `+11.57` | `+8.19` | `+16.64` | 2/4 | 3/3 | 3/4 | 4/5 |
| `sub_no_local_first` | 88.27 | `+8.55` | `+8.68` | `+8.36` | 2/4 | 3/3 | 3/4 | 4/5 |
| `Baseline_S00` | 69.14 | `+27.67` | `+28.83` | `+25.93` | 2/4 | 1/3 | 1/4 | 1/5 |

## 5. Real Azure Functions 2019 Trace Replay (64 Scenarios, Zero-Modification Policy)

> **Provenance Split & Coupled Operating Regime**: `REAL`: Azure 2019 per-minute invocation arrival curve (`counts`) + 7-knot execution duration inverse-CDF (`knots, vmin, vmax, avg`). `SYNTHETIC`: uniform sub-minute arrival expansion into calibrated `window_s = 0.474s` (yielding Zone-III mean coupling `kappa_UL = 1.860` and load `rho_cmp = 0.752`), mean-preserving cycle scaling, round-robin binding to physical devices and ScenarioV1 radio/compute parameters. Across all 64 trace scenarios, `CLWB_Dual` achieves `99.63%`, outperforming `Bambos_ALP+CG` (`+1.78 pp`, `holm_p = 5.18e-10`), `EDF_PreBatch+CG` (`+30.43 pp`, `holm_p = 1.75e-11`), and `DGOLB+CG` (`+31.44 pp`, `holm_p = 1.75e-11`).

| Method | All Traces N=64 (%) | 95% CI (%) | Grant Miss (%) | Max P99 (ms) | Gain by CLWB (pp) | Holm p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CLWB_Dual` | **99.63** | `[99.46, 99.78]` | 0.05 | 7.17 | — | — |
| `Bambos_ALP+CG` | 97.85 | `[97.30, 98.35]` | 0.00 | 2.86 | `+1.78` | `5.18e-10` |
| `Chen_Game+CG` | 95.28 | `[94.02, 96.47]` | 6.39 | 6.41 | `+4.35` | `8.18e-11` |
| `SFA+CG` | 57.29 | `[51.46, 63.10]` | 58.28 | 400.27 | `+42.34` | `3.59e-11` |
| `EDF_PreBatch+CG` | 69.20 | `[66.67, 71.62]` | 82.90 | 1.22 | `+30.43` | `1.75e-11` |
| `DGOLB+CG` | 68.19 | `[65.56, 70.81]` | 84.68 | 1.20 | `+31.44` | `1.75e-11` |

## 6. Honest Failure & Premise Boundary Archive (`FAIL-1..3`)

- **`FAIL-1_CSI_6DB`**: At $\sigma_{\mathrm{CSI}} = 6.0\text{ dB}$, `CLWB_Dual` drops by `-9.77 pp` (`94.15% -> 84.38%`, grant miss `6.43%`). Boundary: Recommended CSI noise <= 2-3 dB; >= 6 dB degrades deterministic fluid prediction.
- **`FAIL-2_SUBTICK_SAME_DEV`**: Sub-tick multi-burst on a single device caps both `CLWB_Dual` and `Bambos_ALP+CG` at `2/5` (`40.0%`). Boundary: Causal idle-radio gate caps per-device admissions at <= 1 per tick_s.
- **`FAIL-3_ZONE1_WEAK_AND_INVERTED_DL`**: In orthogonal weak coupling (Zone I), `CLWB_Dual` and `Bambos_ALP+CG` both reach `100.0%`, while `ALP+CG` uses `0.085 ms` vs `0.498 ms` P50; under full deadline inversion $\eta_{\mathrm{inv}}=1.0$, single-tick oracle match drops by `41.2%` (`100.0% -> 58.8%`). Boundary: Zone I orthogonal weak coupling needs no cross-layer solver (simpler ALP suffices at lower P50); inverted deadlines violate Theorem 3.

## 7. Complete Publication Artifact Inventory

### Figures (`figures/`, Vector PDF with Embedded TrueType Fonts + 300 DPI PNG)
- `fig1_motivation_dual_deadlock`: [PDF](file:///./formal_experiments/figures/fig1_motivation_dual_deadlock.pdf) | [PNG](file:///./formal_experiments/figures/fig1_motivation_dual_deadlock.png)
- `fig2_main_benchmark_comparison`: [PDF](file:///./formal_experiments/figures/fig2_main_benchmark_comparison.pdf) | [PNG](file:///./formal_experiments/figures/fig2_main_benchmark_comparison.png)
- `fig3_oracle_and_ce2`: [PDF](file:///./formal_experiments/figures/fig3_oracle_and_ce2.pdf) | [PNG](file:///./formal_experiments/figures/fig3_oracle_and_ce2.png)
- `fig4_whitebox_xray_and_grafting`: [PDF](file:///./formal_experiments/figures/fig4_whitebox_xray_and_grafting.pdf) | [PNG](file:///./formal_experiments/figures/fig4_whitebox_xray_and_grafting.png)
- `fig5_sensitivity_operating_point`: [PDF](file:///./formal_experiments/figures/fig5_sensitivity_operating_point.pdf) | [PNG](file:///./formal_experiments/figures/fig5_sensitivity_operating_point.png)
- `fig6_regime_map_2d`: [PDF](file:///./formal_experiments/figures/fig6_regime_map_2d.pdf) | [PNG](file:///./formal_experiments/figures/fig6_regime_map_2d.png)
- `fig7_azure_trace_and_robustness`: [PDF](file:///./formal_experiments/figures/fig7_azure_trace_and_robustness.pdf) | [PNG](file:///./formal_experiments/figures/fig7_azure_trace_and_robustness.png)
- `fig8_scalability_and_boundaries`: [PDF](file:///./formal_experiments/figures/fig8_scalability_and_boundaries.pdf) | [PNG](file:///./formal_experiments/figures/fig8_scalability_and_boundaries.png)

### Datasets (`data_csv/`, All 19 Finite CSVs)
- [s50_wallclock_calib.csv](file:///./formal_experiments/data_csv/s50_wallclock_calib.csv) (`65` rows x `8` cols)
- [s50_home_sanity.csv](file:///./formal_experiments/data_csv/s50_home_sanity.csv) (`48` rows x `8` cols)
- [b1a_ortho_vs_shared.csv](file:///./formal_experiments/data_csv/b1a_ortho_vs_shared.csv) (`160` rows x `14` cols)
- [b1b_kappa_sweep.csv](file:///./formal_experiments/data_csv/b1b_kappa_sweep.csv) (`400` rows x `14` cols)
- [b1c_burst_cliff.csv](file:///./formal_experiments/data_csv/b1c_burst_cliff.csv) (`300` rows x `12` cols)
- [b1d_counterexamples.csv](file:///./formal_experiments/data_csv/b1d_counterexamples.csv) (`36` rows x `7` cols)
- [b2_episodes.csv](file:///./formal_experiments/data_csv/b2_episodes.csv) (`1920` rows x `26` cols)
- [b2_workload_agg.csv](file:///./formal_experiments/data_csv/b2_workload_agg.csv) (`192` rows x `17` cols)
- [b2_micro100.csv](file:///./formal_experiments/data_csv/b2_micro100.csv) (`900` rows x `11` cols)
- [b3a_grafting.csv](file:///./formal_experiments/data_csv/b3a_grafting.csv) (`288` rows x `13` cols)
- [b3b_ablation_synergy.csv](file:///./formal_experiments/data_csv/b3b_ablation_synergy.csv) (`216` rows x `9` cols)
- [b3c_xray_traces.csv](file:///./formal_experiments/data_csv/b3c_xray_traces.csv) (`52` rows x `19` cols)
- [b3d_sensitivity_10s.csv](file:///./formal_experiments/data_csv/b3d_sensitivity_10s.csv) (`39` rows x `17` cols)
- [b3e_regime_map_10s.csv](file:///./formal_experiments/data_csv/b3e_regime_map_10s.csv) (`36` rows x `17` cols)
- [b4a_azure_trace.csv](file:///./formal_experiments/data_csv/b4a_azure_trace.csv) (`384` rows x `18` cols)
- [b4b_cross_hw.csv](file:///./formal_experiments/data_csv/b4b_cross_hw.csv) (`1080` rows x `11` cols)
- [b4b_perturb_sweep.csv](file:///./formal_experiments/data_csv/b4b_perturb_sweep.csv) (`880` rows x `10` cols)
- [b4c_scalability.csv](file:///./formal_experiments/data_csv/b4c_scalability.csv) (`350` rows x `14` cols)
- [b4d_theorem_boundaries.csv](file:///./formal_experiments/data_csv/b4d_theorem_boundaries.csv) (`18` rows x `10` cols)

### Statistical Reports (`stats/`)
- [upstream_hashes_baseline.json](file:///./formal_experiments/stats/upstream_hashes_baseline.json)
- [exp_config.json](file:///./formal_experiments/stats/exp_config.json)
- [s50_preflight_report.json](file:///./formal_experiments/stats/s50_preflight_report.json)
- [b1_summary.json](file:///./formal_experiments/stats/b1_summary.json)
- [b2_summary.json](file:///./formal_experiments/stats/b2_summary.json)
- [b3_summary.json](file:///./formal_experiments/stats/b3_summary.json)
- [b4_summary.json](file:///./formal_experiments/stats/b4_summary.json)
- [s55_gates_report.json](file:///./formal_experiments/stats/s55_gates_report.json)
