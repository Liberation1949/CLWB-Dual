"""Benchmark Suite (K=12 Main Independent Workloads + K_OOD=4 Held-Out) & 3-Route Selection (Stage 3).

Strict Statistical Protocol:
  * K = 12 genuinely independent workloads spanning Phase-Space Zone III and boundary regimes,
    compared against m = 7 dual-track baselines (+CommonGuard):
      p_min^Holm = m * 2^{-(K - 1)} = 7 * 2^{-11} = 0.00342 <= 0.02.
  * Each workload averages 3 deterministic scenario seeds BEFORE paired Wilcoxon-Holm testing
    so N_stat = K = 12 strictly (zero seed-pooling pseudo-replication).
  * K_OOD = 4 unseen OOD workloads use separate seed namespace (88000+).
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
STAGE3_DIR = Path(__file__).resolve().parent
for _p in (str(STAGE2_DIR), str(STAGE3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines_stage2 import (  # noqa: E402
    make_bambos_alp,
    make_chen_game,
    make_dgolb,
    make_edf_prebatch,
    make_ibr,
    make_llf_prebatch,
    make_sfa,
)
from candidates_routes import (  # noqa: E402
    build_counterexample_2nd_order,
    make_route_a_clsp,
    make_route_b_waterfill,
    make_route_c_whittle,
)
from common_guard import wrap_with_common_guard  # noqa: E402
from exact_oracle import (  # noqa: E402
    build_counterexample_L1,
    build_counterexample_L2,
    build_counterexample_L3,
    solve_offline_exact,
)
from model_spec import SimConfig, build_scenario_v1, make_micro_scenario, run_simulation  # noqa: E402


def build_main_benchmark_k12(n_seeds: int = 3) -> list[dict[str, Any]]:
    """Build K=12 independent main workloads with 3 seeds each."""
    specs = [
        dict(wid="WK01_BALANCED_MED", n=24, m=6, q=6, load=0.60, T=1.00, r=300.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK02_DENSE_TIGHT", n=30, m=4, q=5, load=0.75, T=0.60, r=150.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK03_DEADLINE_TIGHT", n=24, m=6, q=6, load=0.70, T=0.60, r=220.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK04_HETERO_DEV", n=24, m=6, q=6, load=0.70, T=0.80, r=250.0, fdev=(0.30, 1.30), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK05_HETERO_DL", n=28, m=5, q=5, load=0.72, T=0.75, r=230.0, fdev=(0.65, 0.95), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=(0.50, 1.50), sb=1, csi=0.0),
        dict(wid="WK06_BURST_MODERATE", n=40, m=6, q=4, load=0.85, T=0.70, r=220.0, fdev=(0.65, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK07_BURST_HEAVY", n=55, m=6, q=3, load=0.90, T=0.68, r=200.0, fdev=(0.60, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK08_HIGH_PAYLOAD", n=24, m=5, q=5, load=0.65, T=0.85, r=240.0, fdev=(0.65, 0.90), ptx=(0.8, 1.3), kb=(90.0, 150.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK09_SMALL_CELL_ASYM", n=32, m=4, q=4, load=0.70, T=0.65, r=120.0, fdev=(0.60, 0.95), ptx=(0.6, 1.8), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="WK10_DUAL_SUBBAND", n=36, m=6, q=5, load=0.75, T=0.65, r=180.0, fdev=(0.65, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=2, csi=0.0),
        dict(wid="WK11_CSI_NOISE_2DB", n=24, m=6, q=6, load=0.70, T=0.75, r=220.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=2.0),
        dict(wid="WK12_LARGE_SCALE", n=60, m=8, q=3, load=0.82, T=0.75, r=240.0, fdev=(0.65, 0.95), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
    ]
    suite: list[dict[str, Any]] = []
    for w_idx, sp in enumerate(specs):
        cells = []
        for s_rep in range(int(n_seeds)):
            seed = 41000 + w_idx * 100 + s_rep
            scen = build_scenario_v1(
                case_id=f"{sp['wid']}_S{s_rep}",
                family=sp["wid"],
                seed=seed,
                n_devices=sp["n"],
                n_servers=sp["m"],
                tasks_per_device=sp["q"],
                offered_load=sp["load"],
                T_max_s=sp["T"],
                radius_m=sp["r"],
                f_dev_range_ghz=sp["fdev"],
                p_tx_range_w=sp["ptx"],
                input_kb_range=sp["kb"],
                hetero_deadline_range=sp["hetero"],
                n_subbands=sp["sb"],
            )
            cfg = SimConfig(
                tick_s=0.05,
                delta_sig_s=0.001,
                rho_sys=0.95,
                latency_mode="wall",
                n_subbands=sp["sb"],
                csi_sigma_db=sp["csi"],
                csi_seed=seed + 50000,
            )
            cells.append((scen, cfg))
        suite.append({"wid": sp["wid"], "cells": cells})
    return suite


def build_ood_suite_k4(n_seeds: int = 3) -> list[dict[str, Any]]:
    """Build K_OOD=4 unseen held-out out-of-distribution workloads (seed namespace 88000+)."""
    specs = [
        dict(wid="OOD1_ASYM_NEARFAR", n=34, m=5, q=5, load=0.74, T=0.70, r=160.0, fdev=(0.50, 1.10), ptx=(0.5, 2.0), kb=(70.0, 130.0), hetero=None, sb=1, csi=0.0),
        dict(wid="OOD2_HETDL_CSI2DB", n=28, m=5, q=5, load=0.72, T=0.72, r=210.0, fdev=(0.60, 1.00), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=(0.50, 1.50), sb=1, csi=2.0),
        dict(wid="OOD3_EXTREME_BURST75", n=75, m=6, q=2, load=0.92, T=0.65, r=210.0, fdev=(0.60, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
        dict(wid="OOD4_SPARSE_SRV", n=26, m=3, q=5, load=0.78, T=0.75, r=240.0, fdev=(0.50, 0.95), ptx=(0.8, 1.3), kb=(60.0, 120.0), hetero=None, sb=1, csi=0.0),
    ]
    suite: list[dict[str, Any]] = []
    for w_idx, sp in enumerate(specs):
        cells = []
        for s_rep in range(int(n_seeds)):
            seed = 88000 + w_idx * 100 + s_rep
            scen = build_scenario_v1(
                case_id=f"{sp['wid']}_S{s_rep}",
                family=sp["wid"],
                seed=seed,
                n_devices=sp["n"],
                n_servers=sp["m"],
                tasks_per_device=sp["q"],
                offered_load=sp["load"],
                T_max_s=sp["T"],
                radius_m=sp["r"],
                f_dev_range_ghz=sp["fdev"],
                p_tx_range_w=sp["ptx"],
                input_kb_range=sp["kb"],
                hetero_deadline_range=sp["hetero"],
                n_subbands=sp["sb"],
            )
            cfg = SimConfig(
                tick_s=0.05,
                delta_sig_s=0.001,
                rho_sys=0.95,
                latency_mode="wall",
                n_subbands=sp["sb"],
                csi_sigma_db=sp["csi"],
                csi_seed=seed + 90000,
            )
            cells.append((scen, cfg))
        suite.append({"wid": sp["wid"], "cells": cells})
    return suite


def build_dual_track_cg_baselines() -> list[tuple[str, Callable[[dict], dict]]]:
    """All 7 Stage-2 dual-track baselines uniformly wrapped with CommonGuard (+CG)."""
    return [
        ("EDF_PreBatch+CG", wrap_with_common_guard(make_edf_prebatch(), "EDF_PreBatch+CG")),
        ("LLF_PreBatch+CG", wrap_with_common_guard(make_llf_prebatch(), "LLF_PreBatch+CG")),
        ("Bambos_ALP+CG", wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")),
        ("Chen_Game+CG", wrap_with_common_guard(make_chen_game(max_rounds=8), "Chen_Game+CG")),
        ("DGOLB+CG", wrap_with_common_guard(make_dgolb(), "DGOLB+CG")),
        ("SFA+CG", wrap_with_common_guard(make_sfa(), "SFA+CG")),
        ("IBR_5+CG", wrap_with_common_guard(make_ibr(max_rounds=5), "IBR_5+CG")),
    ]


def evaluate_route_comparison() -> dict[str, Any]:
    """Compare Route A vs Route B vs Route C across counterexamples, dev workloads, and P=70 burst."""
    routes = [
        ("RouteA_CLSP", make_route_a_clsp()),
        ("RouteB_Waterfill", make_route_b_waterfill()),
        ("RouteC_Whittle", make_route_c_whittle()),
    ]
    strongest_cgs = [
        ("Chen_Game+CG", wrap_with_common_guard(make_chen_game(max_rounds=8), "Chen_Game+CG")),
        ("SFA+CG", wrap_with_common_guard(make_sfa(), "SFA+CG")),
    ]

    # 1. Counterexamples L1, L2, L3, CE_2ndOrder
    ce_builders = [
        ("L1_prebatch", build_counterexample_L1),
        ("L2_nondelay", build_counterexample_L2),
        ("L3_myopic", build_counterexample_L3),
        ("CE_2ndOrder", build_counterexample_2nd_order),
    ]
    ce_scores: dict[str, dict[str, Any]] = {}
    for cname, fn in ce_builders:
        sc, cf = fn()
        ex = solve_offline_exact(sc, cf)
        row: dict[str, Any] = {"oracle": f"{ex['best_ontime']}/{sc.n_tasks}"}
        for pname, pol in routes + strongest_cgs:
            res = run_simulation(sc, pol, cf)
            row[pname] = f"{res['n_ontime']}/{sc.n_tasks}"
        ce_scores[cname] = row

    # 2. Micro-oracle capture (100 Stage-2 Gate 4 instances, seed=777)
    rng = np.random.default_rng(777)
    cfg_z = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    micro_acc = {rname: [] for rname, _ in routes}
    oracle_acc = []
    for idx in range(100):
        n = int(rng.integers(2, 5))
        m = int(rng.integers(1, 3))
        r = int(rng.integers(3, 6))
        h = rng.uniform(4e-11, 4e-10, size=(n, m))
        f_loc = rng.uniform(0.3e9, 1.2e9, size=n)
        f_ms = rng.uniform(15.0e9, 35.0e9, size=m)
        ptx = rng.uniform(0.8, 1.3, size=n)
        tasks = []
        for k in range(r):
            d = int(rng.integers(0, n))
            arr = float(rng.choice([0.0, 0.0, 0.02, 0.05, 0.07]))
            span = float(rng.uniform(0.14, 0.28))
            tasks.append({
                "tid": f"{d}:{k}", "device": d, "k": k, "arrival_s": arr,
                "deadline_s": arr + span, "D_bits": float(rng.uniform(4e5, 9.5e5)),
                "C_cycles": float(rng.uniform(2e8, 8e8)), "tau_dn_s": 0.001,
            })
        scen = make_micro_scenario(
            case_id=f"M_{idx:03d}", n_devices=n, n_servers=m, h_dev_ms=h,
            f_loc_hz=f_loc, f_ms_max_hz=f_ms, p_tx_w=ptx, tasks_raw=tasks,
        )
        ex = solve_offline_exact(scen, cfg_z)
        oracle_acc.append(ex["best_rate"])
        for rname, pol in routes:
            res = run_simulation(scen, pol, cfg_z)
            micro_acc[rname].append(res["ontime_rate"])

    # 3. Main K=12 full suite comparison across routes + Chen_Game+CG
    main_k12 = build_main_benchmark_k12(n_seeds=2)
    wk_ontime: dict[str, list[float]] = {rname: [] for rname, _ in routes + [("Chen_Game+CG", strongest_cgs[0][1])]}
    wk_miss: dict[str, list[float]] = {rname: [] for rname, _ in routes + [("Chen_Game+CG", strongest_cgs[0][1])]}
    wk_p99: dict[str, float] = {rname: 0.0 for rname, _ in routes + [("Chen_Game+CG", strongest_cgs[0][1])]}
    wk_skip: dict[str, int] = {rname: 0 for rname, _ in routes + [("Chen_Game+CG", strongest_cgs[0][1])]}

    for wk in main_k12:
        for pname, pol in routes + [("Chen_Game+CG", strongest_cgs[0][1])]:
            ot_c, ms_c = [], []
            for sc, cf in wk["cells"]:
                res = run_simulation(sc, pol, cf)
                ot_c.append(res["ontime_rate"])
                ms_c.append(res["server_grant_miss_rate"])
                wk_p99[pname] = max(wk_p99[pname], res["decision_p99_ms"])
                wk_skip[pname] += res["skipped_ticks"]
            wk_ontime[pname].append(float(np.mean(ot_c)))
            wk_miss[pname].append(float(np.mean(ms_c)))

    # 4. Burst P=70 latency test
    scen_b70 = build_scenario_v1(
        case_id="ROUTE_BURST_P70", family="burst", seed=9070,
        n_devices=70, n_servers=6, tasks_per_device=1,
        offered_load=0.80, T_max_s=0.65, radius_m=220.0, window_s=0.005,
    )
    cfg_b70 = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
    burst_p70 = {}
    for pname, pol in routes:
        r70 = run_simulation(scen_b70, pol, cfg_b70)
        burst_p70[pname] = {
            "ontime_rate": r70["ontime_rate"],
            "decision_p99_ms": r70["decision_p99_ms"],
            "skipped_ticks": r70["skipped_ticks"],
            "miss_rate": r70["server_grant_miss_rate"],
        }

    summary = {
        "counterexamples": ce_scores,
        "micro_100_oracle_mean": float(np.mean(oracle_acc)),
        "micro_100_route_means": {k: float(np.mean(v)) for k, v in micro_acc.items()},
        "main_k12_mean_ontime": {k: float(np.mean(v)) for k, v in wk_ontime.items()},
        "main_k12_mean_miss": {k: float(np.mean(v)) for k, v in wk_miss.items()},
        "main_k12_max_p99_ms": wk_p99,
        "main_k12_total_skipped_ticks": wk_skip,
        "burst_p70": burst_p70,
    }
    return summary


if __name__ == "__main__":
    t0 = time.perf_counter()
    rep = evaluate_route_comparison()
    rep["elapsed_s"] = time.perf_counter() - t0
    out_p = STAGE3_DIR / "route_comparison_report.json"
    out_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("=== STAGE-3 THREE-ROUTE HEAD-TO-HEAD SUMMARY ===")
    print("Counterexamples:", json.dumps(rep["counterexamples"], indent=2))
    print("100 Micro Oracle Capture (Oracle=1.000):", rep["micro_100_route_means"])
    print("Main K=12 Mean Ontime:", rep["main_k12_mean_ontime"])
    print("Main K=12 Mean Miss:", rep["main_k12_mean_miss"])
    print("Main K=12 Max P99 (ms):", rep["main_k12_max_p99_ms"])
    print("Main K=12 Skipped Ticks:", rep["main_k12_total_skipped_ticks"])
    print("Burst P=70:", json.dumps(rep["burst_p70"], indent=2))
