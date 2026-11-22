"""Step 5.1 (Batch 1): Motivation & Dual-Deadlock Diagnosis Experiments.

Produces:
  - data_csv/b1a_ortho_vs_shared.csv
  - data_csv/b1b_kappa_sweep.csv
  - data_csv/b1c_burst_cliff.csv
  - data_csv/b1d_counterexamples.csv
  - stats/b1_summary.json
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from common_utils import (
    DATA_DIR,
    STATS_DIR,
    bootstrap_ci95,
    build_single_frozen_champion,
    save_csv_strict,
    save_json_strict,
    verify_upstream_readonly,
)

from benchmark_suite_s3 import build_dual_track_cg_baselines  # noqa: E402
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
from exact_oracle import (  # noqa: E402
    build_counterexample_L1,
    build_counterexample_L2,
    build_counterexample_L3,
    solve_offline_exact,
)
from model_spec import SimConfig, build_scenario_v1, compute_dimensionless_numbers, run_simulation  # noqa: E402


def run_exp_1a(policies: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s_idx in range(10):
        seed = 51100 + s_idx
        sc = build_scenario_v1(
            case_id=f"B1A_S{s_idx}",
            family="b1a_ortho_shared",
            seed=seed,
            n_devices=24,
            n_servers=6,
            tasks_per_device=5,
            offered_load=0.70,
            T_max_s=0.75,
            radius_m=240.0,
            input_kb_range=(70.0, 120.0),
        )
        for ch_mode, ortho in (("orthogonal", True), ("shared", False)):
            cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=ortho, latency_mode="wall")
            dm = compute_dimensionless_numbers(sc, cfg)
            for pname, pfn in policies:
                r = run_simulation(sc, pfn, cfg)
                rows.append({
                    "channel_mode": ch_mode,
                    "seed_idx": int(s_idx),
                    "seed": int(seed),
                    "kappa_ul": float(dm["kappa_ul"]),
                    "rho_cmp": float(dm["rho_cmp"]),
                    "method": pname,
                    "n_tasks": int(r["n_tasks"]),
                    "n_ontime": int(r["n_ontime"]),
                    "ontime_rate": float(r["ontime_rate"]),
                    "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                    "wasted_srv_cycle_ratio": float(r["wasted_srv_cycle_ratio"]),
                    "decision_p99_ms": float(r["decision_p99_ms"]),
                    "skipped_ticks": int(r["skipped_ticks"]),
                    "illegal_actions": int(r["illegal_actions"]),
                })
    return rows


def run_exp_1b(cg_dict: dict[str, Any], champ: Any) -> list[dict[str, Any]]:
    methods = [
        ("DGOLB+CG", cg_dict["DGOLB+CG"]),
        ("EDF_PreBatch+CG", cg_dict["EDF_PreBatch+CG"]),
        ("Chen_Game+CG", cg_dict["Chen_Game+CG"]),
        ("Bambos_ALP+CG", cg_dict["Bambos_ALP+CG"]),
        ("CLWB_Dual", champ),
    ]
    scales = [0.0, 0.05, 0.15, 0.35, 0.70, 1.0, 1.5, 2.5]
    rows: list[dict[str, Any]] = []
    for isc in scales:
        ortho = (isc == 0.0)
        kb_lo = 60.0 if ortho else 60.0 * isc
        kb_hi = 120.0 if ortho else 120.0 * isc
        eff_isc = 1.0 if ortho else isc
        for s_idx in range(10):
            seed = 51200 + s_idx
            sc = build_scenario_v1(
                case_id=f"B1B_ISC{isc}_S{s_idx}",
                family="b1b_kappa_sweep",
                seed=seed,
                n_devices=20,
                n_servers=5,
                tasks_per_device=5,
                offered_load=0.65,
                T_max_s=0.80,
                input_kb_range=(kb_lo, kb_hi),
                interf_scale=eff_isc,
            )
            cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=ortho, latency_mode="wall")
            dm = compute_dimensionless_numbers(sc, cfg)
            for pname, pfn in methods:
                r = run_simulation(sc, pfn, cfg)
                rows.append({
                    "interf_scale": float(isc),
                    "seed_idx": int(s_idx),
                    "seed": int(seed),
                    "kappa_ul": float(dm["kappa_ul"]),
                    "rho_cmp": float(dm["rho_cmp"]),
                    "method": pname,
                    "n_tasks": int(r["n_tasks"]),
                    "n_ontime": int(r["n_ontime"]),
                    "ontime_rate": float(r["ontime_rate"]),
                    "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                    "wasted_srv_cycle_ratio": float(r["wasted_srv_cycle_ratio"]),
                    "decision_p99_ms": float(r["decision_p99_ms"]),
                    "skipped_ticks": int(r["skipped_ticks"]),
                    "illegal_actions": int(r["illegal_actions"]),
                })
    return rows


def run_exp_1c(cg_dict: dict[str, Any], champ: Any) -> list[dict[str, Any]]:
    methods = [
        ("SFA+CG", cg_dict["SFA+CG"]),
        ("IBR_5+CG", cg_dict["IBR_5+CG"]),
        ("Bambos_ALP+CG", cg_dict["Bambos_ALP+CG"]),
        ("DGOLB+CG", cg_dict["DGOLB+CG"]),
        ("CLWB_Dual", champ),
    ]
    p_vals = [8, 16, 25, 40, 60, 80]
    rows: list[dict[str, Any]] = []
    for p_cnt in p_vals:
        for s_idx in range(10):
            seed = 51300 + s_idx
            sc = build_scenario_v1(
                case_id=f"B1C_P{p_cnt}_S{s_idx}",
                family="b1c_burst_cliff",
                seed=seed,
                n_devices=p_cnt,
                n_servers=6,
                tasks_per_device=1,
                offered_load=0.80,
                T_max_s=0.80,
                radius_m=220.0,
                window_s=0.04,
            )
            cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
            for pname, pfn in methods:
                r = run_simulation(sc, pfn, cfg)
                rows.append({
                    "P_burst": int(p_cnt),
                    "seed_idx": int(s_idx),
                    "seed": int(seed),
                    "method": pname,
                    "n_tasks": int(r["n_tasks"]),
                    "n_ontime": int(r["n_ontime"]),
                    "ontime_rate": float(r["ontime_rate"]),
                    "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                    "decision_p50_ms": float(r["decision_p50_ms"]),
                    "decision_p99_ms": float(r["decision_p99_ms"]),
                    "skipped_ticks": int(r["skipped_ticks"]),
                    "illegal_actions": int(r["illegal_actions"]),
                })
        print(f"[Exp 1C] Completed P_burst={p_cnt}", flush=True)
    return rows


def run_exp_1d(policies: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    builders = [
        ("L1_PreBatch", build_counterexample_L1),
        ("L2_NonDelay", build_counterexample_L2),
        ("L3_MyopicEDF", build_counterexample_L3),
        ("CE_2ndOrder", build_counterexample_2nd_order),
    ]
    rows: list[dict[str, Any]] = []
    for cname, bfn in builders:
        sc, cfg = bfn()
        ex = solve_offline_exact(sc, cfg)
        rows.append({
            "counterexample": cname,
            "method": "Exact_Oracle",
            "n_tasks": int(sc.n_tasks),
            "n_ontime": int(ex["best_ontime"]),
            "ontime_rate": float(ex["best_rate"]),
            "server_grant_miss_rate": 0.0,
            "illegal_actions": 0,
        })
        for pname, pfn in policies:
            r = run_simulation(sc, pfn, cfg)
            rows.append({
                "counterexample": cname,
                "method": pname,
                "n_tasks": int(sc.n_tasks),
                "n_ontime": int(r["n_ontime"]),
                "ontime_rate": float(r["ontime_rate"]),
                "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                "illegal_actions": int(r["illegal_actions"]),
            })
    return rows


def build_b1_summary(
    rows_1a: list[dict[str, Any]],
    rows_1b: list[dict[str, Any]],
    rows_1c: list[dict[str, Any]],
    rows_1d: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    df1a = pd.DataFrame(rows_1a)
    sum_1a: dict[str, Any] = {}
    for ch in ("orthogonal", "shared"):
        sub = df1a[df1a["channel_mode"] == ch]
        sum_1a[ch] = {}
        for m, grp in sub.groupby("method", sort=False):
            ont_m, ont_lo, ont_hi = bootstrap_ci95(grp["ontime_rate"].to_numpy())
            ms_m, ms_lo, ms_hi = bootstrap_ci95(grp["server_grant_miss_rate"].to_numpy())
            ws_m, ws_lo, ws_hi = bootstrap_ci95(grp["wasted_srv_cycle_ratio"].to_numpy())
            sum_1a[ch][str(m)] = {
                "ontime_mean": ont_m,
                "ontime_ci95": [ont_lo, ont_hi],
                "miss_mean": ms_m,
                "miss_ci95": [ms_lo, ms_hi],
                "wasted_mean": ws_m,
                "wasted_ci95": [ws_lo, ws_hi],
                "max_p99_ms": float(grp["decision_p99_ms"].max()),
                "total_skipped_ticks": int(grp["skipped_ticks"].sum()),
            }

    df1b = pd.DataFrame(rows_1b)
    sum_1b: dict[str, Any] = {}
    for isc, grp_i in df1b.groupby("interf_scale", sort=True):
        k_mean = float(grp_i["kappa_ul"].mean())
        sum_1b[str(isc)] = {"kappa_ul_mean": k_mean, "methods": {}}
        for m, grp_m in grp_i.groupby("method", sort=False):
            ont_m, ont_lo, ont_hi = bootstrap_ci95(grp_m["ontime_rate"].to_numpy())
            ms_m, ms_lo, ms_hi = bootstrap_ci95(grp_m["server_grant_miss_rate"].to_numpy())
            sum_1b[str(isc)]["methods"][str(m)] = {
                "ontime_mean": ont_m,
                "ontime_ci95": [ont_lo, ont_hi],
                "miss_mean": ms_m,
                "miss_ci95": [ms_lo, ms_hi],
            }

    df1c = pd.DataFrame(rows_1c)
    sum_1c: dict[str, Any] = {}
    for p_cnt, grp_p in df1c.groupby("P_burst", sort=True):
        sum_1c[str(p_cnt)] = {}
        for m, grp_m in grp_p.groupby("method", sort=False):
            ont_m, ont_lo, ont_hi = bootstrap_ci95(grp_m["ontime_rate"].to_numpy())
            p99_m, p99_lo, p99_hi = bootstrap_ci95(grp_m["decision_p99_ms"].to_numpy())
            sum_1c[str(p_cnt)][str(m)] = {
                "ontime_mean": ont_m,
                "ontime_ci95": [ont_lo, ont_hi],
                "p99_ms_mean": p99_m,
                "p99_ms_ci95": [p99_lo, p99_hi],
                "p99_ms_max": float(grp_m["decision_p99_ms"].max()),
                "total_skipped_ticks": int(grp_m["skipped_ticks"].sum()),
            }

    df1d = pd.DataFrame(rows_1d)
    sum_1d: dict[str, Any] = {}
    for ce, grp_c in df1d.groupby("counterexample", sort=False):
        sum_1d[str(ce)] = {
            str(r["method"]): {
                "n_ontime": int(r["n_ontime"]),
                "n_tasks": int(r["n_tasks"]),
                "ontime_rate": float(r["ontime_rate"]),
            }
            for _, r in grp_c.iterrows()
        }

    return {
        "batch": 1,
        "elapsed_s": float(elapsed_s),
        "exp_1a_ortho_vs_shared": sum_1a,
        "exp_1b_kappa_sweep": sum_1b,
        "exp_1c_burst_cliff": sum_1c,
        "exp_1d_counterexamples": sum_1d,
    }


def main() -> None:
    verify_upstream_readonly()
    t0 = time.perf_counter()
    champ = build_single_frozen_champion()
    cg_baselines = build_dual_track_cg_baselines()
    cg_dict = dict(cg_baselines)
    all_policies = [("CLWB_Dual", champ)] + cg_baselines

    print("[Batch 1] Running Exp 1A (Orthogonal vs Shared, 10 seeds)...", flush=True)
    rows_1a = run_exp_1a(all_policies)
    save_csv_strict(rows_1a, DATA_DIR / "b1a_ortho_vs_shared.csv")

    print("[Batch 1] Running Exp 1B (kappa_ul phase transition, 10 seeds)...", flush=True)
    rows_1b = run_exp_1b(cg_dict, champ)
    save_csv_strict(rows_1b, DATA_DIR / "b1b_kappa_sweep.csv")

    print("[Batch 1] Running Exp 1C (Burst P control-plane cliff, 10 seeds)...", flush=True)
    rows_1c = run_exp_1c(cg_dict, champ)
    save_csv_strict(rows_1c, DATA_DIR / "b1c_burst_cliff.csv")

    print("[Batch 1] Running Exp 1D (4 minimal counterexamples)...", flush=True)
    rows_1d = run_exp_1d(all_policies)
    save_csv_strict(rows_1d, DATA_DIR / "b1d_counterexamples.csv")

    elapsed = time.perf_counter() - t0
    summary = build_b1_summary(rows_1a, rows_1b, rows_1c, rows_1d, elapsed)
    save_json_strict(summary, STATS_DIR / "b1_summary.json")

    illegal_all = sum(int(r["illegal_actions"]) for r in rows_1a + rows_1b + rows_1c + rows_1d)
    print(f"[Batch 1] Complete in {elapsed:.1f}s | illegal_actions={illegal_all}", flush=True)
    if illegal_all > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
