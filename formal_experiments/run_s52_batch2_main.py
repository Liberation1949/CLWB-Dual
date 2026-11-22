"""Step 5.2 (Batch 2): Main Full-Matrix Benchmark (24 Workloads x 10 Seeds) + 100 Micro Oracle.

Supports incremental checkpointing by (suite, wid, seed_idx, method) to data_csv/b2_episodes.csv.
Produces:
  - data_csv/b2_episodes.csv
  - data_csv/b2_workload_agg.csv
  - data_csv/b2_micro100.csv
  - stats/b2_summary.json
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
    holm_bonferroni,
    paired_wilcoxon_and_cliffs,
    save_csv_strict,
    save_json_strict,
    unpack_suite_item,
    verify_upstream_readonly,
)

from benchmark_suite_s3 import (  # noqa: E402
    build_dual_track_cg_baselines,
    build_main_benchmark_k12,
    build_ood_suite_k4,
)
from exact_oracle import solve_offline_exact  # noqa: E402
from model_spec import SimConfig, compute_dimensionless_numbers, make_micro_scenario, run_simulation  # noqa: E402
from step41_blind_and_hardware import build_blind_suite_k8  # noqa: E402


EP_COLUMNS = [
    "suite",
    "wid",
    "seed_idx",
    "seed",
    "kappa_ul",
    "rho_cmp",
    "gamma_tick",
    "method",
    "n_tasks",
    "n_ontime",
    "ontime_rate",
    "n_timeout",
    "n_dropped",
    "n_edge_granted",
    "n_edge_missed",
    "server_grant_miss_rate",
    "wasted_srv_cycle_ratio",
    "skipped_ticks",
    "decision_p50_ms",
    "decision_p99_ms",
    "policy_calls",
    "illegal_actions",
    "max_bit_rel_err",
    "hard_invariant_checks",
    "energy_tx_j",
    "energy_cmp_j",
]


def run_micro100_oracle(policies: list[tuple[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Stage-2 Gate 4 100 micro-instances vs Exact Offline Oracle."""
    rng = np.random.default_rng(777)
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    rows: list[dict[str, Any]] = []
    oracle_rates: list[float] = []
    method_rates: dict[str, list[float]] = {pname: [] for pname, _ in policies}
    inversions = 0
    champ_illegal = 0

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
                "tid": f"{d}:{k}",
                "device": d,
                "k": k,
                "arrival_s": arr,
                "deadline_s": arr + span,
                "D_bits": float(rng.uniform(4e5, 9.5e5)),
                "C_cycles": float(rng.uniform(2e8, 8e8)),
                "tau_dn_s": 0.001,
            })
        scen = make_micro_scenario(
            case_id=f"M_{idx:03d}",
            n_devices=n,
            n_servers=m,
            h_dev_ms=h,
            f_loc_hz=f_loc,
            f_ms_max_hz=f_ms,
            p_tx_w=ptx,
            tasks_raw=tasks,
        )
        ex = solve_offline_exact(scen, cfg)
        oracle_rates.append(float(ex["best_rate"]))
        rows.append({
            "inst_idx": int(idx),
            "case_id": scen.case_id,
            "n_devices": int(n),
            "n_servers": int(m),
            "n_tasks": int(scen.n_tasks),
            "method": "Exact_Oracle",
            "n_ontime": int(ex["best_ontime"]),
            "ontime_rate": float(ex["best_rate"]),
            "server_grant_miss_rate": 0.0,
            "illegal_actions": 0,
            "hard_invariant_checks": 0,
        })
        for pname, pfn in policies:
            res = run_simulation(scen, pfn, cfg)
            method_rates[pname].append(float(res["ontime_rate"]))
            if int(res["n_ontime"]) > int(ex["best_ontime"]):
                inversions += 1
            if pname == "CLWB_Dual":
                champ_illegal += int(res["illegal_actions"])
            rows.append({
                "inst_idx": int(idx),
                "case_id": scen.case_id,
                "n_devices": int(n),
                "n_servers": int(m),
                "n_tasks": int(scen.n_tasks),
                "method": pname,
                "n_ontime": int(res["n_ontime"]),
                "ontime_rate": float(res["ontime_rate"]),
                "server_grant_miss_rate": float(res["server_grant_miss_rate"]),
                "illegal_actions": int(res["illegal_actions"]),
                "hard_invariant_checks": int(res["hard_invariant_checks"]),
            })

    oracle_mean = float(np.mean(oracle_rates))
    champ_mean = float(np.mean(method_rates["CLWB_Dual"]))
    capture_ratio = champ_mean / max(oracle_mean, 1e-12)
    summary = {
        "oracle_mean": oracle_mean,
        "capture_ratio": float(capture_ratio),
        "inversions": int(inversions),
        "champ_illegal_actions": int(champ_illegal),
        "methods": {
            pname: {
                "mean_ontime": float(np.mean(vals)),
                "capture_ratio": float(np.mean(vals) / max(oracle_mean, 1e-12)),
                "ci95": list(bootstrap_ci95(vals)[1:]),
            }
            for pname, vals in method_rates.items()
        },
    }
    return rows, summary


def _summarize_scope(
    df_ep: pd.DataFrame,
    df_agg: pd.DataFrame,
    scope_name: str,
    suite_filter: list[str],
    method_order: list[str],
    baseline_names: list[str],
) -> dict[str, Any]:
    sub_agg = df_agg[df_agg["suite"].isin(suite_filter)].copy()
    sub_ep = df_ep[df_ep["suite"].isin(suite_filter)].copy()
    sub_agg = sub_agg.sort_values(["suite", "wid"])
    wids_unique = list(dict.fromkeys(zip(sub_agg["suite"], sub_agg["wid"])))
    k_workloads = len(wids_unique)

    per_method: dict[str, Any] = {}
    wk_vecs: dict[str, np.ndarray] = {}

    for mname in method_order:
        m_agg = sub_agg[sub_agg["method"] == mname].set_index(["suite", "wid"]).loc[wids_unique]
        m_ep = sub_ep[sub_ep["method"] == mname]
        vals = m_agg["ontime_mean"].to_numpy(dtype=float)
        wk_vecs[mname] = vals
        mean_v, ci_lo, ci_hi = bootstrap_ci95(vals, n_boot=10000, seed=52001)
        ep_ont = m_ep["ontime_rate"].to_numpy(dtype=float)
        ep_p99 = m_ep["decision_p99_ms"].to_numpy(dtype=float)
        per_method[mname] = {
            "n_workloads": int(k_workloads),
            "n_episodes": int(len(m_ep)),
            "ontime_mean": mean_v,
            "ontime_ci95": [ci_lo, ci_hi],
            "worst_wk_ontime": float(np.min(vals)),
            "ep_p05_ontime": float(np.quantile(ep_ont, 0.05)),
            "miss_mean": float(m_agg["miss_mean"].mean()),
            "wasted_mean": float(m_agg["wasted_mean"].mean()),
            "p50_ms_mean": float(m_agg["p50_ms_mean"].mean()),
            "ep_p95_decision_ms": float(np.quantile(ep_p99, 0.95)),
            "max_p99_decision_ms": float(m_agg["p99_ms_max"].max()),
            "total_skipped_ticks": int(m_agg["skipped_ticks_sum"].sum()),
            "total_illegal_actions": int(m_agg["illegal_actions_sum"].sum()),
            "energy_tx_mean": float(m_agg["energy_tx_mean"].mean()),
            "energy_cmp_mean": float(m_agg["energy_cmp_mean"].mean()),
        }

    champ_vec = wk_vecs["CLWB_Dual"]
    raw_ps: list[float] = []
    cliffs_list: list[float] = []
    for bname in baseline_names:
        rp, cd = paired_wilcoxon_and_cliffs(champ_vec, wk_vecs[bname])
        raw_ps.append(rp)
        cliffs_list.append(cd)
    holm_ps = holm_bonferroni(raw_ps)

    pairwise: dict[str, Any] = {}
    for idx, bname in enumerate(baseline_names):
        bvec = wk_vecs[bname]
        diff = champ_vec - bvec
        d_mean_pp, d_lo_pp, d_hi_pp = bootstrap_ci95(diff * 100.0, n_boot=10000, seed=52100 + idx)
        wins = int(np.sum(diff > 1e-6))
        ties = int(np.sum(np.abs(diff) <= 1e-6))
        losses = int(np.sum(diff < -1e-6))
        pairwise[bname] = {
            "n_stat": int(k_workloads),
            "champ_mean_ontime": float(np.mean(champ_vec)),
            "base_mean_ontime": float(np.mean(bvec)),
            "delta_pp": d_mean_pp,
            "delta_pp_ci95": [d_lo_pp, d_hi_pp],
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "raw_p": float(raw_ps[idx]),
            "holm_p": float(holm_ps[idx]),
            "cliffs_delta": float(cliffs_list[idx]),
        }

    return {
        "scope": scope_name,
        "k_workloads": int(k_workloads),
        "per_method": per_method,
        "pairwise_vs_cg": pairwise,
    }


def main() -> None:
    verify_upstream_readonly()
    t0 = time.perf_counter()

    champ = build_single_frozen_champion()
    cg_baselines = build_dual_track_cg_baselines()
    policies: list[tuple[str, Any]] = [("CLWB_Dual", champ)] + cg_baselines
    method_order = [p[0] for p in policies]
    baseline_names = [p[0] for p in cg_baselines]

    suites: list[tuple[str, list[Any]]] = [
        ("Main_K12", build_main_benchmark_k12(n_seeds=10)),
        ("OOD_K4", build_ood_suite_k4(n_seeds=10)),
        ("Blind_K8", build_blind_suite_k8(n_seeds=10)),
    ]

    ep_path = DATA_DIR / "b2_episodes.csv"
    done_keys: set[tuple[str, str, int, str]] = set()
    existing_rows: list[dict[str, Any]] = []
    if ep_path.exists():
        df_existing = pd.read_csv(ep_path)
        existing_rows = df_existing.to_dict(orient="records")
        for r in existing_rows:
            done_keys.add((str(r["suite"]), str(r["wid"]), int(r["seed_idx"]), str(r["method"])))
        print(f"[Batch 2] Loaded {len(done_keys)} checkpointed episode rows from {ep_path.name}", flush=True)

    all_rows = list(existing_rows)
    total_workloads = sum(len(items) for _, items in suites)
    w_counter = 0

    for suite_name, items in suites:
        for item in items:
            w_counter += 1
            wid, cells = unpack_suite_item(item)
            new_wk_rows: list[dict[str, Any]] = []
            for s_idx, (sc, cfg) in enumerate(cells):
                missing_policies = [
                    (pname, pfn)
                    for pname, pfn in policies
                    if (suite_name, wid, int(s_idx), pname) not in done_keys
                ]
                if not missing_policies:
                    continue
                dm = compute_dimensionless_numbers(sc, cfg)
                for pname, pfn in missing_policies:
                    res = run_simulation(sc, pfn, cfg)
                    if res.get("status") != "ok":
                        raise RuntimeError(f"Simulation status != ok for {suite_name}/{wid}/S{s_idx}/{pname}: {res}")
                    row = {
                        "suite": suite_name,
                        "wid": wid,
                        "seed_idx": int(s_idx),
                        "seed": int(sc.seed),
                        "kappa_ul": float(dm["kappa_ul"]),
                        "rho_cmp": float(dm["rho_cmp"]),
                        "gamma_tick": float(dm["gamma_tick"]),
                        "method": pname,
                        "n_tasks": int(res["n_tasks"]),
                        "n_ontime": int(res["n_ontime"]),
                        "ontime_rate": float(res["ontime_rate"]),
                        "n_timeout": int(res["n_timeout"]),
                        "n_dropped": int(res["n_dropped"]),
                        "n_edge_granted": int(res["n_edge_granted"]),
                        "n_edge_missed": int(res["n_edge_missed"]),
                        "server_grant_miss_rate": float(res["server_grant_miss_rate"]),
                        "wasted_srv_cycle_ratio": float(res["wasted_srv_cycle_ratio"]),
                        "skipped_ticks": int(res["skipped_ticks"]),
                        "decision_p50_ms": float(res["decision_p50_ms"]),
                        "decision_p99_ms": float(res["decision_p99_ms"]),
                        "policy_calls": int(res["policy_calls"]),
                        "illegal_actions": int(res["illegal_actions"]),
                        "max_bit_rel_err": float(res["max_bit_rel_err"]),
                        "hard_invariant_checks": int(res["hard_invariant_checks"]),
                        "energy_tx_j": float(res["energy_tx_j"]),
                        "energy_cmp_j": float(res["energy_cmp_j"]),
                    }
                    new_wk_rows.append(row)
                    done_keys.add((suite_name, wid, int(s_idx), pname))

            if new_wk_rows:
                all_rows.extend(new_wk_rows)
                save_csv_strict(all_rows, ep_path)
            pct = 100.0 * float(w_counter) / float(total_workloads)
            print(
                f"[Batch 2] [{w_counter:02d}/{total_workloads:02d} {pct:5.1f}%] "
                f"{suite_name}/{wid} (+{len(new_wk_rows)} rows, elapsed={time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    df_ep = pd.DataFrame(all_rows)[EP_COLUMNS]
    save_csv_strict(df_ep, ep_path)

    # Anti-pooling workload aggregation (N_stat = K)
    agg_df = (
        df_ep.groupby(["suite", "wid", "method"], as_index=False, sort=False)
        .agg(
            n_seeds=("seed_idx", "count"),
            kappa_ul=("kappa_ul", "mean"),
            rho_cmp=("rho_cmp", "mean"),
            gamma_tick=("gamma_tick", "mean"),
            ontime_mean=("ontime_rate", "mean"),
            ontime_std=("ontime_rate", "std"),
            miss_mean=("server_grant_miss_rate", "mean"),
            wasted_mean=("wasted_srv_cycle_ratio", "mean"),
            p50_ms_mean=("decision_p50_ms", "mean"),
            p99_ms_max=("decision_p99_ms", "max"),
            skipped_ticks_sum=("skipped_ticks", "sum"),
            illegal_actions_sum=("illegal_actions", "sum"),
            energy_tx_mean=("energy_tx_j", "mean"),
            energy_cmp_mean=("energy_cmp_j", "mean"),
        )
    )
    agg_df["ontime_std"] = agg_df["ontime_std"].fillna(0.0)
    save_csv_strict(agg_df, DATA_DIR / "b2_workload_agg.csv")

    print("[Batch 2] Running 100 Micro-Instances Exact Oracle comparison...", flush=True)
    micro_rows, micro_summary = run_micro100_oracle(policies)
    save_csv_strict(micro_rows, DATA_DIR / "b2_micro100.csv")

    scopes = {
        "Main_K12": _summarize_scope(df_ep, agg_df, "Main_K12", ["Main_K12"], method_order, baseline_names),
        "OOD_K4": _summarize_scope(df_ep, agg_df, "OOD_K4", ["OOD_K4"], method_order, baseline_names),
        "Blind_K8": _summarize_scope(df_ep, agg_df, "Blind_K8", ["Blind_K8"], method_order, baseline_names),
        "All_24": _summarize_scope(df_ep, agg_df, "All_24", ["Main_K12", "OOD_K4", "Blind_K8"], method_order, baseline_names),
    }

    summary = {
        "batch": 2,
        "elapsed_s": float(time.perf_counter() - t0),
        "total_episode_rows": int(len(df_ep)),
        "total_workloads": int(agg_df[["suite", "wid"]].drop_duplicates().shape[0]),
        "micro_100_oracle": micro_summary,
        "scopes": scopes,
    }
    save_json_strict(summary, STATS_DIR / "b2_summary.json")

    illegal_total = int(df_ep["illegal_actions"].sum())
    main_holm_max = max(v["holm_p"] for v in scopes["Main_K12"]["pairwise_vs_cg"].values())
    all_holm_max = max(v["holm_p"] for v in scopes["All_24"]["pairwise_vs_cg"].values())
    print(
        f"[Batch 2] Done in {summary['elapsed_s']:.1f}s | rows={len(df_ep)} | "
        f"micro_capture={micro_summary['capture_ratio']:.4f} (inv={micro_summary['inversions']}) | "
        f"Main_K12 max holm_p={main_holm_max:.5f} | All_24 max holm_p={all_holm_max:.6f} | illegal={illegal_total}",
        flush=True,
    )
    if (
        illegal_total > 0
        or micro_summary["capture_ratio"] < 0.95
        or micro_summary["inversions"] > 0
        or main_holm_max >= 0.05
        or all_holm_max >= 0.05
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
