"""Step 5.4 (Batch 4): Azure 2019 Trace Replay, Cross-Hardware, Perturbations, P<=200 Scalability & Honest Boundaries.

Produces:
  - data_csv/b4a_azure_trace.csv
  - data_csv/b4b_cross_hw.csv
  - data_csv/b4b_perturb_sweep.csv
  - data_csv/b4c_scalability.csv
  - data_csv/b4d_theorem_boundaries.csv
  - stats/b4_summary.json
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from common_utils import (
    AZURE_EXTRACT_DIR,
    DATA_DIR,
    STATS_DIR,
    bootstrap_ci95,
    build_single_frozen_champion,
    holm_bonferroni,
    paired_wilcoxon_and_cliffs,
    sample_azure_tasks,
    save_csv_strict,
    save_json_strict,
    verify_upstream_readonly,
)

from baselines_stage2 import make_bambos_alp, make_chen_game, make_dgolb, make_edf_prebatch, make_sfa  # noqa: E402
import champion_core  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from exact_oracle import solve_single_tick_exact  # noqa: E402
from model_spec import (  # noqa: E402
    EdgeUplinkEnv,
    ScenarioV1,
    SimConfig,
    TaskSpec,
    build_scenario_v1,
    compute_dimensionless_numbers,
    make_micro_scenario,
    run_simulation,
    shannon_rate,
)


def _make_b4_policies(include_sfa: bool = True) -> list[tuple[str, Any]]:
    pols: list[tuple[str, Any]] = [
        ("CLWB_Dual", build_single_frozen_champion()),
        ("Bambos_ALP+CG", wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")),
        ("Chen_Game+CG", wrap_with_common_guard(make_chen_game(max_rounds=8), "Chen_Game+CG")),
        ("DGOLB+CG", wrap_with_common_guard(make_dgolb(), "DGOLB+CG")),
    ]
    if include_sfa:
        pols.insert(3, ("EDF_PreBatch+CG", wrap_with_common_guard(make_edf_prebatch(), "EDF_PreBatch+CG")))
        pols.append(("SFA+CG", wrap_with_common_guard(make_sfa(), "SFA+CG")))
    return pols


# --------------------------------------------------------------------------- #
# 4A: Azure Functions 2019 Production Trace Zero-Tuning Replay (64 scenarios)
# --------------------------------------------------------------------------- #

def run_4a_azure_trace() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    days = [f"{d:02d}" for d in range(1, 9)]
    windows = ["TR", "RM", "PK", "EV"]
    strata = ["FLEET", "BURST"]
    policies = _make_b4_policies(include_sfa=True)
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="wall")

    rows: list[dict[str, Any]] = []
    idx = 0
    for day in days:
        for win in windows:
            fpath = AZURE_EXTRACT_DIR / f"d{day}_{win}.npz"
            with np.load(fpath, allow_pickle=False) as data:
                pool = {k: data[k] for k in data.files}
            for stratum in strata:
                seed = 54100 + idx
                idx += 1
                rng = np.random.default_rng(seed)
                base_sc = build_scenario_v1(
                    case_id=f"AZ_d{day}_{win}_{stratum}",
                    family=f"AZURE_{stratum}",
                    seed=seed,
                    n_devices=28,
                    n_servers=5,
                    tasks_per_device=6,
                    offered_load=0.75,
                    T_max_s=0.70,
                    window_s=None,
                    radius_m=200.0,
                    input_kb_range=(75.0, 135.0),
                )
                n_tasks = base_sc.n_tasks  # 168
                t_3600, dur_s, raw_avg_ms = sample_azure_tasks(pool, stratum, n_tasks, rng)
                arr_s = np.sort(t_3600 / 3600.0 * base_sc.window_s)
                mean_dur = max(float(np.mean(dur_s)), 1e-12)
                base_total_c = sum(float(t.C_cycles) for t in base_sc.tasks)
                c_raw = np.clip((base_total_c / n_tasks) * (dur_s / mean_dur), 5.0e7, 3.0e9)
                c_arr = c_raw * (base_total_c / max(float(np.sum(c_raw)), 1.0))

                new_tasks: list[TaskSpec] = []
                for k_idx in range(n_tasks):
                    src_t = base_sc.tasks[k_idx]
                    dev_i = k_idx % base_sc.n_devices
                    k_dev = k_idx // base_sc.n_devices
                    h_home = float(base_sc.h_dev_ms[dev_i, int(base_sc.home_ms[dev_i])])
                    r_dn = shannon_rate(base_sc.phys.bandwidth_hz, base_sc.phys.p_ms_w * h_home, base_sc.phys.noise_w, 0.0)
                    tau_dn = float(src_t.B_out_bits / r_dn)
                    a_s = float(arr_s[k_idx])
                    new_tasks.append(
                        TaskSpec(
                            tid=f"{dev_i}:{k_dev}",
                            device=int(dev_i),
                            k=int(k_dev),
                            arrival_s=a_s,
                            deadline_s=a_s + base_sc.T_max_s,
                            D_bits=float(src_t.D_bits),
                            B_out_bits=float(src_t.B_out_bits),
                            C_cycles=float(c_arr[k_idx]),
                            tau_dn_s=tau_dn,
                        )
                    )
                new_tasks.sort(key=lambda t: (t.arrival_s, t.deadline_s, t.device, t.k))

                az_sc = ScenarioV1(
                    case_id=base_sc.case_id,
                    family=base_sc.family,
                    seed=base_sc.seed,
                    n_devices=base_sc.n_devices,
                    n_servers=base_sc.n_servers,
                    tasks=new_tasks,
                    ms_pos=base_sc.ms_pos,
                    dev_pos=base_sc.dev_pos,
                    home_ms=base_sc.home_ms,
                    d_ms=base_sc.d_ms,
                    h_dev_ms=base_sc.h_dev_ms,
                    f_loc_hz=base_sc.f_loc_hz,
                    f_ms_max_hz=base_sc.f_ms_max_hz,
                    p_tx_w=base_sc.p_tx_w,
                    subband=base_sc.subband,
                    phys=base_sc.phys,
                    T_max_s=base_sc.T_max_s,
                    window_s=base_sc.window_s,
                )
                dm = compute_dimensionless_numbers(az_sc, cfg)

                for pname, pfn in policies:
                    r = run_simulation(az_sc, pfn, cfg)
                    rows.append({
                        "scenario_id": az_sc.case_id,
                        "day": f"d{day}",
                        "window": win,
                        "stratum": stratum,
                        "seed": int(seed),
                        "raw_avg_ms": float(raw_avg_ms),
                        "kappa_ul": float(dm["kappa_ul"]),
                        "rho_cmp": float(dm["rho_cmp"]),
                        "window_s": float(az_sc.window_s),
                        "method": pname,
                        "n_tasks": int(r["n_tasks"]),
                        "ontime_rate": float(r["ontime_rate"]),
                        "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                        "wasted_srv_cycle_ratio": float(r["wasted_srv_cycle_ratio"]),
                        "decision_p50_ms": float(r["decision_p50_ms"]),
                        "decision_p99_ms": float(r["decision_p99_ms"]),
                        "skipped_ticks": int(r["skipped_ticks"]),
                        "illegal_actions": int(r["illegal_actions"]),
                    })
        print(f"[4A Azure] Completed day d{day}/d08", flush=True)

    df = pd.DataFrame(rows)
    method_names = [p[0] for p in policies]
    base_names = [m for m in method_names if m != "CLWB_Dual"]

    overall: dict[str, Any] = {}
    scen_order = df[df["method"] == "CLWB_Dual"]["scenario_id"].tolist()
    vecs: dict[str, np.ndarray] = {}
    for m in method_names:
        sub = df[df["method"] == m].set_index("scenario_id").loc[scen_order]
        vals = sub["ontime_rate"].to_numpy(dtype=float)
        vecs[m] = vals
        mv, lo, hi = bootstrap_ci95(vals, n_boot=10000, seed=54150)
        overall[m] = {
            "ontime_mean": mv,
            "ontime_ci95": [lo, hi],
            "miss_mean": float(sub["server_grant_miss_rate"].mean()),
            "p99_ms_max": float(sub["decision_p99_ms"].max()),
            "skipped_ticks_sum": int(sub["skipped_ticks"].sum()),
        }

    raw_ps = []
    cliffs = []
    for b in base_names:
        rp, cd = paired_wilcoxon_and_cliffs(vecs["CLWB_Dual"], vecs[b])
        raw_ps.append(rp)
        cliffs.append(cd)
    holm_ps = holm_bonferroni(raw_ps)
    pairwise = {}
    for i, b in enumerate(base_names):
        diff = (vecs["CLWB_Dual"] - vecs[b]) * 100.0
        dm, dlo, dhi = bootstrap_ci95(diff, n_boot=10000, seed=54180 + i)
        pairwise[b] = {
            "n_stat": len(scen_order),
            "delta_pp": dm,
            "delta_pp_ci95": [dlo, dhi],
            "wins": int(np.sum(diff > 1e-4)),
            "ties": int(np.sum(np.abs(diff) <= 1e-4)),
            "losses": int(np.sum(diff < -1e-4)),
            "raw_p": float(raw_ps[i]),
            "holm_p": float(holm_ps[i]),
            "cliffs_delta": float(cliffs[i]),
        }

    by_stratum: dict[str, Any] = {}
    for st in strata:
        by_stratum[st] = {}
        for m in method_names:
            vals = df[(df["stratum"] == st) & (df["method"] == m)]["ontime_rate"].to_numpy(dtype=float)
            mv, lo, hi = bootstrap_ci95(vals, n_boot=10000, seed=54190)
            by_stratum[st][m] = {"ontime_mean": mv, "ontime_ci95": [lo, hi]}

    by_window: dict[str, Any] = {}
    for w in windows:
        by_window[w] = {}
        for m in method_names:
            vals = df[(df["window"] == w) & (df["method"] == m)]["ontime_rate"].to_numpy(dtype=float)
            mv, lo, hi = bootstrap_ci95(vals, n_boot=10000, seed=54195)
            by_window[w][m] = {"ontime_mean": mv, "ontime_ci95": [lo, hi]}

    sub_clwb = df[df["method"] == "CLWB_Dual"]
    return rows, {
        "n_scenarios": len(scen_order),
        "mean_kappa_ul": float(sub_clwb["kappa_ul"].mean()),
        "mean_rho_cmp": float(sub_clwb["rho_cmp"].mean()),
        "mean_window_s": float(sub_clwb["window_s"].mean()),
        "overall": overall,
        "pairwise_vs_cg": pairwise,
        "by_stratum": by_stratum,
        "by_window": by_window,
    }


# --------------------------------------------------------------------------- #
# 4B: 3 Independent Hardware Profiles & CSI/Signaling Perturbation Sweeps
# --------------------------------------------------------------------------- #

_WORKLOAD_SPECS_6 = [
    dict(wid="WK02_DENSE_TIGHT", base_seed=41100, n=30, m=4, q=5, load=0.75, T=0.60, r=150.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), cyc=(1e8, 1e9), hetero=None),
    dict(wid="WK05_HETERO_DL", base_seed=41400, n=28, m=5, q=5, load=0.72, T=0.75, r=230.0, fdev=(0.65, 0.95), ptx=(0.8, 1.3), kb=(60.0, 120.0), cyc=(1e8, 1e9), hetero=(0.50, 1.50)),
    dict(wid="WK07_BURST_HEAVY", base_seed=41600, n=55, m=6, q=3, load=0.90, T=0.68, r=200.0, fdev=(0.60, 0.90), ptx=(0.8, 1.3), kb=(60.0, 120.0), cyc=(1e8, 1e9), hetero=None),
    dict(wid="WK09_SMALL_CELL_ASYM", base_seed=41800, n=32, m=4, q=4, load=0.70, T=0.65, r=120.0, fdev=(0.60, 0.95), ptx=(0.6, 1.8), kb=(60.0, 120.0), cyc=(1e8, 1e9), hetero=None),
    dict(wid="BL01_MIXED_AR_CTRL", base_seed=99000, n=28, m=6, q=5, load=0.72, T=0.65, r=260.0, fdev=(0.70, 0.90), ptx=(0.8, 1.3), kb=(35.0, 260.0), cyc=(5e7, 1.6e9), hetero=(0.50, 1.60)),
    dict(wid="BL02_NEAR_FAR_HEAVY", base_seed=99100, n=30, m=5, q=5, load=0.78, T=0.70, r=320.0, fdev=(0.70, 0.90), ptx=(0.4, 1.6), kb=(120.0, 240.0), cyc=(1e8, 1e9), hetero=(0.55, 1.50)),
]


def run_4b_cross_hw_and_perturb() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hw_profiles = [
        ("HW1_Macro5G", dict(tick_s=0.05, delta_sig_s=0.001, bandwidth_hz=10e6, f_ms_ghz=50.0, rho_sys=0.95, n_subbands=1, csi_sigma_db=0.0, fdev_override=None, r_override=None)),
        ("HW2_WiFi6_MicroMEC", dict(tick_s=0.025, delta_sig_s=0.0015, bandwidth_hz=20e6, f_ms_ghz=24.0, rho_sys=0.95, n_subbands=1, csi_sigma_db=0.0, fdev_override=(1.0, 1.4), r_override=100.0)),
        ("HW3_SmallCell_DualSB", dict(tick_s=0.05, delta_sig_s=0.002, bandwidth_hz=10e6, f_ms_ghz=36.0, rho_sys=0.95, n_subbands=2, csi_sigma_db=2.0, fdev_override=None, r_override=None)),
    ]
    pols_hw = _make_b4_policies(include_sfa=True)
    hw_rows: list[dict[str, Any]] = []

    for hw_name, hw in hw_profiles:
        for sp in _WORKLOAD_SPECS_6:
            for s_rep in range(10):
                seed = sp["base_seed"] + s_rep
                sc = build_scenario_v1(
                    case_id=f"{hw_name}_{sp['wid']}_S{s_rep}",
                    family=sp["wid"],
                    seed=seed,
                    n_devices=sp["n"],
                    n_servers=sp["m"],
                    tasks_per_device=sp["q"],
                    offered_load=sp["load"],
                    T_max_s=sp["T"],
                    radius_m=hw["r_override"] if hw["r_override"] is not None else sp["r"],
                    f_dev_range_ghz=hw["fdev_override"] if hw["fdev_override"] is not None else sp["fdev"],
                    f_ms_ghz=hw["f_ms_ghz"],
                    bandwidth_hz=hw["bandwidth_hz"],
                    p_tx_range_w=sp["ptx"],
                    cycles_range=sp["cyc"],
                    input_kb_range=sp["kb"],
                    hetero_deadline_range=sp["hetero"],
                    n_subbands=hw["n_subbands"],
                    rho_sys=hw["rho_sys"],
                )
                cfg = SimConfig(
                    tick_s=hw["tick_s"],
                    delta_sig_s=hw["delta_sig_s"],
                    rho_sys=hw["rho_sys"],
                    latency_mode="wall",
                    n_subbands=hw["n_subbands"],
                    csi_sigma_db=hw["csi_sigma_db"],
                    csi_seed=seed + 50000,
                )
                for pname, pfn in pols_hw:
                    r = run_simulation(sc, pfn, cfg)
                    hw_rows.append({
                        "hw_profile": hw_name,
                        "wid": sp["wid"],
                        "seed_idx": int(s_rep),
                        "seed": int(seed),
                        "method": pname,
                        "ontime_rate": float(r["ontime_rate"]),
                        "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                        "wasted_srv_cycle_ratio": float(r["wasted_srv_cycle_ratio"]),
                        "decision_p99_ms": float(r["decision_p99_ms"]),
                        "skipped_ticks": int(r["skipped_ticks"]),
                        "illegal_actions": int(r["illegal_actions"]),
                    })
        print(f"[4B CrossHW] Completed {hw_name}", flush=True)

    # Perturbation Sweeps on WK02_DENSE_TIGHT & BL01_MIXED_AR_CTRL (10 seeds)
    pols_pt = _make_b4_policies(include_sfa=False)
    pt_specs = [sp for sp in _WORKLOAD_SPECS_6 if sp["wid"] in ("WK02_DENSE_TIGHT", "BL01_MIXED_AR_CTRL")]
    pt_rows: list[dict[str, Any]] = []

    for sp in pt_specs:
        for s_rep in range(10):
            seed = sp["base_seed"] + s_rep
            sc = build_scenario_v1(
                case_id=f"PT_{sp['wid']}_S{s_rep}",
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
                cycles_range=sp["cyc"],
                input_kb_range=sp["kb"],
                hetero_deadline_range=sp["hetero"],
            )
            # CSI sweep
            for sig_db in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall", csi_sigma_db=sig_db, csi_seed=seed + 50000)
                for pname, pfn in pols_pt:
                    r = run_simulation(sc, pfn, cfg)
                    pt_rows.append({
                        "sweep_type": "csi_sigma_db",
                        "param_val": float(sig_db),
                        "wid": sp["wid"],
                        "seed_idx": int(s_rep),
                        "method": pname,
                        "ontime_rate": float(r["ontime_rate"]),
                        "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                        "decision_p99_ms": float(r["decision_p99_ms"]),
                        "skipped_ticks": int(r["skipped_ticks"]),
                        "illegal_actions": int(r["illegal_actions"]),
                    })
            # Signaling lag sweep (delta_sig_ms in [0.5, 1.0, 2.0, 4.0, 8.0])
            for dsig_ms in (0.5, 1.0, 2.0, 4.0, 8.0):
                cfg = SimConfig(tick_s=0.05, delta_sig_s=dsig_ms / 1000.0, latency_mode="wall", csi_sigma_db=0.0)
                for pname, pfn in pols_pt:
                    r = run_simulation(sc, pfn, cfg)
                    pt_rows.append({
                        "sweep_type": "delta_sig_ms",
                        "param_val": float(dsig_ms),
                        "wid": sp["wid"],
                        "seed_idx": int(s_rep),
                        "method": pname,
                        "ontime_rate": float(r["ontime_rate"]),
                        "server_grant_miss_rate": float(r["server_grant_miss_rate"]),
                        "decision_p99_ms": float(r["decision_p99_ms"]),
                        "skipped_ticks": int(r["skipped_ticks"]),
                        "illegal_actions": int(r["illegal_actions"]),
                    })

    df_hw = pd.DataFrame(hw_rows)
    hw_summary: dict[str, Any] = {}
    for hw_name, _ in hw_profiles:
        hw_summary[hw_name] = {}
        for pname, _ in pols_hw:
            sub = df_hw[(df_hw["hw_profile"] == hw_name) & (df_hw["method"] == pname)]
            mv, lo, hi = bootstrap_ci95(sub["ontime_rate"].to_numpy(dtype=float), n_boot=10000, seed=54200)
            hw_summary[hw_name][pname] = {
                "ontime_mean": mv,
                "ontime_ci95": [lo, hi],
                "miss_mean": float(sub["server_grant_miss_rate"].mean()),
                "p99_ms_max": float(sub["decision_p99_ms"].max()),
                "skipped_ticks_sum": int(sub["skipped_ticks"].sum()),
            }

    df_pt = pd.DataFrame(pt_rows)
    pt_summary: dict[str, Any] = {"csi_sigma_db": {}, "delta_sig_ms": {}}
    for stype in ("csi_sigma_db", "delta_sig_ms"):
        sub_s = df_pt[df_pt["sweep_type"] == stype]
        for pval, grp_p in sub_s.groupby("param_val", sort=True):
            pt_summary[stype][str(pval)] = {}
            for pname, grp_m in grp_p.groupby("method", sort=False):
                mv, lo, hi = bootstrap_ci95(grp_m["ontime_rate"].to_numpy(dtype=float), n_boot=10000, seed=54250)
                pt_summary[stype][str(pval)][str(pname)] = {
                    "ontime_mean": mv,
                    "ontime_ci95": [lo, hi],
                    "miss_mean": float(grp_m["server_grant_miss_rate"].mean()),
                }

    return hw_rows, pt_rows, {"cross_hw": hw_summary, "perturb_sweep": pt_summary}


# --------------------------------------------------------------------------- #
# 4C: Large-Scale Dual-Metric Scalability Audit (P in {10..200}, 10 seeds)
# --------------------------------------------------------------------------- #

def run_4c_scalability() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p_grid = [10, 25, 50, 75, 100, 150, 200]
    policies = [
        ("CLWB_Dual", build_single_frozen_champion()),
        ("Bambos_ALP+CG", wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")),
        ("Chen_Game+CG", wrap_with_common_guard(make_chen_game(max_rounds=8), "Chen_Game+CG")),
        ("DGOLB+CG", wrap_with_common_guard(make_dgolb(), "DGOLB+CG")),
        ("SFA+CG", wrap_with_common_guard(make_sfa(), "SFA+CG")),
    ]
    champ_pol = policies[0][1]
    rows: list[dict[str, Any]] = []

    gc.disable()
    try:
        for p_cnt in p_grid:
            thm2_bound = 2 * math.ceil(math.log2(max(1, p_cnt) + 1)) + 2 * 6 + 5
            for s_idx in range(10):
                seed = 54300 + s_idx
                sc = build_scenario_v1(
                    case_id=f"SCAL_P{p_cnt}_S{s_idx}",
                    family="scalability",
                    seed=seed,
                    n_devices=max(12, p_cnt),
                    n_servers=6,
                    tasks_per_device=1,
                    offered_load=0.80,
                    T_max_s=1.20,
                    radius_m=220.0,
                    window_s=0.03,
                )
                if p_cnt == 10:
                    sc.tasks = list(sc.tasks[:10])
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")

                # Measure CLWB_Dual unhooked first for clean wall-clock timings, then hooked for exact counts
                for pname, pfn in policies:
                    res = run_simulation(sc, pfn, cfg)
                    max_v_calls = 0
                    tot_v_calls = 0
                    tot_fluid_steps = 0
                    if pname == "CLWB_Dual":
                        orig_v = champion_core._fast_fluid_verify
                        orig_r = champion_core._compute_run_rates
                        tick_calls = 0
                        max_tick_c = 0
                        tot_c = 0
                        f_steps = 0

                        def wrapped_v(obs_a: dict, cand_a: list[dict], act_lag_s: float = 0.0) -> dict[str, Any]:
                            nonlocal tick_calls, tot_c
                            tick_calls += 1
                            tot_c += 1
                            return orig_v(obs_a, cand_a, act_lag_s)

                        def wrapped_r(*args: Any, **kwargs: Any) -> np.ndarray:
                            nonlocal f_steps
                            f_steps += 1
                            return orig_r(*args, **kwargs)

                        class _CountedWrap:
                            def __call__(self, obs: dict) -> dict:
                                nonlocal tick_calls, max_tick_c
                                tick_calls = 0
                                out = champ_pol(obs)
                                if tick_calls > max_tick_c:
                                    max_tick_c = tick_calls
                                return out

                        champion_core._fast_fluid_verify = wrapped_v
                        champion_core._compute_run_rates = wrapped_r
                        try:
                            run_simulation(sc, _CountedWrap(), SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))
                        finally:
                            champion_core._fast_fluid_verify = orig_v
                            champion_core._compute_run_rates = orig_r
                        max_v_calls = max_tick_c
                        tot_v_calls = tot_c
                        tot_fluid_steps = f_steps

                    rows.append({
                        "P": int(p_cnt),
                        "seed_idx": int(s_idx),
                        "seed": int(seed),
                        "method": pname,
                        "ontime_rate": float(res["ontime_rate"]),
                        "decision_p50_ms": float(res["decision_p50_ms"]),
                        "decision_p99_ms": float(res["decision_p99_ms"]),
                        "skipped_ticks": int(res["skipped_ticks"]),
                        "illegal_actions": int(res["illegal_actions"]),
                        "max_tick_verify_calls": int(max_v_calls),
                        "total_verify_calls": int(tot_v_calls),
                        "thm2_bound": int(thm2_bound),
                        "calls_to_bound_ratio": float(max_v_calls / max(1, thm2_bound)) if pname == "CLWB_Dual" else 0.0,
                        "fluid_event_steps": int(tot_fluid_steps),
                    })
            print(f"[4C Scalability] Completed P={p_cnt}", flush=True)
    finally:
        gc.enable()

    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for p_cnt in p_grid:
        summary[str(p_cnt)] = {}
        for pname, _ in policies:
            sub = df[(df["P"] == p_cnt) & (df["method"] == pname)]
            p99_m, p99_lo, p99_hi = bootstrap_ci95(sub["decision_p99_ms"].to_numpy(dtype=float), n_boot=10000, seed=54350)
            ont_m, ont_lo, ont_hi = bootstrap_ci95(sub["ontime_rate"].to_numpy(dtype=float), n_boot=10000, seed=54360)
            summary[str(p_cnt)][pname] = {
                "ontime_mean": ont_m,
                "ontime_ci95": [ont_lo, ont_hi],
                "p50_ms_mean": float(sub["decision_p50_ms"].mean()),
                "p99_ms_mean": p99_m,
                "p99_ms_ci95": [p99_lo, p99_hi],
                "p99_ms_max": float(sub["decision_p99_ms"].max()),
                "skipped_ticks_sum": int(sub["skipped_ticks"].sum()),
                "max_verify_calls": int(sub["max_tick_verify_calls"].max()),
                "thm2_bound": int(sub["thm2_bound"].max()),
                "max_calls_to_bound_ratio": float(sub["calls_to_bound_ratio"].max()),
                "mean_fluid_event_steps": float(sub["fluid_event_steps"].mean()),
            }
    return rows, summary


# --------------------------------------------------------------------------- #
# 4D: Theorem Premise Breakdown & Honest Boundary Archive FAIL-1..3
# --------------------------------------------------------------------------- #

def run_4d_theorem_boundaries(pt_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    champ = build_single_frozen_champion()
    pol_bambos = wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")

    # Curve 1: Theorem 1 zero-noise premise breakdown from pt_rows
    df_pt = pd.DataFrame(pt_rows)
    sub_t1 = df_pt[(df_pt["sweep_type"] == "csi_sigma_db") & (df_pt["method"] == "CLWB_Dual")]
    t1_curve: list[dict[str, Any]] = []
    for sig_db, grp in sub_t1.groupby("param_val", sort=True):
        om, olo, ohi = bootstrap_ci95(grp["ontime_rate"].to_numpy(dtype=float), n_boot=10000, seed=54410)
        mm, mlo, mhi = bootstrap_ci95(grp["server_grant_miss_rate"].to_numpy(dtype=float), n_boot=10000, seed=54415)
        rec = {
            "category": "curve1_thm1_csi_noise",
            "x_label": "csi_sigma_db",
            "x_val": float(sig_db),
            "method": "CLWB_Dual",
            "metric_name": "ontime_rate",
            "metric_val": om,
            "ci_lo": olo,
            "ci_hi": ohi,
            "aux_val": mm,
            "aux_name": "server_grant_miss_rate",
        }
        rows.append(rec)
        t1_curve.append({"sigma_db": float(sig_db), "ontime_mean": om, "miss_mean": mm})

    # Curve 2: Theorem 3 uniform-deadline premise breakdown (eta_inv in [0.0..1.0], 250 Anchor-1 trials each)
    t3_curve: list[dict[str, Any]] = []
    orig_perf = champion_core.time.perf_counter
    champion_core.time.perf_counter = lambda: 0.0
    try:
        for e_idx, eta in enumerate((0.0, 0.1, 0.25, 0.5, 0.75, 1.0)):
            rng = np.random.default_rng(54420 + e_idx)
            matches = []
            for _ in range(250):
                n = int(rng.integers(3, 7))
                h1 = np.full((n, 1), 1e-10, dtype=float)
                f_loc = np.full(n, 0.05e9, dtype=float)
                f_ms = np.full(1, float(rng.uniform(12.0, 26.0)) * 1e9, dtype=float)
                ptx = np.ones(n, dtype=float)
                cycles = rng.uniform(0.8e9, 4.5e9, size=n)
                base_dl = float(rng.uniform(0.30, 0.45))
                rand_dls = rng.uniform(0.22, 0.60, size=n)
                dls = (1.0 - eta) * base_dl + eta * rand_dls
                tasks_raw = [
                    {
                        "tid": f"{i}:0",
                        "device": i,
                        "k": 0,
                        "arrival_s": 0.0,
                        "deadline_s": float(dls[i]),
                        "D_bits": 800_000.0,
                        "C_cycles": float(cycles[i]),
                        "tau_dn_s": 0.001,
                    }
                    for i in range(n)
                ]
                sc = make_micro_scenario(
                    case_id=f"T3_ETA_{e_idx}",
                    n_devices=n,
                    n_servers=1,
                    h_dev_ms=h1,
                    f_loc_hz=f_loc,
                    f_ms_max_hz=f_ms,
                    p_tx_w=ptx,
                    tasks_raw=tasks_raw,
                )
                cf = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="zero")
                env = EdgeUplinkEnv(sc, cf)
                env._reveal_arrivals(0.0)
                obs = env.observe(0.0)
                oracle = solve_single_tick_exact(obs, max_p=n)
                got = len(champ(obs)["actions"])
                matches.append(1.0 if got == int(oracle["best_admitted"]) else 0.0)
            mm, mlo, mhi = bootstrap_ci95(matches, n_boot=10000, seed=54430)
            rows.append({
                "category": "curve2_thm3_deadline_dispersion",
                "x_label": "eta_inversion",
                "x_val": float(eta),
                "method": "CLWB_Dual",
                "metric_name": "oracle_match_rate",
                "metric_val": mm,
                "ci_lo": mlo,
                "ci_hi": mhi,
                "aux_val": float(1.0 - mm),
                "aux_name": "gap_rate",
            })
            t3_curve.append({"eta_inv": float(eta), "oracle_match_rate": mm, "gap_rate": float(1.0 - mm)})
    finally:
        champion_core.time.perf_counter = orig_perf

    # Curve 3: FAIL-1, FAIL-2, FAIL-3 quantitative archive (10 seeds for FAIL-1 and FAIL-3)
    f1_clean, f1_6db, f1_6db_miss = [], [], []
    for s_idx in range(10):
        sd = 96101 + s_idx
        sc1 = build_scenario_v1(
            case_id=f"FAIL1_CSI_6DB_S{s_idx}",
            family="fail_csi",
            seed=sd,
            n_devices=26,
            n_servers=6,
            tasks_per_device=5,
            offered_load=0.75,
            T_max_s=0.65,
            radius_m=240.0,
            input_kb_range=(80.0, 160.0),
        )
        r_c = run_simulation(sc1, champ, SimConfig(tick_s=0.05, delta_sig_s=0.001, csi_sigma_db=0.0, latency_mode="wall"))
        r_n = run_simulation(sc1, champ, SimConfig(tick_s=0.05, delta_sig_s=0.001, csi_sigma_db=6.0, csi_seed=sd, latency_mode="wall"))
        f1_clean.append(float(r_c["ontime_rate"]))
        f1_6db.append(float(r_n["ontime_rate"]))
        f1_6db_miss.append(float(r_n["server_grant_miss_rate"]))

    # FAIL-2: Sub-tick single-device burst
    tasks_f2 = [
        {
            "tid": f"0:{k}",
            "device": 0,
            "k": k,
            "arrival_s": 0.001 * k,
            "deadline_s": 0.08 + 0.005 * k,
            "D_bits": 150_000.0,
            "C_cycles": 2.0e8,
            "tau_dn_s": 0.001,
        }
        for k in range(5)
    ]
    sc2 = make_micro_scenario(
        case_id="FAIL2_SUBTICK_SAME_DEV",
        n_devices=1,
        n_servers=1,
        h_dev_ms=np.array([[1e-9]]),
        f_loc_hz=np.array([0.1e9]),
        f_ms_max_hz=np.array([50e9]),
        p_tx_w=np.array([1.0]),
        tasks_raw=tasks_f2,
    )
    r_f2_clwb = run_simulation(sc2, champ, SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))
    r_f2_alp = run_simulation(sc2, pol_bambos, SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))

    # FAIL-3: Zone I weak coupling (orthogonal=True, wall-clock)
    f3_clwb, f3_alp, f3_clwb_p50, f3_alp_p50 = [], [], [], []
    for s_idx in range(10):
        sd = 96301 + s_idx
        sc3 = build_scenario_v1(
            case_id=f"FAIL3_ZONE1_WEAK_S{s_idx}",
            family="fail_z1",
            seed=sd,
            n_devices=18,
            n_servers=6,
            tasks_per_device=5,
            offered_load=0.35,
            T_max_s=0.95,
            radius_m=180.0,
            input_kb_range=(8.0, 18.0),
        )
        cf3 = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="wall")
        rc3 = run_simulation(sc3, champ, cf3)
        ra3 = run_simulation(sc3, pol_bambos, cf3)
        f3_clwb.append(float(rc3["ontime_rate"]))
        f3_alp.append(float(ra3["ontime_rate"]))
        f3_clwb_p50.append(float(rc3["decision_p50_ms"]))
        f3_alp_p50.append(float(ra3["decision_p50_ms"]))

    fail_records = [
        ("FAIL-1_CSI_6DB", "CLWB_Dual_0dB", float(np.mean(f1_clean)), float(np.min(f1_clean)), float(np.max(f1_clean)), 0.0),
        ("FAIL-1_CSI_6DB", "CLWB_Dual_6dB", float(np.mean(f1_6db)), float(np.min(f1_6db)), float(np.max(f1_6db)), float(np.mean(f1_6db_miss))),
        ("FAIL-2_SUBTICK_DEV", "CLWB_Dual_SingleDevBurst", float(r_f2_clwb["ontime_rate"]), float(r_f2_clwb["ontime_rate"]), float(r_f2_clwb["ontime_rate"]), 0.0),
        ("FAIL-2_SUBTICK_DEV", "Bambos_ALP+CG_SingleDevBurst", float(r_f2_alp["ontime_rate"]), float(r_f2_alp["ontime_rate"]), float(r_f2_alp["ontime_rate"]), float(r_f2_alp["server_grant_miss_rate"])),
        ("FAIL-3_ZONE1_WEAK", "CLWB_Dual_Ortho", float(np.mean(f3_clwb)), float(np.min(f3_clwb)), float(np.max(f3_clwb)), float(np.mean(f3_clwb_p50))),
        ("FAIL-3_ZONE1_WEAK", "Bambos_ALP+CG_Ortho", float(np.mean(f3_alp)), float(np.min(f3_alp)), float(np.max(f3_alp)), float(np.mean(f3_alp_p50))),
    ]
    for cat, mlabel, val, lo, hi, aux in fail_records:
        rows.append({
            "category": "curve3_fail_archive",
            "x_label": cat,
            "x_val": 0.0,
            "method": mlabel,
            "metric_name": "ontime_rate",
            "metric_val": val,
            "ci_lo": lo,
            "ci_hi": hi,
            "aux_val": aux,
            "aux_name": "miss_or_p50_ms",
        })

    fail_summary = {
        "FAIL-1_CSI_6DB": {
            "clean_0db_ontime": float(np.mean(f1_clean)),
            "noisy_6db_ontime": float(np.mean(f1_6db)),
            "noisy_6db_miss": float(np.mean(f1_6db_miss)),
            "drop_pp": float((np.mean(f1_clean) - np.mean(f1_6db)) * 100.0),
            "boundary": "Recommended CSI noise <= 2-3 dB; >= 6 dB degrades deterministic fluid prediction.",
        },
        "FAIL-2_SUBTICK_SAME_DEV": {
            "clwb_ontime": float(r_f2_clwb["ontime_rate"]),
            "clwb_count": f"{r_f2_clwb['n_ontime']}/{sc2.n_tasks}",
            "alp_ontime": float(r_f2_alp["ontime_rate"]),
            "boundary": "Causal idle-radio gate caps per-device admissions at <= 1 per tick_s.",
        },
        "FAIL-3_ZONE1_WEAK_AND_INVERTED_DL": {
            "clwb_ontime": float(np.mean(f3_clwb)),
            "alp_ontime": float(np.mean(f3_alp)),
            "clwb_p50_ms": float(np.mean(f3_clwb_p50)),
            "alp_p50_ms": float(np.mean(f3_alp_p50)),
            "inverted_dl_gap_rate_eta1": float(t3_curve[-1]["gap_rate"]),
            "boundary": "Zone I orthogonal weak coupling needs no cross-layer solver (simpler ALP suffices at lower P50); inverted deadlines violate Theorem 3.",
        },
    }
    return rows, {"t1_csi_curve": t1_curve, "t3_deadline_curve": t3_curve, "fail_archive": fail_summary}


def main() -> None:
    verify_upstream_readonly()
    t0 = time.perf_counter()

    if "--only-4a" in sys.argv:
        print("[Batch 4] Running ONLY 4A (Azure Functions 2019 coupled Zone-III, 64 trace scenarios)...", flush=True)
        rows_4a, sum_4a = run_4a_azure_trace()
        save_csv_strict(rows_4a, DATA_DIR / "b4a_azure_trace.csv")
        summary_path = STATS_DIR / "b4_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["exp_4a_azure"] = sum_4a
        save_json_strict(summary, summary_path)
        illegal_4a = sum(int(r["illegal_actions"]) for r in rows_4a)
        elapsed = time.perf_counter() - t0
        print(
            f"[Batch 4A] Complete in {elapsed:.1f}s | kappa_ul={sum_4a['mean_kappa_ul']:.3f}, rho_cmp={sum_4a['mean_rho_cmp']:.3f} "
            f"| Azure CLWB={sum_4a['overall']['CLWB_Dual']['ontime_mean']*100:.2f}% | illegal={illegal_4a}",
            flush=True,
        )
        if illegal_4a > 0:
            raise SystemExit(1)
        return

    print("[Batch 4] Running 4A (Azure Functions 2019, 64 trace scenarios)...", flush=True)
    rows_4a, sum_4a = run_4a_azure_trace()
    save_csv_strict(rows_4a, DATA_DIR / "b4a_azure_trace.csv")

    print("[Batch 4] Running 4B (3 Hardware Profiles + CSI/Lag Sweeps, 10 seeds)...", flush=True)
    rows_4b_hw, rows_4b_pt, sum_4b = run_4b_cross_hw_and_perturb()
    save_csv_strict(rows_4b_hw, DATA_DIR / "b4b_cross_hw.csv")
    save_csv_strict(rows_4b_pt, DATA_DIR / "b4b_perturb_sweep.csv")

    print("[Batch 4] Running 4C (Dual-Metric Scalability P<=200, 10 seeds)...", flush=True)
    rows_4c, sum_4c = run_4c_scalability()
    save_csv_strict(rows_4c, DATA_DIR / "b4c_scalability.csv")

    print("[Batch 4] Running 4D (Theorem Premise Breakdown & FAIL-1..3)...", flush=True)
    rows_4d, sum_4d = run_4d_theorem_boundaries(rows_4b_pt)
    save_csv_strict(rows_4d, DATA_DIR / "b4d_theorem_boundaries.csv")

    elapsed = time.perf_counter() - t0
    summary = {
        "batch": 4,
        "elapsed_s": float(elapsed),
        "exp_4a_azure": sum_4a,
        "exp_4b_hw_and_perturb": sum_4b,
        "exp_4c_scalability": sum_4c,
        "exp_4d_boundaries": sum_4d,
    }
    save_json_strict(summary, STATS_DIR / "b4_summary.json")

    illegal_all = sum(int(r["illegal_actions"]) for r in rows_4a + rows_4b_hw + rows_4b_pt + rows_4c)
    print(
        f"[Batch 4] Complete in {elapsed:.1f}s | Azure CLWB={sum_4a['overall']['CLWB_Dual']['ontime_mean']*100:.2f}% "
        f"| max Azure holm_p={max(v['holm_p'] for v in sum_4a['pairwise_vs_cg'].values()):.6f} | illegal={illegal_all}",
        flush=True,
    )
    if illegal_all > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
