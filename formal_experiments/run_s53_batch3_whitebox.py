"""Step 5.3 (Batch 3): White-Box Explainability, Causal Grafting, Synergy, X-Ray & 2D Regime Map.

Produces:
  - data_csv/b3a_grafting.csv
  - data_csv/b3b_ablation_synergy.csv
  - data_csv/b3c_xray_traces.csv
  - data_csv/b3d_sensitivity_10s.csv
  - data_csv/b3e_regime_map_10s.csv
  - stats/b3_summary.json
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from common_utils import (
    DATA_DIR,
    STATS_DIR,
    bootstrap_ci95,
    build_single_frozen_champion,
    save_csv_strict,
    save_json_strict,
    unpack_suite_item,
    verify_upstream_readonly,
)

from baselines_stage2 import make_bambos_alp, make_chen_game, make_dgolb, make_edf_prebatch  # noqa: E402
from benchmark_suite_s3 import build_main_benchmark_k12  # noqa: E402
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
import champion_core  # noqa: E402
from champion_core import CLWBDualPolicy, make_clwb_dual  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from exact_oracle import build_counterexample_L1, build_counterexample_L2, build_counterexample_L3  # noqa: E402
from model_spec import SimConfig, build_scenario_v1, compute_dimensionless_numbers, run_simulation  # noqa: E402
from step41_blind_and_hardware import build_blind_suite_k8  # noqa: E402
from step42_anti_patchwork_and_grafting import ProbePolicy, make_grafted_baseline  # noqa: E402
from step43_phase_transition_scan import KnobScanPolicy  # noqa: E402


# --------------------------------------------------------------------------- #
# 3A: 4-Quadrant Cross-Baseline Bidirectional Grafting (10 seeds)
# --------------------------------------------------------------------------- #

def run_3a_grafting(
    main_k12: list[dict[str, Any]],
    blind_k8: list[tuple[str, list[tuple[Any, SimConfig]]]],
    ce_list: list[tuple[str, tuple[Any, SimConfig]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bl_focus = [x for x in blind_k8 if x[0] in ("BL01_MIXED_AR_CTRL", "BL02_NEAR_FAR_HEAVY")]
    workloads: list[tuple[str, str, list[tuple[Any, SimConfig]]]] = []
    for item in main_k12:
        wid, cells = unpack_suite_item(item)
        workloads.append(("Main_K12", wid, cells))
    for wid, cells in bl_focus:
        workloads.append(("Blind_Focus", wid, cells))

    baselines: list[tuple[str, Callable[[], Callable[[dict], dict]]]] = [
        ("Bambos_ALP", lambda: make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", lambda: make_chen_game(max_rounds=8)),
        ("DGOLB", make_dgolb),
        ("EDF_PreBatch", make_edf_prebatch),
    ]
    modes = ["raw", "cg", "cg_plus_fluid_contract", "cg_plus_full_core"]

    rows: list[dict[str, Any]] = []
    summary_table: dict[str, dict[str, Any]] = {}

    for bname, bfactory in baselines:
        summary_table[bname] = {}
        for mode in modes:
            pol = make_grafted_baseline(bname, bfactory, mode)
            k12_onts: list[float] = []
            k12_miss: list[float] = []
            bl_means: dict[str, float] = {}

            for grp, wid, cells in workloads:
                ep_o, ep_m, ep_w, ep_p99 = [], [], [], []
                for sc, cfg in cells:
                    r = run_simulation(sc, pol, cfg)
                    ep_o.append(float(r["ontime_rate"]))
                    ep_m.append(float(r["server_grant_miss_rate"]))
                    ep_w.append(float(r["wasted_srv_cycle_ratio"]))
                    ep_p99.append(float(r["decision_p99_ms"]))
                m_o, lo_o, hi_o = bootstrap_ci95(ep_o, n_boot=5000, seed=53100)
                m_ms = float(np.mean(ep_m))
                m_ws = float(np.mean(ep_w))
                rows.append({
                    "group_type": grp,
                    "wid": wid,
                    "baseline": bname,
                    "mode": mode,
                    "n_runs": int(len(cells)),
                    "ontime_mean": m_o,
                    "ontime_ci_lo": lo_o,
                    "ontime_ci_hi": hi_o,
                    "miss_mean": m_ms,
                    "wasted_mean": m_ws,
                    "p99_ms_max": float(np.max(ep_p99)),
                    "n_ontime_sum": 0,
                    "n_tasks_sum": 0,
                })
                if grp == "Main_K12":
                    k12_onts.append(m_o)
                    k12_miss.append(m_ms)
                else:
                    bl_means[wid] = m_o

            ce_res_map: dict[str, str] = {}
            for cname, (sc, cfg) in ce_list:
                r_ce = run_simulation(sc, pol, cfg)
                ce_res_map[cname] = f"{r_ce['n_ontime']}/{sc.n_tasks}"
                rows.append({
                    "group_type": "Counterexample",
                    "wid": cname,
                    "baseline": bname,
                    "mode": mode,
                    "n_runs": 1,
                    "ontime_mean": float(r_ce["ontime_rate"]),
                    "ontime_ci_lo": float(r_ce["ontime_rate"]),
                    "ontime_ci_hi": float(r_ce["ontime_rate"]),
                    "miss_mean": float(r_ce["server_grant_miss_rate"]),
                    "wasted_mean": float(r_ce["wasted_srv_cycle_ratio"]),
                    "p99_ms_max": float(r_ce["decision_p99_ms"]),
                    "n_ontime_sum": int(r_ce["n_ontime"]),
                    "n_tasks_sum": int(sc.n_tasks),
                })

            k12_m, k12_lo, k12_hi = bootstrap_ci95(k12_onts, n_boot=10000, seed=53150)
            summary_table[bname][mode] = {
                "k12_ontime_mean": k12_m,
                "k12_ontime_ci95": [k12_lo, k12_hi],
                "k12_miss_mean": float(np.mean(k12_miss)),
                "BL01_MIXED_AR_CTRL": bl_means["BL01_MIXED_AR_CTRL"],
                "BL02_NEAR_FAR_HEAVY": bl_means["BL02_NEAR_FAR_HEAVY"],
                "counterexamples": ce_res_map,
            }
            print(
                f"[3A Grafting] {bname:12s} | {mode:22s} | K12={k12_m*100:6.2f}% "
                f"miss={summary_table[bname][mode]['k12_miss_mean']*100:5.2f}% "
                f"BL01={bl_means['BL01_MIXED_AR_CTRL']*100:5.2f}% "
                f"CE2={ce_res_map['CE_2ndOrder']}",
                flush=True,
            )

    return rows, summary_table


# --------------------------------------------------------------------------- #
# 3B: Dual-Layer Ablation + 2x2 Super-Additive Synergy (10 seeds)
# --------------------------------------------------------------------------- #

def run_3b_ablation_synergy(
    main_k12: list[dict[str, Any]],
    blind_k8: list[tuple[str, list[tuple[Any, SimConfig]]]],
    ce_list: list[tuple[str, tuple[Any, SimConfig]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workloads: list[tuple[str, str, list[tuple[Any, SimConfig]]]] = []
    for item in main_k12:
        wid, cells = unpack_suite_item(item)
        workloads.append(("Main_K12", wid, cells))
    for wid, cells in blind_k8:
        workloads.append(("Blind_K8", wid, cells))

    variants: list[tuple[str, Callable[[], CLWBDualPolicy]]] = [
        ("Full", lambda: make_clwb_dual(name="Full")),
        ("LOO_no_ext", lambda: make_clwb_dual(use_externality_pricing=False, name="LOO_no_ext")),
        ("LOO_no_defer", lambda: make_clwb_dual(use_lagrangian_defer=False, name="LOO_no_defer")),
        ("LOO_no_fluid", lambda: make_clwb_dual(use_fluid_contraction=False, name="LOO_no_fluid")),
        ("sub_no_backfill", lambda: ProbePolicy(disable_backfill=True, name="sub_no_backfill")),
        ("sub_no_act_half_slack", lambda: ProbePolicy(disable_act_half_slack=True, name="sub_no_act_half_slack")),
        ("sub_no_edge_shield", lambda: ProbePolicy(disable_edge_shield=True, name="sub_no_edge_shield")),
        ("sub_no_local_first", lambda: ProbePolicy(disable_local_first=True, name="sub_no_local_first")),
        ("Baseline_S00", lambda: make_clwb_dual(use_externality_pricing=False, use_lagrangian_defer=False, name="Baseline_S00")),
    ]

    rows: list[dict[str, Any]] = []
    wk_ont_by_var: dict[str, dict[str, float]] = {}
    wk_miss_by_var: dict[str, dict[str, float]] = {}
    ce_by_var: dict[str, dict[str, str]] = {}

    for vname, vfactory in variants:
        pol = vfactory()
        wk_ont_by_var[vname] = {}
        wk_miss_by_var[vname] = {}
        for grp, wid, cells in workloads:
            ep_o, ep_m, ep_w = [], [], []
            for sc, cfg in cells:
                r = run_simulation(sc, pol, cfg)
                ep_o.append(float(r["ontime_rate"]))
                ep_m.append(float(r["server_grant_miss_rate"]))
                ep_w.append(float(r["wasted_srv_cycle_ratio"]))
            m_o = float(np.mean(ep_o))
            m_ms = float(np.mean(ep_m))
            m_ws = float(np.mean(ep_w))
            wk_ont_by_var[vname][wid] = m_o
            wk_miss_by_var[vname][wid] = m_ms
            rows.append({
                "variant": vname,
                "suite": grp,
                "wid": wid,
                "n_runs": int(len(cells)),
                "ontime_mean": m_o,
                "miss_mean": m_ms,
                "wasted_mean": m_ws,
                "n_ontime_sum": 0,
                "n_tasks_sum": 0,
            })

        ce_scores: dict[str, str] = {}
        for cname, (sc, cfg) in ce_list:
            r_ce = run_simulation(sc, pol, cfg)
            ce_scores[cname] = f"{r_ce['n_ontime']}/{sc.n_tasks}"
            rows.append({
                "variant": vname,
                "suite": "Counterexample",
                "wid": cname,
                "n_runs": 1,
                "ontime_mean": float(r_ce["ontime_rate"]),
                "miss_mean": float(r_ce["server_grant_miss_rate"]),
                "wasted_mean": float(r_ce["wasted_srv_cycle_ratio"]),
                "n_ontime_sum": int(r_ce["n_ontime"]),
                "n_tasks_sum": int(sc.n_tasks),
            })
        ce_by_var[vname] = ce_scores

    full_wk = wk_ont_by_var["Full"]
    full_miss = wk_miss_by_var["Full"]
    wids_20 = [wid for _, wid, _ in workloads]
    main_wids = [wid for grp, wid, _ in workloads if grp == "Main_K12"]
    blind_wids = [wid for grp, wid, _ in workloads if grp == "Blind_K8"]

    ablation_summary: dict[str, Any] = {}
    for vname, _ in variants:
        if vname == "Full":
            continue
        deltas_20 = [(full_wk[w] - wk_ont_by_var[vname][w]) * 100.0 for w in wids_20]
        deltas_main = [(full_wk[w] - wk_ont_by_var[vname][w]) * 100.0 for w in main_wids]
        deltas_blind = [(full_wk[w] - wk_ont_by_var[vname][w]) * 100.0 for w in blind_wids]
        g_m, g_lo, g_hi = bootstrap_ci95(deltas_20, n_boot=10000, seed=53200)
        miss_d_pp = float((np.mean([wk_miss_by_var[vname][w] for w in wids_20]) - np.mean([full_miss[w] for w in wids_20])) * 100.0)
        ablation_summary[vname] = {
            "ontime_20wk": float(np.mean([wk_ont_by_var[vname][w] for w in wids_20])),
            "gain_20wk_pp": g_m,
            "gain_20wk_ci95_pp": [g_lo, g_hi],
            "gain_main12_pp": float(np.mean(deltas_main)),
            "gain_blind8_pp": float(np.mean(deltas_blind)),
            "miss_delta_pp": miss_d_pp,
            "wk_helped": int(sum(1 for d in deltas_20 if d > 1e-4)),
            "wk_hurt": int(sum(1 for d in deltas_20 if d < -1e-4)),
            "ce": ce_by_var[vname],
        }
        print(
            f"[3B Ablation] {vname:22s} gain20={g_m:+6.2f}pp (Main={ablation_summary[vname]['gain_main12_pp']:+5.2f}, "
            f"Blind={ablation_summary[vname]['gain_blind8_pp']:+5.2f}) miss_d={miss_d_pp:+5.2f}pp CE2={ce_by_var[vname]['CE_2ndOrder']}",
            flush=True,
        )

    # 2x2 Synergy: Syn(Ext, Defer) = S11 - S10 - S01 + S00
    syn_per_wk = [
        (wk_ont_by_var["Full"][w] - wk_ont_by_var["LOO_no_defer"][w] - wk_ont_by_var["LOO_no_ext"][w] + wk_ont_by_var["Baseline_S00"][w]) * 100.0
        for w in wids_20
    ]
    syn_m, syn_lo, syn_hi = bootstrap_ci95(syn_per_wk, n_boot=10000, seed=53250)
    synergy_summary = {
        "S00_Baseline": float(np.mean([wk_ont_by_var["Baseline_S00"][w] for w in wids_20])),
        "S10_OnlyExt": float(np.mean([wk_ont_by_var["LOO_no_defer"][w] for w in wids_20])),
        "S01_OnlyDefer": float(np.mean([wk_ont_by_var["LOO_no_ext"][w] for w in wids_20])),
        "S11_Full": float(np.mean([wk_ont_by_var["Full"][w] for w in wids_20])),
        "synergy_20wk_pp": syn_m,
        "synergy_ci95_pp": [syn_lo, syn_hi],
        "ce_2nd_order": {
            "S00": ce_by_var["Baseline_S00"]["CE_2ndOrder"],
            "S10": ce_by_var["LOO_no_defer"]["CE_2ndOrder"],
            "S01": ce_by_var["LOO_no_ext"]["CE_2ndOrder"],
            "S11": ce_by_var["Full"]["CE_2ndOrder"],
        },
    }
    print(f"[3B Synergy] Syn(Ext, Defer) = {syn_m:+.2f} pp (95% CI [{syn_lo:+.2f}, {syn_hi:+.2f}])", flush=True)
    return rows, {"full_20wk_ontime": float(np.mean([full_wk[w] for w in wids_20])), "full_ce": ce_by_var["Full"], "ablation": ablation_summary, "synergy_2x2": synergy_summary}


# --------------------------------------------------------------------------- #
# 3C: White-Box Runtime X-Ray Telemetry (Non-Invasive Subclass Hook)
# --------------------------------------------------------------------------- #

class TelemetryCLWBPolicy(CLWBDualPolicy):
    """Non-invasive telemetry subclass logging internal state per decision tick."""

    def __init__(self, trace_id: str) -> None:
        super().__init__(name="CLWB_Dual_Telemetry")
        self.trace_id = trace_id
        self.records: list[dict[str, Any]] = []
        self._tick_idx = 0
        self._cur_n_local = 0
        self._cur_p_cand = 0
        self._cur_ext: np.ndarray = np.zeros(0, dtype=float)
        self._cur_eq_map: dict[str, float] = {}
        self._cur_verify_calls = 0
        self._cur_min_act_slack_s = 0.0

    def _admit_local_dominant(self, obs: dict, t_exec: float) -> tuple[list[dict], set[str]]:
        acts, loc_ids = super()._admit_local_dominant(obs, t_exec)
        self._cur_n_local = len(acts)
        return acts, loc_ids

    def _compute_two_flow_externality(
        self, obs, t_dev, t_sb, t_ptx, t_d, t_c, t_dl, t_tdn,
        sig_pm, i_base_pm, rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm, srv_rem,
    ) -> np.ndarray:
        eq = super()._compute_two_flow_externality(
            obs, t_dev, t_sb, t_ptx, t_d, t_c, t_dl, t_tdn,
            sig_pm, i_base_pm, rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm, srv_rem,
        )
        self._cur_ext = eq.copy()
        return eq

    def _build_candidate_order(
        self, obs: dict, edge_pool: list[dict], t_exec: float, srv_rem: np.ndarray, f_avail: np.ndarray,
    ) -> tuple[list[dict], np.ndarray]:
        cands, f_iso_pm = super()._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)
        self._cur_p_cand = len(cands)
        # Map candidate task IDs to their computed E_q values
        if self._cur_ext.size > 0:
            for c in cands:
                q_i = int(c["q_i"])
                if 0 <= q_i < self._cur_ext.size:
                    self._cur_eq_map[str(c["id"])] = float(self._cur_ext[q_i])
        return cands, f_iso_pm

    def _compute_active_slack_floor(
        self, obs: dict, verify_fn: Callable[[list[dict]], dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        act_min_slack, act_meta = super()._compute_active_slack_floor(obs, verify_fn)
        self._cur_min_act_slack_s = float(min(act_min_slack.values())) if act_min_slack else 0.0
        return act_min_slack, act_meta

    def __call__(self, obs: dict) -> dict:
        self._cur_n_local = 0
        self._cur_p_cand = 0
        self._cur_ext = np.zeros(0, dtype=float)
        self._cur_eq_map = {}
        self._cur_verify_calls = 0
        self._cur_min_act_slack_s = 0.0

        orig_verify = champion_core._fast_fluid_verify

        def counted_verify(obs_arg: dict, cand_arg: list[dict], act_lag_s: float = 0.0) -> dict[str, Any]:
            self._cur_verify_calls += 1
            return orig_verify(obs_arg, cand_arg, act_lag_s)

        champion_core._fast_fluid_verify = counted_verify
        try:
            out = super().__call__(obs)
        finally:
            champion_core._fast_fluid_verify = orig_verify

        acts = out.get("actions", [])
        n_edge = sum(1 for a in acts if a.get("mode") == "edge")
        n_local = sum(1 for a in acts if a.get("mode") == "local")
        n_pending = len(obs.get("pending", []))
        n_defer = max(0, n_pending - n_edge - n_local)

        if self._cur_ext.size > 0:
            max_eq = float(np.max(self._cur_ext))
            mean_eq = float(np.mean(self._cur_ext))
            gt1_cnt = int(np.sum(self._cur_ext > 1.0))
        else:
            max_eq, mean_eq, gt1_cnt = 0.0, 0.0, 0

        if self._cur_eq_map:
            top_tid = max(self._cur_eq_map, key=lambda k: self._cur_eq_map[k])
            top_eq = float(self._cur_eq_map[top_tid])
        else:
            top_tid = "none"
            top_eq = 0.0

        m_srvs = len(obs["servers"])
        thm2_bound = (2 * math.ceil(math.log2(max(1, self._cur_p_cand) + 1)) + 2 * m_srvs + 5) if self._cur_p_cand > 0 else 0
        f_avail_sum = sum(float(s["f_avail_hz"]) for s in obs["servers"])
        srv_res_sum = sum(float(s["cpu_reserved_hz"]) for s in obs["servers"])
        srv_res_ratio = float(srv_res_sum / max(f_avail_sum, 1.0))

        self.records.append({
            "trace_id": self.trace_id,
            "tick_idx": int(self._tick_idx),
            "time_s": float(obs["time"]),
            "n_pending": int(n_pending),
            "active_uplinks_cnt": int(len(obs["active_uplinks"])),
            "P_cand": int(self._cur_p_cand),
            "n_local": int(n_local),
            "n_edge": int(n_edge),
            "n_defer": int(n_defer),
            "max_E_q": max_eq,
            "mean_E_q": mean_eq,
            "gt1_cnt": int(gt1_cnt),
            "top_eq_tid": top_tid,
            "top_eq_val": top_eq,
            "eq_task_1_0": float(self._cur_eq_map.get("1:0", 0.0)),
            "verify_calls": int(self._cur_verify_calls),
            "thm2_bound": int(thm2_bound),
            "srv_res_ratio": srv_res_ratio,
            "min_act_slack_s": float(self._cur_min_act_slack_s),
        })
        self._tick_idx += 1
        return out


def run_3c_xray(
    main_k12: list[dict[str, Any]],
    blind_k8: list[tuple[str, list[tuple[Any, SimConfig]]]],
    ce2: tuple[Any, SimConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wk02_sc, wk02_cf = next(e["cells"][0] for e in main_k12 if e["wid"] == "WK02_DENSE_TIGHT")
    bl01_sc, bl01_cf = next(cells[0] for wid, cells in blind_k8 if wid == "BL01_MIXED_AR_CTRL")

    traces = [
        ("CE_2ndOrder", ce2[0], ce2[1]),
        ("WK02_DENSE_TIGHT_S0", wk02_sc, wk02_cf),
        ("BL01_MIXED_AR_CTRL_S0", bl01_sc, bl01_cf),
    ]
    all_recs: list[dict[str, Any]] = []
    trace_stats: dict[str, Any] = {}
    for tid, sc, cf in traces:
        pol = TelemetryCLWBPolicy(tid)
        res = run_simulation(sc, pol, cf)
        all_recs.extend(pol.records)
        max_calls = max((r["verify_calls"] for r in pol.records), default=0)
        max_bound = max((r["thm2_bound"] for r in pol.records), default=0)
        trace_stats[tid] = {
            "ontime_rate": float(res["ontime_rate"]),
            "n_ticks": int(len(pol.records)),
            "max_E_q": float(max((r["max_E_q"] for r in pol.records), default=0.0)),
            "max_verify_calls": int(max_calls),
            "max_thm2_bound": int(max_bound),
        }
    return all_recs, trace_stats


# --------------------------------------------------------------------------- #
# 3D: 5-Knob Dual-Extreme Sensitivity Scan (10 seeds)
# --------------------------------------------------------------------------- #

def run_3d_sensitivity(
    main_k12: list[dict[str, Any]],
    blind_k8: list[tuple[str, list[tuple[Any, SimConfig]]]],
    ce_l3: tuple[Any, SimConfig],
    ce_2: tuple[Any, SimConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s3_map = {e["wid"]: e["cells"] for e in main_k12}
    bl_map = dict(blind_k8)
    stress_wids = [
        "WK02_DENSE_TIGHT",
        "WK07_BURST_HEAVY",
        "WK09_SMALL_CELL_ASYM",
        "WK11_CSI_NOISE_2DB",
        "BL01_MIXED_AR_CTRL",
        "BL02_NEAR_FAR_HEAVY",
    ]
    stress_runs = [(w, s3_map[w] if w in s3_map else bl_map[w]) for w in stress_wids]

    scans = [
        ("H / tick_s", "h_mult", 2.0, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf], 20.0),
        ("beta_act", "beta_act", 0.5, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], 1.0),
        ("tau_ext", "tau_ext", 1.0, [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, math.inf], 10.0),
        ("alpha_shield", "alpha_shield", 1.0, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, math.inf], 8.0),
        ("k_fill", "k_fill", 14, [0, 1, 2, 4, 8, 14, 30, 100], 100.0),
    ]

    rows: list[dict[str, Any]] = []
    scan_summary: dict[str, Any] = {}
    dead_flags: list[str] = []

    for knob_title, kw, base_val, grid_vals, inf_plot_val in scans:
        pts_ont: list[float] = []
        ce_pairs: set[tuple[int, int]] = set()
        for val in grid_vals:
            pol = KnobScanPolicy(**{kw: val})
            ep_all: list[float] = []
            miss_all: list[float] = []
            per_wk: dict[str, float] = {}
            for wid, cells in stress_runs:
                os_w = []
                for sc, cfg in cells:
                    r = run_simulation(sc, pol, cfg)
                    ep_all.append(float(r["ontime_rate"]) * 100.0)
                    os_w.append(float(r["ontime_rate"]) * 100.0)
                    miss_all.append(float(r["server_grant_miss_rate"]) * 100.0)
                per_wk[wid] = float(np.mean(os_w))

            r_l3 = run_simulation(ce_l3[0], pol, ce_l3[1])
            r_c2 = run_simulation(ce_2[0], pol, ce_2[1])
            is_inf = isinstance(val, float) and not math.isfinite(val)
            m_pct, lo_pct, hi_pct = bootstrap_ci95(ep_all, n_boot=10000, seed=53400)
            pts_ont.append(m_pct)
            ce_pairs.add((int(r_l3["n_ontime"]), int(r_c2["n_ontime"])))
            rows.append({
                "knob": knob_title,
                "kw": kw,
                "value_str": "inf" if is_inf else str(val),
                "is_inf": bool(is_inf),
                "is_default": bool(val == base_val),
                "value_plot": float(inf_plot_val if is_inf else val),
                "ontime_mean_pct": m_pct,
                "ontime_ci95_lo_pct": lo_pct,
                "ontime_ci95_hi_pct": hi_pct,
                "miss_mean_pct": float(np.mean(miss_all)),
                "wk02_pct": per_wk["WK02_DENSE_TIGHT"],
                "wk11_csi2db_pct": per_wk["WK11_CSI_NOISE_2DB"],
                "bl01_pct": per_wk["BL01_MIXED_AR_CTRL"],
                "l3_ontime": int(r_l3["n_ontime"]),
                "l3_tasks": int(ce_l3[0].n_tasks),
                "ce2_ontime": int(r_c2["n_ontime"]),
                "ce2_tasks": int(ce_2[0].n_tasks),
            })

        spread_pp = float(max(pts_ont) - min(pts_ont))
        dead = bool(spread_pp < 0.30 and len(ce_pairs) == 1)
        if dead:
            dead_flags.append(knob_title)
        scan_summary[knob_title] = {
            "base_val": base_val,
            "spread_pp": spread_pp,
            "ce_unique_states": len(ce_pairs),
            "dead_flag": dead,
        }
        print(f"[3D Sensitivity] {knob_title:14s} spread={spread_pp:5.2f}pp ce_states={len(ce_pairs)} dead={dead}", flush=True)

    return rows, {"dead_line_flags": dead_flags, "knobs": scan_summary}


# --------------------------------------------------------------------------- #
# 3E: Formal 10-Seed 2D Mechanism Regime Map (6 x 6 x 10)
# --------------------------------------------------------------------------- #

def run_3e_regime_map() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_vals = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    kb_scales = [3.0, 22.0, 50.0, 80.0, 115.0, 155.0]
    n_reps = 10

    pol_clwb = build_single_frozen_champion()
    pol_bambos = wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")
    pol_chen = wrap_with_common_guard(make_chen_game(max_rounds=8), "Chen_Game+CG")
    pol_dgolb = wrap_with_common_guard(make_dgolb(), "DGOLB+CG")

    rows: list[dict[str, Any]] = []
    diff_bambos_grid = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    diff_dgolb_grid = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    kappa_rows = np.zeros((len(kb_scales), len(load_vals)), dtype=float)

    for r_i, kb_mid in enumerate(kb_scales):
        for c_i, ld in enumerate(load_vals):
            k_acc, r_acc = [], []
            c_acc, b_acc, ch_acc, d_acc = [], [], [], []
            for rep in range(n_reps):
                sc = build_scenario_v1(
                    case_id=f"REG3E_{r_i}_{c_i}_{rep}",
                    family="b3e_regime_map",
                    seed=53500 + rep,
                    n_devices=24,
                    n_servers=6,
                    tasks_per_device=5,
                    offered_load=ld,
                    T_max_s=0.75,
                    radius_m=220.0,
                    input_kb_range=(0.75 * kb_mid, 1.25 * kb_mid),
                )
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
                dm = compute_dimensionless_numbers(sc, cfg)
                rc = run_simulation(sc, pol_clwb, cfg)
                rb = run_simulation(sc, pol_bambos, cfg)
                rch = run_simulation(sc, pol_chen, cfg)
                rd = run_simulation(sc, pol_dgolb, cfg)
                k_acc.append(float(dm["kappa_ul"]))
                r_acc.append(float(dm["rho_cmp"]))
                c_acc.append(100.0 * float(rc["ontime_rate"]))
                b_acc.append(100.0 * float(rb["ontime_rate"]))
                ch_acc.append(100.0 * float(rch["ontime_rate"]))
                d_acc.append(100.0 * float(rd["ontime_rate"]))

            c_arr = np.asarray(c_acc)
            b_arr = np.asarray(b_acc)
            ch_arr = np.asarray(ch_acc)
            d_arr = np.asarray(d_acc)
            db_m, db_lo, db_hi = bootstrap_ci95(c_arr - b_arr, n_boot=5000, seed=53510)
            dd_m, dd_lo, dd_hi = bootstrap_ci95(c_arr - d_arr, n_boot=5000, seed=53520)

            kappa_m = float(np.mean(k_acc))
            rho_m = float(np.mean(r_acc))
            kappa_rows[r_i, c_i] = kappa_m
            diff_bambos_grid[r_i, c_i] = db_m
            diff_dgolb_grid[r_i, c_i] = dd_m

            rows.append({
                "r_idx": int(r_i),
                "c_idx": int(c_i),
                "kb_mid": float(kb_mid),
                "offered_load": float(ld),
                "kappa_ul_mean": kappa_m,
                "rho_cmp_mean": rho_m,
                "clwb_ontime_pct": float(np.mean(c_arr)),
                "bambos_ontime_pct": float(np.mean(b_arr)),
                "chen_ontime_pct": float(np.mean(ch_arr)),
                "dgolb_ontime_pct": float(np.mean(d_arr)),
                "diff_bambos_pp": db_m,
                "diff_bambos_ci_lo": db_lo,
                "diff_bambos_ci_hi": db_hi,
                "diff_chen_pp": float(np.mean(c_arr - ch_arr)),
                "diff_dgolb_pp": dd_m,
                "diff_dgolb_ci_lo": dd_lo,
                "diff_dgolb_ci_hi": dd_hi,
            })

    k_row_means = np.mean(kappa_rows, axis=1)
    summary = {
        "grid_shape": [len(kb_scales), len(load_vals)],
        "n_reps": int(n_reps),
        "zone1_mean_diff_bambos_pp": float(np.mean(diff_bambos_grid[k_row_means <= 0.12])),
        "zone3_mean_diff_bambos_pp": float(np.mean(diff_bambos_grid[k_row_means > 0.12])),
        "max_gain_vs_bambos_pp": float(np.max(diff_bambos_grid)),
        "min_gain_vs_bambos_pp": float(np.min(diff_bambos_grid)),
        "max_gain_vs_dgolb_pp": float(np.max(diff_dgolb_grid)),
    }
    print(
        f"[3E RegimeMap] Zone1 diff vs ALP={summary['zone1_mean_diff_bambos_pp']:+.2f}pp | "
        f"Zone3 diff vs ALP={summary['zone3_mean_diff_bambos_pp']:+.2f}pp | "
        f"max vs DGOLB={summary['max_gain_vs_dgolb_pp']:+.2f}pp",
        flush=True,
    )
    return rows, summary


def main() -> None:
    verify_upstream_readonly()
    t0 = time.perf_counter()

    main_k12 = build_main_benchmark_k12(n_seeds=10)
    blind_k8 = build_blind_suite_k8(n_seeds=10)
    ce_l1 = build_counterexample_L1()
    ce_l2 = build_counterexample_L2()
    ce_l3 = build_counterexample_L3()
    ce_2 = build_counterexample_2nd_order()
    ce_list = [
        ("L1", ce_l1),
        ("L2", ce_l2),
        ("L3", ce_l3),
        ("CE_2ndOrder", ce_2),
    ]

    print("[Batch 3] Running 3A (4-Quadrant Grafting, 10 seeds)...", flush=True)
    rows_3a, sum_3a = run_3a_grafting(main_k12, blind_k8, ce_list)
    save_csv_strict(rows_3a, DATA_DIR / "b3a_grafting.csv")

    print("[Batch 3] Running 3B (Dual-Layer Ablation + 2x2 Synergy, 10 seeds)...", flush=True)
    rows_3b, sum_3b = run_3b_ablation_synergy(main_k12, blind_k8, ce_list)
    save_csv_strict(rows_3b, DATA_DIR / "b3b_ablation_synergy.csv")

    print("[Batch 3] Running 3C (White-Box X-Ray Telemetry)...", flush=True)
    rows_3c, sum_3c = run_3c_xray(main_k12, blind_k8, ce_2)
    save_csv_strict(rows_3c, DATA_DIR / "b3c_xray_traces.csv")

    print("[Batch 3] Running 3D (5-Knob Dual-Extreme Sensitivity, 10 seeds)...", flush=True)
    rows_3d, sum_3d = run_3d_sensitivity(main_k12, blind_k8, ce_l3, ce_2)
    save_csv_strict(rows_3d, DATA_DIR / "b3d_sensitivity_10s.csv")

    print("[Batch 3] Running 3E (10-Seed 2D Regime Map 6x6x10)...", flush=True)
    rows_3e, sum_3e = run_3e_regime_map()
    save_csv_strict(rows_3e, DATA_DIR / "b3e_regime_map_10s.csv")

    elapsed = time.perf_counter() - t0
    summary = {
        "batch": 3,
        "elapsed_s": float(elapsed),
        "exp_3a_grafting": sum_3a,
        "exp_3b_ablation_synergy": sum_3b,
        "exp_3c_xray_traces": sum_3c,
        "exp_3d_sensitivity": sum_3d,
        "exp_3e_regime_map": sum_3e,
    }
    save_json_strict(summary, STATS_DIR / "b3_summary.json")
    print(f"[Batch 3] Complete in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
