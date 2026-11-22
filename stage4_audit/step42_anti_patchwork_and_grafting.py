"""Stage 4 Step 4.2: Anti-Patchwork Coupling Matrix, Fine-Grained Sub-Ablation & Cross-Baseline Grafting.

Tests:
  Part A (Anti-Patchwork Matrix & Fine-Grained Sub-Component Necessity):
    - 20 workloads (12 Stage-3 + 8 Blind K=8, 2 seeds each = 40 episodes) + 4 Counterexamples (L1, L2, L3, CE_2ndOrder)
    - 3 Macro LOO variants + 7 Fine-Grained sub-component ablations:
        * sub_no_active_dmg      : zero Delta_u^act(q) in E_q
        * sub_no_srv_cost        : pure wireless server choice (ignore CPU share term)
        * sub_no_backfill        : stop after O(log P) prefix contraction (no 2M+2 fill)
        * sub_no_alt_srv_only    : keep same-server spatial reuse, forbid alternate server switch
        * sub_no_act_half_slack  : set active link min-slack floor to 0.0 (deadline-only)
        * sub_no_edge_shield     : allocate exact f_base (zero non-hoarding timing shield)
        * sub_no_local_first     : bypass zero-interference local dominant knapsack
    - Computes patchwork exclusivity index & explicitly flags any zero-gain / negative-synergy sub-component.

  Part B (4-Quadrant Cross-Baseline Bidirectional Grafting):
    - External strong & classic baselines: DGOLB, Bambos_ALP, Chen_Game, EDF_PreBatch
    - Quadrants: Raw -> +CommonGuard -> +CG+FluidContract -> +CG+FullOurCore
    - Evaluated across K=12 (2 seeds) + BL01_MIXED_AR_CTRL + BL02_NEAR_FAR_HEAVY + CE_2ndOrder.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

STAGE4_DIR = Path(__file__).resolve().parent
ROOT_V3 = STAGE4_DIR.parent
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"

for p in (str(STAGE4_DIR), str(STAGE3_DIR), str(STAGE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_stage2 import make_bambos_alp, make_chen_game, make_dgolb, make_edf_prebatch
from benchmark_suite_s3 import build_main_benchmark_k12
from candidates_routes import build_counterexample_2nd_order
from champion_core import CLWBDualPolicy, _fast_fluid_verify, make_clwb_dual
from common_guard import wrap_with_common_guard
from exact_oracle import build_counterexample_L1, build_counterexample_L2, build_counterexample_L3
from model_spec import SimConfig, run_simulation
from step41_blind_and_hardware import build_blind_suite_k8


# --------------------------------------------------------------------------- #
# Part A: Fine-grained surgical ablation subclasses (non-invasive wrappers)
# --------------------------------------------------------------------------- #

class ProbePolicy(CLWBDualPolicy):
    """Surgical probe wrapper toggling single fine-grained sub-components cleanly."""

    def __init__(
        self,
        *,
        disable_backfill: bool = False,
        disable_act_half_slack: bool = False,
        disable_edge_shield: bool = False,
        disable_local_first: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.disable_backfill = disable_backfill
        self.disable_act_half_slack = disable_act_half_slack
        self.disable_edge_shield = disable_edge_shield
        self.disable_local_first = disable_local_first

    def _admit_local_dominant(self, obs: dict, t_exec: float) -> tuple[list[dict], set[str]]:
        if self.disable_local_first:
            return [], set()
        return super()._admit_local_dominant(obs, t_exec)

    def _compute_active_slack_floor(self, obs: dict, verify_fn: Any) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        mins, metas = super()._compute_active_slack_floor(obs, verify_fn)
        if self.disable_act_half_slack:
            return {k: 0.0 for k in mins}, metas
        return mins, metas

    def _contract_prefix(self, cands: list[dict], check_fn: Any) -> tuple[int, dict[str, Any]]:
        k_pref, pref_res = super()._contract_prefix(cands, check_fn)
        self._in_backfill_phase = True
        return k_pref, pref_res

    def _is_batch_admissible(
        self, obs: dict, cand_list: list[dict], check_horizon: bool,
        act_min_slack: dict[str, float], act_meta: dict[str, tuple[float, float]], verify_fn: Any,
    ) -> tuple[bool, dict[str, Any]]:
        if getattr(self, "_in_backfill_phase", False) and self.disable_backfill:
            return False, {}
        return super()._is_batch_admissible(obs, cand_list, check_horizon, act_min_slack, act_meta, verify_fn)

    def _hereditary_select_grants(
        self, obs: dict, cands: list[dict], f_iso_pm: np.ndarray, srv_rem: np.ndarray, t_enter: float,
    ) -> tuple[list[dict], dict[str, Any]]:
        if not self.disable_backfill:
            return super()._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)

        self._in_backfill_phase = False
        try:
            return super()._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)
        finally:
            self._in_backfill_phase = False

    def _allocate_shielded_edge(
        self, obs: dict, kept_grants: list[dict], fluid_res: dict[str, Any], srv_res: np.ndarray, f_avail: np.ndarray,
    ) -> list[dict]:
        acts = super()._allocate_shielded_edge(obs, kept_grants, fluid_res, srv_res, f_avail)
        if self.disable_edge_shield:
            f_min_hz = float(obs["f_min_hz"])
            f_min_map = fluid_res.get("f_min", {})
            for a in acts:
                tid = str(a.get("task", a.get("id")))
                if tid in f_min_map and math.isfinite(float(f_min_map[tid])):
                    a["f_hz"] = max(f_min_hz, float(f_min_map[tid]))
        return acts


def run_part_a() -> dict[str, Any]:
    s3_suite = build_main_benchmark_k12(n_seeds=2)
    bl_suite = build_blind_suite_k8(n_seeds=2)
    workloads: list[tuple[str, str, list[tuple[Any, SimConfig]]]] = []
    for entry in s3_suite:
        workloads.append((entry["wid"], "stage3_k12", entry["cells"]))
    for wid, runs in bl_suite:
        workloads.append((wid, "blind_k8", runs))

    ce_list = [
        ("L1", build_counterexample_L1()),
        ("L2", build_counterexample_L2()),
        ("L3", build_counterexample_L3()),
        ("CE_2ndOrder", build_counterexample_2nd_order()),
    ]

    variants: list[tuple[str, Callable[[], CLWBDualPolicy]]] = [
        ("CLWB_Dual_Full", lambda: make_clwb_dual(name="CLWB_Dual_Full")),
        # 3 Macro LOO
        ("LOO_no_ext", lambda: make_clwb_dual(use_externality_pricing=False, name="LOO_no_ext")),
        ("LOO_no_defer", lambda: make_clwb_dual(use_lagrangian_defer=False, name="LOO_no_defer")),
        ("LOO_no_fluid", lambda: make_clwb_dual(use_fluid_contraction=False, name="LOO_no_fluid")),
        # 4 Fine-grained sub-component probes (remaining after Stage-4 physical slimming)
        ("sub_no_backfill", lambda: ProbePolicy(disable_backfill=True, name="sub_no_backfill")),
        ("sub_no_act_half_slack", lambda: ProbePolicy(disable_act_half_slack=True, name="sub_no_act_half_slack")),
        ("sub_no_edge_shield", lambda: ProbePolicy(disable_edge_shield=True, name="sub_no_edge_shield")),
        ("sub_no_local_first", lambda: ProbePolicy(disable_local_first=True, name="sub_no_local_first")),
    ]

    results: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    for vname, vfactory in variants:
        pol = vfactory()
        wk_scores: dict[str, float] = {}
        miss_all: list[float] = []
        waste_all: list[float] = []
        s3_vals, bl_vals = [], []
        for wid, grp, runs in workloads:
            onts = []
            for sc, cfg in runs:
                r = run_simulation(sc, pol, cfg)
                onts.append(float(r["ontime_rate"]))
                miss_all.append(float(r["server_grant_miss_rate"]))
                waste_all.append(float(r["wasted_srv_cycle_ratio"]))
            mval = float(np.mean(onts))
            wk_scores[wid] = mval
            if grp == "stage3_k12":
                s3_vals.append(mval)
            else:
                bl_vals.append(mval)

        ce_scores = {}
        for cname, (sc, cfg) in ce_list:
            r = run_simulation(sc, pol, cfg)
            ce_scores[cname] = f"{r['n_ontime']}/{sc.n_tasks}"

        results[vname] = {
            "mean_20": float(np.mean(s3_vals + bl_vals)),
            "mean_s3_k12": float(np.mean(s3_vals)),
            "mean_blind_k8": float(np.mean(bl_vals)),
            "mean_miss": float(np.mean(miss_all)),
            "mean_wasted": float(np.mean(waste_all)),
            "ce": ce_scores,
            "per_workload": wk_scores,
        }

    # Summarize deltas relative to CLWB_Dual_Full
    full = results["CLWB_Dual_Full"]
    summary: dict[str, Any] = {}
    flagged_non_positive: list[str] = []
    for vname, data in results.items():
        if vname == "CLWB_Dual_Full":
            continue
        deltas_all = [full["per_workload"][w] - data["per_workload"][w] for w in full["per_workload"]]
        deltas_s3 = [full["per_workload"][w] - data["per_workload"][w] for w, g, _ in workloads if g == "stage3_k12"]
        deltas_bl = [full["per_workload"][w] - data["per_workload"][w] for w, g, _ in workloads if g == "blind_k8"]
        n_pos = int(sum(1 for d in deltas_all if d > 1e-4))
        n_neg = int(sum(1 for d in deltas_all if d < -1e-4))
        ce_drops = [
            c for c in full["ce"]
            if int(str(data["ce"][c]).split("/")[0]) < int(str(full["ce"][c]).split("/")[0])
        ]
        mean_gain_pp = float(np.mean(deltas_all) * 100.0)
        summary[vname] = {
            "gain_20wk_pp": mean_gain_pp,
            "gain_s3_k12_pp": float(np.mean(deltas_s3) * 100.0),
            "gain_blind_k8_pp": float(np.mean(deltas_bl) * 100.0),
            "miss_delta_pp": float((data["mean_miss"] - full["mean_miss"]) * 100.0),
            "wk_helped": n_pos,
            "wk_hurt": n_neg,
            "ce_drops": ce_drops,
            "ce": data["ce"],
            "worst_hurt_pp": float(min(deltas_all) * 100.0),
            "best_help_pp": float(max(deltas_all) * 100.0),
        }
        if mean_gain_pp <= 0.0 and not ce_drops and (data["mean_miss"] - full["mean_miss"]) * 100.0 <= 0.05:
            flagged_non_positive.append(vname)
        print(
            f"[PartA] {vname:24s} gain20={mean_gain_pp:+6.2f}pp (S3={summary[vname]['gain_s3_k12_pp']:+6.2f}, BL={summary[vname]['gain_blind_k8_pp']:+6.2f}) "
            f"miss_d={summary[vname]['miss_delta_pp']:+5.2f}pp helped/hurt={n_pos}/{n_neg} ce_drops={ce_drops}",
            flush=True,
        )

    return {
        "elapsed_s": float(time.perf_counter() - t0),
        "full": full,
        "summary": summary,
        "variants_raw": results,
        "flagged_non_positive": flagged_non_positive,
    }


# --------------------------------------------------------------------------- #
# Part B: 4-Quadrant Cross-Baseline Bidirectional Grafting
# --------------------------------------------------------------------------- #

def make_grafted_baseline(
    base_name: str,
    base_factory: Callable[[], Callable[[dict], dict]],
    mode: str,
) -> Callable[[dict], dict]:
    """Build Q1 (raw), Q2 (+CG), Q3a (+CG+FluidContract), Q3b (+CG+FullCore_ExtRank)."""
    if mode == "raw":
        return base_factory()
    if mode == "cg":
        return wrap_with_common_guard(base_factory(), f"{base_name}+CG")

    base_pol = base_factory()
    champ = make_clwb_dual()

    def grafted(obs: dict) -> dict:
        t_enter = time.perf_counter()
        raw = base_pol(obs)
        pend_by_id = {str(p["id"]): p for p in obs["pending"]}
        devs = obs["devices"]
        srvs = obs["servers"]
        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        srv_rem = np.maximum(f_avail - np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float), float(obs["f_min_hz"]))

        t_exec = float(obs["time"]) + float(obs["delta_sig_s"])
        # First preserve local decisions on idle devices via CLWB local-first
        loc_acts, loc_ids = champ._admit_local_dominant(obs, t_exec)

        # Collect baseline's preferred edge targets among non-local idle-radio devices (1 per device)
        base_srv_pref: dict[str, int] = {}
        base_order_rank: dict[str, int] = {}
        raw_actions = raw.get("actions", [])
        for idx, g in enumerate(raw_actions):
            tid = str(g.get("task", g.get("id")))
            md = str(g.get("mode", ""))
            if md == "edge" and tid not in base_srv_pref and g.get("server") is not None:
                base_order_rank[tid] = idx
                base_srv_pref[tid] = int(g["server"])
            elif md == "local" and tid not in loc_ids and tid not in base_order_rank:
                base_order_rank[tid] = len(base_order_rank)

        # Idle-radio pool: Q3a filters strictly baseline's proposed edge grants; Q3b opens full idle-radio pool with E_q + baseline tie-break.
        best_per_dev: dict[int, tuple[tuple[int, int, float, str], dict]] = {}
        for p in obs["pending"]:
            tid = str(p["id"])
            if tid in loc_ids:
                continue
            if mode == "cg_plus_fluid_contract" and tid not in base_srv_pref:
                continue
            di = int(p["device"])
            if bool(devs[di]["radio_busy"]) or float(devs[di]["tx_queue_bits"]) > 1e-6:
                continue
            rk = (0 if tid in base_srv_pref else 1, base_order_rank.get(tid, 10**6), float(p["deadline_s"]), tid)
            if di not in best_per_dev or rk < best_per_dev[di][0]:
                best_per_dev[di] = (rk, p)
        edge_pool = [v[1] for v in best_per_dev.values()]
        if not edge_pool:
            return {"actions": list(loc_acts)}

        cands_all, f_iso_all = champ._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)
        if not cands_all:
            return {"actions": list(loc_acts)}
        if mode == "cg_plus_full_core":
            cands = cands_all
            f_iso_pm = f_iso_all
            for c in cands:
                tid = str(c["id"])
                if tid in base_srv_pref:
                    bj = base_srv_pref[tid]
                    if bj in c.get("feas_js", []):
                        c["server"] = bj
        else:
            # Q3a: +CG + FluidContract ONLY (retain baseline's ordering and preferred server!)
            cands = sorted(
                cands_all,
                key=lambda c: (
                    0 if str(c["id"]) in base_srv_pref else 1,
                    base_order_rank.get(str(c["id"]), 10**6),
                    float(c["deadline_s"]),
                    str(c["id"]),
                ),
            )
            f_iso_pm = f_iso_all
            for c in cands:
                tid = str(c["id"])
                if tid in base_srv_pref:
                    bj = base_srv_pref[tid]
                    if 0 <= bj < len(srvs):
                        c["server"] = bj
                        feas = [int(j) for j in c.get("feas_js", [])]
                        c["feas_js"] = [bj] + [j for j in feas if j != bj]

        kept, fluid_res = champ._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        edge_acts = champ._allocate_shielded_edge(obs, kept, fluid_res, srv_res, f_avail)
        return {"actions": list(loc_acts) + list(edge_acts)}

    return wrap_with_common_guard(grafted, f"{base_name}+{mode}")


def run_part_b() -> dict[str, Any]:
    s3_suite = build_main_benchmark_k12(n_seeds=2)
    bl_suite = build_blind_suite_k8(n_seeds=3)
    bl_focus = [x for x in bl_suite if x[0] in ("BL01_MIXED_AR_CTRL", "BL02_NEAR_FAR_HEAVY")]
    ce_2nd = build_counterexample_2nd_order()

    baselines = [
        ("Bambos_ALP", lambda: make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", lambda: make_chen_game(max_rounds=8)),
        ("DGOLB", make_dgolb),
        ("EDF_PreBatch", make_edf_prebatch),
    ]
    modes = ["raw", "cg", "cg_plus_fluid_contract", "cg_plus_full_core"]

    table: dict[str, dict[str, Any]] = {}
    for bname, bfactory in baselines:
        table[bname] = {}
        for mode in modes:
            pol = make_grafted_baseline(bname, bfactory, mode)
            k12_onts, k12_miss = [], []
            for entry in s3_suite:
                ep_o, ep_m = [], []
                for sc, cfg in entry["cells"]:
                    r = run_simulation(sc, pol, cfg)
                    ep_o.append(float(r["ontime_rate"]))
                    ep_m.append(float(r["server_grant_miss_rate"]))
                k12_onts.append(float(np.mean(ep_o)))
                k12_miss.append(float(np.mean(ep_m)))

            bl_res = {}
            for wid, runs in bl_focus:
                ep_o = [float(run_simulation(sc, pol, cfg)["ontime_rate"]) for sc, cfg in runs]
                bl_res[wid] = float(np.mean(ep_o))

            r_ce = run_simulation(ce_2nd[0], pol, ce_2nd[1])
            table[bname][mode] = {
                "k12_ontime": float(np.mean(k12_onts)),
                "k12_miss": float(np.mean(k12_miss)),
                "BL01_MIXED_AR_CTRL": bl_res["BL01_MIXED_AR_CTRL"],
                "BL02_NEAR_FAR_HEAVY": bl_res["BL02_NEAR_FAR_HEAVY"],
                "CE_2ndOrder": f"{r_ce['n_ontime']}/{ce_2nd[0].n_tasks}",
            }
            print(
                f"[PartB] {bname:12s} | {mode:22s} | K12={table[bname][mode]['k12_ontime']*100:6.2f}% "
                f"miss={table[bname][mode]['k12_miss']*100:5.2f}% BL01={bl_res['BL01_MIXED_AR_CTRL']*100:6.2f}% "
                f"BL02={bl_res['BL02_NEAR_FAR_HEAVY']*100:6.2f}% CE2={table[bname][mode]['CE_2ndOrder']}",
                flush=True,
            )
    return table


def main() -> None:
    part_a = run_part_a()
    part_b = run_part_b()
    report = {"part_a_anti_patchwork": part_a, "part_b_grafting": part_b}
    out_path = STAGE4_DIR / "step42_grafting_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Step 4.2] Wrote {out_path}")


if __name__ == "__main__":
    main()
