"""Stage 4 Step 4.3: Dual-Extreme Phase-Transition Parameter Scan (Zero Dead-Line Audit).

Scans 5 dimensionless/structural control parameters to extreme bounds (0 and inf / full) on stressful
workloads ({WK02, WK07, WK09, WK11, BL01, BL02} × 3 seeds + L3 + CE_2ndOrder):
  1. H / tick_s     in [0.25, 0.5, 1.0, 1.5, 2.0 (base), 3.0, 5.0, 10.0, inf]
  2. beta_act       in [0.0, 0.1, 0.25, 0.5 (base), 0.75, 0.9, 1.0]
  3. tau_ext        in [0.0, 0.25, 0.5, 1.0 (base), 1.5, 2.5, 5.0, inf]
  4. alpha_shield   in [0.0, 0.25, 0.5, 1.0 (base), 2.0, 4.0, inf]
  5. k_fill         in [0, 1, 2, 4, 8, 14 (base=2M+2 for M=6), 30, 100]

Flags ANY horizontal dead line (curve range < 0.30 pp and zero CE change toward 0 or inf).
Generates `stage4_sensitivity_curves.png` and `step43_sensitivity_report.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGE4_DIR = Path(__file__).resolve().parent
ROOT_V3 = STAGE4_DIR.parent
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"

for p in (str(STAGE4_DIR), str(STAGE3_DIR), str(STAGE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_suite_s3 import build_main_benchmark_k12
from candidates_routes import build_counterexample_2nd_order
from champion_core import CLWBDualPolicy, _fast_fluid_verify
from exact_oracle import build_counterexample_L3
from model_spec import SimConfig, run_simulation
from step41_blind_and_hardware import build_blind_suite_k8


class KnobScanPolicy(CLWBDualPolicy):
    def __init__(
        self,
        *,
        h_mult: float = 2.0,
        beta_act: float = 0.5,
        tau_ext: float = 1.0,
        alpha_shield: float = 1.0,
        k_fill: int | None = None,
        name: str = "CLWB_Knob",
    ) -> None:
        super().__init__(name=name)
        self.h_mult = float(h_mult)
        self.beta_act = float(beta_act)
        self.tau_ext = float(tau_ext)
        self.alpha_shield = float(alpha_shield)
        self.k_fill = k_fill
        self._last_f_iso_pm: np.ndarray | None = None

    def _compute_two_flow_externality(
        self, obs, t_dev, t_sb, t_ptx, t_d, t_c, t_dl, t_tdn,
        sig_pm, i_base_pm, rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm,
        srv_rem,
    ) -> np.ndarray:
        eq = super()._compute_two_flow_externality(
            obs, t_dev, t_sb, t_ptx, t_d, t_c, t_dl, t_tdn,
            sig_pm, i_base_pm, rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm,
            srv_rem,
        )
        if not math.isfinite(self.tau_ext):
            return np.zeros_like(eq)
        if abs(self.tau_ext - 1.0) <= 1e-12:
            return eq
        return np.maximum(0.0, eq - self.tau_ext + 1.0)

    def _compute_active_slack_floor(self, obs: dict, verify_fn: Any) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        self._captured_verify_fn = verify_fn
        mins, metas = super()._compute_active_slack_floor(obs, verify_fn)
        if abs(self.beta_act - 0.5) <= 1e-12:
            return mins, metas
        if self.beta_act <= 0.0:
            return {k: 0.0 for k in mins}, metas
        tick = float(obs["tick_s"])
        res0 = verify_fn([])
        out_mins: dict[str, float] = {}
        for u in obs["active_uplinks"]:
            tid = str(u["id"])
            t_up_e = float(res0["t_tx_end"].get(tid, math.inf))
            if not math.isfinite(t_up_e):
                out_mins[tid] = mins.get(tid, 0.0)
                continue
            s0 = float(u["deadline_s"]) - t_up_e - float(u["C_cycles"]) / max(float(u["f_hz"]), 1.0) - float(u["tau_dn_s"])
            out_mins[tid] = min(self.beta_act * max(0.0, s0), tick)
        return out_mins, metas

    def _contract_prefix(self, cands: list[dict], check_fn: Any) -> tuple[int, dict[str, Any]]:
        k_pref, pref_res = super()._contract_prefix(cands, check_fn)
        self._in_backfill = True
        self._bf_trials = 0
        self._captured_k_pref = k_pref
        return k_pref, pref_res

    def _is_batch_admissible(
        self, obs: dict, cand_list: list[dict], check_horizon: bool,
        act_min_slack: dict[str, float], act_meta: dict[str, tuple[float, float]], verify_fn: Any,
    ) -> tuple[bool, dict[str, Any]]:
        if getattr(self, "_in_backfill", False) and self.k_fill is not None:
            self._bf_trials = getattr(self, "_bf_trials", 0) + 1
            if self._bf_trials > int(self.k_fill):
                return False, {}
        if abs(self.h_mult - 2.0) <= 1e-12:
            return super()._is_batch_admissible(obs, cand_list, check_horizon, act_min_slack, act_meta, verify_fn)
        res = verify_fn(cand_list)
        if not res["feasible"]:
            return False, res
        if check_horizon and self.use_lagrangian_defer and (not bool(obs["links"]["orthogonal"])):
            t_end_m = res["t_tx_end"]
            if math.isfinite(self.h_mult):
                horizon_limit = float(obs["time"]) + self.h_mult * float(obs["tick_s"])
                for g in cand_list:
                    if g.get("can_defer", False) and float(t_end_m[g["id"]]) > horizon_limit:
                        return False, res
            for uid, min_s in act_min_slack.items():
                if act_meta[uid][0] - (float(t_end_m[uid]) + act_meta[uid][1]) < min_s:
                    return False, res
        return True, res

    def _hereditary_select_grants(
        self, obs: dict, cands: list[dict], f_iso_pm: np.ndarray, srv_rem: np.ndarray, t_enter: float,
    ) -> tuple[list[dict], dict[str, Any]]:
        if self.k_fill is None:
            self._in_backfill = False
            return super()._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)

        self._in_backfill = False
        self._bf_trials = 0
        self._captured_k_pref = 0
        self._captured_verify_fn = None
        kept, final_res = super()._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)
        native_budget = 2 * len(obs["servers"]) + 2
        if int(self.k_fill) <= native_budget or self._captured_verify_fn is None or not cands:
            self._in_backfill = False
            return kept, final_res

        # If scanning k_fill > 2M+2, continue backfill trials up to self.k_fill using captured native verify_fn!
        verify_fn = self._captured_verify_fn
        act_min_slack, act_meta = super()._compute_active_slack_floor(obs, verify_fn)
        kept_ids = {str(g["id"]) for g in kept}
        used_devs = {int(g["device"]) for g in kept}
        trials = native_budget
        self._in_backfill = False
        for cand_pos in range(self._captured_k_pref, len(cands)):
            if trials >= int(self.k_fill):
                break
            c = cands[cand_pos]
            if str(c["id"]) in kept_ids or int(c["device"]) in used_devs:
                continue
            trials += 1
            ok, _r = super()._is_batch_admissible(obs, kept + [c], True, act_min_slack, act_meta, verify_fn)
            if ok:
                kept.append(c)
                kept_ids.add(str(c["id"]))
                used_devs.add(int(c["device"]))

        final_res = verify_fn(kept)
        while kept and not final_res["feasible"]:
            kept.pop()
            final_res = verify_fn(kept)
        return kept, final_res

    def _allocate_shielded_edge(
        self, obs: dict, kept_grants: list[dict], fluid_res: dict[str, Any], srv_res: np.ndarray, f_avail: np.ndarray,
    ) -> list[dict]:
        acts = super()._allocate_shielded_edge(obs, kept_grants, fluid_res, srv_res, f_avail)
        if abs(self.alpha_shield - 1.0) <= 1e-12 or not acts:
            return acts

        f_min_hz = float(obs["f_min_hz"])
        f_min_map = fluid_res.get("f_min", {})
        tick = float(obs["tick_s"])
        now = float(obs["time"])
        t_tx_end = fluid_res.get("t_tx_end", {})
        by_id = {str(g["id"]): g for g in kept_grants}

        # Compute base allocation and available remaining headroom across servers
        used_base = np.asarray(srv_res, dtype=float).copy()
        f_bases = []
        for a in acts:
            tid = str(a.get("task", a.get("id")))
            fb = max(f_min_hz, float(f_min_map.get(tid, a["f_hz"])))
            f_bases.append(fb)
            used_base[int(a["server"])] += fb
        headroom = np.maximum(0.0, f_avail - used_base)

        for a, fb in zip(acts, f_bases):
            tid = str(a.get("task", a.get("id")))
            j = int(a["server"])
            if self.alpha_shield <= 0.0:
                a["f_hz"] = fb
                continue
            if not math.isfinite(self.alpha_shield):
                # Greedy hoarding: lock ALL remaining headroom onto earliest admitted tasks
                bump = float(headroom[j])
                a["f_hz"] = fb + bump
                headroom[j] = 0.0
                continue
            g = by_id[tid]
            t_e = float(t_tx_end.get(tid, now + tick))
            slack_cmp = max(1e-9, float(g["deadline_s"]) - t_e - float(g["tau_dn_s"]))
            u_est = max(0.0, t_e - now)
            s_target = min(self.alpha_shield * (tick + u_est), 0.5 * slack_cmp if self.alpha_shield <= 1.0 else (0.9 * slack_cmp))
            f_target = float(g["C_cycles"]) / max(1e-9, slack_cmp - s_target)
            bump = min(float(headroom[j]), max(0.0, f_target - fb), self.alpha_shield * fb)
            a["f_hz"] = fb + bump
            headroom[j] = max(0.0, headroom[j] - bump)
        return acts


def main() -> None:
    s3_suite = {e["wid"]: e["cells"] for e in build_main_benchmark_k12(n_seeds=3)}
    bl_suite = dict(build_blind_suite_k8(n_seeds=3))
    stress_wids = ["WK02_DENSE_TIGHT", "WK07_BURST_HEAVY", "WK09_SMALL_CELL_ASYM", "WK11_CSI_NOISE_2DB", "BL01_MIXED_AR_CTRL", "BL02_NEAR_FAR_HEAVY"]
    stress_runs = []
    for wid in stress_wids:
        src = s3_suite[wid] if wid in s3_suite else bl_suite[wid]
        stress_runs.append((wid, src))

    ce_l3 = build_counterexample_L3()
    ce_2 = build_counterexample_2nd_order()

    scans = [
        ("H / tick_s", "h_mult", 2.0, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf]),
        ("beta_act", "beta_act", 0.5, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        ("tau_ext", "tau_ext", 1.0, [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, math.inf]),
        ("alpha_shield", "alpha_shield", 1.0, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, math.inf]),
        ("k_fill", "k_fill", 14, [0, 1, 2, 4, 8, 14, 30, 100]),
    ]

    t0 = time.perf_counter()
    report: dict[str, Any] = {"stress_workloads": stress_wids, "scans": {}, "dead_line_flags": []}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), dpi=200)
    axes_flat = list(axes.flat)

    for s_idx, (title, kw, base_val, grid_vals) in enumerate(scans):
        points = []
        for val in grid_vals:
            pol = KnobScanPolicy(**{kw: val})
            wk_means = []
            miss_list = []
            per_wk = {}
            for wid, runs in stress_runs:
                os_, ms_ = [], []
                for sc, cfg in runs:
                    r = run_simulation(sc, pol, cfg)
                    os_.append(float(r["ontime_rate"]))
                    ms_.append(float(r["server_grant_miss_rate"]))
                per_wk[wid] = float(np.mean(os_))
                wk_means.append(per_wk[wid])
                miss_list.extend(ms_)
            r_l3 = run_simulation(ce_l3[0], pol, ce_l3[1])
            r_c2 = run_simulation(ce_2[0], pol, ce_2[1])
            pt = {
                "value_str": "inf" if isinstance(val, float) and not math.isfinite(val) else str(val),
                "value_num": None if isinstance(val, float) and not math.isfinite(val) else float(val),
                "is_base": (val == base_val),
                "mean_ontime_pct": float(np.mean(wk_means) * 100.0),
                "mean_miss_pct": float(np.mean(miss_list) * 100.0),
                "wk02_pct": per_wk["WK02_DENSE_TIGHT"] * 100.0,
                "wk11_csi2db_pct": per_wk["WK11_CSI_NOISE_2DB"] * 100.0,
                "bl01_pct": per_wk["BL01_MIXED_AR_CTRL"] * 100.0,
                "l3_score": f"{r_l3['n_ontime']}/{ce_l3[0].n_tasks}",
                "ce2_score": f"{r_c2['n_ontime']}/{ce_2[0].n_tasks}",
            }
            points.append(pt)
            print(
                f"[Scan] {title:14s} val={pt['value_str']:5s} mean={pt['mean_ontime_pct']:5.2f}% "
                f"BL01={pt['bl01_pct']:5.2f}% WK11={pt['wk11_csi2db_pct']:5.2f}% "
                f"miss={pt['mean_miss_pct']:4.2f}% L3={pt['l3_score']} CE2={pt['ce2_score']}",
                flush=True,
            )

        onts = [p["mean_ontime_pct"] for p in points]
        spread_pp = float(max(onts) - min(onts))
        ce_variants = len({(p["l3_score"], p["ce2_score"]) for p in points})
        dead_flag = (spread_pp < 0.30 and ce_variants == 1)
        if dead_flag:
            report["dead_line_flags"].append(title)
        report["scans"][title] = {"base_val": base_val, "spread_pp": spread_pp, "ce_variants": ce_variants, "dead_flag": dead_flag, "points": points}

        ax = axes_flat[s_idx]
        x_idx = np.arange(len(points))
        ax.plot(x_idx, onts, marker="o", linewidth=2.0, color="#1f77b4", label="Mean Ontime (%)")
        ax.plot(x_idx, [p["bl01_pct"] for p in points], marker="s", linestyle="--", linewidth=1.3, color="#2ca02c", label="BL01_MIXED (%)")
        base_pos = [i for i, p in enumerate(points) if p["is_base"]]
        if base_pos:
            ax.axvline(base_pos[0], color="#d62728", linestyle=":", linewidth=1.6, label=f"Theory Pt ({base_val})")
            ax.scatter([base_pos[0]], [onts[base_pos[0]]], color="#d62728", marker="*", s=160, zorder=6)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([p["value_str"] for p in points], fontsize=8)
        ax.set_title(f"{title} (spread={spread_pp:.2f} pp)", fontsize=10)
        ax.set_ylabel("Ontime Rate (%)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right" if "k_fill" in title else "lower left")

    axes_flat[5].axis("off")
    fig.suptitle("Stage 4.3 Dual-Extreme Parameter Sensitivity & Theoretical Operating Points", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig_path = STAGE4_DIR / "stage4_sensitivity_curves.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    report["elapsed_s"] = float(time.perf_counter() - t0)
    out_path = STAGE4_DIR / "step43_sensitivity_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Step 4.3] Wrote {out_path} & {fig_path}, dead_line_flags={report['dead_line_flags']}")


if __name__ == "__main__":
    main()
