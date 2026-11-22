"""Read-only Final Version Fairness & Root-Cause Audit Script (Plan 2).

Strictly read-only toward `stage2_model/`, `stage3_algorithm/`, `stage4_audit/`,
and `formal_experiments/`. Writes output solely to
`supp_verification/final_fairness_results.json`.

Checks:
  A. Equal-Shield Baseline Fairness on Main K=12 (10 seeds, latency_mode='wall'):
     CLWB_Dual vs. Bambos_ALP+CG / Bambos_ALP+2xShd / EDF_PreBatch+CG / EDF_PreBatch+2xShd.
  B. Deterministic Zero-Lag Parity & BL08 Root-Cause Check (10 seeds):
     - BL08_SPARSE_RELAXED under Full / NoShd / NoFld / EDF_PreBatch+CG for BOTH 'wall' and 'fixed=0ms'.
     - Main K=12 under deterministic 'fixed=0ms' for CLWB_Dual / Bambos_ALP+CG / EDF_PreBatch+CG,
       plus bitwise-exact determinism verification across back-to-back replays.
  C. Noiseless 0 dB Zero-Skip Grant-Miss Overshoot Audit (10 seeds, latency_mode='wall'):
     Inspect all missed edge tasks on WK05_HETERO_DL, WK09_SMALL_CELL_ASYM, WK03_DEADLINE_TIGHT
     and classify deadline overshoot (< 2.0 ms sub-ms lag vs >= 2.0 ms).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
for sub in ("stage2_model", "stage3_algorithm", "stage4_audit", "formal_experiments"):
    p = str(V3_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import champion_core  # noqa: E402
from common_utils import (  # noqa: E402
    bootstrap_ci95,
    build_single_frozen_champion,
    unpack_suite_item,
    verify_upstream_readonly,
)
from model_spec import EdgeUplinkEnv, run_simulation  # noqa: E402
from run_s52_batch2_main import (  # noqa: E402
    build_blind_suite_k8,
    build_dual_track_cg_baselines,
    build_main_benchmark_k12,
)
from step42_anti_patchwork_and_grafting import ProbePolicy  # noqa: E402


def wrap_equal_shield(base_pol: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Augment any +CG baseline with the same up-to-2x non-hoarding edge shield as CLWB-Dual."""
    def pol(obs: dict) -> dict:
        out = base_pol(obs)
        acts = [dict(a) for a in out.get("actions", [])]
        srvs = obs["servers"]
        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        srv_used = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        edge_acts = [a for a in acts if a.get("mode") == "edge"]
        for a in edge_acts:
            srv_used[int(a["server"])] += float(a["f_hz"])
        for a in edge_acts:
            j = int(a["server"])
            fb = float(a["f_hz"])
            headroom = max(0.0, float(f_avail[j] - srv_used[j]) - 1e-6 * float(f_avail[j]))
            bump = min(headroom, fb)
            srv_used[j] += bump
            a["f_hz"] = min(float(f_avail[j]), fb + bump)
        return {"actions": acts}

    return pol


def run_check_a_equal_shield(k12: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    policies: list[tuple[str, Callable[[dict], dict]]] = [
        ("CLWB_Dual", build_single_frozen_champion()),
        ("Bambos_ALP+CG", baselines["Bambos_ALP+CG"]),
        ("Bambos_ALP+2xShd", wrap_equal_shield(baselines["Bambos_ALP+CG"])),
        ("EDF_PreBatch+CG", baselines["EDF_PreBatch+CG"]),
        ("EDF_PreBatch+2xShd", wrap_equal_shield(baselines["EDF_PreBatch+CG"])),
    ]

    per_method_wk_ont: dict[str, list[float]] = {nm: [] for nm, _ in policies}
    per_method_wk_miss: dict[str, list[float]] = {nm: [] for nm, _ in policies}
    per_method_illegal: dict[str, int] = {nm: 0 for nm, _ in policies}
    per_workload: dict[str, dict[str, Any]] = {}

    for item in k12:
        wid, cells = unpack_suite_item(item)
        per_workload[wid] = {}
        for nm, pol in policies:
            ots, gms, sks = [], [], []
            for sc, cfg in cells:
                res = run_simulation(sc, pol, cfg=cfg)
                ots.append(float(res["ontime_rate"]) * 100.0)
                gms.append(float(res["server_grant_miss_rate"]) * 100.0)
                sks.append(int(res["skipped_ticks"]))
                per_method_illegal[nm] += int(res["illegal_actions"])
            m_on = float(np.mean(ots))
            m_gm = float(np.mean(gms))
            per_method_wk_ont[nm].append(m_on)
            per_method_wk_miss[nm].append(m_gm)
            per_workload[wid][nm] = {
                "ontime_pct": m_on,
                "grant_miss_pct": m_gm,
                "skipped_ticks_sum": int(sum(sks)),
            }

    clwb_mean = float(np.mean(per_method_wk_ont["CLWB_Dual"]))
    summary: dict[str, Any] = {}
    for nm, _ in policies:
        mean_v, ci_lo, ci_hi = bootstrap_ci95(per_method_wk_ont[nm], n_boot=10000, seed=55001)
        summary[nm] = {
            "k12_ontime_mean_pct": float(mean_v),
            "k12_ontime_ci95_pct": [float(ci_lo), float(ci_hi)],
            "k12_grant_miss_mean_pct": float(np.mean(per_method_wk_miss[nm])),
            "delta_vs_clwb_pp": float(mean_v - clwb_mean),
            "illegal_actions_sum": int(per_method_illegal[nm]),
        }
    return {"summary": summary, "per_workload": per_workload}


def run_check_b_deterministic_parity(
    k12: list[dict[str, Any]],
    k8: list[tuple[str, list[tuple[Any, Any]]]],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    bl08_cells = next(cells for wid, cells in k8 if wid == "BL08_SPARSE_RELAXED")
    bl08_pols: list[tuple[str, Callable[[dict], dict]]] = [
        ("Full", build_single_frozen_champion()),
        ("NoShd", ProbePolicy(disable_edge_shield=True, name="NoShd")),
        ("NoFld", champion_core.make_clwb_dual(use_fluid_contraction=False, name="NoFld")),
        ("EDF_PreBatch+CG", baselines["EDF_PreBatch+CG"]),
    ]

    bl08_summary: dict[str, dict[str, Any]] = {}
    for nm, pol in bl08_pols:
        bl08_summary[nm] = {}
        for mode in ("wall", "fixed"):
            ots, gms, sks = [], [], []
            for sc, cfg0 in bl08_cells:
                cfg = dataclasses.replace(cfg0, latency_mode=mode, fixed_latency_s=0.0)
                r = run_simulation(sc, pol, cfg=cfg)
                ots.append(float(r["ontime_rate"]) * 100.0)
                gms.append(float(r["server_grant_miss_rate"]) * 100.0)
                sks.append(int(r["skipped_ticks"]))
            bl08_summary[nm][mode] = {
                "ontime_pct": float(np.mean(ots)),
                "grant_miss_pct": float(np.mean(gms)),
                "skipped_ticks_sum": int(sum(sks)),
            }

    det_pols: list[tuple[str, Callable[[dict], dict]]] = [
        ("CLWB_Dual", build_single_frozen_champion()),
        ("Bambos_ALP+CG", baselines["Bambos_ALP+CG"]),
        ("EDF_PreBatch+CG", baselines["EDF_PreBatch+CG"]),
    ]
    det_k12: dict[str, Any] = {}
    det_per_wk: dict[str, dict[str, float]] = {}
    for nm, pol in det_pols:
        wk_ots, wk_gms = [], []
        det_per_wk[nm] = {}
        for item in k12:
            wid, cells = unpack_suite_item(item)
            ots, gms = [], []
            for sc, cfg0 in cells:
                cfg = dataclasses.replace(cfg0, latency_mode="fixed", fixed_latency_s=0.0)
                r = run_simulation(sc, pol, cfg=cfg)
                ots.append(float(r["ontime_rate"]) * 100.0)
                gms.append(float(r["server_grant_miss_rate"]) * 100.0)
            m_on = float(np.mean(ots))
            wk_ots.append(m_on)
            wk_gms.append(float(np.mean(gms)))
            det_per_wk[nm][wid] = m_on
        m_v, lo_v, hi_v = bootstrap_ci95(wk_ots, n_boot=10000, seed=55002)
        det_k12[nm] = {
            "ontime_mean_pct": float(m_v),
            "ontime_ci95_pct": [float(lo_v), float(hi_v)],
            "grant_miss_mean_pct": float(np.mean(wk_gms)),
        }

    # Verify bitwise reproducibility across two consecutive fixed=0ms runs on 3 workloads (30 episodes total)
    max_replay_diff = 0.0
    champ = build_single_frozen_champion()
    for item in k12[:3]:
        _wid, cells = unpack_suite_item(item)
        for sc, cfg0 in cells:
            cfg = dataclasses.replace(cfg0, latency_mode="fixed", fixed_latency_s=0.0)
            r1 = run_simulation(sc, champ, cfg=cfg)
            r2 = run_simulation(sc, champ, cfg=cfg)
            max_replay_diff = max(
                max_replay_diff,
                abs(float(r1["ontime_rate"]) - float(r2["ontime_rate"])),
                abs(float(r1["server_grant_miss_rate"]) - float(r2["server_grant_miss_rate"])),
            )

    return {
        "bl08_wall_vs_fixed0": bl08_summary,
        "k12_fixed0": det_k12,
        "k12_fixed0_per_workload": det_per_wk,
        "bitwise_replay_max_diff": float(max_replay_diff),
        "is_bitwise_deterministic": bool(max_replay_diff == 0.0),
    }


def run_check_c_noiseless_overshoot(k12: list[dict[str, Any]]) -> dict[str, Any]:
    targets = ("WK05_HETERO_DL", "WK09_SMALL_CELL_ASYM", "WK03_DEADLINE_TIGHT")
    champ = build_single_frozen_champion()
    out: dict[str, Any] = {}

    for wid in targets:
        cells = next(unpack_suite_item(item)[1] for item in k12 if unpack_suite_item(item)[0] == wid)
        edge_grants = 0
        edge_misses = 0
        overshoots_ms: list[float] = []
        sub_2ms = 0
        ge_2ms = 0
        skips_sum = 0
        ontimes_pct: list[float] = []

        for sc, cfg0 in cells:
            env = EdgeUplinkEnv(sc, cfg0)
            res = env.run(champ)
            ontimes_pct.append(float(res["ontime_rate"]) * 100.0)
            skips_sum += int(res["skipped_ticks"])
            edge_grants += int(res["n_edge_granted"])
            for rt in env.tasks:
                if rt.mode == "edge" and rt.state != "done":
                    edge_misses += 1
                    if rt.t_tx_end_s is not None and rt.f_hz is not None:
                        ov_ms = ((float(rt.t_tx_end_s) + float(rt.C_cycles) / float(rt.f_hz) + float(rt.tau_dn_s)) - float(rt.deadline_s)) * 1000.0
                    else:
                        ov_ms = 999.0
                    overshoots_ms.append(float(ov_ms))
                    if ov_ms < 2.0:
                        sub_2ms += 1
                    else:
                        ge_2ms += 1

        out[wid] = {
            "ontime_mean_pct": float(np.mean(ontimes_pct)),
            "skipped_ticks_sum": int(skips_sum),
            "total_edge_grants": int(edge_grants),
            "total_edge_misses": int(edge_misses),
            "edge_miss_pct": float(100.0 * edge_misses / max(edge_grants, 1)),
            "count_lt_2ms": int(sub_2ms),
            "count_ge_2ms": int(ge_2ms),
            "median_overshoot_ms": float(np.median(overshoots_ms)) if overshoots_ms else 0.0,
            "max_overshoot_ms": float(np.max(overshoots_ms)) if overshoots_ms else 0.0,
        }
    return out


def main() -> None:
    verify_upstream_readonly()
    t0 = time.perf_counter()
    k12 = build_main_benchmark_k12(n_seeds=10)
    k8 = build_blind_suite_k8(n_seeds=10)
    baselines = dict(build_dual_track_cg_baselines())

    check_a = run_check_a_equal_shield(k12, baselines)
    check_b = run_check_b_deterministic_parity(k12, k8, baselines)
    check_c = run_check_c_noiseless_overshoot(k12)
    verify_upstream_readonly()

    payload = {
        "elapsed_s": float(time.perf_counter() - t0),
        "check_a_equal_shield_k12": check_a,
        "check_b_deterministic_parity": check_b,
        "check_c_noiseless_overshoot": check_c,
    }
    out_p = Path(__file__).resolve().parent / "final_fairness_results.json"
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== CHECK A: K=12 Equal 2x Non-Hoarding Shield (10 seeds, wall) ===")
    for nm, d in check_a["summary"].items():
        print(
            f"  {nm:20s}: ontime={d['k12_ontime_mean_pct']:6.2f}% "
            f"[{d['k12_ontime_ci95_pct'][0]:.2f}, {d['k12_ontime_ci95_pct'][1]:.2f}]  "
            f"miss={d['k12_grant_miss_mean_pct']:5.2f}%  delta={d['delta_vs_clwb_pp']:+6.2f}pp  "
            f"illegal={d['illegal_actions_sum']}"
        )
    print("=== CHECK B: BL08 wall vs fixed=0ms & K=12 fixed=0ms ===")
    for nm, mp in check_b["bl08_wall_vs_fixed0"].items():
        print(
            f"  BL08 {nm:16s}: wall={mp['wall']['ontime_pct']:6.2f}% (miss {mp['wall']['grant_miss_pct']:5.2f}%) | "
            f"fixed0={mp['fixed']['ontime_pct']:6.2f}% (miss {mp['fixed']['grant_miss_pct']:5.2f}%)"
        )
    for nm, d in check_b["k12_fixed0"].items():
        print(f"  K12 fixed0 {nm:16s}: ontime={d['ontime_mean_pct']:6.2f}%  miss={d['grant_miss_mean_pct']:5.2f}%")
    print("  Bitwise fixed=0ms max diff:", check_b["bitwise_replay_max_diff"])
    print("=== CHECK C: Noiseless 0dB Zero-Skip Overshoots (wall) ===")
    for wid, d in check_c.items():
        print(
            f"  {wid:22s}: misses={d['total_edge_misses']}/{d['total_edge_grants']} "
            f"({d['edge_miss_pct']:.2f}%), <2ms={d['count_lt_2ms']}, >=2ms={d['count_ge_2ms']}, "
            f"med={d['median_overshoot_ms']:.3f}ms, max={d['max_overshoot_ms']:.3f}ms"
        )
    print(f"Done in {payload['elapsed_s']:.2f}s -> {out_p}")


if __name__ == "__main__":
    main()
