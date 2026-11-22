"""Read-only Plan B Supplemental Audit Verification Script.

Does NOT modify any files in `stage2_model/`, `stage3_algorithm/`, `stage4_audit/`,
or `formal_experiments/`. Writes outputs strictly to `supp_verification/planb_audit_results.json`.

Checks:
  1. Same-Engine SFA Wall-Clock Fairness Check: SFA re-implemented on champion_core._fast_fluid_verify
     vs. CLWB-Dual vs. reference SFA at P in {50, 100, 200} under identical gc.disable() timing.
  2. Model-Aware EDF-Fluid Comparison: Frozen Full CLWB-Dual vs. LOO_no_ext on:
     (2a) Exact frozen build_ood_suite_k4 (OOD1-OOD4, 10 seeds each = 40 episodes/variant).
     (2b) Random near-far asymmetric burst stress check (ratios 10x, 50x, 200x x 20 random trials).
  3. 2-Task Local-vs-Joint Boundary Verification (Reviewer MC1 counterexample).
"""
from __future__ import annotations

import gc
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

import baselines_stage2 as bs2  # noqa: E402
import champion_core  # noqa: E402
from benchmark_suite_s3 import build_ood_suite_k4  # noqa: E402
from common_utils import build_single_frozen_champion, unpack_suite_item, verify_upstream_readonly  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from exact_oracle import solve_single_tick_exact  # noqa: E402
from model_spec import (  # noqa: E402
    EdgeUplinkEnv,
    SimConfig,
    build_scenario_v1,
    make_micro_scenario,
    run_simulation,
)


def make_sfa_fast_verify() -> Callable[[dict], dict]:
    """Exact same sequential EDF admission logic as `bs2.make_sfa()`, using `champion_core._fast_fluid_verify`."""
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
        )
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
        f_min = float(obs["f_min_hz"])

        admitted_edge: list[dict] = []
        last_valid_res: dict[str, Any] | None = None
        local_decisions: dict[str, tuple[str, int | None, float]] = {}

        for t in pending:
            i = int(t["device"])
            dl = float(t["deadline_s"])
            c_cyc = float(t["C_cycles"])

            best_edge = None
            for j in range(m):
                cand = bs2._cand_grant(t, j, devs[i])
                res = champion_core._fast_fluid_verify(obs, admitted_edge + [cand], act_lag_s=0.0)
                if res["feasible"]:
                    t_fin = float(res["t_tx_end"][t["id"]]) + c_cyc / float(res["f_min"][t["id"]]) + float(t["tau_dn_s"])
                    key = (float(res["f_min"][t["id"]]), t_fin, j)
                    if best_edge is None or key < best_edge[0]:
                        best_edge = (key, j, cand, res)

            best_local = None
            rem_loc = dl - t_exec
            if rem_loc > 1e-9:
                f_loc_req = max(f_min, c_cyc / rem_loc)
                if dev_res[i] + f_loc_req <= dev_cap[i] + 1.0:
                    best_local = (f_loc_req, t_exec + c_cyc / f_loc_req)

            choose_local = False
            if best_local is not None and best_edge is not None:
                e_loc = 1e-27 * (best_local[0] ** 2) * c_cyc
                tid = t["id"]
                t_up_dur = float(best_edge[3]["t_tx_end"][tid]) - now
                e_edge = float(devs[i]["p_tx_w"]) * max(t_up_dur, 0.0)
                if e_loc <= e_edge:
                    choose_local = True
            elif best_local is not None:
                choose_local = True

            if best_local is not None:
                f_loc_alloc = min(float(dev_cap[i]), max(f_min, best_local[0] * 1.02))
                best_local = (f_loc_alloc, t_exec + c_cyc / f_loc_alloc)

            if choose_local and best_local is not None:
                f_loc_req, _ = best_local
                dev_res[i] += f_loc_req
                local_decisions[t["id"]] = ("local", None, f_loc_req)
            elif best_edge is not None:
                _k, _j, cand, res = best_edge
                admitted_edge.append(cand)
                last_valid_res = res

        actions: list[dict] = []
        if admitted_edge and last_valid_res is None:
            last_valid_res = champion_core._fast_fluid_verify(obs, admitted_edge, act_lag_s=0.0)
        srv_cap_arr = [float(s["f_avail_hz"]) for s in srvs]
        srv_used_est = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        for g in admitted_edge:
            srv_used_est[int(g["server"])] += float(last_valid_res["f_min"][g["id"]])
        for g in admitted_edge:
            tid = g["id"]
            j = int(g["server"])
            f_req = float(last_valid_res["f_min"][tid])
            headroom = max(0.0, srv_cap_arr[j] - srv_used_est[j])
            bump = min(headroom, f_req * 0.02)
            srv_used_est[j] += bump
            actions.append({"task": tid, "mode": "edge", "server": j, "f_hz": f_req + bump})
        for tid, (md, sv, f_hz) in local_decisions.items():
            actions.append({"task": tid, "mode": md, "server": sv, "f_hz": f_hz if md == "local" else None})
        return {"actions": actions}

    policy.__name__ = "SFA_FastVerify"
    return policy


def run_check1_same_engine_sfa() -> dict[str, Any]:
    champ = build_single_frozen_champion()
    sfa_fast = wrap_with_common_guard(make_sfa_fast_verify(), "SFA_FastVerify+CG")
    sfa_ref = wrap_with_common_guard(bs2.make_sfa(), "SFA_Ref+CG")

    p_grid = [50, 100, 200]
    n_seeds = 5
    single_tick_summary: dict[str, Any] = {}

    gc.disable()
    try:
        for p_cnt in p_grid:
            clwb_ms, fast_ms, ref_ms = [], [], []
            clwb_calls, fast_calls = [], []

            for s_idx in range(n_seeds):
                sc = build_scenario_v1(
                    case_id=f"CHECK1_P{p_cnt}_S{s_idx}",
                    family="scalability",
                    seed=54300 + s_idx,
                    n_devices=p_cnt,
                    n_servers=6,
                    tasks_per_device=1,
                    offered_load=0.80,
                    T_max_s=1.20,
                    radius_m=220.0,
                    window_s=0.03,
                )
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
                env = EdgeUplinkEnv(sc, cfg)
                env._reveal_arrivals(0.05)
                obs = env.observe(0.05)

                t0 = time.perf_counter()
                champ(obs)
                clwb_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                sfa_fast(obs)
                fast_ms.append((time.perf_counter() - t0) * 1000.0)

                # Reference exact_oracle verifier (skip repeat at P=200 after 2 trials if >5s each, here 3 seeds ok)
                if p_cnt <= 100 or s_idx < 3:
                    t0 = time.perf_counter()
                    sfa_ref(obs)
                    ref_ms.append((time.perf_counter() - t0) * 1000.0)

                # Count verifier calls cleanly separately
                orig_v = champion_core._fast_fluid_verify
                cnt = 0

                def _cnt_v(o, c, act_lag_s=0.0):
                    nonlocal cnt
                    cnt += 1
                    return orig_v(o, c, act_lag_s)

                champion_core._fast_fluid_verify = _cnt_v
                try:
                    cnt = 0
                    champ(obs)
                    clwb_calls.append(cnt)
                    cnt = 0
                    sfa_fast(obs)
                    fast_calls.append(cnt)
                finally:
                    champion_core._fast_fluid_verify = orig_v

            single_tick_summary[str(p_cnt)] = {
                "n_trials": n_seeds,
                "clwb_wall_ms_mean": float(np.mean(clwb_ms)),
                "clwb_wall_ms_max": float(np.max(clwb_ms)),
                "clwb_verify_calls_mean": float(np.mean(clwb_calls)),
                "clwb_verify_calls_max": int(np.max(clwb_calls)),
                "sfa_fast_wall_ms_mean": float(np.mean(fast_ms)),
                "sfa_fast_wall_ms_max": float(np.max(fast_ms)),
                "sfa_fast_verify_calls_mean": float(np.mean(fast_calls)),
                "sfa_fast_verify_calls_max": int(np.max(fast_calls)),
                "sfa_ref_wall_ms_mean": float(np.mean(ref_ms)),
                "sfa_ref_wall_ms_max": float(np.max(ref_ms)),
                "sfa_fast_exceeds_50ms": bool(np.mean(fast_ms) > 50.0),
                "speedup_clwb_vs_sfa_fast_mean": float(np.mean(fast_ms) / max(np.mean(clwb_ms), 1e-6)),
            }

        # Also run 3 full wall-clock episodes at P=50 & P=100 comparing CLWB-Dual vs SFA_FastVerify+CG
        ep_summary: dict[str, Any] = {}
        for p_cnt in [50, 100]:
            ep_rows = {"CLWB_Dual": [], "SFA_FastVerify+CG": []}
            for s_idx in range(3):
                sc = build_scenario_v1(
                    case_id=f"CHECK1_EP_P{p_cnt}_S{s_idx}",
                    family="scalability",
                    seed=54300 + s_idx,
                    n_devices=p_cnt,
                    n_servers=6,
                    tasks_per_device=1,
                    offered_load=0.80,
                    T_max_s=1.20,
                    radius_m=220.0,
                    window_s=0.03,
                )
                cfg_wall = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
                for nm, pol in [("CLWB_Dual", champ), ("SFA_FastVerify+CG", sfa_fast)]:
                    r = run_simulation(sc, pol, cfg_wall)
                    ep_rows[nm].append({
                        "ontime_rate": float(r["ontime_rate"]),
                        "p99_ms": float(r["decision_p99_ms"]),
                        "skipped_ticks": int(r["skipped_ticks"]),
                    })
            ep_summary[str(p_cnt)] = {
                nm: {
                    "ontime_mean_pct": float(np.mean([x["ontime_rate"] for x in vals]) * 100.0),
                    "p99_ms_max": float(np.max([x["p99_ms"] for x in vals])),
                    "skipped_ticks_sum": int(np.sum([x["skipped_ticks"] for x in vals])),
                }
                for nm, vals in ep_rows.items()
            }
    finally:
        gc.enable()

    return {"single_tick": single_tick_summary, "wall_episodes": ep_summary}


def run_check2_model_aware_ood_and_asymmetry() -> dict[str, Any]:
    champ_full = build_single_frozen_champion()
    loo_no_ext = champion_core.make_clwb_dual(use_externality_pricing=False, name="LOO_no_ext")

    # (2a) OOD1 - OOD4 under exact frozen SimConfig
    ood_items = build_ood_suite_k4(n_seeds=10)
    ood_per_wk: dict[str, Any] = {}
    full_means, noext_means = [], []
    for item in ood_items:
        wid, cells = unpack_suite_item(item)
        f_on, n_on, f_miss, n_miss = [], [], [], []
        for sc, cfg in cells:
            rf = run_simulation(sc, champ_full, cfg)
            rn = run_simulation(sc, loo_no_ext, cfg)
            f_on.append(float(rf["ontime_rate"]) * 100.0)
            n_on.append(float(rn["ontime_rate"]) * 100.0)
            f_miss.append(float(rf["server_grant_miss_rate"]) * 100.0)
            n_miss.append(float(rn["server_grant_miss_rate"]) * 100.0)
        fm, nm = float(np.mean(f_on)), float(np.mean(n_on))
        full_means.append(fm)
        noext_means.append(nm)
        ood_per_wk[wid] = {
            "full_ontime_pct": fm,
            "no_ext_ontime_pct": nm,
            "diff_pp": fm - nm,
            "full_miss_pct": float(np.mean(f_miss)),
            "no_ext_miss_pct": float(np.mean(n_miss)),
        }

    # (2b) Random near-far asymmetric burst stress check (ratios 10, 50, 200 x 20 trials)
    rng = np.random.default_rng(20261122)
    asym_results: dict[str, Any] = {}
    for ratio in [10, 50, 200]:
        f_rates, n_rates = [], []
        wins, ties, losses = 0, 0, 0
        for t_idx in range(20):
            n_dev = 6
            # Near device 0 has gain `ratio * h_far`, far devices 1..5 have h_far around 1e-10
            h_far = rng.uniform(0.8e-10, 1.3e-10, size=(n_dev, 1))
            h_far[0, 0] = float(np.mean(h_far[1:, 0])) * float(ratio)
            f_loc = np.full(n_dev, 0.05e9, dtype=float)
            # Server capacity sized tightly for the moderate light victims (~16-24 GHz)
            f_ms = np.array([float(rng.uniform(16.0, 24.0)) * 1e9])
            ptx = np.ones(n_dev, dtype=float)
            dl_victims = float(rng.uniform(0.24, 0.30))
            tasks = [
                # Near bully: heavy bits + high compute + slightly earlier EDF deadline
                {
                    "tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0,
                    "deadline_s": dl_victims - float(rng.uniform(0.015, 0.035)),
                    "D_bits": float(rng.uniform(5.0e6, 7.5e6)),
                    "C_cycles": float(rng.uniform(1.8e9, 2.8e9)),
                    "tau_dn_s": 0.001,
                }
            ]
            for di in range(1, n_dev):
                tasks.append({
                    "tid": f"{di}:0", "device": di, "k": 0, "arrival_s": 0.0,
                    "deadline_s": dl_victims + float(rng.uniform(0.0, 0.025)),
                    "D_bits": float(rng.uniform(3.5e5, 6.0e5)),
                    "C_cycles": float(rng.uniform(2.5e8, 5.0e8)),
                    "tau_dn_s": 0.001,
                })
            sc = make_micro_scenario(
                case_id=f"ASYM_R{ratio}_T{t_idx}",
                n_devices=n_dev,
                n_servers=1,
                h_dev_ms=h_far,
                f_loc_hz=f_loc,
                f_ms_max_hz=f_ms,
                p_tx_w=ptx,
                tasks_raw=tasks,
            )
            cfg0 = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
            rf = run_simulation(sc, champ_full, cfg0)
            rn = run_simulation(sc, loo_no_ext, cfg0)
            r_f = float(rf["ontime_rate"]) * 100.0
            r_n = float(rn["ontime_rate"]) * 100.0
            f_rates.append(r_f)
            n_rates.append(r_n)
            if r_f > r_n + 1e-6:
                wins += 1
            elif r_n > r_f + 1e-6:
                losses += 1
            else:
                ties += 1
        asym_results[f"{ratio}x"] = {
            "n_trials": 20,
            "full_mean_pct": float(np.mean(f_rates)),
            "no_ext_mean_pct": float(np.mean(n_rates)),
            "diff_pp": float(np.mean(f_rates) - np.mean(n_rates)),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    return {
        "ood_k4": {
            "per_workload": ood_per_wk,
            "full_mean_pct": float(np.mean(full_means)),
            "no_ext_mean_pct": float(np.mean(noext_means)),
            "diff_pp": float(np.mean(full_means) - np.mean(noext_means)),
        },
        "random_near_far_burst": asym_results,
    }


def run_check3_two_task_local_boundary() -> dict[str, Any]:
    """Verify reviewer MC1 2-task same-device dual-feasible local-first boundary."""
    champ = build_single_frozen_champion()
    # Local window at t=0: dl - (0 + 2*delta_sig) = 1.0 - 0.002 = 0.998s
    rem_loc = 0.998
    c_a = 0.80e9 * rem_loc  # requires 0.80 GHz local
    c_b = 0.50e9 * rem_loc  # requires 0.50 GHz local
    tasks_raw = [
        {
            "tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0, "deadline_s": 1.0,
            "D_bits": 2.0e8, "C_cycles": c_a, "tau_dn_s": 0.001,  # 200 Mb -> edge upload > 2s (edge infeasible)
        },
        {
            "tid": "0:1", "device": 0, "k": 1, "arrival_s": 0.0, "deadline_s": 1.0,
            "D_bits": 2.0e5, "C_cycles": c_b, "tau_dn_s": 0.001,  # 200 kb -> easily edge feasible
        },
    ]
    sc = make_micro_scenario(
        case_id="MC1_TWO_TASK_BOUNDARY",
        n_devices=1,
        n_servers=1,
        h_dev_ms=np.array([[1e-10]]),
        f_loc_hz=np.array([1.0e9]),       # 1.0 GHz local capacity
        f_ms_max_hz=np.array([20.0e9]),   # 19.0 GHz f_avail
        p_tx_w=np.array([1.0]),
        tasks_raw=tasks_raw,
    )
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
    env = EdgeUplinkEnv(sc, cfg)
    env._reveal_arrivals(0.0)
    obs = env.observe(0.0)
    oracle = solve_single_tick_exact(obs, max_p=2)
    res_sim = run_simulation(sc, champ, cfg)
    out = champ(obs)
    return {
        "f_loc_cap_ghz": 1.0,
        "task_a_min_loc_ghz": 0.80,
        "task_b_min_loc_ghz": 0.50,
        "task_a_d_bits": 2.0e8,
        "task_b_d_bits": 2.0e5,
        "oracle_best_admitted": int(oracle["best_admitted"]),
        "clwb_single_tick_admitted": len(out["actions"]),
        "clwb_full_episode_ontime": int(res_sim["n_ontime"]),
        "n_tasks": int(res_sim["n_tasks"]),
        "clwb_actions": out["actions"],
    }


def main() -> None:
    verify_upstream_readonly()
    t_start = time.perf_counter()
    c1 = run_check1_same_engine_sfa()
    c2 = run_check2_model_aware_ood_and_asymmetry()
    c3 = run_check3_two_task_local_boundary()
    verify_upstream_readonly()

    payload = {
        "elapsed_s": float(time.perf_counter() - t_start),
        "check1_same_engine_sfa": c1,
        "check2_model_aware_baseline": c2,
        "check3_two_task_boundary": c3,
    }
    out_p = Path(__file__).resolve().parent / "planb_audit_results.json"
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
