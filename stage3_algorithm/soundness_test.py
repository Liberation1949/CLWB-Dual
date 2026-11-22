"""Independent Soundness & Degeneracy Verification Suite for Champion Candidate (Stage 3 Step 3.1.3).

Verifies:
  1. Read-only Stage-2 `Model v1.0` freeze integrity (8/8 SHA-256 digests unchanged).
  2. Counterexample immunity across `L1`, `L2`, `L3`, and `CE_2ndOrder` vs exact offline oracle.
  3. Zero hard-constraint violations, zero illegal actions, and high oracle capture over 100
     random micro-instances (Stage-2 Gate 4 matrix).
  4. Exact analytical admission count match on all 4 Stage-2 degenerate limit anchors.
  5. Real-time wall-clock scaling across burst P in {10, 25, 45, 70, 100} (P99 < 25 ms, 0 skipped ticks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
STAGE3_DIR = Path(__file__).resolve().parent
for _p in (str(STAGE2_DIR), str(STAGE3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines_stage2 import (  # noqa: E402
    make_dgolb,
    make_edf_prebatch,
    make_sfa,
)
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
from champion_core import make_clwb_dual  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from exact_oracle import (  # noqa: E402
    build_counterexample_L1,
    build_counterexample_L2,
    build_counterexample_L3,
    solve_offline_exact,
)
from model_spec import (  # noqa: E402
    EdgeUplinkEnv,
    SimConfig,
    build_scenario_v1,
    make_micro_scenario,
    run_simulation,
    shannon_rate,
)


def check_stage2_freeze_intact() -> dict[str, Any]:
    manifest_p = STAGE2_DIR / "FREEZE_MANIFEST_v1.0.sha256"
    assert manifest_p.exists(), "Missing Stage-2 FREEZE_MANIFEST_v1.0.sha256"
    checked = 0
    mismatches = []
    for line in manifest_p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected_hash, fname = line.split(None, 1)
        fp = STAGE2_DIR / fname.strip()
        actual_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
        checked += 1
        if actual_hash != expected_hash:
            mismatches.append(fname)
    return {"passed": len(mismatches) == 0 and checked == 8, "checked_files": checked, "mismatches": mismatches}


def check_counterexamples(champ) -> dict[str, Any]:
    dgolb_cg = wrap_with_common_guard(make_dgolb(), "DGOLB+CG")
    edf_cg = wrap_with_common_guard(make_edf_prebatch(), "EDF_PreBatch+CG")
    sfa_cg = wrap_with_common_guard(make_sfa(), "SFA+CG")

    # L1
    sc1, cf1 = build_counterexample_L1()
    ex1 = solve_offline_exact(sc1, cf1)
    ch1 = run_simulation(sc1, champ, cf1)
    dg1 = run_simulation(sc1, dgolb_cg, cf1)
    ed1 = run_simulation(sc1, edf_cg, cf1)
    l1_ok = (ch1["n_ontime"] >= 2 and dg1["n_ontime"] == 0 and ed1["n_ontime"] == 0)

    # L2
    sc2, cf2 = build_counterexample_L2()
    ex2 = solve_offline_exact(sc2, cf2)
    ex2_nd = solve_offline_exact(sc2, cf2, non_delay=True)
    ch2 = run_simulation(sc2, champ, cf2)
    l2_ok = (ch2["n_ontime"] == ex2["best_ontime"] == 3 and ex2_nd["best_ontime"] <= 2)

    # L3
    sc3, cf3 = build_counterexample_L3()
    ex3 = solve_offline_exact(sc3, cf3)
    ch3 = run_simulation(sc3, champ, cf3)
    sf3 = run_simulation(sc3, sfa_cg, cf3)
    l3_ok = (ch3["n_ontime"] == ex3["best_ontime"] == 3 and sf3["n_ontime"] == 1)

    # CE_2ndOrder
    sc4, cf4 = build_counterexample_2nd_order()
    ex4 = solve_offline_exact(sc4, cf4)
    ch4 = run_simulation(sc4, champ, cf4)
    sf4 = run_simulation(sc4, sfa_cg, cf4)
    dg4 = run_simulation(sc4, dgolb_cg, cf4)
    ce2_ok = (ch4["n_ontime"] == ex4["best_ontime"] == 4 and sf4["n_ontime"] <= 2 and dg4["n_ontime"] <= 2)

    return {
        "passed": bool(l1_ok and l2_ok and l3_ok and ce2_ok),
        "L1": {"passed": l1_ok, "champion": ch1["n_ontime"], "oracle": ex1["best_ontime"], "dgolb_cg": dg1["n_ontime"], "edf_cg": ed1["n_ontime"]},
        "L2": {"passed": l2_ok, "champion": ch2["n_ontime"], "oracle": ex2["best_ontime"], "oracle_nondelay": ex2_nd["best_ontime"]},
        "L3": {"passed": l3_ok, "champion": ch3["n_ontime"], "oracle": ex3["best_ontime"], "sfa_cg": sf3["n_ontime"]},
        "CE_2ndOrder": {"passed": ce2_ok, "champion": ch4["n_ontime"], "oracle": ex4["best_ontime"], "sfa_cg": sf4["n_ontime"], "dgolb_cg": dg4["n_ontime"]},
    }


def check_micro_100_oracle(champ) -> dict[str, Any]:
    rng = np.random.default_rng(777)
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    ch_rates = []
    or_rates = []
    inversions = 0
    illegal_tot = 0
    checks_tot = 0

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
        ex = solve_offline_exact(scen, cfg)
        res = run_simulation(scen, champ, cfg)
        or_rates.append(ex["best_rate"])
        ch_rates.append(res["ontime_rate"])
        illegal_tot += res["illegal_actions"]
        checks_tot += res["hard_invariant_checks"]
        if res["n_ontime"] > ex["best_ontime"]:
            inversions += 1

    mean_ch = float(np.mean(ch_rates))
    mean_or = float(np.mean(or_rates))
    capture = mean_ch / max(mean_or, 1e-9)
    return {
        "passed": bool(inversions == 0 and illegal_tot == 0 and capture >= 0.95),
        "oracle_mean": mean_or,
        "champion_mean": mean_ch,
        "capture_ratio": float(capture),
        "inversions": int(inversions),
        "illegal_actions": int(illegal_tot),
        "hard_invariant_checks": int(checks_tot),
    }


def check_degenerate_anchors(champ) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # Anchor 1: Single-server orthogonal -> smallest-share-first exact
    n, m = 5, 1
    h1 = np.array([[1e-10]] * n, dtype=float)
    f_loc1 = np.full(n, 0.2e9, dtype=float)
    f_ms1 = np.full(m, 20.0e9, dtype=float)
    ptx1 = np.ones(n, dtype=float)
    t_specs = [
        {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.25 + 0.05 * i,
         "D_bits": 800_000.0, "C_cycles": (2.0 + 0.8 * i) * 1e9, "tau_dn_s": 0.001}
        for i in range(n)
    ]
    scen1 = make_micro_scenario(case_id="A1", n_devices=n, n_servers=m, h_dev_ms=h1, f_loc_hz=f_loc1, f_ms_max_hz=f_ms1, p_tx_w=ptx1, tasks_raw=t_specs)
    env1 = EdgeUplinkEnv(scen1, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="zero"))
    env1._reveal_arrivals(0.0)
    obs1 = env1.observe(0.0)
    acts1 = champ(obs1)["actions"]
    r0_1 = shannon_rate(10e6, 1e-10, 1e-13, 0.0)
    freqs1 = sorted(float(t["C_cycles"]) / (float(t["deadline_s"]) - 0.001 - float(t["D_bits"]) / r0_1 - float(t["tau_dn_s"])) for t in obs1["pending"])
    cap1, acc1, opt1 = float(obs1["servers"][0]["f_avail_hz"]), 0.0, 0
    for f in freqs1:
        if acc1 + f <= cap1 + 1.0:
            acc1 += f
            opt1 += 1
    results["anchor1_single_srv_ortho"] = {"admitted": len(acts1), "optimal": opt1, "match": len(acts1) == opt1}

    # Anchor 2: Infinite CPU SINR
    tasks2 = [{"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.140, "D_bits": 800_000.0, "C_cycles": 5.0e8, "tau_dn_s": 0.001} for i in range(4)]
    scen2 = make_micro_scenario(case_id="A2", n_devices=4, n_servers=1, h_dev_ms=np.full((4, 1), 1e-10), f_loc_hz=np.full(4, 0.2e9), f_ms_max_hz=np.full(1, 1e14), p_tx_w=np.ones(4), tasks_raw=tasks2)
    env2 = EdgeUplinkEnv(scen2, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero"))
    env2._reveal_arrivals(0.0)
    acts2 = champ(env2.observe(0.0))["actions"]
    opt2 = max(k for k in range(1, 5) if 0.001 + 800_000.0 / shannon_rate(10e6, 1e-10, 1e-13, (k - 1) * 1e-10) + 0.001 <= 0.140)
    results["anchor2_inf_cpu_sinr"] = {"admitted": len(acts2), "optimal": opt2, "match": len(acts2) == opt2}

    # Anchor 3: Infinite BW packing
    tasks3 = [{"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.20, "D_bits": 800_000.0, "C_cycles": (0.8 + 0.5 * i) * 1e9, "tau_dn_s": 0.001} for i in range(5)]
    scen3 = make_micro_scenario(case_id="A3", n_devices=5, n_servers=1, h_dev_ms=np.full((5, 1), 1e-10), f_loc_hz=np.full(5, 0.1e9), f_ms_max_hz=np.full(1, 20.0e9), p_tx_w=np.ones(5), tasks_raw=tasks3, bandwidth_hz=1e12)
    env3 = EdgeUplinkEnv(scen3, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero"))
    env3._reveal_arrivals(0.0)
    obs3 = env3.observe(0.0)
    acts3 = champ(obs3)["actions"]
    freqs3 = sorted(float(t["C_cycles"]) / (float(t["deadline_s"]) - 0.001 - float(t["tau_dn_s"])) for t in obs3["pending"])
    cap3, acc3, opt3 = float(obs3["servers"][0]["f_avail_hz"]), 0.0, 0
    for f in freqs3:
        if acc3 + f <= cap3 + 1.0:
            acc3 += f
            opt3 += 1
    results["anchor3_inf_bw_pack"] = {"admitted": len(acts3), "optimal": opt3, "match": len(acts3) == opt3}

    # Anchor 4: Single task closed form
    scen4 = make_micro_scenario(case_id="A4", n_devices=1, n_servers=1, h_dev_ms=np.array([[1e-10]]), f_loc_hz=np.array([1.0e9]), f_ms_max_hz=np.array([20.0e9]), p_tx_w=np.array([1.0]),
                                tasks_raw=[{"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0, "deadline_s": 0.50, "D_bits": 800_000.0, "C_cycles": 1.0e9, "tau_dn_s": 0.001}])
    env4 = EdgeUplinkEnv(scen4, SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))
    env4._reveal_arrivals(0.0)
    acts4 = champ(env4.observe(0.0))["actions"]
    results["anchor4_single_task"] = {"admitted": len(acts4), "optimal": 1, "match": len(acts4) == 1 and acts4[0]["mode"] == "edge"}

    results["passed"] = all(v["match"] for v in results.values() if isinstance(v, dict))
    return results


def check_burst_scaling(champ) -> dict[str, Any]:
    rows = {}
    all_ok = True
    for p_cnt in (10, 25, 45, 70, 100):
        sc = build_scenario_v1(
            case_id=f"SND_P{p_cnt}", family="burst", seed=9100 + p_cnt,
            n_devices=p_cnt, n_servers=6, tasks_per_device=1,
            offered_load=0.80, T_max_s=0.65, radius_m=220.0, window_s=0.005,
        )
        cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
        res = run_simulation(sc, champ, cfg)
        rows[str(p_cnt)] = {
            "ontime_rate": float(res["ontime_rate"]),
            "miss_rate": float(res["server_grant_miss_rate"]),
            "decision_p99_ms": float(res["decision_p99_ms"]),
            "skipped_ticks": int(res["skipped_ticks"]),
            "hard_invariant_checks": int(res["hard_invariant_checks"]),
        }
        if res["decision_p99_ms"] >= 25.0 or res["skipped_ticks"] > 0 or res["illegal_actions"] > 0:
            all_ok = False
    return {"passed": bool(all_ok), "by_P": rows}


def run_all_soundness() -> dict[str, Any]:
    champ = make_clwb_dual()
    t0 = time.perf_counter()
    c1 = check_stage2_freeze_intact()
    c2 = check_counterexamples(champ)
    c3 = check_micro_100_oracle(champ)
    c4 = check_degenerate_anchors(champ)
    c5 = check_burst_scaling(champ)
    all_ok = bool(c1["passed"] and c2["passed"] and c3["passed"] and c4["passed"] and c5["passed"])
    return {
        "all_passed": all_ok,
        "elapsed_s": float(time.perf_counter() - t0),
        "stage2_freeze_intact": c1,
        "counterexamples": c2,
        "micro_100_oracle": c3,
        "degenerate_anchors": c4,
        "burst_scaling": c5,
    }


if __name__ == "__main__":
    rep = run_all_soundness()
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["all_passed"] else 1)
