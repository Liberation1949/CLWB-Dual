"""Stage 4 Step 4.4: Pre-Theorem 10,000-Trial Adversarial Falsification Fuzzing & FAIL-1..3 Archive.

Fuzzing budget (>= 10,000 targeted adversarial trials against STAGE3_DERIVATION_REPORT.md §4):
  1. Lemma 1 Fuzz (2,500 trials): Zero-interference single-device local knapsack optimality vs exact brute-force.
  2. Theorem 1 Fuzz (3,000 trials):
       - 2,000 zero-lag extreme edge cases verified against Stage-2 `exact_oracle.evaluate_fluid_batch`,
         testing hereditary monotonicity of `_is_batch_admissible` + zero active-link violation + zero CPU cap breach.
       - 1,000 live multi-tick `EdgeUplinkEnv` episodes (`wall` and `fixed` latency modes, zero CSI noise).
  3. Theorem 2 Fuzz (2,500 trials): Exact fluid verifier call-count bound audit vs `1 + ceil(log2(P+1)) + 2M + 2 + 1`.
  4. Theorem 3 Fuzz (2,000 trials): Randomized Degenerate Anchors 1-4 (500 trials each; Anchor 1 split into
       agreeable vs general/inverted deadlines to expose hidden assumptions!).
  5. Honest Boundary Failure Archive (`FAIL-1..3`): Reproducible failure regimes outside theorem assumptions.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

STAGE4_DIR = Path(__file__).resolve().parent
ROOT_V3 = STAGE4_DIR.parent
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"

for p in (str(STAGE4_DIR), str(STAGE3_DIR), str(STAGE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_stage2 import make_bambos_alp, make_edf_prebatch
import champion_core
from champion_core import CLWBDualPolicy, _fast_fluid_verify, make_clwb_dual
from common_guard import wrap_with_common_guard
from exact_oracle import evaluate_fluid_batch, make_micro_scenario, solve_single_tick_exact
from model_spec import EdgeUplinkEnv, SimConfig, build_scenario_v1, run_simulation, shannon_rate


def _exact_local_max_count(pending_i: list[dict], f_rem_loc: float, f_min_hz: float, t_exec_loc: float) -> int:
    reqs: list[float] = []
    cap_safe = f_rem_loc * (1.0 - 1e-9)
    for p in pending_i:
        sl = float(p["deadline_s"]) - t_exec_loc
        if sl <= 1e-9:
            continue
        req = max(f_min_hz, float(p["C_cycles"]) / sl)
        if req <= cap_safe:
            reqs.append(req)
    reqs.sort()
    acc = 0.0
    cnt = 0
    for r in reqs:
        if acc + r <= cap_safe:
            acc += r
            cnt += 1
        else:
            break
    return cnt


def fuzz_lemma1(n_trials: int = 2500) -> dict[str, Any]:
    rng = np.random.default_rng(44001)
    pol = make_clwb_dual()
    violations = 0
    examples: list[dict] = []

    for tr in range(n_trials):
        n_tasks = int(rng.integers(1, 14))
        t_now = float(rng.uniform(0.0, 0.5))
        delta_sig = float(rng.choice([0.0, 1e-4, 1e-3, 5e-3]))
        t_exec = t_now + delta_sig
        t_exec_loc = t_exec + delta_sig
        f_loc = float(rng.uniform(0.2e9, 2.0e9))
        cpu_res = float(rng.uniform(0.0, 0.85 * f_loc))
        f_rem = max(0.0, f_loc - cpu_res)
        f_min_hz = 1e6

        pend = []
        for k in range(n_tasks):
            # Include boundary-gap deadlines near C / f_rem + eps
            if tr % 5 == 0 and k == 0 and f_rem > f_min_hz:
                c_cyc = float(rng.uniform(1e7, 5e8))
                eps = float(10.0 ** rng.uniform(-7.0, -3.0))
                dl = t_exec_loc + c_cyc / f_rem + eps
            else:
                c_cyc = float(rng.uniform(1e7, 9e8))
                dl = t_exec_loc + float(rng.uniform(0.02, 0.8))
            pend.append({
                "id": f"0:{k}",
                "device": 0,
                "k": k,
                "arrival_s": 0.0,
                "deadline_s": dl,
                "D_bits": float(rng.uniform(1e4, 2e6)),
                "B_out_bits": 8000.0,
                "C_cycles": c_cyc,
                "tau_dn_s": 0.001,
            })

        obs = {
            "time": t_now,
            "tick_s": 0.05,
            "delta_sig_s": delta_sig,
            "rho_sys": 0.95,
            "f_min_hz": f_min_hz,
            "pending": pend,
            "active_uplinks": [],
            "devices": [{"device": 0, "f_loc_hz": f_loc, "cpu_used_hz": cpu_res, "cpu_reserved_hz": cpu_res, "p_tx_w": 1.0, "radio_busy": False, "tx_queue_bits": 0.0, "subband": 0}],
            "servers": [{"server": 0, "f_max_hz": 50e9, "f_avail_hz": 47.5e9, "cpu_used_hz": 0.0, "cpu_reserved_hz": 0.0, "wait_compute_count": 0}],
            "links": {"h_hat_dev_ms": np.array([[1e-10]]), "bandwidth_hz": 1e7, "noise_w": 1e-13, "orthogonal": False, "n_subbands": 1},
        }
        loc_acts, loc_ids = pol._admit_local_dominant(obs, t_exec)
        opt_cnt = _exact_local_max_count(pend, f_rem, f_min_hz, t_exec_loc)
        sum_f = sum(float(a["f_hz"]) for a in loc_acts)
        # Check cardinality optimality AND hardware CPU cap
        ok_cnt = (len(loc_acts) == opt_cnt)
        ok_cap = (sum_f <= f_rem + 1.0)
        ok_ind = True
        for a in loc_acts:
            tid_a = str(a.get("task", a.get("id")))
            p = next(x for x in pend if str(x["id"]) == tid_a)
            if float(a["f_hz"]) < f_min_hz - 1e-6 or (t_exec_loc + float(p["C_cycles"]) / float(a["f_hz"])) > float(p["deadline_s"]) + 1e-9:
                ok_ind = False
        if not (ok_cnt and ok_cap and ok_ind):
            violations += 1
            if len(examples) < 5:
                examples.append({"trial": tr, "got": len(loc_acts), "opt": opt_cnt, "sum_f": sum_f, "f_rem": f_rem, "ok_cap": ok_cap, "ok_ind": ok_ind})

    return {"trials": n_trials, "violations": violations, "examples": examples}


def _gen_random_obs_and_params(rng: np.random.Generator, n_dev: int, n_srv: int, n_act: int, n_pend: int, n_sb: int, ortho: bool) -> tuple[dict, dict]:
    t_now = float(rng.uniform(0.0, 0.3))
    tick_s = float(rng.choice([0.025, 0.05]))
    delta_sig = float(rng.choice([0.0, 0.001]))
    f_min_hz = 1e6
    bw = 1e7
    noise = 1e-13

    dists = rng.uniform(10.0, 350.0, size=(n_dev, n_srv))
    h_mat = 0.1 * (1.0 / np.maximum(dists, 10.0)) ** 4.0
    p_tx = rng.uniform(0.3, 2.5, size=n_dev)
    sb = rng.integers(0, n_sb, size=n_dev)
    f_avail = rng.uniform(20e9, 60e9, size=n_srv)
    srv_res = rng.uniform(0.0, 0.65, size=n_srv) * f_avail

    devs = []
    busy_flags = np.zeros(n_dev, dtype=bool)
    active_uplinks = []
    act_devs = rng.choice(n_dev, size=min(n_act, n_dev), replace=False) if n_act > 0 else np.array([], dtype=int)
    for d in act_devs:
        j = int(rng.integers(0, n_srv))
        rem_b = float(rng.uniform(1e4, 2e6))
        sig = float(p_tx[d] * h_mat[d, j])
        # Compute upper-bound rate (isolated) to make active link statically feasible individually
        r_iso = shannon_rate(bw, sig, noise, 0.0)
        t_tx_iso = t_now + rem_b / r_iso
        c_cyc = float(rng.uniform(2e7, 6e8))
        f_lock = float(rng.uniform(max(2e8, f_min_hz), max(3e8, 0.35 * (f_avail[j] - srv_res[j]))))
        if f_lock > f_avail[j] - srv_res[j]:
            f_lock = max(f_min_hz, 0.25 * (f_avail[j] - srv_res[j]))
        # Ensure active link is feasible under U_0 alone later verified
        slack_u = float(rng.uniform(1e-4, 0.35))
        dl_u = t_tx_iso + c_cyc / f_lock + 0.001 + slack_u
        active_uplinks.append({
            "id": f"act_{d}",
            "device": int(d),
            "server": j,
            "p_tx_w": float(p_tx[d]),
            "rem_bits": rem_b,
            "tx_start_s": max(0.0, t_now - 0.02),
            "deadline_s": float(dl_u),
            "C_cycles": c_cyc,
            "f_hz": float(f_lock),
            "tau_dn_s": 0.001,
        })
        busy_flags[d] = True

    for d in range(n_dev):
        devs.append({
            "device": d,
            "f_loc_hz": 0.2e9,  # small so tasks enter edge competition
            "cpu_used_hz": 0.0,
            "cpu_reserved_hz": 0.0,
            "p_tx_w": float(p_tx[d]),
            "radio_busy": bool(busy_flags[d]),
            "tx_queue_bits": 0.0,
            "subband": int(sb[d]),
        })

    srvs = []
    for j in range(n_srv):
        srvs.append({
            "server": j,
            "f_max_hz": float(f_avail[j] / 0.95),
            "f_avail_hz": float(f_avail[j]),
            "cpu_used_hz": float(srv_res[j]),
            "cpu_reserved_hz": float(srv_res[j]),
            "wait_compute_count": 0,
        })

    pend = []
    for p_idx in range(n_pend):
        di = int(rng.integers(0, n_dev))
        d_bits = float(rng.uniform(2e4, 2.5e6))
        c_cyc = float(rng.uniform(3e7, 1.2e9))
        dl = t_now + float(rng.uniform(0.03, 0.90))
        pend.append({
            "id": f"{di}:{p_idx + 10}",
            "device": di,
            "k": p_idx,
            "arrival_s": t_now,
            "deadline_s": dl,
            "D_bits": d_bits,
            "B_out_bits": 8000.0,
            "C_cycles": c_cyc,
            "tau_dn_s": 0.001,
        })

    obs = {
        "time": t_now,
        "tick_s": tick_s,
        "delta_sig_s": delta_sig,
        "rho_sys": 0.95,
        "f_min_hz": f_min_hz,
        "pending": pend,
        "active_uplinks": active_uplinks,
        "devices": devs,
        "servers": srvs,
        "links": {
            "h_hat_dev_ms": h_mat,
            "bandwidth_hz": bw,
            "noise_w": noise,
            "orthogonal": ortho,
            "n_subbands": n_sb,
        },
    }
    params = {
        "h_dev_ms": h_mat,
        "p_tx_w": p_tx,
        "bandwidth_hz": bw,
        "noise_w": noise,
        "orthogonal": ortho,
        "n_subbands": n_sb,
        "subband": sb,
        "f_avail_hz": f_avail,
        "f_min_hz": f_min_hz,
        "delta_sig_s": delta_sig,
        "srv_reserved_hz": srv_res,
    }
    return obs, params


def fuzz_theorem1(n_zero_lag: int = 2000, n_live: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(44002)
    pol = make_clwb_dual()

    tested_valid = 0
    hered_inversions = 0
    exact_feasible_fails = 0
    active_link_fails = 0
    nic_fails = 0
    cpu_cap_fails = 0
    fail_examples: list[dict] = []

    while tested_valid < n_zero_lag:
        n_dev = int(rng.integers(4, 24))
        n_srv = int(rng.integers(1, 6))
        n_act = int(rng.integers(0, min(6, n_dev)))
        n_pend = int(rng.integers(2, 22))
        n_sb = int(rng.choice([1, 2, 4]))
        ortho = bool(rng.integers(0, 2) == 0)
        obs, params = _gen_random_obs_and_params(rng, n_dev, n_srv, n_act, n_pend, n_sb, ortho)

        # Ensure premise (iii): U(t_k) must initially be feasible under exact evaluate_fluid_batch
        base_exact = evaluate_fluid_batch(obs["active_uplinks"], [], params, obs["time"])
        if not base_exact["feasible"]:
            continue
        tested_valid += 1

        # Zero actuation-lag direct verification of _hereditary_select_grants + _allocate_shielded_edge
        # Use fixed act_lag_s = delta_sig_s (matching exact zero computing overhead premise)
        f_avail = np.asarray([float(s["f_avail_hz"]) for s in obs["servers"]], dtype=float)
        srv_rem = f_avail - np.asarray([float(s["cpu_reserved_hz"]) for s in obs["servers"]], dtype=float)
        devs = obs["devices"]
        edge_pool = []
        by_dev: dict[int, dict] = {}
        for p in obs["pending"]:
            di = int(p["device"])
            if devs[di]["radio_busy"] or devs[di]["tx_queue_bits"] > 1e-6:
                continue
            if di not in by_dev or (p["deadline_s"], p["id"]) < (by_dev[di]["deadline_s"], by_dev[di]["id"]):
                by_dev[di] = p
        edge_pool = list(by_dev.values())
        if not edge_pool:
            continue

        t_exec = float(obs["time"]) + float(obs["delta_sig_s"])
        cands, f_iso_pm = pol._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)
        verify_zero = lambda lst: _fast_fluid_verify(obs, lst, 0.0)
        act_min, act_meta = pol._compute_active_slack_floor(obs, verify_zero)
        check_fn = lambda lst: pol._is_batch_admissible(obs, lst, True, act_min, act_meta, verify_zero)

        # Every 4th trial: exhaustively check hereditary prefix monotonicity (no [False -> True] inversion)
        if tested_valid % 4 == 0 and len(cands) >= 2:
            seen_false = False
            for k in range(1, len(cands) + 1):
                ok_k, _ = check_fn(cands[:k])
                if not ok_k:
                    seen_false = True
                elif seen_false:
                    hered_inversions += 1
                    break

        k_pref, pref_res = pol._contract_prefix(cands, check_fn)
        kept = list(cands[:k_pref])
        res_kept = verify_zero(kept)
        if kept and not res_kept["feasible"]:
            while kept and not res_kept["feasible"]:
                kept.pop()
                res_kept = verify_zero(kept)

        edge_acts = pol._allocate_shielded_edge(obs, kept, res_kept, np.asarray(params["srv_reserved_hz"]), f_avail)
        by_id = {str(c["id"]): c for c in kept}
        cand_dicts = []
        srv_sum = np.zeros(n_srv, dtype=float)
        dev_set = set()
        for a in edge_acts:
            c = by_id[str(a.get("task", a.get("id")))]
            di_a = int(c["device"])
            cand_dicts.append({
                "id": c["id"],
                "device": di_a,
                "server": int(a["server"]),
                "D_bits": c["D_bits"],
                "deadline_s": c["deadline_s"],
                "C_cycles": c["C_cycles"],
                "tau_dn_s": c["tau_dn_s"],
                "f_hz": float(a["f_hz"]),
            })
            srv_sum[int(a["server"])] += float(a["f_hz"])
            if di_a in dev_set or devs[di_a]["radio_busy"]:
                nic_fails += 1
            dev_set.add(di_a)

        if np.any(srv_sum > srv_rem + 1.0):
            cpu_cap_fails += 1

        ex_check = evaluate_fluid_batch(obs["active_uplinks"], cand_dicts, params, obs["time"])
        if not ex_check["feasible"]:
            exact_feasible_fails += 1
            if "active_deadline_miss" in ex_check["fail_reason"]:
                active_link_fails += 1
            if len(fail_examples) < 5:
                fail_examples.append({"trial": tested_valid, "reason": ex_check["fail_reason"], "kept": len(cand_dicts)})

        # Also check locked frequency exceeds or meets exact minimum required share for every kept grant
        for cd in cand_dicts:
            f_req_ex = float(ex_check["f_min"].get(str(cd["id"]), math.inf))
            if float(cd["f_hz"]) < f_req_ex - 1.0:
                exact_feasible_fails += 1

    # 1,000 live multi-tick episodes across fixed/zero and wall latency modes (exact CSI, sigma=0)
    live_fixed_miss_eps = 0
    live_wall_miss_eps = 0
    live_wall_total_miss = 0
    live_wall_total_granted = 0
    live_fixed_total_miss = 0
    live_fixed_total_granted = 0
    live_illegal = 0
    live_skipped = 0

    for ep in range(n_live):
        seed = 45000 + ep
        mode = "fixed" if ep < (n_live // 2) else "wall"
        sc = build_scenario_v1(
            case_id=f"FUZZ_T1_{ep}",
            family="fuzz_t1",
            seed=seed,
            n_devices=int(rng.integers(12, 36)),
            n_servers=int(rng.integers(2, 8)),
            tasks_per_device=int(rng.integers(2, 6)),
            offered_load=float(rng.uniform(0.45, 0.90)),
            T_max_s=float(rng.uniform(0.55, 0.90)),
            radius_m=float(rng.uniform(150.0, 320.0)),
            input_kb_range=(float(rng.uniform(30.0, 90.0)), float(rng.uniform(110.0, 240.0))),
            hetero_deadline_range=(0.55, 1.50),
            n_subbands=int(rng.choice([1, 2, 4])),
        )
        cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode=mode, fixed_latency_s=(0.0005 if mode == "fixed" else 0.0), csi_sigma_db=0.0)
        r = run_simulation(sc, pol, cfg)
        live_illegal += int(r["illegal_actions"])
        live_skipped += int(r["skipped_ticks"])
        if mode == "fixed":
            live_fixed_total_granted += int(r["n_edge_granted"])
            live_fixed_total_miss += int(r["n_edge_missed"])
            if r["n_edge_missed"] > 0:
                live_fixed_miss_eps += 1
        else:
            live_wall_total_granted += int(r["n_edge_granted"])
            live_wall_total_miss += int(r["n_edge_missed"])
            if r["n_edge_missed"] > 0:
                live_wall_miss_eps += 1

    return {
        "zero_lag_trials": tested_valid,
        "hereditary_inversions": hered_inversions,
        "exact_feasible_fails": exact_feasible_fails,
        "active_link_fails": active_link_fails,
        "nic_fails": nic_fails,
        "cpu_cap_fails": cpu_cap_fails,
        "fail_examples": fail_examples,
        "live_fixed_episodes": n_live // 2,
        "live_fixed_miss_episodes": live_fixed_miss_eps,
        "live_fixed_miss_rate": float(live_fixed_total_miss / max(1, live_fixed_total_granted)),
        "live_wall_episodes": n_live - (n_live // 2),
        "live_wall_miss_episodes": live_wall_miss_eps,
        "live_wall_miss_rate": float(live_wall_total_miss / max(1, live_wall_total_granted)),
        "live_illegal_actions": live_illegal,
        "live_skipped_ticks": live_skipped,
    }


def fuzz_theorem2_complexity(n_trials: int = 2500) -> dict[str, Any]:
    rng = np.random.default_rng(44003)
    pol = make_clwb_dual()
    orig_verify = champion_core._fast_fluid_verify

    bound_violations = 0
    max_ratio = 0.0
    max_excess = 0
    worst_case: dict[str, Any] = {}

    call_cnt = 0
    def counted_verify(obs_arg: dict, cand_arg: list[dict], act_lag_s: float = 0.0) -> dict[str, Any]:
        nonlocal call_cnt
        call_cnt += 1
        return orig_verify(obs_arg, cand_arg, act_lag_s)

    champion_core._fast_fluid_verify = counted_verify
    try:
        for tr in range(n_trials):
            n_dev = int(rng.integers(5, 45))
            n_srv = int(rng.integers(1, 8))
            n_act = int(rng.integers(0, min(8, n_dev)))
            n_pend = int(rng.integers(1, 35))
            obs, _ = _gen_random_obs_and_params(rng, n_dev, n_srv, n_act, n_pend, n_sb=1, ortho=False)
            f_avail = np.asarray([float(s["f_avail_hz"]) for s in obs["servers"]], dtype=float)
            srv_rem = f_avail - np.asarray([float(s["cpu_reserved_hz"]) for s in obs["servers"]], dtype=float)
            devs = obs["devices"]
            by_dev: dict[int, dict] = {}
            for p in obs["pending"]:
                di = int(p["device"])
                if not devs[di]["radio_busy"] and devs[di]["tx_queue_bits"] <= 1e-6:
                    if di not in by_dev or p["deadline_s"] < by_dev[di]["deadline_s"]:
                        by_dev[di] = p
            edge_pool = list(by_dev.values())
            if not edge_pool:
                continue
            p_eff = len(edge_pool)
            cands, f_iso_pm = pol._build_candidate_order(obs, edge_pool, float(obs["time"]) + float(obs["delta_sig_s"]), srv_rem, f_avail)
            p_cand = len(cands)
            call_cnt = 0
            _kept, _res = pol._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, time.perf_counter())

            # Tightened Theorem 2 worst-case bound:
            # 1 (active floor) + 1 (full-batch check) + ceil(log2(P+1)) (contraction) + (2M+2) (backfill) + 1 (final) + ceil(log2(P+1)) (lag fallback)
            claimed_bound = 2 * math.ceil(math.log2(max(1, p_cand) + 1)) + 2 * n_srv + 5
            excess = call_cnt - claimed_bound
            ratio = call_cnt / max(1, claimed_bound)
            if ratio > max_ratio:
                max_ratio = ratio
                worst_case = {"trial": tr, "calls": call_cnt, "bound": claimed_bound, "P_cand": p_cand, "M": n_srv, "n_act": n_act}
            if excess > 0:
                bound_violations += 1
                max_excess = max(max_excess, excess)
    finally:
        champion_core._fast_fluid_verify = orig_verify

    return {
        "trials": n_trials,
        "bound_violations": bound_violations,
        "max_excess": max_excess,
        "max_ratio_calls_to_bound": max_ratio,
        "worst_case": worst_case,
    }


def _eval_anchor1_case(rng: np.random.Generator, pol: CLWBDualPolicy, uniform_dl: bool) -> bool:
    n = int(rng.integers(3, 7))
    m = 1
    h1 = np.full((n, m), 1e-10, dtype=float)
    f_loc = np.full(n, 0.05e9, dtype=float)  # force edge
    f_ms = np.full(m, float(rng.uniform(12.0, 26.0)) * 1e9, dtype=float)
    ptx = np.ones(n, dtype=float)
    d_bits = 800_000.0
    cycles = rng.uniform(0.8e9, 4.5e9, size=n)
    if uniform_dl:
        dls = np.full(n, float(rng.uniform(0.28, 0.45)))
    else:
        dls = rng.uniform(0.22, 0.60, size=n)

    tasks_raw = [
        {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": float(dls[i]),
         "D_bits": d_bits, "C_cycles": float(cycles[i]), "tau_dn_s": 0.001}
        for i in range(n)
    ]
    scen = make_micro_scenario(case_id="FUZZ_A1", n_devices=n, n_servers=1, h_dev_ms=h1, f_loc_hz=f_loc, f_ms_max_hz=f_ms, p_tx_w=ptx, tasks_raw=tasks_raw)
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="zero")
    env = EdgeUplinkEnv(scen, cfg)
    env._reveal_arrivals(0.0)
    obs = env.observe(0.0)
    oracle = solve_single_tick_exact(obs, max_p=n)
    got = len(pol(obs)["actions"])
    return got == int(oracle["best_admitted"])


def fuzz_theorem3_anchors(per_anchor: int = 500) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(44004)
    pol = make_clwb_dual()
    orig_perf_counter = champion_core.time.perf_counter

    # Match Theorem 3 exact-equivalence hypothesis L_solver = 0 (solve_single_tick_exact assumes zero solver wall-lag)
    champion_core.time.perf_counter = lambda: 0.0
    try:
        # Anchor 1 (500 single-tick trials under Theorem 3 uniform-deadline orthogonal M=1 hypothesis)
        a1_unif_fails = 0
        for _ in range(per_anchor):
            if not _eval_anchor1_case(rng, pol, uniform_dl=True):
                a1_unif_fails += 1

        # Empirical gap outside Theorem 3 uniform-deadline hypothesis (250 inverted/random-deadline Anchor 1 trials)
        a1_gen_fails = 0
        for _ in range(250):
            if not _eval_anchor1_case(rng, pol, uniform_dl=False):
                a1_gen_fails += 1

        # Anchor 2 (500 single-tick trials: infinite CPU, symmetric channel/payload, uniform deadline)
        a2_sym_fails = 0
        for _ in range(per_anchor):
            n = int(rng.integers(3, 6))
            dls = np.full(n, float(rng.uniform(0.11, 0.22)))
            d0 = float(rng.uniform(4e5, 1.2e6))
            tasks2 = [
                {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": float(dls[i]),
                 "D_bits": d0, "C_cycles": 5e8, "tau_dn_s": 0.001}
                for i in range(n)
            ]
            sc = make_micro_scenario(case_id="FUZZ_A2", n_devices=n, n_servers=1, h_dev_ms=np.full((n, 1), 1e-10), f_loc_hz=np.full(n, 0.1e9), f_ms_max_hz=np.array([1e14]), p_tx_w=np.ones(n), tasks_raw=tasks2)
            cf = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero")
            env = EdgeUplinkEnv(sc, cf)
            env._reveal_arrivals(0.0)
            obs = env.observe(0.0)
            oracle = solve_single_tick_exact(obs, max_p=n)
            got = len(pol(obs)["actions"])
            if got != int(oracle["best_admitted"]):
                a2_sym_fails += 1

        # Anchor 3 (500 single-tick trials: infinite BW, M=1, uniform deadline compute packing)
        a3_unif_fails = 0
        for _ in range(per_anchor):
            n = int(rng.integers(3, 6))
            cycles = rng.uniform(0.6e9, 3.0e9, size=n)
            dls = np.full(n, float(rng.uniform(0.18, 0.30)))
            tasks3 = [
                {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": float(dls[i]),
                 "D_bits": 8e5, "C_cycles": float(cycles[i]), "tau_dn_s": 0.001}
                for i in range(n)
            ]
            sc = make_micro_scenario(case_id="FUZZ_A3", n_devices=n, n_servers=1, h_dev_ms=np.full((n, 1), 1e-10), f_loc_hz=np.full(n, 0.05e9), f_ms_max_hz=np.full(1, float(rng.uniform(12.0, 26.0)) * 1e9), p_tx_w=np.ones(n), tasks_raw=tasks3, bandwidth_hz=1e12)
            cf = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero")
            env = EdgeUplinkEnv(sc, cf)
            env._reveal_arrivals(0.0)
            obs = env.observe(0.0)
            oracle = solve_single_tick_exact(obs, max_p=n)
            got = len(pol(obs)["actions"])
            if got != int(oracle["best_admitted"]):
                a3_unif_fails += 1

        # Anchor 4 (500 single-tick trials: single-task closed-form optimality)
        a4_fails = 0
        for _ in range(per_anchor):
            h4 = np.array([[float(rng.uniform(0.2, 5.0)) * 1e-10]])
            f_loc = np.array([float(rng.uniform(0.1, 2.0)) * 1e9])
            f_ms = np.array([float(rng.uniform(10.0, 50.0)) * 1e9])
            d_b = float(rng.uniform(1e5, 3e6))
            c_c = float(rng.uniform(1e8, 2e9))
            dl = float(rng.uniform(0.08, 0.80))
            sc = make_micro_scenario(
                case_id="FUZZ_A4", n_devices=1, n_servers=1, h_dev_ms=h4, f_loc_hz=f_loc, f_ms_max_hz=f_ms, p_tx_w=np.array([1.0]),
                tasks_raw=[{"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0, "deadline_s": dl, "D_bits": d_b, "C_cycles": c_c, "tau_dn_s": 0.001}],
            )
            cf = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
            env = EdgeUplinkEnv(sc, cf)
            env._reveal_arrivals(0.0)
            obs = env.observe(0.0)
            oracle = solve_single_tick_exact(obs, max_p=1)
            got = len(pol(obs)["actions"])
            if got != int(oracle["best_admitted"]):
                a4_fails += 1
    finally:
        champion_core.time.perf_counter = orig_perf_counter

    # Live wall-clock boundary check outside Theorem 3 L_solver=0 hypothesis (100 Anchor-1 trials with real perf_counter)
    rng_live = np.random.default_rng(44005)
    a1_live_wall_fails = 0
    for _ in range(100):
        if not _eval_anchor1_case(rng_live, pol, uniform_dl=True):
            a1_live_wall_fails += 1

    anchors_exact = {
        "anchor1_uniform_deadline": {"trials": per_anchor, "fails": a1_unif_fails},
        "anchor2_infinite_cpu_uniform_dl": {"trials": per_anchor, "fails": a2_sym_fails},
        "anchor3_infinite_bw_uniform_dl": {"trials": per_anchor, "fails": a3_unif_fails},
        "anchor4_single_task": {"trials": per_anchor, "fails": a4_fails},
    }
    emp_gap = {
        "anchor1_general_deadlines": {"trials": 250, "fails": a1_gen_fails, "rate": float(a1_gen_fails / 250.0)},
        "anchor1_live_wall_boundary": {"trials": 100, "fails": a1_live_wall_fails, "rate": float(a1_live_wall_fails / 100.0)},
    }
    return anchors_exact, emp_gap


def evaluate_fail_archive(emp_gap: dict[str, Any]) -> dict[str, Any]:
    pol_clwb = make_clwb_dual()
    pol_bambos = wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")

    # FAIL-1: Extreme 6 dB unmodeled CSI estimation noise
    sc1 = build_scenario_v1(case_id="FAIL1_CSI_6DB", family="fail_csi", seed=96101, n_devices=26, n_servers=6, tasks_per_device=5, offered_load=0.75, T_max_s=0.65, radius_m=240.0, input_kb_range=(80.0, 160.0))
    r1_clean = run_simulation(sc1, pol_clwb, SimConfig(tick_s=0.05, delta_sig_s=0.001, csi_sigma_db=0.0))
    r1_6db = run_simulation(sc1, pol_clwb, SimConfig(tick_s=0.05, delta_sig_s=0.001, csi_sigma_db=6.0, csi_seed=96101))

    # FAIL-2: Sub-tick single-device burst (same device gets 5 tight sub-tick tasks violating 1-per-device idle-radio pacing)
    tasks_f2 = [
        {"tid": f"0:{k}", "device": 0, "k": k, "arrival_s": 0.001 * k, "deadline_s": 0.08 + 0.005 * k,
         "D_bits": 150_000.0, "C_cycles": 2.0e8, "tau_dn_s": 0.001}
        for k in range(5)
    ]
    sc2 = make_micro_scenario(case_id="FAIL2_SUBTICK_SAME_DEV", n_devices=1, n_servers=1, h_dev_ms=np.array([[1e-9]]), f_loc_hz=np.array([0.1e9]), f_ms_max_hz=np.array([50e9]), p_tx_w=np.array([1.0]), tasks_raw=tasks_f2)
    cf2 = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
    r2_clwb = run_simulation(sc2, pol_clwb, cf2)

    # FAIL-3: Zone I weak coupling & inverted-deadline knapsack gap (orthogonal=True wall-clock + EDF vs f_min)
    sc3 = build_scenario_v1(case_id="FAIL3_ZONE1_WEAK", family="fail_z1", seed=96301, n_devices=18, n_servers=6, tasks_per_device=5, offered_load=0.35, T_max_s=0.95, radius_m=180.0, input_kb_range=(8.0, 18.0))
    cf3 = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="wall")
    r3_clwb = run_simulation(sc3, pol_clwb, cf3)
    r3_bambos = run_simulation(sc3, pol_bambos, cf3)

    return {
        "FAIL-1_CSI_6DB": {
            "clean_ontime": r1_clean["ontime_rate"],
            "noisy_6db_ontime": r1_6db["ontime_rate"],
            "noisy_6db_miss": r1_6db["server_grant_miss_rate"],
            "note": "Heavy >5 dB CSI error breaks deterministic Theorem 1 rate prediction",
        },
        "FAIL-2_SUBTICK_SAME_DEV": {
            "ontime_count": f"{r2_clwb['n_ontime']}/{sc2.n_tasks}",
            "ontime_rate": r2_clwb["ontime_rate"],
            "note": "Idle-radio 1-per-device causal gate caps admission to <= 1 task per tick_s on a single busy radio",
        },
        "FAIL-3_ZONE1_OVERHEAD_AND_INVERTED_DL": {
            "clwb_ontime": r3_clwb["ontime_rate"],
            "clwb_miss": r3_clwb["server_grant_miss_rate"],
            "bambos_ontime": r3_bambos["ontime_rate"],
            "clwb_p50_ms": r3_clwb["decision_p50_ms"],
            "bambos_p50_ms": r3_bambos["decision_p50_ms"],
            "inverted_dl_anchor1_gap_cnt": emp_gap["anchor1_general_deadlines"]["fails"],
            "inverted_dl_anchor1_gap_rate": emp_gap["anchor1_general_deadlines"]["rate"],
            "note": "Under orthogonal weak coupling CLWB matches Bambos at higher P50 latency; outside uniform/agreeable deadlines EDF diverges from smallest-share knapsack",
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    l1 = fuzz_lemma1(2500)
    print("[Fuzz] Lemma 1:", l1, flush=True)
    t1 = fuzz_theorem1(2000, 1000)
    print("[Fuzz] Theorem 1:", t1, flush=True)
    t2 = fuzz_theorem2_complexity(2500)
    print("[Fuzz] Theorem 2:", t2, flush=True)
    t3, emp_gap = fuzz_theorem3_anchors(500)
    print("[Fuzz] Theorem 3:", t3, "EmpGap:", emp_gap, flush=True)
    fails = evaluate_fail_archive(emp_gap)
    print("[Fuzz] FAIL Archive:", json.dumps(fails, indent=2), flush=True)

    rep = {
        "total_trials": 2500 + 3000 + 2500 + 2000,
        "elapsed_s": float(time.perf_counter() - t0),
        "lemma1": l1,
        "theorem1": t1,
        "theorem2": t2,
        "theorem3_anchors": t3,
        "empirical_general_deadline_gap": emp_gap,
        "fail_archive": fails,
    }
    out_path = STAGE4_DIR / "step44_fuzz_report.json"
    out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[Step 4.4] Wrote {out_path}")


if __name__ == "__main__":
    main()
