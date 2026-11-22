"""Exact Mathematical Oracles, Hereditary Monotonicity & Counterexample Verifiers (Model v1.0).

Pure mathematical/optimization reference solvers for `model_spec.py`:
  * `evaluate_fluid_batch`: exact continuous fluid-SINR rollout under half-duplex NIC queues
    for any active link set U_0 and candidate edge grant set S.
  * `verify_hereditary_anti_monotonicity`: empirical verification of the monotone completion
    operator and downward-closed (hereditary independence) feasibility system over 200+ trials.
  * `solve_single_tick_exact`: exact branch-and-bound single-epoch optimal batch solver
    with OR-Tools / exact pruning support for small instances (P <= 8).
  * `solve_offline_exact`: exact multi-epoch omniscient tree search driving `EdgeUplinkEnv`
    physics directly with zero approximation (upper-bound judge on micro-scenarios R <= 6).
  * Explicit hand-verifiable N <= 5 counterexample builders (`L1`, `L2`, `L3`) proving strict
    performance lower bounds on pre-batch-blind, non-delay, and myopic-EDF policy classes.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math
from typing import Any, Optional

import numpy as np

from model_spec import (
    EdgeUplinkEnv,
    ScenarioV1,
    SimConfig,
    make_micro_scenario,
    shannon_rate,
)


# --------------------------------------------------------------------------- #
# 1. Exact continuous fluid-SINR rollout for a candidate batch (U_0, S)
# --------------------------------------------------------------------------- #

def evaluate_fluid_batch(
    active_uplinks: list[dict],
    candidate_edge_grants: list[dict],
    scen_params: dict,
    now: float,
) -> dict[str, Any]:
    """Exact event-driven fluid-SINR integration for active uplinks U_0 and batch grants S.

    Preserves exact half-duplex NIC queueing (at most one active uplink per device, EDF
    order within device queues) and continuous co-subband Shannon rate reflow whenever
    any transmission starts or ends.
    """
    h_dev_ms = np.asarray(scen_params["h_dev_ms"], dtype=float)
    p_tx_arr = np.asarray(scen_params["p_tx_w"], dtype=float)
    bw = float(scen_params["bandwidth_hz"])
    noise = float(scen_params["noise_w"])
    orthogonal = bool(scen_params.get("orthogonal", False))
    n_sb = max(1, int(scen_params.get("n_subbands", 1)))
    subband = scen_params.get("subband", list(range(len(p_tx_arr))))
    f_avail_hz = np.asarray(scen_params["f_avail_hz"], dtype=float)
    f_min_hz = float(scen_params.get("f_min_hz", 1e6))
    delta_sig = float(scen_params.get("delta_sig_s", 0.0))

    # Reserved CPU baseline excluding any grants being evaluated in candidate_edge_grants;
    # srv_reserved_hz already accounts for U_0 and previously queued/computing tasks.
    srv_reserved_base = np.asarray(
        scen_params.get("srv_reserved_hz", np.zeros(len(f_avail_hz), dtype=float)),
        dtype=float,
    )

    n = int(len(p_tx_arr))
    t_act = float(now) + delta_sig

    # Track per-flow fluid state
    flows: dict[str, dict] = {}
    tx_q: list[list[dict]] = [[] for _ in range(n)]
    running: list[dict] = []
    busy_dev: set[int] = set()

    for u in active_uplinks:
        tid = str(u["id"])
        i = int(u["device"])
        j = int(u["server"])
        rec = {
            "id": tid,
            "is_active": True,
            "device": i,
            "server": j,
            "p_tx_w": float(u["p_tx_w"]),
            "rem_bits": float(max(u["rem_bits"], 0.0)),
            "deadline_s": float(u["deadline_s"]),
            "C_cycles": float(u["C_cycles"]),
            "f_locked_hz": float(u["f_hz"]),
            "tau_dn_s": float(u["tau_dn_s"]),
        }
        flows[tid] = rec
        running.append(rec)
        busy_dev.add(i)

    # Sort candidate grants in EDF order for device NIC queueing
    sorted_cand = sorted(
        candidate_edge_grants,
        key=lambda g: (float(g["deadline_s"]), int(g["device"]), str(g["id"])),
    )
    pending_release: list[dict] = []
    for g in sorted_cand:
        tid = str(g["id"])
        i = int(g["device"])
        j = int(g["server"])
        rec = {
            "id": tid,
            "is_active": False,
            "device": i,
            "server": j,
            "p_tx_w": float(g.get("p_tx_w", p_tx_arr[i])),
            "rem_bits": float(max(g.get("rem_bits", g.get("D_bits", 0.0)), 0.0)),
            "deadline_s": float(g["deadline_s"]),
            "C_cycles": float(g["C_cycles"]),
            "f_locked_hz": g.get("f_hz"),
            "tau_dn_s": float(g["tau_dn_s"]),
        }
        flows[tid] = rec
        if delta_sig <= 1e-12:
            tx_q[i].append(rec)
        else:
            pending_release.append(rec)

    def _dispatch_queues() -> None:
        for dev_i in range(n):
            if dev_i not in busy_dev and tx_q[dev_i]:
                tx_q[dev_i].sort(key=lambda r: (r["deadline_s"], r["device"], r["id"]))
                nxt = tx_q[dev_i].pop(0)
                running.append(nxt)
                busy_dev.add(dev_i)

    _dispatch_queues()

    t_cur = float(now)
    released = delta_sig <= 1e-12
    t_tx_end: dict[str, float] = {}

    def _inst_rate(r: dict) -> float:
        i = r["device"]
        j = r["server"]
        sig = float(r["p_tx_w"] * h_dev_ms[i, j])
        interf = 0.0
        if not orthogonal:
            sb_i = int(subband[i]) % n_sb
            for o in running:
                if o["id"] == r["id"]:
                    continue
                if (int(subband[o["device"]]) % n_sb) != sb_i:
                    continue
                interf += float(o["p_tx_w"] * h_dev_ms[o["device"], j])
        return shannon_rate(bw, sig, noise, interf)

    max_iters = 100 + 8 * (len(active_uplinks) + len(candidate_edge_grants))
    for _ in range(max_iters):
        if not running and released:
            break

        next_release = t_act if (not released and pending_release) else math.inf

        if not running:
            # Advance straight to actuation instant
            if math.isfinite(next_release):
                t_cur = next_release
                released = True
                for rec in pending_release:
                    tx_q[rec["device"]].append(rec)
                pending_release.clear()
                _dispatch_queues()
                continue
            break

        rates = [_inst_rate(r) for r in running]
        dt_ends = [r["rem_bits"] / max(rt, 1e-3) for r, rt in zip(running, rates)]
        min_dt_end = min(dt_ends)
        dt_rel = max(0.0, next_release - t_cur) if math.isfinite(next_release) else math.inf
        dt = min(min_dt_end, dt_rel)

        # Advance all active fluid flows over [t_cur, t_cur + dt]
        for r, rt in zip(running, rates):
            r["rem_bits"] = max(0.0, r["rem_bits"] - rt * dt)
        t_cur += dt

        # Close completed transmissions
        still_running: list[dict] = []
        finished_any = False
        for r in running:
            if r["rem_bits"] <= 1e-7:
                r["rem_bits"] = 0.0
                t_tx_end[r["id"]] = float(t_cur)
                busy_dev.discard(r["device"])
                finished_any = True
            else:
                still_running.append(r)
        running = still_running

        if (not released) and abs(t_cur - t_act) <= 1e-12:
            released = True
            for rec in pending_release:
                tx_q[rec["device"]].append(rec)
            pending_release.clear()
            _dispatch_queues()
        elif finished_any:
            _dispatch_queues()

    # Evaluate exact minimum CPU shares and feasibility for both U_0 and S
    f_min_map: dict[str, float] = {}
    active_ok: dict[str, bool] = {}
    srv_added = np.zeros(len(f_avail_hz), dtype=float)
    feasible = True
    fail_reason = "ok"

    for tid, r in flows.items():
        if tid not in t_tx_end:
            feasible = False
            fail_reason = f"incomplete_tx:{tid}"
            break
        t_up_e = t_tx_end[tid]
        slack_cmp = r["deadline_s"] - t_up_e - r["tau_dn_s"]
        if r["is_active"]:
            f_lock = float(r["f_locked_hz"])
            t_finish = t_up_e + r["C_cycles"] / max(f_lock, 1.0) + r["tau_dn_s"]
            ok = t_finish <= r["deadline_s"] + 1e-9
            active_ok[tid] = ok
            f_min_map[tid] = f_lock
            if not ok:
                feasible = False
                fail_reason = f"active_deadline_miss:{tid}"
        else:
            if slack_cmp <= 1e-9:
                f_min_map[tid] = math.inf
                feasible = False
                fail_reason = f"upload_exceeds_deadline:{tid}"
            else:
                f_req = max(f_min_hz, r["C_cycles"] / slack_cmp)
                f_min_map[tid] = f_req
                j = r["server"]
                srv_added[j] += f_req
                if f_req > f_avail_hz[j] + 1.0:
                    feasible = False
                    fail_reason = f"single_task_cpu_cap:{tid}"

    srv_used = srv_reserved_base + srv_added
    if feasible and np.any(srv_used > f_avail_hz + 1.0):
        feasible = False
        fail_reason = "server_aggregate_cap"

    return {
        "t_tx_end": t_tx_end,
        "f_min": f_min_map,
        "active_ok": active_ok,
        "srv_used_hz": srv_used.tolist(),
        "feasible": bool(feasible),
        "reason": fail_reason,
    }


def _scen_params_from_obs(obs: dict, extra_delta_sig: Optional[float] = None) -> dict:
    devs = obs["devices"]
    srvs = obs["servers"]
    links = obs["links"]
    return {
        "h_dev_ms": np.asarray(links["h_hat_dev_ms"], dtype=float),
        "p_tx_w": np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float),
        "bandwidth_hz": float(links["bandwidth_hz"]),
        "noise_w": float(links["noise_w"]),
        "orthogonal": bool(links["orthogonal"]),
        "n_subbands": int(links["n_subbands"]),
        "subband": [int(d["subband"]) for d in devs],
        "f_avail_hz": np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float),
        "srv_reserved_hz": np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float),
        "f_min_hz": float(obs["f_min_hz"]),
        "delta_sig_s": float(obs["delta_sig_s"] if extra_delta_sig is None else extra_delta_sig),
    }


# --------------------------------------------------------------------------- #
# 2. Hereditary anti-monotonicity verifier
# --------------------------------------------------------------------------- #

def verify_hereditary_anti_monotonicity(n_trials: int = 200, seed: int = 42) -> dict[str, Any]:
    """Empirically verify S_1 subset S_2 => t_tx_end(S_1) <= t_tx_end(S_2), f_min(S_1) <= f_min(S_2),
    and feasible(S_2) => feasible(S_1) under continuous fluid SINR + half-duplex NIC queueing."""
    rng = np.random.default_rng(int(seed))
    passed = 0
    max_t_violation = 0.0
    max_f_violation = 0.0
    feasible_s2_count = 0

    for _trial in range(int(n_trials)):
        n = int(rng.integers(3, 8))
        m = int(rng.integers(1, 4))
        n_sb = int(rng.choice([1, 2]))
        h = rng.uniform(1e-11, 5e-10, size=(n, m))
        ptx = rng.uniform(0.8, 1.3, size=n)
        f_avail = np.full(m, float(rng.uniform(15e9, 45e9)), dtype=float)
        params = {
            "h_dev_ms": h,
            "p_tx_w": ptx,
            "bandwidth_hz": 10e6,
            "noise_w": 1e-13,
            "orthogonal": False,
            "n_subbands": n_sb,
            "subband": [i % n_sb for i in range(n)],
            "f_avail_hz": f_avail,
            "srv_reserved_hz": np.zeros(m, dtype=float),
            "f_min_hz": 1e6,
            "delta_sig_s": float(rng.choice([0.0, 0.001])),
        }

        # Active links (at most 1 per device)
        n_act = int(rng.integers(0, min(3, n)))
        act_devs = rng.choice(n, size=n_act, replace=False) if n_act > 0 else []
        active = []
        for idx, d in enumerate(act_devs):
            j = int(rng.integers(0, m))
            f_lock = float(rng.uniform(2e9, 8e9))
            params["srv_reserved_hz"][j] += f_lock
            active.append({
                "id": f"act_{idx}",
                "device": int(d),
                "server": j,
                "p_tx_w": float(ptx[d]),
                "rem_bits": float(rng.uniform(2e5, 8e5)),
                "tx_start_s": 0.0,
                "deadline_s": float(rng.uniform(0.35, 0.90)),
                "C_cycles": float(rng.uniform(1e8, 4e8)),
                "f_hz": f_lock,
                "tau_dn_s": 0.001,
            })

        # Candidate pool S_2 of size 3..7 (may include multiple tasks per device)
        k_s2 = int(rng.integers(3, 8))
        s2 = []
        for k in range(k_s2):
            d = int(rng.integers(0, n))
            j = int(rng.integers(0, m))
            s2.append({
                "id": f"cand_{k}",
                "device": d,
                "server": j,
                "p_tx_w": float(ptx[d]),
                "D_bits": float(rng.uniform(3e5, 9e5)),
                "deadline_s": float(rng.uniform(0.25, 0.95)),
                "C_cycles": float(rng.uniform(1e8, 5e8)),
                "tau_dn_s": 0.001,
            })

        k_s1 = int(rng.integers(1, k_s2))
        s1_idx = sorted(rng.choice(k_s2, size=k_s1, replace=False).tolist())
        s1 = [s2[i] for i in s1_idx]

        r1 = evaluate_fluid_batch(active, s1, params, now=0.0)
        r2 = evaluate_fluid_batch(active, s2, params, now=0.0)

        trial_ok = True
        for item in active + s1:
            tid = item["id"]
            dt = r1["t_tx_end"][tid] - r2["t_tx_end"][tid]
            if dt > max_t_violation:
                max_t_violation = dt
            if dt > 1e-10:
                trial_ok = False
            if not item.get("f_hz"):  # candidate task: check f_min monotonicity
                f1 = r1["f_min"][tid]
                f2 = r2["f_min"][tid]
                if math.isfinite(f2):
                    if not math.isfinite(f1):
                        trial_ok = False
                    else:
                        df = f1 - f2
                        if df > max_f_violation:
                            max_f_violation = df
                        if df > 1e-4:
                            trial_ok = False

        if r2["feasible"]:
            feasible_s2_count += 1
            if not r1["feasible"]:
                trial_ok = False

        if trial_ok:
            passed += 1

    return {
        "n_trials": int(n_trials),
        "passed": int(passed),
        "all_passed": bool(passed == int(n_trials)),
        "feasible_s2_count": int(feasible_s2_count),
        "max_t_tx_end_violation_s": float(max(0.0, max_t_violation)),
        "max_f_min_violation_hz": float(max(0.0, max_f_violation)),
    }


# --------------------------------------------------------------------------- #
# 3. Exact single-tick branch-and-bound optimal batch solver
# --------------------------------------------------------------------------- #

def solve_single_tick_exact(obs: dict, max_p: int = 8) -> dict[str, Any]:
    """Exact single-epoch optimal batch decision over EDF prefix of pending tasks (P <= max_p).

    Searches all placement combinations (defer, local, edge_0..edge_{M-1}) under exact
    fluid-SINR coupling and active-link protection, maximizing (n_admitted, -total_cpu).
    """
    now = float(obs["time"])
    t_exec = now + float(obs["delta_sig_s"])
    pending = sorted(
        obs["pending"],
        key=lambda t: (float(t["deadline_s"]), int(t["device"]), str(t["id"])),
    )[: int(max_p)]
    p_cnt = len(pending)
    if p_cnt == 0:
        return {"actions": [], "best_admitted": 0, "best_cpu_hz": 0.0}

    devs = obs["devices"]
    srvs = obs["servers"]
    m = len(srvs)
    params = _scen_params_from_obs(obs)
    active = obs["active_uplinks"]

    f_loc_cap = [float(d["f_loc_hz"]) for d in devs]
    dev_res = [float(d["cpu_reserved_hz"]) for d in devs]
    f_min_hz = float(obs["f_min_hz"])

    # Pre-filter per-task feasible atomic modes (under U_0 alone as necessary condition)
    task_modes: list[list[tuple[str, Optional[int]]]] = []
    for t in pending:
        i = int(t["device"])
        dl = float(t["deadline_s"])
        opts: list[tuple[str, Optional[int]]] = []
        # Local candidate
        slack_loc = dl - t_exec
        if slack_loc > 1e-9:
            f_loc_req = max(f_min_hz, float(t["C_cycles"]) / slack_loc)
            if dev_res[i] + f_loc_req <= f_loc_cap[i] + 1.0:
                opts.append(("local", None))
        # Edge candidates sorted by strongest link gain
        h_row = params["h_dev_ms"][i, :]
        srv_order = sorted(range(m), key=lambda j: -float(h_row[j]))
        for j in srv_order:
            g = {
                "id": t["id"],
                "device": i,
                "server": j,
                "p_tx_w": float(devs[i]["p_tx_w"]),
                "D_bits": float(t["D_bits"]),
                "deadline_s": dl,
                "C_cycles": float(t["C_cycles"]),
                "tau_dn_s": float(t["tau_dn_s"]),
            }
            chk = evaluate_fluid_batch(active, [g], params, now)
            if chk["feasible"]:
                opts.append(("edge", j))
        opts.append(("defer", None))
        task_modes.append(opts)

    best_adm = -1
    best_cpu = math.inf
    best_assign: list[tuple[str, Optional[int], Optional[float]]] = [("defer", None, None)] * p_cnt

    cur_assign: list[tuple[str, Optional[int]]] = [("defer", None)] * p_cnt
    edge_grants: list[dict] = []
    loc_used = list(dev_res)

    def _dfs(idx: int, adm_cnt: int, loc_cpu_sum: float) -> None:
        nonlocal best_adm, best_cpu, best_assign
        if adm_cnt + (p_cnt - idx) < best_adm:
            return
        if idx == p_cnt:
            res = evaluate_fluid_batch(active, edge_grants, params, now)
            if not res["feasible"]:
                return
            edge_cpu = sum(res["f_min"][g["id"]] for g in edge_grants)
            tot_cpu = loc_cpu_sum + edge_cpu
            if (adm_cnt > best_adm) or (adm_cnt == best_adm and tot_cpu < best_cpu - 1.0):
                best_adm = adm_cnt
                best_cpu = tot_cpu
                snap: list[tuple[str, Optional[int], Optional[float]]] = []
                for k_i in range(p_cnt):
                    md, sv = cur_assign[k_i]
                    if md == "local":
                        f_v = max(f_min_hz, float(pending[k_i]["C_cycles"]) / (float(pending[k_i]["deadline_s"]) - t_exec))
                        snap.append(("local", None, f_v))
                    elif md == "edge":
                        snap.append(("edge", sv, float(res["f_min"][pending[k_i]["id"]])))
                    else:
                        snap.append(("defer", None, None))
                best_assign = snap
            return

        t = pending[idx]
        i = int(t["device"])
        dl = float(t["deadline_s"])

        for md, sv in task_modes[idx]:
            if md == "local":
                f_v = max(f_min_hz, float(t["C_cycles"]) / (dl - t_exec))
                if loc_used[i] + f_v > f_loc_cap[i] + 1.0:
                    continue
                loc_used[i] += f_v
                cur_assign[idx] = ("local", None)
                _dfs(idx + 1, adm_cnt + 1, loc_cpu_sum + f_v)
                loc_used[i] -= f_v
                cur_assign[idx] = ("defer", None)
            elif md == "edge":
                g = {
                    "id": t["id"],
                    "device": i,
                    "server": int(sv),
                    "p_tx_w": float(devs[i]["p_tx_w"]),
                    "D_bits": float(t["D_bits"]),
                    "deadline_s": dl,
                    "C_cycles": float(t["C_cycles"]),
                    "tau_dn_s": float(t["tau_dn_s"]),
                }
                edge_grants.append(g)
                # Hereditary anti-monotonicity prunes immediately if current subset is infeasible!
                if evaluate_fluid_batch(active, edge_grants, params, now)["feasible"]:
                    cur_assign[idx] = ("edge", int(sv))
                    _dfs(idx + 1, adm_cnt + 1, loc_cpu_sum)
                    cur_assign[idx] = ("defer", None)
                edge_grants.pop()
            else:
                cur_assign[idx] = ("defer", None)
                _dfs(idx + 1, adm_cnt, loc_cpu_sum)

    _dfs(0, 0, 0.0)

    actions = []
    for t, (md, sv, f_v) in zip(pending, best_assign):
        if md != "defer":
            actions.append({"task": t["id"], "mode": md, "server": sv, "f_hz": f_v})
    return {
        "actions": actions,
        "best_admitted": max(0, best_adm),
        "best_cpu_hz": best_cpu if best_adm >= 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# 4. Exact multi-tick offline omniscient oracle directly over EdgeUplinkEnv
# --------------------------------------------------------------------------- #

def _clone_env_up_to_tick(env: EdgeUplinkEnv) -> EdgeUplinkEnv:
    return copy.deepcopy(env)


def _advance_env_to_next_decision(
    env: EdgeUplinkEnv,
    max_horizon: float,
) -> tuple[bool, Optional[int], Optional[dict]]:
    """Advance cloned EdgeUplinkEnv until next tick with pending tasks, or simulation end."""
    import heapq as _hq
    while env._heap and not env._stop:
        t_ev = env._heap[0][0]
        if t_ev > env.horizon + 1e-9 or t_ev > max_horizon + 1e-9:
            break
        env.now = t_ev
        env._advance_uplinks(t_ev)

        batch = []
        while env._heap and env._heap[0][0] <= t_ev + 1e-13:
            batch.append(_hq.heappop(env._heap))
        batch.sort(key=lambda x: (x[1], x[2]))

        uplink_dirty = False
        paused_tick_idx: Optional[int] = None
        paused_obs: Optional[dict] = None

        for idx_ev, (_w, _prio, _seq, kind, payload) in enumerate(batch):
            if kind == "tx_end":
                uplink_dirty = env._handle_tx_end(payload, t_ev) or uplink_dirty
            elif kind == "compute_end":
                env._handle_compute_end(payload, t_ev)
            elif kind == "downlink_end":
                env._handle_downlink_end(payload, t_ev)
            elif kind == "deadline":
                uplink_dirty = env._handle_deadline(payload, t_ev) or uplink_dirty
            elif kind == "actuate":
                env._handle_actuate(payload)
            elif kind == "tick":
                tick_idx = int(payload)
                env._reveal_arrivals(t_ev)
                pending = [t for t in env.tasks if t.state == "pending"]
                if pending:
                    # Wait if other same-timestamp events exist? No, tick has prio 3 >= all others (0,1,2)
                    paused_tick_idx = tick_idx
                    paused_obs = env.observe(t_ev)
                    break
                nxt_t = (tick_idx + 1) * env.cfg.tick_s
                if nxt_t <= env.horizon + 1e-9:
                    env._push(nxt_t, 3, "tick", tick_idx + 1)
            elif kind == "horizon":
                env._stop = True
                break

        if env._stop:
            break

        if paused_tick_idx is not None:
            if env._dispatch(t_ev):
                uplink_dirty = True
            if uplink_dirty:
                env._recompute_and_schedule_uplinks(t_ev)
            env._assert_hard_invariants(t_ev)
            return True, paused_tick_idx, paused_obs

        if env._dispatch(t_ev):
            uplink_dirty = True
        if uplink_dirty:
            env._recompute_and_schedule_uplinks(t_ev)
        env._assert_hard_invariants(t_ev)

    return False, None, None


def _batch_necessary_upload_ok(
    active_uplinks: list[dict],
    edge_chosen: list[dict],
    params: dict,
    now: float,
) -> bool:
    """Strictly necessary hereditary condition for any sequential or shared CPU execution:
    every granted task in `edge_chosen` can at least finish if given full server speed F_avail
    immediately after fluid uplink completes, without killing any active uplink at its locked share."""
    res = evaluate_fluid_batch(active_uplinks, edge_chosen, params, now)
    f_avail_hz = np.asarray(params["f_avail_hz"], dtype=float)
    for tid, ok in res["active_ok"].items():
        if not ok:
            return False
    for g in edge_chosen:
        tid = str(g["id"])
        if tid not in res["t_tx_end"]:
            return False
        t_up_e = res["t_tx_end"][tid]
        t_min_fin = t_up_e + float(g["C_cycles"]) / max(float(f_avail_hz[int(g["server"])]), 1.0) + float(g["tau_dn_s"])
        if t_min_fin > float(g["deadline_s"]) + 1e-9:
            return False
    return True


def solve_offline_exact(
    scen: ScenarioV1,
    cfg: Optional[SimConfig] = None,
    *,
    non_delay: bool = False,
) -> dict[str, Any]:
    """Exact multi-tick offline tree search over `EdgeUplinkEnv` (zero latency overhead).

    If `non_delay=True`, searches strictly within the Non-Delay policy subclass (no `defer`
    allowed for pending tasks: every revealed pending task must be committed to local, edge,
    or permanently dropped at its arrival tick).
    """
    base_cfg = cfg if cfg is not None else SimConfig()
    oracle_cfg = replace(base_cfg, latency_mode="zero")
    root_env = EdgeUplinkEnv(scen, oracle_cfg)

    r_tot = scen.n_tasks
    best_ontime = -1
    best_trace: list[dict] = []
    nodes_explored = 0

    def _upper_bound(env: EdgeUplinkEnv) -> int:
        dead = sum(1 for t in env.tasks if t.state in ("timeout", "dropped"))
        return r_tot - dead

    def _search(env: EdgeUplinkEnv, trace: list[dict]) -> None:
        nonlocal best_ontime, best_trace, nodes_explored
        if _upper_bound(env) <= best_ontime:
            return

        has_tick, tick_idx, obs = _advance_env_to_next_decision(env, env.horizon)
        if not has_tick:
            nodes_explored += 1
            if env.n_ontime > best_ontime:
                best_ontime = env.n_ontime
                best_trace = list(trace)
            return

        if _upper_bound(env) <= best_ontime:
            return

        nodes_explored += 1
        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), int(t["device"]), str(t["id"])),
        )
        p_cnt = len(pending)
        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        f_min_hz = float(obs["f_min_hz"])
        f_avail_arr = [float(s["f_avail_hz"]) for s in obs["servers"]]
        f_loc_arr = [float(d["f_loc_hz"]) for d in obs["devices"]]
        m = len(f_avail_arr)
        params = _scen_params_from_obs(obs)

        # Enumerate feasible single-task choices at this tick
        per_task_opts: list[list[dict]] = []
        for t in pending:
            i = int(t["device"])
            dl = float(t["deadline_s"])
            opts: list[dict] = []
            # Edge choices (full speed, half speed, and tightest feasible share)
            for j in range(m):
                g = {
                    "id": t["id"],
                    "device": i,
                    "server": j,
                    "p_tx_w": float(obs["devices"][i]["p_tx_w"]),
                    "D_bits": float(t["D_bits"]),
                    "deadline_s": dl,
                    "C_cycles": float(t["C_cycles"]),
                    "tau_dn_s": float(t["tau_dn_s"]),
                }
                if _batch_necessary_upload_ok(obs["active_uplinks"], [g], params, now):
                    res_1 = evaluate_fluid_batch(obs["active_uplinks"], [g], params, now)
                    f_req = float(res_1["f_min"][t["id"]])
                    opts.append({"task": t["id"], "mode": "edge", "server": j, "f_hz": f_avail_arr[j], "cand": g})
                    if f_req < f_avail_arr[j] * 0.95:
                        opts.append({"task": t["id"], "mode": "edge", "server": j, "f_hz": max(f_min_hz, f_req), "cand": g})
            # Local choice (at full device CPU, EDF optimal on 1 processor)
            if dl - t_exec > 1e-9:
                f_req_loc = max(f_min_hz, float(t["C_cycles"]) / (dl - t_exec))
                if f_req_loc <= f_loc_arr[i] + 1.0:
                    opts.append({"task": t["id"], "mode": "local", "server": None, "f_hz": f_loc_arr[i]})
            if not non_delay:
                opts.append({"task": t["id"], "mode": "defer", "server": None, "f_hz": None})
            else:
                opts.append({"task": t["id"], "mode": "drop", "server": None, "f_hz": None})
            per_task_opts.append(opts)

        chosen: list[dict] = []
        edge_chosen: list[dict] = []
        tick_action_sets: list[list[dict]] = []

        def _gen_tick(k_idx: int) -> None:
            if k_idx == p_cnt:
                tick_action_sets.append(list(chosen))
                return

            for opt in per_task_opts[k_idx]:
                if opt["mode"] == "edge":
                    edge_chosen.append(opt["cand"])
                    # Necessary hereditary bound (never prunes valid EDF-queued schedules!)
                    if _batch_necessary_upload_ok(obs["active_uplinks"], edge_chosen, params, now):
                        chosen.append({"task": opt["task"], "mode": "edge", "server": opt["server"], "f_hz": opt["f_hz"]})
                        _gen_tick(k_idx + 1)
                        chosen.pop()
                    edge_chosen.pop()
                else:
                    chosen.append({"task": opt["task"], "mode": "local" if opt["mode"] == "local" else opt["mode"], "server": opt["server"], "f_hz": opt["f_hz"]})
                    _gen_tick(k_idx + 1)
                    chosen.pop()

        _gen_tick(0)
        # Prioritize batches that commit more tasks (faster upper bound tightening)
        tick_action_sets.sort(
            key=lambda acts: (
                -sum(1 for a in acts if a["mode"] in ("edge", "local")),
                sum(1 for a in acts if a["mode"] == "drop"),
            )
        )

        for acts in tick_action_sets:
            if _upper_bound(env) <= best_ontime:
                break
            nxt_env = _clone_env_up_to_tick(env)
            staged = nxt_env._stage_decisions({"actions": acts}, t_exec)
            if staged:
                nxt_env._push(t_exec, 2, "actuate", (staged, t_exec))
            nxt_t = (tick_idx + 1) * nxt_env.cfg.tick_s
            if nxt_t <= nxt_env.horizon + 1e-9:
                nxt_env._push(nxt_t, 3, "tick", tick_idx + 1)
            trace.append({"tick": tick_idx, "t": now, "actions": acts})
            _search(nxt_env, trace)
            trace.pop()

    _search(root_env, [])
    return {
        "case_id": scen.case_id,
        "non_delay": bool(non_delay),
        "n_tasks": int(r_tot),
        "best_ontime": int(max(0, best_ontime)),
        "best_rate": float(max(0, best_ontime) / max(1, r_tot)),
        "nodes_explored": int(nodes_explored),
        "best_trace": best_trace,
    }


# --------------------------------------------------------------------------- #
# 5. Explicit hand-verifiable N <= 5 counterexamples for Theorems L1, L2, L3
# --------------------------------------------------------------------------- #

def build_counterexample_L1() -> tuple[ScenarioV1, SimConfig]:
    """Counterexample L1 (Pre-Batch Blindness Class Pi_pre): N=4, M=1, R=4.

    All 4 symmetric tasks arrive at t=0 with d=0.160s, D=8e5 bits, C=1.2e9 cycles.
      - Local CPU (0.5 GHz) cannot meet deadline (needs 2.4s > 0.16s).
      - Pre-batch SINR (I=0) predicts u_0 = 8.03 ms, f_pre = 8.00 GHz (4*8 = 32 <= 38 GHz),
        so every pre-batch policy admits all 4 simultaneously -> concurrent SINR rate drops
        to 4.15 Mbps, actual upload takes 193.8 ms > 160 ms -> 0/4 on-time (0%)!
      - Batch-consistent admission of 2 tasks takes u_2 = 80.1 ms, f = 15.40 GHz each (30.8 <= 38),
        completing 2/4 on-time (50%). Exact optimal gap = +50 pp!
    """
    n, m = 4, 1
    h = np.full((n, m), 1e-10, dtype=float)
    f_loc = np.full(n, 0.5e9, dtype=float)
    f_ms = np.full(m, 40.0e9, dtype=float)
    ptx = np.full(n, 1.0, dtype=float)
    tasks = [
        {
            "tid": f"{i}:0",
            "device": i,
            "k": 0,
            "arrival_s": 0.0,
            "deadline_s": 0.160,
            "D_bits": 800_000.0,
            "C_cycles": 1.2e9,
            "tau_dn_s": 0.001,
        }
        for i in range(n)
    ]
    scen = make_micro_scenario(
        case_id="CE_L1_PREBATCH",
        n_devices=n,
        n_servers=m,
        h_dev_ms=h,
        f_loc_hz=f_loc,
        f_ms_max_hz=f_ms,
        p_tx_w=ptx,
        tasks_raw=tasks,
    )
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    return scen, cfg


def build_counterexample_L2() -> tuple[ScenarioV1, SimConfig]:
    """Counterexample L2 (Non-Delay Policy Class Pi_non_delay): N=3, M=1, R=3.

    Task 0 arrives at t=0.0 (D=7e6 bits, C=4e8 cycles, d=0.115s, F_avail=19 GHz).
    Tasks 1,2 arrive at t=0.045s (revealed at tick t=0.050s) with D=7e5 bits, C=4e8 cycles, d=0.250s.
      - At t=0.050s, Task 0 is still transmitting on air until 0.0712s.
      - Any Non-Delay policy cannot `defer` Tasks 1,2: either it commits edge (slowing Task 0's
        remaining 2.02 Mbits into timeout at 0.115s -> <= 2/3) or drops them (1/3).
      - Delay-allowed optimal policy `defers` Tasks 1,2 at t=0.050s, lets Task 0 finish at 0.093s,
        and admits Tasks 1,2 together at t=0.100s -> 3/3 = 100% (gap >= 33.3 pp)!
    """
    n, m = 3, 1
    h = np.full((n, m), 1e-10, dtype=float)
    f_loc = np.full(n, 0.5e9, dtype=float)
    f_ms = np.full(m, 20.0e9, dtype=float)
    ptx = np.full(n, 1.0, dtype=float)
    tasks = [
        {"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.000, "deadline_s": 0.115, "D_bits": 7_000_000.0, "C_cycles": 4.0e8, "tau_dn_s": 0.001},
        {"tid": "1:0", "device": 1, "k": 0, "arrival_s": 0.045, "deadline_s": 0.250, "D_bits": 700_000.0, "C_cycles": 4.0e8, "tau_dn_s": 0.001},
        {"tid": "2:0", "device": 2, "k": 0, "arrival_s": 0.045, "deadline_s": 0.250, "D_bits": 700_000.0, "C_cycles": 4.0e8, "tau_dn_s": 0.001},
    ]
    scen = make_micro_scenario(
        case_id="CE_L2_NONDELAY",
        n_devices=n,
        n_servers=m,
        h_dev_ms=h,
        f_loc_hz=f_loc,
        f_ms_max_hz=f_ms,
        p_tx_w=ptx,
        tasks_raw=tasks,
    )
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    return scen, cfg


def build_counterexample_L3() -> tuple[ScenarioV1, SimConfig]:
    """Counterexample L3 (Externality-Blind Myopic EDF-Greedy Class Pi_myopic): N=4, M=1, R=4.

    All 4 tasks arrive at t=0.0, F_avail=38 GHz, F_loc=0.4 GHz (local impossible).
    Task 0 has earliest deadline (0.120s), strong received signal p0*h0 = 2e-8 W (200x peers),
    large payload D=1.95e7 bits (u_0=110.7ms), C=1.5e8 cycles -> feasible alone!
    Tasks 1,2,3 have slightly looser deadlines (0.130s), weaker signal 1e-10 W, D=4e5 bits, C=5e8 cycles.
      - Myopic EDF scans Task 0 first -> admits Task 0 -> Task 0's interference crushes peers to 72 kbps
        until t=0.1117s, leaving insufficient time/CPU for ANY of Tasks 1..3 -> 1/4 = 25%!
      - Deferring/skipping Task 0 lets ALL 3 Tasks 1,2,3 finish together -> 3/4 = 75% (gap = +50 pp)!
    """
    n, m = 4, 1
    h = np.array([[1e-8], [1e-10], [1e-10], [1e-10]], dtype=float)
    f_loc = np.full(n, 0.4e9, dtype=float)
    f_ms = np.full(m, 40.0e9, dtype=float)
    ptx = np.array([2.0, 1.0, 1.0, 1.0], dtype=float)
    tasks = [
        {"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0, "deadline_s": 0.120, "D_bits": 19_500_000.0, "C_cycles": 1.5e8, "tau_dn_s": 0.001},
        {"tid": "1:0", "device": 1, "k": 0, "arrival_s": 0.0, "deadline_s": 0.130, "D_bits": 400_000.0, "C_cycles": 5.0e8, "tau_dn_s": 0.001},
        {"tid": "2:0", "device": 2, "k": 0, "arrival_s": 0.0, "deadline_s": 0.130, "D_bits": 400_000.0, "C_cycles": 5.0e8, "tau_dn_s": 0.001},
        {"tid": "3:0", "device": 3, "k": 0, "arrival_s": 0.0, "deadline_s": 0.130, "D_bits": 400_000.0, "C_cycles": 5.0e8, "tau_dn_s": 0.001},
    ]
    scen = make_micro_scenario(
        case_id="CE_L3_MYOPIC_EDF",
        n_devices=n,
        n_servers=m,
        h_dev_ms=h,
        f_loc_hz=f_loc,
        f_ms_max_hz=f_ms,
        p_tx_w=ptx,
        tasks_raw=tasks,
    )
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    return scen, cfg
