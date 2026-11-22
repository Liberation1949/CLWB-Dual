"""Three Competing Mathematical Routes & 2nd-Order Coupled Counterexample (Stage 3 Step 3.1).

Strictly derived from Stage-2 `Model v1.0` Mathematical Leverage Inventory (`MODEL_v1.0_SPEC.md` §2.4):
  * Cut 1 (Low-Complexity Batch-Consistent Projection & Quadratic Externality Pricing):
    Cross-layer marginal sensitivity d f_r^min / d I_j proportional to 1 / (slack_r^cmp)^2
    plus hereditary fluid prefix contraction (Theorem 2.1).
  * Cut 2 (Causal Deferral Stopping & Actuation-Jitter Resilience):
    Exact Shannon active-link interference headroom Delta I_u, Lagrangian deferral bound,
    and self-measured runtime compensation t_exec = t_k + L_elapsed + delta_sig.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from model_spec import SimConfig, make_micro_scenario, shannon_rate  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Second-order anti-triviality counterexample (defeats +CommonGuard too!)
# --------------------------------------------------------------------------- #

def build_counterexample_2nd_order() -> tuple[Any, SimConfig]:
    """Second-order coupled counterexample (N=5, M=1, R=5):

    Combines cross-tick active-link overlap (tick t_1) with extreme intra-tick quadratic
    cross-layer externality (tick t_2) on a single 45 GHz server (F_avail = 42.75 GHz):
      - Task 0 arrives at t=0.0 (D=5.5e6, d=0.105s, C=3.0e8, h=1e-10). Runs until 0.0562s.
      - Tasks 1..4 arrive at t=0.045s (revealed at t_1=0.050s). Local CPUs are 0.4 GHz (impossible).
        * CommonGuard protects Task 0 at t_1 by deferring new edge requests to t_2=0.100s.
        * At t_2=0.100s, Task 1 has earlier deadline (0.216s < 0.228s) and 200x interferer
          power (p=2.0 W, h=1e-8). 1st-order myopic EDF+CG (`SFA+CG`) admits Task 1 first,
          crushing Tasks 2..4 to 72 kbps until 0.2032s and leaving <= 0.0198s slack (requires
          32.8 GHz > 28.85 GHz residual CPU) -> strictly 2/5 = 40%!
        * Batch-consistent cross-layer pricing skips Task 1 and jointly admits Tasks 2, 3, 4
          at t_2=0.100s (each needs ~11.29 GHz <= 36.35 GHz) -> exact oracle 4/5 = 80%!
    """
    n, m = 5, 1
    h = np.array([[1e-10], [1e-8], [1e-10], [1e-10], [1e-10]], dtype=float)
    f_loc = np.full(n, 0.4e9, dtype=float)
    f_ms = np.full(m, 45.0e9, dtype=float)
    ptx = np.array([1.0, 2.0, 1.0, 1.0, 1.0], dtype=float)
    tasks = [
        {"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.000, "deadline_s": 0.105, "D_bits": 5_500_000.0, "C_cycles": 3.0e8, "tau_dn_s": 0.001},
        {"tid": "1:0", "device": 1, "k": 0, "arrival_s": 0.045, "deadline_s": 0.216, "D_bits": 18_000_000.0, "C_cycles": 1.5e8, "tau_dn_s": 0.001},
        {"tid": "2:0", "device": 2, "k": 0, "arrival_s": 0.045, "deadline_s": 0.228, "D_bits": 400_000.0, "C_cycles": 6.5e8, "tau_dn_s": 0.001},
        {"tid": "3:0", "device": 3, "k": 0, "arrival_s": 0.045, "deadline_s": 0.228, "D_bits": 400_000.0, "C_cycles": 6.5e8, "tau_dn_s": 0.001},
        {"tid": "4:0", "device": 4, "k": 0, "arrival_s": 0.045, "deadline_s": 0.228, "D_bits": 400_000.0, "C_cycles": 6.5e8, "tau_dn_s": 0.001},
    ]
    scen = make_micro_scenario(
        case_id="CE_2ND_ORDER_COUPLED",
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


# --------------------------------------------------------------------------- #
# 2. Vectorized exact fluid batch rollout & active headroom helpers
# --------------------------------------------------------------------------- #

def fast_evaluate_fluid_batch(
    obs: dict,
    candidate_edge_grants: list[dict],
    act_lag_s: float = 0.0,
) -> dict[str, Any]:
    """O(K)-vectorized exact continuous fluid-SINR + half-duplex NIC evaluator."""
    now = float(obs["time"])
    t_act = now + float(obs["delta_sig_s"]) + float(act_lag_s)
    devs = obs["devices"]
    srvs = obs["servers"]
    links = obs["links"]
    h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
    p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
    n_sb = max(1, int(links["n_subbands"]))
    sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
    bw = float(links["bandwidth_hz"])
    noise = float(links["noise_w"])
    ortho = bool(links["orthogonal"])
    f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    f_min = float(obs["f_min_hz"])
    n = len(devs)
    m = len(srvs)

    active = obs["active_uplinks"]
    if not active and not candidate_edge_grants:
        return {"t_tx_end": {}, "f_min": {}, "active_ok": {}, "feasible": True, "reason": "empty"}

    tx_q: list[list[dict]] = [[] for _ in range(n)]
    running: list[dict] = []
    busy_dev = np.zeros(n, dtype=bool)
    flows: dict[str, dict] = {}

    for u in active:
        tid = str(u["id"])
        i = int(u["device"])
        j = int(u["server"])
        rec = {
            "id": tid, "is_active": True, "device": i, "server": j, "sb": int(sb_arr[i]),
            "ptx": float(u["p_tx_w"]), "sig": float(u["p_tx_w"]) * float(h_mat[i, j]),
            "rem_bits": float(max(u["rem_bits"], 0.0)), "deadline_s": float(u["deadline_s"]),
            "C_cycles": float(u["C_cycles"]), "f_locked": float(u["f_hz"]), "tau_dn_s": float(u["tau_dn_s"]),
        }
        flows[tid] = rec
        running.append(rec)
        busy_dev[i] = True

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
            "id": tid, "is_active": False, "device": i, "server": j, "sb": int(sb_arr[i]),
            "ptx": float(p_arr[i]), "sig": float(p_arr[i]) * float(h_mat[i, j]),
            "rem_bits": float(g["D_bits"]), "deadline_s": float(g["deadline_s"]),
            "C_cycles": float(g["C_cycles"]), "f_locked": None, "tau_dn_s": float(g["tau_dn_s"]),
        }
        flows[tid] = rec
        if t_act <= now + 1e-12:
            tx_q[i].append(rec)
        else:
            pending_release.append(rec)

    def _dispatch_queues() -> None:
        for di in range(n):
            if (not busy_dev[di]) and tx_q[di]:
                tx_q[di].sort(key=lambda r: (r["deadline_s"], r["device"], r["id"]))
                nxt = tx_q[di].pop(0)
                running.append(nxt)
                busy_dev[di] = True

    _dispatch_queues()
    t_cur = now
    released = t_act <= now + 1e-12
    t_tx_end: dict[str, float] = {}
    p_sb_srv = np.zeros((n_sb, m), dtype=float)
    max_iters = 64 + 4 * (len(active) + len(candidate_edge_grants))

    for _ in range(max_iters):
        if not running and released:
            break
        if not running:
            t_cur = t_act
            released = True
            for rec in pending_release:
                tx_q[rec["device"]].append(rec)
            pending_release.clear()
            _dispatch_queues()
            continue

        k_run = len(running)
        if ortho:
            rates = [
                max(bw * math.log2(1.0 + r["sig"] / max(noise, 1e-30)), 1e-3)
                for r in running
            ]
        else:
            p_sb_srv.fill(0.0)
            for r in running:
                p_sb_srv[r["sb"], :] += r["ptx"] * h_mat[r["device"], :]
            rates = [
                max(
                    bw * math.log2(
                        1.0 + r["sig"] / max(noise + max(0.0, float(p_sb_srv[r["sb"], r["server"]] - r["sig"])), 1e-30)
                    ),
                    1e-3,
                )
                for r in running
            ]

        min_dt_end = min(r["rem_bits"] / rt for r, rt in zip(running, rates))
        dt_rel = max(0.0, t_act - t_cur) if (not released and pending_release) else math.inf
        dt = min(min_dt_end, dt_rel)

        for idx_r in range(k_run):
            running[idx_r]["rem_bits"] = max(0.0, running[idx_r]["rem_bits"] - rates[idx_r] * dt)
        t_cur += dt

        still: list[dict] = []
        finished_any = False
        for r in running:
            if r["rem_bits"] <= 1e-7:
                r["rem_bits"] = 0.0
                t_tx_end[r["id"]] = float(t_cur)
                busy_dev[r["device"]] = False
                finished_any = True
            else:
                still.append(r)
        running = still

        if (not released) and abs(t_cur - t_act) <= 1e-12:
            released = True
            for rec in pending_release:
                tx_q[rec["device"]].append(rec)
            pending_release.clear()
            _dispatch_queues()
        elif finished_any:
            _dispatch_queues()

    f_min_map: dict[str, float] = {}
    active_ok: dict[str, bool] = {}
    srv_added = np.zeros(m, dtype=float)
    feasible = True
    fail_reason = "ok"

    for tid, r in flows.items():
        if tid not in t_tx_end:
            feasible = False
            fail_reason = f"incomplete:{tid}"
            break
        t_up_e = t_tx_end[tid]
        if r["is_active"]:
            f_lock = float(r["f_locked"])
            ok = (t_up_e + r["C_cycles"] / max(f_lock, 1.0) + r["tau_dn_s"]) <= (r["deadline_s"] + 1e-9)
            active_ok[tid] = ok
            f_min_map[tid] = f_lock
            if not ok:
                feasible = False
                fail_reason = f"active_miss:{tid}"
        else:
            slack_cmp = r["deadline_s"] - t_up_e - r["tau_dn_s"]
            if slack_cmp <= 1e-9:
                f_min_map[tid] = math.inf
                feasible = False
                fail_reason = f"dl_miss:{tid}"
            else:
                f_req = max(f_min, r["C_cycles"] / slack_cmp)
                f_min_map[tid] = f_req
                j = r["server"]
                srv_added[j] += f_req
                if f_req > f_avail[j] + 1.0:
                    feasible = False
                    fail_reason = f"cap_single:{tid}"

    if feasible and np.any(srv_res + srv_added > f_avail + 1.0):
        feasible = False
        fail_reason = "cap_agg"

    return {
        "t_tx_end": t_tx_end,
        "f_min": f_min_map,
        "active_ok": active_ok,
        "srv_used_hz": (srv_res + srv_added).tolist(),
        "feasible": bool(feasible),
        "reason": fail_reason,
    }


def _shadow_externality_tables(
    obs: dict,
    extra_grants: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Compute interference I[sb, j], server dual weights lam[j], cross-layer sensitivity
    pressure Pi[sb, j'], and per-active-link interference headroom Delta I_u."""
    now = float(obs["time"])
    t_exec = now + float(obs["delta_sig_s"])
    devs = obs["devices"]
    srvs = obs["servers"]
    links = obs["links"]
    h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
    p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
    n_sb = max(1, int(links["n_subbands"]))
    sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
    bw = float(links["bandwidth_hz"])
    noise = float(links["noise_w"])
    ortho = bool(links["orthogonal"])
    m = len(srvs)
    ln2 = math.log(2.0)

    f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    f_min = float(obs["f_min_hz"])
    f_rem = np.maximum(f_avail - srv_res, f_min)
    lam = 1.0 / f_rem

    interf = np.zeros((n_sb, m), dtype=float)
    seen_devs: set[int] = set()
    for u in obs["active_uplinks"]:
        di = int(u["device"])
        if di not in seen_devs:
            seen_devs.add(di)
            interf[sb_arr[di], :] += p_arr[di] * h_mat[di, :]

    if extra_grants:
        for g in extra_grants:
            di = int(g["device"])
            if di not in seen_devs:
                seen_devs.add(di)
                interf[sb_arr[di], :] += p_arr[di] * h_mat[di, :]

    # Exact closed-form Shannon active-link interference headroom Delta I_u
    act_headrooms: list[dict] = []
    for u in obs["active_uplinks"]:
        di = int(u["device"])
        ji = int(u["server"])
        sb = sb_arr[di]
        sig = float(u["p_tx_w"]) * float(h_mat[di, ji])
        i_cur = 0.0 if ortho else max(0.0, float(interf[sb, ji] - sig))
        t_req = float(u["deadline_s"]) - now - float(u["C_cycles"]) / max(float(u["f_hz"]), 1.0) - float(u["tau_dn_s"])
        if t_req <= 1e-12:
            i_max = -1.0
        else:
            r_min = float(u["rem_bits"]) / t_req
            exp_term = min(r_min / bw, 50.0)
            denom_sinr = max(2.0 ** exp_term - 1.0, 1e-12)
            i_max = sig / denom_sinr - noise
        act_headrooms.append({
            "sb": int(sb), "srv": int(ji), "dev": int(di),
            "sig": sig, "i_cur": i_cur, "headroom_w": i_max - i_cur,
        })

    if ortho:
        return interf, lam, np.zeros((n_sb, m), dtype=float), act_headrooms

    pi_sens = np.zeros((n_sb, m), dtype=float)
    for u in obs["active_uplinks"]:
        di = int(u["device"])
        ji = int(u["server"])
        sb = sb_arr[di]
        sig = float(u["p_tx_w"]) * float(h_mat[di, ji])
        i_other = max(0.0, float(interf[sb, ji] - sig))
        denom_sum = max(noise + i_other, 1e-30)
        sinr = sig / denom_sum
        log1p = math.log2(1.0 + sinr)
        rate = max(bw * log1p, 1e-3)
        rem_bits = float(u["rem_bits"])
        t_tx = rem_bits / rate
        slack_act = max(float(u["deadline_s"]) - now - float(u["C_cycles"]) / max(float(u["f_hz"]), 1.0) - float(u["tau_dn_s"]) - t_tx, float(obs["delta_sig_s"]))
        du_di = (rem_bits / (bw * ln2 * max(log1p, 1e-9) ** 2)) * (sig / (denom_sum * (denom_sum + sig)))
        pi_sens[sb, ji] += lam[ji] * float(u["C_cycles"]) * du_di / (slack_act ** 2)

    if extra_grants:
        for g in extra_grants:
            di = int(g["device"])
            ji = int(g["server"])
            sb = sb_arr[di]
            sig = p_arr[di] * float(h_mat[di, ji])
            i_other = max(0.0, float(interf[sb, ji] - sig))
            denom_sum = max(noise + i_other, 1e-30)
            sinr = sig / denom_sum
            log1p = math.log2(1.0 + sinr)
            rate = max(bw * log1p, 1e-3)
            d_bits = float(g["D_bits"])
            t_up = d_bits / rate
            slack_cmp = max(float(g["deadline_s"]) - t_exec - t_up - float(g["tau_dn_s"]), float(obs["delta_sig_s"]))
            du_di = (d_bits / (bw * ln2 * max(log1p, 1e-9) ** 2)) * (sig / (denom_sum * (denom_sum + sig)))
            pi_sens[sb, ji] += lam[ji] * float(g["C_cycles"]) * du_di / (slack_cmp ** 2)

    return interf, lam, pi_sens, act_headrooms


def _hereditary_prefix_contract(
    obs: dict,
    ordered_grants: list[dict],
    act_lag_s: float = 0.0,
) -> tuple[list[dict], dict[str, Any]]:
    """Binary search on prefix length via Theorem 2.1 (Hereditary Anti-Monotonicity) in O(log P)."""
    if not ordered_grants:
        return [], {"feasible": True, "f_min": {}, "t_tx_end": {}}

    full_res = fast_evaluate_fluid_batch(obs, ordered_grants, act_lag_s=act_lag_s)
    if full_res["feasible"]:
        return ordered_grants, full_res

    lo, hi = 0, len(ordered_grants)
    best_res: dict[str, Any] = {"feasible": True, "f_min": {}, "t_tx_end": {}}
    while lo < hi:
        mid = (lo + hi + 1) // 2
        res_mid = fast_evaluate_fluid_batch(obs, ordered_grants[:mid], act_lag_s=act_lag_s)
        if res_mid["feasible"]:
            lo = mid
            best_res = res_mid
        else:
            hi = mid - 1
    return ordered_grants[:lo], best_res


def _finalize_edge_and_local(
    obs: dict,
    kept_grants: list[dict],
    fluid_res: dict[str, Any],
    fallback_local_pool: list[dict],
    t_enter: float,
) -> list[dict]:
    """Refresh exact f_min against actual self-latency, distribute residual sensitivity headroom,
    and allocate residual local fallback cleanly."""
    now = float(obs["time"])
    delta_sig = float(obs["delta_sig_s"])
    rho_sys = float(obs["rho_sys"])
    f_min = float(obs["f_min_hz"])
    devs = obs["devices"]
    srvs = obs["servers"]
    m = len(srvs)
    ortho = bool(obs["links"]["orthogonal"])

    f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
    dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
    dev_rem = np.maximum(dev_cap - dev_res, 0.0)

    actions: list[dict] = []
    kept_ids = {g["id"] for g in kept_grants}

    if kept_grants:
        l_now = time.perf_counter() - t_enter
        res_fin = fast_evaluate_fluid_batch(obs, kept_grants, act_lag_s=l_now)
        if not res_fin["feasible"]:
            kept_grants, fluid_res = _hereditary_prefix_contract(obs, kept_grants, act_lag_s=l_now)
            kept_ids = {g["id"] for g in kept_grants}
        else:
            fluid_res = res_fin

        srv_used_batch = srv_res.copy()
        for g in kept_grants:
            srv_used_batch[int(g["server"])] += float(fluid_res["f_min"][g["id"]])

        idle_budget = np.maximum(0.0, f_avail - srv_used_batch)
        # Distribute idle headroom proportionally weighted by dimensionless upload sensitivity
        # w_r = t_up_r / (t_up_r + C_r / f_r) up to rho_sys fraction of residual idle CPU
        srv_weights = np.zeros(m, dtype=float)
        task_w: dict[str, float] = {}
        for g in kept_grants:
            tid = g["id"]
            j = int(g["server"])
            f_base = float(fluid_res["f_min"][tid])
            t_up_est = max(0.0, float(fluid_res["t_tx_end"][tid]) - now - delta_sig)
            t_cmp_est = float(g["C_cycles"]) / max(f_base, 1.0)
            sens_ratio = t_up_est / max(t_up_est + t_cmp_est, 1e-9)
            w_r = f_base * ((1.0 - rho_sys) + sens_ratio)
            task_w[tid] = w_r
            srv_weights[j] += w_r

        for g in kept_grants:
            tid = g["id"]
            j = int(g["server"])
            f_base = float(fluid_res["f_min"][tid])
            if not ortho and srv_weights[j] > 0.0 and idle_budget[j] > 0.0:
                # Allocate rho_sys share of currently idle server CPU weighted by upload exposure
                alloc = (rho_sys * float(idle_budget[j])) * (task_w[tid] / srv_weights[j])
                bump = min(alloc, float(f_avail[j] - srv_used_batch[j]))
                srv_used_batch[j] += bump
            else:
                bump = 0.0
            actions.append({"task": tid, "mode": "edge", "server": j, "f_hz": min(float(f_avail[j]), f_base + bump)})

    used_loc = np.zeros(len(devs), dtype=float)
    t_exec_loc = now + delta_sig + (time.perf_counter() - t_enter)
    for t in sorted(fallback_local_pool, key=lambda x_t: (float(x_t["deadline_s"]), str(x_t["id"]))):
        tid = t["id"]
        if tid in kept_ids:
            continue
        i = int(t["device"])
        rem_l = float(t["deadline_s"]) - t_exec_loc
        if rem_l <= 1e-9:
            continue
        f_req_l = max(f_min, float(t["C_cycles"]) / rem_l)
        if used_loc[i] + f_req_l <= dev_rem[i] + 1.0:
            headroom_l = max(0.0, dev_rem[i] - (used_loc[i] + f_req_l))
            bump_l = min(headroom_l, (1.0 - rho_sys) * f_req_l)
            used_loc[i] += f_req_l + bump_l
            actions.append({"task": tid, "mode": "local", "server": None, "f_hz": float(f_req_l + bump_l)})

    return actions


# --------------------------------------------------------------------------- #
# 3. Route A: Cross-Layer Shadow-Price Dual + Analytical Incremental Sizing
# --------------------------------------------------------------------------- #

def make_route_a_clsp() -> Callable[[dict], dict]:
    """Route A (CLSP-WF): Unified Cross-Layer Shadow-Price Water-Filling + Exact Fluid
    Hereditary Prefix Contraction + Active-Link Shannon Headroom Deferral."""
    def policy(obs: dict) -> dict:
        t_enter = time.perf_counter()
        pending = obs["pending"]
        if not pending:
            return {"actions": []}

        now = float(obs["time"])
        delta_sig = float(obs["delta_sig_s"])
        tick_s = float(obs["tick_s"])
        t_exec = now + delta_sig
        devs = obs["devices"]
        srvs = obs["servers"]
        links = obs["links"]
        h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
        p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
        n_sb = max(1, int(links["n_subbands"]))
        sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        ortho = bool(links["orthogonal"])
        f_min = float(obs["f_min_hz"])
        m = len(srvs)
        ln2 = math.log(2.0)

        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_rem = np.maximum(f_avail - srv_res, 0.0)

        interf_act, lam, pi_act, act_headrooms = _shadow_externality_tables(obs, extra_grants=None)
        sorted_p = sorted(pending, key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])))
        p_cnt = len(sorted_p)

        t_dev = np.asarray([int(t["device"]) for t in sorted_p], dtype=int)
        t_sb = sb_arr[t_dev]
        t_ptx = p_arr[t_dev]
        t_dl = np.asarray([float(t["deadline_s"]) for t in sorted_p], dtype=float)
        t_d = np.asarray([float(t["D_bits"]) for t in sorted_p], dtype=float)
        t_c = np.asarray([float(t["C_cycles"]) for t in sorted_p], dtype=float)
        t_tdn = np.asarray([float(t["tau_dn_s"]) for t in sorted_p], dtype=float)

        # NIC earliest start per pending task
        nic_busy_until = np.full(len(devs), t_exec, dtype=float)
        for u in obs["active_uplinks"]:
            di = int(u["device"])
            ji = int(u["server"])
            r_u = max(bw * math.log2(1.0 + p_arr[di] * float(h_mat[di, ji]) / max(noise, 1e-30)), 1e-3)
            nic_busy_until[di] = max(nic_busy_until[di], now + float(u["rem_bits"]) / r_u)
        t_start_dev = nic_busy_until[t_dev]  # (P,)

        # Vectorized O(PM) Signal & Path-loss Matrices (P x M)
        h_pm = h_mat[t_dev, :]                  # (P, M)
        sig_pm = t_ptx[:, None] * h_pm          # (P, M)

        # Fast 3-pass vectorized Cross-Layer Water-Filling Equilibrium (no nested Python loops!)
        x = np.full((p_cnt, m), 1.0 / max(1, m + 1), dtype=float)

        # Aggregate per-device uplink contribution (P -> N -> sb x M)
        for _wf in range(3):
            if ortho:
                i_batch_pm = np.zeros((p_cnt, m), dtype=float)
                pi_batch_pm = np.zeros((p_cnt, m), dtype=float)
            else:
                tx_prob = np.clip(np.sum(x, axis=1), 0.0, 1.0)
                # Deduplicated device tx probability weighted sum per subband
                p_sb_srv = np.zeros((n_sb, m), dtype=float)
                for q_i in range(p_cnt):
                    p_sb_srv[t_sb[q_i], :] += tx_prob[q_i] * sig_pm[q_i, :]
                i_tot_pm = interf_act[t_sb, :] + p_sb_srv[t_sb, :]
                i_batch_pm = np.maximum(0.0, i_tot_pm - tx_prob[:, None] * sig_pm)

            denom_pm = np.maximum(noise + i_batch_pm, 1e-30)
            sinr_pm = sig_pm / denom_pm
            log1p_pm = np.log2(1.0 + sinr_pm)
            rates_pm = np.maximum(bw * log1p_pm, 1e-3)
            u_pm = t_d[:, None] / rates_pm
            rem_cmp_pm = t_dl[:, None] - t_start_dev[:, None] - u_pm - t_tdn[:, None]
            feas_pm = rem_cmp_pm > 1e-9
            safe_rem = np.where(feas_pm, rem_cmp_pm, 1e-6)
            f_pm = np.where(feas_pm, np.maximum(f_min, t_c[:, None] / safe_rem), 1e18)
            feas_pm = feas_pm & (f_pm <= f_avail[None, :] + 1.0)

            srv_load = np.sum(np.where(feas_pm, x * f_pm, 0.0), axis=0)
            lam_wf = 1.0 / np.maximum(f_avail - srv_res - srv_load, f_min)

            if ortho:
                ext_tot_p = np.zeros(p_cnt, dtype=float)
            else:
                # Batch quadratic sensitivity contribution per (sb, j) weighted by lam_wf[j]
                du_di_pm = (t_d[:, None] / (bw * ln2 * np.maximum(log1p_pm, 1e-9) ** 2)) * (sig_pm / (denom_pm * (denom_pm + sig_pm)))
                sens_pm = np.where(feas_pm, x * (lam_wf[None, :] * t_c[:, None] * du_di_pm / np.maximum(safe_rem, delta_sig) ** 2), 0.0)
                pi_batch_sb = np.zeros((n_sb, m), dtype=float)
                for q_i in range(p_cnt):
                    pi_batch_sb[t_sb[q_i], :] += sens_pm[q_i, :]
                pi_comb_sb = pi_act + pi_batch_sb
                ext_tot_p = np.einsum("pm,pm->p", sig_pm, pi_comb_sb[t_sb, :]) - np.einsum("pm,pm->p", sig_pm, sens_pm)
                ext_tot_p = np.maximum(0.0, ext_tot_p)

            # Gradient step balancing throughput reward (1) vs shadow cost (lam_wf * f + ext)
            step = 1.0 / max(2.0, float(m))
            cost_pm = lam_wf[None, :] * f_pm + ext_tot_p[:, None]
            grad_pm = np.where(feas_pm, 1.0 - cost_pm, -1.0)
            x = np.maximum(0.0, x + step * grad_pm)
            row_s = np.sum(x, axis=1, keepdims=True)
            over = row_s > 1.0
            x = np.where(over, x / np.maximum(row_s, 1e-12), x)

        # Rank equilibrium options using cross-layer shadow score & equilibrium fractional confidence
        # Key property: under ortho (kappa=0), cost_pm reduces to lam * f_req (smallest share first)!
        # Under asymmetric high-interference (L3, CE_2nd), extreme externality puts bullies at back!
        cand_items: list[tuple[float, dict, int]] = []
        act_rem_h = [dict(h_item) for h_item in act_headrooms]
        for q_i, t in enumerate(sorted_p):
            i = int(t_dev[q_i])
            sb = int(t_sb[q_i])

            # Active-link Shannon interference guard + sub-tick NIC deferral check:
            # If NIC won't free BEFORE next_tick and can cleanly wait 1 tick, defer!
            if bool(devs[i]["radio_busy"]) and t_start_dev[q_i] >= now + tick_s - 1e-9:
                # Check if at least one server survives waiting 1 tick
                r_iso_best = max(bw * math.log2(1.0 + float(np.max(sig_pm[q_i, :])) / max(noise, 1e-30)), 1e-3)
                if t_dl[q_i] - (now + tick_s + delta_sig) - t_d[q_i] / r_iso_best - t_tdn[q_i] > t_c[q_i] / max(float(np.max(f_avail)), 1.0):
                    continue

            # If admitting device i onto subband sb strictly violates active link Shannon headroom, defer!
            if not ortho and act_rem_h:
                breaks_act = False
                for ah in act_rem_h:
                    if ah["sb"] == sb and ah["dev"] != i:
                        if ah["headroom_w"] - float(sig_pm[q_i, ah["srv"]]) < -1e-18:
                            breaks_act = True
                            break
                if breaks_act:
                    continue

            # Pick best feasible server j for task q_i using waterfilling weights + shadow cost
            best_j = -1
            best_sc = math.inf
            for j in range(m):
                if not feas_pm[q_i, j]:
                    continue
                # Score = shadow cost penalized inversely by equilibrium fraction x[q_i, j] (smooth load spread!)
                wf_bonus = float(x[q_i, j])
                sc = float(cost_pm[q_i, j]) - 0.5 * wf_bonus
                if sc < best_sc:
                    best_sc = sc
                    best_j = j
            if best_j >= 0:
                cand_items.append((best_sc, t, best_j))

        cand_items.sort(key=lambda item: (item[0], float(item[1]["deadline_s"]), item[1]["id"]))

        # Assemble candidate grants up to approximate server headroom, preserving active-link headroom
        # and checking exact continuous fluid feasibility in O(log P)!
        ordered_grants: list[dict] = []
        seen_devs_ord: set[int] = {int(u["device"]) for u in obs["active_uplinks"]}
        est_srv_load = srv_res.copy()

        # First pass: take primary low-externality tasks that respect active headroom and server caps
        tail_items: list[tuple[float, dict, int]] = []
        for sc, t, srv_j in cand_items:
            i = int(t["device"])
            sb = sb_arr[i]
            new_dev = i not in seen_devs_ord
            if new_dev and not ortho and act_rem_h:
                viol_act = False
                for ah in act_rem_h:
                    if ah["sb"] == sb and ah["dev"] != i:
                        if ah["headroom_w"] - float(p_arr[i] * h_mat[i, ah["srv"]]) < 0.0:
                            viol_act = True
                            break
                if viol_act:
                    continue

            # Estimate CPU demand on srv_j
            r_iso = max(bw * math.log2(1.0 + p_arr[i] * float(h_mat[i, srv_j]) / max(noise, 1e-30)), 1e-3)
            f_est = max(f_min, float(t["C_cycles"]) / max(float(t["deadline_s"]) - nic_busy_until[i] - float(t["D_bits"]) / r_iso - float(t["tau_dn_s"]), 1e-6))
            if est_srv_load[srv_j] + f_est <= f_avail[srv_j] + 1.0:
                if new_dev and not ortho:
                    seen_devs_ord.add(i)
                    for ah in act_rem_h:
                        if ah["sb"] == sb and ah["dev"] != i:
                            ah["headroom_w"] -= float(p_arr[i] * h_mat[i, ah["srv"]])
                est_srv_load[srv_j] += f_est
                ordered_grants.append({
                    "id": t["id"], "device": i, "server": int(srv_j),
                    "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                    "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
                })
            else:
                tail_items.append((sc, t, srv_j))

        l_el = time.perf_counter() - t_enter
        kept_grants, fluid_res = _hereditary_prefix_contract(obs, ordered_grants, act_lag_s=l_el)
        kept_ids = {g["id"] for g in kept_grants}

        # Lightweight residual salvage on up to m skipped/tail items (keeps total fluid calls <= log P + m ~ 12!)
        salvage_pool = [g for g in ordered_grants if g["id"] not in kept_ids]
        for _sc, t, srv_j in tail_items[:m]:
            salvage_pool.append({
                "id": t["id"], "device": int(t["device"]), "server": int(srv_j),
                "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
            })
        for g in salvage_pool[: max(4, m)]:
            trial = kept_grants + [g]
            res_t = fast_evaluate_fluid_batch(obs, trial, act_lag_s=time.perf_counter() - t_enter)
            if res_t["feasible"]:
                kept_grants = trial
                fluid_res = res_t
                kept_ids.add(g["id"])

        fallback_local = [t for t in sorted_p if t["id"] not in kept_ids]
        return {"actions": _finalize_edge_and_local(obs, kept_grants, fluid_res, fallback_local, t_enter)}

    policy.__name__ = "RouteA_CLSP"
    return policy


# --------------------------------------------------------------------------- #
# 4. Route B: Smooth Log-Barrier Fractional Convex Relaxation + Hereditary Round
# --------------------------------------------------------------------------- #

def make_route_b_waterfill(n_iters: int = 12) -> Callable[[dict], dict]:
    """Route B: Projected ascent on smooth log-barrier fractional CPU capacity + fluid rounding."""
    def policy(obs: dict) -> dict:
        t_enter = time.perf_counter()
        pending = obs["pending"]
        if not pending:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        links = obs["links"]
        h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
        p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
        n_sb = max(1, int(links["n_subbands"]))
        sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        ortho = bool(links["orthogonal"])
        f_min = float(obs["f_min_hz"])
        m = len(srvs)

        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_rem = np.maximum(f_avail - srv_res, f_min)

        p_cnt = len(pending)
        x = np.full((p_cnt, m), 1.0 / max(1, m + 1), dtype=float)
        interf_act, _lam, pi_act, _ah = _shadow_externality_tables(obs, extra_grants=None)

        d_bits = np.asarray([float(t["D_bits"]) for t in pending], dtype=float)
        c_cyc = np.asarray([float(t["C_cycles"]) for t in pending], dtype=float)
        dl = np.asarray([float(t["deadline_s"]) for t in pending], dtype=float)
        tau_dn = np.asarray([float(t["tau_dn_s"]) for t in pending], dtype=float)
        t_dev = np.asarray([int(t["device"]) for t in pending], dtype=int)
        t_sb = sb_arr[t_dev]

        step = 1.0 / max(2.0, float(p_cnt))
        for _it in range(int(n_iters)):
            interf_frac = interf_act.copy()
            if not ortho:
                tx_prob = np.clip(np.sum(x, axis=1), 0.0, 1.0)
                for q in range(p_cnt):
                    di = t_dev[q]
                    interf_frac[t_sb[q], :] += tx_prob[q] * p_arr[di] * h_mat[di, :]

            f_req_mat = np.full((p_cnt, m), 1e18, dtype=float)
            load_j = np.zeros(m, dtype=float)
            for q in range(p_cnt):
                di = t_dev[q]
                for j in range(m):
                    sig = p_arr[di] * float(h_mat[di, j])
                    i_o = max(0.0, float(interf_frac[t_sb[q], j] - np.sum(x[q, :]) * sig))
                    rate = max(bw * math.log2(1.0 + sig / max(noise + i_o, 1e-30)), 1e-3)
                    rem = dl[q] - t_exec - d_bits[q] / rate - tau_dn[q]
                    if rem > 1e-9:
                        fv = max(f_min, c_cyc[q] / rem)
                        if fv <= srv_rem[j] + 1.0:
                            f_req_mat[q, j] = fv
                            load_j[j] += x[q, j] * fv

            dual_j = 1.0 / np.maximum(srv_rem - 0.5 * load_j, f_min)
            for q in range(p_cnt):
                di = t_dev[q]
                act_pen = 0.0 if ortho else float(np.dot(p_arr[di] * h_mat[di, :], pi_act[t_sb[q], :]))
                grad = 1.0 - dual_j * f_req_mat[q, :] - act_pen
                cand_x = np.maximum(0.0, x[q, :] + step * grad)
                s_sum = float(np.sum(cand_x))
                if s_sum > 1.0:
                    cand_x /= s_sum
                x[q, :] = cand_x

        ranked = []
        for q, t in enumerate(pending):
            j_best = int(np.argmax(x[q, :]))
            score = float(x[q, j_best])
            if score > 0.0:
                ranked.append((score, t, j_best))
        ranked.sort(key=lambda item: (-item[0], float(item[1]["deadline_s"]), item[1]["id"]))

        grants = [
            {
                "id": t["id"], "device": int(t["device"]), "server": int(j),
                "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
            }
            for _sc, t, j in ranked
        ]
        l_el = time.perf_counter() - t_enter
        kept, fluid_res = _hereditary_prefix_contract(obs, grants, act_lag_s=l_el)
        kept_ids = {g["id"] for g in kept}
        return {"actions": _finalize_edge_and_local(obs, kept, fluid_res, [t for t in pending if t["id"] not in kept_ids], t_enter)}

    policy.__name__ = "RouteB_Waterfill"
    return policy


# --------------------------------------------------------------------------- #
# 5. Route C: Whittle Urgency-to-Externality Index Ratio Policy
# --------------------------------------------------------------------------- #

def make_route_c_whittle() -> Callable[[dict], dict]:
    """Route C: Whittle index ranking by delta_c_defer(q) / c_eff(q, j) + hereditary check."""
    def policy(obs: dict) -> dict:
        t_enter = time.perf_counter()
        pending = obs["pending"]
        if not pending:
            return {"actions": []}

        now = float(obs["time"])
        tick_s = float(obs["tick_s"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        links = obs["links"]
        h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
        p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
        n_sb = max(1, int(links["n_subbands"]))
        sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        ortho = bool(links["orthogonal"])
        f_min = float(obs["f_min_hz"])
        m = len(srvs)

        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        tent = []
        seen_d: set[int] = set()
        for t in pending:
            i = int(t["device"])
            if i not in seen_d:
                seen_d.add(i)
                j_best = int(np.argmax(h_mat[i, :]))
                tent.append({
                    "id": t["id"], "device": i, "server": j_best,
                    "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                    "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
                })
        _i_tot, lam, pi_tot, _ah = _shadow_externality_tables(obs, extra_grants=tent)

        whittle_list: list[tuple[float, float, dict, int]] = []
        for t in pending:
            i = int(t["device"])
            sb = sb_arr[i]
            dl = float(t["deadline_s"])
            c_cyc = float(t["C_cycles"])
            d_bits = float(t["D_bits"])
            tau_dn = float(t["tau_dn_s"])

            best_w = -math.inf
            best_c = math.inf
            best_srv = -1
            ext = 0.0 if ortho else float(np.dot(p_arr[i] * h_mat[i, :], pi_tot[sb, :]))

            for j in range(m):
                sig = p_arr[i] * float(h_mat[i, j])
                r_iso = max(bw * math.log2(1.0 + sig / max(noise, 1e-30)), 1e-3)
                slack_now = dl - t_exec - d_bits / r_iso - tau_dn
                if slack_now <= 1e-9:
                    continue
                f_iso = max(f_min, c_cyc / slack_now)
                if f_iso > f_avail[j] + 1.0:
                    continue
                c_eff = lam[j] * f_iso + ext
                slack_nxt = slack_now - tick_s
                if slack_nxt <= c_cyc / max(f_avail[j], 1.0):
                    w_idx = 1e6 / max(c_eff, 1e-9)
                else:
                    dc_def = lam[j] * c_cyc * (1.0 / slack_nxt - 1.0 / slack_now)
                    w_idx = dc_def / max(c_eff, 1e-9)
                if (w_idx > best_w) or (abs(w_idx - best_w) <= 1e-12 and c_eff < best_c):
                    best_w = w_idx
                    best_c = c_eff
                    best_srv = j

            if best_srv >= 0:
                whittle_list.append((best_w, -best_c, t, best_srv))

        whittle_list.sort(key=lambda item: (-item[0], -item[1], float(item[2]["deadline_s"]), item[2]["id"]))

        grants = [
            {
                "id": t["id"], "device": int(t["device"]), "server": int(j),
                "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
            }
            for _w, _mc, t, j in whittle_list
        ]
        l_el = time.perf_counter() - t_enter
        kept, fluid_res = _hereditary_prefix_contract(obs, grants, act_lag_s=l_el)
        kept_ids = {g["id"] for g in kept}
        return {"actions": _finalize_edge_and_local(obs, kept, fluid_res, [t for t in pending if t["id"] not in kept_ids], t_enter)}

    policy.__name__ = "RouteC_Whittle"
    return policy
