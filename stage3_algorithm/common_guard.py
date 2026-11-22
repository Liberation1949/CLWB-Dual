"""CommonGuard: Strictly Tier-3 Engineering Hygiene + 1st-Order Obvious Fixes (Stage 3).

Applied uniformly to ALL comparison baselines so no candidate gains unfair advantage from
trivial sanity protections:
  1. Strict action schema & hardware-cap sanitization (no duplicate IDs, f_min <= f_hz <= cap).
  2. Nominal 2% leftover-headroom scheduling slack against sub-millisecond OS actuation jitter.
  3. 1st-order half-duplex NIC check: defers tasks queued behind an already-busy NIC whose
     minimum remaining active + queued transmission alone breaches the deadline.
  4. 1st-order static active-link protection: checks whether newly proposed edge grants statically
     push any in-flight uplink u in U(t_k) past its deadline, deferring the tail edge additions
     until all active on-air links remain statically protected.

Strictly excludes any Stage-3 OurCore logic (no cross-layer quadratic externality pricing,
no Lagrangian inter-epoch deferral stopping, no fast fluid hereditary contraction).
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import numpy as np

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from model_spec import shannon_rate  # noqa: E402


def apply_common_guard(obs: dict, raw_out: dict) -> dict:
    """Sanitize baseline output with Tier-3 checks and 1st-order active-link / NIC guard."""
    if not isinstance(raw_out, dict) or not isinstance(raw_out.get("actions"), list):
        return {"actions": []}

    now = float(obs["time"])
    t_exec = now + float(obs["delta_sig_s"])
    f_min = float(obs["f_min_hz"])
    devs = obs["devices"]
    srvs = obs["servers"]
    links = obs["links"]
    h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
    bw = float(links["bandwidth_hz"])
    noise = float(links["noise_w"])
    ortho = bool(links["orthogonal"])
    n_sb = max(1, int(links["n_subbands"]))

    pend_map = {t["id"]: t for t in obs["pending"]}
    srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
    dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)

    # Active link static remaining lower-bound per busy radio
    act_finish_lb_by_dev: dict[int, float] = {}
    for u in obs["active_uplinks"]:
        di = int(u["device"])
        ji = int(u["server"])
        r_iso = shannon_rate(bw, float(u["p_tx_w"]) * float(h_mat[di, ji]), noise, 0.0)
        act_finish_lb_by_dev[di] = now + float(u["rem_bits"]) / r_iso

    seen: set[str] = set()
    clean_non_edge: list[dict] = []
    edge_actions: list[dict] = []

    for act in raw_out["actions"]:
        if not isinstance(act, dict):
            continue
        tid = act.get("task")
        if tid not in pend_map or tid in seen:
            continue
        seen.add(tid)
        t = pend_map[tid]
        i = int(t["device"])
        md = act.get("mode", "defer")

        if md == "defer":
            continue
        if md == "drop":
            clean_non_edge.append({"task": tid, "mode": "drop", "server": None, "f_hz": None})
            continue
        if md not in ("local", "edge"):
            continue
        try:
            f_hz = float(act.get("f_hz"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(f_hz) or f_hz < f_min - 1e-3:
            continue

        if md == "local":
            cap = float(dev_cap[i])
            if f_hz > cap + 1.0:
                continue
            rem_loc = float(t["deadline_s"]) - t_exec
            if rem_loc <= 1e-9:
                continue
            f_safe = min(cap, max(f_min, f_hz))
            if dev_res[i] + f_safe > cap + 1.0:
                continue
            headroom = max(0.0, cap - (dev_res[i] + f_safe))
            f_safe += min(headroom, 0.02 * f_safe)
            dev_res[i] += f_safe
            clean_non_edge.append({"task": tid, "mode": "local", "server": None, "f_hz": float(f_safe)})
        else:
            sv = act.get("server")
            if not isinstance(sv, (int, np.integer)) or isinstance(sv, bool):
                continue
            j = int(sv)
            if not (0 <= j < len(srvs)):
                continue
            cap = float(srv_cap[j])
            if f_hz > cap + 1.0:
                continue

            # 1st-order NIC queueing sanity: if NIC is busy and min serial upload already misses deadline, defer
            if bool(devs[i]["radio_busy"]):
                t_start_lb = max(t_exec, act_finish_lb_by_dev.get(i, t_exec))
                r_iso = shannon_rate(bw, float(devs[i]["p_tx_w"]) * float(h_mat[i, j]), noise, 0.0)
                t_end_lb = t_start_lb + float(t["D_bits"]) / r_iso
                if t_end_lb + float(t["C_cycles"]) / cap + float(t["tau_dn_s"]) > float(t["deadline_s"]) + 1e-9:
                    continue

            f_safe = min(cap, max(f_min, f_hz))
            if srv_res[j] + f_safe > cap + 1.0:
                continue
            srv_res[j] += f_safe
            edge_actions.append({"task": tid, "mode": "edge", "server": j, "f_hz": float(f_safe)})

    # 1st-order static active-link protection: withdraw tail edge grants if they statically kill U(t_k)
    active = obs["active_uplinks"]
    if active and edge_actions and not ortho:
        while edge_actions:
            # Build co-subband device-deduplicated interference from active + kept batch
            kept_devs_by_sb: dict[int, set[int]] = {sb: set() for sb in range(n_sb)}
            for a in edge_actions:
                di = int(pend_map[a["task"]]["device"])
                sb = int(devs[di]["subband"]) % n_sb
                kept_devs_by_sb[sb].add(di)
            for u in active:
                du = int(u["device"])
                sb = int(devs[du]["subband"]) % n_sb
                kept_devs_by_sb[sb].add(du)

            all_active_safe = True
            for u in active:
                du = int(u["device"])
                ju = int(u["server"])
                sb = int(devs[du]["subband"]) % n_sb
                sig = float(u["p_tx_w"]) * float(h_mat[du, ju])
                interf = sum(
                    float(devs[ov]["p_tx_w"]) * float(h_mat[ov, ju])
                    for ov in kept_devs_by_sb[sb]
                    if ov != du
                )
                r_stat = shannon_rate(bw, sig, noise, interf)
                t_fin = now + float(u["rem_bits"]) / r_stat + float(u["C_cycles"]) / float(u["f_hz"]) + float(u["tau_dn_s"])
                if t_fin > float(u["deadline_s"]) + 1e-9:
                    all_active_safe = False
                    break
            if all_active_safe:
                break
            popped = edge_actions.pop()
            srv_res[int(popped["server"])] -= float(popped["f_hz"])

    # Nominal 2% jitter margin into unused headroom for kept edge actions
    for a in edge_actions:
        j = int(a["server"])
        headroom = max(0.0, float(srv_cap[j] - srv_res[j]))
        bump = min(headroom, 0.02 * float(a["f_hz"]))
        srv_res[j] += bump
        a["f_hz"] = float(a["f_hz"]) + bump

    return {"actions": clean_non_edge + edge_actions}


def wrap_with_common_guard(policy: Callable[[dict], dict], name: str | None = None) -> Callable[[dict], dict]:
    """Return baseline wrapped with Stage-3 CommonGuard (+CG)."""
    def guarded_policy(obs: dict) -> dict:
        raw = policy(obs)
        return apply_common_guard(obs, raw)

    base_name = getattr(policy, "__name__", "baseline")
    guarded_policy.__name__ = name if name is not None else f"{base_name}+CG"
    return guarded_policy
