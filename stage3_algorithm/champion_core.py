"""CLWB-Dual: Cross-Layer Pigouvian Externality & Hereditary Fluid Control (Stage 3 Champion).

Strictly derived from Stage-2 `Model v1.0` Mathematical Leverage Inventory (`MODEL_v1.0_SPEC.md` §2.4):
  * Mechanism 1 (`use_externality_pricing`): Vectorized convex quadratic two-flow continuous-fluid
    cross-layer externality E_q = sum_v Delta_v(q)^2 over pre-deduplicated co-subband pending
    representatives with best-SNR feasible server selection and Pigouvian-EDF urgency ordering.
  * Mechanism 2 (`use_lagrangian_defer`): Causal inter-epoch deferral stopping via idle-radio
    NIC eligibility, active-link half-slack shield preservation, and bounded 2-tick fluid horizon.
  * Mechanism 3 (`use_fluid_contraction`): Exact continuous fluid-SINR verification, O(log P)
    hereditary prefix contraction (Theorem 2.1) with primary-server O(M) spatial-reuse backfill,
    predictive actuation-lag compensation, and non-hoarding tick+upload timing shield allocation.

Zero magic constants: every numeric literal is a mathematical identity {0, 0.5, 1, 2},
numerical convergence tolerance, or directly read from `obs`. Zero scenario-specific branches.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))


def _dispatch_device_queues(
    n: int,
    busy_dev: np.ndarray,
    tx_q: list[list[int]],
    running_mask: np.ndarray,
) -> None:
    """Release head-of-line EDF uplink on every idle terminal radio."""
    for di in range(n):
        if (not busy_dev[di]) and tx_q[di]:
            nxt_idx = tx_q[di].pop(0)
            running_mask[nxt_idx] = True
            busy_dev[di] = True


def _init_rollout_state(
    obs: dict,
    sorted_cand: list[dict],
    rel_immediate: bool,
) -> tuple[Any, ...]:
    """Allocate contiguous state vectors for fast continuous fluid-SINR rollout."""
    devs = obs["devices"]
    links = obs["links"]
    h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
    p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
    n_sb = max(1, int(links["n_subbands"]))
    sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
    n = len(devs)

    active = obs["active_uplinks"]
    n_act = len(active)
    k_tot = n_act + len(sorted_cand)

    f_tids: list[str] = []
    f_is_act = np.zeros(k_tot, dtype=bool)
    f_dev = np.zeros(k_tot, dtype=int)
    f_srv = np.zeros(k_tot, dtype=int)
    f_sb = np.zeros(k_tot, dtype=int)
    f_ptx = np.zeros(k_tot, dtype=float)
    f_sig = np.zeros(k_tot, dtype=float)
    f_rem = np.zeros(k_tot, dtype=float)
    f_dl = np.zeros(k_tot, dtype=float)
    f_cyc = np.zeros(k_tot, dtype=float)
    f_flock = np.zeros(k_tot, dtype=float)
    f_tdn = np.zeros(k_tot, dtype=float)
    running_mask = np.zeros(k_tot, dtype=bool)
    busy_dev = np.zeros(n, dtype=bool)
    tx_q: list[list[int]] = [[] for _ in range(n)]
    pending_release: list[int] = []

    for idx, u in enumerate(active):
        di = int(u["device"])
        ji = int(u["server"])
        f_tids.append(str(u["id"]))
        f_is_act[idx] = True
        f_dev[idx] = di
        f_srv[idx] = ji
        f_sb[idx] = int(sb_arr[di])
        f_ptx[idx] = float(u["p_tx_w"])
        f_sig[idx] = float(u["p_tx_w"]) * float(h_mat[di, ji])
        f_rem[idx] = float(max(u["rem_bits"], 0.0))
        f_dl[idx] = float(u["deadline_s"])
        f_cyc[idx] = float(u["C_cycles"])
        f_flock[idx] = float(u["f_hz"])
        f_tdn[idx] = float(u["tau_dn_s"])
        running_mask[idx] = True
        busy_dev[di] = True

    for offset, g in enumerate(sorted_cand):
        idx = n_act + offset
        di = int(g["device"])
        ji = int(g["server"])
        f_tids.append(str(g["id"]))
        f_dev[idx] = di
        f_srv[idx] = ji
        f_sb[idx] = int(sb_arr[di])
        f_ptx[idx] = float(p_arr[di])
        f_sig[idx] = float(p_arr[di]) * float(h_mat[di, ji])
        f_rem[idx] = float(g["D_bits"])
        f_dl[idx] = float(g["deadline_s"])
        f_cyc[idx] = float(g["C_cycles"])
        f_tdn[idx] = float(g["tau_dn_s"])
        if rel_immediate:
            tx_q[di].append(idx)
        else:
            pending_release.append(idx)

    f_rad_m = f_ptx[:, None] * h_mat[f_dev, :]
    return (
        f_tids, f_is_act, f_dev, f_srv, f_sb, f_sig, f_rem,
        f_dl, f_cyc, f_flock, f_tdn, f_rad_m,
        running_mask, busy_dev, tx_q, pending_release,
    )


def _compute_run_rates(
    run_idx: np.ndarray,
    ortho: bool,
    n_sb: int,
    bw: float,
    noise: float,
    f_sb: np.ndarray,
    f_srv: np.ndarray,
    f_sig: np.ndarray,
    f_rad_m: np.ndarray,
    p_sb_srv: np.ndarray,
) -> np.ndarray:
    """Compute instantaneous Shannon uplink rates across active running flows."""
    if ortho:
        return np.maximum(bw * np.log2(1.0 + f_sig[run_idx] / max(noise, 1e-30)), 1e-3)
    p_sb_srv.fill(0.0)
    r_sbs = f_sb[run_idx]
    if n_sb == 1:
        p_sb_srv[0, :] = np.sum(f_rad_m[run_idx, :], axis=0)
    else:
        for sb_i in range(n_sb):
            sub = run_idx[r_sbs == sb_i]
            if sub.size:
                p_sb_srv[sb_i, :] = np.sum(f_rad_m[sub, :], axis=0)
    interf = np.maximum(0.0, p_sb_srv[r_sbs, f_srv[run_idx]] - f_sig[run_idx])
    return np.maximum(bw * np.log2(1.0 + f_sig[run_idx] / np.maximum(noise + interf, 1e-30)), 1e-3)


def _run_fluid_rollout(
    obs: dict,
    sorted_cand: list[dict],
    t_act: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate continuous-fluid SINR dynamics and return upload finish timestamps."""
    now = float(obs["time"])
    n = len(obs["devices"])
    m = len(obs["servers"])
    links = obs["links"]
    n_sb = max(1, int(links["n_subbands"]))
    bw = float(links["bandwidth_hz"])
    noise = float(links["noise_w"])
    ortho = bool(links["orthogonal"])
    rel_immediate = t_act <= now + 1e-12

    (
        f_tids, f_is_act, f_dev, f_srv, f_sb, f_sig, f_rem,
        f_dl, f_cyc, f_flock, f_tdn, f_rad_m,
        running_mask, busy_dev, tx_q, pending_release,
    ) = _init_rollout_state(obs, sorted_cand, rel_immediate)

    t_tx_end_arr = np.full(len(f_tids), math.inf, dtype=float)
    _dispatch_device_queues(n, busy_dev, tx_q, running_mask)
    t_cur = now
    released = rel_immediate
    p_sb_srv = np.zeros((n_sb, m), dtype=float)

    while np.any(running_mask) or not released:
        if not np.any(running_mask):
            t_cur = t_act
            released = True
            for q_idx in pending_release:
                tx_q[int(f_dev[q_idx])].append(q_idx)
            pending_release.clear()
            _dispatch_device_queues(n, busy_dev, tx_q, running_mask)
            continue

        run_idx = np.flatnonzero(running_mask)
        rates = _compute_run_rates(run_idx, ortho, n_sb, bw, noise, f_sb, f_srv, f_sig, f_rad_m, p_sb_srv)
        min_dt_end = float(np.min(f_rem[run_idx] / rates))
        dt_rel = max(0.0, t_act - t_cur) if (not released and pending_release) else math.inf
        dt = min(min_dt_end, dt_rel)

        f_rem[run_idx] = np.maximum(0.0, f_rem[run_idx] - rates * dt)
        t_cur += dt

        done_sub = run_idx[f_rem[run_idx] <= 1e-6]
        finished_any = done_sub.size > 0
        if finished_any:
            f_rem[done_sub] = 0.0
            t_tx_end_arr[done_sub] = t_cur
            running_mask[done_sub] = False
            for idx_d in done_sub:
                busy_dev[int(f_dev[idx_d])] = False

        if (not released) and abs(t_cur - t_act) <= 1e-12:
            released = True
            for q_idx in pending_release:
                tx_q[int(f_dev[q_idx])].append(q_idx)
            pending_release.clear()
            _dispatch_device_queues(n, busy_dev, tx_q, running_mask)
        elif finished_any:
            _dispatch_device_queues(n, busy_dev, tx_q, running_mask)

    return f_tids, f_is_act, f_srv, f_dl, f_cyc, f_flock, f_tdn, t_tx_end_arr


def _check_fluid_feasibility(
    obs: dict,
    rollout: tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Verify compute window deadlines, active-link safety, and per-server RT CPU caps."""
    srvs = obs["servers"]
    f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    f_min = float(obs["f_min_hz"])
    f_tids, f_is_act, f_srv, f_dl, f_cyc, f_flock, f_tdn, t_tx_end_arr = rollout

    f_min_map: dict[str, float] = {}
    t_tx_end: dict[str, float] = {}
    active_ok: dict[str, bool] = {}
    if np.any(~np.isfinite(t_tx_end_arr)):
        return {"t_tx_end": t_tx_end, "f_min": f_min_map, "active_ok": active_ok, "feasible": False}

    srv_added = np.zeros(len(srvs), dtype=float)
    feasible = True
    for idx, tid in enumerate(f_tids):
        t_up_e = float(t_tx_end_arr[idx])
        t_tx_end[tid] = t_up_e
        if f_is_act[idx]:
            f_lock = float(f_flock[idx])
            ok = (t_up_e + float(f_cyc[idx]) / max(f_lock, 1.0) + float(f_tdn[idx])) <= (float(f_dl[idx]) + 1e-9)
            active_ok[tid] = ok
            f_min_map[tid] = f_lock
            if not ok:
                feasible = False
        else:
            slack_cmp = float(f_dl[idx]) - t_up_e - float(f_tdn[idx])
            if slack_cmp <= 1e-9:
                f_min_map[tid] = math.inf
                feasible = False
            else:
                f_req = max(f_min, float(f_cyc[idx]) / slack_cmp)
                f_min_map[tid] = f_req
                j = int(f_srv[idx])
                srv_added[j] += f_req
                if f_req > f_avail[j] * (1.0 - 1e-9):
                    feasible = False

    if feasible and np.any(srv_res + srv_added > f_avail * (1.0 - 1e-9)):
        feasible = False

    return {"t_tx_end": t_tx_end, "f_min": f_min_map, "active_ok": active_ok, "feasible": bool(feasible)}


def _fast_fluid_verify(
    obs: dict,
    candidate_edge_grants: list[dict],
    act_lag_s: float = 0.0,
) -> dict[str, Any]:
    """Sort edge candidates in EDF NIC order and evaluate exact continuous fluid feasibility."""
    if not obs["active_uplinks"] and not candidate_edge_grants:
        return {"t_tx_end": {}, "f_min": {}, "active_ok": {}, "feasible": True}
    t_act = float(obs["time"]) + float(obs["delta_sig_s"]) + float(act_lag_s)
    sorted_cand = sorted(
        candidate_edge_grants,
        key=lambda g: (float(g["deadline_s"]), int(g["device"]), str(g["id"])),
    )
    rollout = _run_fluid_rollout(obs, sorted_cand, t_act)
    return _check_fluid_feasibility(obs, rollout)


class CLWBDualPolicy:
    """Single-core Cross-Layer Pigouvian Externality & Hereditary Fluid Control Solver."""

    def __init__(
        self,
        *,
        use_externality_pricing: bool = True,
        use_lagrangian_defer: bool = True,
        use_fluid_contraction: bool = True,
        name: str = "CLWB_Dual",
    ):
        self.use_externality_pricing = bool(use_externality_pricing)
        self.use_lagrangian_defer = bool(use_lagrangian_defer)
        self.use_fluid_contraction = bool(use_fluid_contraction)
        self.__name__ = str(name)

    def _admit_local_dominant(
        self,
        obs: dict,
        t_exec: float,
    ) -> tuple[list[dict], set[str]]:
        """Pack zero-interference local-feasible tasks via exact unit-value knapsack."""
        devs = obs["devices"]
        n = len(devs)
        delta_sig = float(obs["delta_sig_s"])
        rho_sys = float(obs["rho_sys"])
        f_min = float(obs["f_min_hz"])
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_rem = np.maximum(dev_cap - dev_res, 0.0)

        loc_cands: list[tuple[float, float, str, dict]] = []
        for t in obs["pending"]:
            i = int(t["device"])
            rem_l = float(t["deadline_s"]) - (t_exec + delta_sig)
            if rem_l > 1e-9:
                f_req_l = max(f_min, float(t["C_cycles"]) / rem_l)
                if f_req_l <= dev_rem[i] * (1.0 - 1e-9):
                    loc_cands.append((f_req_l, float(t["deadline_s"]), str(t["id"]), t))
        loc_cands.sort(key=lambda z: (z[0], z[1], z[2]))

        packed: list[tuple[str, int, float]] = []
        used_loc = np.zeros(n, dtype=float)
        for f_req_l, _dl, tid, t in loc_cands:
            i = int(t["device"])
            if used_loc[i] + f_req_l <= dev_rem[i] * (1.0 - 1e-9):
                used_loc[i] += f_req_l
                packed.append((tid, i, f_req_l))

        rem_hr = np.maximum(0.0, dev_rem - used_loc - 1e-6 * dev_cap)
        actions: list[dict] = []
        loc_ids: set[str] = set()
        for tid, i, f_req_l in packed:
            bump_l = min(float(rem_hr[i]), (1.0 - rho_sys) * f_req_l) if self.use_fluid_contraction else 0.0
            rem_hr[i] = max(0.0, float(rem_hr[i]) - bump_l)
            actions.append({"task": tid, "mode": "local", "server": None, "f_hz": float(f_req_l + bump_l)})
            loc_ids.add(tid)
        return actions, loc_ids

    def _compute_two_flow_externality(
        self,
        obs: dict,
        t_dev: np.ndarray,
        t_sb: np.ndarray,
        t_ptx: np.ndarray,
        t_d: np.ndarray,
        t_c: np.ndarray,
        t_dl: np.ndarray,
        t_tdn: np.ndarray,
        sig_pm: np.ndarray,
        i_base_pm: np.ndarray,
        rates_iso: np.ndarray,
        u_iso_pm: np.ndarray,
        f_iso_pm: np.ndarray,
        feas_iso_pm: np.ndarray,
        srv_rem: np.ndarray,
    ) -> np.ndarray:
        """Compute convex quadratic two-flow fluid cross-layer externality E_q over pending pool P."""
        p_cnt = len(t_dev)
        links = obs["links"]
        if bool(links["orthogonal"]):
            return np.zeros(p_cnt, dtype=float)

        now = float(obs["time"])
        delta_sig = float(obs["delta_sig_s"])
        f_min = float(obs["f_min_hz"])
        t_exec = now + delta_sig
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)

        snr_cost = np.where(feas_iso_pm, -h_mat[t_dev, :], 1e18)
        best_j_p = np.argmin(snr_cost, axis=1)
        has_feas_p = np.any(feas_iso_pm, axis=1)
        idx_p = np.arange(p_cnt)
        sig_v = sig_pm[idx_p, best_j_p]
        r_v0 = rates_iso[idx_p, best_j_p]
        u_q0 = u_iso_pm[idx_p, best_j_p]
        f_v0 = f_iso_pm[idx_p, best_j_p]
        cap_v = srv_rem[best_j_p]
        base_v = i_base_pm[idx_p, best_j_p]

        # Co-subband representative pending task damage (1 earliest-deadline task per victim device)
        same_sb = (t_sb[:, None] == t_sb[None, :]) & (t_dev[:, None] != t_dev[None, :])
        i_qv = np.where(same_sb, t_ptx[:, None] * h_mat[t_dev][:, best_j_p], 0.0)
        r_vq = np.maximum(bw * np.log2(1.0 + sig_v[None, :] / np.maximum(noise + base_v[None, :] + i_qv, 1e-30)), 1e-3)
        rem_bits_v = np.maximum(0.0, t_d[None, :] - r_vq * u_q0[:, None])
        u_vq = np.where(rem_bits_v <= 0.0, t_d[None, :] / r_vq, u_q0[:, None] + rem_bits_v / r_v0[None, :])
        slack_vq = t_dl[None, :] - t_exec - u_vq - t_tdn[None, :]
        killed = (slack_vq <= 1e-9) | (t_c[None, :] / np.maximum(slack_vq, 1e-9) > cap_v[None, :] * (1.0 - 1e-9))
        f_vq = np.where(killed, cap_v[None, :] + f_v0[None, :], np.maximum(f_min, t_c[None, :] / np.maximum(slack_vq, 1e-6)))
        rel_d = np.where(
            same_sb & has_feas_p[:, None] & has_feas_p[None, :],
            np.minimum(1.0, np.maximum(0.0, (f_vq - f_v0[None, :]) / np.maximum(cap_v[None, :], f_min))),
            0.0,
        )
        dmg_qv = rel_d * rel_d

        rep_mask = np.zeros(p_cnt, dtype=bool)
        seen_vdev: set[int] = set()
        for idx_e in np.argsort(t_dl):
            if has_feas_p[idx_e]:
                dv = int(t_dev[idx_e])
                if dv not in seen_vdev:
                    seen_vdev.add(dv)
                    rep_mask[idx_e] = True
        return np.sum(dmg_qv[:, rep_mask], axis=1)

    @staticmethod
    def _pre_dedup_idle_indices(
        edge_pool: list[dict],
        t_dl: np.ndarray,
        feas_iso_pm: np.ndarray,
    ) -> list[int]:
        """Select at most one earliest-deadline isolated-feasible task index per device."""
        has_feas = np.any(feas_iso_pm, axis=1)
        seen_dev: set[int] = set()
        keep_idx: list[int] = []
        order = sorted(range(len(edge_pool)), key=lambda idx: (float(t_dl[idx]), str(edge_pool[idx]["id"])))
        for idx in order:
            if not has_feas[idx]:
                continue
            dv = int(edge_pool[idx]["device"])
            if dv not in seen_dev:
                seen_dev.add(dv)
                keep_idx.append(idx)
        return keep_idx

    def _build_candidate_order(
        self,
        obs: dict,
        edge_pool: list[dict],
        t_exec: float,
        srv_rem: np.ndarray,
        f_avail: np.ndarray,
    ) -> tuple[list[dict], np.ndarray]:
        """Rank edge candidates via convex Pigouvian-EDF urgency and best-SNR server selection."""
        now = float(obs["time"])
        tick_s = float(obs["tick_s"])
        delta_sig = float(obs["delta_sig_s"])
        f_min = float(obs["f_min_hz"])
        devs = obs["devices"]
        links = obs["links"]
        m = len(obs["servers"])
        h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
        p_arr = np.asarray([float(d["p_tx_w"]) for d in devs], dtype=float)
        n_sb = max(1, int(links["n_subbands"]))
        sb_arr = np.asarray([int(d["subband"]) % n_sb for d in devs], dtype=int)
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        ortho = bool(links["orthogonal"])

        interf_act = np.zeros((n_sb, m), dtype=float)
        if not ortho:
            for u in obs["active_uplinks"]:
                di = int(u["device"])
                interf_act[sb_arr[di], :] += float(u["p_tx_w"]) * h_mat[di, :]

        p_cnt = len(edge_pool)
        t_dev = np.asarray([int(t["device"]) for t in edge_pool], dtype=int)
        t_sb = sb_arr[t_dev]
        t_ptx = p_arr[t_dev]
        t_d = np.asarray([float(t["D_bits"]) for t in edge_pool], dtype=float)
        t_c = np.asarray([float(t["C_cycles"]) for t in edge_pool], dtype=float)
        t_dl = np.asarray([float(t["deadline_s"]) for t in edge_pool], dtype=float)
        t_tdn = np.asarray([float(t["tau_dn_s"]) for t in edge_pool], dtype=float)
        sig_pm = t_ptx[:, None] * h_mat[t_dev, :]

        i_base_pm = np.zeros((p_cnt, m), dtype=float) if ortho else interf_act[t_sb, :]
        rates_iso = np.maximum(bw * np.log2(1.0 + sig_pm / np.maximum(noise + i_base_pm, 1e-30)), 1e-3)
        u_iso_pm = t_d[:, None] / rates_iso
        rem_iso_pm = t_dl[:, None] - t_exec - u_iso_pm - t_tdn[:, None]
        feas_iso_pm = rem_iso_pm > 1e-9
        f_iso_pm = np.where(feas_iso_pm, np.maximum(f_min, t_c[:, None] / np.maximum(rem_iso_pm, 1e-6)), 1e18)
        feas_iso_pm = feas_iso_pm & (f_iso_pm <= srv_rem[None, :] * (1.0 - 1e-9))

        if self.use_lagrangian_defer:
            keep = self._pre_dedup_idle_indices(edge_pool, t_dl, feas_iso_pm)
            if not keep:
                return [], np.zeros((0, m), dtype=float)
            k_arr = np.asarray(keep, dtype=int)
            edge_pool = [edge_pool[idx] for idx in keep]
            t_dev, t_sb, t_ptx = t_dev[k_arr], t_sb[k_arr], t_ptx[k_arr]
            t_d, t_c, t_dl, t_tdn = t_d[k_arr], t_c[k_arr], t_dl[k_arr], t_tdn[k_arr]
            sig_pm, i_base_pm = sig_pm[k_arr], i_base_pm[k_arr]
            rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm = rates_iso[k_arr], u_iso_pm[k_arr], f_iso_pm[k_arr], feas_iso_pm[k_arr]

        rates_clean = np.maximum(bw * np.log2(1.0 + sig_pm / max(noise, 1e-30)), 1e-3)
        rem_next_pm = t_dl[:, None] - (now + tick_s + delta_sig) - t_d[:, None] / rates_clean - t_tdn[:, None]
        can_defer_p = np.any((rem_next_pm > 1e-9) & (t_c[:, None] / np.maximum(rem_next_pm, 1e-6) <= f_avail[None, :]), axis=1)

        if self.use_externality_pricing:
            ext_pen_p = self._compute_two_flow_externality(
                obs, t_dev, t_sb, t_ptx, t_d, t_c, t_dl, t_tdn,
                sig_pm, i_base_pm, rates_iso, u_iso_pm, f_iso_pm, feas_iso_pm, srv_rem,
            )
        else:
            ext_pen_p = np.zeros(len(edge_pool), dtype=float)

        ranked: list[dict[str, Any]] = []
        for q_i, t in enumerate(edge_pool):
            feas_js = [j for j in range(m) if feas_iso_pm[q_i, j]]
            if not feas_js:
                continue
            feas_js.sort(key=lambda j: (-float(h_mat[t_dev[q_i], j]), float(f_iso_pm[q_i, j]), j))
            eff_dl = max(0.0, float(t["deadline_s"]) - now) * (1.0 + max(0.0, float(ext_pen_p[q_i]) - 1.0)) if self.use_externality_pricing else float(t["deadline_s"])
            s_key = (eff_dl, float(f_iso_pm[q_i, feas_js[0]]), float(t["deadline_s"]), str(t["id"]))
            ranked.append({"sort": s_key, "q_i": q_i, "t": t, "feas_js": feas_js})

        ranked.sort(key=lambda item: item["sort"])

        cands: list[dict] = []
        for item in ranked:
            q_i = int(item["q_i"])
            t = item["t"]
            feas_js = item["feas_js"]
            is_fin = float(f_iso_pm[q_i, feas_js[0]]) / max(float(f_avail[feas_js[0]]), 1.0) > 1e-3
            cands.append({
                "id": t["id"], "device": int(t["device"]), "server": int(feas_js[0]),
                "D_bits": float(t["D_bits"]), "deadline_s": float(t["deadline_s"]),
                "C_cycles": float(t["C_cycles"]), "tau_dn_s": float(t["tau_dn_s"]),
                "feas_js": feas_js, "q_i": q_i,
                "can_defer": bool(can_defer_p[q_i]) and is_fin,
            })
        return cands, f_iso_pm

    def _compute_active_slack_floor(
        self,
        obs: dict,
        verify_fn: Callable[[list[dict]], dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Measure active uplink slack baseline to preserve half-slack headroom against interference."""
        active = obs["active_uplinks"]
        act_meta = {
            u["id"]: (float(u["deadline_s"]), float(u["C_cycles"]) / max(float(u["f_hz"]), 1.0) + float(u["tau_dn_s"]))
            for u in active
        }
        act_min_slack: dict[str, float] = {}
        if self.use_lagrangian_defer and (not bool(obs["links"]["orthogonal"])) and active:
            tick_s = float(obs["tick_s"])
            res_base = verify_fn([])
            if res_base["feasible"]:
                for u in active:
                    uid = u["id"]
                    s0 = act_meta[uid][0] - (float(res_base["t_tx_end"][uid]) + act_meta[uid][1])
                    if s0 > 0.0:
                        act_min_slack[uid] = min(0.5 * s0, tick_s)
        return act_min_slack, act_meta

    def _is_batch_admissible(
        self,
        obs: dict,
        cand_list: list[dict],
        check_horizon: bool,
        act_min_slack: dict[str, float],
        act_meta: dict[str, tuple[float, float]],
        verify_fn: Callable[[list[dict]], dict[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        """Check exact fluid feasibility, 2-tick deferrable radio horizon, and active half-slack shield."""
        res = verify_fn(cand_list)
        if not res["feasible"]:
            return False, res
        if check_horizon and self.use_lagrangian_defer and (not bool(obs["links"]["orthogonal"])):
            horizon_limit = float(obs["time"]) + 2.0 * float(obs["tick_s"])
            t_end_m = res["t_tx_end"]
            for g in cand_list:
                if g["can_defer"] and float(t_end_m[g["id"]]) > horizon_limit:
                    return False, res
            for uid, min_s in act_min_slack.items():
                if act_meta[uid][0] - (float(t_end_m[uid]) + act_meta[uid][1]) < min_s:
                    return False, res
        return True, res

    def _contract_prefix(
        self,
        cands: list[dict],
        check_fn: Callable[[list[dict]], tuple[bool, dict[str, Any]]],
    ) -> tuple[int, dict[str, Any]]:
        """Binary prefix contraction exploiting Theorem 2.1 downward-closed feasibility."""
        lo, hi = 0, len(cands)
        best_res: dict[str, Any] = {"t_tx_end": {}, "f_min": {}, "active_ok": {}, "feasible": True}
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ok_m, res_m = check_fn(cands[:mid])
            if ok_m:
                lo = mid
                best_res = res_m
            else:
                hi = mid - 1
        return lo, best_res

    def _hereditary_select_grants(
        self,
        obs: dict,
        cands: list[dict],
        f_iso_pm: np.ndarray,
        srv_rem: np.ndarray,
        t_enter: float,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Select batch via O(log P) hereditary contraction, O(M) alternate fill, and lag refresh."""
        if not cands:
            return [], {"t_tx_end": {}, "f_min": {}, "active_ok": {}, "feasible": True}

        now = float(obs["time"])
        delta_sig = float(obs["delta_sig_s"])
        t_exec = now + delta_sig
        if not self.use_fluid_contraction:
            tmp_rem = srv_rem.copy()
            kept_nf: list[dict] = []
            f_map_nf: dict[str, float] = {}
            t_end_nf: dict[str, float] = {}
            for g in cands:
                j = int(g["server"])
                fe = float(f_iso_pm[int(g["q_i"]), j])
                if tmp_rem[j] + 1.0 >= fe:
                    tmp_rem[j] -= fe
                    kept_nf.append(g)
                    f_map_nf[g["id"]] = fe
                    t_end_nf[g["id"]] = t_exec
            return kept_nf, {"t_tx_end": t_end_nf, "f_min": f_map_nf, "active_ok": {}, "feasible": True}

        dt_verify_est = 0.0

        def _verify_safe(cand_list: list[dict]) -> dict[str, Any]:
            nonlocal dt_verify_est
            t_v0 = time.perf_counter()
            lag_pred = (t_v0 - t_enter) + min(dt_verify_est, delta_sig)
            res = _fast_fluid_verify(obs, cand_list, act_lag_s=lag_pred)
            dt_verify_est = max(dt_verify_est, time.perf_counter() - t_v0)
            return res

        act_min_slack, act_meta = self._compute_active_slack_floor(obs, _verify_safe)
        ok_all, res_all = self._is_batch_admissible(obs, cands, True, act_min_slack, act_meta, _verify_safe)
        if ok_all:
            kept_grants = list(cands)
            fluid_res = res_all
        else:
            lo, fluid_res = self._contract_prefix(
                cands[:-1],
                lambda sub: self._is_batch_admissible(obs, sub, True, act_min_slack, act_meta, _verify_safe),
            )
            kept_grants = list(cands[:lo])
            budget_checks = 2 * len(obs["servers"]) + 2
            checks = 0
            for g in cands[lo:]:
                if checks >= budget_checks:
                    break
                checks += 1
                ok_t, res_t = self._is_batch_admissible(
                    obs, kept_grants + [g], True, act_min_slack, act_meta, _verify_safe,
                )
                if ok_t:
                    kept_grants.append(g)
                    fluid_res = res_t

        if kept_grants:
            res_fin = _verify_safe(kept_grants)
            if not res_fin["feasible"]:
                lo_f, fluid_res = self._contract_prefix(kept_grants, lambda sub: (res_s["feasible"], res_s) if (res_s := _verify_safe(sub)) else (False, {}))
                kept_grants = kept_grants[:lo_f]
            else:
                fluid_res = res_fin
        return kept_grants, fluid_res

    def _allocate_shielded_edge(
        self,
        obs: dict,
        kept_grants: list[dict],
        fluid_res: dict[str, Any],
        srv_res: np.ndarray,
        f_avail: np.ndarray,
    ) -> list[dict]:
        """Assign exact edge frequencies augmented with non-hoarding tick+upload timing shield."""
        if not kept_grants:
            return []
        now = float(obs["time"])
        tick_s = float(obs["tick_s"])
        delta_sig = float(obs["delta_sig_s"])

        srv_used_b = srv_res.copy()
        for g in kept_grants:
            srv_used_b[int(g["server"])] += float(fluid_res["f_min"][g["id"]])

        edge_actions: list[dict] = []
        for g in kept_grants:
            tid = g["id"]
            j = int(g["server"])
            fb = float(fluid_res["f_min"][tid])
            if self.use_fluid_contraction:
                t_up_end = float(fluid_res["t_tx_end"][tid])
                slack_cmp = max(float(g["deadline_s"]) - t_up_end - float(g["tau_dn_s"]), 1e-6)
                t_up_e = max(0.0, t_up_end - now - delta_sig)
                shield = min(tick_s + t_up_e, 0.5 * slack_cmp)
                f_target = float(g["C_cycles"]) / max(slack_cmp - shield, 0.5 * slack_cmp)
                headroom = max(0.0, float(f_avail[j] - srv_used_b[j]) - 1e-6 * float(f_avail[j]))
                bump = min(headroom, max(0.0, f_target - fb), fb)
                srv_used_b[j] += bump
            else:
                bump = 0.0
            edge_actions.append({"task": tid, "mode": "edge", "server": j, "f_hz": min(float(f_avail[j]), fb + bump)})
        return edge_actions

    def __call__(self, obs: dict) -> dict:
        t_enter = time.perf_counter()
        pending = obs["pending"]
        if not pending:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        actions, loc_ids = self._admit_local_dominant(obs, t_exec)
        rem_pending = [t for t in pending if t["id"] not in loc_ids]
        if not rem_pending:
            return {"actions": actions}

        devs = obs["devices"]
        srvs = obs["servers"]
        f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_rem = np.maximum(f_avail - srv_res, float(obs["f_min_hz"]))

        if self.use_lagrangian_defer:
            edge_pool = [
                t for t in rem_pending
                if (not bool(devs[int(t["device"])]["radio_busy"]))
                and float(devs[int(t["device"])]["tx_queue_bits"]) <= 1e-9
            ]
        else:
            edge_pool = list(rem_pending)

        kept_ids: set[str] = set()
        if edge_pool:
            cands, f_iso_pm = self._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)
            kept_grants, fluid_res = self._hereditary_select_grants(obs, cands, f_iso_pm, srv_rem, t_enter)
            kept_ids = {g["id"] for g in kept_grants}
            actions.extend(self._allocate_shielded_edge(obs, kept_grants, fluid_res, srv_res, f_avail))

        if not self.use_lagrangian_defer:
            for t in rem_pending:
                if t["id"] not in kept_ids:
                    actions.append({"task": t["id"], "mode": "drop", "server": None, "f_hz": None})

        return {"actions": actions}


def make_clwb_dual(
    *,
    use_externality_pricing: bool = True,
    use_lagrangian_defer: bool = True,
    use_fluid_contraction: bool = True,
    name: str = "CLWB_Dual",
) -> CLWBDualPolicy:
    return CLWBDualPolicy(
        use_externality_pricing=use_externality_pricing,
        use_lagrangian_defer=use_lagrangian_defer,
        use_fluid_contraction=use_fluid_contraction,
        name=name,
    )
