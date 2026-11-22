"""Dual-Track Baseline Pool (1960-2020 Classical + Stage-1 SOTA) & Degenerate Anchors (Model v1.0).

All policies operate strictly through the causal observation mask `obs` returned by
`model_spec.EdgeUplinkEnv.observe(t_k)`:
  * 1960-2020 Classics:
      1. `EDF_PreBatch`  (Liu-Layland 1973 / Horn 1974): EDF priority, pre-batch SINR.
      2. `LLF_PreBatch`  (Mok 1983): least-laxity-first, pre-batch SINR.
      3. `Bambos_ALP`    (Bambos-Chen-Pottie ToN 2000): active-link protection + gradual
         removal via fixed wireless SINR threshold (no CPU slack feedback).
      4. `Chen_Game`     (Chen et al. ToN 2016): Gauss-Seidel multi-user best-response game.
  * Stage-1 SOTA Comparators:
      5. `DGOLB`         (Xu et al. IEEE TMC 2026): slack-ordered energy-minimizing offloading
         with pre-batch interference pricing and dynamic server reservation cap.
      6. `SFA`           (Sequential Feasible Admission): EDF scan running full exact fluid-SINR
         batch verification on every candidate addition (O(P^2 M) scaling).
      7. `IBR_5`         (Iterative Batch Reoptimization, 5 rounds): multi-round batch SINR
         re-pricing + Gauss-Seidel feasible relocation refinement.

Also provides `verify_degenerate_anchors()` verifying the 4 exact analytical limit solutions
against `exact_oracle.solve_single_tick_exact`.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from exact_oracle import evaluate_fluid_batch, solve_single_tick_exact
from model_spec import SimConfig, make_micro_scenario, shannon_rate


# --------------------------------------------------------------------------- #
# Shared causal helper utilities (strictly zero future-info access)
# --------------------------------------------------------------------------- #

def _prebatch_interference(obs: dict) -> np.ndarray:
    """Interference matrix I_pre[sb, j] from currently active uplinks ONLY (pre-batch)."""
    links = obs["links"]
    h = np.asarray(links["h_hat_dev_ms"], dtype=float)
    n_sb = max(1, int(links["n_subbands"]))
    m = len(obs["servers"])
    interf = np.zeros((n_sb, m), dtype=float)
    if bool(links["orthogonal"]):
        return interf
    devs = obs["devices"]
    for u in obs["active_uplinks"]:
        du = int(u["device"])
        sb = int(devs[du]["subband"]) % n_sb
        p_u = float(u["p_tx_w"])
        interf[sb, :] += p_u * h[du, :]
    return interf


def _estimate_option(
    obs: dict,
    task: dict,
    mode: str,
    server: Optional[int],
    interf_arr: np.ndarray,
    t_exec: float,
    extra_wait_s: float = 0.0,
    jitter_guard: float = 1.02,
) -> Optional[dict[str, float]]:
    """Compute required frequency and device energy under a given scalar interference vector."""
    i = int(task["device"])
    dl = float(task["deadline_s"])
    c_cyc = float(task["C_cycles"])
    d_bits = float(task["D_bits"])
    tau_dn = float(task["tau_dn_s"])
    f_min = float(obs["f_min_hz"])
    dev = obs["devices"][i]
    p_tx = float(dev["p_tx_w"])

    if mode == "local":
        rem = dl - t_exec
        if rem <= 1e-9:
            return None
        f_req = max(f_min, c_cyc / rem)
        f_cap = float(dev["f_loc_hz"])
        if f_req > f_cap + 1.0:
            return None
        f_req = min(f_cap, max(f_min, f_req * float(jitter_guard)))
        e_dev = 1e-27 * (f_req ** 2) * c_cyc
        return {
            "f_hz": float(f_req),
            "t_up_s": 0.0,
            "finish_s": float(t_exec + c_cyc / f_req),
            "energy_j": float(e_dev),
        }

    assert server is not None
    j = int(server)
    links = obs["links"]
    h = float(links["h_hat_dev_ms"][i, j])
    bw = float(links["bandwidth_hz"])
    noise = float(links["noise_w"])
    sig = p_tx * h
    i_w = float(max(0.0, interf_arr[j]))
    rate = shannon_rate(bw, sig, noise, i_w)
    t_up = d_bits / rate
    t_tx_e = t_exec + extra_wait_s + t_up
    rem = dl - t_tx_e - tau_dn
    if rem <= 1e-9:
        return None
    f_req = max(f_min, c_cyc / rem)
    f_cap = float(obs["servers"][j]["f_avail_hz"])
    if f_req > f_cap + 1.0:
        return None
    f_req = min(f_cap, max(f_min, f_req * float(jitter_guard)))
    return {
        "f_hz": float(f_req),
        "t_up_s": float(t_up),
        "finish_s": float(t_tx_e + c_cyc / f_req + tau_dn),
        "energy_j": float(p_tx * t_up),
    }


def _build_params_from_obs(obs: dict, *, delta_sig_override: Optional[float] = None) -> dict:
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
        "delta_sig_s": float(obs["delta_sig_s"] if delta_sig_override is None else delta_sig_override),
    }


def _cand_grant(task: dict, server: int, dev: dict) -> dict:
    return {
        "id": task["id"],
        "device": int(task["device"]),
        "server": int(server),
        "p_tx_w": float(dev["p_tx_w"]),
        "D_bits": float(task["D_bits"]),
        "deadline_s": float(task["deadline_s"]),
        "C_cycles": float(task["C_cycles"]),
        "tau_dn_s": float(task["tau_dn_s"]),
    }


# --------------------------------------------------------------------------- #
# 1960-2020 Classical Baselines
# --------------------------------------------------------------------------- #

def make_edf_prebatch(non_delay: bool = False) -> Callable[[dict], dict]:
    """Liu-Layland 1973 / Horn 1974 projected baseline: EDF + pre-batch SINR."""
    def policy(obs: dict) -> dict:
        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
        )
        if not pending:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        n = len(devs)
        n_sb = max(1, int(obs["links"]["n_subbands"]))

        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)

        interf_mat = _prebatch_interference(obs)
        actions: list[dict] = []

        for t in pending:
            i = int(t["device"])
            sb = int(devs[i]["subband"]) % n_sb
            interf_vec = interf_mat[sb]

            # Subtract self active uplink if device i currently active
            for u in obs["active_uplinks"]:
                if int(u["device"]) == i:
                    interf_vec = np.maximum(
                        interf_vec - float(u["p_tx_w"]) * np.asarray(obs["links"]["h_hat_dev_ms"])[i, :],
                        0.0,
                    )

            best: Optional[tuple[tuple, str, Optional[int], float]] = None

            # Edge options
            for j in range(m):
                est = _estimate_option(obs, t, "edge", j, interf_vec, t_exec)
                if est is None:
                    continue
                if srv_res[j] + est["f_hz"] <= srv_cap[j] + 1.0:
                    key = (est["finish_s"], j)
                    if best is None or key < best[0]:
                        best = (key, "edge", j, est["f_hz"])

            # Local option
            est_loc = _estimate_option(obs, t, "local", None, interf_vec, t_exec)
            if est_loc is not None and dev_res[i] + est_loc["f_hz"] <= dev_cap[i] + 1.0:
                key = (est_loc["finish_s"], -1)
                if best is None or key < best[0]:
                    best = (key, "local", None, est_loc["f_hz"])

            if best is not None:
                _, md, sv, f_hz = best
                if md == "edge":
                    srv_res[int(sv)] += f_hz
                else:
                    dev_res[i] += f_hz
                actions.append({"task": t["id"], "mode": md, "server": sv, "f_hz": f_hz})
            elif non_delay:
                actions.append({"task": t["id"], "mode": "drop", "server": None, "f_hz": None})

        return {"actions": actions}

    policy.__name__ = "EDF_PreBatch" + ("_NoDefer" if non_delay else "")
    return policy


def make_llf_prebatch() -> Callable[[dict], dict]:
    """Mok 1983 Least-Laxity-First baseline with pre-batch SINR estimation."""
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}
        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        n_sb = max(1, int(obs["links"]["n_subbands"]))
        interf_mat = _prebatch_interference(obs)

        laxities: dict[str, float] = {}
        for t in obs["pending"]:
            i = int(t["device"])
            sb = int(devs[i]["subband"]) % n_sb
            est_times = []
            el = _estimate_option(obs, t, "local", None, interf_mat[sb], t_exec)
            if el is not None:
                est_times.append(el["finish_s"] - t_exec)
            for j in range(m):
                ee = _estimate_option(obs, t, "edge", j, interf_mat[sb], t_exec)
                if ee is not None:
                    est_times.append(ee["finish_s"] - t_exec)
            min_exec = min(est_times) if est_times else (float(t["deadline_s"]) - t_exec + 1.0)
            laxities[t["id"]] = float(t["deadline_s"]) - t_exec - min_exec

        pending = sorted(
            obs["pending"],
            key=lambda t: (laxities[t["id"]], float(t["deadline_s"]), int(t["device"]), str(t["id"])),
        )
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)

        actions: list[dict] = []
        for t in pending:
            i = int(t["device"])
            sb = int(devs[i]["subband"]) % n_sb
            interf_vec = interf_mat[sb]
            best: Optional[tuple[tuple, str, Optional[int], float]] = None

            for j in range(m):
                est = _estimate_option(obs, t, "edge", j, interf_vec, t_exec)
                if est is not None and srv_res[j] + est["f_hz"] <= srv_cap[j] + 1.0:
                    key = (est["finish_s"], j)
                    if best is None or key < best[0]:
                        best = (key, "edge", j, est["f_hz"])

            est_loc = _estimate_option(obs, t, "local", None, interf_vec, t_exec)
            if est_loc is not None and dev_res[i] + est_loc["f_hz"] <= dev_cap[i] + 1.0:
                key = (est_loc["finish_s"], -1)
                if best is None or key < best[0]:
                    best = (key, "local", None, est_loc["f_hz"])

            if best is not None:
                _, md, sv, f_hz = best
                if md == "edge":
                    srv_res[int(sv)] += f_hz
                else:
                    dev_res[i] += f_hz
                actions.append({"task": t["id"], "mode": md, "server": sv, "f_hz": f_hz})

        return {"actions": actions}

    policy.__name__ = "LLF_PreBatch"
    return policy


def make_bambos_alp(sinr_thresh: float = 0.5) -> Callable[[dict], dict]:
    """Bambos-Chen-Pottie (IEEE/ACM ToN 2000) Active-Link Protection + Gradual Removal.

    Classic fixed-SINR-threshold wireless admission gate without cross-layer CPU deadline
    inversion feedback: protects on-air links from dipping below `sinr_thresh` and limits
    per-epoch wireless admissions by SINR feasibility, assigning nominal server shares
    assuming threshold rate `R(sinr_thresh)` rather than coupled fluid completion times.
    """
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        links = obs["links"]
        h = np.asarray(links["h_hat_dev_ms"], dtype=float)
        bw = float(links["bandwidth_hz"])
        noise = float(links["noise_w"])
        ortho = bool(links["orthogonal"])
        n_sb = max(1, int(links["n_subbands"]))
        f_min = float(obs["f_min_hz"])

        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)

        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), int(t["device"]), str(t["id"])),
        )

        # Rate corresponding to fixed wireless SINR threshold gate
        r_thresh = shannon_rate(bw, sinr_thresh, 1.0, 0.0)

        active_set = [
            {
                "device": int(u["device"]),
                "server": int(u["server"]),
                "p_tx_w": float(u["p_tx_w"]),
                "subband": int(devs[int(u["device"])]["subband"]) % n_sb,
            }
            for u in obs["active_uplinks"]
        ]
        admitted_edge: list[tuple[dict, int, float]] = []
        actions: list[dict] = []

        def _sinr_set_ok(links_list: list[dict]) -> bool:
            if ortho:
                return True
            for idx_a, la in enumerate(links_list):
                da = la["device"]
                ja = la["server"]
                sba = la["subband"]
                sig = la["p_tx_w"] * float(h[da, ja])
                interf = 0.0
                for idx_b, lb in enumerate(links_list):
                    if idx_a == idx_b or lb["subband"] != sba:
                        continue
                    # Only distinct active devices radiate simultaneously at threshold check
                    if lb["device"] == da:
                        continue
                    interf += lb["p_tx_w"] * float(h[lb["device"], ja])
                sinr = sig / max(noise + interf, 1e-30)
                if sinr < sinr_thresh:
                    return False
            return True

        # Admit edge links greedily under fixed SINR threshold + local fallback
        for t in pending:
            i = int(t["device"])
            sb = int(devs[i]["subband"]) % n_sb
            p_tx = float(devs[i]["p_tx_w"])
            dl = float(t["deadline_s"])
            c_cyc = float(t["C_cycles"])

            # Try best-SNR edge server if radio idle and SINR threshold gate passes
            srv_order = sorted(range(len(srvs)), key=lambda j: -float(h[i, j]))
            placed_edge = False
            if not devs[i]["radio_busy"] and all(g[0]["device"] != i for g in admitted_edge):
                for j in srv_order:
                    cand_link = {"device": i, "server": j, "p_tx_w": p_tx, "subband": sb}
                    trial = active_set + [cand_link]
                    if _sinr_set_ok(trial):
                        # Nominal share computed from fixed SINR target rate without fluid coupling
                        u_nom = float(t["D_bits"]) / r_thresh
                        rem_nom = dl - (t_exec + u_nom) - float(t["tau_dn_s"])
                        if rem_nom > 1e-9:
                            f_req = max(f_min, c_cyc / rem_nom)
                            if f_req <= srv_cap[j] + 1.0 and srv_res[j] + f_req <= srv_cap[j] + 1.0:
                                active_set.append(cand_link)
                                srv_res[j] += f_req
                                admitted_edge.append((t, j, f_req))
                                actions.append({"task": t["id"], "mode": "edge", "server": j, "f_hz": f_req})
                                placed_edge = True
                                break

            if not placed_edge:
                rem_loc = dl - t_exec
                if rem_loc > 1e-9:
                    f_loc_req = max(f_min, c_cyc / rem_loc)
                    if f_loc_req <= dev_cap[i] + 1.0 and dev_res[i] + f_loc_req <= dev_cap[i] + 1.0:
                        dev_res[i] += f_loc_req
                        actions.append({"task": t["id"], "mode": "local", "server": None, "f_hz": f_loc_req})

        return {"actions": actions}

    policy.__name__ = "Bambos_ALP"
    return policy


def make_chen_game(max_rounds: int = 8) -> Callable[[dict], dict]:
    """Chen et al. (IEEE/ACM ToN 2016) Multi-User Best-Response Interference Game.

    Each pending task plays best-response moves over `(local, edge_j, defer)` against current
    active links plus device-deduplicated static concurrent interference of other tasks' choices.
    Uses static instantaneous rate (no fluid reflow or half-duplex NIC queue accounting).
    """
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        links = obs["links"]
        h = np.asarray(links["h_hat_dev_ms"], dtype=float)
        ortho = bool(links["orthogonal"])
        n_sb = max(1, int(links["n_subbands"]))
        n = len(devs)
        m = len(srvs)

        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
        )
        p_cnt = len(pending)

        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)

        im_act = _prebatch_interference(obs)
        # state per task: ("defer", None, 0.0, 0.0) or ("local", None, f, E) or ("edge", j, f, E)
        choice: list[tuple[str, Optional[int], float, float]] = [("defer", None, 0.0, 0.0)] * p_cnt

        def _batch_device_interf(exclude_dev: int, sb: int) -> np.ndarray:
            arr = im_act[sb].copy()
            if ortho:
                return arr
            seen_devs: set[int] = {int(u["device"]) for u in obs["active_uplinks"]}
            for idx_k, ch in enumerate(choice):
                if ch[0] != "edge":
                    continue
                dk = int(pending[idx_k]["device"])
                if dk == exclude_dev or dk in seen_devs:
                    continue
                if (int(devs[dk]["subband"]) % n_sb) != sb:
                    continue
                seen_devs.add(dk)
                arr += float(devs[dk]["p_tx_w"]) * h[dk, :]
            return arr

        for _r in range(int(max_rounds)):
            changed = 0
            for idx_k, t in enumerate(pending):
                old_md, old_sv, old_f, old_e = choice[idx_k]
                i = int(t["device"])
                sb = int(devs[i]["subband"]) % n_sb

                # Temporarily release old reservation
                if old_md == "edge" and old_sv is not None:
                    srv_res[int(old_sv)] -= old_f
                elif old_md == "local":
                    dev_res[i] -= old_f

                interf_vec = _batch_device_interf(i, sb)

                # Evaluate best response minimizing energy (equivalently game utility = -energy)
                best: Optional[tuple[float, str, Optional[int], float, float]] = None

                el = _estimate_option(obs, t, "local", None, interf_vec, t_exec)
                if el is not None and dev_res[i] + el["f_hz"] <= dev_cap[i] + 1.0:
                    best = (el["energy_j"], "local", None, el["f_hz"], el["energy_j"])

                for j in range(m):
                    ee = _estimate_option(obs, t, "edge", j, interf_vec, t_exec)
                    if ee is not None and srv_res[j] + ee["f_hz"] <= srv_cap[j] + 1.0:
                        if best is None or ee["energy_j"] < best[0] - 1e-12:
                            best = (ee["energy_j"], "edge", j, ee["f_hz"], ee["energy_j"])

                if best is not None:
                    _sc, new_md, new_sv, new_f, new_e = best
                    if new_md == "edge" and new_sv is not None:
                        srv_res[int(new_sv)] += new_f
                    elif new_md == "local":
                        dev_res[i] += new_f
                    if (new_md, new_sv) != (old_md, old_sv) or abs(new_f - old_f) > 10.0:
                        changed += 1
                    choice[idx_k] = (new_md, new_sv, new_f, new_e)
                else:
                    if old_md != "defer":
                        changed += 1
                    choice[idx_k] = ("defer", None, 0.0, 0.0)

            if changed == 0:
                break

        actions = []
        for t, (md, sv, f_hz, _e) in zip(pending, choice):
            if md != "defer":
                actions.append({"task": t["id"], "mode": md, "server": sv, "f_hz": f_hz})
        return {"actions": actions}

    policy.__name__ = f"Chen_Game_{max_rounds}"
    return policy


# --------------------------------------------------------------------------- #
# Stage-1 Recent SOTA Baselines
# --------------------------------------------------------------------------- #

def make_dgolb() -> Callable[[dict], dict]:
    """DGOLB (Xu et al., IEEE TMC 2026) Stage-1 faithful projection:

    Slack-ordered admission + energy-minimizing local/edge mode choice under pre-batch
    interference pricing + dynamic aggregate server reservation cap.
    """
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        n_sb = max(1, int(obs["links"]["n_subbands"]))

        # Sort by remaining slack ascending (TOPRFM order)
        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]) - now, int(t["device"]), str(t["id"])),
        )
        srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        srv_cap = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
        im_pre = _prebatch_interference(obs)

        actions: list[dict] = []
        for t in pending:
            i = int(t["device"])
            sb = int(devs[i]["subband"]) % n_sb
            interf_vec = im_pre[sb]

            best: Optional[tuple[tuple, str, Optional[int], float]] = None
            el = _estimate_option(obs, t, "local", None, interf_vec, t_exec)
            if el is not None and dev_res[i] + el["f_hz"] <= dev_cap[i] + 1.0:
                key = (el["energy_j"], 0)
                if best is None or key < best[0]:
                    best = (key, "local", None, el["f_hz"])

            for j in range(m):
                ee = _estimate_option(obs, t, "edge", j, interf_vec, t_exec)
                if ee is not None and srv_res[j] + ee["f_hz"] <= srv_cap[j] + 1.0:
                    key = (ee["energy_j"], 1, j)
                    if best is None or key < best[0]:
                        best = (key, "edge", j, ee["f_hz"])

            if best is not None:
                _, md, sv, f_hz = best
                if md == "edge":
                    srv_res[int(sv)] += f_hz
                else:
                    dev_res[i] += f_hz
                actions.append({"task": t["id"], "mode": md, "server": sv, "f_hz": f_hz})

        return {"actions": actions}

    policy.__name__ = "DGOLB"
    return policy


def make_sfa(non_delay: bool = False) -> Callable[[dict], dict]:
    """Sequential Feasible Admission (SFA):

    Scans pending tasks in EDF order and tests exact simultaneous fluid-SINR + NIC-queue +
    CPU feasibility (`evaluate_fluid_batch`) for `A union {cand}` on every candidate step.
    Exhibits exact O(P^2 M) per-epoch decision latency scaling under large bursts P.
    """
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        active = obs["active_uplinks"]
        params = _build_params_from_obs(obs)

        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
        )

        dev_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
        f_min = float(obs["f_min_hz"])

        admitted_edge: list[dict] = []
        last_valid_res: Optional[dict[str, Any]] = None
        local_decisions: dict[str, tuple[str, Optional[int], float]] = {}

        for t in pending:
            i = int(t["device"])
            dl = float(t["deadline_s"])
            c_cyc = float(t["C_cycles"])

            best_edge: Optional[tuple[tuple, int, dict, dict]] = None
            for j in range(m):
                cand = _cand_grant(t, j, devs[i])
                res = evaluate_fluid_batch(active, admitted_edge + [cand], params, now)
                if res["feasible"]:
                    t_fin = float(res["t_tx_end"][t["id"]]) + c_cyc / float(res["f_min"][t["id"]]) + float(t["tau_dn_s"])
                    key = (float(res["f_min"][t["id"]]), t_fin, j)
                    if best_edge is None or key < best_edge[0]:
                        best_edge = (key, j, cand, res)

            # Check local alternative
            best_local: Optional[tuple[float, float]] = None
            rem_loc = dl - t_exec
            if rem_loc > 1e-9:
                f_loc_req = max(f_min, c_cyc / rem_loc)
                if dev_res[i] + f_loc_req <= dev_cap[i] + 1.0:
                    best_local = (f_loc_req, t_exec + c_cyc / f_loc_req)

            # Prefer feasible option; compare energy if both exist
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
                f_loc_cap = float(dev_cap[i])
                f_loc_alloc = min(f_loc_cap, max(f_min, best_local[0] * 1.02))
                best_local = (f_loc_alloc, t_exec + c_cyc / f_loc_alloc)

            if choose_local and best_local is not None:
                f_loc_req, _ = best_local
                dev_res[i] += f_loc_req
                local_decisions[t["id"]] = ("local", None, f_loc_req)
            elif best_edge is not None:
                _k, _j, cand, res = best_edge
                admitted_edge.append(cand)
                last_valid_res = res
            elif non_delay:
                local_decisions[t["id"]] = ("drop", None, 0.0)

        actions: list[dict] = []
        if admitted_edge and last_valid_res is None:
            last_valid_res = evaluate_fluid_batch(active, admitted_edge, params, now)
        srv_cap_arr = [float(s["f_avail_hz"]) for s in srvs]
        srv_used_est = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
        for g in admitted_edge:
            tid = g["id"]
            j = int(g["server"])
            srv_used_est[j] += float(last_valid_res["f_min"][tid])
        for g in admitted_edge:
            tid = g["id"]
            j = int(g["server"])
            f_req = float(last_valid_res["f_min"][tid])
            headroom = max(0.0, srv_cap_arr[j] - srv_used_est[j])
            bump = min(headroom, f_req * 0.02)
            srv_used_est[j] += bump
            actions.append({
                "task": tid,
                "mode": "edge",
                "server": j,
                "f_hz": f_req + bump,
            })
        for tid, (md, sv, f_hz) in local_decisions.items():
            actions.append({"task": tid, "mode": md, "server": sv, "f_hz": f_hz if md == "local" else None})

        return {"actions": actions}

    policy.__name__ = "SFA" + ("_NoDefer" if non_delay else "")
    return policy


def make_ibr(max_rounds: int = 5) -> Callable[[dict], dict]:
    """Iterative Batch Reoptimization (IBR-5):

    Multi-round interference re-pricing + exact `evaluate_fluid_batch` sequential projection +
    Gauss-Seidel single-task feasible relocation refinement.
    """
    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        now = float(obs["time"])
        t_exec = now + float(obs["delta_sig_s"])
        devs = obs["devices"]
        srvs = obs["servers"]
        m = len(srvs)
        n_sb = max(1, int(obs["links"]["n_subbands"]))
        h = np.asarray(obs["links"]["h_hat_dev_ms"], dtype=float)
        ortho = bool(obs["links"]["orthogonal"])
        active = obs["active_uplinks"]
        params = _build_params_from_obs(obs)

        pending = sorted(
            obs["pending"],
            key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
        )
        task_map = {t["id"]: t for t in pending}
        f_min = float(obs["f_min_hz"])
        dev_cap = np.asarray([float(d["f_loc_hz"]) for d in devs], dtype=float)
        dev_base_res = np.asarray([float(d["cpu_reserved_hz"]) for d in devs], dtype=float)

        im_act = _prebatch_interference(obs)
        prev_edge_devs: dict[int, set[int]] = {sb: set() for sb in range(n_sb)}

        best_score = (-1, -math.inf)
        best_actions: list[dict] = []

        for _rd in range(int(max_rounds)):
            # 1. Re-price options under previous round's co-subband transmitting device set
            interf_by_sb = im_act.copy()
            if not ortho:
                act_devs = {int(u["device"]) for u in active}
                for sb in range(n_sb):
                    for du in prev_edge_devs[sb]:
                        if du not in act_devs:
                            interf_by_sb[sb, :] += float(devs[du]["p_tx_w"]) * h[du, :]

            # 2. Rank options and build feasible batch via exact fluid check
            admitted_edge: list[dict] = []
            local_map: dict[str, float] = {}
            dev_res = dev_base_res.copy()
            last_res: Optional[dict[str, Any]] = None

            for t in pending:
                i = int(t["device"])
                sb = int(devs[i]["subband"]) % n_sb
                dl = float(t["deadline_s"])
                c_cyc = float(t["C_cycles"])

                ranked_srvs = []
                for j in range(m):
                    est = _estimate_option(obs, t, "edge", j, interf_by_sb[sb], t_exec)
                    score = est["energy_j"] if est is not None else 1e9 + float(h[i, j]) * (-1.0)
                    ranked_srvs.append((score, j))
                ranked_srvs.sort()

                placed = False
                # Check local option
                rem_loc = dl - t_exec
                loc_ok = False
                f_loc_req = math.inf
                if rem_loc > 1e-9:
                    f_loc_req = max(f_min, c_cyc / rem_loc)
                    if dev_res[i] + f_loc_req <= dev_cap[i] + 1.0:
                        loc_ok = True

                for _sc, j in ranked_srvs:
                    cand = _cand_grant(t, j, devs[i])
                    res = evaluate_fluid_batch(active, admitted_edge + [cand], params, now)
                    if res["feasible"]:
                        admitted_edge.append(cand)
                        last_res = res
                        placed = True
                        break

                if not placed and loc_ok:
                    dev_res[i] += f_loc_req
                    local_map[t["id"]] = f_loc_req

            # 3. Gauss-Seidel single-task feasible relocation / un-admitted admission pass
            for idx_e in range(len(admitted_edge)):
                cur_g = admitted_edge[idx_e]
                tid = cur_g["id"]
                t = task_map[tid]
                i = int(t["device"])
                if last_res is None:
                    last_res = evaluate_fluid_batch(active, admitted_edge, params, now)
                best_g = cur_g
                best_f_share = float(last_res["f_min"][tid])
                others = admitted_edge[:idx_e] + admitted_edge[idx_e + 1:]
                for j2 in range(m):
                    if j2 == int(cur_g["server"]):
                        continue
                    alt = _cand_grant(t, j2, devs[i])
                    res2 = evaluate_fluid_batch(active, others + [alt], params, now)
                    if res2["feasible"] and float(res2["f_min"][tid]) < best_f_share - 1e3:
                        best_f_share = float(res2["f_min"][tid])
                        best_g = alt
                        last_res = res2
                admitted_edge[idx_e] = best_g

            # Track incumbent
            if admitted_edge and last_res is None:
                last_res = evaluate_fluid_batch(active, admitted_edge, params, now)
            adm_total = len(admitted_edge) + len(local_map)
            tot_e = sum(1e-27 * (fv ** 2) * float(task_map[tid]["C_cycles"]) for tid, fv in local_map.items())
            if admitted_edge and last_res is not None:
                tot_e += sum(
                    float(devs[int(g["device"])]["p_tx_w"]) * max(float(last_res["t_tx_end"][g["id"]]) - now, 0.0)
                    for g in admitted_edge
                )
            cur_score = (adm_total, -tot_e)
            if cur_score > best_score:
                best_score = cur_score
                acts = []
                srv_cap_arr = [float(s["f_avail_hz"]) for s in srvs]
                srv_used_est = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
                for g in admitted_edge:
                    srv_used_est[int(g["server"])] += float(last_res["f_min"][g["id"]])
                for g in admitted_edge:
                    j = int(g["server"])
                    f_req = float(last_res["f_min"][g["id"]])
                    headroom = max(0.0, srv_cap_arr[j] - srv_used_est[j])
                    bump = min(headroom, f_req * 0.02)
                    srv_used_est[j] += bump
                    acts.append({
                        "task": g["id"],
                        "mode": "edge",
                        "server": j,
                        "f_hz": f_req + bump,
                    })
                for tid, fv in local_map.items():
                    i_d = int(task_map[tid]["device"])
                    f_loc_alloc = min(float(dev_cap[i_d]), fv * 1.02)
                    acts.append({"task": tid, "mode": "local", "server": None, "f_hz": float(f_loc_alloc)})
                best_actions = acts

            new_edge_devs: dict[int, set[int]] = {sb: set() for sb in range(n_sb)}
            for g in admitted_edge:
                di = int(g["device"])
                sb = int(devs[di]["subband"]) % n_sb
                new_edge_devs[sb].add(di)
            if new_edge_devs == prev_edge_devs and _rd > 0:
                break
            prev_edge_devs = new_edge_devs

        return {"actions": best_actions}

    policy.__name__ = f"IBR_{max_rounds}"
    return policy


# --------------------------------------------------------------------------- #
# 4 Exact Degenerate Limit Anchors vs. solve_single_tick_exact
# --------------------------------------------------------------------------- #

def verify_degenerate_anchors() -> dict[str, Any]:
    """Verify the 4 classical degenerate exact optimal solutions match `solve_single_tick_exact`."""
    results: dict[str, Any] = {}

    # Anchor 1: Single-server orthogonal regime (kappa_UL = 0, M = 1) -> Smallest-share-first exact
    n, m = 5, 1
    h1 = np.array([[1e-10]] * n, dtype=float)
    f_loc1 = np.full(n, 0.2e9, dtype=float)   # local impossible
    f_ms1 = np.full(m, 20.0e9, dtype=float)   # F_avail = 19 GHz
    ptx1 = np.ones(n, dtype=float)
    d_bits = 800_000.0
    t_specs = [
        {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.25 + 0.05 * i,
         "D_bits": d_bits, "C_cycles": (2.0 + 0.8 * i) * 1e9, "tau_dn_s": 0.001}
        for i in range(n)
    ]
    scen1 = make_micro_scenario(
        case_id="ANCHOR1_ORTHO_M1", n_devices=n, n_servers=m,
        h_dev_ms=h1, f_loc_hz=f_loc1, f_ms_max_hz=f_ms1, p_tx_w=ptx1, tasks_raw=t_specs,
    )
    from model_spec import EdgeUplinkEnv
    env1 = EdgeUplinkEnv(scen1, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="zero"))
    env1._reveal_arrivals(0.0)
    obs1 = env1.observe(0.0)
    oracle1 = solve_single_tick_exact(obs1, max_p=5)

    # Analytical smallest-f_min knapsack optimal for M=1 orthogonal
    f_reqs = []
    for t in obs1["pending"]:
        i = int(t["device"])
        r0 = shannon_rate(obs1["links"]["bandwidth_hz"], ptx1[i] * float(h1[i, 0]), obs1["links"]["noise_w"], 0.0)
        t_tx_e = 0.001 + float(t["D_bits"]) / r0
        rem = float(t["deadline_s"]) - t_tx_e - float(t["tau_dn_s"])
        f_reqs.append((max(1e6, float(t["C_cycles"]) / rem), t["id"]))
    f_reqs.sort()
    cap1 = float(obs1["servers"][0]["f_avail_hz"])
    acc = 0.0
    analyt1_cnt = 0
    for f_v, _tid in f_reqs:
        if acc + f_v <= cap1 + 1.0:
            acc += f_v
            analyt1_cnt += 1
    results["anchor1_single_srv_orthogonal"] = {
        "oracle_admitted": oracle1["best_admitted"],
        "analytical_admitted": analyt1_cnt,
        "match": oracle1["best_admitted"] == analyt1_cnt and abs(oracle1["best_cpu_hz"] - acc) <= 1e3,
    }

    # Anchor 2: Infinite-CPU SINR regime (F_avail -> infinity) -> Pure wireless deadline admission
    f_ms2 = np.full(1, 1e14, dtype=float)
    tasks2 = [
        {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.140,
         "D_bits": 800_000.0, "C_cycles": 5.0e8, "tau_dn_s": 0.001}
        for i in range(4)
    ]
    scen2 = make_micro_scenario(
        case_id="ANCHOR2_INF_CPU", n_devices=4, n_servers=1,
        h_dev_ms=np.full((4, 1), 1e-10), f_loc_hz=np.full(4, 0.2e9),
        f_ms_max_hz=f_ms2, p_tx_w=np.ones(4), tasks_raw=tasks2,
    )
    env2 = EdgeUplinkEnv(scen2, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero"))
    env2._reveal_arrivals(0.0)
    obs2 = env2.observe(0.0)
    oracle2 = solve_single_tick_exact(obs2, max_p=4)
    # Closed-form symmetric wireless rate for k concurrent transmitters: R_k = B log2(1 + 1/(k - 1 + N0/(p h)))
    analyt2_cnt = 0
    for k in range(1, 5):
        rk = shannon_rate(10e6, 1e-10, 1e-13, (k - 1) * 1e-10)
        if 0.001 + 800_000.0 / rk + 0.001 <= 0.140:
            analyt2_cnt = k
    results["anchor2_infinite_cpu_sinr"] = {
        "oracle_admitted": oracle2["best_admitted"],
        "analytical_admitted": analyt2_cnt,
        "match": oracle2["best_admitted"] == analyt2_cnt,
    }

    # Anchor 3: Infinite-bandwidth compute packing regime (BW -> infinity)
    tasks3 = [
        {"tid": f"{i}:0", "device": i, "k": 0, "arrival_s": 0.0, "deadline_s": 0.20,
         "D_bits": 800_000.0, "C_cycles": (0.8 + 0.5 * i) * 1e9, "tau_dn_s": 0.001}
        for i in range(5)
    ]
    scen3 = make_micro_scenario(
        case_id="ANCHOR3_INF_BW", n_devices=5, n_servers=1,
        h_dev_ms=np.full((5, 1), 1e-10), f_loc_hz=np.full(5, 0.1e9),
        f_ms_max_hz=np.full(1, 20.0e9), p_tx_w=np.ones(5), tasks_raw=tasks3,
        bandwidth_hz=1e12,
    )
    env3 = EdgeUplinkEnv(scen3, SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="zero"))
    env3._reveal_arrivals(0.0)
    obs3 = env3.observe(0.0)
    oracle3 = solve_single_tick_exact(obs3, max_p=5)
    f_reqs3 = sorted(float(t["C_cycles"]) / (float(t["deadline_s"]) - 0.001 - float(t["tau_dn_s"])) for t in obs3["pending"])
    cap3 = float(obs3["servers"][0]["f_avail_hz"])
    acc3, analyt3_cnt = 0.0, 0
    for fv in f_reqs3:
        if acc3 + fv <= cap3 + 1.0:
            acc3 += fv
            analyt3_cnt += 1
    results["anchor3_infinite_bw_packing"] = {
        "oracle_admitted": oracle3["best_admitted"],
        "analytical_admitted": analyt3_cnt,
        "oracle_cpu_hz": oracle3["best_cpu_hz"],
        "analytical_cpu_hz": acc3,
        "match": oracle3["best_admitted"] == analyt3_cnt and abs(oracle3["best_cpu_hz"] - acc3) <= 1e-4 * max(acc3, 1.0),
    }

    # Anchor 4: Single-task closed-form optimality (R = 1, U_0 = empty)
    scen4 = make_micro_scenario(
        case_id="ANCHOR4_SINGLE_TASK", n_devices=1, n_servers=1,
        h_dev_ms=np.array([[1e-10]]), f_loc_hz=np.array([1.0e9]),
        f_ms_max_hz=np.array([20.0e9]), p_tx_w=np.array([1.0]),
        tasks_raw=[{"tid": "0:0", "device": 0, "k": 0, "arrival_s": 0.0, "deadline_s": 0.50,
                    "D_bits": 800_000.0, "C_cycles": 1.0e9, "tau_dn_s": 0.001}],
    )
    env4 = EdgeUplinkEnv(scen4, SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))
    env4._reveal_arrivals(0.0)
    obs4 = env4.observe(0.0)
    oracle4 = solve_single_tick_exact(obs4, max_p=1)
    r0_4 = shannon_rate(10e6, 1e-10, 1e-13, 0.0)
    f_edge4 = 1.0e9 / (0.50 - 0.001 - 800_000.0 / r0_4 - 0.001)
    results["anchor4_single_task_closed_form"] = {
        "oracle_admitted": oracle4["best_admitted"],
        "oracle_cpu_hz": oracle4["best_cpu_hz"],
        "closed_form_cpu_hz": f_edge4,
        "match": oracle4["best_admitted"] == 1 and abs(oracle4["best_cpu_hz"] - f_edge4) <= 1e2,
    }

    results["all_passed"] = all(v["match"] for v in results.values() if isinstance(v, dict))
    return results
