"""Read-only Major Revision Engineering Audit Script.

Strictly read-only toward `stage2_model/`, `stage3_algorithm/`, `stage4_audit/`,
and `formal_experiments/`. Writes outputs strictly to
`supp_verification/revision_audit_results.json`.

Checks:
  1. Compiled Native x64 Fluid Verifier Benchmark (`fluid_bench_x64.exe`):
     - Numerical parity vs Python `_fast_fluid_verify` on feasible & full batches.
     - Single-tick wall-clock scalability for CLWB_x64 vs SFA_x64 across P in {50, 100, 200, 400}.
  2. Time-Budgeted Same-Engine Strong Baseline (`Budgeted-SFA+CG`, 45 ms cap, zero skip-ticks):
     - Main K=12 (10 seeds = 120 episodes) vs CLWB_Dual and Bambos_ALP+CG.
     - High-burst scalability P in {50, 100, 200} (3 wall episodes each).
  3. Decoupled Deterministic Clock Replay & Theorem 2 Lag-Envelope Coverage:
     - (3a) Decoupled `champion_core.time` on Main K=12 (`fixed=0ms`, back-to-back bitwise diff).
     - (3b) Theorem 2 runtime lag-envelope coverage rate + noiseless residual misses on Main K=12 (wall).
  4. Multi-Server Selection Comparison (`Best-SNR` vs `LoadAware`) on Main K=12 & OOD4_SPARSE_SRV.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
V3_DIR = THIS_DIR.parent
for sub in ("stage2_model", "stage3_algorithm", "stage4_audit", "formal_experiments"):
    p = str(V3_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines_stage2 as bs2  # noqa: E402
import champion_core  # noqa: E402
from benchmark_suite_s3 import build_ood_suite_k4  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from common_utils import (  # noqa: E402
    bootstrap_ci95,
    build_single_frozen_champion,
    paired_wilcoxon_and_cliffs,
    unpack_suite_item,
    verify_upstream_readonly,
)
from model_spec import EdgeUplinkEnv, SimConfig, build_scenario_v1, run_simulation  # noqa: E402
from run_s52_batch2_main import build_dual_track_cg_baselines, build_main_benchmark_k12  # noqa: E402

X64_EXE = THIS_DIR / "fluid_bench_x64.exe"
OUT_JSON = THIS_DIR / "revision_audit_results.json"


# --------------------------------------------------------------------------- #
# Check 1: Compiled Native x64 Fluid Benchmark
# --------------------------------------------------------------------------- #

def _export_snapshot(obs: dict, champ: champion_core.CLWBDualPolicy, path: Path) -> tuple[list[dict], list[dict]]:
    """Export single-tick snapshot + precomputed Python candidate lists matching CLWB & SFA."""
    now = float(obs["time"])
    tick_s = float(obs["tick_s"])
    delta_sig = float(obs["delta_sig_s"])
    t_exec = now + delta_sig
    t_act = t_exec
    devs = obs["devices"]
    srvs = obs["servers"]
    links = obs["links"]
    n_sb = max(1, int(links["n_subbands"]))
    h_mat = np.asarray(links["h_hat_dev_ms"], dtype=float)
    f_avail = np.asarray([float(s["f_avail_hz"]) for s in srvs], dtype=float)
    srv_res = np.asarray([float(s["cpu_reserved_hz"]) for s in srvs], dtype=float)
    srv_rem = np.maximum(f_avail - srv_res, float(obs["f_min_hz"]))

    # Match champion_core edge_pool and candidate ranking
    pending_sfa = sorted(
        obs["pending"],
        key=lambda t: (float(t["deadline_s"]), float(t["arrival_s"]), int(t["device"]), str(t["id"])),
    )
    edge_pool = [
        t for t in obs["pending"]
        if (not bool(devs[int(t["device"])]["radio_busy"]))
        and float(devs[int(t["device"])]["tx_queue_bits"]) <= 1e-9
    ]
    cands_clwb, _ = champ._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)

    # Exact EDF sort rank matching _fast_fluid_verify: (deadline_s, device, str(id))
    all_tasks = sorted(
        obs["pending"],
        key=lambda t: (float(t["deadline_s"]), int(t["device"]), str(t["id"])),
    )
    edf_rank = {str(t["id"]): idx for idx, t in enumerate(all_tasks)}

    lines: list[str] = []
    lines.append(
        f"{now:.17g} {t_act:.17g} {tick_s:.17g} {delta_sig:.17g} "
        f"{float(links['bandwidth_hz']):.17g} {float(links['noise_w']):.17g} "
        f"{1 if bool(links['orthogonal']) else 0} {n_sb} {float(obs['f_min_hz']):.17g} "
        f"{len(devs)} {len(srvs)} {len(obs['active_uplinks'])} {len(pending_sfa)} {len(cands_clwb)}"
    )
    for d in devs:
        lines.append(
            f"{float(d['p_tx_w']):.17g} {int(d['subband']) % n_sb} "
            f"{float(d['f_loc_hz']):.17g} {float(d['cpu_reserved_hz']):.17g}"
        )
    for s in srvs:
        lines.append(f"{float(s['f_avail_hz']):.17g} {float(s['cpu_reserved_hz']):.17g}")
    for i in range(len(devs)):
        lines.append(" ".join(f"{float(h_mat[i, j]):.17g}" for j in range(len(srvs))))
    for u in obs["active_uplinks"]:
        lines.append(
            f"{u['id']} {int(u['device'])} {int(u['server'])} {float(u['p_tx_w']):.17g} "
            f"{float(u['rem_bits']):.17g} {float(u['deadline_s']):.17g} "
            f"{float(u['C_cycles']):.17g} {float(u['f_hz']):.17g} {float(u['tau_dn_s']):.17g}"
        )
    for t in pending_sfa:
        tid = str(t["id"])
        lines.append(
            f"{tid} {int(t['device'])} 0 {float(t['D_bits']):.17g} "
            f"{float(t['deadline_s']):.17g} {float(t['C_cycles']):.17g} "
            f"{float(t['tau_dn_s']):.17g} 1 {edf_rank[tid]}"
        )
    for g in cands_clwb:
        tid = str(g["id"])
        lines.append(
            f"{tid} {int(g['device'])} {int(g['server'])} {float(g['D_bits']):.17g} "
            f"{float(g['deadline_s']):.17g} {float(g['C_cycles']):.17g} "
            f"{float(g['tau_dn_s']):.17g} {1 if g['can_defer'] else 0} {edf_rank[tid]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pending_sfa, cands_clwb


def _parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in line.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def run_check1_compiled_x64() -> dict[str, Any]:
    if not X64_EXE.exists():
        raise FileNotFoundError(f"Missing compiled native benchmark: {X64_EXE}")

    champ = build_single_frozen_champion()
    p_grid = [50, 100, 200, 400]
    n_seeds = 5
    max_t_diff_s = 0.0
    parity_checks = 0
    parity_matches = 0
    per_scale: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="clwb_x64_") as tmpdir:
        tmp_path = Path(tmpdir)
        for p_cnt in p_grid:
            clwb_ms_list, sfa_ms_list = [], []
            clwb_calls_list, sfa_calls_list = [], []
            clwb_adm_list, sfa_adm_list = [], []
            clwb_reps = 200 if p_cnt <= 200 else 100
            sfa_reps = 5 if p_cnt <= 200 else 3

            for s_idx in range(n_seeds):
                sc = build_scenario_v1(
                    case_id=f"X64_P{p_cnt}_S{s_idx}",
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

                snap_file = tmp_path / f"snap_p{p_cnt}_s{s_idx}.txt"
                _pending_sfa, cands_clwb = _export_snapshot(obs, champ, snap_file)

                # Parity check 1: Full batch
                py_res = champion_core._fast_fluid_verify(obs, cands_clwb, act_lag_s=0.0)
                cp = subprocess.run(
                    [str(X64_EXE), "parity", str(snap_file)],
                    capture_output=True, text=True, check=True,
                )
                toks = cp.stdout.strip().split()
                feas_x64 = toks[0] == "FEAS=1"
                parity_checks += 1
                if feas_x64 == bool(py_res["feasible"]):
                    parity_matches += 1
                for i, g in enumerate(cands_clwb):
                    t_x64 = float(toks[1 + i])
                    t_py = float(py_res["t_tx_end"][g["id"]])
                    if np.isfinite(t_x64) and np.isfinite(t_py):
                        max_t_diff_s = max(max_t_diff_s, abs(t_x64 - t_py))

                # Parity check 2: Feasible prefix subset (cands_clwb[:min(8, P)])
                sub_file = tmp_path / f"snap_sub_p{p_cnt}_s{s_idx}.txt"
                sub_cands = cands_clwb[:min(8, len(cands_clwb))]
                obs_sub = dict(obs)
                # Write snapshot with RankedClwb replaced by sub_cands
                snap_lines = snap_file.read_text(encoding="utf-8").splitlines()
                hdr = snap_lines[0].split()
                hdr[-1] = str(len(sub_cands))
                n_keep = len(snap_lines) - len(cands_clwb)
                sub_lines = [" ".join(hdr)] + snap_lines[1:n_keep] + snap_lines[n_keep:n_keep + len(sub_cands)]
                sub_file.write_text("\n".join(sub_lines) + "\n", encoding="utf-8")
                py_sub = champion_core._fast_fluid_verify(obs_sub, sub_cands, act_lag_s=0.0)
                cp_sub = subprocess.run(
                    [str(X64_EXE), "parity", str(sub_file)],
                    capture_output=True, text=True, check=True,
                )
                toks_sub = cp_sub.stdout.strip().split()
                parity_checks += 1
                if (toks_sub[0] == "FEAS=1") == bool(py_sub["feasible"]):
                    parity_matches += 1
                for i, g in enumerate(sub_cands):
                    t_x64 = float(toks_sub[1 + i])
                    t_py = float(py_sub["t_tx_end"][g["id"]])
                    if np.isfinite(t_x64) and np.isfinite(t_py):
                        max_t_diff_s = max(max_t_diff_s, abs(t_x64 - t_py))

                # Benchmark mode
                cp_b = subprocess.run(
                    [str(X64_EXE), "bench", str(snap_file), str(clwb_reps), str(sfa_reps)],
                    capture_output=True, text=True, check=True,
                )
                kv = _parse_kv_line(cp_b.stdout)
                clwb_ms_list.append(float(kv["CLWB_MS"]))
                clwb_calls_list.append(int(kv["CLWB_CALLS"]))
                clwb_adm_list.append(int(kv["CLWB_ADM"]))
                sfa_ms_list.append(float(kv["SFA_MS"]))
                sfa_calls_list.append(int(kv["SFA_CALLS"]))
                sfa_adm_list.append(int(kv["SFA_ADM"]))

            per_scale[str(p_cnt)] = {
                "n_trials": n_seeds,
                "clwb_x64_ms_mean": float(np.mean(clwb_ms_list)),
                "clwb_x64_ms_max": float(np.max(clwb_ms_list)),
                "clwb_calls_mean": float(np.mean(clwb_calls_list)),
                "clwb_admitted_mean": float(np.mean(clwb_adm_list)),
                "sfa_x64_ms_mean": float(np.mean(sfa_ms_list)),
                "sfa_x64_ms_max": float(np.max(sfa_ms_list)),
                "sfa_calls_mean": float(np.mean(sfa_calls_list)),
                "sfa_admitted_mean": float(np.mean(sfa_adm_list)),
                "speedup_x64_mean": float(np.mean(sfa_ms_list) / max(np.mean(clwb_ms_list), 1e-9)),
                "sfa_x64_exceeds_50ms": bool(np.mean(sfa_ms_list) > 50.0),
            }

    return {
        "parity_checks": parity_checks,
        "parity_matches": parity_matches,
        "max_t_tx_end_diff_s": float(max_t_diff_s),
        "per_scale": per_scale,
    }


# --------------------------------------------------------------------------- #
# Check 2: Budget-Truncated Same-Engine Strong Baseline (Budgeted-SFA+CG)
# --------------------------------------------------------------------------- #

def make_budgeted_sfa_fast(budget_ms: float = 42.0, stats_out: dict[str, list[float]] | None = None) -> Callable[[dict], dict]:
    """Same sequential _fast_fluid_verify baseline as SFA_FastVerify, stopping cleanly before budget_ms with elapsed-lag compensation."""
    budget_s = budget_ms / 1000.0

    def policy(obs: dict) -> dict:
        if not obs["pending"]:
            return {"actions": []}

        t_start = time.perf_counter()
        now = float(obs["time"])
        delta_sig_s = float(obs["delta_sig_s"])
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

        inspected = 0
        truncated = False
        for t in pending:
            if (time.perf_counter() - t_start) >= budget_s:
                truncated = True
                break
            inspected += 1
            i = int(t["device"])
            dl = float(t["deadline_s"])
            c_cyc = float(t["C_cycles"])

            best_edge = None
            for j in range(m):
                elapsed_s = time.perf_counter() - t_start
                if elapsed_s >= budget_s:
                    truncated = True
                    break
                cand = bs2._cand_grant(t, j, devs[i])
                lag_now = elapsed_s + 0.001
                res = champion_core._fast_fluid_verify(obs, admitted_edge + [cand], act_lag_s=lag_now)
                if res["feasible"]:
                    t_fin = float(res["t_tx_end"][t["id"]]) + c_cyc / float(res["f_min"][t["id"]]) + float(t["tau_dn_s"])
                    key = (float(res["f_min"][t["id"]]), t_fin, j)
                    if best_edge is None or key < best_edge[0]:
                        best_edge = (key, j, cand, res)

            if truncated:
                break

            t_exec_now = now + delta_sig_s + (time.perf_counter() - t_start)
            best_local = None
            rem_loc = dl - t_exec_now
            if rem_loc > 1e-9:
                f_loc_req = max(f_min, c_cyc / rem_loc)
                if dev_res[i] + f_loc_req <= dev_cap[i] + 1.0:
                    best_local = (f_loc_req, t_exec_now + c_cyc / f_loc_req)

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
                best_local = (f_loc_alloc, t_exec_now + c_cyc / f_loc_alloc)

            if choose_local and best_local is not None:
                f_loc_req, _ = best_local
                dev_res[i] += f_loc_req
                local_decisions[t["id"]] = ("local", None, f_loc_req)
            elif best_edge is not None:
                _k, _j, cand, res = best_edge
                admitted_edge.append(cand)
                last_valid_res = res

        if stats_out is not None:
            stats_out["inspected_frac"].append(float(inspected / max(1, len(pending))))
            stats_out["truncated"].append(1.0 if truncated else 0.0)

        while admitted_edge:
            lag_fin = (time.perf_counter() - t_start) + 0.001
            res_fin = champion_core._fast_fluid_verify(obs, admitted_edge, act_lag_s=lag_fin)
            if res_fin["feasible"]:
                last_valid_res = res_fin
                break
            admitted_edge.pop()

        actions: list[dict] = []
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

    policy.__name__ = "Budgeted_SFA_FastVerify"
    return policy


def run_check2_budgeted_sfa(k12: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    champ = build_single_frozen_champion()
    alp = baselines["Bambos_ALP+CG"]
    bud_stats: dict[str, list[float]] = {"inspected_frac": [], "truncated": []}
    bud_sfa = wrap_with_common_guard(make_budgeted_sfa_fast(42.0, bud_stats), "Budgeted_SFA+CG")

    # (2a) Main K=12 (10 seeds = 120 episodes)
    policies = [("CLWB_Dual", champ), ("Bambos_ALP+CG", alp), ("Budgeted_SFA+CG", bud_sfa)]
    wk_ont: dict[str, list[float]] = {nm: [] for nm, _ in policies}
    wk_miss: dict[str, list[float]] = {nm: [] for nm, _ in policies}
    max_p99: dict[str, float] = {nm: 0.0 for nm, _ in policies}
    skip_sum: dict[str, int] = {nm: 0 for nm, _ in policies}
    illegal_sum: dict[str, int] = {nm: 0 for nm, _ in policies}
    per_wk: dict[str, dict[str, Any]] = {}

    for item in k12:
        wid, cells = unpack_suite_item(item)
        per_wk[wid] = {}
        for nm, pol in policies:
            ots, gms, p99s, sks = [], [], [], []
            for sc, cfg in cells:
                r = run_simulation(sc, pol, cfg)
                ots.append(float(r["ontime_rate"]) * 100.0)
                gms.append(float(r["server_grant_miss_rate"]) * 100.0)
                p99s.append(float(r["decision_p99_ms"]))
                sks.append(int(r["skipped_ticks"]))
                illegal_sum[nm] += int(r["illegal_actions"])
            m_on = float(np.mean(ots))
            m_gm = float(np.mean(gms))
            wk_ont[nm].append(m_on)
            wk_miss[nm].append(m_gm)
            max_p99[nm] = max(max_p99[nm], float(np.max(p99s)))
            skip_sum[nm] += int(sum(sks))
            per_wk[wid][nm] = {"ontime_pct": m_on, "miss_pct": m_gm, "p99_max_ms": float(np.max(p99s)), "skipped_ticks": int(sum(sks))}

    clwb_arr = np.asarray(wk_ont["CLWB_Dual"], dtype=float)
    k12_summary: dict[str, Any] = {}
    for nm, _ in policies:
        arr = np.asarray(wk_ont[nm], dtype=float)
        mv, lo, hi = bootstrap_ci95(arr, n_boot=10000, seed=56101)
        row: dict[str, Any] = {
            "ontime_mean_pct": float(mv),
            "ontime_ci95_pct": [float(lo), float(hi)],
            "grant_miss_mean_pct": float(np.mean(wk_miss[nm])),
            "p99_ms_max": float(max_p99[nm]),
            "skipped_ticks_sum": int(skip_sum[nm]),
            "illegal_actions_sum": int(illegal_sum[nm]),
        }
        if nm != "CLWB_Dual":
            raw_p, c_delta = paired_wilcoxon_and_cliffs(clwb_arr, arr)
            diff = clwb_arr - arr
            row["delta_clwb_minus_base_pp"] = float(np.mean(diff))
            row["wilcoxon_p_raw"] = float(raw_p)
            row["cliffs_delta"] = float(c_delta)
            row["wins"] = int(np.sum(diff > 1e-12))
            row["ties"] = int(np.sum(np.abs(diff) <= 1e-12))
            row["losses"] = int(np.sum(diff < -1e-12))
        k12_summary[nm] = row

    k12_bud_insp_pct = float(np.mean(bud_stats["inspected_frac"]) * 100.0) if bud_stats["inspected_frac"] else 100.0
    k12_bud_trunc_pct = float(np.mean(bud_stats["truncated"]) * 100.0) if bud_stats["truncated"] else 0.0

    # (2b) High-burst scalability wall episodes at P in {50, 100, 200}
    burst_summary: dict[str, Any] = {}
    for p_cnt in [50, 100, 200]:
        ep_rows: dict[str, list[dict[str, Any]]] = {"CLWB_Dual": [], "Budgeted_SFA+CG": []}
        bud_stats["inspected_frac"].clear()
        bud_stats["truncated"].clear()
        for s_idx in range(3):
            sc = build_scenario_v1(
                case_id=f"BUD_P{p_cnt}_S{s_idx}",
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
            for nm, pol in [("CLWB_Dual", champ), ("Budgeted_SFA+CG", bud_sfa)]:
                r = run_simulation(sc, pol, cfg_wall)
                ep_rows[nm].append({
                    "ontime_pct": float(r["ontime_rate"]) * 100.0,
                    "miss_pct": float(r["server_grant_miss_rate"]) * 100.0,
                    "p99_ms": float(r["decision_p99_ms"]),
                    "skipped_ticks": int(r["skipped_ticks"]),
                })
        burst_summary[str(p_cnt)] = {
            "CLWB_Dual": {
                "ontime_mean_pct": float(np.mean([x["ontime_pct"] for x in ep_rows["CLWB_Dual"]])),
                "p99_ms_max": float(np.max([x["p99_ms"] for x in ep_rows["CLWB_Dual"]])),
                "skipped_ticks_sum": int(sum(x["skipped_ticks"] for x in ep_rows["CLWB_Dual"])),
            },
            "Budgeted_SFA+CG": {
                "ontime_mean_pct": float(np.mean([x["ontime_pct"] for x in ep_rows["Budgeted_SFA+CG"]])),
                "miss_mean_pct": float(np.mean([x["miss_pct"] for x in ep_rows["Budgeted_SFA+CG"]])),
                "p99_ms_max": float(np.max([x["p99_ms"] for x in ep_rows["Budgeted_SFA+CG"]])),
                "skipped_ticks_sum": int(sum(x["skipped_ticks"] for x in ep_rows["Budgeted_SFA+CG"])),
                "inspected_cand_pct": float(np.mean(bud_stats["inspected_frac"]) * 100.0),
                "truncated_tick_pct": float(np.mean(bud_stats["truncated"]) * 100.0),
            },
        }

    return {
        "k12_summary": k12_summary,
        "k12_budgeted_inspected_cand_pct": k12_bud_insp_pct,
        "k12_budgeted_truncated_tick_pct": k12_bud_trunc_pct,
        "k12_per_workload": per_wk,
        "high_burst_episodes": burst_summary,
    }


# --------------------------------------------------------------------------- #
# Check 3: Decoupled Deterministic Clock Replay & Theorem 2 Lag Envelope Audit
# --------------------------------------------------------------------------- #

def run_check3_clock_and_thm2(k12: list[dict[str, Any]]) -> dict[str, Any]:
    champ = build_single_frozen_champion()

    # (3a) Decouple champion_core internal timer under fixed=0ms
    orig_time_mod = champion_core.time
    champion_core.time = types.SimpleNamespace(perf_counter=lambda: 0.0)
    run_a: list[float] = []
    run_b: list[float] = []
    miss_a: list[float] = []
    wk_ont_a: dict[str, list[float]] = {unpack_suite_item(it)[0]: [] for it in k12}
    try:
        for item in k12:
            wid, cells = unpack_suite_item(item)
            for sc, cfg0 in cells:
                cfg_det = dataclasses.replace(cfg0, latency_mode="fixed", fixed_latency_s=0.0)
                ra = run_simulation(sc, champ, cfg_det)
                rb = run_simulation(sc, champ, cfg_det)
                oa = float(ra["ontime_rate"]) * 100.0
                ob = float(rb["ontime_rate"]) * 100.0
                run_a.append(oa)
                run_b.append(ob)
                miss_a.append(float(ra["server_grant_miss_rate"]) * 100.0)
                wk_ont_a[wid].append(oa)
    finally:
        champion_core.time = orig_time_mod

    max_diff_pp = float(np.max(np.abs(np.asarray(run_a) - np.asarray(run_b))))
    wk_means = [float(np.mean(vals)) for vals in wk_ont_a.values()]
    det_mean, det_lo, det_hi = bootstrap_ci95(wk_means, n_boot=10000, seed=56201)

    # (3b) Instrument Theorem 2 runtime lag-envelope coverage on Main K=12 in real 'wall' mode
    orig_verify = champion_core._fast_fluid_verify
    last_lag_pred: list[float] = [0.0]

    def _rec_verify(obs: dict, cands: list[dict], act_lag_s: float = 0.0) -> dict[str, Any]:
        last_lag_pred[0] = float(act_lag_s)
        return orig_verify(obs, cands, act_lag_s=act_lag_s)

    total_edge_ticks = 0
    viol_ticks = 0
    max_underest_ms = 0.0
    max_margin_ms = 0.0

    def _inst_policy(obs: dict) -> dict:
        nonlocal total_edge_ticks, viol_ticks, max_underest_ms, max_margin_ms
        t0 = time.perf_counter()
        last_lag_pred[0] = 0.0
        out = champ(obs)
        t1 = time.perf_counter()
        has_edge = any(a.get("mode") == "edge" for a in out.get("actions", []))
        if has_edge:
            total_edge_ticks += 1
            actual_lag = t1 - t0
            pred_lag = last_lag_pred[0]
            diff_ms = (actual_lag - pred_lag) * 1000.0
            if actual_lag > pred_lag:
                viol_ticks += 1
                max_underest_ms = max(max_underest_ms, diff_ms)
            else:
                max_margin_ms = max(max_margin_ms, -diff_ms)
        return out

    noiseless_grants = 0
    noiseless_misses = 0
    noiseless_overshoots_ms: list[float] = []
    noisy_grants = 0
    noisy_misses = 0

    champion_core._fast_fluid_verify = _rec_verify
    try:
        for item in k12:
            wid, cells = unpack_suite_item(item)
            is_noisy = "CSI_NOISE" in wid
            for sc, cfg in cells:
                env = EdgeUplinkEnv(sc, cfg)
                r = env.run(_inst_policy)
                if is_noisy:
                    noisy_grants += int(r["n_edge_granted"])
                    noisy_misses += int(r["n_edge_missed"])
                else:
                    noiseless_grants += int(r["n_edge_granted"])
                    noiseless_misses += int(r["n_edge_missed"])
                    for rt in env.tasks:
                        if rt.mode == "edge" and rt.state != "done":
                            if rt.t_tx_end_s is not None and rt.f_hz is not None:
                                ov_ms = ((float(rt.t_tx_end_s) + float(rt.C_cycles) / float(rt.f_hz) + float(rt.tau_dn_s)) - float(rt.deadline_s)) * 1000.0
                                noiseless_overshoots_ms.append(max(0.0, float(ov_ms)))
    finally:
        champion_core._fast_fluid_verify = orig_verify

    cov_pct = 100.0 * (1.0 - viol_ticks / max(1, total_edge_ticks))
    return {
        "decoupled_fixed0": {
            "bitwise_max_diff_pp": max_diff_pp,
            "is_bitwise_deterministic": bool(max_diff_pp == 0.0),
            "k12_ontime_mean_pct": float(det_mean),
            "k12_ontime_ci95_pct": [float(det_lo), float(det_hi)],
            "k12_grant_miss_mean_pct": float(np.mean(miss_a)),
        },
        "theorem2_runtime_audit_k12_wall": {
            "total_edge_ticks": int(total_edge_ticks),
            "envelope_violation_ticks": int(viol_ticks),
            "envelope_violation_pct": float(100.0 * viol_ticks / max(1, total_edge_ticks)),
            "envelope_coverage_pct": float(cov_pct),
            "max_lag_underest_ms": float(max_underest_ms),
            "noiseless_edge_grants": int(noiseless_grants),
            "noiseless_edge_misses": int(noiseless_misses),
            "noiseless_edge_miss_pct": float(100.0 * noiseless_misses / max(1, noiseless_grants)),
            "noiseless_max_overshoot_ms": float(max(noiseless_overshoots_ms)) if noiseless_overshoots_ms else 0.0,
            "noisy_edge_grants": int(noisy_grants),
            "noisy_edge_misses": int(noisy_misses),
        },
    }


# --------------------------------------------------------------------------- #
# Check 4: Multi-Server Best-SNR vs Load-Aware Server Selection
# --------------------------------------------------------------------------- #

class LoadAwareCLWBPolicy(champion_core.CLWBDualPolicy):
    """Variant choosing feasible servers by lowest residual CPU utilization first, then best SNR."""

    def _build_candidate_order(
        self,
        obs: dict,
        edge_pool: list[dict],
        t_exec: float,
        srv_rem: np.ndarray,
        f_avail: np.ndarray,
    ) -> tuple[list[dict], np.ndarray]:
        cands, f_iso_pm = super()._build_candidate_order(obs, edge_pool, t_exec, srv_rem, f_avail)
        h_mat = np.asarray(obs["links"]["h_hat_dev_ms"], dtype=float)
        updated: list[dict] = []
        for g in cands:
            g2 = dict(g)
            feas_js = list(g2["feas_js"])
            q_i = int(g2["q_i"])
            di = int(g2["device"])
            feas_js.sort(
                key=lambda j: (
                    float(f_iso_pm[q_i, j]) / max(float(srv_rem[j]), 1.0),
                    -float(h_mat[di, j]),
                    j,
                )
            )
            g2["feas_js"] = feas_js
            g2["server"] = int(feas_js[0])
            updated.append(g2)
        return updated, f_iso_pm


def run_check4_server_selection(k12: list[dict[str, Any]], ood4: list[dict[str, Any]]) -> dict[str, Any]:
    best_snr = build_single_frozen_champion()
    load_aware = LoadAwareCLWBPolicy(name="CLWB_LoadAware")

    def _eval_suite(suite: list[dict[str, Any]], label: str) -> dict[str, Any]:
        snr_wk, ld_wk = [], []
        snr_miss, ld_miss = [], []
        per_wk: dict[str, Any] = {}
        for item in suite:
            wid, cells = unpack_suite_item(item)
            o_s, m_s, o_l, m_l = [], [], [], []
            for sc, cfg in cells:
                rs = run_simulation(sc, best_snr, cfg)
                rl = run_simulation(sc, load_aware, cfg)
                o_s.append(float(rs["ontime_rate"]) * 100.0)
                m_s.append(float(rs["server_grant_miss_rate"]) * 100.0)
                o_l.append(float(rl["ontime_rate"]) * 100.0)
                m_l.append(float(rl["server_grant_miss_rate"]) * 100.0)
            ms, ml = float(np.mean(o_s)), float(np.mean(o_l))
            snr_wk.append(ms)
            ld_wk.append(ml)
            snr_miss.append(float(np.mean(m_s)))
            ld_miss.append(float(np.mean(m_l)))
            per_wk[wid] = {
                "best_snr_ontime_pct": ms,
                "load_aware_ontime_pct": ml,
                "delta_load_minus_snr_pp": float(ml - ms),
            }
        m_snr, lo_s, hi_s = bootstrap_ci95(snr_wk, n_boot=10000, seed=56401)
        m_ld, lo_l, hi_l = bootstrap_ci95(ld_wk, n_boot=10000, seed=56402)
        return {
            "suite": label,
            "best_snr_ontime_pct": float(m_snr),
            "best_snr_ci95_pct": [float(lo_s), float(hi_s)],
            "best_snr_miss_pct": float(np.mean(snr_miss)),
            "load_aware_ontime_pct": float(m_ld),
            "load_aware_ci95_pct": [float(lo_l), float(hi_l)],
            "load_aware_miss_pct": float(np.mean(ld_miss)),
            "delta_load_minus_snr_pp": float(m_ld - m_snr),
            "per_workload": per_wk,
        }

    k12_res = _eval_suite(k12, "Main_K12")
    ood4_sparse = [it for it in ood4 if it["wid"] == "OOD4_SPARSE_SRV"]
    ood_all_res = _eval_suite(ood4, "OOD_K4")
    return {
        "main_k12": k12_res,
        "ood_k4": ood_all_res,
        "ood4_sparse_srv": ood_all_res["per_workload"].get("OOD4_SPARSE_SRV", {}),
    }


def main() -> None:
    only_check2 = "--only-check2" in sys.argv
    t0 = time.perf_counter()
    print("[Audit] Building frozen benchmarks (K=12, OOD4, 10 seeds)...")
    k12 = build_main_benchmark_k12(n_seeds=10)
    baselines = dict(build_dual_track_cg_baselines())

    if only_check2 and OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print("[Audit] Re-running Check 2 ONLY with elapsed-lag compensation (42 ms cap)...")
        c2 = run_check2_budgeted_sfa(k12, baselines)
        print(f"  K12 Budgeted-SFA+CG={c2['k12_summary']['Budgeted_SFA+CG']['ontime_mean_pct']:.2f}% (miss {c2['k12_summary']['Budgeted_SFA+CG']['grant_miss_mean_pct']:.2f}%) vs CLWB={c2['k12_summary']['CLWB_Dual']['ontime_mean_pct']:.2f}%")
        upstream = verify_upstream_readonly()
        assert upstream["intact"], f"Upstream integrity violated: {upstream}"
        existing["check2_budgeted_sfa"] = c2
        existing["upstream_readonly"] = upstream
        OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Audit] Updated check2_budgeted_sfa in {time.perf_counter() - t0:.1f}s -> {OUT_JSON}")
        return

    ood4 = build_ood_suite_k4(n_seeds=10)

    print("[Audit] Check 1: Compiled native x64 benchmark & bit-parity...")
    c1 = run_check1_compiled_x64()
    print(f"  Parity: {c1['parity_matches']}/{c1['parity_checks']}, max_diff={c1['max_t_tx_end_diff_s']:.3e} s")

    print("[Audit] Check 2: Budget-truncated Budgeted-SFA+CG (42 ms cap, lag-aware)...")
    c2 = run_check2_budgeted_sfa(k12, baselines)
    print(f"  K12 Budgeted-SFA+CG={c2['k12_summary']['Budgeted_SFA+CG']['ontime_mean_pct']:.2f}% vs CLWB={c2['k12_summary']['CLWB_Dual']['ontime_mean_pct']:.2f}%")

    print("[Audit] Check 3: Decoupled clock deterministic parity + Theorem 2 envelope coverage...")
    c3 = run_check3_clock_and_thm2(k12)
    print(f"  Decoupled max_diff={c3['decoupled_fixed0']['bitwise_max_diff_pp']:.4f} pp, Thm2 cov={c3['theorem2_runtime_audit_k12_wall']['envelope_coverage_pct']:.2f}%")

    print("[Audit] Check 4: Best-SNR vs Load-Aware multi-server selection...")
    c4 = run_check4_server_selection(k12, ood4)
    print(f"  K12 delta={c4['main_k12']['delta_load_minus_snr_pp']:+.2f} pp, OOD4 delta={c4['ood4_sparse_srv']['delta_load_minus_snr_pp']:+.2f} pp")

    upstream = verify_upstream_readonly()
    assert upstream["intact"], f"Upstream integrity violated: {upstream}"

    out = {
        "elapsed_s": float(time.perf_counter() - t0),
        "upstream_readonly": upstream,
        "check1_compiled_x64": c1,
        "check2_budgeted_sfa": c2,
        "check3_clock_and_thm2": c3,
        "check4_server_selection": c4,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Audit] Done in {out['elapsed_s']:.1f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
