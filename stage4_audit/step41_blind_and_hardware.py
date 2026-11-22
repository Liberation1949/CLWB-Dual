"""Stage 4 Step 4.1: Blind Sliced Benchmark, Hardware Wall-Clock Audit & Sim-to-Real Verification.

Verifies:
  1. Stage-2 Model v1.0 cryptographic freeze integrity (FREEZE_MANIFEST_v1.0.sha256).
  2. Strict Sim-to-Real hardware proxy contract (`TrackedDict`/`TrackedList` runtime key & immutability audit
     against `hardware_mapping_table.md`, zero forbidden oracle keys).
  3. Unseen Blind Benchmark (`BLIND_K8`, seeds 99000+, 3 seeds/workload = 24 paired episodes) evaluated
     exclusively with domain-native metrics (no artificial composite score) against all 7 `+CommonGuard` baselines.
  4. Real wall-clock scaling (`P in {10, 25, 50, 75, 100}`) verifying `P99` decision latency << `tick_s`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

STAGE4_DIR = Path(__file__).resolve().parent
ROOT_V3 = STAGE4_DIR.parent
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"

for p in (str(STAGE3_DIR), str(STAGE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_suite_s3 import build_dual_track_cg_baselines
from champion_core import make_clwb_dual
from model_spec import SimConfig, build_scenario_v1, compute_dimensionless_numbers, run_simulation

ALLOWED_TOP_KEYS = {
    "time", "tick_s", "delta_sig_s", "rho_sys", "f_min_hz",
    "pending", "active_uplinks", "devices", "servers", "links",
}
ALLOWED_PENDING_KEYS = {
    "id", "device", "k", "arrival_s", "deadline_s",
    "D_bits", "B_out_bits", "C_cycles", "tau_dn_s",
}
ALLOWED_ACTIVE_KEYS = {
    "id", "device", "server", "p_tx_w", "rem_bits",
    "tx_start_s", "deadline_s", "C_cycles", "f_hz", "tau_dn_s",
}
ALLOWED_DEVICE_KEYS = {
    "device", "f_loc_hz", "cpu_used_hz", "cpu_reserved_hz",
    "p_tx_w", "radio_busy", "tx_queue_bits", "subband",
}
ALLOWED_SERVER_KEYS = {
    "server", "f_max_hz", "f_avail_hz", "cpu_used_hz",
    "cpu_reserved_hz", "wait_compute_count",
}
ALLOWED_LINK_KEYS = {
    "h_hat_dev_ms", "bandwidth_hz", "noise_w", "orthogonal", "n_subbands",
}
FORBIDDEN_KEYS = {"end_s", "t_tx_end_s", "cur_rate_bps", "h_dev_ms", "future_tasks", "seed"}


class AuditLog:
    def __init__(self) -> None:
        self.accessed: dict[str, set[str]] = {
            "top": set(), "pending": set(), "active_uplinks": set(),
            "devices": set(), "servers": set(), "links": set(),
        }
        self.forbidden_hits: list[str] = []
        self.illegal_hits: list[str] = []


class TrackedDict(dict):
    def __init__(self, data: dict, section: str, allowed: set[str], log: AuditLog) -> None:
        super().__init__(data)
        self._section = section
        self._allowed = allowed
        self._log = log

    def _record(self, key: Any) -> None:
        k = str(key)
        self._log.accessed[self._section].add(k)
        if k in FORBIDDEN_KEYS:
            self._log.forbidden_hits.append(f"{self._section}.{k}")
        if k not in self._allowed:
            self._log.illegal_hits.append(f"{self._section}.{k}")

    def __getitem__(self, key: Any) -> Any:
        self._record(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._record(key)
        return super().get(key, default)

    def __contains__(self, key: Any) -> bool:
        self._record(key)
        return super().__contains__(key)


def _wrap_obs(obs: dict, log: AuditLog) -> TrackedDict:
    wrapped_pending = [TrackedDict(p, "pending", ALLOWED_PENDING_KEYS, log) for p in obs["pending"]]
    wrapped_active = [TrackedDict(u, "active_uplinks", ALLOWED_ACTIVE_KEYS, log) for u in obs["active_uplinks"]]
    wrapped_devs = [TrackedDict(d, "devices", ALLOWED_DEVICE_KEYS, log) for d in obs["devices"]]
    wrapped_srvs = [TrackedDict(s, "servers", ALLOWED_SERVER_KEYS, log) for s in obs["servers"]]
    links = dict(obs["links"])
    h_ro = np.asarray(links["h_hat_dev_ms"], dtype=float).copy()
    h_ro.setflags(write=False)
    links["h_hat_dev_ms"] = h_ro
    wrapped_links = TrackedDict(links, "links", ALLOWED_LINK_KEYS, log)
    top_copy = {
        "time": obs["time"],
        "tick_s": obs["tick_s"],
        "delta_sig_s": obs["delta_sig_s"],
        "rho_sys": obs["rho_sys"],
        "f_min_hz": obs["f_min_hz"],
        "pending": wrapped_pending,
        "active_uplinks": wrapped_active,
        "devices": wrapped_devs,
        "servers": wrapped_srvs,
        "links": wrapped_links,
    }
    return TrackedDict(top_copy, "top", ALLOWED_TOP_KEYS, log)


def check_stage2_freeze() -> dict[str, Any]:
    manifest_path = STAGE2_DIR / "FREEZE_MANIFEST_v1.0.sha256"
    lines = [ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mismatches: list[str] = []
    checked = 0
    for ln in lines:
        parts = ln.split()
        expected_hash = parts[0]
        fname = parts[-1].lstrip("*")
        fpath = STAGE2_DIR / fname
        if not fpath.exists():
            mismatches.append(f"MISSING:{fname}")
            continue
        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        checked += 1
        if actual_hash != expected_hash:
            mismatches.append(f"HASH_MISMATCH:{fname}")
    return {"manifest_entries": len(lines), "checked": checked, "mismatches": mismatches, "intact": (len(mismatches) == 0 and checked == 8)}


def run_hardware_proxy_audit() -> dict[str, Any]:
    log = AuditLog()
    pol = make_clwb_dual()

    def audited_policy(obs: dict) -> dict:
        t_obs = _wrap_obs(obs, log)
        return pol(t_obs)

    sc_noisy = build_scenario_v1(
        case_id="HW_AUDIT_NOISY",
        family="blind_audit",
        seed=98501,
        n_devices=26,
        n_servers=6,
        tasks_per_device=5,
        offered_load=0.72,
        T_max_s=0.68,
        radius_m=240.0,
        input_kb_range=(40.0, 220.0),
        hetero_deadline_range=(0.55, 1.55),
    )
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, csi_sigma_db=1.5, csi_seed=98501)
    res = run_simulation(sc_noisy, audited_policy, cfg)
    return {
        "ontime_rate": float(res["ontime_rate"]),
        "illegal_actions": int(res["illegal_actions"]),
        "hard_invariant_checks": int(res["hard_invariant_checks"]),
        "forbidden_hits": log.forbidden_hits,
        "illegal_key_hits": log.illegal_hits,
        "accessed_keys": {k: sorted(v) for k, v in log.accessed.items()},
        "sim_to_real_clean": (len(log.forbidden_hits) == 0 and len(log.illegal_hits) == 0 and int(res["illegal_actions"]) == 0),
    }


def build_blind_suite_k8(n_seeds: int = 3) -> list[tuple[str, list[tuple[Any, SimConfig]]]]:
    specs = [
        ("BL01_MIXED_AR_CTRL", dict(n_devices=28, n_servers=6, tasks_per_device=5, offered_load=0.72, T_max_s=0.65, radius_m=260.0, input_kb_range=(35.0, 260.0), cycles_range=(5e7, 1.6e9), hetero_deadline_range=(0.50, 1.60)), {}),
        ("BL02_NEAR_FAR_HEAVY", dict(n_devices=30, n_servers=5, tasks_per_device=5, offered_load=0.78, T_max_s=0.70, radius_m=320.0, input_kb_range=(120.0, 240.0), p_tx_range_w=(0.4, 1.6), hetero_deadline_range=(0.55, 1.50)), {}),
        ("BL03_CSI_NOISY_2DB", dict(n_devices=24, n_servers=6, tasks_per_device=6, offered_load=0.70, T_max_s=0.75, radius_m=230.0, input_kb_range=(70.0, 140.0)), {"csi_sigma_db": 2.0}),
        ("BL04_SUBBAND_REUSE4", dict(n_devices=36, n_servers=6, tasks_per_device=5, offered_load=0.75, T_max_s=0.70, radius_m=240.0, n_subbands=4, input_kb_range=(80.0, 160.0)), {"n_subbands": 4}),
        ("BL05_ULTRA_BURST85", dict(n_devices=50, n_servers=7, tasks_per_device=2, offered_load=0.85, T_max_s=0.65, radius_m=250.0, window_s=0.18, input_kb_range=(60.0, 130.0)), {}),
        ("BL06_MICRO_EDGE_CELL", dict(n_devices=32, n_servers=4, tasks_per_device=5, offered_load=0.82, T_max_s=0.65, radius_m=120.0, area_m=1500.0, input_kb_range=(90.0, 180.0)), {}),
        ("BL07_TICK_FAST25MS", dict(n_devices=24, n_servers=6, tasks_per_device=6, offered_load=0.70, T_max_s=0.70, radius_m=220.0, input_kb_range=(70.0, 140.0)), {"tick_s": 0.025}),
        ("BL08_SPARSE_RELAXED", dict(n_devices=18, n_servers=6, tasks_per_device=5, offered_load=0.35, T_max_s=1.00, radius_m=180.0, input_kb_range=(10.0, 25.0)), {}),
    ]
    out: list[tuple[str, list[tuple[Any, SimConfig]]]] = []
    for idx, (wid, s_kw, c_kw) in enumerate(specs):
        base_seed = 99000 + idx * 100
        runs: list[tuple[Any, SimConfig]] = []
        for s_off in range(n_seeds):
            seed = base_seed + s_off
            sc = build_scenario_v1(case_id=f"{wid}_S{s_off}", family=wid, seed=seed, **s_kw)
            cfg_kw = {"tick_s": 0.05, "delta_sig_s": 0.001, "latency_mode": "wall", "csi_seed": seed}
            cfg_kw.update(c_kw)
            runs.append((sc, SimConfig(**cfg_kw)))
        out.append((wid, runs))
    return out


def _exact_sign_flip_p(diffs: np.ndarray) -> float:
    n = len(diffs)
    obs_mean = float(np.mean(diffs))
    if np.all(np.abs(diffs) <= 1e-12):
        return 1.0
    if n <= 14:
        extreme = 0
        total = 1 << n
        for bits in range(total):
            signs = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(n)])
            if abs(float(np.mean(diffs * signs))) >= abs(obs_mean) - 1e-12:
                extreme += 1
        return float(extreme / total)
    rng = np.random.default_rng(202611)
    n_perm = 20000
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.mean(signs * diffs[None, :], axis=1)
    return float((1 + np.sum(np.abs(perm_means) >= abs(obs_mean) - 1e-12)) / (n_perm + 1))


def _bootstrap_ci95(diffs: np.ndarray) -> list[float]:
    rng = np.random.default_rng(202611)
    idx = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    boots = np.mean(diffs[idx], axis=1)
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def _holm_adjust(raw_ps: list[float]) -> list[float]:
    m = len(raw_ps)
    order = sorted(range(m), key=lambda i: raw_ps[i])
    adj = [0.0] * m
    cur = 0.0
    for rank, orig in enumerate(order):
        val = (m - rank) * raw_ps[orig]
        cur = max(cur, val)
        adj[orig] = min(1.0, float(cur))
    return adj


def evaluate_blind_k8() -> dict[str, Any]:
    suite = build_blind_suite_k8(n_seeds=3)
    champ = make_clwb_dual()
    cg_baselines = dict(build_dual_track_cg_baselines())
    policies: list[tuple[str, Any]] = [("CLWB_Dual", champ)] + list(cg_baselines.items())

    per_workload: dict[str, dict[str, Any]] = {}
    ep_vectors: dict[str, list[float]] = {pname: [] for pname, _ in policies}
    wk_vectors: dict[str, list[float]] = {pname: [] for pname, _ in policies}

    t0 = time.perf_counter()
    for wid, runs in suite:
        dm = compute_dimensionless_numbers(runs[0][0], runs[0][1])
        wk_entry: dict[str, Any] = {"kappa_ul": float(dm["kappa_ul"]), "rho_cmp": float(dm["rho_cmp"]), "policies": {}}
        for pname, pfn in policies:
            ep_res = [run_simulation(sc, pfn, cfg) for sc, cfg in runs]
            onts = [float(r["ontime_rate"]) for r in ep_res]
            ep_vectors[pname].extend(onts)
            wk_vectors[pname].append(float(np.mean(onts)))
            wk_entry["policies"][pname] = {
                "ontime_mean": float(np.mean(onts)),
                "ontime_std": float(np.std(onts)),
                "miss_mean": float(np.mean([r["server_grant_miss_rate"] for r in ep_res])),
                "wasted_mean": float(np.mean([r["wasted_srv_cycle_ratio"] for r in ep_res])),
                "p50_ms_mean": float(np.mean([r["decision_p50_ms"] for r in ep_res])),
                "p99_ms_max": float(np.max([r["decision_p99_ms"] for r in ep_res])),
                "skipped_ticks": int(sum(r["skipped_ticks"] for r in ep_res)),
                "illegal_actions": int(sum(r["illegal_actions"] for r in ep_res)),
                "energy_tx_j": float(np.mean([r["energy_tx_j"] for r in ep_res])),
                "energy_cmp_j": float(np.mean([r["energy_cmp_j"] for r in ep_res])),
            }
        per_workload[wid] = wk_entry
        print(
            f"[BLIND] {wid:20s} CLWB={wk_entry['policies']['CLWB_Dual']['ontime_mean']*100:5.2f}% "
            f"Bambos+CG={wk_entry['policies']['Bambos_ALP+CG']['ontime_mean']*100:5.2f}% "
            f"Chen+CG={wk_entry['policies']['Chen_Game+CG']['ontime_mean']*100:5.2f}% "
            f"SFA+CG={wk_entry['policies']['SFA+CG']['ontime_mean']*100:5.2f}%",
            flush=True,
        )

    champ_wk = np.asarray(wk_vectors["CLWB_Dual"], dtype=float)
    champ_ep = np.asarray(ep_vectors["CLWB_Dual"], dtype=float)
    base_names = [pname for pname, _ in cg_baselines.items()]

    raw_ep_p = [_exact_sign_flip_p(champ_ep - np.asarray(ep_vectors[b], dtype=float)) for b in base_names]
    raw_wk_p = [_exact_sign_flip_p(champ_wk - np.asarray(wk_vectors[b], dtype=float)) for b in base_names]
    holm_ep_p = _holm_adjust(raw_ep_p)
    holm_wk_p = _holm_adjust(raw_wk_p)

    pairwise: dict[str, Any] = {}
    for idx, b in enumerate(base_names):
        d_wk = champ_wk - np.asarray(wk_vectors[b], dtype=float)
        d_ep = champ_ep - np.asarray(ep_vectors[b], dtype=float)
        pairwise[b] = {
            "champ_mean_ontime": float(np.mean(champ_wk)),
            "base_mean_ontime": float(np.mean(wk_vectors[b])),
            "mean_diff_pp": float(np.mean(d_wk) * 100.0),
            "wk_wins": int(np.sum(d_wk > 1e-6)),
            "wk_ties": int(np.sum(np.abs(d_wk) <= 1e-6)),
            "wk_losses": int(np.sum(d_wk < -1e-6)),
            "ep_wins": int(np.sum(d_ep > 1e-6)),
            "ep_ties": int(np.sum(np.abs(d_ep) <= 1e-6)),
            "ep_losses": int(np.sum(d_ep < -1e-6)),
            "wk_ci95_pp": [x * 100.0 for x in _bootstrap_ci95(d_wk * 1.0)],
            "ep_ci95_pp": [x * 100.0 for x in _bootstrap_ci95(d_ep * 1.0)],
            "raw_p_wk": float(raw_wk_p[idx]),
            "holm_p_wk": float(holm_wk_p[idx]),
            "raw_p_ep": float(raw_ep_p[idx]),
            "holm_p_ep": float(holm_ep_p[idx]),
        }

    overall = {}
    for pname, _ in policies:
        wk_stats = [per_workload[w]["policies"][pname] for w in per_workload]
        overall[pname] = {
            "mean_ontime": float(np.mean(wk_vectors[pname])),
            "min_ontime": float(np.min(wk_vectors[pname])),
            "mean_miss": float(np.mean([s["miss_mean"] for s in wk_stats])),
            "mean_wasted": float(np.mean([s["wasted_mean"] for s in wk_stats])),
            "max_p99_ms": float(max(s["p99_ms_max"] for s in wk_stats)),
            "total_skipped_ticks": int(sum(s["skipped_ticks"] for s in wk_stats)),
            "total_illegal_actions": int(sum(s["illegal_actions"] for s in wk_stats)),
        }

    return {
        "n_workloads": len(suite),
        "n_episodes_per_policy": len(ep_vectors["CLWB_Dual"]),
        "eval_wall_s": float(time.perf_counter() - t0),
        "overall": overall,
        "pairwise_vs_cg": pairwise,
        "per_workload": per_workload,
    }


def evaluate_hardware_burst_scaling() -> dict[str, Any]:
    cg = dict(build_dual_track_cg_baselines())
    targets = {
        "CLWB_Dual": make_clwb_dual(),
        "Bambos_ALP+CG": cg["Bambos_ALP+CG"],
        "SFA+CG": cg["SFA+CG"],
    }
    p_vals = [10, 25, 50, 75, 100]
    rows: dict[str, Any] = {}
    for p_cnt in p_vals:
        entry: dict[str, Any] = {}
        for pname, pfn in targets.items():
            p50s, p99s, onts, skips = [], [], [], 0
            for rep in range(3):
                seed = 97000 + p_cnt * 10 + rep
                sc = build_scenario_v1(
                    case_id=f"BURST_P{p_cnt}_{rep}",
                    family="hw_burst",
                    seed=seed,
                    n_devices=p_cnt,
                    n_servers=6,
                    tasks_per_device=1,
                    offered_load=0.70,
                    T_max_s=0.75,
                    radius_m=220.0,
                    window_s=0.015,
                    input_kb_range=(60.0, 120.0),
                )
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
                res = run_simulation(sc, pfn, cfg)
                p50s.append(float(res["decision_p50_ms"]))
                p99s.append(float(res["decision_p99_ms"]))
                onts.append(float(res["ontime_rate"]))
                skips += int(res["skipped_ticks"])
            entry[pname] = {
                "p50_ms": float(np.mean(p50s)),
                "p99_ms_max": float(np.max(p99s)),
                "ontime_mean": float(np.mean(onts)),
                "skipped_ticks": int(skips),
            }
        rows[f"P={p_cnt}"] = entry
    return rows


def main() -> None:
    freeze = check_stage2_freeze()
    print("[Step 4.1] Stage-2 Freeze:", freeze)
    hw_audit = run_hardware_proxy_audit()
    print("[Step 4.1] Sim-to-Real Clean:", hw_audit["sim_to_real_clean"], "forbidden=", hw_audit["forbidden_hits"])
    blind = evaluate_blind_k8()
    burst = evaluate_hardware_burst_scaling()

    report = {
        "stage2_freeze": freeze,
        "sim_to_real_audit": hw_audit,
        "blind_k8": blind,
        "hardware_burst_scaling": burst,
    }
    out_path = STAGE4_DIR / "step41_blind_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Step 4.1] Wrote {out_path}")


if __name__ == "__main__":
    main()
