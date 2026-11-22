"""Automated 8-Gate Stage-2 Acceptance & Verification Suite (Model v1.0).

Runs all Stage-2 verification checks without modifying any frozen workspace assets:
  Gate 1: Zero new-algorithm smuggling scan across `stage2_model/*.py` and `*.md`.
  Gate 2: Real-hardware fidelity, zero future completion leaks in `observe()`,
          and hard physical invariant checks under wall-clock actuation lag.
  Gate 3: Dimensionless parameter sanity screening (no <1e-2 pseudo-bottleneck).
  Gate 4: 100 random micro-instance tests validating `solve_offline_exact >= baseline`.
  Gate 5: Phenomenon recovery (pre-batch server-grant miss collapse vs orthogonal,
          and burst-P superlinear latency explosion with real controller tick-skipping).
  Gate 6: 200/200 hereditary anti-monotonicity trials + 4/4 degenerate analytical anchors.
  Gate 7: Hand-verifiable N <= 5 counterexamples (L1, L2, L3) + 2D dimensionless phase plane.
  Gate 8: Model v1.0 SHA-256 freeze manifest generation.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines_stage2 import (
    make_bambos_alp,
    make_chen_game,
    make_dgolb,
    make_edf_prebatch,
    make_ibr,
    make_llf_prebatch,
    make_sfa,
    verify_degenerate_anchors,
)
from exact_oracle import (
    build_counterexample_L1,
    build_counterexample_L2,
    build_counterexample_L3,
    solve_offline_exact,
    solve_single_tick_exact,
    verify_hereditary_anti_monotonicity,
)
from model_spec import (
    EdgeUplinkEnv,
    SimConfig,
    build_calibrated_suite,
    build_scenario_v1,
    compute_dimensionless_numbers,
    make_micro_scenario,
    run_simulation,
)

STAGE2_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Gate 1: Zero Algorithm Smuggling Audit
# --------------------------------------------------------------------------- #

def check_gate1_no_smuggling() -> dict[str, Any]:
    # Construct banned tokens dynamically so this verifier does not match itself.
    banned_patterns = [
        ("ri" + "cos"),
        ("two" + r"[\s_-]*" + "pass"),
        ("monotone" + r"[\s_-]*" + "withdrawal"),
        ("proposed" + r"[\s_-]+" + "algorithm"),
        ("our" + r"[\s_-]+" + "algorithm"),
        ("our" + r"[\s_-]+" + "method"),
    ]
    combined = re.compile("|".join(banned_patterns), re.IGNORECASE)
    hits: list[dict[str, Any]] = []
    scanned: list[str] = []

    for ext in ("*.py", "*.md"):
        for fp in sorted(STAGE2_DIR.glob(ext)):
            scanned.append(fp.name)
            txt = fp.read_text(encoding="utf-8", errors="replace")
            for ln_no, line in enumerate(txt.splitlines(), 1):
                m = combined.search(line)
                if m:
                    hits.append({"file": fp.name, "line": ln_no, "match": m.group(0)})

    return {
        "passed": len(hits) == 0,
        "scanned_files": scanned,
        "hits": hits,
    }


# --------------------------------------------------------------------------- #
# Gate 2 & Gate 3: Hardware Fidelity, Causal Isolation & Dimensionless Sanity
# --------------------------------------------------------------------------- #

def check_gate2_and_gate3() -> tuple[dict[str, Any], dict[str, Any]]:
    suite = build_calibrated_suite(seed=2026)

    # Verify causal isolation on observe(): no future `end_s` and zero unrevealed tasks
    scen0, cfg0 = suite[0]
    env0 = EdgeUplinkEnv(scen0, cfg0)
    env0._reveal_arrivals(0.25)
    # Put one task into active uplink manually for observation check
    first_pend = next(t for t in env0.tasks if t.state == "pending")
    first_pend.state = "tx_up"
    first_pend.server = 0
    first_pend.f_hz = 5e9
    first_pend.t_tx_start_s = 0.25
    env0.active_uplinks.append(first_pend)
    obs0 = env0.observe(0.25)
    active_keys = set(obs0["active_uplinks"][0].keys())
    no_end_s = ("end_s" not in active_keys) and ("t_tx_end_s" not in active_keys)
    unrev_ids = {t.tid for t in env0.tasks if t.state == "unrevealed"}
    pend_ids = {t["id"] for t in obs0["pending"]}
    no_future_tasks = len(unrev_ids & pend_ids) == 0

    dim_report = {}
    hard_checks_total = 0
    max_bit_err_global = 0.0
    illegal_total = 0
    wall_nonzero_ok = True

    policies = [
        ("EDF_PreBatch", make_edf_prebatch()),
        ("LLF_PreBatch", make_llf_prebatch()),
        ("Bambos_ALP", make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", make_chen_game(max_rounds=8)),
        ("DGOLB", make_dgolb()),
        ("SFA", make_sfa()),
        ("IBR_5", make_ibr(max_rounds=5)),
    ]

    for scen, cfg in suite:
        dims = compute_dimensionless_numbers(scen, cfg)
        dim_report[scen.case_id] = dims
        for pname, pol in policies:
            res = run_simulation(scen, pol, cfg)
            hard_checks_total += res["hard_invariant_checks"]
            max_bit_err_global = max(max_bit_err_global, res["max_bit_rel_err"])
            illegal_total += res["illegal_actions"]
            if res["policy_calls"] > 0 and res["decision_p50_ms"] <= 0.0:
                wall_nonzero_ok = False

    # Confirm kappa_ul is non-trivial (>= 0.15) across calibrated scenarios
    kappas = [v["kappa_ul"] for v in dim_report.values()]
    rhos = [v["rho_cmp"] for v in dim_report.values()]
    gammas = [v["gamma_tick"] for v in dim_report.values()]

    gate2 = {
        "passed": bool(no_end_s and no_future_tasks and illegal_total == 0 and max_bit_err_global <= 1e-8 and wall_nonzero_ok),
        "no_end_s_leak": bool(no_end_s),
        "no_future_task_leak": bool(no_future_tasks),
        "hard_invariant_checks_total": int(hard_checks_total),
        "max_bit_rel_err": float(max_bit_err_global),
        "illegal_actions_total": int(illegal_total),
        "wall_clock_nonzero": bool(wall_nonzero_ok),
    }
    gate3 = {
        "passed": bool(min(kappas) >= 0.15 and min(rhos) >= 0.30 and 0.04 <= min(gammas) and max(gammas) <= 0.12),
        "kappa_ul_range": [float(min(kappas)), float(max(kappas))],
        "rho_cmp_range": [float(min(rhos)), float(max(rhos))],
        "gamma_tick_range": [float(min(gammas)), float(max(gammas))],
        "by_scenario": dim_report,
    }
    return gate2, gate3


# --------------------------------------------------------------------------- #
# Gate 4: 100 Micro-Instances Exact Offline Oracle vs All Causal Baselines
# --------------------------------------------------------------------------- #

def check_gate4_micro_oracle(n_instances: int = 100, seed: int = 777) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, rho_sys=0.95, latency_mode="zero")
    baselines = [
        ("EDF_PreBatch", make_edf_prebatch()),
        ("LLF_PreBatch", make_llf_prebatch()),
        ("Bambos_ALP", make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", make_chen_game(max_rounds=6)),
        ("DGOLB", make_dgolb()),
        ("SFA", make_sfa()),
        ("IBR_5", make_ibr(max_rounds=5)),
    ]

    inversions = 0
    nonempty_feasible = 0
    oracle_rates = []
    baseline_rates: dict[str, list[float]] = {name: [] for name, _p in baselines}
    t_start = time.perf_counter()

    for idx in range(int(n_instances)):
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
                "tid": f"{d}:{k}",
                "device": d,
                "k": k,
                "arrival_s": arr,
                "deadline_s": arr + span,
                "D_bits": float(rng.uniform(4e5, 9.5e5)),
                "C_cycles": float(rng.uniform(2e8, 8e8)),
                "tau_dn_s": 0.001,
            })

        scen = make_micro_scenario(
            case_id=f"M_{idx:03d}",
            n_devices=n,
            n_servers=m,
            h_dev_ms=h,
            f_loc_hz=f_loc,
            f_ms_max_hz=f_ms,
            p_tx_w=ptx,
            tasks_raw=tasks,
        )
        exact = solve_offline_exact(scen, cfg)
        if exact["best_ontime"] >= 1:
            nonempty_feasible += 1
        oracle_rates.append(exact["best_rate"])

        for bname, pol in baselines:
            res = run_simulation(scen, pol, cfg)
            baseline_rates[bname].append(res["ontime_rate"])
            if res["n_ontime"] > exact["best_ontime"]:
                inversions += 1

    elapsed_s = time.perf_counter() - t_start
    mean_oracle = float(np.mean(oracle_rates))
    mean_baselines = {k: float(np.mean(v)) for k, v in baseline_rates.items()}

    return {
        "passed": bool(inversions == 0 and nonempty_feasible == n_instances),
        "n_instances": int(n_instances),
        "nonempty_feasible": int(nonempty_feasible),
        "inversions": int(inversions),
        "elapsed_s": float(elapsed_s),
        "mean_oracle_rate": mean_oracle,
        "mean_baseline_rates": mean_baselines,
    }


# --------------------------------------------------------------------------- #
# Gate 5: Phenomenon Recovery (Pre-Batch Blindness + Burst Control Lag)
# --------------------------------------------------------------------------- #

def check_gate5_phenomena() -> dict[str, Any]:
    suite = build_calibrated_suite(seed=2026)
    policies = [
        ("EDF_PreBatch", make_edf_prebatch()),
        ("LLF_PreBatch", make_llf_prebatch()),
        ("Bambos_ALP", make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", make_chen_game(max_rounds=8)),
        ("DGOLB", make_dgolb()),
        ("SFA", make_sfa()),
        ("IBR_5", make_ibr(max_rounds=5)),
    ]

    shared_summary: dict[str, dict[str, float]] = {}
    orth_summary: dict[str, dict[str, float]] = {}

    for pname, pol in policies:
        s_rows = []
        o_rows = []
        for scen, cfg in suite:
            s_rows.append(run_simulation(scen, pol, cfg))
            o_rows.append(run_simulation(scen, pol, replace(cfg, orthogonal=True)))
        shared_summary[pname] = {
            "ontime_rate": float(np.mean([r["ontime_rate"] for r in s_rows])),
            "server_grant_miss_rate": float(np.mean([r["server_grant_miss_rate"] for r in s_rows])),
            "wasted_srv_cycle_ratio": float(np.mean([r["wasted_srv_cycle_ratio"] for r in s_rows])),
            "decision_p99_ms": float(np.max([r["decision_p99_ms"] for r in s_rows])),
            "skipped_ticks": int(sum(r["skipped_ticks"] for r in s_rows)),
        }
        orth_summary[pname] = {
            "ontime_rate": float(np.mean([r["ontime_rate"] for r in o_rows])),
            "server_grant_miss_rate": float(np.mean([r["server_grant_miss_rate"] for r in o_rows])),
            "wasted_srv_cycle_ratio": float(np.mean([r["wasted_srv_cycle_ratio"] for r in o_rows])),
        }

    # Burst sweep P in {10, 25, 45, 70} measuring real wall-clock latency L(t_k) and skipped ticks
    burst_policies = [
        ("EDF_PreBatch", make_edf_prebatch()),
        ("DGOLB", make_dgolb()),
        ("Bambos_ALP", make_bambos_alp(sinr_thresh=0.5)),
        ("Chen_Game", make_chen_game(max_rounds=8)),
        ("SFA", make_sfa()),
        ("IBR_5", make_ibr(max_rounds=5)),
    ]
    burst_results: dict[int, dict[str, dict[str, float]]] = {}
    for p_cnt in (10, 25, 45, 70):
        # All P tasks arrive in tight burst within [0, 0.001s] across P devices and M=6 servers
        scen_b = build_scenario_v1(
            case_id=f"BURST_P{p_cnt}",
            family="burst",
            seed=9000 + p_cnt,
            n_devices=p_cnt,
            n_servers=6,
            tasks_per_device=1,
            offered_load=0.80,
            T_max_s=0.65,
            radius_m=220.0,
            window_s=0.005,
        )
        cfg_b = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
        burst_results[p_cnt] = {}
        for pname, pol in burst_policies:
            res = run_simulation(scen_b, pol, cfg_b)
            burst_results[p_cnt][pname] = {
                "ontime_rate": float(res["ontime_rate"]),
                "decision_p99_ms": float(res["decision_p99_ms"]),
                "skipped_ticks": int(res["skipped_ticks"]),
                "server_grant_miss_rate": float(res["server_grant_miss_rate"]),
            }

    # Smoking guns criteria:
    # 1. Pre-batch baselines (DGOLB & EDF_PreBatch) suffer >= 12% server grant miss rate in shared channel
    #    which collapses dramatically in orthogonal channel;
    # 2. Both SFA and IBR_5 exceed Delta_t = 50 ms at P = 70 with skipped_ticks > 0!
    dgolb_shared_miss = shared_summary["DGOLB"]["server_grant_miss_rate"]
    dgolb_orth_miss = orth_summary["DGOLB"]["server_grant_miss_rate"]
    edf_shared_miss = shared_summary["EDF_PreBatch"]["server_grant_miss_rate"]
    ibr_p70_ms = burst_results[70]["IBR_5"]["decision_p99_ms"]
    sfa_p70_ms = burst_results[70]["SFA"]["decision_p99_ms"]
    ibr_skipped = burst_results[70]["IBR_5"]["skipped_ticks"]
    sfa_skipped = burst_results[70]["SFA"]["skipped_ticks"]

    passed = (
        dgolb_shared_miss >= 0.12
        and edf_shared_miss >= 0.12
        and dgolb_orth_miss < 0.05
        and ibr_p70_ms > 50.0
        and sfa_p70_ms > 50.0
        and ibr_skipped > 0
        and sfa_skipped > 0
    )

    return {
        "passed": bool(passed),
        "shared_calibrated_summary": shared_summary,
        "orthogonal_calibrated_summary": orth_summary,
        "burst_sweep": {str(k): v for k, v in burst_results.items()},
    }


# --------------------------------------------------------------------------- #
# Gate 6 & Gate 7: Hereditary, Anchors, Counterexamples L1-L3, 2D Phase Space
# --------------------------------------------------------------------------- #

def check_gate6_and_gate7() -> tuple[dict[str, Any], dict[str, Any]]:
    # Gate 6: Hereditary anti-monotonicity (200 trials) + 4 degenerate anchors
    hered = verify_hereditary_anti_monotonicity(n_trials=200, seed=42)
    anchors = verify_degenerate_anchors()
    gate6 = {
        "passed": bool(hered["all_passed"] and anchors["all_passed"]),
        "hereditary_anti_monotonicity": hered,
        "degenerate_anchors": anchors,
    }

    # Gate 7a: Counterexamples L1, L2, L3 exact checks
    # L1: Pre-batch class
    scen_l1, cfg_l1 = build_counterexample_L1()
    ex_l1 = solve_offline_exact(scen_l1, cfg_l1)
    edf_l1 = run_simulation(scen_l1, make_edf_prebatch(), cfg_l1)
    llf_l1 = run_simulation(scen_l1, make_llf_prebatch(), cfg_l1)
    dgolb_l1 = run_simulation(scen_l1, make_dgolb(), cfg_l1)
    l1_ok = (ex_l1["best_ontime"] >= 2 and edf_l1["n_ontime"] == 0 and llf_l1["n_ontime"] == 0 and dgolb_l1["n_ontime"] == 0)

    # L2: Non-delay class
    scen_l2, cfg_l2 = build_counterexample_L2()
    ex_l2_delay = solve_offline_exact(scen_l2, cfg_l2, non_delay=False)
    ex_l2_nondelay = solve_offline_exact(scen_l2, cfg_l2, non_delay=True)
    edf_l2_nd = run_simulation(scen_l2, make_edf_prebatch(non_delay=True), cfg_l2)
    sfa_l2_nd = run_simulation(scen_l2, make_sfa(non_delay=True), cfg_l2)
    l2_ok = (ex_l2_delay["best_ontime"] == 3 and ex_l2_nondelay["best_ontime"] <= 2 and sfa_l2_nd["n_ontime"] <= 2)

    # L3: Myopic EDF class
    scen_l3, cfg_l3 = build_counterexample_L3()
    ex_l3 = solve_offline_exact(scen_l3, cfg_l3)
    sfa_l3 = run_simulation(scen_l3, make_sfa(), cfg_l3)
    edf_l3 = run_simulation(scen_l3, make_edf_prebatch(), cfg_l3)
    l3_ok = (ex_l3["best_ontime"] == 3 and sfa_l3["n_ontime"] == 1 and edf_l3["n_ontime"] <= 1)

    ce_report = {
        "L1_prebatch_blindness": {
            "passed": bool(l1_ok),
            "oracle_ontime": ex_l1["best_ontime"],
            "edf_prebatch": edf_l1["n_ontime"],
            "llf_prebatch": llf_l1["n_ontime"],
            "dgolb": dgolb_l1["n_ontime"],
            "n_tasks": scen_l1.n_tasks,
            "dgolb_miss_rate": dgolb_l1["server_grant_miss_rate"],
        },
        "L2_non_delay": {
            "passed": bool(l2_ok),
            "oracle_delay_allowed": ex_l2_delay["best_ontime"],
            "oracle_non_delay": ex_l2_nondelay["best_ontime"],
            "edf_non_delay": edf_l2_nd["n_ontime"],
            "sfa_non_delay": sfa_l2_nd["n_ontime"],
            "n_tasks": scen_l2.n_tasks,
        },
        "L3_myopic_edf": {
            "passed": bool(l3_ok),
            "oracle_ontime": ex_l3["best_ontime"],
            "sfa_myopic": sfa_l3["n_ontime"],
            "edf_prebatch": edf_l3["n_ontime"],
            "n_tasks": scen_l3.n_tasks,
        },
    }

    # Gate 7b: 2D Dimensionless Phase Space Sweep (rho_cmp vs kappa_ul) using Common Random Numbers
    load_vals = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    # Sweep uplink coupling kappa_ul monotonically via input payload D_kb while holding topology seed fixed!
    kb_scales = [3.0, 22.0, 50.0, 80.0, 115.0, 155.0]  # mean input KB

    grid_kappa = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_rho = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_dgolb_miss = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_edf_ontime = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_sfa_zero_ontime = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_gap_pp = np.zeros((len(kb_scales), len(load_vals)), dtype=float)

    pol_dgolb = make_dgolb()
    pol_edf = make_edf_prebatch()
    pol_sfa = make_sfa()

    for r_idx, kb_mid in enumerate(kb_scales):
        for c_idx, ld in enumerate(load_vals):
            k_acc, rho_acc, miss_acc, edf_acc, sfa_acc = [], [], [], [], []
            for s_rep in range(2):
                scen_g = build_scenario_v1(
                    case_id=f"PS_{r_idx}_{c_idx}_{s_rep}",
                    family="phase_grid",
                    seed=31000 + s_rep,
                    n_devices=24,
                    n_servers=6,
                    tasks_per_device=5,
                    offered_load=ld,
                    T_max_s=0.75,
                    radius_m=220.0,
                    input_kb_range=(0.75 * kb_mid, 1.25 * kb_mid),
                )
                cfg_g = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero")
                dm = compute_dimensionless_numbers(scen_g, cfg_g)
                r_dg = run_simulation(scen_g, pol_dgolb, cfg_g)
                r_ed = run_simulation(scen_g, pol_edf, cfg_g)
                r_sf = run_simulation(scen_g, pol_sfa, cfg_g)

                k_acc.append(dm["kappa_ul"])
                rho_acc.append(dm["rho_cmp"])
                miss_acc.append(100.0 * r_dg["server_grant_miss_rate"])
                edf_acc.append(100.0 * r_ed["ontime_rate"])
                sfa_acc.append(100.0 * r_sf["ontime_rate"])

            grid_kappa[r_idx, c_idx] = float(np.mean(k_acc))
            grid_rho[r_idx, c_idx] = float(np.mean(rho_acc))
            grid_dgolb_miss[r_idx, c_idx] = float(np.mean(miss_acc))
            grid_edf_ontime[r_idx, c_idx] = float(np.mean(edf_acc))
            grid_sfa_zero_ontime[r_idx, c_idx] = float(np.mean(sfa_acc))
            grid_gap_pp[r_idx, c_idx] = float(np.mean(sfa_acc) - max(np.mean(edf_acc), 100.0 * (1.0 - np.mean(miss_acc))))

    # Plot 2D Dimensionless Phase Space Figure
    suite_named = [(sc.case_id, compute_dimensionless_numbers(sc, cf)) for sc, cf in build_calibrated_suite(seed=2026)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=220)

    k_mean_rows = np.mean(grid_kappa, axis=1)
    r_mean_cols = np.mean(grid_rho, axis=0)
    x_lo, x_hi = float(min(r_mean_cols)), float(max(r_mean_cols))
    y_lo, y_hi = float(min(k_mean_rows)), float(max(k_mean_rows))

    # Panel A: DGOLB Pre-Batch Server-Grant Miss Rate (%)
    ax0 = axes[0]
    im0 = ax0.contourf(r_mean_cols, k_mean_rows, grid_dgolb_miss, levels=12, cmap="YlOrRd")
    ax0.contour(r_mean_cols, k_mean_rows, grid_dgolb_miss, levels=[15.0, 35.0, 55.0], colors="black", linewidths=0.8, linestyles="--")
    fig.colorbar(im0, ax=ax0, label="DGOLB Server-Grant Miss Rate (%)")
    ax0.set_title("(a) Pre-Batch Collapse on $(\\rho_{\\mathrm{cmp}},\\,\\kappa_{\\mathrm{UL}})$", fontsize=10.5)
    ax0.set_xlabel(r"Normalized Offered Compute Load $\rho_{\mathrm{cmp}}$", fontsize=10)
    ax0.set_ylabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$", fontsize=10)

    # Panel B: Fluid Joint Feasibility Gain over EDF_PreBatch (pp) & Regimes
    diff_pp = grid_sfa_zero_ontime - grid_edf_ontime
    ax1 = axes[1]
    im1 = ax1.contourf(r_mean_cols, k_mean_rows, diff_pp, levels=12, cmap="viridis")
    fig.colorbar(im1, ax=ax1, label="Batch-Consistent Gain vs Pre-Batch (pp)")
    ax1.set_title("(b) Phase-Space Zones & Calibrated Scenarios", fontsize=10.5)
    ax1.set_xlabel(r"Normalized Offered Compute Load $\rho_{\mathrm{cmp}}$", fontsize=10)
    ax1.set_ylabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$", fontsize=10)

    # Overlay calibrated suite markers and regime boxes on both panels
    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_hi + 0.04)
        # Zone I boundary: kappa_ul <= 0.12 (weak coupling)
        ax.axhline(0.12, color="#ffffff" if ax_idx == 1 else "#1f77b4", linestyle=":", linewidth=1.3)
        # Zone II boundary: rho_cmp >= 0.90 (compute-saturated)
        ax.axvline(0.90, color="#ffffff" if ax_idx == 1 else "#2ca02c", linestyle="-.", linewidth=1.3)
        for cid, pt in suite_named:
            rx = float(pt["rho_cmp"])
            ky = float(pt["kappa_ul"])
            if ky <= y_hi:
                ax.scatter(rx, ky, c="cyan", edgecolors="black", s=50, zorder=5)
                if ax_idx == 1:
                    lbl = cid.replace("CAL_", "")
                    dy = 0.05 if "DEADLINE" not in cid else -0.09
                    dx = -0.06 if rx > 0.85 else 0.015
                    ax.text(rx + dx, max(0.16, ky + dy), lbl, fontsize=7.0, color="white", weight="bold")
            else:
                ax.scatter(rx, y_hi - 0.015, marker="^", c="cyan", edgecolors="black", s=70, zorder=6)
                if ax_idx == 1:
                    ax.text(rx - 0.08, y_hi - 0.10, f"INTERF_DENSE\n(kappa={ky:.2f})", fontsize=7.0, color="white", weight="bold",
                            bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.55))

    ax1.text(0.28, 0.045, "Zone I: Weakly Coupled (kappa <= 0.12)", fontsize=7.8, color="white", weight="bold")
    ax1.text(0.32, 1.07, "Zone III: Core Coupled Phase-Transition Regime", fontsize=8.0, color="white", weight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.45))
    ax1.text(0.908, 0.45, "Zone II:\nCompute\nSat.", fontsize=7.5, color="white", weight="bold")

    fig.tight_layout()
    fig.savefig(STAGE2_DIR / "phase_space_2d.png", dpi=220)
    plt.close(fig)

    gate7 = {
        "passed": bool(l1_ok and l2_ok and l3_ok and np.max(grid_dgolb_miss) >= 20.0),
        "counterexamples": ce_report,
        "phase_grid": {
            "rho_cols": [float(x) for x in r_mean_cols],
            "kappa_rows": [float(x) for x in k_mean_rows],
            "dgolb_miss_rate_pct": grid_dgolb_miss.tolist(),
            "edf_prebatch_ontime_pct": grid_edf_ontime.tolist(),
            "sfa_zero_ontime_pct": grid_sfa_zero_ontime.tolist(),
            "gain_over_prebatch_pp": diff_pp.tolist(),
        },
    }
    return gate6, gate7


# --------------------------------------------------------------------------- #
# Gate 8: Model v1.0 SHA-256 Freeze Manifest
# --------------------------------------------------------------------------- #

def write_freeze_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    lines: list[str] = []
    for p in sorted(STAGE2_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.name in ("FREEZE_MANIFEST_v1.0.sha256",):
            continue
        if p.suffix in (".py", ".md", ".png", ".json"):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest[p.name] = digest
            lines.append(f"{digest}  {p.name}")
    (STAGE2_DIR / "FREEZE_MANIFEST_v1.0.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    t0 = time.perf_counter()
    print("[Gate 1] Auditing Stage-2 directory for algorithm smuggling...")
    g1 = check_gate1_no_smuggling()
    print(f"  -> passed={g1['passed']} ({len(g1['scanned_files'])} files scanned, {len(g1['hits'])} hits)")

    print("[Gate 2 & 3] Auditing causal isolation, real-hardware caps & dimensionless scaling...")
    g2, g3 = check_gate2_and_gate3()
    print(f"  -> Gate 2 passed={g2['passed']} (checks={g2['hard_invariant_checks_total']}, max_bit_err={g2['max_bit_rel_err']:.2e}, illegal={g2['illegal_actions_total']})")
    print(f"  -> Gate 3 passed={g3['passed']} (kappa_ul={g3['kappa_ul_range']}, rho_cmp={g3['rho_cmp_range']}, gamma_tick={g3['gamma_tick_range']})")

    print("[Gate 4] Running 100 random micro-instances exact offline oracle vs causal baselines...")
    g4 = check_gate4_micro_oracle(n_instances=100, seed=777)
    print(f"  -> passed={g4['passed']} (feasible={g4['nonempty_feasible']}/100, inversions={g4['inversions']}, elapsed={g4['elapsed_s']:.2f}s)")

    print("[Gate 5] Verifying Stage-1 smoking-gun phenomena recovery & classical breakpoints...")
    g5 = check_gate5_phenomena()
    print(f"  -> passed={g5['passed']} (DGOLB shared miss={g5['shared_calibrated_summary']['DGOLB']['server_grant_miss_rate']:.3f} vs orth={g5['orthogonal_calibrated_summary']['DGOLB']['server_grant_miss_rate']:.3f}, IBR5 P70 p99={g5['burst_sweep']['70']['IBR_5']['decision_p99_ms']:.1f}ms)")

    print("[Gate 6 & 7] Running hereditary anti-monotonicity (200x), 4 anchors, L1-L3 counterexamples & 2D phase sweep...")
    g6, g7 = check_gate6_and_gate7()
    print(f"  -> Gate 6 passed={g6['passed']} (hered={g6['hereditary_anti_monotonicity']['passed']}/200, anchors={g6['degenerate_anchors']['all_passed']})")
    print(f"  -> Gate 7 passed={g7['passed']} (L1={g7['counterexamples']['L1_prebatch_blindness']['passed']}, L2={g7['counterexamples']['L2_non_delay']['passed']}, L3={g7['counterexamples']['L3_myopic_edf']['passed']})")

    all_passed = bool(g1["passed"] and g2["passed"] and g3["passed"] and g4["passed"] and g5["passed"] and g6["passed"] and g7["passed"])

    report = {
        "model_version": "v1.0",
        "elapsed_s": float(time.perf_counter() - t0),
        "all_passed": all_passed,
        "gate1_no_smuggling": g1,
        "gate2_hardware_fidelity": g2,
        "gate3_dimensionless": g3,
        "gate4_micro_oracle": g4,
        "gate5_phenomena_recovery": g5,
        "gate6_structure_and_anchors": g6,
        "gate7_lower_bounds_and_phase_space": g7,
    }
    (STAGE2_DIR / "stage2_verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[Gate 8] Generating FREEZE_MANIFEST_v1.0.sha256...")
    manifest = write_freeze_manifest()
    print(f"  -> Frozen {len(manifest)} files into FREEZE_MANIFEST_v1.0.sha256")

    if all_passed:
        print("\nALL 8 STAGE-2 EXIT GATES PASSED — MODEL v1.0 FROZEN")
        return 0
    print("\nSTAGE-2 GATE FAILURE DETECTED — CHECK stage2_verification_report.json")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
