"""Step 5.0: Preflight Fidelity, Calibration & Home-Regime Sanity Gate.

Produces:
  - stats/exp_config.json
  - stats/s50_preflight_report.json
  - data_csv/s50_wallclock_calib.csv
  - data_csv/s50_home_sanity.csv
"""
from __future__ import annotations

import gc
import json
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import linregress, spearmanr

from common_utils import (
    AZURE_EXTRACT_DIR,
    DATA_DIR,
    FORMAL_DIR,
    STAGE2_DIR,
    STAGE3_DIR,
    STAGE4_DIR,
    STATS_DIR,
    build_single_frozen_champion,
    sample_azure_tasks,
    save_csv_strict,
    save_json_strict,
    verify_upstream_readonly,
)

from benchmark_suite_s3 import build_dual_track_cg_baselines  # noqa: E402
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
import champion_core  # noqa: E402
from exact_oracle import (  # noqa: E402
    build_counterexample_L1,
    build_counterexample_L2,
    build_counterexample_L3,
)
from model_spec import EdgeUplinkEnv, SimConfig, build_scenario_v1, run_simulation  # noqa: E402
from soundness_test import check_counterexamples, check_degenerate_anchors  # noqa: E402


def run_parity_gate(champ: Any) -> dict[str, Any]:
    """Verify exact equivalence with Stage-3 reference results under zero latency."""
    s3_ref = json.loads((STAGE3_DIR / "stage3_results.json").read_text(encoding="utf-8"))
    ref_g1 = s3_ref["gate1_soundness"]

    ce_res = check_counterexamples(champ)
    an_res = check_degenerate_anchors(champ)

    # Compare exact counts against stage3_results.json
    parity_diffs: list[dict[str, Any]] = []
    for ce_key in ("L1", "L2", "L3", "CE_2ndOrder"):
        got_c = int(ce_res[ce_key]["champion"])
        ref_c = int(ref_g1["counterexamples"][ce_key]["champion"])
        got_o = int(ce_res[ce_key]["oracle"])
        ref_o = int(ref_g1["counterexamples"][ce_key]["oracle"])
        err = max(abs(got_c - ref_c), abs(got_o - ref_o))
        parity_diffs.append({"item": ce_key, "got": got_c, "ref": ref_c, "err": float(err)})

    for an_key in ("anchor1_single_srv_ortho", "anchor2_inf_cpu_sinr", "anchor3_inf_bw_pack", "anchor4_single_task"):
        got_a = int(an_res[an_key]["admitted"])
        ref_a = int(ref_g1["degenerate_anchors"][an_key]["admitted"])
        err = abs(got_a - ref_a)
        parity_diffs.append({"item": an_key, "got": got_a, "ref": ref_a, "err": float(err)})

    # Check zero-noise bit conservation error across all 4 counterexamples + calibrated zero-noise scenario
    max_bit_err = 0.0
    illegal_tot = 0
    for builder in (build_counterexample_L1, build_counterexample_L2, build_counterexample_L3, build_counterexample_2nd_order):
        sc, cf = builder()
        sim = run_simulation(sc, champ, cf)
        max_bit_err = max(max_bit_err, float(sim["max_bit_rel_err"]))
        illegal_tot += int(sim["illegal_actions"])

    sc_cal = build_scenario_v1(
        case_id="PARITY_CAL_ZERO",
        family="balanced",
        seed=2027,
        n_devices=24,
        n_servers=6,
        tasks_per_device=6,
        offered_load=0.60,
        T_max_s=1.0,
        radius_m=300.0,
    )
    sim_cal = run_simulation(sc_cal, champ, SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="zero"))
    max_bit_err = max(max_bit_err, float(sim_cal["max_bit_rel_err"]))
    illegal_tot += int(sim_cal["illegal_actions"])

    max_parity_err = max(d["err"] for d in parity_diffs)
    passed = bool(
        ce_res["passed"]
        and an_res["passed"]
        and max_parity_err <= 1e-4
        and max_bit_err < 1e-8
        and illegal_tot == 0
    )
    return {
        "passed": passed,
        "max_parity_err": float(max_parity_err),
        "max_bit_rel_err": float(max_bit_err),
        "illegal_actions": int(illegal_tot),
        "counterexamples": ce_res,
        "degenerate_anchors": an_res,
        "parity_diffs": parity_diffs,
    }


def run_wallclock_calibration(champ: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure CLWB_Dual wall-clock vs theoretical operation proxy (R^2) and 6-policy rank order."""
    cg_dict = dict(build_dual_track_cg_baselines())
    calib_rows: list[dict[str, Any]] = []

    # Part 1: CLWB_Dual native wall-clock vs theoretical op proxy across P in {10, 25, 50, 75, 100}
    orig_verify = champion_core._fast_fluid_verify
    v_calls = 0
    v_act_plus_cand = 0

    def counted_verify(obs_arg: dict, cand_arg: list[dict], act_lag_s: float = 0.0) -> dict[str, Any]:
        nonlocal v_calls, v_act_plus_cand
        v_calls += 1
        v_act_plus_cand += len(obs_arg["active_uplinks"]) + len(cand_arg)
        return orig_verify(obs_arg, cand_arg, act_lag_s)

    xs_ops: list[float] = []
    ys_ms: list[float] = []
    p_clwb_grid = [10, 25, 50, 75, 100]

    champion_core._fast_fluid_verify = counted_verify
    gc.disable()
    try:
        for p_val in p_clwb_grid:
            m_srv = max(2, p_val // 5)
            sc = build_scenario_v1(
                case_id=f"CALIB_CLWB_P{p_val}",
                family="WK02_DENSE_TIGHT",
                seed=50500,
                n_devices=p_val,
                n_servers=m_srv,
                tasks_per_device=1,
                offered_load=0.80,
                T_max_s=0.45,
                radius_m=150.0,
                f_dev_range_ghz=(0.3, 0.5),
                cycles_range=(6e8, 9e8),
                n_subbands=1,
                window_s=0.04,
            )
            cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, n_subbands=1, latency_mode="wall")
            env = EdgeUplinkEnv(sc, cfg)
            env._reveal_arrivals(0.05)
            obs = env.observe(0.05)
            for _ in range(3):
                champ(obs)
            v_calls = 0
            v_act_plus_cand = 0
            n_rep = 30
            t0 = time.perf_counter()
            for _ in range(n_rep):
                champ(obs)
            wall_ms = (time.perf_counter() - t0) * 1000.0 / float(n_rep)
            mean_vcalls = v_calls / float(n_rep)
            theo_ops = (v_act_plus_cand / float(n_rep)) + float(p_val * m_srv)
            xs_ops.append(float(theo_ops))
            ys_ms.append(float(wall_ms))
            calib_rows.append({
                "calib_kind": "clwb_regression",
                "family": "WK02_DENSE_TIGHT",
                "P": int(p_val),
                "M": int(m_srv),
                "method": "CLWB_Dual",
                "mean_verify_calls": float(mean_vcalls),
                "theo_ops_proxy": float(theo_ops),
                "wall_ms": float(wall_ms),
            })

        reg = linregress(xs_ops, ys_ms)
        r2 = float(reg.rvalue ** 2)

        # Part 2: 6-method cross-algorithm wall-clock rank order over CAL_BALANCED and WK02_DENSE_TIGHT
        rank_methods = [
            ("EDF_PreBatch+CG", cg_dict["EDF_PreBatch+CG"], 1),
            ("DGOLB+CG", cg_dict["DGOLB+CG"], 2),
            ("Bambos_ALP+CG", cg_dict["Bambos_ALP+CG"], 3),
            ("CLWB_Dual", champ, 4),
            ("SFA+CG", cg_dict["SFA+CG"], 5),
            ("IBR_5+CG", cg_dict["IBR_5+CG"], 6),
        ]
        per_method_ms: dict[str, list[float]] = {m_name: [] for m_name, _, _ in rank_methods}
        for fam, m_srv, rad in [("CAL_BALANCED", 6, 300.0), ("WK02_DENSE_TIGHT", 4, 150.0)]:
            for p_val in (10, 20, 35, 50, 75):
                sc = build_scenario_v1(
                    case_id=f"RANK_{fam}_{p_val}",
                    family=fam,
                    seed=50000 + p_val,
                    n_devices=p_val,
                    n_servers=m_srv,
                    tasks_per_device=1,
                    offered_load=0.60,
                    T_max_s=1.0,
                    radius_m=rad,
                    f_dev_range_ghz=(0.7, 0.9),
                    cycles_range=(6e8, 8.5e8),
                    input_kb_range=(60.0, 120.0),
                    f_ms_ghz=10.0,
                    window_s=0.04,
                )
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
                env = EdgeUplinkEnv(sc, cfg)
                env._reveal_arrivals(0.05)
                obs = env.observe(0.05)
                for m_name, pol_fn, theo_rank in rank_methods:
                    # Warmup
                    n_warm = 3 if m_name not in ("SFA+CG", "IBR_5+CG") else (1 if p_val <= 35 else 0)
                    for _ in range(n_warm):
                        pol_fn(obs)
                    n_rep = 30 if m_name not in ("SFA+CG", "IBR_5+CG") else (3 if p_val <= 35 else 1)
                    t0 = time.perf_counter()
                    for _ in range(n_rep):
                        pol_fn(obs)
                    wall_ms = (time.perf_counter() - t0) * 1000.0 / float(n_rep)
                    per_method_ms[m_name].append(float(wall_ms))
                    calib_rows.append({
                        "calib_kind": "rank_comparison",
                        "family": fam,
                        "P": int(p_val),
                        "M": int(m_srv),
                        "method": m_name,
                        "mean_verify_calls": 0.0,
                        "theo_ops_proxy": float(theo_rank),
                        "wall_ms": float(wall_ms),
                    })
    finally:
        champion_core._fast_fluid_verify = orig_verify
        gc.enable()

    mean_wall_by_method = {m_name: float(np.mean(per_method_ms[m_name])) for m_name, _, _ in rank_methods}
    theo_ranks = [r for _, _, r in rank_methods]
    measured_vals = [mean_wall_by_method[m_name] for m_name, _, _ in rank_methods]
    rho_s = float(spearmanr(theo_ranks, measured_vals).statistic)

    passed = bool(r2 >= 0.98 and abs(rho_s - 1.0) <= 1e-12)
    return {
        "passed": passed,
        "clwb_r2": r2,
        "clwb_slope": float(reg.slope),
        "clwb_intercept": float(reg.intercept),
        "spearman_rho": rho_s,
        "mean_wall_ms_by_method": mean_wall_by_method,
    }, calib_rows


def run_azure_calibration(n_samples: int = 2000, n_mc_reps: int = 100) -> dict[str, Any]:
    """Validate Azure Functions 2019 inverse-CDF sampling reconstruction MAPE < 5%."""
    files = ["d01_PK.npz", "d04_EV.npz", "d08_RM.npz"]
    records: list[dict[str, Any]] = []
    max_mape = 0.0
    for idx, fname in enumerate(files):
        fpath = AZURE_EXTRACT_DIR / fname
        if not fpath.exists():
            raise RuntimeError(f"Missing Azure trace file: {fpath}")
        with np.load(fpath, allow_pickle=False) as data:
            pool = {k: data[k] for k in data.files}
        for s_idx, stratum in enumerate(("FLEET", "BURST")):
            rng = np.random.default_rng(50300 + idx * 10 + s_idx)
            d_acc: list[float] = []
            r_acc: list[float] = []
            for _ in range(int(n_mc_reps)):
                _t3600, dur_s, raw_avg_ms = sample_azure_tasks(pool, stratum, n_samples, rng)
                d_acc.append(float(np.mean(dur_s) * 1000.0))
                r_acc.append(float(raw_avg_ms))
            sampled_mean_ms = float(np.mean(d_acc))
            raw_mean_ms = float(np.mean(r_acc))
            rel_err_pct = abs(sampled_mean_ms - raw_mean_ms) / max(raw_mean_ms, 1e-12) * 100.0
            max_mape = max(max_mape, rel_err_pct)
            records.append({
                "file": fname,
                "stratum": stratum,
                "n_samples": int(n_samples),
                "n_mc_reps": int(n_mc_reps),
                "raw_avg_ms": raw_mean_ms,
                "sampled_mean_ms": sampled_mean_ms,
                "rel_err_pct": float(rel_err_pct),
            })
    mean_mape = float(np.mean([r["rel_err_pct"] for r in records]))
    return {
        "passed": bool(mean_mape < 5.0 and max_mape < 5.0),
        "mean_mape_pct": mean_mape,
        "max_mape_pct": float(max_mape),
        "samples": records,
    }


def run_home_regime_sanity(champ: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify all 7 +CG baselines achieve strong on-time rates in their comfort zones."""
    cg_baselines = build_dual_track_cg_baselines()
    all_policies = [("CLWB_Dual", champ)] + cg_baselines
    rows: list[dict[str, Any]] = []

    # Regime 1: Orthogonal light-load comfort zone
    ortho_scores: dict[str, list[float]] = {name: [] for name, _ in all_policies}
    ortho_skips: dict[str, int] = {name: 0 for name, _ in all_policies}
    for s_idx, seed in enumerate((50101, 50102, 50103)):
        sc = build_scenario_v1(
            case_id=f"SANITY_ORTHO_S{s_idx}",
            family="sanity_ortho",
            seed=seed,
            n_devices=12,
            n_servers=6,
            tasks_per_device=4,
            offered_load=0.35,
            T_max_s=1.0,
        )
        cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=True, latency_mode="wall")
        for pname, pfn in all_policies:
            res = run_simulation(sc, pfn, cfg)
            ortho_scores[pname].append(float(res["ontime_rate"]))
            ortho_skips[pname] += int(res["skipped_ticks"])
            rows.append({
                "regime": "ortho_light",
                "seed": int(seed),
                "method": pname,
                "ontime_rate": float(res["ontime_rate"]),
                "server_grant_miss_rate": float(res["server_grant_miss_rate"]),
                "skipped_ticks": int(res["skipped_ticks"]),
                "illegal_actions": int(res["illegal_actions"]),
                "decision_p99_ms": float(res["decision_p99_ms"]),
            })

    # Regime 2: Small-scale shared relaxed zone
    shared_scores: dict[str, list[float]] = {name: [] for name, _ in all_policies}
    for s_idx, seed in enumerate((50201, 50202, 50203)):
        sc = build_scenario_v1(
            case_id=f"SANITY_SHARED_RELAXED_S{s_idx}",
            family="sanity_shared_relaxed",
            seed=seed,
            n_devices=8,
            n_servers=4,
            tasks_per_device=3,
            offered_load=0.30,
            T_max_s=1.0,
            radius_m=150.0,
            input_kb_range=(20.0, 40.0),
        )
        cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, orthogonal=False, latency_mode="wall")
        for pname, pfn in all_policies:
            res = run_simulation(sc, pfn, cfg)
            shared_scores[pname].append(float(res["ontime_rate"]))
            rows.append({
                "regime": "shared_relaxed",
                "seed": int(seed),
                "method": pname,
                "ontime_rate": float(res["ontime_rate"]),
                "server_grant_miss_rate": float(res["server_grant_miss_rate"]),
                "skipped_ticks": int(res["skipped_ticks"]),
                "illegal_actions": int(res["illegal_actions"]),
                "decision_p99_ms": float(res["decision_p99_ms"]),
            })

    ortho_means = {k: float(np.mean(v)) for k, v in ortho_scores.items()}
    shared_means = {k: float(np.mean(v)) for k, v in shared_scores.items()}

    ortho_ok = all(ortho_means[name] >= 0.95 and ortho_skips[name] == 0 for name, _ in cg_baselines)
    shared_targets = ("Bambos_ALP+CG", "Chen_Game+CG", "SFA+CG", "IBR_5+CG")
    shared_ok = all(shared_means[name] >= 0.90 for name in shared_targets)
    no_illegal = all(r["illegal_actions"] == 0 for r in rows)

    return {
        "passed": bool(ortho_ok and shared_ok and no_illegal),
        "ortho_ok": bool(ortho_ok),
        "shared_ok": bool(shared_ok),
        "ortho_mean_ontime": ortho_means,
        "ortho_skipped_ticks": ortho_skips,
        "shared_mean_ontime": shared_means,
    }, rows


def main() -> None:
    t0 = time.perf_counter()
    upstream_status = verify_upstream_readonly()
    champ = build_single_frozen_champion()

    exp_config = {
        "stage": 5,
        "champion_factory": "make_clwb_dual(use_externality_pricing=True, use_lagrangian_defer=True, use_fluid_contraction=True)",
        "paths": {
            "root_v3": str(FORMAL_DIR.parent),
            "stage2_dir": str(STAGE2_DIR),
            "stage3_dir": str(STAGE3_DIR),
            "stage4_dir": str(STAGE4_DIR),
            "formal_dir": str(FORMAL_DIR),
            "azure_extract_dir": str(AZURE_EXTRACT_DIR),
        },
        "default_sim_config": {
            "tick_s": 0.05,
            "delta_sig_s": 0.001,
            "rho_sys": 0.95,
            "f_min_hz": 1_000_000.0,
            "latency_mode": "wall",
        },
        "seed_namespaces": {
            "s50_calib_clwb": 50500,
            "s50_calib_rank": 50000,
            "s50_sanity_ortho": [50101, 50102, 50103],
            "s50_sanity_shared": [50201, 50202, 50203],
            "b1a_ortho_vs_shared": "51100..51109",
            "b1b_kappa_sweep": "51200..51209",
            "b1c_burst_cliff": "51300..51309",
            "b2_main_k12": "41000+ (10 seeds/workload)",
            "b2_ood_k4": "88000+ (10 seeds/workload)",
            "b2_blind_k8": "99000+ (10 seeds/workload)",
            "b2_micro100": 777,
            "b3e_regime_map": "53500..53509",
            "b4a_azure_trace": "54100..54163",
        },
        "azure_transparency_manifest": {
            "REAL_from_Azure_2019_trace": [
                "Per-minute invocation arrival density across fleet and bursty top-24 apps (counts matrix F x 60)",
                "Per-function execution duration 7-quantile distribution knots (0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0), vmin, vmax",
                "Published per-function empirical mean execution duration (avg_ms)",
            ],
            "SYNTHETIC_ScenarioV1_binding": [
                "Uniform sub-minute arrival expansion linearly rescaled onto physical window [0, 1.25s)",
                "Dimensionless cycle scaling C_r = clip(5.5e8 * dur_s / mean(dur_s), 5e7, 2.5e9) preserving relative duration ratios",
                "Round-robin binding across 24 terminals, 6 MEC servers (50 GHz), 10 MHz shared wireless channel, radius 220m, T_max=0.85s",
            ],
        },
    }
    save_json_strict(exp_config, STATS_DIR / "exp_config.json")

    parity = run_parity_gate(champ)
    wallclock, calib_rows = run_wallclock_calibration(champ)
    azure_cal = run_azure_calibration(n_samples=2000)
    sanity, sanity_rows = run_home_regime_sanity(champ)

    save_csv_strict(calib_rows, DATA_DIR / "s50_wallclock_calib.csv")
    save_csv_strict(sanity_rows, DATA_DIR / "s50_home_sanity.csv")

    all_passed = bool(
        upstream_status["intact"]
        and parity["passed"]
        and wallclock["passed"]
        and azure_cal["passed"]
        and sanity["passed"]
    )
    report = {
        "all_preflight_passed": all_passed,
        "elapsed_s": float(time.perf_counter() - t0),
        "upstream_readonly": upstream_status,
        "parity_gate": parity,
        "wallclock_calibration": wallclock,
        "azure_calibration": azure_cal,
        "home_regime_sanity": sanity,
    }
    save_json_strict(report, STATS_DIR / "s50_preflight_report.json")

    print(
        f"[Step 5.0] all_preflight_passed={all_passed} | "
        f"parity={parity['passed']} (bit_err={parity['max_bit_rel_err']:.2e}) | "
        f"wallclock={wallclock['passed']} (R2={wallclock['clwb_r2']:.4f}, rho={wallclock['spearman_rho']:.3f}) | "
        f"azure={azure_cal['passed']} (mean_MAPE={azure_cal['mean_mape_pct']:.3f}%, max_MAPE={azure_cal['max_mape_pct']:.3f}%) | "
        f"sanity={sanity['passed']}",
        flush=True,
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
