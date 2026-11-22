"""Stage 4 Step 4.5: 2D Mechanism Regime Map & Double-Layer Red-Light Verdict Gate.

Generates:
  1. `stage4_regime_map_2d.png`: 6x6x3 Common-Random-Numbers sweep across Stage-2's orthogonal
     dimensionless control axes (rho_cmp, kappa_ul) comparing CLWB_Dual vs Bambos_ALP+CG and DGOLB+CG,
     with Stage-2 Zone I/II boundaries and CAL_* scenarios overlaid.
  2. `stage4_verdict.json`: Honest machine-readable evaluation of all 5 Red Lights (A-E), fine-grained
     sub-component necessity flags, parameter dead-line flags, and theorem fuzz violation counts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np

STAGE4_DIR = Path(__file__).resolve().parent
ROOT_V3 = STAGE4_DIR.parent
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"

for p in (str(STAGE4_DIR), str(STAGE3_DIR), str(STAGE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_stage2 import make_bambos_alp, make_dgolb
from champion_core import make_clwb_dual
from common_guard import wrap_with_common_guard
from model_spec import SimConfig, build_calibrated_suite, build_scenario_v1, compute_dimensionless_numbers, run_simulation


def compute_2d_regime_map() -> dict[str, Any]:
    load_vals = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    kb_scales = [3.0, 22.0, 50.0, 80.0, 115.0, 155.0]
    n_reps = 3

    pol_clwb = make_clwb_dual()
    pol_bambos = wrap_with_common_guard(make_bambos_alp(sinr_thresh=0.5), "Bambos_ALP+CG")
    pol_dgolb = wrap_with_common_guard(make_dgolb(), "DGOLB+CG")

    grid_kappa = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_rho = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_clwb = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_bambos = np.zeros((len(kb_scales), len(load_vals)), dtype=float)
    grid_dgolb = np.zeros((len(kb_scales), len(load_vals)), dtype=float)

    t0 = time.perf_counter()
    for r_i, kb_mid in enumerate(kb_scales):
        for c_i, ld in enumerate(load_vals):
            k_acc, r_acc, c_acc, b_acc, d_acc = [], [], [], [], []
            for rep in range(n_reps):
                sc = build_scenario_v1(
                    case_id=f"PS4_{r_i}_{c_i}_{rep}",
                    family="stage4_phase",
                    seed=95000 + rep,
                    n_devices=24,
                    n_servers=6,
                    tasks_per_device=5,
                    offered_load=ld,
                    T_max_s=0.75,
                    radius_m=220.0,
                    input_kb_range=(0.75 * kb_mid, 1.25 * kb_mid),
                )
                cfg = SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall")
                dm = compute_dimensionless_numbers(sc, cfg)
                rc = run_simulation(sc, pol_clwb, cfg)
                rb = run_simulation(sc, pol_bambos, cfg)
                rd = run_simulation(sc, pol_dgolb, cfg)
                k_acc.append(float(dm["kappa_ul"]))
                r_acc.append(float(dm["rho_cmp"]))
                c_acc.append(100.0 * float(rc["ontime_rate"]))
                b_acc.append(100.0 * float(rb["ontime_rate"]))
                d_acc.append(100.0 * float(rd["ontime_rate"]))
            grid_kappa[r_i, c_i] = float(np.mean(k_acc))
            grid_rho[r_i, c_i] = float(np.mean(r_acc))
            grid_clwb[r_i, c_i] = float(np.mean(c_acc))
            grid_bambos[r_i, c_i] = float(np.mean(b_acc))
            grid_dgolb[r_i, c_i] = float(np.mean(d_acc))

    diff_bambos = grid_clwb - grid_bambos
    diff_dgolb = grid_clwb - grid_dgolb
    k_rows = np.mean(grid_kappa, axis=1)
    r_cols = np.mean(grid_rho, axis=0)

    suite_named = [(sc.case_id, compute_dimensionless_numbers(sc, cf)) for sc, cf in build_calibrated_suite(seed=2026)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)

    # Panel (a): CLWB_Dual - Bambos_ALP+CG (pp)
    vmin_a = min(-1.0, float(np.min(diff_bambos)))
    vmax_a = max(1.0, float(np.max(diff_bambos)))
    norm_a = TwoSlopeNorm(vmin=vmin_a, vcenter=0.0, vmax=vmax_a)
    cf0 = axes[0].contourf(r_cols, k_rows, diff_bambos, levels=14, cmap="RdBu_r", norm=norm_a)
    axes[0].contour(r_cols, k_rows, diff_bambos, levels=[0.0, 5.0, 15.0], colors="black", linewidths=0.8, linestyles="--")
    fig.colorbar(cf0, ax=axes[0], label=r"Ontime Gain: CLWB_Dual $-$ Bambos_ALP+CG (pp)")
    axes[0].set_title(r"(a) Gain vs Classic Wireless ALP+CG on ($\rho_{\mathrm{cmp}},\,\kappa_{\mathrm{UL}}$)", fontsize=10)
    axes[0].set_xlabel(r"Normalized Offered Compute Load $\rho_{\mathrm{cmp}}$", fontsize=9.5)
    axes[0].set_ylabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$", fontsize=9.5)

    # Panel (b): CLWB_Dual - DGOLB+CG (pp)
    cf1 = axes[1].contourf(r_cols, k_rows, diff_dgolb, levels=14, cmap="viridis")
    axes[1].contour(r_cols, k_rows, diff_dgolb, levels=[10.0, 35.0, 60.0], colors="white", linewidths=0.8, linestyles="--")
    fig.colorbar(cf1, ax=axes[1], label=r"Ontime Gain: CLWB_Dual $-$ DGOLB+CG (pp)")
    axes[1].set_title(r"(b) Gain vs Stage-1 SOTA DGOLB+CG & Zone Boundaries", fontsize=10)
    axes[1].set_xlabel(r"Normalized Offered Compute Load $\rho_{\mathrm{cmp}}$", fontsize=9.5)
    axes[1].set_ylabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$", fontsize=9.5)

    x_lo, x_hi = float(min(r_cols)), float(max(r_cols))
    y_hi = float(max(k_rows))
    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_hi + 0.04)
        ax.axhline(0.12, color="#1f77b4" if ax_idx == 0 else "#ffffff", linestyle=":", linewidth=1.4, label=r"Zone I ($\kappa_{\mathrm{UL}}=0.12$)")
        ax.axvline(0.90, color="#2ca02c" if ax_idx == 0 else "#ffdd57", linestyle="-.", linewidth=1.4, label=r"Zone II ($\rho_{\mathrm{cmp}}=0.90$)")
        for cid, pt in suite_named:
            rx = float(pt["rho_cmp"])
            ky = min(float(pt["kappa_ul"]), y_hi - 0.015)
            mk = "^" if float(pt["kappa_ul"]) > y_hi else "o"
            ax.scatter(rx, ky, marker=mk, c="cyan", edgecolors="black", s=46, zorder=5)
            if ax_idx == 1:
                lbl = cid.replace("CAL_", "")
                ax.text(rx + (-0.06 if rx > 0.82 else 0.012), max(0.08, ky + 0.035), lbl, fontsize=6.5, color="white", weight="bold")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Stage 4.5 Two-Dimensional Mechanism Regime Map", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig_path = STAGE4_DIR / "stage4_regime_map_2d.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return {
        "elapsed_s": float(time.perf_counter() - t0),
        "rho_cols": [float(x) for x in r_cols],
        "kappa_rows": [float(x) for x in k_rows],
        "clwb_grid": grid_clwb.tolist(),
        "bambos_grid": grid_bambos.tolist(),
        "dgolb_grid": grid_dgolb.tolist(),
        "diff_vs_bambos_pp": diff_bambos.tolist(),
        "diff_vs_dgolb_pp": diff_dgolb.tolist(),
        "zone1_mean_diff_bambos_pp": float(np.mean(diff_bambos[k_rows <= 0.12])) if np.any(k_rows <= 0.12) else None,
        "zone3_mean_diff_bambos_pp": float(np.mean(diff_bambos[k_rows > 0.12])) if np.any(k_rows > 0.12) else None,
        "max_gain_vs_bambos_pp": float(np.max(diff_bambos)),
        "min_gain_vs_bambos_pp": float(np.min(diff_bambos)),
        "max_gain_vs_dgolb_pp": float(np.max(diff_dgolb)),
    }


def evaluate_verdict(regime_map: dict[str, Any]) -> dict[str, Any]:
    r41 = json.loads((STAGE4_DIR / "step41_blind_report.json").read_text(encoding="utf-8"))
    r42 = json.loads((STAGE4_DIR / "step42_grafting_report.json").read_text(encoding="utf-8"))
    r43 = json.loads((STAGE4_DIR / "step43_sensitivity_report.json").read_text(encoding="utf-8"))
    r44 = json.loads((STAGE4_DIR / "step44_fuzz_report.json").read_text(encoding="utf-8"))

    # Evaluate Red Light A: Patchwork or non-positive fine-grained sub-component
    part_a = r42["part_a_anti_patchwork"]
    flagged_sub = list(part_a["flagged_non_positive"])
    for vname, st in part_a["summary"].items():
        if vname.startswith("sub_") and st["gain_20wk_pp"] <= 0.0 and st["miss_delta_pp"] <= 0.05 and not st["ce_drops"]:
            if vname not in flagged_sub:
                flagged_sub.append(vname)
    red_a_reasons = []
    if flagged_sub:
        red_a_reasons.append(f"non_positive_subcomponents={flagged_sub}")

    # Evaluate Red Light B: Net gain & statistical power on Blind K8 (+CG baselines)
    pair41 = r41["blind_k8"]["pairwise_vs_cg"]
    failed_holm_ep = [b for b, st in pair41.items() if st["holm_p_ep"] >= 0.05]
    red_b_reasons = []
    if failed_holm_ep:
        red_b_reasons.append(f"blind_k8_ep_holm_ge_0.05={failed_holm_ep}")

    # Evaluate Red Light C: Dead lines in parameter sensitivity scan
    dead_knobs = r43["dead_line_flags"]
    red_c_reasons = [f"dead_line_knobs={dead_knobs}"] if dead_knobs else []

    # Evaluate Red Light D: Theorem / lemma violations in 10,000-trial fuzzing (tightened hypotheses)
    red_d_reasons = []
    if r44["lemma1"]["violations"] > 0:
        red_d_reasons.append(f"lemma1_violations={r44['lemma1']['violations']}")
    t1 = r44["theorem1"]
    for k in ("hereditary_inversions", "exact_feasible_fails", "active_link_fails", "nic_fails", "cpu_cap_fails", "live_illegal_actions", "live_skipped_ticks"):
        if t1[k] > 0:
            red_d_reasons.append(f"theorem1_{k}={t1[k]}")
    if r44["theorem2"]["bound_violations"] > 0:
        red_d_reasons.append(f"theorem2_bound_violations={r44['theorem2']['bound_violations']}")
    for k, sub in r44["theorem3_anchors"].items():
        if sub["fails"] > 0:
            red_d_reasons.append(f"theorem3_{k}_fails={sub['fails']}/{sub['trials']}")

    # Evaluate Red Light E: Stage-2 Freeze or Sim-to-Real breach
    red_e_reasons = []
    if not r41["stage2_freeze"]["intact"]:
        red_e_reasons.append("stage2_freeze_broken")
    if not r41["sim_to_real_audit"]["sim_to_real_clean"]:
        red_e_reasons.append("sim_to_real_forbidden_or_illegal_key")
    if r41["blind_k8"]["overall"]["CLWB_Dual"]["total_illegal_actions"] > 0:
        red_e_reasons.append("illegal_actions_gt_0")

    red_lights = {
        "A_patchwork_or_redundant_subcomponent": {"triggered": bool(red_a_reasons), "reasons": red_a_reasons},
        "B_net_gain_or_statistics": {"triggered": bool(red_b_reasons), "reasons": red_b_reasons},
        "C_dead_line_parameter": {"triggered": bool(red_c_reasons), "reasons": red_c_reasons},
        "D_theorem_falsification": {"triggered": bool(red_d_reasons), "reasons": red_d_reasons},
        "E_model_or_sim_to_real": {"triggered": bool(red_e_reasons), "reasons": red_e_reasons},
    }
    all_green = not any(v["triggered"] for v in red_lights.values())
    return {
        "all_green": all_green,
        "red_lights": red_lights,
        "regime_map_summary": regime_map,
    }


def main() -> None:
    rmap = compute_2d_regime_map()
    verdict = evaluate_verdict(rmap)
    out_path = STAGE4_DIR / "stage4_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("[Step 4.5] all_green =", verdict["all_green"])
    print(json.dumps(verdict["red_lights"], indent=2))


if __name__ == "__main__":
    main()
