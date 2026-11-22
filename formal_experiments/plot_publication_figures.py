"""Step 5.5: Zero-Hardcoding Publication Figure Generator (8 Main Figures, PDF + 300 DPI PNG).

Reads exclusively from data_csv/ and stats/. Never hardcodes empirical numbers,
error bars, or significance stars. All labels use clean academic English without
version tags, file names, or code identifiers.
"""
from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_utils import DATA_DIR, FIG_DIR, STATS_DIR, bootstrap_ci95, verify_upstream_readonly

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito colorblind-safe palette + distinct markers & linestyles
METHOD_STYLE: dict[str, dict[str, Any]] = {
    "CLWB_Dual": {"label": "CLWB-Dual (Proposed)", "color": "#0072B2", "marker": "o", "ls": "-", "lw": 2.0, "zorder": 6},
    "Bambos_ALP+CG": {"label": "Bambos ALP + CG", "color": "#E69F00", "marker": "s", "ls": "--", "lw": 1.4, "zorder": 5},
    "Chen_Game+CG": {"label": "Chen Game + CG", "color": "#009E73", "marker": "^", "ls": "-.", "lw": 1.4, "zorder": 4},
    "DGOLB+CG": {"label": "DGOLB + CG", "color": "#D55E00", "marker": "D", "ls": ":", "lw": 1.5, "zorder": 4},
    "EDF_PreBatch+CG": {"label": "EDF Pre-Batch + CG", "color": "#CC79A7", "marker": "v", "ls": "--", "lw": 1.2, "zorder": 3},
    "LLF_PreBatch+CG": {"label": "LLF Pre-Batch + CG", "color": "#56B4E9", "marker": "P", "ls": "-.", "lw": 1.2, "zorder": 3},
    "SFA+CG": {"label": "SFA + CG", "color": "#B8860B", "marker": "X", "ls": ":", "lw": 1.4, "zorder": 3},
    "IBR_5+CG": {"label": "IBR-5 + CG", "color": "#4D4D4D", "marker": "*", "ls": "-.", "lw": 1.2, "zorder": 2},
    "Exact_Oracle": {"label": "Exact Oracle", "color": "#000000", "marker": "h", "ls": "-", "lw": 1.6, "zorder": 7},
}


def _holm_stars(p_val: float) -> str:
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def _save_fig(fig: plt.Figure, stem: str) -> None:
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: Dual Deadlock Motivation (Batch 1)
# --------------------------------------------------------------------------- #

def plot_fig1() -> None:
    df1a = pd.read_csv(DATA_DIR / "b1a_ortho_vs_shared.csv")
    df1b = pd.read_csv(DATA_DIR / "b1b_kappa_sweep.csv")
    df1c = pd.read_csv(DATA_DIR / "b1c_burst_cliff.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.50), dpi=300)
    fig.subplots_adjust(wspace=0.30, top=0.72)

    # Panel (a): Orthogonal vs Shared On-Time Rate & Shared Miss Rate
    methods_1a = [
        "Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG",
        "LLF_PreBatch+CG", "DGOLB+CG", "SFA+CG", "IBR_5+CG",
    ]
    short_1a = ["ALP", "Chen", "EDF", "LLF", "DGOLB", "SFA", "IBR-5"]
    x = np.arange(len(methods_1a))
    w = 0.34

    ortho_means, ortho_err = [], []
    shared_means, shared_err = [], []
    miss_means = []
    for m in methods_1a:
        vo = df1a[(df1a["channel_mode"] == "orthogonal") & (df1a["method"] == m)]["ontime_rate"].to_numpy() * 100.0
        vs = df1a[(df1a["channel_mode"] == "shared") & (df1a["method"] == m)]["ontime_rate"].to_numpy() * 100.0
        vm = df1a[(df1a["channel_mode"] == "shared") & (df1a["method"] == m)]["server_grant_miss_rate"].to_numpy() * 100.0
        mo, loo, hio = bootstrap_ci95(vo, n_boot=5000, seed=6001)
        ms, los, his = bootstrap_ci95(vs, n_boot=5000, seed=6002)
        ortho_means.append(mo)
        ortho_err.append([max(0.0, mo - loo), max(0.0, hio - mo)])
        shared_means.append(ms)
        shared_err.append([max(0.0, ms - los), max(0.0, his - ms)])
        miss_means.append(float(np.mean(vm)))

    ax0.bar(x - w / 2, ortho_means, width=w, color="#56B4E9", edgecolor="black", linewidth=0.5,
            yerr=np.array(ortho_err).T, capsize=2.0, label="Orthogonal On-Time")
    ax0.bar(x + w / 2, shared_means, width=w, color="#0072B2", edgecolor="black", linewidth=0.5,
            yerr=np.array(shared_err).T, capsize=2.0, label="Shared On-Time")
    ax0.plot(x + w / 2, miss_means, color="#D55E00", marker="D", markersize=4.5, linestyle="none",
             markeredgecolor="black", markeredgewidth=0.5, zorder=5, label="Shared Grant Miss")
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_1a, rotation=22, ha="right")
    ax0.set_ylim(0, 109)
    ax0.set_ylabel("Rate (%)")
    ax0.set_title("(a) Orthogonal vs. Shared Channel (+CG)", pad=44)
    ax0.grid(axis="y", alpha=0.25, linestyle="--")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
               frameon=True, framealpha=0.95, fontsize=5.9, columnspacing=0.6, handletextpad=0.3)

    # Panel (b): Pure baseline collapse + dashed theoretical/batch-consistent reference (SOP compliant)
    m_1b = ["Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG", "DGOLB+CG", "CLWB_Dual"]
    for m in m_1b:
        sub = df1b[df1b["method"] == m]
        grp = sub.groupby("interf_scale", sort=True)
        kx = grp["kappa_ul"].mean().to_numpy()
        ym, ylo, yhi = [], [], []
        for _, g in grp:
            mv, lv, hv = bootstrap_ci95(g["ontime_rate"].to_numpy() * 100.0, n_boot=5000, seed=6003)
            ym.append(mv)
            ylo.append(lv)
            yhi.append(hv)
        if m == "CLWB_Dual":
            ax1.plot(kx, ym, color="#111111", marker="", ls="--", lw=1.4, zorder=7,
                     label="Batch-Consistent Target")
        else:
            st = METHOD_STYLE[m]
            ax1.plot(kx, ym, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                     markersize=3.8, label=st["label"])
            ax1.fill_between(kx, ylo, yhi, color=st["color"], alpha=0.12)

    ax1.axvline(0.12, color="#4D4D4D", ls=":", lw=1.2)
    ax1.text(0.135, 8.0, r"Zone I ($\kappa_{\mathrm{UL}}=0.12$)", fontsize=6.6, color="#333333")
    ax1.set_xlabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$")
    ax1.set_ylabel("On-Time Completion Rate (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title(r"(b) Coupling Phase Collapse & Burst Cliff", pad=44)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=True, framealpha=0.95, fontsize=5.7, columnspacing=0.6, handletextpad=0.3)

    # Inset placed inside open region with white opaque background and compact legend
    ax_in = ax1.inset_axes([0.50, 0.36, 0.48, 0.35], zorder=5)
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.98)
    for m in ("IBR_5+CG", "SFA+CG", "CLWB_Dual"):
        grp_p = df1c[df1c["method"] == m].groupby("P_burst", sort=True)
        px = np.array(sorted(grp_p.groups.keys()), dtype=float)
        py = [float(grp_p.get_group(p)["decision_p99_ms"].mean()) for p in px]
        if m == "CLWB_Dual":
            ax_in.plot(px, py, color="#111111", marker="", ls="--", lw=1.2, label="Target")
        else:
            st = METHOD_STYLE[m]
            lbl_in = "IBR-5" if m.startswith("IBR") else "SFA"
            ax_in.plot(px, py, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.1,
                       markersize=2.4, label=lbl_in)
    ax_in.axhline(50.0, color="red", ls="--", lw=0.9)
    ax_in.text(10, 82.0, "50 ms", color="red", fontsize=5.4)
    ax_in.set_yscale("log")
    ax_in.set_ylim(0.25, 3e4)
    ax_in.set_xlabel(r"Burst $P$", fontsize=5.8, labelpad=1)
    ax_in.set_ylabel("P99 (ms)", fontsize=5.8, labelpad=1)
    ax_in.tick_params(labelsize=5.3)
    ax_in.set_title("Control Latency Cliff", fontsize=6.1, pad=2)
    ax_in.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=4.8,
                 handlelength=1.4, labelspacing=0.2, borderpad=0.25)
    ax_in.grid(True, alpha=0.25, linestyle=":")

    _save_fig(fig, "fig1_motivation_dual_deadlock")


# --------------------------------------------------------------------------- #
# Figure 2: Main Benchmark Comparison (Batch 2)
# --------------------------------------------------------------------------- #

def plot_fig2() -> None:
    b2_sum = json.loads((STATS_DIR / "b2_summary.json").read_text(encoding="utf-8"))
    df_ep = pd.read_csv(DATA_DIR / "b2_episodes.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.35), dpi=300)
    fig.subplots_adjust(wspace=0.28, top=0.78)

    scope_keys = ["Main_K12", "OOD_K4", "Blind_K8", "All_24"]
    scope_labels = ["Standard\n(K=12)", "OOD\n(K=4)", "Blind\n(K=8)", "Combined\n(K=24)"]
    methods = [
        "CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG",
        "LLF_PreBatch+CG", "DGOLB+CG", "SFA+CG", "IBR_5+CG",
    ]
    n_m = len(methods)
    x = np.arange(len(scope_keys))
    bw = 0.82 / n_m

    pair_all = b2_sum["scopes"]["All_24"]["pairwise_vs_cg"]
    comb_caps: list[tuple[float, float, str]] = []
    for idx, m in enumerate(methods):
        st = METHOD_STYLE[m]
        means, err_lo, err_hi = [], [], []
        for sk in scope_keys:
            pm = b2_sum["scopes"][sk]["per_method"][m]
            mv = pm["ontime_mean"] * 100.0
            ci = pm["ontime_ci95"]
            means.append(mv)
            err_lo.append(max(0.0, mv - ci[0] * 100.0))
            err_hi.append(max(0.0, ci[1] * 100.0 - mv))
        xpos = x + (idx - (n_m - 1) / 2.0) * bw
        ax0.bar(xpos, means, width=bw, color=st["color"], edgecolor="black", linewidth=0.35,
                yerr=[err_lo, err_hi], capsize=1.3, error_kw={"elinewidth": 0.7}, label=st["label"])
        if m != "CLWB_Dual":
            hp = float(pair_all[m]["holm_p"])
            star = _holm_stars(hp)
            if star != "ns":
                comb_caps.append((float(xpos[-1]), float(means[-1] + err_hi[-1]), star))

    prev_y = -1.0
    for xp_s, y_cap, star in comb_caps:
        y_s = y_cap + 2.2
        if prev_y >= 0.0 and abs(y_s - prev_y) < 4.5:
            y_s = prev_y + 5.0
        ax0.text(xp_s, min(110.5, y_s), star, ha="center", va="bottom",
                 fontsize=4.9, fontweight="bold", rotation=0)
        prev_y = y_s

    ax0.set_xticks(x)
    ax0.set_xticklabels(scope_labels)
    ax0.set_ylim(0, 116)
    ax0.set_ylabel("Mean On-Time Rate (%)")
    ax0.set_title("(a) Cross-Domain On-Time Rate (95% CI)")
    ax0.grid(axis="y", alpha=0.25, linestyle="--")

    # Shared figure-level legend cleanly above both panels
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=True, framealpha=0.95, fontsize=6.5, columnspacing=0.8)

    # Panel (b): 240-episode On-Time Rate ECDF + Miss Rate inset shielded by opaque white backdrop
    for m in methods:
        st = METHOD_STYLE[m]
        vals = np.sort(df_ep[df_ep["method"] == m]["ontime_rate"].to_numpy() * 100.0)
        ecdf_y = np.arange(1, len(vals) + 1) / float(len(vals))
        ax1.step(vals, ecdf_y, where="post", color=st["color"], ls=st["ls"], lw=st["lw"])

    ax1.set_xlabel("Episode On-Time Rate (%)")
    ax1.set_ylabel("Empirical CDF (240 Episodes)")
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0, 1.03)
    ax1.set_title("(b) Tail Robustness ECDF & Grant Miss Rate")
    ax1.grid(True, alpha=0.25, linestyle="--")

    bg_patch = FancyBboxPatch(
        (0.02, 0.34), 0.57, 0.59, transform=ax1.transAxes,
        boxstyle="round,pad=0.01", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5, alpha=0.97, zorder=4,
    )
    ax1.add_patch(bg_patch)
    ax_in = ax1.inset_axes([0.13, 0.47, 0.44, 0.39], zorder=5)
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.98)
    miss_m = [b2_sum["scopes"]["All_24"]["per_method"][m]["miss_mean"] * 100.0 for m in methods]
    miss_c = [METHOD_STYLE[m]["color"] for m in methods]
    ax_in.bar(np.arange(len(methods)), miss_m, color=miss_c, edgecolor="black", linewidth=0.35)
    ax_in.set_xticks(np.arange(len(methods)))
    ax_in.set_xticklabels(["CLWB", "ALP", "Chen", "EDF", "LLF", "DGOLB", "SFA", "IBR"],
                          rotation=42, ha="right", fontsize=5.2)
    ax_in.set_ylim(0, 105)
    ax_in.set_ylabel("Miss (%)", fontsize=5.8, labelpad=1)
    ax_in.tick_params(axis="y", labelsize=5.4)
    ax_in.set_title("Grant Miss (240 Eps)", fontsize=6.2, pad=2)
    ax_in.grid(axis="y", alpha=0.25, linestyle=":")

    _save_fig(fig, "fig2_main_benchmark_comparison")


# --------------------------------------------------------------------------- #
# Figure 3: 100 Micro-Oracle Capture & 4 Analytical Counterexamples
# --------------------------------------------------------------------------- #

def plot_fig3() -> None:
    df_m = pd.read_csv(DATA_DIR / "b2_micro100.csv")
    df_ce = pd.read_csv(DATA_DIR / "b1d_counterexamples.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.35), dpi=300)
    fig.subplots_adjust(wspace=0.28, top=0.75)

    methods_m = [
        "CLWB_Dual", "SFA+CG", "IBR_5+CG", "Chen_Game+CG",
        "Bambos_ALP+CG", "EDF_PreBatch+CG", "LLF_PreBatch+CG", "DGOLB+CG",
    ]
    short_m = ["CLWB-Dual", "SFA + CG", "IBR-5 + CG", "Chen + CG", "ALP + CG", "EDF + CG", "LLF + CG", "DGOLB + CG"]
    oracle_mean = float(df_m[df_m["method"] == "Exact_Oracle"]["ontime_rate"].mean()) * 100.0

    cap_means, cap_lo, cap_hi, cols = [], [], [], []
    for m in methods_m:
        vals = df_m[df_m["method"] == m]["ontime_rate"].to_numpy() * 100.0
        mv, lv, hv = bootstrap_ci95(vals, n_boot=5000, seed=6030)
        cap_means.append(mv)
        cap_lo.append(max(0.0, mv - lv))
        cap_hi.append(max(0.0, hv - mv))
        cols.append(METHOD_STYLE[m]["color"])

    xp = np.arange(len(methods_m))
    ax0.bar(xp, cap_means, color=cols, edgecolor="black", linewidth=0.45, yerr=[cap_lo, cap_hi], capsize=2.0)
    ax0.axhline(oracle_mean, color="black", ls="--", lw=1.3, label=f"Exact Oracle ({oracle_mean:.1f}%)")
    ax0.set_xticks(xp)
    ax0.set_xticklabels(short_m, rotation=26, ha="right")
    ax0.set_ylim(0, 108)
    ax0.set_ylabel("On-Time Completion Rate (%)")
    ax0.set_title("(a) 100 Micro-Instances vs. Exact Oracle", pad=34)
    ax0.grid(axis="y", alpha=0.25, linestyle="--")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=True, framealpha=0.95, fontsize=6.6)

    ce_order = ["L1_PreBatch", "L2_NonDelay", "L3_MyopicEDF", "CE_2ndOrder"]
    ce_labels = ["L1\nPre-Batch", "L2\nNon-Delay", "L3\nMyopic EDF", "2nd-Order\nCoupled"]
    ce_methods = ["Exact_Oracle", "CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "SFA+CG", "DGOLB+CG"]
    xc = np.arange(len(ce_order))
    bw = 0.80 / len(ce_methods)

    for idx, m in enumerate(ce_methods):
        st = METHOD_STYLE[m]
        ys = []
        for ce in ce_order:
            sub = df_ce[(df_ce["counterexample"] == ce) & (df_ce["method"] == m)]
            ys.append(float(sub["ontime_rate"].iloc[0]) * 100.0)
        xpos = xc + (idx - (len(ce_methods) - 1) / 2.0) * bw
        ax1.bar(xpos, ys, width=bw, color=st["color"], edgecolor="black", linewidth=0.4, label=st["label"])

    ax1.set_xticks(xc)
    ax1.set_xticklabels(ce_labels, fontsize=7.0)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Completion Rate (%)")
    ax1.set_title("(b) Minimal Structural Counterexamples", pad=34)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
               frameon=True, framealpha=0.95, fontsize=5.6, columnspacing=0.6, handletextpad=0.3)

    _save_fig(fig, "fig3_oracle_and_ce2")


# --------------------------------------------------------------------------- #
# Figure 4: Cross-Baseline Causal Grafting & White-Box Runtime X-Ray
# --------------------------------------------------------------------------- #

def plot_fig4() -> None:
    df_g = pd.read_csv(DATA_DIR / "b3a_grafting.csv")
    df_x = pd.read_csv(DATA_DIR / "b3c_xray_traces.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.45), dpi=300)
    fig.subplots_adjust(wspace=0.32, top=0.75)

    sub_g = df_g[df_g["group_type"] == "Main_K12"]
    bases = ["Bambos_ALP", "Chen_Game", "DGOLB", "EDF_PreBatch"]
    base_lbls = ["Bambos ALP", "Chen Game", "DGOLB", "EDF Pre-Batch"]
    modes = ["raw", "cg", "cg_plus_fluid_contract", "cg_plus_full_core"]
    mode_cfg = {
        "raw": ("Raw Baseline", "#999999"),
        "cg": ("+Common Guard", "#E69F00"),
        "cg_plus_fluid_contract": ("+Fluid Contract", "#56B4E9"),
        "cg_plus_full_core": ("+Full Core", "#0072B2"),
    }
    xb = np.arange(len(bases))
    bw = 0.80 / len(modes)

    for idx, md in enumerate(modes):
        mlbl, mcol = mode_cfg[md]
        means, lo_e, hi_e = [], [], []
        for b in bases:
            vals = sub_g[(sub_g["baseline"] == b) & (sub_g["mode"] == md)]["ontime_mean"].to_numpy() * 100.0
            mv, lv, hv = bootstrap_ci95(vals, n_boot=5000, seed=6040)
            means.append(mv)
            lo_e.append(max(0.0, mv - lv))
            hi_e.append(max(0.0, hv - mv))
        xpos = xb + (idx - (len(modes) - 1) / 2.0) * bw
        ax0.bar(xpos, means, width=bw, color=mcol, edgecolor="black", linewidth=0.4,
                yerr=[lo_e, hi_e], capsize=1.5, error_kw={"elinewidth": 0.7}, label=mlbl)

    ax0.set_xticks(xb)
    ax0.set_xticklabels(base_lbls, rotation=16, ha="right")
    ax0.set_ylim(0, 108)
    ax0.set_ylabel("Mean On-Time Rate (%)")
    ax0.set_title("(a) Cross-Baseline Causal Grafting (K=12)", pad=34)
    ax0.grid(axis="y", alpha=0.25, linestyle="--")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=True, framealpha=0.95, fontsize=6.1, columnspacing=0.8)

    wk_tr = df_x[df_x["trace_id"] == "WK02_DENSE_TIGHT_S0"]
    t_s = wk_tr["time_s"].to_numpy()
    v_calls = wk_tr["verify_calls"].to_numpy()
    t2_bnd = wk_tr["thm2_bound"].to_numpy()
    max_eq = wk_tr["max_E_q"].to_numpy()

    l1 = ax1.plot(t_s, v_calls, color="#0072B2", marker="o", markersize=3.0, lw=1.6, label="Actual Checks")
    l2 = ax1.plot(t_s, t2_bnd, color="#D55E00", ls="--", lw=1.4, label="Thm. 2 Bound")
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Fluid Verifier Calls / Tick", color="#0072B2")
    ax1.set_ylim(0, 48.0)
    ax1.grid(True, alpha=0.25, linestyle="--")

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    l3 = ax1_r.plot(t_s, max_eq, color="#009E73", marker="^", markersize=3.0, ls="-.", lw=1.3,
                    label=r"Peak $\max E_q$")
    ax1_r.axhline(1.0, color="#009E73", ls=":", lw=0.9, alpha=0.7)
    ax1_r.set_ylabel(r"Peak Externality $\max E_q$", color="#009E73")
    ax1_r.set_ylim(-0.15, 5.2)

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
               frameon=True, framealpha=0.95, fontsize=5.8, columnspacing=0.6, handletextpad=0.3)
    ax1.set_title("(b) White-Box Runtime Telemetry (Dense Tight)", pad=34)

    # Inset cleanly above Thm. 2 bound with unclipped peak pulse annotation
    ce_tr = df_x[df_x["trace_id"] == "CE_2ndOrder"].reset_index(drop=True)
    ax_in = ax1.inset_axes([0.56, 0.65, 0.42, 0.30], zorder=5)
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.97)
    eq_bars = ce_tr["eq_task_1_0"].to_numpy()
    eq_ticks = ce_tr["tick_idx"].to_numpy()
    y_peak = float(np.max(eq_bars)) if len(eq_bars) else 1.5
    y_in_top = max(1.6, y_peak * 1.32)
    ax_in.bar(eq_ticks, eq_bars, color="#CC79A7", edgecolor="black", lw=0.4)
    ax_in.axhline(1.0, color="red", ls="--", lw=0.9)
    p_idx = int(np.argmax(eq_bars))
    ax_in.text(float(eq_ticks[p_idx]), y_peak + 0.04 * y_in_top, f"{y_peak:.2f}",
               ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#333333")
    ax_in.set_xticks(eq_ticks)
    ax_in.set_ylim(0.0, y_in_top)
    ax_in.set_xlabel("Tick", fontsize=5.5, labelpad=1)
    ax_in.set_ylabel(r"$E_{1:0}$", fontsize=5.5, labelpad=1)
    ax_in.tick_params(labelsize=5.2)
    ax_in.set_title("2nd-Order Pulse", fontsize=6.0, pad=2)

    _save_fig(fig, "fig4_whitebox_xray_and_grafting")


# --------------------------------------------------------------------------- #
# Figure 5: 5-Knob Dual-Extreme Sensitivity & Operating Point
# --------------------------------------------------------------------------- #

def plot_fig5() -> None:
    df = pd.read_csv(DATA_DIR / "b3d_sensitivity_10s.csv")
    df_ce = pd.read_csv(DATA_DIR / "b1d_counterexamples.csv")
    l3_oracle = int(df_ce[(df_ce["counterexample"] == "L3_MyopicEDF") & (df_ce["method"] == "Exact_Oracle")]["n_ontime"].iloc[0])
    ce2_oracle = int(df_ce[(df_ce["counterexample"] == "CE_2ndOrder") & (df_ce["method"] == "Exact_Oracle")]["n_ontime"].iloc[0])

    knob_meta = [
        ("H / tick_s", r"(a) Horizon Multiplier $H / \Delta t$"),
        ("beta_act", r"(b) Active-Link Slack Floor $\beta_{\mathrm{act}}$"),
        ("tau_ext", r"(c) Externality Threshold $\tau_{\mathrm{ext}}$"),
        ("alpha_shield", r"(d) Timing Shield Scale $\alpha_{\mathrm{shield}}$"),
        ("k_fill", r"(e) Spatial-Reuse Fill Budget $K_{\mathrm{fill}}$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.5), dpi=300)
    fig.subplots_adjust(hspace=0.44, wspace=0.40)
    axes_flat = list(axes.flat)

    for idx, (kw_key, title) in enumerate(knob_meta):
        ax = axes_flat[idx]
        sub = df[df["knob"] == kw_key].reset_index(drop=True)
        xi = np.arange(len(sub))
        ym = sub["ontime_mean_pct"].to_numpy()
        ylo = sub["ontime_ci95_lo_pct"].to_numpy()
        yhi = sub["ontime_ci95_hi_pct"].to_numpy()
        bl01 = sub["bl01_pct"].to_numpy()
        ce2_ok = (sub["ce2_ontime"] >= ce2_oracle) & (sub["l3_ontime"] >= l3_oracle)

        for pos in range(len(sub)):
            if not bool(ce2_ok.iloc[pos]):
                ax.axvspan(pos - 0.45, pos + 0.45, color="#FDE0DD", alpha=0.65, zorder=0)

        ax.plot(xi, ym, color="#0072B2", marker="o", markersize=3.2, lw=1.6, label="Stress Suite Mean (%)")
        ax.fill_between(xi, ylo, yhi, color="#0072B2", alpha=0.16)
        ax.plot(xi, bl01, color="#009E73", marker="s", markersize=2.6, ls="--", lw=1.1, label="Mixed AR Blind (%)")

        def_rows = sub[sub["is_default"] == True]  # noqa: E712
        if not def_rows.empty:
            d_pos = int(def_rows.index[0])
            d_val = float(def_rows["ontime_mean_pct"].iloc[0])
            ax.scatter([d_pos], [d_val], marker="*", s=95, color="#D55E00", edgecolor="black", linewidth=0.5,
                       zorder=8, label="Derived Operating Point")

        ax.set_xticks(xi)
        ax.set_xticklabels(sub["value_str"].tolist(), rotation=28, ha="right", fontsize=6.4)
        y_min = max(40.0, float(min(np.min(ylo), np.min(bl01))) - 5.0)
        ax.set_ylim(y_min, 101.0)
        ax.set_ylabel("On-Time (%)")
        ax.set_title(title, fontsize=8.0)
        ax.grid(True, alpha=0.25, linestyle="--")

    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ce_patch = Patch(facecolor="#FDE0DD", edgecolor="#E0A09A", label="Counterexample Failure Zone")
    ax_leg.legend(handles + [ce_patch], labels + ["Counterexample Failure Zone"],
                  loc="center", frameon=True, framealpha=0.95, fontsize=7.0)
    ax_leg.set_title("(f) Legend & Operating Regime", fontsize=8.0)

    _save_fig(fig, "fig5_sensitivity_operating_point")


# --------------------------------------------------------------------------- #
# Figure 6: Formal 10-Seed 2D Mechanism Regime Map
# --------------------------------------------------------------------------- #

def plot_fig6() -> None:
    df = pd.read_csv(DATA_DIR / "b3e_regime_map_10s.csv")
    n_r = int(df["r_idx"].max()) + 1
    n_c = int(df["c_idx"].max()) + 1

    kappa_grid = np.zeros((n_r, n_c), dtype=float)
    rho_grid = np.zeros((n_r, n_c), dtype=float)
    diff_alp = np.zeros((n_r, n_c), dtype=float)
    diff_dgolb = np.zeros((n_r, n_c), dtype=float)

    for _, row in df.iterrows():
        ri, ci = int(row["r_idx"]), int(row["c_idx"])
        kappa_grid[ri, ci] = float(row["kappa_ul_mean"])
        rho_grid[ri, ci] = float(row["rho_cmp_mean"])
        diff_alp[ri, ci] = float(row["diff_bambos_pp"])
        diff_dgolb[ri, ci] = float(row["diff_dgolb_pp"])

    k_rows = np.mean(kappa_grid, axis=1)
    r_cols = np.mean(rho_grid, axis=0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.05), dpi=300)
    fig.subplots_adjust(wspace=0.34)

    vmin_a = min(-1.0, float(np.min(diff_alp)))
    vmax_a = max(1.0, float(np.max(diff_alp)))
    norm_a = TwoSlopeNorm(vmin=vmin_a, vcenter=0.0, vmax=vmax_a)
    cf0 = ax0.contourf(r_cols, k_rows, diff_alp, levels=14, cmap="RdBu_r", norm=norm_a)
    ax0.contour(r_cols, k_rows, diff_alp, levels=[0.0, 4.0, 10.0], colors="black", linewidths=0.7, linestyles="--")
    fig.colorbar(cf0, ax=ax0, label="Gain vs. ALP + CG (pp)")
    ax0.set_title("(a) Gain vs. Bambos ALP + CG")

    cf1 = ax1.contourf(r_cols, k_rows, diff_dgolb, levels=14, cmap="viridis")
    ax1.contour(r_cols, k_rows, diff_dgolb, levels=[15.0, 40.0, 65.0], colors="white", linewidths=0.7, linestyles="--")
    fig.colorbar(cf1, ax=ax1, label="Gain vs. DGOLB + CG (pp)")
    ax1.set_title("(b) Gain vs. DGOLB + CG")

    x_lo, x_hi = float(min(r_cols)), float(max(r_cols))
    y_hi = float(max(k_rows))
    for ax_i, ax in enumerate((ax0, ax1)):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_hi + 0.03)
        ax.axhline(0.12, color="#1f77b4" if ax_i == 0 else "#ffffff", ls=":", lw=1.4,
                   label=r"Zone I ($\kappa_{\mathrm{UL}}=0.12$)")
        ax.axvline(0.90, color="#2ca02c" if ax_i == 0 else "#ffdd57", ls="-.", lw=1.4,
                   label=r"Zone II ($\rho_{\mathrm{cmp}}=0.90$)")
        ax.set_xlabel(r"Normalized Compute Load $\rho_{\mathrm{cmp}}$")
        ax.set_ylabel(r"Uplink Coupling Index $\kappa_{\mathrm{UL}}$")
        leg_loc = "lower right" if ax_i == 0 else "upper left"
        leg = ax.legend(loc=leg_loc, frameon=True, framealpha=0.92, fontsize=6.4)
        if ax_i == 1:
            leg.get_frame().set_facecolor("#222536")
            for txt in leg.get_texts():
                txt.set_color("white")

    ax1.text(x_lo + 0.04, 0.04, "Zone I: Weak Coupling", color="white", fontsize=6.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="#1f1f2e", alpha=0.65, edgecolor="none"))
    ax1.text(x_lo + 0.06, 0.42, "Zone III: Coupled\nPhase Regime", color="white", fontsize=6.8, fontweight="bold")

    _save_fig(fig, "fig6_regime_map_2d")


# --------------------------------------------------------------------------- #
# Figure 7: Azure Functions 2019 Trace Replay & Cross-Hardware / CSI Robustness
# --------------------------------------------------------------------------- #

def plot_fig7() -> None:
    df_az = pd.read_csv(DATA_DIR / "b4a_azure_trace.csv")
    b4_sum = json.loads((STATS_DIR / "b4_summary.json").read_text(encoding="utf-8"))
    df_hw = pd.read_csv(DATA_DIR / "b4b_cross_hw.csv")
    df_pt = pd.read_csv(DATA_DIR / "b4b_perturb_sweep.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.60), dpi=300)
    fig.subplots_adjust(wspace=0.28, top=0.71)

    az_methods = ["CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "SFA+CG", "EDF_PreBatch+CG", "DGOLB+CG"]
    groups = [("FLEET", "Fleet\n(N=32)"), ("BURST", "Bursty\n(N=32)"), ("ALL", "All Traces\n(N=64)")]
    xg = np.arange(len(groups))
    bw = 0.80 / len(az_methods)
    pair_az = b4_sum["exp_4a_azure"]["pairwise_vs_cg"]

    min_az_lo = 100.0
    all_caps: list[tuple[float, float, str]] = []
    for idx, m in enumerate(az_methods):
        st = METHOD_STYLE[m]
        means, lo_e, hi_e = [], [], []
        for gk, _glbl in groups:
            sub = df_az[df_az["method"] == m] if gk == "ALL" else df_az[(df_az["stratum"] == gk) & (df_az["method"] == m)]
            mv, lv, hv = bootstrap_ci95(sub["ontime_rate"].to_numpy() * 100.0, n_boot=5000, seed=6070)
            means.append(mv)
            elo = max(0.0, mv - lv)
            ehi = max(0.0, hv - mv)
            lo_e.append(elo)
            hi_e.append(ehi)
            min_az_lo = min(min_az_lo, mv - elo)
        xpos = xg + (idx - (len(az_methods) - 1) / 2.0) * bw
        ax0.bar(xpos, means, width=bw, color=st["color"], edgecolor="black", linewidth=0.35,
                yerr=[lo_e, hi_e], capsize=1.4, error_kw={"elinewidth": 0.7}, label=st["label"])
        if m != "CLWB_Dual":
            star = _holm_stars(float(pair_az[m]["holm_p"]))
            if star != "ns":
                all_caps.append((float(xpos[-1]), float(means[-1] + hi_e[-1]), star))

    prev_ay = -1.0
    for xp_a, yc_a, st_a in all_caps:
        ys_a = yc_a + 1.5
        if prev_ay >= 0.0 and abs(ys_a - prev_ay) < 4.0:
            ys_a = prev_ay + 4.2
        ax0.text(xp_a, min(107.0, ys_a), st_a, ha="center", va="bottom", fontsize=5.2, fontweight="bold")
        prev_ay = ys_a

    ax0.set_xticks(xg)
    ax0.set_xticklabels([g[1] for g in groups])
    y_az_lo = max(0.0, 5.0 * np.floor((min_az_lo - 8.0) / 5.0))
    ax0.set_ylim(y_az_lo, 112.0)
    ax0.set_ylabel("On-Time Rate (%)")
    ax0.set_title("(a) Azure Functions 2019 Trace Replay", pad=44)
    ax0.grid(axis="y", alpha=0.25, linestyle="--")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=True, framealpha=0.95, fontsize=5.8, columnspacing=0.6)

    csi_df = df_pt[df_pt["sweep_type"] == "csi_sigma_db"]
    for m in ("CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "DGOLB+CG"):
        st = METHOD_STYLE[m]
        sub = csi_df[csi_df["method"] == m]
        xs = sorted(sub["param_val"].unique())
        ym, ylo, yhi = [], [], []
        for xv in xs:
            mv, lv, hv = bootstrap_ci95(sub[sub["param_val"] == xv]["ontime_rate"].to_numpy() * 100.0, n_boot=5000, seed=6075)
            ym.append(mv)
            ylo.append(lv)
            yhi.append(hv)
        ax1.plot(xs, ym, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                 markersize=3.6, label=st["label"])
        ax1.fill_between(xs, ylo, yhi, color=st["color"], alpha=0.12)

    ax1.axvline(2.0, color="#0072B2", ls=":", lw=1.0)
    ax1.text(2.08, 10.0, "2 dB Nominal", fontsize=6.0, color="#0072B2")
    ax1.axvline(6.0, color="#D55E00", ls=":", lw=1.0)
    ax1.text(5.88, 10.0, "6 dB Limit", fontsize=6.0, color="#D55E00", ha="right")
    ax1.set_xlabel(r"CSI Estimation Error $\sigma_{\mathrm{CSI}}$ (dB)")
    ax1.set_ylabel("On-Time Rate (%)")
    ax1.set_ylim(8, 105)
    ax1.set_title("(b) CSI Error & Multi-Hardware Profiles", pad=44)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=True, framealpha=0.95, fontsize=5.8, columnspacing=0.6)

    # Inset cleanly inside empty band [26.4..46.8%] between DGOLB (20%) and Chen (52% CI)
    ax_in = ax1.inset_axes([0.38, 0.19, 0.58, 0.21])
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.97)
    hws = [("HW1_Macro5G", "5G"), ("HW2_WiFi6_MicroMEC", "WiFi6"), ("HW3_SmallCell_DualSB", "DualSB")]
    xhw = np.arange(len(hws))
    for i_m, m in enumerate(("CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG")):
        st = METHOD_STYLE[m]
        vals_hw = [float(df_hw[(df_hw["hw_profile"] == h[0]) & (df_hw["method"] == m)]["ontime_rate"].mean() * 100.0) for h in hws]
        ax_in.bar(xhw + (i_m - 1) * 0.25, vals_hw, width=0.25, color=st["color"], edgecolor="black", lw=0.3)
    ax_in.set_xticks(xhw)
    ax_in.set_xticklabels([h[1] for h in hws], fontsize=5.6)
    ax_in.set_ylim(60, 105)
    ax_in.tick_params(axis="y", labelsize=5.2)
    ax_in.set_title("3 Hardware Profiles (%)", fontsize=5.6, pad=1.5)
    ax_in.grid(axis="y", alpha=0.25, linestyle=":")

    _save_fig(fig, "fig7_azure_trace_and_robustness")


# --------------------------------------------------------------------------- #
# Figure 8: Scalability P<=200 & Theorem Premise / Honest Boundaries
# --------------------------------------------------------------------------- #

def plot_fig8() -> None:
    df_sc = pd.read_csv(DATA_DIR / "b4c_scalability.csv")
    df_bd = pd.read_csv(DATA_DIR / "b4d_theorem_boundaries.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.6, 3.25), dpi=300)
    fig.subplots_adjust(wspace=0.30)

    for m in ("CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "DGOLB+CG", "SFA+CG"):
        st = METHOD_STYLE[m]
        sub = df_sc[df_sc["method"] == m]
        ps = sorted(sub["P"].unique())
        ym, ylo, yhi = [], [], []
        for pv in ps:
            mv, lv, hv = bootstrap_ci95(sub[sub["P"] == pv]["decision_p99_ms"].to_numpy(), n_boot=5000, seed=6080)
            ym.append(mv)
            ylo.append(lv)
            yhi.append(hv)
        ax0.plot(ps, ym, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                 markersize=3.5, label=st["label"])
        ax0.fill_between(ps, ylo, yhi, color=st["color"], alpha=0.12)

    ax0.axhline(50.0, color="red", ls="--", lw=1.2)
    ax0.text(65, 62.0, "50 ms Control Budget", color="red", fontsize=6.2, ha="center")
    ax0.set_yscale("log")
    ax0.set_ylim(0.05, 6e4)
    ax0.set_xlabel(r"Concurrent Burst Size $P$")
    ax0.set_ylabel("P99 Decision Latency (ms)")
    ax0.set_title(r"(a) Wall-Clock Scalability ($P \leq 200$)")
    ax0.grid(True, alpha=0.25, linestyle="--")
    ax0.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=5.6)

    # Inset placed cleanly above 50ms budget line with upper-left legend clear of curve
    ax_in = ax0.inset_axes([0.56, 0.58, 0.41, 0.25], zorder=5)
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.98)
    sub_c = df_sc[df_sc["method"] == "CLWB_Dual"]
    ps_c = sorted(sub_c["P"].unique())
    v_max = [float(sub_c[sub_c["P"] == pv]["max_tick_verify_calls"].max()) for pv in ps_c]
    t2_b = [float(sub_c[sub_c["P"] == pv]["thm2_bound"].max()) for pv in ps_c]
    ax_in.plot(ps_c, v_max, color="#0072B2", marker="o", markersize=2.5, lw=1.2, label="Actual")
    ax_in.plot(ps_c, t2_b, color="#D55E00", ls="--", lw=1.1, label="Thm. 2")
    ax_in.set_ylim(0.0, max(t2_b) * 1.32)
    ax_in.set_xlabel(r"$P$", fontsize=5.6, labelpad=1)
    ax_in.set_ylabel("Checks", fontsize=5.6, labelpad=1)
    ax_in.tick_params(labelsize=5.2)
    ax_in.set_title("Checks vs. Thm. 2 Bound", fontsize=5.8, pad=2)
    ax_in.legend(loc="upper left", ncol=2, fontsize=4.6, columnspacing=0.45,
                 handlelength=1.1, frameon=True, framealpha=0.92, borderpad=0.2)

    t3_df = df_bd[df_bd["category"] == "curve2_thm3_deadline_dispersion"].sort_values("x_val")
    b4_sum = json.loads((STATS_DIR / "b4_summary.json").read_text(encoding="utf-8"))
    f3_info = b4_sum["exp_4d_boundaries"]["fail_archive"]["FAIL-3_ZONE1_WEAK_AND_INVERTED_DL"]
    xv = t3_df["x_val"].to_numpy()
    yv = t3_df["metric_val"].to_numpy() * 100.0
    ylo_b = t3_df["ci_lo"].to_numpy() * 100.0
    yhi_b = t3_df["ci_hi"].to_numpy() * 100.0
    ax1.plot(xv, yv, color="#0072B2", marker="o", markersize=3.8, lw=1.8, label="Exact Single-Tick Oracle Match (%)")
    ax1.fill_between(xv, ylo_b, yhi_b, color="#0072B2", alpha=0.15)
    ax1.set_xlabel(r"Deadline Dispersion Ratio $\eta_{\mathrm{inv}}$")
    ax1.set_ylabel("Oracle Match Rate (%)")
    ax1.set_ylim(25, 112)
    ax1.set_title("(b) Premise Breakdown & Failure Boundaries")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.0)

    # Inset placed cleanly above the diagonal curve CI upper bound in upper-right empty space
    fail_df = df_bd[df_bd["category"] == "curve3_fail_archive"]
    ax_f = ax1.inset_axes([0.45, 0.62, 0.52, 0.34], zorder=5)
    ax_f.set_facecolor("white")
    ax_f.patch.set_alpha(0.98)
    fail_items = [
        ("FAIL-1_CSI_6DB", "CLWB_Dual_0dB", "0 dB"),
        ("FAIL-1_CSI_6DB", "CLWB_Dual_6dB", "6 dB"),
        ("FAIL-2_SUBTICK_DEV", "CLWB_Dual_SingleDevBurst", "Sub-Tick"),
        ("FAIL-3_ZONE1_WEAK", "CLWB_Dual_Ortho", "Zone I"),
    ]
    f_vals, f_lbls, f_tags = [], [], []
    clwb_p50 = float(f3_info["clwb_p50_ms"])
    alp_p50 = max(float(f3_info["alp_p50_ms"]), 1e-6)
    for cat, m_k, lbl in fail_items:
        row_f = fail_df[(fail_df["x_label"] == cat) & (fail_df["method"] == m_k)]
        val_pct = float(row_f["metric_val"].iloc[0]) * 100.0
        f_vals.append(val_pct)
        f_lbls.append(lbl)
        if lbl == "Zone I":
            f_tags.append(f"{clwb_p50 / alp_p50:.1f}x P50\n({clwb_p50:.2f}ms)")
        else:
            f_tags.append(f"{val_pct:.1f}%")
    f_cols = ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]
    ax_f.bar(np.arange(len(f_vals)), f_vals, color=f_cols, edgecolor="black", lw=0.4)
    for bi, (bv, btag) in enumerate(zip(f_vals, f_tags)):
        ax_f.text(bi, bv + 3.0, btag, ha="center", va="bottom", fontsize=4.4, color="#222222")
    ax_f.set_xticks(np.arange(len(f_vals)))
    ax_f.set_xticklabels(f_lbls, fontsize=5.3)
    ax_f.set_ylim(0, 138)
    ax_f.set_ylabel("On-Time (%)", fontsize=5.5, labelpad=1)
    ax_f.tick_params(axis="y", labelsize=5.2)
    ax_f.set_title("Honest Failure Boundaries", fontsize=5.9, pad=2)
    ax_f.grid(axis="y", alpha=0.25, linestyle=":")

    _save_fig(fig, "fig8_scalability_and_boundaries")


def main() -> None:
    verify_upstream_readonly()
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    print("[Plot] Successfully regenerated all 8 figures (.pdf + .png) in", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
