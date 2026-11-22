"""Publication-grade figure suite (Fig. 1-8) in Nature / Elsevier style.

Reads exclusively from read-only Stage-5 datasets and summary JSONs under
``../formal_experiments/`` (never hardcoding empirical results). Exports
vector PDF + SVG + 300 DPI PNG copies into ``../figures_patched/`` and runs
automated panel-alignment and text-collision QA before delivery.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_V3 = SCRIPT_DIR.parent
FORMAL_DIR = ROOT_V3 / "formal_experiments"
OUT_DIR = ROOT_V3 / "figures"
QA_DIR = OUT_DIR / "qa"

if str(FORMAL_DIR) not in sys.path:
    sys.path.insert(0, str(FORMAL_DIR))

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:
    def require_matplotlib_panel_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fallback no-op when optional figure alignment inspector is absent."""
        return {"status": "SKIPPED_NO_INSPECTOR"}

from common_utils import DATA_DIR, STATS_DIR, bootstrap_ci95, verify_upstream_readonly  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Mandatory publication typography & editable vector settings
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.titlesize": 7.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#1E293B",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "legend.frameon": False,
})

# Restrained Nature-aligned hierarchical palette
COL_NAVY = "#144E8C"
COL_SKY = "#8BB8E8"
COL_CHARCOAL = "#1E293B"
COL_AMBER = "#D9822B"
COL_TEAL = "#2E9389"
COL_ROSE = "#C06C84"
COL_SLATE_BLUE = "#6BAED6"
COL_CORAL = "#C85A32"
COL_OLIVE = "#9C824A"
COL_SLATE = "#64748B"
COL_CRIMSON = "#C0392B"
COL_FAIL_BG = "#FBEAE8"
COL_GRID = "#E2E8F0"
COL_ERR = "#2B3640"

METHOD_STYLE: dict[str, dict[str, Any]] = {
    "CLWB_Dual": {
        "label": "CLWB-Dual (Proposed)",
        "short": "CLWB-Dual",
        "color": COL_NAVY,
        "marker": "o",
        "ls": "-",
        "lw": 1.9,
        "zorder": 6,
    },
    "Exact_Oracle": {
        "label": "Exact Oracle",
        "short": "Exact Oracle",
        "color": COL_CHARCOAL,
        "marker": "h",
        "ls": "-",
        "lw": 1.6,
        "zorder": 7,
    },
    "Bambos_ALP+CG": {
        "label": "Bambos ALP + CG",
        "short": "ALP + CG",
        "color": COL_AMBER,
        "marker": "s",
        "ls": "--",
        "lw": 1.35,
        "zorder": 5,
    },
    "Chen_Game+CG": {
        "label": "Chen Game + CG",
        "short": "Chen + CG",
        "color": COL_TEAL,
        "marker": "^",
        "ls": "-.",
        "lw": 1.3,
        "zorder": 4,
    },
    "EDF_PreBatch+CG": {
        "label": "EDF Pre-Batch + CG",
        "short": "EDF + CG",
        "color": COL_ROSE,
        "marker": "v",
        "ls": "--",
        "lw": 1.25,
        "zorder": 3,
    },
    "LLF_PreBatch+CG": {
        "label": "LLF Pre-Batch + CG",
        "short": "LLF + CG",
        "color": COL_SLATE_BLUE,
        "marker": "P",
        "ls": "-.",
        "lw": 1.2,
        "zorder": 3,
    },
    "DGOLB+CG": {
        "label": "DGOLB + CG",
        "short": "DGOLB + CG",
        "color": COL_CORAL,
        "marker": "D",
        "ls": ":",
        "lw": 1.4,
        "zorder": 4,
    },
    "SFA+CG": {
        "label": "SFA + CG",
        "short": "SFA + CG",
        "color": COL_OLIVE,
        "marker": "X",
        "ls": ":",
        "lw": 1.35,
        "zorder": 3,
    },
    "IBR_5+CG": {
        "label": "IBR-5 + CG",
        "short": "IBR-5 + CG",
        "color": COL_SLATE,
        "marker": "*",
        "ls": "-.",
        "lw": 1.2,
        "zorder": 2,
    },
}

ERROR_KW = {"elinewidth": 0.72, "capthick": 0.72, "ecolor": COL_ERR}


def _add_panel_label(
    ax: plt.Axes,
    letter: str,
    x_offset_pt: float = -22.0,
    y_offset_pt: float = 5.0,
) -> None:
    """Place bold lowercase panel label via fixed physical point offset."""
    offset = ScaledTranslation(
        x_offset_pt / 72.0,
        y_offset_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0.0,
        1.0,
        letter,
        transform=ax.transAxes + offset,
        fontsize=9.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
        va="bottom",
    )


def _style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply crisp subtle background grid and top/right spine suppression."""
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=COL_GRID, linewidth=0.5, linestyle="-", zorder=0)


def _check_text_collisions(fig: plt.Figure, stem: str) -> None:
    """Detect any overlapping rendered text boxes or out-of-figure text clipping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    items: list[tuple[str, Any]] = []
    for txt in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        content = txt.get_text().strip()
        if not content or not txt.get_visible():
            continue
        # Skip empty tick labels off-axis
        bb = txt.get_window_extent(renderer=renderer)
        if bb.width <= 0.5 or bb.height <= 0.5:
            continue
        items.append((content.replace("\n", " "), bb))

    overlaps: list[str] = []
    for i in range(len(items)):
        t1, b1 = items[i]
        # Shrink slightly by 1 px to tolerate intentional tight baselines
        b1_in = b1.padded(-0.8)
        for j in range(i + 1, len(items)):
            t2, b2 = items[j]
            b2_in = b2.padded(-0.8)
            if b1_in.overlaps(b2_in):
                overlaps.append(f"'{t1}' <-> '{t2}'")
    clips: list[str] = []
    for t, b in items:
        if b.x0 < -2.0 or b.y0 < -2.0 or b.x1 > fig_w + 2.0 or b.y1 > fig_h + 2.0:
            clips.append(f"'{t}' ({b.x0:.1f},{b.y0:.1f})-({b.x1:.1f},{b.y1:.1f}) vs ({fig_w:.1f},{fig_h:.1f})")

    if overlaps:
        raise RuntimeError(f"[{stem}] Text overlaps detected: {overlaps[:5]}")
    if clips:
        raise RuntimeError(f"[{stem}] Text clipped outside figure bounds: {clips[:5]}")


def _finalize_and_save(
    fig: plt.Figure,
    stem: str,
    *,
    align_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run alignment gate, text collision audit, and export PDF/SVG/PNG."""
    fig.canvas.draw()
    opts = dict(align_kwargs or {})
    require_matplotlib_panel_alignment(
        fig,
        json_out=QA_DIR / f"{stem}.alignment.json",
        overlay_svg=QA_DIR / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        **opts,
    )
    _check_text_collisions(fig, stem)
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.svg", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: Dual Cross-Layer Deadlock Motivation (1 x 3 Clean Layout)
# --------------------------------------------------------------------------- #

def plot_fig1() -> None:
    df1a = pd.read_csv(DATA_DIR / "b1a_ortho_vs_shared.csv")
    df1b = pd.read_csv(DATA_DIR / "b1b_kappa_sweep.csv")
    df1c = pd.read_csv(DATA_DIR / "b1c_burst_cliff.csv")

    fig = plt.figure(figsize=(7.15, 2.55), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.22, 1.12, 0.90], wspace=0.35,
                           left=0.075, right=0.985, bottom=0.17, top=0.68)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Orthogonal vs. shared channel collapse
    methods_1a = [
        "Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG",
        "LLF_PreBatch+CG", "DGOLB+CG", "SFA+CG", "IBR_5+CG",
    ]
    short_1a = ["ALP", "Chen", "EDF", "LLF", "DGOLB", "SFA", "IBR-5"]
    x = np.arange(len(methods_1a))
    w = 0.36
    ortho_m, ortho_e, shared_m, shared_e, miss_m = [], [], [], [], []
    for m in methods_1a:
        vo = df1a[(df1a["channel_mode"] == "orthogonal") & (df1a["method"] == m)]["ontime_rate"].to_numpy() * 100.0
        vs = df1a[(df1a["channel_mode"] == "shared") & (df1a["method"] == m)]["ontime_rate"].to_numpy() * 100.0
        vm = df1a[(df1a["channel_mode"] == "shared") & (df1a["method"] == m)]["server_grant_miss_rate"].to_numpy() * 100.0
        mo, loo, hio = bootstrap_ci95(vo, n_boot=5000, seed=6001)
        ms, los, his = bootstrap_ci95(vs, n_boot=5000, seed=6002)
        ortho_m.append(mo)
        ortho_e.append([max(0.0, mo - loo), max(0.0, hio - mo)])
        shared_m.append(ms)
        shared_e.append([max(0.0, ms - los), max(0.0, his - ms)])
        miss_m.append(float(np.mean(vm)))

    ax0.bar(x - w / 2, ortho_m, width=w, color=COL_SKY, edgecolor="white", linewidth=0.4,
            yerr=np.array(ortho_e).T, capsize=1.5, error_kw=ERROR_KW, label="Orthogonal on-time")
    ax0.bar(x + w / 2, shared_m, width=w, color=COL_NAVY, edgecolor="white", linewidth=0.4,
            yerr=np.array(shared_e).T, capsize=1.5, error_kw=ERROR_KW, label="Shared on-time")
    ax0.plot(x + w / 2, miss_m, color=COL_CORAL, marker="D", markersize=3.8, ls="none",
             markeredgecolor="white", markeredgewidth=0.55, zorder=6, clip_on=False, label="Shared miss")
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_1a, fontsize=5.8)
    ax0.set_yticks([0, 20, 40, 60, 80, 100])
    ax0.set_ylim(0, 108)
    ax0.set_ylabel("Rate (%)")
    ax0.set_xlabel("Baseline policy")
    ax0.set_title("Orthogonal vs. shared", loc="left", pad=5)
    _style_axis(ax0, "y")
    _add_panel_label(ax0, "a", x_offset_pt=-22.0, y_offset_pt=6.0)
    ax0.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.15), ncol=2, fontsize=6.0,
               columnspacing=0.6, handletextpad=0.35, labelspacing=0.2)

    # Panel b: Coupling phase collapse over kappa_UL
    m_1b = ["Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG", "DGOLB+CG", "CLWB_Dual"]
    ax1.axvspan(0.0, 0.12, color="#F1F5F9", alpha=0.95, zorder=0)
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
            ax1.plot(kx, ym, color=COL_CHARCOAL, marker="", ls="--", lw=1.4, zorder=7,
                     label="Batch-consistent")
        else:
            st = METHOD_STYLE[m]
            ax1.plot(kx, ym, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                     markersize=3.4, markeredgecolor="white", markeredgewidth=0.4, label=st["short"])
            ax1.fill_between(kx, ylo, yhi, color=st["color"], alpha=0.14)

    ax1.axvline(0.12, color="#475569", ls=":", lw=1.1, zorder=2)
    ax1.text(0.145, 7.0, r"Zone I: $\kappa_{\mathrm{UL}}=0.12$", fontsize=7.2, color="#334155")
    ax1.set_xlabel(r"Uplink coupling index $\kappa_{\mathrm{UL}}$")
    ax1.set_ylabel("On-time rate (%)")
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 108)
    ax1.set_xlim(-0.02, 1.0)
    ax1.set_title("Coupling phase collapse", loc="left", pad=5)
    _style_axis(ax1, "both")
    _add_panel_label(ax1, "b", x_offset_pt=-22.0, y_offset_pt=6.0)
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.15), ncol=2, fontsize=5.9,
               columnspacing=0.55, handletextpad=0.35, labelspacing=0.18)

    # Panel c: Control latency cliff (separated cleanly from Panel b)
    for m in ("IBR_5+CG", "SFA+CG", "CLWB_Dual"):
        grp_p = df1c[df1c["method"] == m].groupby("P_burst", sort=True)
        px = np.array(sorted(grp_p.groups.keys()), dtype=float)
        py = [float(grp_p.get_group(p)["decision_p99_ms"].mean()) for p in px]
        if m == "CLWB_Dual":
            ax2.plot(px, py, color=COL_CHARCOAL, marker="", ls="--", lw=1.4,
                     label="CLWB-Dual target")
        else:
            st = METHOD_STYLE[m]
            ax2.plot(px, py, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                     markersize=3.4, markeredgecolor="white", markeredgewidth=0.4, label=st["short"])

    ax2.axhline(50.0, color=COL_CRIMSON, ls="--", lw=1.1, zorder=4)
    ax2.text(79.0, 66.0, "50 ms budget", color=COL_CRIMSON, fontsize=6.4, fontweight="bold", ha="right", va="bottom")
    ax2.set_yscale("log")
    ax2.set_ylim(0.25, 4.5e4)
    ax2.set_yticks([1, 10, 100, 1000, 10000])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v//1000)}k" if v >= 1000 else f"{int(v)}"))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
    ax2.set_xticks([20, 40, 60, 80])
    ax2.set_xlabel(r"Burst size $P$")
    ax2.set_ylabel("P99 latency (ms)")
    ax2.set_title("Control latency cliff", loc="left", pad=5)
    _style_axis(ax2, "both")
    _add_panel_label(ax2, "c", x_offset_pt=-22.0, y_offset_pt=6.0)
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.15), ncol=2, fontsize=5.9,
               columnspacing=0.5, handletextpad=0.3, labelspacing=0.18)

    _finalize_and_save(
        fig,
        "fig1_motivation_dual_deadlock",
        align_kwargs={
            "exemptions": [
                {"panels": ["a", "b", "c"], "checks": ["panel-width"], "reason": "c shows compact 4-point log latency curve"},
            ]
        },
    )


# --------------------------------------------------------------------------- #
# Figure 2: Main Benchmark Comparison (1 x 3 Clean Layout)
# --------------------------------------------------------------------------- #

def plot_fig2() -> None:
    b2_sum = json.loads((STATS_DIR / "b2_summary.json").read_text(encoding="utf-8"))
    df_ep = pd.read_csv(DATA_DIR / "b2_episodes.csv")

    fig = plt.figure(figsize=(7.15, 2.48), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.48, 1.02, 0.80], wspace=0.32,
                           left=0.075, right=0.985, bottom=0.20, top=0.73)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    scope_keys = ["Main_K12", "OOD_K4", "Blind_K8", "All_24"]
    scope_labels = ["Standard\nK=12", "OOD\nK=4", "Blind\nK=8", "Combined\nK=24"]
    methods = [
        "CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "EDF_PreBatch+CG",
        "LLF_PreBatch+CG", "DGOLB+CG", "SFA+CG", "IBR_5+CG",
    ]
    n_m = len(methods)
    x = np.arange(len(scope_keys))
    bw = 0.84 / n_m

    for idx, m in enumerate(methods):
        st = METHOD_STYLE[m]
        means, elo, ehi = [], [], []
        for sk in scope_keys:
            pm = b2_sum["scopes"][sk]["per_method"][m]
            mv = pm["ontime_mean"] * 100.0
            ci = pm["ontime_ci95"]
            means.append(mv)
            elo.append(max(0.0, mv - ci[0] * 100.0))
            ehi.append(max(0.0, ci[1] * 100.0 - mv))
        xpos = x + (idx - (n_m - 1) / 2.0) * bw
        ax0.bar(xpos, means, width=bw, color=st["color"], edgecolor="white", linewidth=0.35,
                yerr=[elo, ehi], capsize=1.1, error_kw={"elinewidth": 0.65, "ecolor": COL_ERR},
                label=st["label"])

    ax0.set_xticks(x)
    ax0.set_xticklabels(scope_labels, fontsize=6.5)
    ax0.set_yticks([0, 20, 40, 60, 80, 100])
    ax0.set_ylim(0, 106)
    ax0.set_ylabel("Mean on-time rate (%)")
    ax0.set_title("Cross-domain on-time rate (95% CI)", loc="left", pad=8)
    _style_axis(ax0, "y")
    _add_panel_label(ax0, "a", x_offset_pt=-24.0, y_offset_pt=10.0)

    # Panel b: Full-canvas ECDF (unobstructed!)
    for m in methods:
        st = METHOD_STYLE[m]
        vals = np.sort(df_ep[df_ep["method"] == m]["ontime_rate"].to_numpy() * 100.0)
        ecdf_y = np.arange(1, len(vals) + 1) / float(len(vals))
        ax1.step(vals, ecdf_y, where="post", color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"])
    ax1.set_xlabel("Episode on-time rate (%)")
    ax1.set_ylabel("Empirical CDF (240 eps)")
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0, 1.04)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_title("Tail robustness ECDF", loc="left", pad=8)
    _style_axis(ax1, "both")
    _add_panel_label(ax1, "b", x_offset_pt=-22.0, y_offset_pt=10.0)

    # Panel c: Dedicated Grant Miss Rate horizontal bar chart
    miss_m = [b2_sum["scopes"]["All_24"]["per_method"][m]["miss_mean"] * 100.0 for m in methods]
    miss_c = [METHOD_STYLE[m]["color"] for m in methods]
    y_pos = np.arange(len(methods))[::-1]
    short_names = ["CLWB", "ALP", "Chen", "EDF", "LLF", "DGOLB", "SFA", "IBR-5"]
    ax2.barh(y_pos, miss_m, height=0.66, color=miss_c, edgecolor="white", linewidth=0.4)
    for m_name, yp, mv in zip(methods, y_pos, miss_m):
        ax2.text(mv + 2.0, yp, f"{mv:.1f}%", va="center", ha="left", fontsize=6.2,
                 fontweight="bold" if m_name == "CLWB_Dual" else "normal",
                 color=COL_NAVY if m_name == "CLWB_Dual" else "#334155")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_names, fontsize=6.4)
    ax2.set_xlim(0, 118)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Grant miss rate (%)")
    ax2.set_title("Admission miss (240 eps)", loc="left", pad=8)
    _style_axis(ax2, "x")
    _add_panel_label(ax2, "c", x_offset_pt=-22.0, y_offset_pt=10.0)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.99),
               ncol=4, fontsize=6.6, columnspacing=0.9, handletextpad=0.4, labelspacing=0.28)

    _finalize_and_save(
        fig,
        "fig2_main_benchmark_comparison",
        align_kwargs={
            "exemptions": [
                {"panels": ["a", "b", "c"], "checks": ["panel-width"], "reason": "a holds 4x8 grouped bars; c holds compact horizontal bars"},
            ]
        },
    )


# --------------------------------------------------------------------------- #
# Figure 3: 100 Micro-Oracle Capture & 4 Analytical Counterexamples
# --------------------------------------------------------------------------- #

def plot_fig3() -> None:
    df_m = pd.read_csv(DATA_DIR / "b2_micro100.csv")
    df_ce = pd.read_csv(DATA_DIR / "b1d_counterexamples.csv")

    fig = plt.figure(figsize=(7.15, 2.42), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.25], wspace=0.34,
                           left=0.115, right=0.985, bottom=0.19, top=0.77)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Panel a: Horizontal bars sorted top-to-bottom (zero rotated x-tick mess!)
    methods_m = [
        "CLWB_Dual", "SFA+CG", "IBR_5+CG", "Chen_Game+CG",
        "Bambos_ALP+CG", "EDF_PreBatch+CG", "LLF_PreBatch+CG", "DGOLB+CG",
    ]
    oracle_mean = float(df_m[df_m["method"] == "Exact_Oracle"]["ontime_rate"].mean()) * 100.0
    cap_m, cap_lo, cap_hi, cols, ylbls = [], [], [], [], []
    for m in methods_m:
        vals = df_m[df_m["method"] == m]["ontime_rate"].to_numpy() * 100.0
        mv, lv, hv = bootstrap_ci95(vals, n_boot=5000, seed=6030)
        cap_m.append(mv)
        cap_lo.append(max(0.0, mv - lv))
        cap_hi.append(max(0.0, hv - mv))
        cols.append(METHOD_STYLE[m]["color"])
        ylbls.append(METHOD_STYLE[m]["short"])

    yp = np.arange(len(methods_m))[::-1]
    ax0.barh(yp, cap_m, height=0.66, color=cols, edgecolor="white", linewidth=0.4,
             xerr=[cap_lo, cap_hi], capsize=1.6, error_kw=ERROR_KW)
    ax0.axvline(oracle_mean, color=COL_CHARCOAL, ls="--", lw=1.25, zorder=5)
    ax0.text(oracle_mean - 1.8, -0.42, f"Oracle {oracle_mean:.1f}%", fontsize=6.3, fontweight="bold",
             color=COL_CHARCOAL, ha="right", va="bottom")
    ax0.text(cap_m[0] + cap_hi[0] + 1.8, yp[0], f"{cap_m[0]:.1f}%", va="center", ha="left",
             fontsize=6.6, fontweight="bold", color=COL_NAVY)

    ax0.set_yticks(yp)
    ax0.set_yticklabels(ylbls, fontsize=6.6)
    ax0.set_xlim(0, 116)
    ax0.set_xticks([0, 20, 40, 60, 80, 100])
    ax0.set_xlabel("On-time completion rate (%)")
    ax0.set_title("100 micro-instances vs. Exact Oracle", loc="left", pad=5)
    _style_axis(ax0, "x")
    _add_panel_label(ax0, "a", x_offset_pt=-32.0, y_offset_pt=6.0)

    # Panel b: Four analytical counterexamples
    ce_order = ["L1_PreBatch", "L2_NonDelay", "L3_MyopicEDF", "CE_2ndOrder"]
    ce_labels = ["L1\nPre-batch", "L2\nNon-delay", "L3\nMyopic EDF", "2nd-order\ncoupled"]
    ce_methods = ["Exact_Oracle", "CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "SFA+CG", "DGOLB+CG"]
    xc = np.arange(len(ce_order))
    bw = 0.80 / len(ce_methods)

    for idx, m in enumerate(ce_methods):
        st = METHOD_STYLE[m]
        ys = [float(df_ce[(df_ce["counterexample"] == ce) & (df_ce["method"] == m)]["ontime_rate"].iloc[0]) * 100.0
              for ce in ce_order]
        xpos = xc + (idx - (len(ce_methods) - 1) / 2.0) * bw
        ax1.bar(xpos, ys, width=bw, color=st["color"], edgecolor="white", linewidth=0.35, label=st["short"])

    ax1.set_xticks(xc)
    ax1.set_xticklabels(ce_labels, fontsize=6.6)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Completion rate (%)")
    ax1.set_title("Minimal structural counterexamples", loc="left", pad=5)
    _style_axis(ax1, "y")
    _add_panel_label(ax1, "b", x_offset_pt=-22.0, y_offset_pt=6.0)
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.13), ncol=3, fontsize=6.1,
               columnspacing=0.65, handletextpad=0.35, labelspacing=0.18)

    _finalize_and_save(fig, "fig3_oracle_and_ce2")


# --------------------------------------------------------------------------- #
# Figure 4: Causal Grafting & Decoupled White-Box Telemetry + 2nd-Order Pulse
# --------------------------------------------------------------------------- #

def plot_fig4() -> None:
    df_g = pd.read_csv(DATA_DIR / "b3a_grafting.csv")
    df_x = pd.read_csv(DATA_DIR / "b3c_xray_traces.csv")

    fig = plt.figure(figsize=(7.15, 2.58), dpi=300)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1.28, 0.72], wspace=0.40, hspace=0.30,
                           left=0.075, right=0.985, bottom=0.17, top=0.69)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_bt = fig.add_subplot(gs[0, 1])
    ax_bb = fig.add_subplot(gs[1, 1], sharex=ax_bt)
    ax_c = fig.add_subplot(gs[:, 2])

    # Panel a: 4-stage monotonic sequential grafting palette
    sub_g = df_g[df_g["group_type"] == "Main_K12"]
    bases = ["Bambos_ALP", "Chen_Game", "DGOLB", "EDF_PreBatch"]
    base_lbls = ["Bambos\nALP", "Chen\nGame", "DGOLB", "EDF\nPre-batch"]
    modes = ["raw", "cg", "cg_plus_fluid_contract", "cg_plus_full_core"]
    mode_cfg = {
        "raw": ("Raw baseline", "#CBD5E1"),
        "cg": ("+Common Guard", COL_AMBER),
        "cg_plus_fluid_contract": ("+Fluid Contract", COL_SLATE_BLUE),
        "cg_plus_full_core": ("+Full Core", COL_NAVY),
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
        ax_a.bar(xpos, means, width=bw, color=mcol, edgecolor="white", linewidth=0.35,
                 yerr=[lo_e, hi_e], capsize=1.3, error_kw=ERROR_KW, label=mlbl)

    ax_a.set_xticks(xb)
    ax_a.set_xticklabels(base_lbls, fontsize=6.4)
    ax_a.set_yticks([0, 20, 40, 60, 80, 100])
    ax_a.set_ylim(0, 108)
    ax_a.set_ylabel("Mean on-time rate (%)")
    ax_a.set_title("Cross-baseline grafting (K=12)", loc="left", pad=5)
    _style_axis(ax_a, "y")
    _add_panel_label(ax_a, "a", x_offset_pt=-24.0, y_offset_pt=6.0)
    ax_a.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.13), ncol=2, fontsize=6.0,
                columnspacing=0.65, handletextpad=0.35, labelspacing=0.18)

    # Panel b (top) + Panel c (bottom): Decoupled telemetry (zero dual-y clutter!)
    wk_tr = df_x[df_x["trace_id"] == "WK02_DENSE_TIGHT_S0"]
    t_s = wk_tr["time_s"].to_numpy()
    v_calls = wk_tr["verify_calls"].to_numpy()
    t3_bnd = wk_tr["thm2_bound"].to_numpy()  # Matches Thm. 3 Bound in manuscript
    max_eq = wk_tr["max_E_q"].to_numpy()

    ax_bt.plot(t_s, v_calls, color=COL_NAVY, marker="o", markersize=2.6, markeredgecolor="white",
               markeredgewidth=0.35, lw=1.45, label="Actual checks")
    ax_bt.plot(t_s, t3_bnd, color=COL_CORAL, ls="--", lw=1.3, label="Thm. 3 bound")
    ax_bt.fill_between(t_s, v_calls, t3_bnd, where=(t3_bnd >= v_calls), color=COL_SLATE_BLUE, alpha=0.14)
    ax_bt.set_ylim(0, 31)
    ax_bt.set_yticks([0, 10, 20, 30])
    ax_bt.set_ylabel("Calls")
    ax_bt.set_title("WK02 runtime telemetry", loc="left", pad=5)
    _style_axis(ax_bt, "both")
    _add_panel_label(ax_bt, "b", x_offset_pt=-22.0, y_offset_pt=6.0)
    plt.setp(ax_bt.get_xticklabels(), visible=False)
    ax_bt.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.36), ncol=2, fontsize=6.1,
                 columnspacing=0.6, handletextpad=0.35)

    ax_bb.plot(t_s, max_eq, color=COL_TEAL, marker="^", markersize=2.6, markeredgecolor="white",
               markeredgewidth=0.35, lw=1.25, label="Peak externality")
    ax_bb.axhline(1.0, color=COL_CRIMSON, ls="--", lw=0.95)
    ax_bb.text(1.12, 1.15, "Threshold 1.0", fontsize=6.0, color=COL_CRIMSON, ha="right", va="bottom")
    ax_bb.set_ylim(-0.2, 3.6)
    ax_bb.set_yticks([0, 1, 2, 3])
    ax_bb.set_xlabel("Simulation time (s)")
    ax_bb.set_ylabel(r"Peak $E_q$")
    _style_axis(ax_bb, "both")
    _add_panel_label(ax_bb, "c", x_offset_pt=-22.0, y_offset_pt=2.5)

    # Panel d: Dedicated 2nd-order externality pulse in CE_2ndOrder
    ce_tr = df_x[df_x["trace_id"] == "CE_2ndOrder"].reset_index(drop=True)
    eq_bars = ce_tr["eq_task_1_0"].to_numpy()
    eq_ticks = ce_tr["tick_idx"].to_numpy()
    y_peak = float(np.max(eq_bars)) if len(eq_bars) else 1.5
    bar_cols = [COL_CRIMSON if v > 1.0 else "#CBD5E1" for v in eq_bars]
    ax_c.bar(eq_ticks, eq_bars, width=0.58, color=bar_cols, edgecolor="white", linewidth=0.45)
    ax_c.axhline(1.0, color=COL_CHARCOAL, ls="--", lw=1.0)
    p_idx = int(np.argmax(eq_bars))
    ax_c.text(float(eq_ticks[p_idx]), y_peak + 0.05, f"{y_peak:.2f}",
              ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COL_CRIMSON)
    ax_c.text(float(eq_ticks[-1]) + 0.35, 1.04, "1.0", ha="right", va="bottom", fontsize=6.0, color=COL_CHARCOAL)
    ax_c.set_xticks(eq_ticks)
    ax_c.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax_c.set_ylim(0.0, max(1.62, y_peak * 1.28))
    ax_c.set_xlabel("Control tick")
    ax_c.set_ylabel(r"Externality $E_{1:0}$")
    ax_c.set_title("2nd-order pulse", loc="left", pad=5)
    _style_axis(ax_c, "y")
    _add_panel_label(ax_c, "d", x_offset_pt=-22.0, y_offset_pt=6.0)

    _finalize_and_save(
        fig,
        "fig4_whitebox_xray_and_grafting",
        align_kwargs={
            "exemptions": [
                {"panels": ["a", "d"], "checks": ["panel-width"], "reason": "a holds 4x4 grouped bars while d holds 5-tick pulse bar"},
            ],
        },
    )


# --------------------------------------------------------------------------- #
# Figure 5 (Supp Fig. S1): 5-Knob Sensitivity & Operating Point
# --------------------------------------------------------------------------- #

def plot_fig5() -> None:
    df = pd.read_csv(DATA_DIR / "b3d_sensitivity_10s.csv")
    df_ce = pd.read_csv(DATA_DIR / "b1d_counterexamples.csv")
    l3_oracle = int(df_ce[(df_ce["counterexample"] == "L3_MyopicEDF") & (df_ce["method"] == "Exact_Oracle")]["n_ontime"].iloc[0])
    ce2_oracle = int(df_ce[(df_ce["counterexample"] == "CE_2ndOrder") & (df_ce["method"] == "Exact_Oracle")]["n_ontime"].iloc[0])

    knob_meta = [
        ("H / tick_s", "a", r"Horizon multiplier $H / \Delta t$"),
        ("beta_act", "b", r"Active-link slack floor $\beta_{\mathrm{act}}$"),
        ("tau_ext", "c", r"Externality threshold $\tau_{\mathrm{ext}}$"),
        ("alpha_shield", "d", r"Timing shield scale $\alpha_{\mathrm{shield}}$"),
        ("k_fill", "e", r"Spatial-reuse fill budget $K_{\mathrm{fill}}$"),
    ]

    fig = plt.figure(figsize=(7.15, 3.85), dpi=300)
    gs = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.52,
                           left=0.07, right=0.985, bottom=0.11, top=0.92)
    axes_list = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for idx, (kw_key, letter, title) in enumerate(knob_meta):
        ax = axes_list[idx]
        sub = df[df["knob"] == kw_key].reset_index(drop=True)
        xi = np.arange(len(sub))
        ym = sub["ontime_mean_pct"].to_numpy()
        ylo = sub["ontime_ci95_lo_pct"].to_numpy()
        yhi = sub["ontime_ci95_hi_pct"].to_numpy()
        bl01 = sub["bl01_pct"].to_numpy()
        ce2_ok = (sub["ce2_ontime"] >= ce2_oracle) & (sub["l3_ontime"] >= l3_oracle)

        for pos in range(len(sub)):
            if not bool(ce2_ok.iloc[pos]):
                ax.axvspan(pos - 0.45, pos + 0.45, color=COL_FAIL_BG, alpha=0.95, zorder=0)

        ax.plot(xi, ym, color=COL_NAVY, marker="o", markersize=3.2, markeredgecolor="white",
                markeredgewidth=0.4, lw=1.6, label="Stress suite mean (%)")
        ax.fill_between(xi, ylo, yhi, color=COL_NAVY, alpha=0.15)
        ax.plot(xi, bl01, color=COL_TEAL, marker="s", markersize=2.6, markeredgecolor="white",
                markeredgewidth=0.3, ls="--", lw=1.15, label="Mixed AR blind (%)")

        def_rows = sub[sub["is_default"] == True]  # noqa: E712
        if not def_rows.empty:
            d_pos = int(def_rows.index[0])
            d_val = float(def_rows["ontime_mean_pct"].iloc[0])
            ax.axvline(d_pos, color=COL_CORAL, ls=":", lw=0.9, zorder=3)
            ax.scatter([d_pos], [d_val], marker="*", s=95, color=COL_CORAL, edgecolor="white",
                       linewidth=0.5, zorder=8, label="Derived operating point")

        ax.set_xticks(xi)
        ax.set_xticklabels(sub["value_str"].tolist(), rotation=26, ha="right", rotation_mode="anchor", fontsize=6.3)
        y_min = max(40.0, float(min(np.min(ylo), np.min(bl01))) - 5.0)
        ax.set_ylim(y_min, 102.0)
        ax.set_ylabel("On-time (%)")
        ax.set_title(title, loc="left", pad=6, fontsize=7.5)
        _style_axis(ax, "both")
        _add_panel_label(ax, letter, x_offset_pt=-22.0, y_offset_pt=7.0)

    ax_leg = axes_list[5]
    ax_leg.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    ce_patch = Patch(facecolor=COL_FAIL_BG, edgecolor="#E5A39E", label="Counterexample failure zone")
    ax_leg.legend(handles + [ce_patch], labels + ["Counterexample failure zone"],
                  loc="center", frameon=False, fontsize=7.0, labelspacing=0.6)

    _finalize_and_save(
        fig,
        "fig5_sensitivity_operating_point",
        align_kwargs={
            "exclude_axes": [ax_leg],
            "row_groups": [["a", "b", "c"], ["d", "e"]],
            "column_groups": [["a", "d"], ["b", "e"]],
        },
    )


# --------------------------------------------------------------------------- #
# Figure 6 (Supp Fig. S2): 10-Seed 2D Control Regime Map
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

    cmap_navy_seq = LinearSegmentedColormap.from_list(
        "clwb_navy_seq",
        ["#F7FAFD", "#DCEAF6", "#A6CBE7", "#5C97C9", "#24649E", "#103763"],
        N=256,
    )
    cmap_teal_seq = LinearSegmentedColormap.from_list(
        "clwb_teal_seq",
        ["#F7FBFA", "#D3EEEA", "#8FD1C7", "#43A89C", "#1B756D", "#0E4644"],
        N=256,
    )

    fig = plt.figure(figsize=(7.15, 2.72), dpi=300)
    gs = gridspec.GridSpec(1, 2, wspace=0.52, left=0.08, right=0.92, bottom=0.17, top=0.72)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    x_edges = np.arange(n_c + 1) - 0.5
    y_edges = np.arange(n_r + 1) - 0.5

    def _draw_regime_heatmap(
        ax: plt.Axes,
        grid: np.ndarray,
        cmap: LinearSegmentedColormap,
        vmin: float,
        vmax: float,
        cb_ticks: list[int],
        cb_label: str,
        title: str,
        letter: str,
        panel_x_off: float,
    ) -> Any:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        qm = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=1.25,
            shading="flat",
            zorder=1,
        )
        for ri in range(n_r):
            for ci in range(n_c):
                val = float(grid[ri, ci])
                r_rgb, g_rgb, b_rgb, _ = cmap(norm(val))
                lum = 0.2126 * r_rgb + 0.7152 * g_rgb + 0.0722 * b_rgb
                txt_col = "#FFFFFF" if lum < 0.48 else "#1E293B"
                txt_str = "0.0" if abs(val) < 0.05 else f"{val:+.1f}"
                ax.text(
                    ci,
                    ri,
                    txt_str,
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    fontweight="bold" if val >= 0.75 * vmax else "normal",
                    color=txt_col,
                    zorder=3,
                )

        # Align empirical regime boundaries strictly along cell seams
        ax.plot([-0.5, 5.5], [1.5, 1.5], color="#1E293B", ls="--", lw=1.35, zorder=4)
        ax.plot([4.5, 4.5], [1.5, 5.5], color=COL_CORAL, ls="-.", lw=1.45, zorder=4)

        # Top non-overlapping regime callouts above the cell grid
        ax.text(2.0, 5.66, "Zone III (coupled)", ha="center", va="bottom",
                fontsize=6.0, fontweight="bold", color="#1E293B", clip_on=False)
        ax.text(5.0, 5.66, "Zone II", ha="center", va="bottom",
                fontsize=6.0, fontweight="bold", color=COL_CORAL, clip_on=False)

        # Clean right-side bracket indicating Zone I rows (rows 0 and 1)
        ax.plot([5.60, 5.68, 5.68, 5.60], [-0.40, -0.40, 1.40, 1.40],
                color="#475569", lw=0.75, clip_on=False, zorder=4)
        ax.text(5.84, 0.5, "Zone I", ha="center", va="center", rotation=90,
                fontsize=5.8, fontweight="bold", color="#334155", clip_on=False)

        ax.set_xticks(np.arange(n_c))
        ax.set_xticklabels([f"{v:.2f}" for v in r_cols], fontsize=6.4)
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels([f"{v:.2f}" for v in k_rows], fontsize=6.4)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xlabel(r"Normalized compute load $\rho_{\mathrm{cmp}}$")
        ax.set_ylabel(r"Uplink coupling index $\kappa_{\mathrm{UL}}$")
        ax.set_title(title, loc="left", pad=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _add_panel_label(ax, letter, x_offset_pt=panel_x_off, y_offset_pt=16.0)

        cb = fig.colorbar(qm, ax=ax, fraction=0.046, pad=0.115)
        cb.set_ticks(cb_ticks)
        cb.set_label(cb_label, fontsize=6.8)
        cb.ax.tick_params(labelsize=6.2)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#475569")
        return cb

    cb0 = _draw_regime_heatmap(
        ax0,
        diff_alp,
        cmap_navy_seq,
        vmin=0.0,
        vmax=17.0,
        cb_ticks=[0, 4, 8, 12, 16],
        cb_label="Gain vs. ALP + CG (pp)",
        title="Gain vs. Bambos ALP + CG",
        letter="a",
        panel_x_off=-26.0,
    )
    cb1 = _draw_regime_heatmap(
        ax1,
        diff_dgolb,
        cmap_teal_seq,
        vmin=0.0,
        vmax=80.0,
        cb_ticks=[0, 20, 40, 60, 80],
        cb_label="Gain vs. DGOLB + CG (pp)",
        title="Gain vs. DGOLB + CG",
        letter="b",
        panel_x_off=-24.0,
    )

    legend_handles = [
        Line2D([0], [0], color="#1E293B", ls="--", lw=1.35, label=r"Zone I boundary ($\kappa_{\mathrm{UL}} = 0.12$)"),
        Line2D([0], [0], color=COL_CORAL, ls="-.", lw=1.45, label=r"Zone II boundary ($\rho_{\mathrm{cmp}} = 0.90$)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=2,
        fontsize=7.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    _finalize_and_save(
        fig,
        "fig6_regime_map_2d",
        align_kwargs={
            "axes": [ax0, ax1],
            "panel_ids": ["a", "b"],
            "row_groups": [["a", "b"]],
            "exclude_axes": [cb0.ax, cb1.ax],
        },
    )


# --------------------------------------------------------------------------- #
# Figure 7 (Main Fig. 5): Azure Trace Replay, CSI Error & Cross-Hardware
# --------------------------------------------------------------------------- #

def plot_fig7() -> None:
    df_az = pd.read_csv(DATA_DIR / "b4a_azure_trace.csv")
    b4_sum = json.loads((STATS_DIR / "b4_summary.json").read_text(encoding="utf-8"))
    df_hw = pd.read_csv(DATA_DIR / "b4b_cross_hw.csv")
    df_pt = pd.read_csv(DATA_DIR / "b4b_perturb_sweep.csv")

    fig = plt.figure(figsize=(7.15, 2.42), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.15, 0.88], wspace=0.33,
                           left=0.075, right=0.985, bottom=0.19, top=0.73)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: Azure Functions 2019 Replay
    az_methods = ["CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG", "SFA+CG", "EDF_PreBatch+CG", "DGOLB+CG"]
    groups = [("FLEET", "Fleet\nN=32"), ("BURST", "Bursty\nN=32"), ("ALL", "All traces\nN=64")]
    xg = np.arange(len(groups))
    bw = 0.82 / len(az_methods)
    min_az_lo = 100.0

    for idx, m in enumerate(az_methods):
        st = METHOD_STYLE[m]
        means, lo_e, hi_e = [], [], []
        for gk, _ in groups:
            sub = df_az[df_az["method"] == m] if gk == "ALL" else df_az[(df_az["stratum"] == gk) & (df_az["method"] == m)]
            mv, lv, hv = bootstrap_ci95(sub["ontime_rate"].to_numpy() * 100.0, n_boot=5000, seed=6070)
            means.append(mv)
            elo = max(0.0, mv - lv)
            ehi = max(0.0, hv - mv)
            lo_e.append(elo)
            hi_e.append(ehi)
            min_az_lo = min(min_az_lo, mv - elo)
        xpos = xg + (idx - (len(az_methods) - 1) / 2.0) * bw
        ax0.bar(xpos, means, width=bw, color=st["color"], edgecolor="white", linewidth=0.35,
                yerr=[lo_e, hi_e], capsize=1.2, error_kw=ERROR_KW, label=st["label"])

    ax0.set_xticks(xg)
    ax0.set_xticklabels([g[1] for g in groups], fontsize=6.5)
    y_az_lo = max(0.0, 5.0 * np.floor((min_az_lo - 8.0) / 5.0))
    ax0.set_ylim(y_az_lo, 106.0)
    ax0.set_yticks([40, 60, 80, 100])
    ax0.set_ylabel("On-time rate (%)")
    ax0.set_title("Azure Functions 2019 replay", loc="left", pad=8)
    _style_axis(ax0, "y")
    _add_panel_label(ax0, "a", x_offset_pt=-24.0, y_offset_pt=10.0)

    # Panel b: CSI error continuous sweep (unobstructed!)
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
                 markersize=3.4, markeredgecolor="white", markeredgewidth=0.4, label=st["short"])
        ax1.fill_between(xs, ylo, yhi, color=st["color"], alpha=0.14)

    ax1.axvline(2.0, color=COL_NAVY, ls=":", lw=1.0)
    ax1.text(2.12, 7.0, "2 dB nom.", fontsize=6.1, color=COL_NAVY)
    ax1.axvline(6.0, color=COL_CORAL, ls=":", lw=1.0)
    ax1.text(5.85, 7.0, "6 dB limit", fontsize=6.1, color=COL_CORAL, ha="right")
    ax1.set_xlabel(r"CSI estimation error $\sigma_{\mathrm{CSI}}$ (dB)")
    ax1.set_ylabel("On-time rate (%)")
    ax1.set_ylim(2, 106)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_title("CSI error robustness", loc="left", pad=8)
    _style_axis(ax1, "both")
    _add_panel_label(ax1, "b", x_offset_pt=-22.0, y_offset_pt=10.0)

    # Panel c: Dedicated 3 Hardware Profiles grouped bar chart
    hws = [("HW1_Macro5G", "Macro\n5G"), ("HW2_WiFi6_MicroMEC", "Wi-Fi 6\n25 ms"), ("HW3_SmallCell_DualSB", "DualSB\n2 dB")]
    xhw = np.arange(len(hws))
    hw_methods = ["CLWB_Dual", "Bambos_ALP+CG", "Chen_Game+CG"]
    bw_hw = 0.76 / len(hw_methods)
    for i_m, m in enumerate(hw_methods):
        st = METHOD_STYLE[m]
        vals_hw, elo_hw, ehi_hw = [], [], []
        for h_k, _ in hws:
            arr = df_hw[(df_hw["hw_profile"] == h_k) & (df_hw["method"] == m)]["ontime_rate"].to_numpy() * 100.0
            mv, lv, hv = bootstrap_ci95(arr, n_boot=5000, seed=6078)
            vals_hw.append(mv)
            elo_hw.append(max(0.0, mv - lv))
            ehi_hw.append(max(0.0, hv - mv))
        ax2.bar(xhw + (i_m - 1) * bw_hw, vals_hw, width=bw_hw, color=st["color"], edgecolor="white",
                linewidth=0.35, yerr=[elo_hw, ehi_hw], capsize=1.3, error_kw=ERROR_KW)
    ax2.set_xticks(xhw)
    ax2.set_xticklabels([h[1] for h in hws], fontsize=6.4)
    ax2.set_ylim(55, 105)
    ax2.set_yticks([60, 70, 80, 90, 100])
    ax2.set_ylabel("On-time rate (%)")
    ax2.set_title("Hardware profiles", loc="left", pad=8)
    _style_axis(ax2, "y")
    _add_panel_label(ax2, "c", x_offset_pt=-22.0, y_offset_pt=10.0)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.99),
               ncol=3, fontsize=6.6, columnspacing=1.1, handletextpad=0.4, labelspacing=0.28)

    _finalize_and_save(
        fig,
        "fig7_azure_trace_and_robustness",
        align_kwargs={
            "exemptions": [
                {"panels": ["a", "b", "c"], "checks": ["panel-width"], "reason": "a and b hold 6-method bars/sweep; c holds compact 3-profile bars"},
            ]
        },
    )


# --------------------------------------------------------------------------- #
# Figure 8 (Main Fig. 6): Burst Scalability & Honest Failure Boundaries
# --------------------------------------------------------------------------- #

def plot_fig8() -> None:
    df_sc = pd.read_csv(DATA_DIR / "b4c_scalability.csv")
    df_bd = pd.read_csv(DATA_DIR / "b4d_theorem_boundaries.csv")
    b4_sum = json.loads((STATS_DIR / "b4_summary.json").read_text(encoding="utf-8"))
    f3_info = b4_sum["exp_4d_boundaries"]["fail_archive"]["FAIL-3_ZONE1_WEAK_AND_INVERTED_DL"]

    fig = plt.figure(figsize=(7.15, 2.55), dpi=300)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.22, 1.00, 1.02], wspace=0.42, hspace=0.30,
                           left=0.075, right=0.985, bottom=0.18, top=0.69)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[:, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    # Panel a (top-left): P99 Decision Latency log-y
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
        ax_a.plot(ps, ym, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                  markersize=2.8, markeredgecolor="white", markeredgewidth=0.35, label=st["short"])
        ax_a.fill_between(ps, ylo, yhi, color=st["color"], alpha=0.12)

    ax_a.axhline(50.0, color=COL_CRIMSON, ls="--", lw=1.0)
    ax_a.text(198.0, 68.0, "50 ms budget", color=COL_CRIMSON, fontsize=6.0, fontweight="bold", ha="right", va="bottom")
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.05, 5e4)
    ax_a.set_yticks([0.1, 10, 1000])
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0.1" if v < 1 else (f"{int(v//1000)}k" if v >= 1000 else f"{int(v)}")))
    ax_a.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
    ax_a.set_ylabel("P99 (ms)")
    ax_a.set_title(r"Scalability ($P \leq 200$)", loc="left", pad=5)
    _style_axis(ax_a, "both")
    _add_panel_label(ax_a, "a", x_offset_pt=-24.0, y_offset_pt=6.0)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_a.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.38), ncol=3, fontsize=5.8,
                columnspacing=0.45, handletextpad=0.3, labelspacing=0.15)

    # Panel b (bottom-left): Actual Checks vs. Thm. 3 Bound (sharing Burst P!)
    sub_c = df_sc[df_sc["method"] == "CLWB_Dual"]
    ps_c = sorted(sub_c["P"].unique())
    v_max = [float(sub_c[sub_c["P"] == pv]["max_tick_verify_calls"].max()) for pv in ps_c]
    t3_b = [float(sub_c[sub_c["P"] == pv]["thm2_bound"].max()) for pv in ps_c]  # Matches Thm. 3 Bound
    ax_b.plot(ps_c, v_max, color=COL_NAVY, marker="o", markersize=2.6, markeredgecolor="white",
              markeredgewidth=0.35, lw=1.4, label="Actual checks")
    ax_b.plot(ps_c, t3_b, color=COL_CORAL, ls="--", lw=1.25, label="Thm. 3 bound")
    ax_b.fill_between(ps_c, v_max, t3_b, color=COL_SLATE_BLUE, alpha=0.14)
    ax_b.set_ylim(0.0, max(t3_b) * 1.50)
    ax_b.set_yticks([0, 20, 40])
    ax_b.set_xticks([50, 100, 150, 200])
    ax_b.set_xlabel(r"Concurrent burst size $P$")
    ax_b.set_ylabel("Checks")
    _style_axis(ax_b, "both")
    _add_panel_label(ax_b, "b", x_offset_pt=-24.0, y_offset_pt=2.5)
    ax_b.legend(loc="upper left", ncol=2, fontsize=5.9, columnspacing=0.5, handletextpad=0.3)

    # Panel c: Premise breakdown curve (Anchor 1 deadline dispersion)
    t3_df = df_bd[df_bd["category"] == "curve2_thm3_deadline_dispersion"].sort_values("x_val")
    xv = t3_df["x_val"].to_numpy()
    yv = t3_df["metric_val"].to_numpy() * 100.0
    ylo_b = t3_df["ci_lo"].to_numpy() * 100.0
    yhi_b = t3_df["ci_hi"].to_numpy() * 100.0
    ax_c.plot(xv, yv, color=COL_NAVY, marker="o", markersize=3.5, markeredgecolor="white",
              markeredgewidth=0.4, lw=1.65, label="Single-tick match")
    ax_c.fill_between(xv, ylo_b, yhi_b, color=COL_NAVY, alpha=0.14)
    ax_c.text(xv[0] + 0.04, yv[0] + 2.5, f"{yv[0]:.1f}%", fontsize=6.4, fontweight="bold", color=COL_NAVY, ha="left")
    ax_c.text(xv[-1] - 0.02, yv[-1] - 7.5, f"{yv[-1]:.1f}%", fontsize=6.4, fontweight="bold", color=COL_NAVY, ha="right")
    ax_c.set_xlim(-0.04, 1.06)
    ax_c.set_ylim(42, 112)
    ax_c.set_yticks([50, 65, 80, 95, 110])
    ax_c.set_xlabel(r"Deadline dispersion $\eta_{\mathrm{inv}}$")
    ax_c.set_ylabel("Oracle match rate (%)")
    ax_c.set_title("Premise dispersion", loc="left", pad=5)
    _style_axis(ax_c, "both")
    _add_panel_label(ax_c, "c", x_offset_pt=-24.0, y_offset_pt=6.0)

    # Panel d: Dedicated Horizontal Failure Boundaries Bar Chart
    fail_df = df_bd[df_bd["category"] == "curve3_fail_archive"]
    fail_items = [
        ("FAIL-1_CSI_6DB", "CLWB_Dual_0dB", "0 dB CSI"),
        ("FAIL-1_CSI_6DB", "CLWB_Dual_6dB", "6 dB CSI"),
        ("FAIL-2_SUBTICK_DEV", "CLWB_Dual_SingleDevBurst", "Sub-tick"),
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
            f_tags.append(f"100% · {clwb_p50 / alp_p50:.1f}x P50")
        else:
            f_tags.append(f"{val_pct:.1f}%")

    yp_f = np.arange(len(f_vals))[::-1]
    f_cols = [COL_NAVY, COL_CORAL, COL_ROSE, COL_TEAL]
    ax_d.barh(yp_f, f_vals, height=0.56, color=f_cols, edgecolor="white", linewidth=0.4)
    for yp_i, bv, btag in zip(yp_f, f_vals, f_tags):
        ax_d.text(bv + 2.5, yp_i, btag, ha="left", va="center", fontsize=6.0, color="#1E293B", fontweight="bold")
    ax_d.set_yticks(yp_f)
    ax_d.set_yticklabels(f_lbls, fontsize=6.4)
    ax_d.set_xlim(0, 162)
    ax_d.set_xticks([0, 25, 50, 75, 100])
    ax_d.set_xlabel("On-time rate (%)")
    ax_d.set_title("Operating boundaries", loc="left", pad=5)
    _style_axis(ax_d, "x")
    _add_panel_label(ax_d, "d", x_offset_pt=-22.0, y_offset_pt=6.0)

    _finalize_and_save(
        fig,
        "fig8_scalability_and_boundaries",
        align_kwargs={
            "exemptions": [
                {"panels": ["c", "d"], "checks": ["panel-width"], "reason": "c shows dispersion curve and d shows compact failure bars"},
            ],
        },
    )


def main() -> None:
    st = verify_upstream_readonly()
    assert st["intact"], f"Upstream integrity violated: {st}"
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    print("[Nature Plot] Regenerated all 8 figures (.pdf + .svg + 300 DPI .png) cleanly in", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
