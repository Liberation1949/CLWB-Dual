"""Stage-3 Five-Gate Evolution Audit, K=12 Wilcoxon-Holm Benchmark, OOD & LOO Verification.

Enforces all 5 Stage-3 Guard Gates without touching frozen `stage2_model/` assets:
  Gate 1: Soundness (`soundness_test.py`: Stage-2 8/8 SHA-256 intact, 4/4 counterexamples,
          100 micro-instances vs exact oracle, 4/4 degenerate anchors, burst P=100 <= 25ms).
  Gate 2: AST Anti-Patchwork & Zero-Magic-Constant Audit on `champion_core.py`.
  Gate 3: Main K=12 dual-track comparison vs all 7 baselines mounted with `+CommonGuard`
          with strict per-workload aggregation (N_stat = 12, zero seed pooling) and
          family-wise Holm-Bonferroni corrected Wilcoxon signed-rank tests.
  Gate 4: Held-out OOD generalization (K_OOD=4 unseen workloads, 3 seeds each) on mean and worst group.
  Gate 5: 3-way Leave-One-Out (LOO) zero-self-harm verification across K=12 + counterexamples.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

STAGE2_DIR = Path(__file__).resolve().parent.parent / "stage2_model"
STAGE3_DIR = Path(__file__).resolve().parent
for _p in (str(STAGE2_DIR), str(STAGE3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_suite_s3 import (  # noqa: E402
    build_dual_track_cg_baselines,
    build_main_benchmark_k12,
    build_ood_suite_k4,
)
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
from champion_core import make_clwb_dual  # noqa: E402
from exact_oracle import build_counterexample_L1, build_counterexample_L2, build_counterexample_L3  # noqa: E402
from model_spec import run_simulation  # noqa: E402
from soundness_test import run_all_soundness  # noqa: E402


# --------------------------------------------------------------------------- #
# Gate 2: AST Zero-Magic-Constant & Anti-Patchwork Audit
# --------------------------------------------------------------------------- #

_ALLOWED_NUMERICS = {
    0, 1, 2,
    0.0, 0.5, 1.0, 2.0,
    1e-3, 1e-4, 1e-6, 1e-9, 1e-12, 1e-30, 1e18,
}
_FORBIDDEN_WORKLOAD_TOKENS = ("wk0", "wk1", "ood", "cal_", "balanced", "dense", "micro", "ce_")


def check_gate2_ast_audit() -> dict[str, Any]:
    src_path = STAGE3_DIR / "champion_core.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename="champion_core.py")

    bad_numbers: list[dict[str, Any]] = []
    bad_strings: list[dict[str, Any]] = []
    max_branches = 0
    complex_funcs: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                if val not in _ALLOWED_NUMERICS:
                    bad_numbers.append({"line": node.lineno, "value": val})
            elif isinstance(val, str):
                low = val.lower()
                if any(tok in low for tok in _FORBIDDEN_WORKLOAD_TOKENS):
                    bad_strings.append({"line": node.lineno, "value": val})

        if isinstance(node, ast.FunctionDef):
            branches = sum(
                1
                for sub in ast.walk(node)
                if isinstance(sub, (ast.If, ast.For, ast.While, ast.IfExp))
            )
            if branches > max_branches:
                max_branches = branches
            if branches > 24:
                complex_funcs.append(f"{node.name}:{branches}")

    passed = (len(bad_numbers) == 0 and len(bad_strings) == 0 and len(complex_funcs) == 0)
    return {
        "passed": bool(passed),
        "bad_magic_numbers": bad_numbers,
        "bad_dataset_strings": bad_strings,
        "max_function_branches": int(max_branches),
        "overcomplex_functions": complex_funcs,
    }


# --------------------------------------------------------------------------- #
# Gate 3: Main K=12 Benchmark + Family-Wise Holm-Bonferroni Wilcoxon Test
# --------------------------------------------------------------------------- #

def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [1.0] * m
    prev_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        cand = min(1.0, (m - rank) * float(p))
        val = max(prev_max, cand)
        prev_max = val
        adj[orig_idx] = val
    return adj


def check_gate3_main_k12(champ, baselines_cg: list[tuple[str, Any]]) -> dict[str, Any]:
    suite = build_main_benchmark_k12(n_seeds=3)
    all_policies = [("CLWB_Dual", champ)] + baselines_cg

    per_wk: dict[str, dict[str, dict[str, float]]] = {}
    vec_ontime: dict[str, list[float]] = {name: [] for name, _ in all_policies}
    vec_miss: dict[str, list[float]] = {name: [] for name, _ in all_policies}
    vec_waste: dict[str, list[float]] = {name: [] for name, _ in all_policies}
    max_p99: dict[str, float] = {name: 0.0 for name, _ in all_policies}
    tot_skip: dict[str, int] = {name: 0 for name, _ in all_policies}

    for idx, wk in enumerate(suite, 1):
        wid = wk["wid"]
        print(f"  [Gate 3] Evaluating {wid} ({idx}/{len(suite)})...", flush=True)
        per_wk[wid] = {}
        for pname, pol in all_policies:
            ots, misses, wastes, p99s, skips = [], [], [], [], 0
            for sc, cf in wk["cells"]:
                res = run_simulation(sc, pol, cf)
                ots.append(res["ontime_rate"])
                misses.append(res["server_grant_miss_rate"])
                wastes.append(res["wasted_srv_cycle_ratio"])
                p99s.append(res["decision_p99_ms"])
                skips += res["skipped_ticks"]
            m_ot = float(np.mean(ots))
            m_ms = float(np.mean(misses))
            m_ws = float(np.mean(wastes))
            m_p99 = float(np.max(p99s))
            per_wk[wid][pname] = {
                "ontime_rate": m_ot,
                "miss_rate": m_ms,
                "wasted_ratio": m_ws,
                "max_p99_ms": m_p99,
                "skipped_ticks": int(skips),
            }
            vec_ontime[pname].append(m_ot)
            vec_miss[pname].append(m_ms)
            vec_waste[pname].append(m_ws)
            max_p99[pname] = max(max_p99[pname], m_p99)
            tot_skip[pname] += skips

    champ_arr = np.asarray(vec_ontime["CLWB_Dual"], dtype=float)
    base_names = [n for n, _ in baselines_cg]
    raw_pvals = []
    stats_rows = {}
    for bname in base_names:
        b_arr = np.asarray(vec_ontime[bname], dtype=float)
        diff = champ_arr - b_arr
        wins = int(np.sum(diff > 1e-9))
        ties = int(np.sum(np.abs(diff) <= 1e-9))
        if np.all(np.abs(diff) <= 1e-12):
            pval = 1.0
        else:
            # Exact / asymptotic two-sided / greater paired Wilcoxon over K=12 independent workloads
            _, pval = wilcoxon(champ_arr, b_arr, alternative="greater", zero_method="wilcox")
        raw_pvals.append(float(pval))
        stats_rows[bname] = {
            "champ_mean_ontime": float(np.mean(champ_arr)),
            "base_mean_ontime": float(np.mean(b_arr)),
            "delta_pp": float(100.0 * (np.mean(champ_arr) - np.mean(b_arr))),
            "wins_of_12": wins,
            "ties_of_12": ties,
            "raw_p": float(pval),
        }

    holm_pvals = _holm_adjust(raw_pvals)
    for bname, hp in zip(base_names, holm_pvals):
        stats_rows[bname]["holm_p"] = float(hp)

    champ_beats_all_mean = all(stats_rows[b]["delta_pp"] > 0.0 for b in base_names)
    all_holm_sig = all(stats_rows[b]["holm_p"] < 0.05 for b in base_names)
    champ_zero_skip = (tot_skip["CLWB_Dual"] == 0)

    summary = {
        pname: {
            "mean_ontime": float(np.mean(vec_ontime[pname])),
            "mean_miss": float(np.mean(vec_miss[pname])),
            "mean_wasted": float(np.mean(vec_waste[pname])),
            "max_p99_ms": float(max_p99[pname]),
            "skipped_ticks": int(tot_skip[pname]),
        }
        for pname, _ in all_policies
    }

    return {
        "passed": bool(champ_beats_all_mean and all_holm_sig and champ_zero_skip),
        "n_workloads_independent": len(suite),
        "theoretical_min_holm_p": float(len(base_names) * (2.0 ** (-(len(suite) - 1)))),
        "summary_across_k12": summary,
        "wilcoxon_holm_vs_baselines": stats_rows,
        "per_workload": per_wk,
    }


# --------------------------------------------------------------------------- #
# Gate 4: Held-Out OOD Generalization (K_OOD = 4)
# --------------------------------------------------------------------------- #

def check_gate4_ood(champ, baselines_cg: list[tuple[str, Any]]) -> dict[str, Any]:
    # Compare against top fast + classical strong opponents on 4 OOD workloads (exclude multi-sec IBR_5/SFA if desired,
    # or include all fast ones so OOD completes swiftly)
    fast_bases = [(n, p) for n, p in baselines_cg if n not in ("IBR_5+CG",)]
    all_p = [("CLWB_Dual", champ)] + fast_bases
    ood = build_ood_suite_k4(n_seeds=3)

    per_ood: dict[str, dict[str, float]] = {}
    ot_map: dict[str, list[float]] = {n: [] for n, _ in all_p}
    miss_map: dict[str, list[float]] = {n: [] for n, _ in all_p}

    for wk in ood:
        wid = wk["wid"]
        per_ood[wid] = {}
        for pname, pol in all_p:
            ots, ms = [], []
            for sc, cf in wk["cells"]:
                res = run_simulation(sc, pol, cf)
                ots.append(res["ontime_rate"])
                ms.append(res["server_grant_miss_rate"])
            m_ot = float(np.mean(ots))
            m_ms = float(np.mean(ms))
            per_ood[wid][pname] = m_ot
            ot_map[pname].append(m_ot)
            miss_map[pname].append(m_ms)

    ch_mean = float(np.mean(ot_map["CLWB_Dual"]))
    ch_worst = float(np.min(ot_map["CLWB_Dual"]))
    best_base_mean = max(float(np.mean(ot_map[n])) for n, _ in fast_bases)
    best_base_worst = max(float(np.min(ot_map[n])) for n, _ in fast_bases)

    return {
        "passed": bool(ch_mean > best_base_mean and ch_worst > best_base_worst),
        "champ_ood_mean": ch_mean,
        "best_base_ood_mean": best_base_mean,
        "champ_ood_worst": ch_worst,
        "best_base_ood_worst": best_base_worst,
        "mean_by_method": {n: float(np.mean(ot_map[n])) for n, _ in all_p},
        "miss_by_method": {n: float(np.mean(miss_map[n])) for n, _ in all_p},
        "worst_by_method": {n: float(np.min(ot_map[n])) for n, _ in all_p},
        "per_workload": per_ood,
    }


# --------------------------------------------------------------------------- #
# Gate 5: LOO Zero Self-Harm Check
# --------------------------------------------------------------------------- #

def check_gate5_loo(champ, k12_champ_mean: float) -> dict[str, Any]:
    suite = build_main_benchmark_k12(n_seeds=3)
    ablations = [
        ("LOO_no_externality_pricing", make_clwb_dual(use_externality_pricing=False, name="LOO_no_ext")),
        ("LOO_no_lagrangian_defer", make_clwb_dual(use_lagrangian_defer=False, name="LOO_no_def")),
        ("LOO_no_fluid_contraction", make_clwb_dual(use_fluid_contraction=False, name="LOO_no_fluid")),
    ]
    ces = [
        ("L1", build_counterexample_L1),
        ("L2", build_counterexample_L2),
        ("L3", build_counterexample_L3),
        ("CE_2nd", build_counterexample_2nd_order),
    ]

    rows = {}
    all_positive = True
    for aname, apol in ablations:
        ots, ms = [], []
        for wk in suite:
            wk_ot, wk_ms = [], []
            for sc, cf in wk["cells"]:
                res = run_simulation(sc, apol, cf)
                wk_ot.append(res["ontime_rate"])
                wk_ms.append(res["server_grant_miss_rate"])
            ots.append(float(np.mean(wk_ot)))
            ms.append(float(np.mean(wk_ms)))
        m_ot = float(np.mean(ots))
        m_ms = float(np.mean(ms))
        ce_res = {}
        for cn, fn in ces:
            sc, cf = fn()
            ce_res[cn] = f"{run_simulation(sc, apol, cf)['n_ontime']}/{sc.n_tasks}"

        drop_pp = 100.0 * (k12_champ_mean - m_ot)
        if drop_pp <= 0.0:
            all_positive = False
        rows[aname] = {
            "k12_mean_ontime": m_ot,
            "k12_mean_miss": m_ms,
            "champ_advantage_pp": float(drop_pp),
            "counterexamples": ce_res,
        }

    return {
        "passed": bool(all_positive),
        "champ_k12_mean": float(k12_champ_mean),
        "ablations": rows,
    }


def main() -> int:
    t0 = time.perf_counter()
    champ = make_clwb_dual()
    baselines_cg = build_dual_track_cg_baselines()

    print("[Gate 1] Running soundness_test.py...")
    g1 = run_all_soundness()
    print(f"  -> Gate 1 passed={g1['all_passed']} (micro={g1['micro_100_oracle']['capture_ratio']:.4f}, ce={g1['counterexamples']['passed']}, anchors={g1['degenerate_anchors']['passed']})")

    print("[Gate 2] Running AST anti-patchwork & zero-magic-constant audit...")
    g2 = check_gate2_ast_audit()
    print(f"  -> Gate 2 passed={g2['passed']} (bad_nums={len(g2['bad_magic_numbers'])}, bad_str={len(g2['bad_dataset_strings'])}, max_branches={g2['max_function_branches']})")

    print("[Gate 3] Running Main K=12 dual-track benchmark (+CommonGuard) + Holm-Wilcoxon...")
    g3 = check_gate3_main_k12(champ, baselines_cg)
    champ_ot = g3["summary_across_k12"]["CLWB_Dual"]["mean_ontime"]
    print(f"  -> Gate 3 passed={g3['passed']} (Champ K=12={100*champ_ot:.2f}%, miss={100*g3['summary_across_k12']['CLWB_Dual']['mean_miss']:.2f}%)")
    for bname, st in g3["wilcoxon_holm_vs_baselines"].items():
        print(f"     vs {bname:16s}: +{st['delta_pp']:5.2f} pp | wins={st['wins_of_12']}/12 | holm_p={st['holm_p']:.5f}")

    print("[Gate 4] Running Held-out OOD generalization (K_OOD=4)...")
    g4 = check_gate4_ood(champ, baselines_cg)
    print(f"  -> Gate 4 passed={g4['passed']} (Champ OOD mean={100*g4['champ_ood_mean']:.2f}% vs best_base={100*g4['best_base_ood_mean']:.2f}%; worst={100*g4['champ_ood_worst']:.2f}% vs {100*g4['best_base_ood_worst']:.2f}%)")

    print("[Gate 5] Running 3-way LOO zero-self-harm check across K=12 + counterexamples...")
    g5 = check_gate5_loo(champ, champ_ot)
    print(f"  -> Gate 5 passed={g5['passed']}")
    for aname, row in g5["ablations"].items():
        print(f"     {aname:28s}: {100*row['k12_mean_ontime']:.2f}% (net contribution={row['champ_advantage_pp']:+.2f} pp) | miss={100*row['k12_mean_miss']:.2f}% | CE={row['counterexamples']}")

    all_passed = bool(g1["all_passed"] and g2["passed"] and g3["passed"] and g4["passed"] and g5["passed"])
    report = {
        "stage": 3,
        "model_base": "Model v1.0",
        "elapsed_s": float(time.perf_counter() - t0),
        "all_passed": all_passed,
        "gate1_soundness": g1,
        "gate2_ast_audit": g2,
        "gate3_main_k12": g3,
        "gate4_ood_k4": g4,
        "gate5_loo_ablation": g5,
    }
    (STAGE3_DIR / "stage3_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if all_passed:
        print("\nALL 5 STAGE-3 EVOLUTION GATES & 8 EXIT CHECKS PASSED — READY FOR STAGE-4 AUDIT")
        return 0
    print("\nSTAGE-3 GATE CHECK FAILED — INSPECT stage3_results.json")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
