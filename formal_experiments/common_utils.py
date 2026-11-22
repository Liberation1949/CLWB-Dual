"""Shared utilities for Stage 5 formal experiments.

Enforces:
  1. Strict read-only SHA-256 verification of Stage 2-4 upstream codebases.
  2. Single frozen champion construction (`make_clwb_dual()` with default args).
  3. Anti-pooling statistical helpers (10,000-resample Bootstrap 95% CI,
     two-sided paired Wilcoxon, Holm-Bonferroni correction, Cliff's delta).
  4. Strict finite CSV/JSON serialization (rejecting NaN / Inf).
  5. Verified Azure Functions 2019 inverse-CDF trace sampling kernel.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT_V3 = Path(__file__).resolve().parents[1]
STAGE2_DIR = ROOT_V3 / "stage2_model"
STAGE3_DIR = ROOT_V3 / "stage3_algorithm"
STAGE4_DIR = ROOT_V3 / "stage4_audit"
FORMAL_DIR = ROOT_V3 / "formal_experiments"
DATA_DIR = FORMAL_DIR / "data_csv"
STATS_DIR = FORMAL_DIR / "stats"
FIG_DIR = FORMAL_DIR / "figures"
AZURE_EXTRACT_DIR = ROOT_V3 / "trace_data" / "trace_extract"

for _d in (FORMAL_DIR, DATA_DIR, STATS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

for _p in (str(STAGE4_DIR), str(STAGE3_DIR), str(STAGE2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from champion_core import CLWBDualPolicy, make_clwb_dual  # noqa: E402


QUANTILES = np.array([0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0], dtype=float)

UPSTREAM_STAGE3_FILES = [
    "champion_core.py",
    "common_guard.py",
    "benchmark_suite_s3.py",
    "candidates_routes.py",
    "soundness_test.py",
    "guard_stage3.py",
]

UPSTREAM_STAGE4_FILES = [
    "step41_blind_and_hardware.py",
    "step42_anti_patchwork_and_grafting.py",
    "step43_phase_transition_scan.py",
    "step44_fuzz_theorems.py",
    "step45_regime_map_and_verdict.py",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_no_nan_inf(obj: Any, path_str: str = "root") -> None:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Non-finite float detected at {path_str}: {obj}")
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite numpy float detected at {path_str}: {val}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _validate_no_nan_inf(v, f"{path_str}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _validate_no_nan_inf(v, f"{path_str}[{idx}]")


def save_json_strict(obj: Any, path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_no_nan_inf(obj, out_path.name)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def save_csv_strict(df_or_rows: pd.DataFrame | list[dict[str, Any]], path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df = df_or_rows.copy()
    else:
        df = pd.DataFrame(df_or_rows)
    if df.empty:
        raise ValueError(f"Refusing to write empty CSV: {out_path}")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        arr = df[col].to_numpy(dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"Non-finite value in CSV column '{col}' for {out_path.name}")
    # Also check object columns for raw float NaNs/Infs
    for col in df.columns:
        if col not in num_cols:
            for idx, val in enumerate(df[col].tolist()):
                if isinstance(val, (float, np.floating)):
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        raise ValueError(f"Non-finite float at {out_path.name}[{idx}, {col}]: {val}")
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def verify_upstream_readonly() -> dict[str, Any]:
    """Verify Stage-2 manifest and Stage-3/4 cryptographic hashes."""
    manifest_path = STAGE2_DIR / "FREEZE_MANIFEST_v1.0.sha256"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing Stage-2 manifest: {manifest_path}")

    stage2_hashes: dict[str, str] = {}
    stage2_mismatches: list[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        expected = parts[0]
        fname = parts[-1].lstrip("*")
        fpath = STAGE2_DIR / fname
        if not fpath.exists():
            stage2_mismatches.append(f"stage2_model/{fname}:MISSING")
            continue
        actual = _sha256_file(fpath)
        stage2_hashes[fname] = actual
        if actual != expected:
            stage2_mismatches.append(f"stage2_model/{fname}:EXPECTED={expected}:ACTUAL={actual}")

    if stage2_mismatches or len(stage2_hashes) != 8:
        raise RuntimeError(f"Stage-2 freeze manifest verification failed: {stage2_mismatches}")

    current_snapshot: dict[str, dict[str, str]] = {
        "stage2_model": stage2_hashes,
        "stage3_algorithm": {},
        "stage4_audit": {},
    }
    for fname in UPSTREAM_STAGE3_FILES:
        fp = STAGE3_DIR / fname
        if not fp.exists():
            raise RuntimeError(f"Missing upstream Stage-3 file: {fp}")
        current_snapshot["stage3_algorithm"][fname] = _sha256_file(fp)

    for fname in UPSTREAM_STAGE4_FILES:
        fp = STAGE4_DIR / fname
        if not fp.exists():
            raise RuntimeError(f"Missing upstream Stage-4 file: {fp}")
        current_snapshot["stage4_audit"][fname] = _sha256_file(fp)

    baseline_path = STATS_DIR / "upstream_hashes_baseline.json"
    if not baseline_path.exists():
        save_json_strict(current_snapshot, baseline_path)
    else:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        drifts: list[str] = []
        for group, files_dict in current_snapshot.items():
            base_group = baseline.get(group, {})
            for fname, digest in files_dict.items():
                if base_group.get(fname) != digest:
                    drifts.append(f"{group}/{fname}:BASELINE={base_group.get(fname)}:ACTUAL={digest}")
        if drifts:
            raise RuntimeError(f"Upstream read-only violation detected: {drifts}")

    return {
        "intact": True,
        "stage2_files": len(current_snapshot["stage2_model"]),
        "stage3_files": len(current_snapshot["stage3_algorithm"]),
        "stage4_files": len(current_snapshot["stage4_audit"]),
        "baseline_path": str(baseline_path),
    }


def build_single_frozen_champion() -> CLWBDualPolicy:
    """Return the single frozen champion with strictly default parameters."""
    return make_clwb_dual()


def bootstrap_ci95(values: Any, n_boot: int = 10000, seed: int = 50001) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap CI on empty array.")
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boots = np.mean(arr[idx], axis=1)
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return mean_val, lo, hi


def paired_wilcoxon_and_cliffs(champ_vals: Any, base_vals: Any) -> tuple[float, float]:
    x = np.asarray(list(champ_vals), dtype=float)
    y = np.asarray(list(base_vals), dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError(f"Shape mismatch or empty input: {x.shape} vs {y.shape}")
    diff = x - y
    if np.all(np.abs(diff) <= 1e-12):
        raw_p = 1.0
    else:
        res = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        raw_p = float(res.pvalue)
    # Exact Cliff's delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    cliffs = float((gt - lt) / float(x.size * y.size))
    return raw_p, cliffs


def holm_bonferroni(raw_ps: list[float]) -> list[float]:
    m = len(raw_ps)
    order = sorted(range(m), key=lambda i: float(raw_ps[i]))
    adj = [0.0] * m
    cur = 0.0
    for rank, orig_idx in enumerate(order):
        val = (m - rank) * float(raw_ps[orig_idx])
        cur = max(cur, val)
        adj[orig_idx] = min(1.0, float(cur))
    return adj


def unpack_suite_item(item: Any) -> tuple[str, list[tuple[Any, Any]]]:
    if isinstance(item, dict):
        return str(item["wid"]), list(item["cells"])
    wid, cells = item
    return str(wid), list(cells)


def select_bursty_mask(pool: dict, top_n: int = 24, min_inv: int = 120, busiest_frac: float = 0.25) -> np.ndarray:
    counts = pool["counts"].astype(np.int64)
    apps, inv = np.unique(pool["app"], return_inverse=True)
    per_app = np.zeros((len(apps), counts.shape[1]), dtype=np.int64)
    for m_idx in range(counts.shape[1]):
        per_app[:, m_idx] = np.bincount(inv, weights=counts[:, m_idx], minlength=len(apps)).astype(np.int64)
    total = per_app.sum(axis=1)
    k = max(1, int(round(busiest_frac * counts.shape[1])))
    top_k_sum = np.sort(per_app, axis=1)[:, -k:].sum(axis=1)
    share = np.where(total > 0, top_k_sum / np.maximum(total, 1), 0.0)
    eligible = total >= min_inv
    order = np.lexsort((apps, -total, -share))
    picked = [i for i in order if eligible[i]][:top_n]
    mask_apps = np.zeros(len(apps), dtype=bool)
    mask_apps[picked] = True
    return mask_apps[inv]


def sample_azure_tasks(pool: dict, stratum: str, n_tasks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    if stratum == "BURST":
        mask = select_bursty_mask(pool)
    else:
        mask = np.ones(pool["counts"].shape[0], dtype=bool)
    counts = pool["counts"][mask].astype(np.int64)
    flat = counts.T.reshape(-1)
    cum = np.cumsum(flat)
    func_idx = np.tile(np.flatnonzero(mask), counts.shape[1])
    minute_idx = np.repeat(np.arange(counts.shape[1]), counts.shape[0])
    keep = flat > 0
    cum, func_idx, minute_idx = cum[keep], func_idx[keep], minute_idx[keep]
    total_events = int(cum[-1])
    spacing = total_events / float(n_tasks)
    u0 = float(rng.uniform(0.0, spacing))
    pts = u0 + np.arange(n_tasks) * spacing
    cells = np.clip(np.searchsorted(cum, pts, side="right"), 0, len(cum) - 1)
    funcs = func_idx[cells].copy()
    mins = minute_idx[cells].astype(float)
    t_3600 = np.minimum((mins + rng.uniform(0.0, 1.0, size=n_tasks)) * 60.0, 3600.0 - 1e-9)

    valid = pool["valid"]
    bad = ~valid[funcs]
    if np.any(bad):
        w = np.where(valid, pool["count_total"].astype(float), 0.0)
        w = w / np.sum(w)
        funcs[bad] = rng.choice(len(w), size=int(np.sum(bad)), p=w)

    knots = np.clip(pool["knots"][funcs].astype(float), pool["vmin"][funcs][:, None], pool["vmax"][funcs][:, None])
    avg = pool["avg"][funcs].astype(float)
    qmean = np.trapezoid(knots, QUANTILES, axis=1) if hasattr(np, "trapezoid") else np.trapz(knots, QUANTILES, axis=1)
    factor = np.where(qmean > 0, avg / np.maximum(qmean, 1e-12), 1.0)
    u = np.clip(rng.uniform(0.0, 1.0, size=n_tasks), 0.0, 1.0 - 1e-12)
    dur_ms = np.array([np.interp(u[i], QUANTILES, knots[i]) * factor[i] for i in range(n_tasks)], dtype=float)
    dur_s = np.maximum(dur_ms / 1000.0, 1e-9)
    raw_avg_ms = float(np.mean(avg))
    return t_3600, dur_s, raw_avg_ms
