"""Four-gate artifact verifier for CLWB-Dual reproducibility package."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("stage2_model", "stage3_algorithm", "stage4_audit", "formal_experiments"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_stage2 import make_dgolb  # noqa: E402
from benchmark_suite_s3 import build_main_benchmark_k12  # noqa: E402
from candidates_routes import build_counterexample_2nd_order  # noqa: E402
from common_guard import wrap_with_common_guard  # noqa: E402
from common_utils import build_single_frozen_champion, verify_upstream_readonly  # noqa: E402
from model_spec import SimConfig, run_simulation  # noqa: E402


def gate1_frozen_and_traces() -> None:
    print("[Gate 1/4] Verifying frozen Stage 2-4 hashes and Azure trace assets...")
    status = verify_upstream_readonly()
    assert status.get("intact") is True, f"Upstream integrity failed: {status}"
    traces = sorted((ROOT / "trace_data" / "trace_extract").glob("*.npz"))
    assert len(traces) == 32, f"Expected 32 Azure .npz traces, found {len(traces)}"
    print(f"  OK Stage 2-4 hashes intact ({status['stage2_files']}+{status['stage3_files']}+{status['stage4_files']} files) & 32 Azure .npz traces present.")


def gate2_csv_and_headline_metrics() -> None:
    print("[Gate 2/4] Auditing 19 aggregated CSVs and published headline metrics...")
    csv_files = sorted((ROOT / "formal_experiments" / "data_csv").glob("*.csv"))
    assert len(csv_files) == 19, f"Expected 19 CSV datasets, found {len(csv_files)}"

    b2 = json.loads((ROOT / "formal_experiments" / "stats" / "b2_summary.json").read_text(encoding="utf-8"))
    k12 = b2["scopes"]["Main_K12"]["per_method"]["CLWB_Dual"]["ontime_mean"] * 100.0
    k24 = b2["scopes"]["All_24"]["per_method"]["CLWB_Dual"]["ontime_mean"] * 100.0
    assert abs(k12 - 97.94) < 0.02, f"Unexpected K=12 mean: {k12:.4f}"
    assert abs(k24 - 96.89) < 0.02, f"Unexpected K=24 mean: {k24:.4f}"

    b4 = json.loads((ROOT / "formal_experiments" / "stats" / "b4_summary.json").read_text(encoding="utf-8"))
    azure = b4["exp_4a_azure"]["overall"]["CLWB_Dual"]["ontime_mean"] * 100.0
    p200_max = b4["exp_4c_scalability"]["200"]["CLWB_Dual"]["p99_ms_max"]
    assert abs(azure - 99.63) < 0.02, f"Unexpected Azure mean: {azure:.4f}"
    assert abs(p200_max - 9.03) < 0.05, f"Unexpected P=200 max P99: {p200_max:.4f}"

    rev = json.loads((ROOT / "supp_verification" / "revision_audit_results.json").read_text(encoding="utf-8"))
    x64_ms = rev["check1_compiled_x64"]["per_scale"]["200"]["clwb_x64_ms_mean"]
    assert abs(x64_ms - 0.568) < 0.02, f"Unexpected P=200 x64 mean: {x64_ms:.4f}"
    print(f"  OK 19 CSVs verified | Main K=12={k12:.2f}%, All K=24={k24:.2f}%, Azure={azure:.2f}%, P200 P99={p200_max:.2f} ms, x64={x64_ms:.3f} ms.")


def gate3_live_smoke_episodes() -> None:
    print("[Gate 3/4] Running live CLWB-Dual simulation smoke checks...")
    champ = build_single_frozen_champion()
    dgolb_cg = wrap_with_common_guard(make_dgolb())
    ce_scen, ce_cfg = build_counterexample_2nd_order()
    res_champ = run_simulation(ce_scen, champ, ce_cfg)
    res_dgolb = run_simulation(ce_scen, dgolb_cg, ce_cfg)
    assert res_champ["n_ontime"] == 4 and res_champ["n_tasks"] == 5, f"CE_2ndOrder CLWB failed: {res_champ}"
    assert res_dgolb["n_ontime"] == 1 and res_dgolb["n_tasks"] == 5, f"CE_2ndOrder DGOLB failed: {res_dgolb}"

    wk01_item = build_main_benchmark_k12(n_seeds=1)[0]
    scen, cfg = wk01_item["cells"][0]
    res_wk01 = run_simulation(scen, champ, cfg)
    assert res_wk01["ontime_rate"] == 1.0 and res_wk01["skipped_ticks"] == 0, f"WK01 failed: {res_wk01}"
    print("  OK CE_2ndOrder: CLWB-Dual 4/5 vs DGOLB+CG 1/5 | WK01 S0: 100.00%, 0 skipped ticks.")


def gate4_manifest_sha256() -> None:
    print("[Gate 4/4] Verifying MANIFEST.sha256 digests...")
    manifest_path = ROOT / "MANIFEST.sha256"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    entries = [ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) >= 80, f"Suspiciously small manifest: {len(entries)} entries"
    for line in entries:
        expected, rel = line.split("  ", 1)
        fp = ROOT / rel
        assert fp.exists(), f"Missing file in manifest: {rel}"
        actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert actual == expected, f"SHA-256 mismatch on {rel}: expected {expected}, got {actual}"
    print(f"  OK Verified all {len(entries)} tracked files against MANIFEST.sha256.")


def main() -> int:
    gate1_frozen_and_traces()
    gate2_csv_and_headline_metrics()
    gate3_live_smoke_episodes()
    gate4_manifest_sha256()
    print("\nALL 4 VERIFICATION GATES PASSED FOR CLWB-Dual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
