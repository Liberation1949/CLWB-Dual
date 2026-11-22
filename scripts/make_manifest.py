"""Generate MANIFEST.sha256 across all tracked artifacts in CLWB-Dual."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", "__pycache__", "qa", "generated_tables"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".exe"}
EXCLUDE_NAMES = {"MANIFEST.sha256", ".DS_Store", "Thumbs.db"}


def main() -> int:
    entries: list[str] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_EXTS:
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = p.relative_to(ROOT).as_posix()
        entries.append(f"{digest}  {rel}")

    manifest_path = ROOT / "MANIFEST.sha256"
    manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} SHA-256 digests to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
