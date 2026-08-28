#!/usr/bin/env python3
"""Execute every chapter notebook on a fresh kernel, in order, and write VALIDATION_REPORT.md.

Usage:
    python scripts/validate_all.py            # execute all notebooks in notebooks/
    python scripts/validate_all.py 06 07      # execute only chapters 06 and 07

Each notebook is executed in-place (outputs saved) with a fresh kernel via nbclient.
Exit code 0 only if every notebook executed without errors.
"""
from __future__ import annotations

import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
REPORT = REPO / "VALIDATION_REPORT.md"
TIMEOUT_S = 1200


def notebook_stats(nb) -> dict:
    code = [c for c in nb.cells if c.cell_type == "code"]
    executed = [c for c in code if c.get("execution_count") is not None]
    errors = [
        out for c in code for out in c.get("outputs", []) if out.get("output_type") == "error"
    ]
    return {"cells": len(nb.cells), "code": len(code), "executed": len(executed), "errors": len(errors)}


def run(nb_path: Path) -> dict:
    t0 = time.time()
    nb = nbformat.read(nb_path, as_version=4)
    status, err = "PASS", ""
    try:
        NotebookClient(
            nb,
            timeout=TIMEOUT_S,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO)}},
        ).execute()
    except Exception:
        status, err = "FAIL", traceback.format_exc(limit=8)
    nbformat.write(nb, nb_path)  # persist outputs either way
    s = notebook_stats(nb)
    return {"name": nb_path.name, "status": status, "runtime_s": time.time() - t0, "error": err, **s}


def main() -> int:
    prefixes = sys.argv[1:]
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if prefixes:
        notebooks = [n for n in notebooks if any(n.name.startswith(p) for p in prefixes)]
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 2

    results, t0 = [], time.time()
    for nb_path in notebooks:
        print(f"→ executing {nb_path.name} ...", flush=True)
        r = run(nb_path)
        print(f"  {r['status']} ({r['runtime_s']:.0f}s, {r['executed']}/{r['code']} code cells)", flush=True)
        results.append(r)

    total = time.time() - t0
    failed = [r for r in results if r["status"] != "PASS"]
    verdict = "PASS" if not failed else "FAIL"

    lines = [
        "# VALIDATION_REPORT",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Python: {sys.version.split()[0]}  Platform: {platform.platform()}",
        f"- Notebooks executed: {len(results)}   Failed: {len(failed)}",
        f"- Total runtime: {total:.0f}s",
        "",
        "| Notebook | Status | Runtime | Code cells | Executed | Error outputs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['status']} | {r['runtime_s']:.0f}s | {r['code']} | {r['executed']} | {r['errors']} |"
        )
    for r in failed:
        lines += ["", f"## Failure: {r['name']}", "```", r["error"], "```"]
    lines += ["", f"**SERIES VALIDATION: {verdict}**", ""]
    REPORT.write_text("\n".join(lines))
    print(f"\nSERIES VALIDATION: {verdict}  → {REPORT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
