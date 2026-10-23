"""Summarize every CV case/round without weakening the fixed 5ms gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verify_cv_acceptance import LIMIT_MS, verify_timing


def summarize(metadata: dict, paths: dict[str, Path]) -> dict:
    result = dict(requirement_id="REQ-CV-001", limit_ms=LIMIT_MS,
                  project_version=metadata["project_version"],
                  git_source_sha256=metadata["git_source_sha256"],
                  accepted=True, architectures={})
    for arch, path in paths.items():
        data = path.read_bytes()
        report = json.loads(data)
        expected = (metadata["binary_fingerprints"]["CQ_X86.dll"]["sha256"]
                    if arch == "x86" else metadata["cv_x64_sha256"])
        reason = None
        try:
            verify_timing(report, expected, arch)
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
            result["accepted"] = False
        cases = []
        for case in report["cases"]:
            rounds = case["rounds"]
            cases.append(dict(name=case["name"], cold_ms=case["cold_ms"],
                              p95_ms_each_round=[r["p95_ms"] for r in rounds],
                              worst_p95_ms=max(r["p95_ms"] for r in rounds),
                              worst_p99_ms=max(r["p99_ms"] for r in rounds),
                              max_ms=max(r["max_ms"] for r in rounds),
                              count=sum(r["count"] for r in rounds),
                              errors=sum(r["errors"] for r in rounds),
                              worst_over5_fraction=max(r["over5_fraction"] for r in rounds)))
        result["architectures"][arch] = dict(
            accepted=reason is None, rejection=reason, dll_sha256=report["dll_sha256"],
            evidence=dict(file=path.name, size=len(data), sha256=hashlib.sha256(data).hexdigest()),
            cases=cases)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--x86", type=Path, required=True)
    parser.add_argument("--x64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    summary = summarize(metadata, {"x86": args.x86, "x64": args.x64})
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(accepted=summary["accepted"], limit_ms=LIMIT_MS,
                         report=str(args.output)), ensure_ascii=False))
    raise SystemExit(0 if summary["accepted"] else 3)


if __name__ == "__main__":
    main()
