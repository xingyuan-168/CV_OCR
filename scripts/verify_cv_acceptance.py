"""Binary-bound transparent CV acceptance; never infer success from old DLLs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

LIMIT_MS = 5.0
INPUT_HASHES = {
    "1.bmp": "b4c8df7314fb91fbbc6f118013d503d6c4ef281e24af1f07a397fca09036939c",
    "2.bmp": "a6eab91fc30c641c618e2398b71b9205189c77257393467352128cd554cc0357",
}
REQUIRED_CASES = {
    "old_fixed_hit", "old_fixed_miss", "old_changed_hit", "old_changed_miss",
    "new_color", "new_gray", "new_multi", "new_changed_color", "new_changed_gray",
    "new_changed_multi", "new_miss", "new_gray_miss", "new_multi_miss",
}
LEGACY_CASES = {"threshold_single", "single", "transparent_single", "multi_text",
                "transparent_multi_text", "changing_multi", "changing_transparent"}


def required(metadata: dict) -> bool:
    version = tuple(int(n) for n in metadata["project_version"].split("."))
    return version >= (0, 14, 9) or "cv_transparent_validation" in metadata


def verify_timing(report: dict, expected_sha: str, architecture: str, *, legacy: bool = False) -> None:
    if report["dll_sha256"].lower() != expected_sha.lower():
        raise ValueError("CV timing used a different DLL")
    if report.get("architecture") != architecture or report.get("test_hooks") is not False:
        raise ValueError("CV acceptance requires the correct architecture and production DLL")
    if not legacy and report.get("input_sha256") != INPUT_HASHES:
        raise ValueError("CV acceptance used different frozen images")
    if (report["warmup"] != 64 or report["samples"] != 1000 or report["rounds"] != 3
            or report["limit_ms"] != LIMIT_MS):
        raise ValueError("CV acceptance requires warmup64/sample1000/three rounds/5ms")
    cases = report["cases"]
    required_cases = LEGACY_CASES if legacy else REQUIRED_CASES
    if len(cases) != len(required_cases) or {c["name"] for c in cases} != required_cases:
        raise ValueError("CV acceptance is missing required old/new scenarios")
    for case in cases:
        if len(case["rounds"]) != 3:
            raise ValueError("CV case requires three independent rounds")
        for r in case["rounds"]:
            values = [r[k] for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms")]
            if r["count"] != 1000 or r["errors"] != 0:
                raise ValueError("CV functional/sample acceptance failed")
            if not all(math.isfinite(v) and v >= 0 for v in values) or values != sorted(values):
                raise ValueError("Invalid CV percentile values")
            if r["p95_ms"] > LIMIT_MS:
                raise ValueError(f"CV5ms failed: {architecture}/{case['name']} P95={r['p95_ms']}ms")
            if not 0 <= r["over5_fraction"] <= 1:
                raise ValueError("Invalid CV over-limit fraction")
    if report.get("accepted_5ms") is not True:
        raise ValueError("CV timing report did not pass5ms")


def verify_candidate_cv(metadata: dict, validation: Path, binaries: dict) -> dict:
    if not required(metadata):
        return {}
    if metadata.get("cv_transparent_validation") != "passed" or metadata.get("cv_transparent_p95_limit_ms") != LIMIT_MS:
        raise ValueError("REQ-CV-001 has not passed correctness and every5ms acceptance round")
    evidence = metadata["cv_transparent_evidence"]
    def read_evidence(item: dict) -> dict:
        path = (validation / item["file"]).resolve()
        if not path.is_relative_to(validation.resolve()):
            raise ValueError("CV evidence escapes validation directory")
        data = path.read_bytes()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("CV evidence hash differs")
        return json.loads(data)

    reports = {}
    for architecture in ("x86", "x64"):
        item = evidence[f"timing_{architecture}"]
        expected = binaries["CQ_X86.dll"]["sha256"] if architecture == "x86" else metadata["cv_x64_sha256"]
        verify_timing(read_evidence(item), expected, architecture)
        verify_timing(read_evidence(evidence[f"legacy_{architecture}"]), expected, architecture, legacy=True)
        reports[architecture] = item
        for threads in (1, 4, 16):
            stress = read_evidence(evidence[f"stress_{architecture}_{threads}"])
            if (stress["dll_sha256"] != expected or stress["threads"] != threads
                    or stress.get("architecture") != architecture or stress.get("test_hooks") is not False
                    or not math.isfinite(stress["elapsed_s"]) or stress["elapsed_s"] < 600
                    or stress["errors"] != 0 or stress.get("calls", 0) <= 0
                    or stress.get("changed_cases", 0) < 7
                    or stress.get("template_release_passed") is not True
                    or stress.get("memory_growth_detected") is not False):
                raise ValueError("CV stress evidence failed or used a different DLL")
    if metadata.get("cv_independent_reference_validation") != "passed":
        raise ValueError("Independent exhaustive CV reference did not pass")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    result = verify_candidate_cv(metadata, args.validation, metadata["binary_fingerprints"])
    print(json.dumps({"passed": True, "requirement_id": "REQ-CV-001", "reports": result}))


if __name__ == "__main__":
    main()
