"""Run every legacy CV latency scenario on adjacent production DLLs, serially."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"threshold_single", "single", "transparent_single", "multi_text",
         "transparent_multi_text", "changing_multi", "changing_transparent"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--arch", choices=("x86", "x64"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build, out = args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dll = build / ("CQ_X86.dll" if args.arch == "x86" else "CQ_AI_x64.dll")
    sha = digest(dll)
    # The configured build is also checked; private test exports cannot silently
    # turn instrumentation timings into release acceptance.
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    if "AIENGINE_CV_TEST_HOOKS:BOOL=OFF" not in cache:
        raise ValueError("Production build with test hooks OFF required")
    cases = {name: [] for name in NAMES}
    cold = {name: [] for name in NAMES}
    for round_id in range(3):
        raw = out / f"legacy-{args.arch}-{round_id}.jsonl"
        if raw.exists():
            raise FileExistsError(raw)
        env = dict(os.environ, CQ_CV_BENCHMARK_REPORT=str(raw))
        for exe, options in (("ai_engine_cv_business_benchmark.exe", ["--enforce", "all"]),
                             ("ai_engine_cv_changing_benchmark.exe", [])):
            if digest(dll) != sha:
                raise ValueError("DLL changed during acceptance")
            command = [str(build / exe), str(ROOT / "tests/fixtures/cv/base"), "1000", *options]
            with (out / f"{exe}-{args.arch}-{round_id}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=240)
            # Exit3 is the explicit latency failure, retained with all raw data.
            if result.returncode not in (0, 3):
                raise RuntimeError(f"{exe} functional failure {result.returncode}")
            if digest(dll) != sha:
                raise ValueError("DLL changed during acceptance")
        records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        if len(records) != len(NAMES) or {r["name"] for r in records} != NAMES:
            raise ValueError("Missing legacy scenario")
        for record in records:
            values = sorted(record["samples_ms"])
            if len(values) != 1000 or record["warmup"] != 64 or record["errors"]:
                raise ValueError("Invalid samples")
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Invalid latency")
            row = {f"p{q}_ms": values[math.ceil(q / 100 * len(values)) - 1] for q in (50, 95, 99)}
            row.update(count=len(values), errors=0, max_ms=values[-1],
                       over5_fraction=sum(v > 5 for v in values) / len(values))
            cases[record["name"]].append(row)
            cold[record["name"]].append(record["prewarm_first_call_ms"])
        print(args.arch, round_id, "worst P95", max(v[-1]["p95_ms"] for v in cases.values()), flush=True)
    report = dict(schema=1, architecture=args.arch, test_hooks=False, dll_sha256=sha,
                  warmup=64, samples=1000, rounds=3, limit_ms=5,
                  cases=[dict(name=n, rounds=cases[n], prewarm_first_call_ms=cold[n]) for n in sorted(cases)],
                  accepted_5ms=all(r["p95_ms"] <= 5 for rows in cases.values() for r in rows))
    (out / f"legacy-timing-{args.arch}.json").write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    raise SystemExit(0 if report["accepted_5ms"] else 3)


if __name__ == "__main__":
    main()
