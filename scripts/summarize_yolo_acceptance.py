"""Summarize real per-lane performance and long-run memory without claiming unmeasured GPU success."""
import argparse
import csv
import json
from pathlib import Path

from calibrate_yolo import verify, worst_lanes


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def memory_summary(path):
    with path.open(encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if float(row["elapsed_s"]) >= 120]
    if len(rows) < 60:
        raise ValueError("Insufficient memory samples after the 120-second warm period")
    result = {}
    for field in ("client_private", "worker_private"):
        values = [int(row[field]) / 1024**2 for row in rows]
        first, last = sum(values[:60])/60, sum(values[-60:])/60
        result[field + "_mib"] = dict(min=min(values), max=max(values), initial_mean=first, final_mean=last, growth=last-first)
    result["continuous_growth_detected"] = any(result[k]["growth"] > 2 for k in ("client_private_mib", "worker_private_mib"))
    result["samples_after_warmup"] = len(rows)
    return result


def gpu_memory_summary(path):
    if not path.is_file() or not path.stat().st_size:
        return dict(available=False, growth_detected=None)
    with path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file, skipinitialspace=True))
    devices = {}
    for row in rows:
        index = row["index"].strip()
        key = next(k for k in row if k.startswith("memory.used"))
        try:
            devices.setdefault(index, []).append(float(row[key].split()[0]))
        except ValueError:
            continue
    measured = {}
    for index, values in devices.items():
        values = values[120:]
        if len(values) < 60:
            continue
        initial, final = sum(values[:60])/60, sum(values[-60:])/60
        measured[index] = dict(peak_used_mib=max(values), initial_mean=initial, final_mean=final, growth_mib=final-initial)
    return dict(available=bool(measured), devices=measured, growth_detected=any(v["growth_mib"] > 16 for v in measured.values()),
                note="nvidia-smi reports total device memory including the five business windows; a growth flag needs investigation before acceptance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    folder = args.directory.resolve()
    inventory = read(folder / "hardware.json")
    calibration = read(folder / "calibration-report.json")
    preliminary = read(folder / "acceptance-summary.json")
    reports = list(calibration["best"]["reports"]) + [calibration["auto_report"]]
    stability = {}
    if not args.calibrate_only:
        for name in ("soak.json", "soak-tensorrt-fp32.json"):
            path = folder / name
            if path.exists():
                report = read(path)
                if any(r["errors"] or r["elapsed_s"] < 1800 for r in report["rounds"]):
                    raise ValueError("A completed error-free 30-minute soak is required")
                stability[name] = memory_summary(Path(str(path) + ".resources.csv"))
                if name == "soak.json":
                    reports.append(report)
    cache_files = [folder / ("cache-" + name + ".detections.json") for name in ("first-build", "cache-hit", "corrupt-rebuild")]
    cache_consistent = False
    if all(path.is_file() for path in cache_files):
        reference = read(cache_files[0])
        cache_consistent = all(verify(reference, read(path), exact=True) for path in cache_files[1:])
        if not cache_consistent:
            raise ValueError("FP32 cache hit/rebuild detections changed from the initial engine")
    missing = read(folder / "optional-dependency-checks/summary.json")
    p50, p95 = worst_lanes(reports)
    timing_passed = p50 <= 20 and p95 <= 30 and all(not r["errors"] for report in reports for r in report["rounds"])
    grew = any(item["continuous_growth_detected"] for item in stability.values())
    gpu_memory = gpu_memory_summary(folder / ("soak-tensorrt-fp32.gpu.csv" if (folder / "soak-tensorrt-fp32.json").is_file() else "soak.gpu.csv"))
    preliminary.update(measured_worst_lane_p50_ms=p50, measured_worst_lane_p95_ms=p95, threshold_met=timing_passed,
                       stability=stability, memory_growth_detected=grew, cache_detections_consistent=cache_consistent,
                       optional_dependency_checks=missing, calibration_only=args.calibrate_only, gpu_memory=gpu_memory)
    preliminary["target_performance_accepted"] = bool(preliminary["target_host"] and inventory["business_windows_active"] and not inventory["baseline_only"]
        and not args.calibrate_only and timing_passed and not grew and stability and missing["passed"]
        and preliminary["gpu"]["gpu_soak"] and preliminary["gpu"]["cache_hit"] and preliminary["gpu"]["corruption_rebuilt"] and cache_consistent
        and gpu_memory["available"] and not gpu_memory["growth_detected"])
    (folder / "acceptance-summary.json").write_text(json.dumps(preliminary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Worst lane P50={p50:.3f}ms P95={p95:.3f}ms; target accepted={preliminary['target_performance_accepted']}")


if __name__ == "__main__":
    main()
