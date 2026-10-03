"""Join DLL/Worker request traces and report measured stage percentiles."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--output", type=Path, default=Path("yolo-stages.json"))
    args = parser.parse_args()
    records = {}
    discarded_rows = 0
    for path in args.prefix.parent.glob(args.prefix.name + "-*.csv"):
        for row in csv.DictReader(path.open()):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Incomplete diagnostic row")
                numeric = {key: int(value) for key, value in row.items()}
            except (TypeError, ValueError):
                discarded_rows += 1
                continue
            if numeric["status"] < 0:
                continue
            key = row["request_id"]
            # Caller total contains the public ABI and transport. Worker total
            # contains only execution; keep the longer joined observation.
            if key not in records or int(row["total_us"]) > int(records[key]["total_us"]):
                records[key] = row
    if not records:
        raise SystemExit("No successful diagnostic requests found")
    stages = {}
    for name in next(iter(records.values())):
        if not name.endswith("_us"):
            continue
        values = sorted(int(row[name])/1000 for row in records.values())
        stages[name.replace("_us", "_ms")] = {f"p{p}": values[math.ceil(len(values)*p/100)-1] for p in (50,95,99)}
    ipc95 = stages["pack_ms"]["p95"] + stages["connect_ms"]["p95"] + stages["transport_ms"]["p95"]
    # The sum of marginal percentiles is a diagnostic bound. The decision is
    # based on the percentile of each request's combined IPC stages.
    combined = sorted((int(row["pack_us"])+int(row["connect_us"])+int(row["transport_us"]))/1000 for row in records.values())
    ipc95 = combined[math.ceil(len(combined)*.95)-1]
    result = dict(requests=len(records), discarded_incomplete_rows=discarded_rows, stages=stages, ipc_p95_ms=ipc95, shared_memory_measurement_gate_exceeded=ipc95 > 2,
                  note="Diagnostic tracing adds overhead; formal performance runs must disable it. A target-host IPC P95 >2ms requires a shared-memory follow-up.")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
