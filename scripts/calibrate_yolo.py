"""Measure the product ABI with real BMPs; qualify precision before publishing AUTO records."""
from __future__ import annotations
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading


def physical_cores():
    size = ctypes.c_ulong()
    kernel = ctypes.windll.kernel32
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise OSError("Cannot obtain physical CPU core budget")
    offset = count = 0
    while offset < size.value:
        length = int.from_bytes(buffer.raw[offset + 4:offset + 8], "little")
        if length < 8 or offset + length > size.value:
            raise ValueError("Invalid Windows CPU topology")
        count += 1
        offset += length
    return count


def detections_match(reference, candidate, coordinate_tolerance=1.0, score_tolerance=0.02):
    """Bipartite matching prevents two nearby detections matching the same box."""
    if len(reference) != len(candidate):
        return False
    edges = []
    for original in reference:
        matches = []
        for i, box in enumerate(candidate):
            if original["class_id"] != box["class_id"]:
                continue
            fields = ("x1", "y1", "x2", "y2", "score")
            if not all(math.isfinite(box[k]) and math.isfinite(original[k]) for k in fields):
                continue
            if all(abs(original[k] - box[k]) <= coordinate_tolerance for k in fields[:4]) and abs(original["score"] - box["score"]) <= score_tolerance:
                matches.append(i)
        edges.append(matches)
    assigned = {}

    def assign(index, visited):
        for box in edges[index]:
            if box in visited:
                continue
            visited.add(box)
            if box not in assigned or assign(assigned[box], visited):
                assigned[box] = index
                return True
        return False
    return all(assign(i, set()) for i in range(len(reference)))


def verify(reference, candidate, exact=False):
    if len(reference) != len(candidate):
        return False
    for original, changed in zip(reference, candidate):
        if original["sha256"] != changed["sha256"]:
            return False
        if exact:
            if original["detections"] != changed["detections"]:
                return False
        elif not detections_match(original["detections"], changed["detections"]):
            return False
    return True


def worst_lanes(reports):
    lanes = [lane for report in reports for round_ in report["rounds"] for lane in round_["lanes"]]
    return max(x["p50_ms"] for x in lanes), max(x["p95_ms"] for x in lanes)


def graph_qualified(normal, graph):
    # Every independently repeated round in both modes must improve P95 >=10%.
    if len(normal) != len(graph):
        return False
    for a, b in zip(normal, graph):
        if not b["runtime"]["cuda_graph"] or len(a["rounds"]) != len(b["rounds"]):
            return False
        for ar, br in zip(a["rounds"], b["rounds"]):
            previous = max(x["p95_ms"] for x in ar["lanes"])
            current = max(x["p95_ms"] for x in br["lanes"])
            if current > previous * 0.9:
                return False
    return True


def choose_candidate(entries):
    def throughput(entry):
        return min(r["throughput_per_s"] for p in entry["reports"] for r in p["rounds"])
    def rank(entry):
        meets_p95 = entry["p95"] <= 30
        return not meets_p95, entry["p50"] if meets_p95 else entry["p95"], -throughput(entry)
    fastest = min(entries, key=rank)
    comparable = [x for x in entries if (x["p95"] <= 30) == (fastest["p95"] <= 30)
                  and x["p50"] <= fastest["p50"] * 1.05 and x["p95"] <= fastest["p95"] * 1.05
                  and throughput(x) >= throughput(fastest) * .95]
    # Within 5% prefer freeing the GPU, fewer CPU workers, and fewer slots.
    return min(comparable, key=lambda x: (x["device"] != 2, x["sessions"] * x["threads"], x["sessions"], -throughput(x)))


def run_benchmark(args, folder, name, device, sessions, threads=1, precision="fp32", graph=0, mode="continuous", profile=None):
    report_path = folder / (name + ".json")
    detections_path = folder / (name + ".detections.json")
    command = [str(args.benchmark), "--model", str(args.model), "--device", str(device), "--sessions", str(sessions),
               "--threads", str(threads), "--precision", precision, "--graph", str(graph), "--mode", mode,
               "--warmup", str(args.warmup), "--samples", str(args.samples), "--rounds", str(args.rounds),
               "--output", str(report_path), "--verify-output", str(detections_path), "--workload", args.workload]
    if args.preserve_worker:
        command += ["--shutdown-worker", "0"]
    if profile:
        command += ["--profile", str(profile)]
    for image in args.images:
        command += ["--image", str(image)]
    environment = dict(os.environ, CQ_AI_YOLO_TRACE_PREFIX="")
    gpu_monitor = None
    smi = shutil.which("nvidia-smi")
    if smi:
        gpu_file = (folder / (name + ".gpu.csv")).open("wb")
        gpu_monitor = subprocess.Popen([smi, "--query-gpu=timestamp,index,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv", "-l", "1"], stdout=gpu_file, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        process = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
    finally:
        if gpu_monitor:
            gpu_monitor.terminate()
            gpu_monitor.wait()
            gpu_file.close()
    (folder / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"benchmark exit {process.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual = report["runtime"]
    expected = {0: None, 1: "directml", 2: "cpu", 3: "tensorrt"}[device]
    if expected and (actual["active"] != expected or actual["precision"] != precision):
        raise RuntimeError(f"Requested backend/precision differs from actual: {actual}")
    if threads and device == 2 and actual["intra_op_threads"] != threads:
        raise RuntimeError("CPU thread configuration did not reach the runtime")
    if any(round_["errors"] for round_ in report["rounds"]):
        raise RuntimeError("Benchmark contains inference errors")
    return report, json.loads(detections_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outpush/yolo-calibration"))
    parser.add_argument("--sessions", default="1,2,3,5")
    parser.add_argument("--cpu-threads", default="1,2,4,6,8")
    parser.add_argument("--devices", default="2,1,3")
    parser.add_argument("--workload", default="")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--preserve-worker", action="store_true", help="Release test models without shutting down a Worker used by business windows")
    args = parser.parse_args()
    args.images = sorted({p.resolve() for p in args.image} | (set(args.images_dir.resolve().glob("*.bmp")) if args.images_dir else set()))
    if not args.images:
        parser.error("Provide --image or --images-dir with real BMP files")
    args.benchmark = args.benchmark.resolve()
    args.model = args.model.resolve()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if not args.workload:
        args.workload = hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in args.images).encode()).hexdigest()
    reference_report, reference = run_benchmark(args, folder, "reference-cpu", 2, 1, 1)
    candidates, failures = [], []
    budget = physical_cores()
    for sessions in map(int, args.sessions.split(",")):
        for device in map(int, args.devices.split(",")):
            precisions = ["fp32", "fp16"] if device == 3 and args.fp16 else ["fp32"]
            for threads in (list(map(int, args.cpu_threads.split(","))) if device == 2 else [1]):
                if threads * sessions > budget:
                    continue
                fp32_reference = None
                for precision in precisions:
                    base_name = f"d{device}-s{sessions}-t{threads}-{precision}"
                    try:
                        if precision == "fp16" and fp32_reference is None:
                            raise RuntimeError("TensorRT FP32 reference did not qualify; FP16 remains disabled")
                        reports = []
                        for mode in ("continuous", "simultaneous"):
                            report, detections = run_benchmark(args, folder, base_name + "-" + mode, device, sessions, threads, precision, mode=mode)
                            if not verify(reference, detections, exact=device == 2):
                                raise RuntimeError("Detection count/class/coordinate/score gate failed; FP16 is rejected")
                            if precision == "fp16" and not verify(fp32_reference, detections):
                                raise RuntimeError("FP16 differs from the qualified TensorRT FP32 reference")
                            reports.append(report)
                        if device == 3 and precision == "fp32":
                            fp32_reference = detections
                        p50, p95 = worst_lanes(reports)
                        entry = dict(name=base_name, device=device, sessions=sessions, threads=threads, precision=precision, graph=0, p50=p50, p95=p95, reports=reports)
                        candidates.append(entry)
                        print(f"{base_name}: worst lane P50={p50:.3f} P95={p95:.3f} ms", flush=True)
                        if device == 3 and args.graphs:
                            captured = []
                            for mode in ("continuous", "simultaneous"):
                                report, detections = run_benchmark(args, folder, base_name + "-graph-" + mode, device, sessions, threads, precision, graph=1, mode=mode)
                                if not verify(reference, detections):
                                    raise RuntimeError("CUDA Graph detection gate failed")
                                captured.append(report)
                            if graph_qualified(reports, captured):
                                p50, p95 = worst_lanes(captured)
                                candidates.append(dict(entry, name=base_name + "-graph", graph=1, p50=p50, p95=p95, reports=captured))
                            else:
                                failures.append(dict(name=base_name + "-graph", error="Graph capture failed or P95 improvement is below 10% in at least one round; use ordinary execution"))
                    except Exception as error:
                        failures.append(dict(name=base_name, error=str(error)))
                        print(f"{base_name}: {error}", flush=True)
    if not candidates:
        raise RuntimeError("No backend passed detection and inference checks")
    selected = []
    for sessions in sorted({x["sessions"] for x in candidates}):
        selected.append(choose_candidate([x for x in candidates if x["sessions"] == sessions]))
    profile = folder / "business-calibration.tsv"
    temporary = profile.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write("# yolo-business-v4 key device threads fp16 graph verified worst_lane_p50 worst_lane_p95\n")
        for entry in selected:
            key = entry["reports"][0]["runtime"]["calibration_key"]
            file.write(f"{key}\t{entry['device']}\t{entry['threads']}\t{int(entry['precision']=='fp16')}\t{entry['graph']}\t1\t{entry['p50']:.9f}\t{entry['p95']:.9f}\n")
    best = choose_candidate(selected)
    # Exercise the generated record through AUTO, rather than merely trusting
    # the writer and loader have compatible keys and settings.
    auto_report, auto_detections = run_benchmark(args, folder, "selected-auto", 0, best["sessions"], 0, profile=temporary)
    if auto_report["runtime"]["selection_basis"] != "business_calibration" or not verify(reference, auto_detections):
        raise RuntimeError("Published AUTO record did not load or failed detection validation")
    temporary.replace(profile)
    result = dict(schema=1, workload=args.workload, physical_cores=budget, reference=reference_report, candidates=candidates, selected=selected, best=best, failures=failures,
                  host_sample_threshold_met=best["p50"] <= 20 and best["p95"] <= 30, auto_report=auto_report,
                  caveat="Acceptance applies only to this host and these BMPs; run with all five business windows on E5/RTX2070")
    (folder / "calibration-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    example = f"设置环境变量 CQ_AI_YOLO_CALIBRATION_FILE = {profile}\n设置环境变量 CQ_AI_YOLO_WORKLOAD_ID = {args.workload}\nYOLO_创建 (模型句柄)\nYOLO_加载模型从路径 (模型句柄, 模型路径, 标签路径, 0, 0, 0, {best['sessions']})\n五个调用线程共享模型句柄。最后一个参数是执行槽数。\n每路用 QPC 包住 YOLO_InferJson 调用进行验收。\n"
    (folder / "易语言加载参数.txt").write_text(example, encoding="utf-8-sig")
    print(f"AUTO profile: {profile}; recommended sessions={best['sessions']}")


if __name__ == "__main__":
    main()
