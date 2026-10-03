"""Check optional-backend failure and CPU startup in disposable x64 runtime copies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    empty = folder / "empty-nvidia"
    empty.mkdir()
    stage = folder / "runtime"
    stage.mkdir()
    source = args.runtime.resolve()
    for path in source.glob("*.dll"):
        if path.name != "CQ_YOLO_TensorRT.dll":
            shutil.copy2(path, stage / path.name)
    shutil.copy2(source / "yolo_benchmark_x64.exe", stage / "yolo_benchmark_x64.exe")
    environment = dict(os.environ, CQ_AI_NVIDIA_DIR=str(empty), CQ_AI_YOLO_TRACE_PREFIX="", CQ_AI_ORT_PROFILE_PREFIX="")
    checks = {}
    for name, device in (("missing-module", 3), ("missing-nvidia-runtime", 3), ("cpu-without-nvidia", 2)):
        if name == "missing-nvidia-runtime":
            shutil.copy2(source / "CQ_YOLO_TensorRT.dll", stage / "CQ_YOLO_TensorRT.dll")
        command = [str(stage / "yolo_benchmark_x64.exe"), "--model", str(args.model.resolve()), "--image", str(args.image.resolve()),
                   "--device", str(device), "--sessions", "1", "--callers", "1", "--threads", "1", "--warmup", "0", "--samples", "1", "--rounds", "1", "--shutdown-worker", "0", "--output", str(folder / (name + ".json"))]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        log = result.stdout + result.stderr
        (folder / (name + ".log")).write_text(log, encoding="utf-8")
        if name == "cpu-without-nvidia":
            passed = result.returncode == 0 and json.loads((folder / (name + ".json")).read_text())["runtime"]["active"] == "cpu"
        elif name == "missing-module":
            passed = result.returncode != 0 and "Cannot load optional CQ_YOLO_TensorRT.dll" in log
        else:
            passed = result.returncode != 0 and ("nvcuda.dll is missing" in log or ("Cannot load" in log and "nvinfer_10.dll" in log))
        checks[name] = dict(passed=passed, exit_code=result.returncode, log=str(folder / (name + ".log")))
        if not passed:
            raise RuntimeError(f"Optional dependency check failed: {name}: {log}")
    (folder / "summary.json").write_text(json.dumps(dict(passed=True, checks=checks), indent=2), encoding="utf-8")
    print("Optional module/runtime failure and CPU-without-NVIDIA checks passed")


if __name__ == "__main__":
    main()
