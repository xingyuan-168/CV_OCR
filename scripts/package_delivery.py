"""Build a verified three-file delivery and separate optional/tool packages under outpush."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from generate_e_language_api_doc import exported_names, parse_exports, source_metadata
from package_yolo_candidate import copy, digest, zip_directory

ROOT = Path(__file__).resolve().parents[1]
CORE_NAMES = {"CQ_X86.dll", "CQ_AI_worker.exe", "易语言_DLL_API_说明.html"}


def run(command, log=None, cwd=None):
    result = subprocess.run([str(x) for x in command], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def info(path):
    return dict(size=path.stat().st_size, sha256=digest(path))


def check_zip(path, expected=None):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError(f"ZIP CRC failure: {path}")
        names = set(archive.namelist())
        if expected is not None and names != set(expected):
            raise ValueError(f"ZIP members differ: {path}")
        return names


def safe_output(path):
    path = path.resolve()
    base = (ROOT / "outpush").resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError("Packaging output must be a child of the workspace outpush directory")
    return path


def native_checks(args, folder, metadata):
    expected = {name for _, name, _ in parse_exports(ROOT / "include/ai_engine.h")}
    if len(expected) != 60 or exported_names(args.dumpbin, args.x86 / "CQ_X86.dll") != expected:
        raise ValueError("The x86 public export table differs from the 60 declarations")
    for path, machine in ((args.x86 / "CQ_X86.dll", "x86"), (args.worker, "x64")):
        headers = run([args.dumpbin, "/HEADERS", path])
        if f"machine ({machine})" not in headers:
            raise ValueError(f"Wrong PE architecture: {path}")
        dependencies = re.findall(r"^\s+([A-Za-z0-9_.-]+\.dll)\s*$", run([args.dumpbin, "/DEPENDENTS", path]), re.M)
        if any(re.search(r"(?i)(onnxruntime|directml|msvcp|vcruntime|cuda|cublas|nvrtc|nvinfer)", name) for name in dependencies):
            raise ValueError(f"Unexpected ordinary runtime dependency: {path}: {dependencies}")
        if machine == "x86" and dependencies != ["KERNEL32.dll"]:
            raise ValueError("The base x86 DLL must depend only on KERNEL32.dll")
    exports = run([args.dumpbin, "/EXPORTS", args.x86 / "CQ_X86.dll"])
    aliases = re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(_\w+@\d+)\s*$", exports, re.M)
    if len(aliases) != 60 or "_YOLO_InferJson@24" not in aliases:
        raise ValueError("The 60 stdcall aliases were not preserved")
    embedded = json.loads(run([args.worker, "--verify-embedded-runtime"], folder / "embedded-runtime.json"))
    probe = json.loads(run([args.worker, "--runtime-probe"], folder / "worker-probe.json"))
    if not embedded["valid"] or embedded["file_count"] < 13:
        raise ValueError("The embedded runtime failed verification")
    manifest = embedded["manifest"]
    for key in ("project_version", "delivery_version", "worker_protocol"):
        if manifest[key] != metadata[key]:
            raise ValueError(f"Embedded runtime version differs: {key}")
    if manifest["ort_version"] != "1.24.4" or manifest["directml_version"] != "1.15.4" or probe["worker_protocol"] != metadata["worker_protocol"]:
        raise ValueError("Pinned runtime/protocol differs")
    if not {"CPUExecutionProvider", "DmlExecutionProvider"}.issubset(probe["available_providers"]):
        raise ValueError("Worker CPU/DirectML providers are missing")
    return probe


def verified_reports(args, binaries, metadata):
    reports = {}
    for name in ("delivery-build.json", "continuous.json", "simultaneous.json", "soak-30min.json"):
        path = args.validation / name
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["dll_sha256"].lower() != binaries["CQ_X86.dll"]["sha256"] or report["worker_sha256"].lower() != binaries["CQ_AI_worker.exe"]["sha256"]:
            raise ValueError(f"Validation used a different DLL/Worker: {name}")
        if report["worker_protocol"] != metadata["worker_protocol"]:
            raise ValueError(f"Validation protocol differs: {name}")
        if name == "delivery-build.json":
            if report["errors"] or report["public_exports"] != 60 or report["version"] != "CQ_X86/" + metadata["project_version"]:
                raise ValueError("Independent delivery loader did not pass")
        else:
            if report["api_version"] != "CQ_X86/" + metadata["project_version"] or report["callers"] != 5 or any(r["errors"] for r in report["rounds"]):
                raise ValueError(f"Five-caller verification did not pass: {name}")
            if name == "soak-30min.json":
                if len(report["rounds"]) != 1 or report["rounds"][0]["elapsed_s"] < 1800:
                    raise ValueError("A new 30-minute soak is required for these binaries")
            elif len(report["rounds"]) != 3 or report["warmup_per_lane"] != 100 or report["samples_per_lane"] != 1000 or any(lane["count"] != 1000 for r in report["rounds"] for lane in r["lanes"]):
                raise ValueError("Formal timing requires warmup100/sample1000/three rounds per lane")
        reports[name] = dict(path=str(path.resolve()), **info(path))
    summary = json.loads((args.validation / "summary.json").read_text(encoding="utf-8"))
    if not summary["functional_passed"] or summary["memory_growth_detected"]:
        raise ValueError("Functional/stability summary did not pass")
    return reports


def nvidia_package(args, metadata):
    output = safe_output(args.nvidia_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    dependency = args.nvidia_deps.resolve()
    manifest = json.loads((dependency / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (dependency / name).resolve()
        if not path.is_relative_to(dependency) or info(path) != expected:
            raise ValueError(f"Optional NVIDIA dependency differs: {name}")
    module = args.x64 / "CQ_YOLO_TensorRT.dll"
    module_info = info(module)
    dependencies = re.findall(r"^\s+([A-Za-z0-9_.-]+\.dll)\s*$", run([args.dumpbin, "/DEPENDENTS", module]), re.M)
    if any(x.lower() not in {"kernel32.dll", "bcrypt.dll", "version.dll"} for x in dependencies):
        raise ValueError(f"TensorRT module has an ordinary third-party dependency: {dependencies}")
    package_metadata = dict(project_version=metadata["project_version"], delivery_version=metadata["delivery_version"], module=module_info, dependencies=manifest, target_gpu_validation="pending")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(module, module.name)
        for path in sorted(dependency.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "README.txt"}:
                archive.write(path, path.relative_to(dependency).as_posix())
        archive.writestr("manifest.json", json.dumps(package_metadata, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "Copy CQ_YOLO_TensorRT.dll and nvidia/ beside the paired x64 Worker, or set CQ_AI_NVIDIA_DIR to the nvidia directory. Base CPU/DirectML needs neither. Driver API >=12.8, SM >=7.5. nvcuda.dll comes from the system driver. TensorRT 10.13.3.9 / CUDA runtime12.8.90 / NVRTC12.8.93 / cuBLAS12.8.4.1. Target RTX2070 execution, FP16, Graph, cache and stability validation are pending.\n")
    names = check_zip(output)
    if "CQ_YOLO_TensorRT.dll" not in names or not any("licenses/tensorrt" in name for name in names):
        raise ValueError("Optional module/licenses missing from NVIDIA package")
    return dict(path=str(output), **info(output), module=module_info)


def python_wheel(args, output, metadata):
    expected = {name for _, name, _ in parse_exports(ROOT / "include/ai_engine.h")}
    if exported_names(args.dumpbin, args.x64 / "CQ_AI_x64.dll") != expected:
        raise ValueError("The x64 public export table differs from the public header")
    stage = output / "wheel-source"
    if stage.exists():
        raise ValueError("Choose a fresh Python package output; existing staging is preserved")
    for name in ("ai_engine.py", "setup.py", "pyproject.toml", "README.md"):
        copy(ROOT / "python" / name, stage / name)
    shutil.copytree(ROOT / "python/cq_ai_engine", stage / "cq_ai_engine", ignore=shutil.ignore_patterns("__pycache__", "*.dll"), dirs_exist_ok=True)
    for path in args.x64.glob("*.dll"):
        copy(path, stage / "cq_ai_engine/_native" / path.name)
    for path in (ROOT / "third_party/runtime/ort-directml-1.24.4/licenses").glob("*.txt"):
        copy(path, stage / "cq_ai_engine/_licenses" / path.name)
    run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", output], output / "wheel-build.log", cwd=stage)
    wheel = output / f"cq_ai_engine-{metadata['project_version']}-py3-none-win_amd64.whl"
    names = check_zip(wheel)
    if any(re.search(r"(?i)\.(exe|onnx|ini)$|(^|/)CQ_X86\.dll$", name) for name in names):
        raise ValueError("Wheel contains a forbidden payload")
    with zipfile.ZipFile(wheel) as archive:
        tag = archive.read(next(x for x in archive.namelist() if x.endswith(".dist-info/WHEEL"))).decode()
        if "Root-Is-Purelib: false" not in tag or "Tag: py3-none-win_amd64" not in tag:
            raise ValueError("The Wheel must contain an x64 native payload with the stable platform tag")
        for path in args.x64.glob("*.dll"):
            data = archive.read("cq_ai_engine/_native/" + path.name)
            if hashlib.sha256(data).hexdigest() != digest(path):
                raise ValueError(f"Wheel native payload differs: {path.name}")
    return dict(path=str(wheel), **info(wheel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x86", type=Path, default=ROOT / "build-cv-candidate-x86/Release")
    parser.add_argument("--x64", type=Path, default=ROOT / "build-cv-candidate-x64/Release")
    parser.add_argument("--worker", type=Path, default=ROOT / "build-cv-candidate-worker/Release/CQ_AI_worker.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "outpush/v23.6-delivery")
    parser.add_argument("--validation", type=Path, default=ROOT / "outpush/v23.6-validation")
    parser.add_argument("--dumpbin", type=Path, required=True)
    parser.add_argument("--nvidia-deps", type=Path, default=ROOT / "outpush/nvidia-runtime-trt10.13.3-cuda12.8")
    parser.add_argument("--nvidia-zip", type=Path, default=ROOT / "outpush/CQ_AI_NVIDIA_v23.6_TRT10.13.3_CUDA12.8.zip")
    parser.add_argument("--nvidia-only", action="store_true")
    parser.add_argument("--skip-nvidia", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    for key in ("x86", "x64", "worker", "validation", "dumpbin"):
        setattr(args, key, getattr(args, key).resolve())
    metadata_path = ROOT / "docs/YOLO_CANDIDATE_METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_metadata(ROOT / "include/ai_engine.h", ROOT / "src/worker_protocol.h", metadata_path)
    if args.nvidia_only:
        package = nvidia_package(args, metadata)
        args.nvidia_zip.with_suffix(".manifest.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
        print(json.dumps(package)); return
    output = safe_output(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if args.python_only:
        print(json.dumps(python_wheel(args, output, metadata))); return
    core = output / "core-stage"
    if core.exists():
        raise ValueError("Choose a fresh --out directory; existing deliverables are preserved")
    if metadata["functional_validation"] != "passed":
        raise ValueError("Finish functional and new-binary soak validation before staging delivery")
    binaries = {"CQ_X86.dll": info(args.x86 / "CQ_X86.dll"), "CQ_AI_worker.exe": info(args.worker)}
    if metadata.get("binary_fingerprints") != binaries:
        raise ValueError("HTML metadata must contain the latest DLL/Worker fingerprints")
    reports = verified_reports(args, binaries, metadata)
    for relative, expected_sha in metadata["native_source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected_sha:
            raise ValueError(f"Native source changed after the tested build snapshot: {relative}")
    probe = native_checks(args, output, metadata)
    run([sys.executable, ROOT / "scripts/generate_e_language_api_doc.py", "--manifest", metadata_path, "--dll", args.x86 / "CQ_X86.dll", "--dumpbin", args.dumpbin, "--module", ROOT / "examples/e_language_onnx_module.txt", "--check"], output / "html-check.log", cwd=ROOT)
    for name, path in (("CQ_X86.dll", args.x86 / "CQ_X86.dll"), ("CQ_AI_worker.exe", args.worker), ("易语言_DLL_API_说明.html", ROOT / "docs/易语言_DLL_API_说明.html")):
        copy(path, core / name)
    if {p.name for p in core.iterdir()} != CORE_NAMES:
        raise ValueError("Core stage must contain exactly the three delivery files")
    run([args.x86 / "ai_engine_delivery_verify.exe", "--dll", core / "CQ_X86.dll", "--header", ROOT / "include/ai_engine.h", "--model", args.validation / "中文路径/best.onnx", "--image", ROOT / "input/yolo测试图.bmp", "--expected-json", args.validation / "frozen-detections.json", "--report", output / "delivery-staged.json"], output / "delivery-staged.log")
    artifacts = {}
    if not args.no_zip:
        base_zip = output / f"CQ_AI_e_language_{metadata['delivery_version']}.zip"
        zip_directory(core, base_zip)
        check_zip(base_zip, CORE_NAMES)
        artifacts["easy_language_zip"] = dict(path=str(base_zip), **info(base_zip))
    tools = output / "diagnostics"
    copy(args.x86 / "ai_engine_yolo_benchmark.exe", tools / "yolo_benchmark_x86.exe")
    copy(args.x86 / "ai_engine_delivery_verify.exe", tools / "delivery_verify_x86.exe")
    copy(args.x64 / "ai_engine_yolo_benchmark.exe", tools / "x64/yolo_benchmark_x64.exe")
    for path in args.x64.glob("*.dll"):
        copy(path, tools / "x64" / path.name)
    for path in (ROOT / "third_party/runtime/ort-directml-1.24.4/licenses").glob("*.txt"):
        copy(path, tools / "x64/licenses" / path.name)
    copy(ROOT / "include/ai_engine.h", tools / "include/ai_engine.h")
    for name in ("calibrate_yolo.py", "summarize_yolo_trace.py", "summarize_yolo_acceptance.py", "verify_yolo_optional_dependencies.py", "monitor_yolo_benchmark.py", "run_yolo_acceptance.ps1"):
        copy(ROOT / "scripts" / name, tools / "tools" / name)
    for name in ("e_language_onnx_module.txt", "e_language_yolo_qpc.txt"):
        copy(ROOT / "examples" / name, tools / "examples" / name)
    for name in ("YOLO_OPTIMIZATION_CN.md", "YOLO_VALIDATION_CN.md", "V23_6_DELIVERY_CN.md"):
        copy(ROOT / "docs" / name, tools / name)
    copy(ROOT / "input/best.onnx", tools / "validation/best.onnx")
    for directory in (ROOT / "input", ROOT / "tests/fixtures/yolo"):
        for path in directory.glob("*.bmp"):
            copy(path, tools / "validation" / path.name)
    (tools / "README.txt").write_text("Requires Windows x64 and Python3.9+. Run powershell -File tools/run_yolo_acceptance.ps1 -RuntimeDir D:/your/application -Model D:/models/best.onnx -ImagesDir D:/businessBMP. RuntimeDir supplies the three delivered files; reports/staging do not modify it. Read V23_6_DELIVERY_CN.md. Baseline fixtures alone cannot accept five real business windows.\n", encoding="utf-8")
    tools_zip = output / f"CQ_AI_YOLO_tools_{metadata['delivery_version']}.zip"
    zip_directory(tools, tools_zip); check_zip(tools_zip)
    artifacts["diagnostics_zip"] = dict(path=str(tools_zip), **info(tools_zip))
    if args.skip_nvidia:
        package = json.loads(args.nvidia_zip.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if info(args.nvidia_zip) != {k: package[k] for k in ("size", "sha256")} or package["module"] != info(args.x64 / "CQ_YOLO_TensorRT.dll"):
            raise ValueError("The existing optional NVIDIA package differs from the current module")
    else:
        package = nvidia_package(args, metadata)
    artifacts["nvidia_zip"] = package
    caches = {}
    for name in ("build-cv-candidate-x86", "build-cv-candidate-worker", "build-cv-candidate-x64"):
        path = ROOT / name / "CMakeCache.txt"
        selected = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"(CMAKE_BUILD_TYPE|CMAKE_GENERATOR|CMAKE_GENERATOR_PLATFORM|AIENGINE_\w+|CMAKE_CXX_COMPILER):", line):
                key, value = line.split("=", 1); selected[key] = value
        caches[name] = selected
    result = dict(metadata, source_committed_revision=run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip(), source_snapshot_verified=True, created_utc=datetime.now(timezone.utc).isoformat(), core_directory=str(core), output_directory=str(ROOT / "output"), output_promotion="pending", files={p.name: info(p) for p in core.iterdir()}, worker_probe=probe, validation_reports=reports, build_configuration=caches, artifacts=artifacts, historical_release="release/v23.5 retained unchanged")
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output / "manifest.json"))


if __name__ == "__main__":
    main()
