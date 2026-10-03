"""Stage versioned diagnostic/Python artifacts under outpush; preserve historical release files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def copy(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def zip_directory(directory, output):
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x86", type=Path, default=ROOT / "build-cv-candidate-x86/Release")
    parser.add_argument("--x64", type=Path, default=ROOT / "build-cv-candidate-x64/Release")
    parser.add_argument("--worker", type=Path, default=ROOT / "build-cv-candidate-worker/Release/CQ_AI_worker.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "outpush/yolo-v23.6-candidate")
    parser.add_argument("--nvidia-sdk", type=Path)
    parser.add_argument("--cuda-wheels", type=Path, default=ROOT / "third_party/nvidia-download")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outpush").resolve()) or output == (ROOT / "outpush").resolve():
        raise ValueError("Candidate output must be a child directory under outpush")
    output.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([str(args.worker.resolve()), "--runtime-probe"], capture_output=True, text=True, encoding="utf-8", check=True)
    runtime = json.loads(probe.stdout)
    if runtime["worker_protocol"] != 26:
        raise ValueError("Candidate Worker must use protocol 26")
    e = output / "e_language"
    p = output / "python_x64"
    copy(args.x86 / "CQ_X86.dll", e / "CQ_X86.dll")
    copy(args.worker, e / "CQ_AI_worker.exe")
    copy(args.x86 / "ai_engine_yolo_benchmark.exe", e / "yolo_benchmark_x86.exe")
    copy(args.x64 / "ai_engine_yolo_benchmark.exe", p / "yolo_benchmark_x64.exe")
    for path in args.x64.glob("*.dll"):
        copy(path, p / path.name)
    copy(args.x64 / "CQ_YOLO_TensorRT.dll", e / "CQ_YOLO_TensorRT.dll")
    for path in (ROOT / "python").glob("*.py"):
        copy(path, p / path.name)
    shutil.copytree(ROOT / "python/cq_ai_engine", p / "cq_ai_engine", ignore=shutil.ignore_patterns("__pycache__", "*.dll"), dirs_exist_ok=True)
    for path in (ROOT / "third_party/runtime/ort-directml-1.24.4/licenses").glob("*"):
        if path.is_file(): copy(path, p / "licenses" / path.name)
    for folder in (e, p):
        copy(ROOT / "docs/易语言_DLL_API_说明.html", folder / "易语言_DLL_API_说明.html")
        copy(ROOT / "docs/YOLO_OPTIMIZATION_CN.md", folder / "YOLO_OPTIMIZATION_CN.md")
        copy(ROOT / "docs/YOLO_VALIDATION_CN.md", folder / "YOLO_VALIDATION_CN.md")
        for name in ("calibrate_yolo.py", "summarize_yolo_trace.py", "run_yolo_acceptance.ps1"):
            copy(ROOT / "scripts" / name, folder / "tools" / name)
        for name in ("e_language_onnx_module.txt", "e_language_yolo_qpc.txt"):
            copy(ROOT / "examples" / name, folder / "examples" / name)
        copy(ROOT / "input/best.onnx", folder / "validation/best.onnx")
        for path in (ROOT / "input").glob("*.bmp"):
            copy(path, folder / "validation" / path.name)
        for path in (ROOT / "tests/fixtures/yolo").glob("*.bmp"):
            copy(path, folder / "validation" / path.name)
    for name in ("calibrate_yolo.py", "summarize_yolo_trace.py"):
        copy(ROOT / "scripts" / name, output / "tools" / name)
    for name in ("e_language_onnx_module.txt", "e_language_yolo_qpc.txt"):
        copy(ROOT / "examples" / name, output / "examples" / name)
    copy(ROOT / "input/best.onnx", output / "validation/best.onnx")
    for path in (ROOT / "input").glob("*.bmp"):
        copy(path, output / "validation" / path.name)
    metadata = json.loads((ROOT / "docs/YOLO_CANDIDATE_METADATA.json").read_text())
    metadata["worker_probe"] = runtime
    metadata["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata["dirty_source"] = bool(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).strip())
    # Build a separately named candidate Wheel from the same native payload.
    stage = output / "wheel-source"
    for name in ("ai_engine.py", "setup.py", "pyproject.toml", "README.md"):
        copy(ROOT / "python" / name, stage / name)
    shutil.copytree(ROOT / "python/cq_ai_engine", stage / "cq_ai_engine", ignore=shutil.ignore_patterns("__pycache__", "*.dll"), dirs_exist_ok=True)
    for path in p.glob("*.dll"):
        copy(path, stage / "cq_ai_engine/_native" / path.name)
    for path in (p / "licenses").glob("*.txt"):
        copy(path, stage / "cq_ai_engine/_licenses" / path.name)
    subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--build-number", "1yolov236", "--dist-dir", str(output.parent)], cwd=stage, check=True, stdout=subprocess.PIPE)
    wheel = output.parent / f"cq_ai_engine-{metadata['project_version']}-1yolov236-py3-none-win_amd64.whl"
    metadata["python_wheel"] = dict(path=wheel.name, size=wheel.stat().st_size, sha256=digest(wheel))
    if args.nvidia_sdk:
        dependency = output / "nvidia-dependencies"
        for path in args.x64.glob("*.dll"):
            if path.name.lower().startswith(("msvcp", "vcruntime", "concrt")):
                copy(path, dependency / "nvidia" / path.name)
        for path in (p / "licenses").glob("*.txt"):
            if "vcrt" in path.name.lower() or "vc_redist" in path.name.lower() or "visual-cpp" in path.name.lower():
                copy(path, dependency / "nvidia/licenses" / path.name)
        for path in args.nvidia_sdk.rglob("*.dll"):
            copy(path, dependency / "nvidia" / path.name)
        for path in args.nvidia_sdk.rglob("*"):
            if path.is_file() and ("license" in path.name.lower() or path.name.lower() in ("acknowledgements.txt", "readme.txt")):
                copy(path, dependency / "nvidia/licenses/tensorrt" / path.name)
        packages = ("nvidia_cuda_runtime_cu12-12.8.90", "nvidia_cuda_nvrtc_cu12-12.8.93", "nvidia_cublas_cu12-12.8.4.1")
        for package in packages:
            wheel = next(args.cuda_wheels.glob(package + "*win_amd64.whl"))
            with zipfile.ZipFile(wheel) as archive:
                for name in archive.namelist():
                    destination = None
                    if name.lower().endswith(".dll"):
                        destination = dependency / "nvidia" / Path(name).name
                    elif "license" in name.lower() and not name.endswith("/"):
                        destination = dependency / "nvidia/licenses" / package / Path(name).name
                    if destination:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(name) as source, destination.open("wb") as target:
                            shutil.copyfileobj(source, target, 8*1024*1024)
        hashes = {path.relative_to(dependency).as_posix(): dict(size=path.stat().st_size, sha256=digest(path)) for path in dependency.rglob("*") if path.is_file()}
        (dependency / "manifest.json").write_text(json.dumps(dict(tensorrt="10.13.3.9", cuda_runtime="12.8.90", cuda_nvrtc="12.8.93", cublas="12.8.4.1", files=hashes), indent=2))
        (dependency / "README.txt").write_text("Copy the nvidia folder beside CQ_AI_worker.exe and CQ_AI_x64.dll, or set CQ_AI_NVIDIA_DIR. Install a NVIDIA driver exposing CUDA >=12.8. nvcuda.dll is supplied by the driver and is never bundled. GPU execution is pending target-machine validation.\n")
        dependency_zip = output.parent / "CQ_AI_NVIDIA_TRT10.13.3_CUDA12.8.zip"
        zip_directory(dependency, dependency_zip)
        metadata["nvidia_package"] = dict(path=dependency_zip.name, size=dependency_zip.stat().st_size, sha256=digest(dependency_zip))
    else:
        # Refresh base artifacts without rebuilding the unchanged optional archive.
        dependency_zip = output.parent / "CQ_AI_NVIDIA_TRT10.13.3_CUDA12.8.zip"
        if dependency_zip.is_file():
            metadata["nvidia_package"] = dict(path=dependency_zip.name, size=dependency_zip.stat().st_size, sha256=digest(dependency_zip))
    files = {}
    for folder in (e, p, output / "tools", output / "examples", output / "validation"):
        for path in folder.rglob("*"):
            if path.is_file(): files[path.relative_to(output).as_posix()] = dict(size=path.stat().st_size, sha256=digest(path))
    metadata["files"] = files
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Large NVIDIA binaries are delivered in their own optional archive.
    for folder in (e, p):
        zip_directory(folder, output.parent / (folder.name + "_" + metadata["delivery_version"] + "_candidate.zip"))
    print(output)


if __name__ == "__main__":
    main()
