"""Prepare pinned official headers without installing CUDA or requiring a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

TRT_COMMIT = "94e2b9ef6d2cce74c76cdad499cca36cc4949197"  # NVIDIA v10.13.3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "third_party")
    parser.add_argument("--tensorrt-sdk", type=Path, help="Official TensorRT-10.13.3.9 SDK directory; recommended for exact parser header")
    args = parser.parse_args()
    root = args.root.resolve()
    headers = root / "nvidia-headers"
    headers.mkdir(parents=True, exist_ok=True)
    if args.tensorrt_sdk:
        source = args.tensorrt_sdk / "include"
        for path in source.glob("*.h"):
            shutil.copy2(path, headers / path.name)
    else:
        entries = json.load(urllib.request.urlopen(f"https://api.github.com/repos/NVIDIA/TensorRT/contents/include?ref={TRT_COMMIT}"))
        for entry in entries:
            if entry["name"].endswith(".h"):
                (headers / entry["name"]).write_bytes(urllib.request.urlopen(entry["download_url"]).read())
        parser_url = "https://raw.githubusercontent.com/onnx/onnx-tensorrt/10.13-GA/NvOnnxParser.h"
        (headers / "NvOnnxParser.h").write_bytes(urllib.request.urlopen(parser_url).read())
        (headers / "LICENSE.Apache-2.0.txt").write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/NVIDIA/TensorRT/{TRT_COMMIT}/LICENSE").read())
    downloads = root / "nvidia-download"
    downloads.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(downloads),
                    "nvidia-cuda-runtime-cu12==12.8.90", "nvidia-cuda-nvcc-cu12==12.8.93"], check=True)
    for name, folder in (("nvidia_cuda_runtime_cu12-12.8.90", "nvidia-cuda-runtime-12.8.90"), ("nvidia_cuda_nvcc_cu12-12.8.93", "nvidia-cuda-nvcc-12.8.93")):
        wheel = next(downloads.glob(name + "*win_amd64.whl"))
        with zipfile.ZipFile(wheel) as archive:
            destination = root / folder
            for member in archive.namelist():
                if not (destination / member).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(destination)
    crt = next((root / "nvidia-cuda-nvcc-12.8.93").rglob("crt"))
    shutil.copytree(crt, root / "nvidia-cuda-runtime-12.8.90/nvidia/cuda_runtime/include/crt", dirs_exist_ok=True)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in headers.glob("*.h")}
    (headers / "headers-manifest.json").write_text(json.dumps(dict(tensorrt_commit=TRT_COMMIT, cuda_runtime="12.8.90", cuda_nvcc="12.8.93", sha256=hashes), indent=2))
    print(headers)


if __name__ == "__main__":
    main()
