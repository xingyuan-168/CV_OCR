"""Export pinned small official checkpoints in an isolated offline-test environment.

Use Python3.12, torch2.6.0+cpu, torchvision0.21.0+cpu, ultralytics8.4.28,
onnx1.19.1 and onnxruntime1.24.4. --work is disposable, never the repository.
The output manifest records checkpoint/source/export hashes and package versions.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    work, output = a.work.resolve(), a.output.resolve()
    if work == root or work.is_relative_to(root):
        raise ValueError('Export environments must stay outside the retained repository')
    if not output.is_relative_to((root / 'outpush').resolve()):
        raise ValueError('Reference outputs must be staged under outpush')
    if version('ultralytics') != '8.4.28' or version('onnxruntime') != '1.24.4':
        raise ValueError('Use the declared pinned export/reference environment')
    output.mkdir(parents=True, exist_ok=True)
    weights = work / 'weights'; weights.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(work / 'ultralytics-config')
    (work / 'ultralytics-config' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(work / 'torch-home')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'  # trusted pinned official legacy checkpoints
    lock_path = root / 'configs/yolo-reference-sources.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    sources = {}

    def fetch(url, path):
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                path.write_bytes(response.read())
        record = dict(url=url, sha256=digest(path), size=path.stat().st_size)
        if url not in lock or record != lock[url]:
            raise ValueError('Reference source differs from the pinned hash: ' + url)
        sources[url] = record
        return record

    archive = work / 'yolov5-v7.0.zip'
    source = fetch('https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip', archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (work / name).resolve().is_relative_to(work):
                raise ValueError('Source archive escapes the export work directory')
        z.extractall(work)
    records = []
    checkpoint = weights / 'yolov5n.pt'
    info = fetch('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt', checkpoint)
    command = [sys.executable, str(work / 'yolov5-7.0/export.py'), '--weights', str(checkpoint), '--imgsz', '320',
               '--batch-size', '1', '--include', 'onnx', '--opset', '12', '--device', 'cpu']
    result = subprocess.run(command, cwd=work / 'yolov5-7.0', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError('Official v5 export failed')
    file = output / 'yolov5n-fp32-320.onnx'
    file.write_bytes(checkpoint.with_suffix('.onnx').read_bytes())
    records.append(dict(file=file.name, checkpoint=info, export_source=source, sha256=digest(file), export_args=command[2:]))
    from ultralytics import YOLO
    for name in ('yolo11n', 'yolo26n'):
        checkpoint = weights / (name + '.pt')
        info = fetch('https://github.com/ultralytics/assets/releases/download/v8.4.0/' + name + '.pt', checkpoint)
        options = dict(format='onnx', imgsz=320, batch=1, half=False, dynamic=False, simplify=False, opset=18, device='cpu', nms=False)
        if name == 'yolo26n':
            options['end2end'] = False
        exported = Path(YOLO(str(checkpoint)).export(**options))
        file = output / (name + '-fp32-320' + ('-dense' if name == 'yolo26n' else '') + '.onnx')
        file.write_bytes(exported.read_bytes())
        records.append(dict(file=file.name, checkpoint=info, exporter='ultralytics8.4.28', export_args=options, sha256=digest(file)))
    (output / 'exports.json').write_text(json.dumps(dict(models=records, sources=sources,
        packages={x: version(x) for x in ('torch','torchvision','ultralytics','onnx','onnxruntime','numpy')}), indent=2), encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
