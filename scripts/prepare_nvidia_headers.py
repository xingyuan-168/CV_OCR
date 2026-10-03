"""Restore exact SDK/CUDA headers from locked sources; reuse valid caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
from dependency_cache import cache
from release_manifest import verify_file, safe_member

ROOT = Path(__file__).resolve().parents[1]


def download(spec, folder):
    p = folder / spec['file']
    with urllib.request.urlopen(spec['url'], timeout=90) as response, p.open('wb') as stream:
        shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
    verify_file(p, dict(spec, size=spec.get('size', p.stat().st_size)))
    return p


def valid(group):
    try:
        cache(ROOT, group)
        return True
    except (OSError, ValueError, KeyError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'third_party')
    p.add_argument('--tensorrt-sdk', type=Path, help='Exact official SDK 10.13.3.9 directory')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if a.root.resolve() != (ROOT / 'third_party').resolve():
        raise ValueError('Dependency cache root is defined by the checked-in lock')
    lock = json.loads((ROOT / 'configs/dependencies.lock.json').read_text(encoding='utf-8'))
    header_hit = not a.force and valid('nvidia-headers')
    cuda_hit = not a.force and valid('cuda-headers')
    if header_hit and cuda_hit:
        print(json.dumps({'cache_hit':True,'downloaded_bytes':0}))
        return
    with tempfile.TemporaryDirectory(prefix='cq-nvidia-headers-') as temporary:
        temp = Path(temporary)
        if not header_hit:
            headers = ROOT / 'third_party/nvidia-headers'; headers.mkdir(parents=True, exist_ok=True)
            source = None if a.tensorrt_sdk else download(lock['tensorrt_sdk'], temp)
            archive = zipfile.ZipFile(source) if source else None
            try:
                for spec in lock['nvidia_header_downloads']:
                    dest = headers / spec['file']
                    if archive: dest.write_bytes(archive.read(spec['sdk_member']))
                    else: shutil.copy2(a.tensorrt_sdk / 'include' / spec['file'], dest)
                    verify_file(dest, spec)
                license_path = headers / 'NVIDIA_TensorRT_LICENSE.html'
                if archive: license_path.write_bytes(archive.read('TensorRT-10.13.3.9/doc/NVIDIA_TensorRT_LICENSE.html'))
                else: shutil.copy2(a.tensorrt_sdk / 'doc/NVIDIA_TensorRT_LICENSE.html', license_path)
            finally:
                if archive: archive.close()
            (headers / 'headers-manifest.json').write_text(json.dumps({'source':'TensorRT SDK 10.13.3.9','cuda_runtime':'12.8.90','cuda_nvcc':'12.8.93','sha256':{x['file']:x['sha256'] for x in lock['nvidia_header_downloads']}}, indent=2), encoding='utf-8')
            cache(ROOT, 'nvidia-headers', True)
        if not cuda_hit:
            destination = ROOT / 'third_party/nvidia-cuda-runtime-12.8.90'
            for spec in lock['nvidia_wheels']:
                if spec['package'] not in {'nvidia-cuda-runtime-cu12','nvidia-cuda-nvcc-cu12'}: continue
                wheel = download(spec, temp)
                with zipfile.ZipFile(wheel) as archive:
                    for name in archive.namelist():
                        if name.endswith('/'): continue
                        if spec['package'] == 'nvidia-cuda-runtime-cu12': output = safe_member(destination, name)
                        elif name.startswith('nvidia/cuda_nvcc/include/crt/'):
                            output = safe_member(destination, name.replace('nvidia/cuda_nvcc/include/', 'nvidia/cuda_runtime/include/', 1))
                        elif name.endswith('License.txt'): output = destination / 'licenses/nvcc-License.txt'
                        else: continue
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(archive.read(name))
            cache(ROOT, 'cuda-headers', True)
    print(json.dumps({'cache_hit':False,'restored':True}))


if __name__ == '__main__': main()
