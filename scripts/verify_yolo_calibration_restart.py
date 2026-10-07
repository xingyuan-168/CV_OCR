"""Verify calibration identity across lazy DirectML extraction and Worker restart."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time


def wait_worker_exit(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # Worker already exited.
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if kernel.WaitForSingleObject(handle, 15000) != 0:
            raise RuntimeError('Worker did not exit after its test owner; no process was terminated')
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'model', 'image', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, LOCALAPPDATA=str(folder / 'localappdata'),
                       CQ_AI_YOLO_TRACE_PREFIX='', CQ_AI_ORT_PROFILE_PREFIX='')
    reports, cached = {}, {}
    for name, device in (('cold-cpu', 2), ('extract-directml', 1), ('restarted-cpu', 2)):
        path = folder / (name + '.json')
        command = [str(args.benchmark.resolve()), '--model', str(args.model.resolve()), '--image', str(args.image.resolve()),
                   '--device', str(device), '--sessions', '2', '--threads', '1', '--callers', '1', '--warmup', '0',
                   '--samples', '2', '--rounds', '1', '--shutdown-worker', '0', '--output', str(path)]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        (folder / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name}: {result.stderr}')
        reports[name] = json.loads(path.read_text(encoding='utf-8-sig'))
        cached[name] = (Path(reports[name]['runtime']['ort_path']).parent / 'DirectML.dll').exists()
        pid = reports[name].get('worker_pid')
        if pid:
            wait_worker_exit(pid)
        else:
            # Earlier benchmark versions did not report the Worker PID.
            time.sleep(5)
    keys = {name: report['runtime']['calibration_key'] for name, report in reports.items()}
    passed = (not cached['cold-cpu'] and cached['extract-directml'] and cached['restarted-cpu']
              and len(set(keys.values())) == 1
              and all(not r['errors'] for report in reports.values() for r in report['rounds']))
    pids = [report.get('worker_pid') for report in reports.values()]
    fresh_pids = all(pids) and len(set(pids)) == len(pids)
    if all(pids):
        passed = passed and fresh_pids
    summary = dict(passed=passed, keys=keys, directml_cached=cached, worker_pids=pids, fresh_worker_pids=bool(fresh_pids),
                   scope='Owned LOCALAPPDATA; cold CPU, DirectML extraction, fresh CPU benchmark/Worker',
                   dll_sha256=reports['cold-cpu']['dll_sha256'], worker_sha256=reports['cold-cpu']['worker_sha256'])
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
