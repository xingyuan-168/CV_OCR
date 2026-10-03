"""Inventory, retain evidence and prepare explicitly registered cache cleanup."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from release_manifest import current_manifest, digest, safe_member, verify_current, verify_file

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outpush/governance-v23.6'
DATA = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.onnx', '.pt', '.pth'}
EVIDENCE = {'.json', '.jsonl', '.csv', '.log', '.txt', '.md', '.ini', '.yaml', '.yml'}


def info(p):
    return {'size': p.stat().st_size, 'sha256': digest(p)}


def files(folder):
    if folder.is_file():
        yield folder
        return
    if not folder.exists():
        return
    for directory, dirs, names in os.walk(folder, followlinks=False):
        for name in dirs + names:
            p = Path(directory) / name
            if p.is_symlink() or (p.stat().st_file_attributes if os.name == 'nt' else 0) & 1024:
                raise ValueError(f'Reparse point in cleanup tree: {p}')
        for name in names:
            yield Path(directory) / name


def inventory(root):
    sizes = {}
    count = 0
    for p in files(root):
        parts = p.relative_to(root).parts
        for n in range(1, min(3, len(parts)) + 1):
            k = '/'.join(parts[:n]); sizes[k] = sizes.get(k, 0) + p.stat().st_size
        count += 1
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'files': count,
            'total_bytes': sum(v for k, v in sizes.items() if '/' not in k), 'sizes': sizes}


def copy_verified(source, destination):
    expected = info(source)
    if destination.exists():
        verify_file(destination, expected)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        shutil.copy2(source, partial)
        verify_file(partial, expected)
        partial.replace(destination)
    return expected


def verify_nvidia(path, expected):
    verify_file(path, expected)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate NVIDIA ZIP entries')
        for name in names:
            safe_member(path.parent, name.rstrip('/'))
        if archive.testzip():
            raise ValueError('NVIDIA ZIP CRC failed')
        m = json.loads(archive.read('manifest.json'))
        import hashlib
        members = dict(m['dependencies']['files'])
        members['CQ_YOLO_TensorRT.dll'] = m['module']
        for name, fingerprint in members.items():
            h = hashlib.sha256(); size = 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk); size += len(chunk)
            if size != fingerprint['size'] or h.hexdigest() != fingerprint['sha256']:
                raise ValueError(f'NVIDIA ZIP member hash failed: {name}')


def prepare(archive_dir, builds=False):
    AUDIT.mkdir(parents=True, exist_ok=True)
    verify_current(ROOT, ROOT / 'output')
    baseline = json.loads((AUDIT / 'protected-before.json').read_text(encoding='utf-8'))
    for cohort in baseline['protected'].values():
        for name, expected in cohort.items(): verify_file(safe_member(ROOT, name), expected)
    policy = json.loads((ROOT / 'configs/storage-policy.json').read_text(encoding='utf-8'))
    names = policy['build_targets'] if builds else policy['scratch_targets']
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in names:
        if any(t == name or t.startswith(name + '/') for t in filter(None, tracked)):
            raise ValueError(f'Registered cleanup target contains tracked files: {name}')
    protected = {}
    for name in ('input', 'output', 'release', 'outpush/rollback', 'models'):
        for p in files(ROOT / name): protected[p.relative_to(ROOT).as_posix()] = info(p)
    for cohort in baseline['protected'].values(): protected.update(cohort)
    retained = {}
    _, current = current_manifest(ROOT)
    nv = current['external_archives']['nvidia_zip']
    nv_dest = archive_dir / nv['file']
    if not nv_dest.exists(): copy_verified(ROOT / 'outpush' / nv['file'], nv_dest)
    verify_nvidia(nv_dest, nv)
    retained[str(nv_dest)] = info(nv_dest)
    nv_manifest = ROOT / 'outpush' / Path(nv['file']).with_suffix('.manifest.json')
    if nv_manifest.exists(): retained[str(archive_dir / nv_manifest.name)] = copy_verified(nv_manifest, archive_dir / nv_manifest.name)
    known = {}
    for base in ('input', 'models', 'tests/fixtures'):
        for p in files(ROOT / base):
            if p.suffix.lower() in DATA: known.setdefault(digest(p), str(p))
    evidence_index = []
    targets = []
    for name in names:
        target = safe_member(ROOT, name)
        entries = []
        for p in files(target):
            relative = p.relative_to(ROOT).as_posix()
            entry = {'path': relative, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            if name.startswith('outpush/') and p.suffix.lower() in EVIDENCE:
                dest = AUDIT / 'retained-evidence' / relative
                expected = copy_verified(p, dest)
                retained[str(dest)] = expected
                entry['sha256'] = expected['sha256']
                evidence_index.append({'source': relative, 'retained': str(dest), **expected})
            elif name.startswith('outpush/') and p.suffix.lower() in DATA:
                fingerprint = digest(p); entry['sha256'] = fingerprint
                if fingerprint not in known:
                    dest = archive_dir / 'unique-fixtures' / (fingerprint + p.suffix.lower())
                    retained[str(dest)] = copy_verified(p, dest); known[fingerprint] = str(dest)
                evidence_index.append({'source': relative, 'retained': known[fingerprint], 'sha256': fingerprint})
            if p.name == 'CMakeCache.txt':
                dest = AUDIT / 'retained-evidence' / relative
                retained[str(dest)] = copy_verified(p, dest)
            entries.append(entry)
        targets.append({'path': name, 'bytes': sum(e['size'] for e in entries), 'files': entries})
    plan = {'schema': 1, 'root': str(ROOT), 'policy_sha256': digest(ROOT / 'configs/storage-policy.json'),
            'targets': targets, 'protected': protected, 'retained': retained,
            'nvidia_external': {'path': str(nv_dest), **info(nv_dest)}, 'evidence_index': evidence_index}
    path = AUDIT / ('cleanup-builds.json' if builds else 'cleanup-plan.json')
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    if not (AUDIT / 'cleanup-before.json').exists():
        (AUDIT / 'cleanup-before.json').write_text(json.dumps(inventory(ROOT), indent=2), encoding='utf-8')
    print(json.dumps({'plan': str(path), 'delete_bytes': sum(t['bytes'] for t in targets), 'retained_nvidia_bytes': nv['size']}))


def verify_plan(path):
    plan = json.loads(path.read_text(encoding='utf-8-sig'))
    if plan['schema'] != 1 or Path(plan['root']).resolve() != ROOT:
        raise ValueError('Wrong cleanup workspace')
    if digest(ROOT / 'configs/storage-policy.json') != plan['policy_sha256']:
        raise ValueError('Cleanup policy changed')
    policy = json.loads((ROOT / 'configs/storage-policy.json').read_text(encoding='utf-8'))
    allowed = set(policy['scratch_targets'] + policy['build_targets'])
    for t in plan['targets']:
        if t['path'] not in allowed: raise ValueError('Unregistered cleanup target')
        folder = safe_member(ROOT, t['path'])
        if folder.exists():
            actual = {p.relative_to(ROOT).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in files(folder)}
            if actual != {e['path']: (e['size'], e['mtime_ns']) for e in t['files']}:
                raise ValueError(f'Cleanup target changed: {t["path"]}')
    for name, expected in plan['protected'].items(): verify_file(safe_member(ROOT, name), expected)
    for name, expected in plan['retained'].items(): verify_file(Path(name), expected)
    print(json.dumps({'verified': True, 'protected_files': len(plan['protected'])}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'verify-plan', 'measure'])
    p.add_argument('--archive-dir', type=Path, default=Path('D:/Projects-tools/交付归档/CV_OCR/v23.6'))
    p.add_argument('--include-builds', action='store_true')
    p.add_argument('--plan', type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.archive_dir.resolve(), a.include_builds)
    elif a.action == 'verify-plan': verify_plan(a.plan.resolve())
    else:
        result = inventory(ROOT)
        if a.report: a.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'bytes': result['total_bytes'], 'gib': result['total_bytes'] / 2**30, 'files': result['files']}))


if __name__ == '__main__':
    main()
