"""Verify installed dependency trees without downloading or rebuilding them."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from release_manifest import digest, safe_member

ROOT = Path(__file__).resolve().parents[1]
STATE = '.cq-cache.json'


def recipe_id(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def cache(root, group, seal=False):
    lock = json.loads((root / 'configs/dependencies.lock.json').read_text(encoding='utf-8'))
    spec = lock['groups'][group]
    folder = safe_member(root, spec['directory'])
    state = folder / STATE
    if seal:
        files = {p.relative_to(folder).as_posix(): {'size': p.stat().st_size, 'sha256': digest(p)}
                 for p in folder.rglob('*') if p.is_file() and p.name != STATE}
        for name in spec['required']:
            if name not in files:
                raise ValueError(f'Missing required dependency: {group}/{name}')
        state.write_text(json.dumps({'recipe': recipe_id(spec), 'files': files}, indent=2), encoding='utf-8')
    data = json.loads(state.read_text(encoding='utf-8'))
    if data['recipe'] != recipe_id(spec):
        raise ValueError('Dependency recipe changed')
    if not set(spec['required']).issubset(data['files']):
        raise ValueError('Incomplete dependency state')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != STATE}
    if actual != set(data['files']):
        raise ValueError('Dependency tree changed')
    for name, expected in data['files'].items():
        p = safe_member(folder, name)
        if p.stat().st_size != expected['size'] or digest(p) != expected['sha256']:
            raise ValueError(f'Dependency corrupted: {group}/{name}')
    return {'group': group, 'cache_hit': True, 'files': len(actual)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'seal'])
    p.add_argument('--group', required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    try:
        print(json.dumps(cache(a.root, a.group, a.action == 'seal')))
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({'cache_hit': False, 'reason': str(e)}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
