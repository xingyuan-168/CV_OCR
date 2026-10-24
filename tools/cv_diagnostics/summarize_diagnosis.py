"""Freeze local diagnostic evidence; these short runs are not release acceptance."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'outpush/cv-transparent-mask-fix'
RAW = TASK / 'performance-diagnosis'
ARCHIVE = Path('D:/codex-delivery/CV_OCR/cv-transparent-mask-fix-20261023/build')

def fingerprint(path):
    data = path.read_bytes()
    return {'path': str(path), 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'status': 'diagnosis_only_not_5ms_acceptance', 'limit_ms': 5, 'architectures': {}, 'files': {}}
    for arch, dllname in [('x86', 'CQ_X86.dll'), ('x64', 'CQ_AI_x64.dll')]:
        current = json.loads((TASK / f'cv-timing-{arch}.json').read_text())
        old = json.loads((TASK / f'baseline-{arch}.json').read_text())
        production = ARCHIVE / f'cv-mask-{arch}-production' / dllname
        assert fingerprint(production)['sha256'] == current['dll_sha256']
        old_dll = ROOT/'output/CQ_X86.dll' if arch=='x86' else Path('D:/codex-diagnostics/CV_OCR/20261016/x64/CQ_AI_x64.dll')
        assert fingerprint(old_dll)['sha256'] == old['dll_sha256']
        hooks = json.loads((RAW / f'hook-{arch}.json').read_text())
        hook_dll = ARCHIVE / f'cv-mask-{arch}-test' / dllname
        assert fingerprint(hook_dll)['sha256'] == hooks['dll_sha256']
        assert hooks['test_hooks'] and len(hooks['cases']) == 13
        log = (RAW / f'hook-{arch}.log').read_text(encoding='utf-8-sig')
        profiles = {}
        for name, bound, exact, count in re.findall(r'(\w+) round=.*?\n\s+bounds=([\d.]+) exact=([\d.]+) candidates=(\d+)', log):
            profiles[name] = {'last_call_bounds_ms': float(bound), 'last_call_exact_ms': float(exact), 'last_call_candidates': int(count)}
        assert len(profiles) == 13
        phases = [json.loads(s) for s in (RAW / f'phase-audit-{arch}.jsonl').read_text().splitlines()]
        assert len(phases) == 6 and all(p['arch']==arch for p in phases)
        result['architectures'][arch] = {'formal_old': old, 'formal_candidate': current, 'hook_short_run': hooks,
                                         'hook_last_call_profiles': profiles, 'phase_warmup': 8, 'phase_samples': 64, 'phases': phases}
        for p in (production, old_dll, hook_dll): result['files'][str(p)] = fingerprint(p)
    for p in [*sorted(RAW.glob('*.json*')), *sorted(RAW.glob('*.log')), RAW/'phase_probe.cpp', RAW/'build-phase.ps1',
              ROOT/'src/ai_engine.cpp', ROOT/'src/cv_correlation.cpp', ROOT/'src/cv_masked_score_avx2.cpp',
              *[ROOT/'input'/s for s in ('1.bmp','2.bmp','xyq.bmp','CV_123.bmp','CV_大图.bmp')],
              *sorted((ROOT/'output').iterdir()), ROOT/'release/current.json']:
        result['files'][str(p)] = fingerprint(p)
    path=ROOT/'docs/evidence/CV_TRANSPARENT_PERFORMANCE_DIAGNOSIS_20261023.json'
    if path.exists(): raise RuntimeError('Do not overwrite a frozen diagnosis')
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(path)

if __name__ == '__main__': main()
