"""Alternate old/new five-lane measurements on the same owned-process CPU set.

This diagnostic is for noisy hosts. It neither changes system affinity nor
terminates business workers. Run x64 direct binaries after other timed work.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--affinity',type=lambda s:int(s,0),required=True)
    p.add_argument('--old-version',required=True,help='Expected AI_GetVersion result')
    p.add_argument('--new-version',required=True,help='Expected AI_GetVersion result')
    p.add_argument('--old-sha256',required=True)
    p.add_argument('--new-sha256',required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype=ctypes.c_int
    rows=[]
    for mode in ('continuous','simultaneous'):
        for repeat in range(3):
            # Reverse alternate pairs to reduce ordering/thermal bias.
            pair=('old','new') if repeat%2==0 else ('new','old')
            measured={}
            for tag in pair:
                report=a.out/f'{mode}-{repeat}-{tag}.json'
                command=[str(getattr(a,tag).resolve()),'--model',str(a.model.resolve()),
                    '--image',str(a.image.resolve()),'--output',str(report.resolve()),
                    '--device','2','--sessions','5','--callers','5','--threads','1',
                    '--warmup','100','--samples','1000','--rounds','1','--mode',mode,
                    '--shutdown-worker','0']
                with report.with_suffix('.log').open('w',encoding='utf-8') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                    if not kernel.SetProcessAffinityMask(int(child._handle),a.affinity):
                        # Own child only; never a Worker or arbitrary process.
                        child.terminate();child.wait()
                        raise ctypes.WinError(ctypes.get_last_error())
                    if child.wait(timeout=600):raise RuntimeError(f'Benchmark failed: {tag}, {mode}')
                result=json.loads(report.read_text(encoding='utf-8'))
                if result['api_version'] != getattr(a,tag+'_version') or result['dll_sha256'].lower() != getattr(a,tag+'_sha256').lower():
                    raise RuntimeError(f'Wrong {tag} baseline DLL version/hash; comparison rejected')
                assert result['rounds'][0]['errors']==0
                assert all(l['count']==1000 for l in result['rounds'][0]['lanes'])
                measured[tag]=result
            ratios={metric:measured['new']['rounds'][0]['overall'][metric]/measured['old']['rounds'][0]['overall'][metric]
                    for metric in ('p50_ms','p95_ms')}
            for metric in ('p50_ms','p95_ms'):
                ratios['worst_lane_'+metric]=max(x[metric] for x in measured['new']['rounds'][0]['lanes'])/max(x[metric] for x in measured['old']['rounds'][0]['lanes'])
            rows.append(dict(mode=mode,repeat=repeat,ratios=ratios,
                             old=measured['old']['rounds'][0]['overall'],new=measured['new']['rounds'][0]['overall']))
            print(mode,repeat,ratios,flush=True)
    aggregates={mode:{metric:statistics.median(x['ratios'][metric] for x in rows if x['mode']==mode)
                       for metric in ('p50_ms','p95_ms','worst_lane_p50_ms','worst_lane_p95_ms')} for mode in ('continuous','simultaneous')}
    result=dict(affinity_mask=hex(a.affinity),method='alternating five-lane x64 processes, same CPU set; no concurrent timed test',
                old_version=a.old_version,new_version=a.new_version,
                old_dll_sha256=measured['old']['dll_sha256'],new_dll_sha256=measured['new']['dll_sha256'],
                rounds=rows,median_ratios=aggregates,passed=all(v<=1.05 for m in aggregates.values() for v in m.values()))
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['median_ratios']),flush=True)
    if not result['passed']:raise SystemExit('Repeatable regression exceeds5%; inspect the measured stages before delivery')


if __name__=='__main__':main()
