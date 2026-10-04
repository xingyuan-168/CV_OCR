"""Validate paired x86 path/memory loads against recorded x64 detections.

Use only a disposable test runtime; the verifier confirms the absolute Worker
path before closing its own Worker. Never point this at a running business app.
"""
import argparse
import json
from pathlib import Path
import subprocess


def execute(command, log):
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,timeout=240)
    if result.returncode:raise RuntimeError('x86 validation failed: '+str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--header',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--bus-bmp',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    runtime=a.runtime.resolve();cases=json.loads(a.cases.read_text(encoding='utf-8'))
    refs={(r['case'],r['device']):r for r in json.loads(a.reference.read_text(encoding='utf-8'))['cases']}
    reports=[]
    for case in cases:
        image=Path(case['images'][0]);image=a.bus_bmp if image.suffix.lower() in ('.jpg','.jpeg') else image
        for device in (2,1):
            name=f"{case['name']}-{device}"
            expected=a.out/(name+'-expected.json');expected.write_text(refs[(case['name'],device)]['results'][0],encoding='utf-8')
            report=a.out/(name+'-path.json')
            execute([runtime/'ai_engine_delivery_verify.exe','--dll',runtime/'CQ_X86.dll',
                '--header',a.header.resolve(),'--model',case['model'],'--image',image.resolve(),
                '--device',device,'--input-size',case.get('input_size',0),'--expected-json',expected.resolve(),
                '--report',report.resolve()],a.out/(name+'-path.log'))
            record=json.loads(report.read_text(encoding='utf-8'));assert record['errors']==0
            assert record['runtime']['active']==('cpu' if device==2 else 'directml')
            if not case.get('input_size'):
                # Benchmark uses the public memory-loading entry point.
                detections=a.out/(name+'-memory-detections.json')
                execute([runtime/'ai_engine_yolo_benchmark.exe','--model',case['model'],'--image',image.resolve(),
                    '--device',device,'--sessions',2,'--callers',5,'--threads',1,'--warmup',2,'--samples',10,
                    '--rounds',1,'--mode','continuous','--verify-output',detections.resolve(),
                    '--output',(a.out/(name+'-memory.json')).resolve()],a.out/(name+'-memory.log'))
                assert json.loads(detections.read_text(encoding='utf-8'))[0]['detections']==record['detections']
            reports.append(dict(case=case['name'],device=device,path=True,memory=not bool(case.get('input_size')),
                                dll_sha256=record['dll_sha256'],worker_sha256=record['worker_sha256'],
                                detections=record['detections'],runtime=record['runtime'],errors=0))
            print('Passed',name,flush=True)
    (a.out.parent/'compatibility-x86.json').write_text(json.dumps(dict(passed=True,cases=reports),indent=2),encoding='utf-8')


if __name__=='__main__':main()
