"""Register verified packages once; never replace a historical delivery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from package_common import copy
from release_manifest import digest, safe_member, verify_archive, verify_sources, verify_current

ROOT = Path(__file__).resolve().parents[1]


def register(package, wheel, select=False):
    m = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    version = m['delivery_version']
    import re
    if not re.fullmatch(r'v\d+\.\d+(?:\.\d+)?', version): raise ValueError('Invalid delivery version')
    base = ROOT / 'release' / version
    if base.exists(): raise ValueError('Registered releases are immutable; choose a new delivery version')
    if 'git_source_sha256' not in m: raise ValueError('Portable normalized source fingerprints are required')
    verify_sources(ROOT, m)
    artifact = m['artifacts']['easy_language_zip']
    easy = Path(artifact['path'])
    if not easy.resolve().is_relative_to(package.resolve()): raise ValueError('Package archive escapes staging')
    verify_archive(easy, artifact, m['files'])
    wheel_info = {'file':wheel.name,'size':wheel.stat().st_size,'sha256':digest(wheel)}
    verify_archive(wheel, wheel_info)
    # Validate evidence before creating a permanent directory.
    from package_delivery import verified_reports
    from types import SimpleNamespace
    validation = Path(m['validation_reports']['delivery-build.json']['path']).parent
    verified_reports(SimpleNamespace(validation=validation),m['files'],m)
    stage = ROOT / 'outpush' / ('register-' + version)
    if stage.exists(): raise ValueError('Registration staging already exists')
    stage.mkdir(parents=True)
    try:
        copy(easy,stage/easy.name); copy(wheel,stage/wheel.name)
        m['schema']=2
        m['artifacts']={'easy_language_zip':dict(file=easy.name,size=easy.stat().st_size,sha256=digest(easy)), 'python_wheel':wheel_info}
        m['evidence']={}
        for name in ['delivery-build.json','continuous.json','simultaneous.json','soak-30min.json','summary.json','frozen-detections.json','html-interactions.json']:
            dest=stage/'evidence'/name; copy(validation/name,dest)
            m['evidence'][name]=dict(file='evidence/'+name,size=dest.stat().st_size,sha256=digest(dest))
        m['formal_release']=m['target_performance_validation']=='passed'
        (stage/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        stage.rename(base)
    except BaseException:
        if stage.exists(): shutil.rmtree(stage)
        raise
    if select:
        pointer=ROOT/'release/current.json'
        old=pointer.read_bytes()
        pointer.write_text(json.dumps({'schema':1,'manifest':version+'/manifest.json'},indent=2),encoding='utf-8')
        try: verify_current(ROOT)
        except BaseException: pointer.write_bytes(old); raise
    return base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir',type=Path,required=True)
    p.add_argument('--wheel',type=Path,required=True)
    p.add_argument('--select-current',action='store_true')
    a=p.parse_args()
    package=a.package_dir.resolve()
    if not package.is_relative_to((ROOT/'outpush').resolve()): raise ValueError('Package staging must be below outpush')
    print(register(package,a.wheel.resolve(),a.select_current))


if __name__=='__main__': main()
