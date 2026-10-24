"""Measure the first matching call in a new process for every masked scenario."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    summary={}
    for arch in ("x86","x64"):
        formal=json.loads((out/f"cv-timing-{arch}.json").read_text(encoding="utf-8"))
        build=ROOT/f"build/cv-cascade-{arch}-production"
        dll=build/("CQ_X86.dll" if arch=="x86" else "CQ_AI_x64.dll")
        if hashlib.sha256(dll.read_bytes()).hexdigest()!=formal["dll_sha256"]:raise ValueError("DLL mismatch")
        cases={}
        for item in formal["cases"]:
            name=item["name"];report=out/f"cold-{arch}-{name}.json"
            with report.with_suffix(".log").open("w",encoding="utf-8") as log:
                result=subprocess.run([str(build/"ai_engine_cv_transparent_acceptance.exe"),
                    "--dll",str(dll),"--input",str(ROOT/"input"),"--output",str(report),
                    "--scope","all","--samples","1","--rounds","1","--case",name],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            if result.returncode not in (0,3):raise RuntimeError(f"Cold-call correctness failure {arch}/{name}")
            data=json.loads(report.read_text(encoding="utf-8"))
            if data["dll_sha256"]!=formal["dll_sha256"] or len(data["cases"])!=1:raise ValueError("Wrong cold DLL/case")
            cases[name]=data["cases"][0]["cold_ms"]
        summary[arch]=dict(dll_sha256=formal["dll_sha256"],first_matching_call_ms=cases)
    (out/"cv-cold-summary.json").write_text(json.dumps(dict(
        scope="Each case in a new process; file read and template loading outside call; no5ms cold-call gate",
        architectures=summary),indent=2),encoding="utf-8")


if __name__=="__main__":main()
