"""Run six serial ten-minute CV stress tests and sample owned-process memory."""
from __future__ import annotations
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


class Counters(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("faults",wintypes.DWORD)]+[
        (name,ctypes.c_size_t) for name in ("peak_ws","ws","peak_paged","paged",
            "peak_nonpaged","nonpaged","pagefile","peak_pagefile","private")]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--arch",choices=("x86","x64"),action="append")
    parser.add_argument("--threads",type=int,action="append")
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    for arch in args.arch or ("x86","x64"):
        build=ROOT/f"build/cv-cascade-{arch}-production"
        dll=build/("CQ_X86.dll" if arch=="x86" else "CQ_AI_x64.dll")
        sha=hashlib.sha256(dll.read_bytes()).hexdigest()
        for threads in args.threads or (1,4,16):
            report=out/f"cv-stress-{arch}-{threads}.json"
            if report.exists():raise FileExistsError(report)
            command=[str(build/"ai_engine_cv_transparent_acceptance.exe"),"--dll",str(dll),
                "--input",str(ROOT/"input"),"--output",str(report),"--scope","all",
                "--stress-seconds","600","--threads",str(threads)]
            print(f"START {arch}/{threads} callers",flush=True)
            rows=[];start=time.monotonic();last_update=0
            with report.with_suffix(".log").open("w",encoding="utf-8") as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                handle=kernel.OpenProcess(0x410,False,process.pid)
                if not handle:raise ctypes.WinError(ctypes.get_last_error())
                try:
                    while process.poll() is None:
                        counters=Counters();counters.cb=ctypes.sizeof(counters)
                        if psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):
                            elapsed=time.monotonic()-start;rows.append((elapsed,counters.private,counters.ws))
                            if elapsed-last_update>=60:
                                print(f"{arch}/{threads} elapsed={elapsed:.0f}s private={counters.private/2**20:.1f}MiB",flush=True)
                                last_update=elapsed
                        if time.monotonic()-start>900:
                            process.terminate();process.wait(timeout=30)
                            raise TimeoutError(f"Owned CV stress process hung: {arch}/{threads}")
                        time.sleep(5)
                finally:kernel.CloseHandle(handle)
            with report.with_suffix(".resources.csv").open("w",newline="",encoding="utf-8") as file:
                writer=csv.writer(file);writer.writerow(("elapsed_s","private_bytes","working_set"));writer.writerows(rows)
            if process.returncode:raise RuntimeError(f"Stress failed {arch}/{threads}: {process.returncode}")
            measured=json.loads(report.read_text(encoding="utf-8"))
            if measured["dll_sha256"]!=sha or hashlib.sha256(dll.read_bytes()).hexdigest()!=sha:
                raise ValueError("Stress DLL changed")
            # Sampling begins during frame construction. Allow180s for templates,
            # thread-pool startup and the bounded10000-sample lane buffers.
            stable=[r[1] for r in rows if r[0]>=180]
            if len(stable)<24:raise ValueError("Insufficient memory samples")
            first=sum(stable[:12])/12;last=sum(stable[-12:])/12
            measured["memory_samples_after_warmup"]=len(stable)
            measured["initial_mean_private_bytes"]=first
            measured["final_mean_private_bytes"]=last
            measured["memory_growth_bytes"]=last-first
            measured["memory_growth_detected"]=last-first>2*1024*1024
            measured["memory_warmup_seconds"]=180
            report.write_text(json.dumps(measured,indent=2)+"\n",encoding="utf-8")
            if measured["memory_growth_detected"]:raise ValueError("CV sustained memory growth exceeds2MiB")
            print(f"PASS {arch}/{threads}: calls={measured['calls']}, errors={measured['errors']}, memory_growth={last-first:.0f}B",flush=True)


if __name__=="__main__":main()
