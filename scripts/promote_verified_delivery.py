"""Promote a verified cohort and current pointer together, with rollback."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import zipfile

from release_manifest import (CORE_NAMES, digest, verify_cohort, verify_current,
                              verify_current_delivery, verify_sources)

ROOT=Path(__file__).resolve().parents[1]


def exclusive_files(folder: Path) -> None:
    """Fail before any rename if a user's process has a deployed file mapped."""
    if os.name!="nt":return
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles=[]
    try:
        for name in sorted(CORE_NAMES):
            handle=kernel.CreateFileW(str(folder/name),0xC0000000,0,None,3,0,None)
            if handle==ctypes.c_void_p(-1).value:
                raise OSError(f"Output occupied or not writable; preserved: {folder/name}")
            handles.append(handle)
    finally:
        for handle in handles:kernel.CloseHandle(handle)


def promote_transaction(root: Path, prepared: Path, files: dict, version: str, post_check) -> dict:
    root,prepared=root.resolve(),prepared.resolve()
    if not re.fullmatch(r"v\d+\.\d+(?:\.\d+)?",version):raise ValueError("Invalid delivery version")
    output=root/"output";pointer=root/"release/current.json"
    if (not prepared.is_relative_to(root/"outpush") or prepared==root/"outpush"
            or output.is_symlink() or output.resolve()!=output
            or pointer.is_symlink() or pointer.resolve()!=pointer):
        raise ValueError("Promotion paths must be real workspace output/outpush paths")
    verify_cohort(prepared,files)
    if ({p.name for p in output.iterdir()}!=CORE_NAMES or
            any(not p.is_file() or p.is_symlink() for p in output.iterdir())):
        raise ValueError("Preserve unexpected output contents")
    exclusive_files(output)
    old_pointer=pointer.read_bytes()
    old_files={p.name:dict(size=p.stat().st_size,sha256=digest(p)) for p in output.iterdir()}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    rollback=root/"outpush/rollback"/f"before-{version}-{stamp}"
    rejected=root/"outpush"/f"rejected-{version}-{stamp}"
    # Resolve and check each recursive directory move before mutating anything.
    for path in (rollback,rejected):
        if not path.resolve().is_relative_to(root/"outpush") or path.exists():
            raise ValueError("Invalid or existing rollback destination")
    rollback.parent.mkdir(parents=True,exist_ok=True)
    rollback.with_suffix(".json").write_text(json.dumps(dict(pointer=old_pointer.decode("utf-8-sig"),
        files=old_files),indent=2),encoding="utf-8")
    output.rename(rollback)
    temporary=pointer.with_name("current.promotion.tmp")
    temporary_owned=False
    try:
        prepared.rename(output)
        new_pointer=json.dumps(dict(schema=1,manifest=f"{version}/manifest.json"),indent=2).encode()
        with temporary.open("xb") as stream:
            temporary_owned=True
            stream.write(new_pointer)
        os.replace(temporary,pointer)
        verify_cohort(output,files)
        post_check()
    except BaseException:
        if temporary_owned and temporary.exists():temporary.unlink()
        if output.exists():output.rename(rejected)
        rollback.rename(output)
        pointer.write_bytes(old_pointer)
        verify_cohort(output,old_files)
        if pointer.read_bytes()!=old_pointer:raise RuntimeError("Pointer rollback failed")
        raise
    return dict(output=str(output),rollback=str(rollback),version=version,verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--wheel",type=Path,required=True)
    parser.add_argument("--verifier",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--post-check-json",type=Path,required=True,help="JSON argv for the real Finish gate")
    args=parser.parse_args();package=args.package.resolve()
    if not package.is_relative_to(ROOT/"outpush"):raise ValueError("Package must be staged under outpush")
    metadata=json.loads((package/"manifest.json").read_text(encoding="utf-8-sig"))
    verify_current_delivery(ROOT);verify_sources(ROOT,metadata)
    from register_release import register
    registered=register(package,args.wheel.resolve())
    version=registered.name
    prepared=package/"promotion-core"
    prepared.mkdir()
    archive=Path(metadata["artifacts"]["easy_language_zip"]["path"])
    with zipfile.ZipFile(archive) as zip_file:
        for name in CORE_NAMES:(prepared/name).write_bytes(zip_file.read(name))
    validation=Path(metadata["validation_reports"]["delivery-build.json"]["path"]).parent
    finish_command=json.loads(args.post_check_json.read_text(encoding="utf-8-sig"))
    if not isinstance(finish_command,list) or not all(isinstance(x,str) for x in finish_command):
        raise ValueError("Post-check must be an argv array")
    def post_check():
        command=[str(args.verifier.resolve()),"--dll",str(ROOT/"output/CQ_X86.dll"),
            "--header",str(ROOT/"include/ai_engine.h"),"--model",str(args.model.resolve()),
            "--image",str(args.image.resolve()),"--expected-json",str(validation/"frozen-detections.json"),
            "--report",str(validation/"delivery-output.json"),"--preserve-worker","0"]
        with (validation/"delivery-output.log").open("w",encoding="utf-8") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_current(ROOT)
        result=subprocess.run(finish_command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        (validation/"finish-current.json").write_text(result.stdout,encoding="utf-8")
        (validation/"finish-current.stderr.log").write_text(result.stderr,encoding="utf-8")
        result.check_returncode()
        if json.loads(result.stdout).get("ok") is not True:raise RuntimeError("Actual Finish gate rejected promotion")
    result=promote_transaction(ROOT,prepared,metadata["files"],version,post_check)
    (validation/"promotion.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
