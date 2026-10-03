"""Portable, fail-closed checks for immutable CQ_AI delivery manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import zipfile

CORE_NAMES = {"CQ_X86.dll", "CQ_AI_worker.exe", "易语言_DLL_API_说明.html"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def safe_member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid relative path: {relative!r}")
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or PureWindowsPath(relative).drive:
        raise ValueError(f"Path escapes manifest root: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError(f"Path escapes manifest root: {relative}")
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def current_manifest(root: Path) -> tuple[Path, dict]:
    pointer = read_json(root / "release/current.json")
    if pointer.get("schema") != 1:
        raise ValueError("Unsupported current-release pointer")
    path = safe_member(root / "release", pointer["manifest"])
    manifest = read_json(path)
    if manifest.get("schema") != 2:
        raise ValueError("Current delivery must use manifest schema2")
    return path, manifest


def verify_file(path: Path, expected: dict) -> None:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        raise ValueError(f"Missing file or incorrect size: {path}")
    if digest(path) != expected["sha256"].lower():
        raise ValueError(f"SHA-256 differs: {path}")


def verify_cohort(folder: Path, files: dict) -> None:
    if set(files) != CORE_NAMES or {p.name for p in folder.iterdir()} != CORE_NAMES:
        raise ValueError("Output must contain exactly the three declared delivery files")
    for name, expected in files.items():
        verify_file(safe_member(folder, name), expected)


def verify_archive(path: Path, expected: dict, members: dict | None = None) -> None:
    verify_file(path, expected)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP members: {path}")
        for name in names:
            safe_member(path.parent, name.rstrip("/"))
        if archive.testzip():
            raise ValueError(f"ZIP CRC failure: {path}")
        if members is not None:
            if set(names) != set(members):
                raise ValueError(f"ZIP cohort differs: {path}")
            for name, info in members.items():
                data = archive.read(name)
                if len(data) != info["size"] or hashlib.sha256(data).hexdigest() != info["sha256"].lower():
                    raise ValueError(f"ZIP member differs: {name}")


def verify_sources(root: Path, manifest: dict) -> None:
    expected = manifest["git_source_sha256"]
    for relative, fingerprint in expected.items():
        data = safe_member(root, relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != fingerprint:
            raise ValueError(f"Native source differs from registered delivery: {relative}")
    header = (root / "include/ai_engine.h").read_text(encoding="utf-8")
    version = ".".join(re.search(rf"#define\s+AIENGINE_VERSION_{key}\s+(\d+)", header).group(1) for key in ("MAJOR", "MINOR", "PATCH"))
    protocol = re.search(r"kVersion\s*=\s*(\d+)", (root / "src/worker_protocol.h").read_text(encoding="utf-8")).group(1)
    if version != manifest["project_version"] or int(protocol) != manifest["worker_protocol"]:
        raise ValueError("Header version or Worker protocol differs from manifest")


def verify_current(root: Path, output: Path | None = None) -> dict:
    path, manifest = current_manifest(root)
    if manifest["public_exports"] != 60 or set(manifest["files"]) != CORE_NAMES:
        raise ValueError("Public ABI or delivery cohort differs")
    verify_sources(root, manifest)
    for kind, artifact in manifest["artifacts"].items():
        verify_archive(safe_member(path.parent, artifact["file"]), artifact,
                       manifest["files"] if kind == "easy_language_zip" else None)
    for evidence in manifest["evidence"].values():
        verify_file(safe_member(path.parent, evidence["file"]), evidence)
    delivery = read_json(safe_member(path.parent, manifest["evidence"]["delivery-build.json"]["file"]))
    if delivery["worker_protocol"] != manifest["worker_protocol"] or delivery["public_exports"] != 60 or delivery["errors"]:
        raise ValueError("Registered delivery verification did not pass")
    for name, key in (("CQ_X86.dll", "dll_sha256"), ("CQ_AI_worker.exe", "worker_sha256")):
        if delivery[key] != manifest["files"][name]["sha256"]:
            raise ValueError("Evidence belongs to a different binary cohort")
    if manifest["target_performance_validation"] == "pending" and manifest["formal_release"]:
        raise ValueError("Pending target performance cannot be certified as formal release")
    if output is not None:
        verify_cohort(output, manifest["files"])
    return {"version": manifest["project_version"], "delivery": manifest["delivery_version"],
            "protocol": manifest["worker_protocol"], "source_files": len(manifest["git_source_sha256"]),
            "functional_validation": manifest["functional_validation"],
            "target_performance": manifest["target_performance_validation"]}


def verify_historical(root: Path) -> None:
    base = root / "release/v23.5"
    manifest = read_json(base / "manifest.json")
    expected = {a["file"] for a in manifest["artifacts"]} | {"manifest.json"}
    if {p.name for p in base.iterdir()} != expected:
        raise ValueError("Historical release directory differs")
    for artifact in manifest["artifacts"]:
        members = {m["file"]: m for m in artifact.get("members", [])} or None
        verify_archive(safe_member(base, artifact["file"]), artifact, members)
