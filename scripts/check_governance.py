"""Portable repository, documentation and immutable-delivery acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

from generate_e_language_api_doc import parse_exports, source_metadata
from release_manifest import current_manifest, verify_current, verify_historical

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {".gitattributes", ".github", ".gitignore", ".codex", ".codex-os", "AGENTS.md", "CMakeLists.txt", "CMakePresets.json", "README.md", "configs", "docs", "examples", "include", "models", "python", "release", "scripts", "src", "tests", "tools"}
LOCAL_ROOTS = {".git", ".cache", ".ruff_cache", ".pytest_cache", ".venv", ".worktrees", ".vs", ".vscode", "input", "output", "outpush", "third_party", "build"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8")


def hygiene(working_tree: bool, clean: bool) -> None:
    tracked = git("ls-files", "-z").split("\0")
    current_path, current = current_manifest(ROOT)
    approved = {"release/v23.5/" + a["file"] for a in json.loads((ROOT / "release/v23.5/manifest.json").read_text(encoding="utf-8-sig"))["artifacts"]}
    approved |= {a.relative_to(ROOT).as_posix() for a in (current_path.parent / x["file"] for x in current["artifacts"].values())}
    for name in filter(None, tracked):
        if name.split("/", 1)[0] not in ROOTS:
            raise ValueError(f"Unregistered tracked root: {name}")
        if re.search(r"(^|/)(__pycache__|\.venv|\.ruff_cache|\.pytest_cache|node_modules|build|dist)(/|$)|\.(pyc|log|bak)$", name):
            raise ValueError(f"Tracked runtime pollution: {name}")
        if Path(name).suffix.lower() in {".dll", ".exe", ".zip", ".whl"} and name not in approved:
            raise ValueError(f"Unregistered compiled artifact: {name}")
    if git("diff", "--name-only", "--diff-filter=U").strip():
        raise ValueError("Unresolved Git conflicts")
    if working_tree:
        for path in ROOT.iterdir():
            if path.name in ROOTS | LOCAL_ROOTS | {".codex"}:
                continue
            if path.is_dir() and re.fullmatch(r"(build[-\w]*|cmake-build[-\w]*)", path.name):
                if not (path / "CMakeCache.txt").is_file():
                    raise ValueError(f"Build directory has no CMake provenance: {path.name}")
                continue
            raise ValueError(f"Unexpected working-tree root: {path.name}")
    if clean and git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Task changes are not committed; user input is excluded by its read-only contract")


def docs() -> int:
    checked = 0
    for path in [ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        # Inline-code paths are descriptive; actual Markdown links must resolve.
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0].split("\"", 1)[0].strip().strip("<>")
            if not target or re.match(r"[a-zA-Z]+:", target):
                continue
            if not (path.parent / unquote(target)).exists():
                raise ValueError(f"Broken document link: {path.relative_to(ROOT)} -> {target}")
        if "待补充。" in text:
            raise ValueError(f"Unfilled governance template: {path}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-tree", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all", action="store_true", help="Run affected script/wrapper/calibration tests")
    args = parser.parse_args()
    hygiene(args.working_tree, args.require_clean)
    result = verify_current(ROOT, args.output)
    verify_historical(ROOT)
    result["documents"] = docs()
    manifest_path, manifest = current_manifest(ROOT)
    source_metadata(ROOT / "include/ai_engine.h", ROOT / "src/worker_protocol.h", manifest_path)
    if len(parse_exports(ROOT / "include/ai_engine.h")) != manifest["public_exports"]:
        raise ValueError("Header declaration count differs")
    if args.all:
        for pattern in ("test_governance.py", "test_repository_storage.py", "test_packaging.py", "test_yolo_calibration.py", "python_wrapper_test.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-p", pattern], cwd=ROOT, check=True)
    print(json.dumps(dict(passed=True, **result), ensure_ascii=False))


if __name__ == "__main__":
    main()
