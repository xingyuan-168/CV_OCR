"""Installed cq-ai-engine x64 Wheel acceptance and DirectML soak test.

Run this file with ``python -I`` from the clean Wheel virtual environment.  The
script deliberately lives below ``tests`` so importing ``ai_engine`` cannot
fall back to the repository's ``python/ai_engine.py`` source file.
"""

from __future__ import annotations

import argparse
import ctypes as C
from ctypes import wintypes
import json
from pathlib import Path
import struct
import time

import cq_ai_engine
from ai_engine import (
    AI_DEVICE_CPU,
    AI_DEVICE_DIRECTML,
    Engine,
    image_from_bmp_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ProcessMemoryCountersEx(C.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", C.c_size_t),
        ("working_set_size", C.c_size_t),
        ("quota_peak_paged_pool_usage", C.c_size_t),
        ("quota_paged_pool_usage", C.c_size_t),
        ("quota_peak_non_paged_pool_usage", C.c_size_t),
        ("quota_non_paged_pool_usage", C.c_size_t),
        ("pagefile_usage", C.c_size_t),
        ("peak_pagefile_usage", C.c_size_t),
        ("private_usage", C.c_size_t),
    ]


def _process_memory_mib() -> dict[str, float]:
    kernel32 = C.WinDLL("kernel32", use_last_error=True)
    psapi = C.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        C.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _ProcessMemoryCountersEx()
    counters.cb = C.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), C.byref(counters), counters.cb
    ):
        raise C.WinError(C.get_last_error())
    mib = 1024.0 * 1024.0
    return {
        "working_set_mib": round(counters.working_set_size / mib, 3),
        "private_mib": round(counters.private_usage / mib, 3),
    }


def _installed_package_checks() -> dict[str, object]:
    assert struct.calcsize("P") * 8 == 64
    assert cq_ai_engine.__version__ == "0.14.6"
    with Engine() as engine:
        assert engine.version() == "CQ_AI_x64/0.14.6"
        assert engine.dll_path.name == "CQ_AI_x64.dll"
        assert "cq_ai_engine" in str(engine.dll_path.parent.parent)
        assert not any(engine.dll_path.parent.glob("*.exe"))
        engine.shutdown_worker()
        return {
            "version": engine.version(),
            "dll": str(engine.dll_path),
        }


def _cv_checks() -> dict[str, object]:
    source_path = PROJECT_ROOT / "tests" / "fixtures" / "cv" / "base" / "大图1.bmp"
    template_paths = [
        PROJECT_ROOT / "tests" / "fixtures" / "cv" / "base" / "电.bmp",
        PROJECT_ROOT / "tests" / "fixtures" / "cv" / "base" / "游.bmp",
    ]
    source = image_from_bmp_bytes(source_path.read_bytes())
    templates = [image_from_bmp_bytes(path.read_bytes()) for path in template_paths]
    with Engine() as engine:
        single = engine.cv_find_image_dict(source, templates[0], min_score=0.80)
        multiple = engine.cv_find_images_dicts(source, templates, min_score=0.80)
    assert single is not None, "single-template CV match was not found"
    assert multiple, "multi-template CV matches were not found"
    return {
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "single": single,
        "multi_count": len(multiple),
    }


def _ocr_cpu_checks() -> dict[str, object]:
    bmp = (PROJECT_ROOT / "examples" / "ocr_test_abc123.bmp").read_bytes()
    with Engine() as engine:
        engine.ocr_load_embedded_models(runtime_device=AI_DEVICE_CPU)
        text = engine.ocr_recognize_bmp(bmp, min_confidence=0.0)
        status = engine.runtime_status()
        engine.ocr_release()
    assert isinstance(text, str) and text
    assert status.get("active") == "cpu", status
    return {"text": text, "runtime": status}


def _yolo_checks() -> dict[str, object]:
    model_path = PROJECT_ROOT / "models" / "yolo" / "best.onnx"
    memory_model_path = PROJECT_ROOT / "models" / "yolo" / "best.onnx"
    image_path = PROJECT_ROOT / "tests" / "fixtures" / "yolo" / "1.bmp"
    image = image_from_bmp_bytes(image_path.read_bytes())
    model_bytes = memory_model_path.read_bytes()

    with Engine() as engine:
        path_model = engine.yolo_model(0, AI_DEVICE_CPU, 0, 2)
        memory_model = engine.yolo_model(0, AI_DEVICE_CPU, 0, 2)
        try:
            path_model.load_model(model_path)
            path_result = path_model.infer(image, confidence=0.25)
            path_status = path_model.runtime_status()

            # Keep the first handle alive while loading and executing the second
            # handle.  This exercises simultaneous handles and two-session pools.
            memory_model.load_model_from_memory(model_bytes)
            memory_result = memory_model.infer(image, confidence=0.25)
            memory_status = memory_model.runtime_status()
            path_result_again = path_model.infer(image, confidence=0.25)
        finally:
            memory_model.release()
            path_model.release()

    assert isinstance(path_result, list)
    assert isinstance(memory_result, list)
    assert path_result == path_result_again
    assert path_status.get("active") == "cpu", path_status
    assert memory_status.get("active") == "cpu", memory_status
    return {
        "path_detections": len(path_result),
        "memory_detections": len(memory_result),
        "path_runtime": path_status,
        "memory_runtime": memory_status,
    }


def _directml_soak(iterations: int) -> dict[str, object]:
    if iterations <= 0:
        return {"iterations": 0, "skipped": True}

    bmp = (PROJECT_ROOT / "examples" / "ocr_test_abc123.bmp").read_bytes()
    samples: list[dict[str, object]] = []
    sample_every = max(1, iterations // 10)
    worker_before = _worker_pids()
    started = time.perf_counter()

    with Engine() as engine:
        engine.ocr_load_embedded_model(
            runtime_device=AI_DEVICE_DIRECTML, session_count=1
        )
        status = engine.runtime_status()
        assert status.get("active") == "directml", status
        reference = engine.ocr_recognize_bmp(bmp, min_confidence=0.0)
        samples.append({"iteration": 0, **_process_memory_mib()})
        for iteration in range(1, iterations + 1):
            current = engine.ocr_recognize_bmp(bmp, min_confidence=0.0)
            assert current == reference, f"OCR output changed at iteration {iteration}"
            if iteration % sample_every == 0 or iteration == iterations:
                samples.append({"iteration": iteration, **_process_memory_mib()})
        engine.ocr_release()

    # A new Engine and DirectML session must still load after full release.
    with Engine() as engine:
        engine.ocr_load_embedded_model(
            runtime_device=AI_DEVICE_DIRECTML, session_count=1
        )
        reloaded = engine.ocr_recognize_bmp(bmp, min_confidence=0.0)
        reload_status = engine.runtime_status()
        engine.ocr_release()
    assert reloaded == reference
    assert reload_status.get("active") == "directml", reload_status
    assert _worker_pids() == worker_before, "DirectML soak unexpectedly started a worker"

    private_values = [float(sample["private_mib"]) for sample in samples]
    return {
        "iterations": iterations,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stable_output": True,
        "released_and_reloaded": True,
        "runtime": status,
        "reload_runtime": reload_status,
        "private_mib_min": min(private_values),
        "private_mib_max": max(private_values),
        "private_mib_delta": round(private_values[-1] - private_values[0], 3),
        "samples": samples,
    }


def _worker_pids() -> list[int]:
    # Avoid psutil and shelling out: Toolhelp enumerates processes using only
    # Windows system DLLs.  The helper is intentionally local to the test.
    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = C.c_void_p(-1).value

    class ProcessEntry32W(C.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", C.POINTER(C.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", C.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", C.c_wchar * 260),
        ]

    kernel32 = C.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, C.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, C.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if handle == INVALID_HANDLE_VALUE:
        raise C.WinError(C.get_last_error())
    pids: list[int] = []
    try:
        entry = ProcessEntry32W()
        entry.dwSize = C.sizeof(entry)
        ok = kernel32.Process32FirstW(handle, C.byref(entry))
        while ok:
            if entry.szExeFile.casefold() == "cq_ai_worker.exe":
                pids.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(handle, C.byref(entry))
    finally:
        kernel32.CloseHandle(handle)
    return sorted(pids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dml-iterations", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = {
        "package": _installed_package_checks(),
        "cv": _cv_checks(),
        "ocr_cpu": _ocr_cpu_checks(),
        "yolo": _yolo_checks(),
        "directml_soak": _directml_soak(args.dml_iterations),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
