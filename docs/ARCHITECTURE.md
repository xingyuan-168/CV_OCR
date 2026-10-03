# Architecture

CQ_AI is a Windows C++17 library with a stable C ABI (60 public functions).

```mermaid
flowchart LR
  E[32-bit caller] --> D[CQ_X86.dll]
  D --> CV[In-process OpenCV]
  D --> P[Persistent named pipe / protocol26]
  P --> W[CQ_AI_worker.exe / x64]
  PY[Python x64] --> X[CQ_AI_x64.dll]
  W --> ORT[ORT CPU / DirectML]
  X --> ORT
  W --> TRT[Optional TensorRT module / CUDA]
  X --> TRT
```

YOLO shares preprocessing, decoding and backend selection across x86 and x64.
Session count is execution-slot capacity, separate from callers and ORT threads.
Each DirectML Session serializes Run; each TensorRT slot owns its context, stream
and buffers. BMP views, reusable workspaces and request IDs cover the common path.
OCR retains devices0..2 and ORT. CV runs in the host DLL.

The core Worker embeds its fixed ORT/DirectML runtime and OCR assets; optional
NVIDIA dependencies remain a separate package. A model pool owns the backend and
leases slots with RAII. Runtime status identifies the actual backend/device.

Repository flow: source + locked dependencies -> build -> verified candidate ->
HTML/archives -> cohort promotion. `release/current.json` locates the immutable
current manifest; `output/` is the local three-file deployment directory.

Detailed source mapping: [project guide](PROJECT_GUIDE_CN.md).
