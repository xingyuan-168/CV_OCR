# Scope

This task implements REQ-YOLO-002 from the registered v23.6 source: validated ONNX
detection descriptors, v5/v11/v26 decode paths, ORT/TensorRT integration, offline
model tools, regression tests and the0.14.7/v23.7 paired DLL/Worker/HTML delivery.
Public ABI, result contracts and protocol26 remain compatible. Native TensorRT
private ABI becomes3; version-specific Worker names prevent mixed deliveries.

Standard batch1 square FP32 detection exports are covered. Additional tasks,
embedded traditional NMS, TensorRT dynamic shapes, INT8 and batch inference are
outside this task. Existing ORT dynamic-input behavior remains supported. New
model fixtures/export environments are temporary or hash-pinned; no heavy new
runtime dependency is shipped. v23.6 and older registered artifacts remain frozen.

The AIOS directory-preservation compatibility patch is isolated from the user's
AI-OS working directory and is recorded with its upstream revision and patch hash.

User input, unrelated projects and business Worker processes are protected.
Historical packages stay immutable. Model and test fixture retention follows
actual build/test use. Temporary build and dependency state stays outside Git.

Target RTX2070 performance tuning remains a separate acceptance task using the
delivered one-click tools and real business BMPs. Shared memory is conditional
on isolated target-machine communication P95 exceeding2ms.
