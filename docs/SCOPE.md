# Scope

Current task: REQ-CV-001, transparent candidate completeness and local single
caller P95 <=5ms on both x86/x64. Use codex/cv-transparent-mask-fix from fb4aa6c,
stage only under outpush/cv-transparent-mask-fix, and preserve current v23.8 until
the new candidate passes independent correctness, all old/new5ms scenarios,
stability, paired-binary/document verification and actual governance gates.
Reuse pinned OpenCV5.0/IPP/AVX2, true masked correlation and exact integer
verification. Only accepted matches participate in NMS; no coarse-score pruning
or candidate cap may discard a qualifying match. Preserve existing gray rules,
ordinary CV paths, alpha-API score semantics and all public ABI/protocol facts.
No scaling, rotation, PNG import, lighting strategy, OCR fixes or business YOLO
tuning is part of this task. Published validation never applies to new binaries
without fresh evidence. Planned candidate0.14.9/v23.9; failing5ms means no promotion.

Prior delivered scope (REQ-YOLO-003):

REQ-YOLO-003, local YOLO26/bus.jpg five-caller complete-call
P95 <=20ms, revised by the user from15ms; pursue lower latency. Restore Git
history without overwriting local files; use a codex/
branch from the current source. Reuse the existing TensorRT ABI3 module and
locked10.13.3.9/CUDA12.8 dependencies, qualify FP32/FP16 and CUDA Graph on RTX2070,
optimize bit-identical preprocessing, and propagate an explicit P95 limit through
calibration/acceptance tools. Stage all testing/candidates under outpush/.

Do not resize/re-export the frozen640 model, change inference/result contracts,
add INT8/batching, or claim five-business-window acceptance from a bus benchmark.
The measured baseline IPC P95 is1.905ms, below the2ms shared-memory trigger.
Only a verified candidate meeting the new requirement may create/promote a new
delivery cohort; retain the currentv23.7 and all historical evidence.

Prior delivered scope (REQ-YOLO-002):

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
