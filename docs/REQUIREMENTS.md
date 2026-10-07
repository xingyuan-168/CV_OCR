# Requirements

## REQ-YOLO-003 — Local YOLO26 five-caller P95 <=20ms

On the local E5-2696 v4 / RTX2070, five callers share one model handle through
the x86 DLL and paired x64 Worker. Freeze input/yolov26/yolo26n.onnx
(SHA256355c4f26cc1d48e581aaeb4a7f837baf7d5993e6f83337d30d50d0476bc5b158),
FP32 input [1,3,640,640], native end-to-end output [1,300,6], bus.jpg and
confidence0.5. Decode the JPEG once to a 24-bit BMP outside input/ without resizing.

Acceptance: every lane in each of three independent rounds has caller QPC
complete-call P95 <=20ms, in continuous and simultaneous modes, with100 warmups
and1000 samples per lane. The user revised acceptance from15ms to20ms on2026-10-07;
15ms remains a stretch target. Report P50/P95/P99, maximum and fractions above
both15ms and20ms, preserving the original15ms scan as historical evidence.
Require zero inference errors, CPU-reference count/class agreement and matched
coordinate error <=1px / score difference <=0.02. FP16 also matches qualified
TensorRT FP32. Graph requires >=10% P95 improvement in every round of both modes.
Preprocessing remains bit-identical for all supported formats and signed strides.
Verify engine cache build/hit/corrupt rebuild, AUTO profile readback, affected
regressions and30-minute local stability before promoting a paired delivery.
AUTO calibration identity must survive process restart and lazy DirectML cache
extraction. Test this with an isolated, test-owned runtime cache.

Keep the60 public signatures, stdcall aliases, protocol26 and positive
session_count semantics. Existing tool invocations retain the20/30ms defaults;
an explicit20ms P95 limit controls this task's final selection and acceptance. NVIDIA
dependencies remain optional and separate. Historical releases and input/ are
immutable. This bus-only benchmark does not complete REQ-YOLO-001 five-window
business acceptance. Baseline formal worst-lane P95 is39.7853ms continuous and
28.6854ms simultaneous; all30,000 samples succeeded but the latency target failed.

Additional user-requested validation: input/best.onnx with input/yolo测试图.bmp
(V8 dense-head workload), five callers, both modes, the same20ms per-lane P95
limit and formal sampling. Report its result separately; do not infer V8 success
from YOLO26. Keep its native static input shape and compare the old model's
same-backend/same-slot latency with v23.7 to detect >5% repeatable regression.

## REQ-YOLO-002 — Compatible YOLO detection layouts

Deliver 0.14.7 / v23.7 with unchanged 60 public signatures, result structures,
JSON/encoding/error contracts and Worker protocol26. Support classic YOLOv5
objectness outputs, YOLOv5u/v8/11/26 dense outputs and native YOLO26 end-to-end
outputs through one validated model descriptor and shared decoder. Preserve
legacy preprocessing, class-agnostic NMS0.45, labels and valid-model results.

Accept standard batch1 square FP32 ONNX detection models from paths and memory;
retain existing ORT dynamic-input behavior. Reject ambiguous, embedded-NMS,
segmentation/pose/OBB and undecoded outputs with existing errors. An offline
inspection/preparation tool may write validated metadata-only copies outside
input/. No Python runtime dependency is added to the delivered binaries.

Acceptance: the user's input/yolov26/yolo26n.onnx and bus.jpg produce valid
end-to-end results (initial CPU reference at confidence0.5: four people and one
bus); pinned v5/v11 reference models, synthetic layout/error cases, old-result
equivalence, x86/x64 CPU/DirectML regressions, paired delivery checks, five-lane
100 warmups +1000 samples x3 rounds and 30-minute changing-frame stability.
Repeatable old-model P50/P95 regression above5% blocks promotion. TensorRT GPU
execution/FP16 and the E5/RTX2070 performance target remain explicitly pending
until hardware evidence exists. Historical deliveries/input remain immutable;
promote verified candidates transactionally and retain the1.5GiB storage budget.

## REQ-GOV-001 — Govern the delivered v23.6 source

Use the delivered 0.14.6 / v23.6 / Worker protocol26 files and source commit
`5e05f7f86d7a361f52cdfa459f61d1eb3f8db8bf` as the baseline. The user's selected
canonical repository is CV_OCR. Preserve input data, the three deployed files,
their hashes, and the original v23.5 delivery.

Acceptance: a portable current manifest; clean-clone build and checks; one
packaging entry point; accurate examples; CI; observable AIOS Start/Finish results;
and a reviewable governance commit on CV_OCR.

Storage acceptance: retain the verified NVIDIA optional ZIP once outside the
repository, actually delete registered duplicated artifacts/caches, preserve
protected hashes, and keep the finished workspace including Git below1.5GiB.
Report deleted bytes separately from externally retained bytes.

## REQ-YOLO-001 — Target-machine performance

Five callers share one model handle on E5-2696 v4 / 128GB / RTX2070, with five
business windows. Each lane's complete call must reach P50 <=20ms and P95 <=30ms.
The frozen model is FP32 input [1,3,320,320], output [1,10,2100], six classes;
confidence0.5 and NMS0.45. Time the caller with QPC, including communication.

FP16 acceptance requires equal counts/classes, matched coordinate error <=1px
and score difference <=0.02, including critical targets and boundary cases.
Graph requires correctness and repeatable P95 improvement >=10% in each round.
Local functional evidence is not target-machine performance acceptance.

Current status is usable, functional validation passed in its recorded scope;
NVIDIA hardware execution and target performance remain pending.
