# Open Source Research

## Requirement

requirement_id: REQ-CV-001
summary: Fix loss of exact transparent matches while requiring local x86/x64 complete-call P95 <=5ms in old and new scenarios.
scope: masked correlation, candidate selection/NMS, numerical reference tests, resource accounting and binary-bound delivery checks
updated_at: 2026-10-23

## Candidates

### Existing OpenCV5.0 masked template matching

- Source: https://docs.opencv.org/4.13.0/df/dfb/group__imgproc__object.html and the pinned OpenCV5.0 static core/imgproc already used by this repository.
- License: existing OpenCV Apache-2.0 distribution; no new runtime dependency.
- Evidence:322 visible pixels in1.bmp exactly match2.bmp at(596,165), but both current product architectures miss. Direct full masked matching finds it, taking263-300ms in single-call diagnostics.
- Limits: changing only ignored target-background pixels makes the current DLL find score1.0. Mean-filled gray coarse scoring and suppression of failed neighbors violate mask invariance. Direct OpenCV masked matching is a correctness cross-check, not evidence of5ms performance.

### Existing correlation workspaces and masked scalar/AVX2 accumulator

- Source: src/cv_correlation.cpp, src/cv_masked_score_avx2.cpp, src/ai_engine.cpp and tests/cv_real_transparent_benchmark.cpp.
- Reuse: per-workspace IPP transform plans, verified frame-content reuse, immutable template snapshots, bounded idle memory and integer exact masked score.
- Extension: true masked window moments/correlation, safe numerical candidate bounds, exact score ordering and accepted-match-only NMS. Independent exhaustive tests replace shared-heuristic equivalence as the completeness oracle.
- Limits:32-bit FFT estimates must not silently reject boundary/low-energy candidates. Track every new retained buffer; production builds exclude test hooks. Current two gray conversions remain unchanged.

## Decision

decision: extract
reason: Extend the existing native primitives with complete masked candidate evaluation, preserving scores and public contracts. First establish independent correctness, then measure and optimize toward the user-confirmed5ms hard gate. A candidate failing correctness or any5ms round remains unpromoted; no finite candidate cap, old coarse suppression or relaxed threshold is an acceptable performance shortcut. Diagnosis evidence is outpush/cv-transparent-20261023/ROOT_CAUSE_AND_REPAIR_PLAN_CN.md.

### Performance follow-up — 2026-10-24

Fresh x86/x64 probes show20,769 candidates for the old miss template,18,404 with
zero full masked source energy. Gray leaves17,319 candidates;128 sampled points
reduce this to1 but the bound still costs about7.8ms. Changed frames pay the full
source transform cost even with one exact candidate. Detailed evidence and the
implementation decision are in [the5ms improvement plan](CV_TRANSPARENT_5MS_IMPROVEMENT_PLAN_CN.md).

Decision: independently implement and verify a mask-only orthogonal-projection
upper bound using incremental integer sample moments, then compare a spatial
SIMD cascade against the existing FFT path. This is a proposed optimization,
not measured5ms success. Preserve uncertainty for exact scoring, including score
rounding and degenerate templates. Keep the old path as a correctness fallback;
a fallback exceeding any formal5ms scenario still prevents delivery.
Reuse existing OpenCV/IPP/AVX2 and add no runtime dependency. Intel's DFT API
requires plan/work buffers, already reused here:
https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/dftfwd.html .
Do not attribute regression to missing plan reuse or remove FFT numerical guards.

## Prior research — REQ-YOLO-003

## Requirement

requirement_id: REQ-YOLO-003
summary: Reach local YOLO26/bus five-caller complete-call P95 <=20ms with unchanged public contracts; the user revised acceptance from15ms on2026-10-07.
scope: optional NVIDIA runtime recovery, native preprocessing, calibration/acceptance, RTX2070 evidence and paired delivery
updated_at: 2026-10-08

## Candidates

### Existing native TensorRT10.13.3.9 module and pinned NVIDIA distribution

- URL: https://github.com/NVIDIA/TensorRT ; https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html
- License: existing headers Apache-2.0; runtime distributions carry separate NVIDIA terms and retained licenses.
- Reuse: ABI3 engine/context/stream/pinned buffers, FP16, CUDA Graph, per-driver/model engine and timing cache.
- Limits: this host lacks nvinfer_10.dll; a real load fails with Windows error126. A compiled module does not prove GPU execution. FP16 and Graph remain gated by correctness and repeated measurements.

### Existing preprocessing and native caller benchmark

- Source: src/yolo_preprocess.h, tests/yolo_pipeline_test.cpp, tests/yolo_benchmark.cpp and scripts/calibrate_yolo.py.
- Reuse: exact nearest-neighbor letterbox,114 padding, FP32 CHW normalization, QPC complete-call timing and per-request traces.
- Extension: a256-entry FP32 division lookup and separate content/padding loops; configurable P95 limit with legacy defaults preserved.
- Limits: current selection/acceptance use30ms, not the requested15ms. Baseline diagnostic P95: ORT Run32.086ms, preprocessing3.448ms, combined IPC1.905ms. Stage percentiles are diagnostic and are not additive formal acceptance evidence.

### Existing calibration identity and embedded runtime bundle

- Source: src/yolo_pool.cpp runtime_fingerprint(), src/runtime_bundle_loader.cpp runtime_bundle_hash()/ensure_embedded_directml().
- Evidence: after native cache tests, the same model/driver/slots/workload changed key from773c9aee... to153b8a22...; AUTO missed the record and used FP32. The embedded CPU runtime lazily extracts DirectML.dll, while the old fingerprint hashed that cache file or the word missing once per process.
- Reuse: the verified embedded bundle SHA already covers both ORT and DirectML payloads independently of extraction state. Use that identity for embedded Workers; retain actual file fingerprints for the external x64 runtime.
- Acceptance: an owned cold LOCALAPPDATA cache, subsequent DirectML extraction and a fresh process must produce the same key; AUTO must read the newly validated record. Version the key algorithm to reject earlier ambiguous records. Public ABI/protocol remain unchanged.

### Portable build tooling for this host

- Source: https://github.com/tgbender/portablemsvc ; CMake/Ninja PyPI binary wheels.
- Reuse: extract official Microsoft MSVC14.44/SDK26100 packages to an independent directory using the pure-Python MSI extractor. No system installer, registry registration or permanent environment change.
- Evidence: external portablemsvc.lock records resolved MSVC14.44.35207 (compiler19.44.35229), SDK10.0.26100.0, package URLs and hashes. The Microsoft channel manifest's declared vsman hash differs from its downloaded response; both are retained, and individual payload hashes are checked. This build tooling is not shipped with the product.
- Extension: allow compiler-discovered X86/x64 AVX2 paths with Ninja, and discover packaging tools from the active PATH before the existing VS2022 fallback.

### CUDA stage timing after long-run failure

- Source: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__EVENT.html ; src/tensorrt_module.cpp run().
- Evidence: the final core pair passed short formal20ms sampling, but a second1800-second run reached worst-laneP95 27.0985ms with621,628 successful calls. Both retained runs show higher minute latency after about13-15minutes. This does not establish a thermal, scheduling or background-load cause.
- Extension: allocate/record four CUDA timing events and query elapsed times only when CQ_AI_YOLO_TRACE_PREFIX is configured before model creation. Normal requests retain asynchronous copies, enqueueV3/Graph and cuStreamSynchronize before output; zero stage timings mean instrumentation disabled.
- Limits: this removes known diagnostic work, not the demonstrated cause of the long-run tail. Requalify CPU/FP32/FP16 detections, both modes/three rounds, Graph and1800-second stability. A failed candidate remains unpromoted; do not repeat until a favorable result and discard failures.

## Decision

decision: use
reason: Reuse and qualify the already implemented NVIDIA path before adding infrastructure. Restore exact locked dependencies, retain the640 model, optimize only bit-identical preprocessing, and expose the P95 limit explicitly. Keep the original15ms scan and use the user-revised20ms limit for final acceptance. CUDA Graph is enabled only after every-round >=10% P95 improvement. IPC is below the registered2ms trigger, so shared memory is outside the current change unless new measurements exceed it. Baseline evidence is outpush/yolo26-local-20261007/measurement-summary.json;30,000 formal samples passed correctness but failed latency. No five-window business acceptance claim follows from this workload.

## Prior research — REQ-YOLO-002

## Requirement

requirement_id: REQ-YOLO-002
summary: Support YOLOv5, YOLO11 and YOLO26 ONNX detection without public ABI or result-contract changes.
scope: native model inspection/decoding, ORT/TensorRT adapters, offline tools, tests and delivery
updated_at: 2026-10-04

## Candidates

### Ultralytics YOLOv5 v7.0 and Ultralytics8.4.28

- URL: https://github.com/ultralytics/yolov5/tree/v7.0 and https://github.com/ultralytics/ultralytics/tree/v8.4.28
- License: YOLOv5 v7.0 GPL-3.0; Ultralytics8.4.28 AGPL-3.0. Used as offline export/reference tools; their implementation is not copied into the DLL.
- Solves: authoritative decoded detection tensor semantics and reproducible small reference models.
- Direct reuse: offline exporters and model metadata; no runtime Python/PyTorch dependency.
- Extension: export/validation scripts with fixed options and hashes.
- Useful design: classic v5 [B,N,5+C] uses objectness*class probability; dense heads use [B,4+C,N]; native end-to-end [B,K,6] is xyxy/score/class and requires no second NMS.
- Limits: export flags vary by release;8.4.28 uses end2end explicitly. Metadata tool versions are not model architectures. Official preprocessing/class-aware defaults differ from this library's legacy contract.

### ONNX IR and existing inference adapters

- URL: https://onnx.ai/onnx/repo-docs/IR.html ; https://github.com/microsoft/onnxruntime/tree/v1.24.4 ; https://github.com/NVIDIA/TensorRT
- License: ONNX Apache-2.0, ORT MIT, TensorRT headers Apache-2.0 with separate NVIDIA runtime terms.
- Solves: serialized model metadata, tensor declarations, CPU/DirectML and optional NVIDIA execution.
- Direct reuse: existing ORT1.24.4/TensorRT10.13.3.9 infrastructure and Python ONNX tooling offline.
- Extension: shared bounded read-only model descriptor reader, rejecting unsupported tasks/NMS and ambiguous layouts; private TensorRT ABI3.
- Useful design: parse descriptors once, skip weights, validate runtime tensors against descriptors, preserve persistent buffers.
- Limits: ORT metadata normally requires a Session; a separate probe Session would penalize TensorRT loading. A small bounded wire reader avoids that and adds no deployment dependency.

## Decision

decision: build
reason: Independently implement a narrowly scoped descriptor reader and three shared decode paths, reuse the existing runtimes, and use pinned upstream exports only for offline validation. Keep legacy NMS/preprocessing/labels and all public contracts. Metadata-only prepared copies resolve ambiguity without changing user inputs or model tensor outputs.

Evidence: input/yolov26/yolo26n.onnx SHA256355c4f26cc1d48e581aaeb4a7f837baf7d5993e6f83337d30d50d0476bc5b158 has FP32 [1,3,640,640] -> [1,300,6], task=detect and end2end=True. The0.14.6/ORT1.24.4 native CPU path loads it but emits score561.258/class156, proving a decode bug rather than an unsupported inference graph. Native end-to-end semantics are documented at https://docs.ultralytics.com/guides/end2end-detection/ .

## Prior governance research — REQ-GOV-001

### Requirement

requirement_id: REQ-GOV-001
summary: Register the delivered v23.6 source and integrate observable repository governance.
scope: scripts, configs, docs, CI, release manifests, AIOS compatibility
updated_at: 2026-10-03

## Candidates

### AI Engineering OS

- URL: https://github.com/xingyuan-168/AI_Engineering_OS
- Upstream revision: 3ba90c2d1962c6129ca03c0d4e305156f60bdc0d
- License: upstream has no LICENSE or declared project license at this revision; it is the user-owned governance tool. Record this explicitly; dependency licenses remain in their distributions.
- Solves: Start/Finish gates, document checks, project memory and disposable worktrees.
- Direct reuse: official CLI and its frozen uv.lock; Codex retains task execution.
- Extension: minimal regression-tested preservation of existing input/output directories.
- Useful design: deterministic observable gates and Git-tracked engineering facts.
- Limits: stock init/check requires .gitkeep files; the existing three-file output forbids them.

### Existing CQ_AI scripts

- Source: scripts/package_delivery.py and scripts/check_repository_hygiene.ps1.
- Reuse: native verification, exact ABI/protocol checks, report hash binding and transactional output promotion.
- Extension: current-manifest selection, portable paths, declared fixtures, shared NVIDIA preparation.
- Limits: hardcoded historical release and dependence on machine-local input/build paths.

## Decision

decision: use
reason: Reuse official AIOS gates with a narrowly scoped compatibility patch and strengthen existing delivery scripts; avoid adding another agent runtime or rewriting native inference.

The patch, runtime lock and actual gate results are recorded with the governance report.

## Prior backend research — REQ-YOLO-001

### YOLO 五路并发：后端调研与决策

## Requirement

E5-2696 v4 / RTX2070 上，五个线程共享一个 YOLO 模型句柄；完整调用每路 P50≤20ms、P95≤30ms。保留 CPU、DirectML、OCR 和现有 ABI。输入静态 FP32 `[1,3,320,320]`，输出 `[1,10,2100]`；FP16 必须验证检测一致性。

## Candidates

| 候选 | 来源 / 许可 | 可复用部分 | 限制与风险 |
|---|---|---|---|
| ONNX Runtime 1.24.4 | [官方源码](https://github.com/microsoft/onnxruntime/tree/v1.24.4)，MIT | 现有 CPU / DirectML Session、静态张量、profiling | DirectML 要求关闭 memory pattern、顺序执行；同一 Session 不得并行 Run。节点回退和复制成本必须实测。 |
| 传统 TensorRT 10.13.3 | [版本说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html)、[官方源码](https://github.com/NVIDIA/TensorRT)，开源接口/示例 Apache-2.0；二进制运行库遵循 NVIDIA 许可 | ONNX parser、共享 engine、独立 execution context、enqueueV3、timing cache | Windows x64，CUDA 12.8 Update 1 为固定构建基线；engine 与 GPU/软件版本绑定。目标机驱动及精度必须验证。 |
| CUDA Driver API | [12.8.1 官方接口](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/index.html)，NVIDIA 驱动 / SDK 许可 | 持久设备内存、pinned host memory、异步复制、stream、event / graph | 动态加载 nvcuda.dll；无 NVIDIA 驱动时只让显式 TensorRT 请求失败，不阻塞其他后端。 |
| ORT TensorRT EP | [官方文档](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)，ORT MIT | 子图转换、缓存及 CUDA EP 回退 | 当前 DirectML 分发包不含此 EP；另换 ORT 构建增加部署和 OCR 回归范围，也难以控制每个执行槽的 buffer / stream。 |

## Decision

**build** 原生可选 `CQ_YOLO_TensorRT.dll` 适配模块，使用官方 TensorRT / CUDA API，不 fork 运行时。基础 DLL/Worker 动态加载模块，基础包无需 NVIDIA 依赖。ORT 继续承担 CPU、DirectML 和 OCR。公共最近邻预处理、坐标还原、类别和 NMS 语义保持一致。

同一模型共享 engine，各槽独立 context / stream / pinned host buffer / device buffer。缓存完整记录模型 SHA-256、GPU、驱动、CUDA、TensorRT、精度和构建参数；损坏或不匹配时重建。首先 FP32，FP16 只有离线业务验证通过后可选。CUDA Graph 作为测量后启用的选项，不预设性能收益。

性能依据来自完整业务调用的 QPC 测量，不能以 GPU kernel 时间替代验收。可用版在功能验证通过后可以交付，目标机性能认证另列；RTX2070 实测通过前不得标注20/30ms已达标。
