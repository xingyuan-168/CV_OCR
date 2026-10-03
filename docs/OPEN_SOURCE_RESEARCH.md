# Open Source Research

## Requirement

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
