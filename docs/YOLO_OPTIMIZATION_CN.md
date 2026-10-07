# YOLO：构建、校准与目标机验收

本轮0.14.8/v23.8候选兼容V5/V11/V26检测输出，保持60个公开函数、正整数`session_count`及协议26。设备3=TensorRT仅用于YOLO。模型要求见[兼容说明](YOLO_MODEL_COMPATIBILITY_CN.md)。本机bus和V8图片的五线程P95≤20ms结果见[本机报告](YOLO26_LOCAL_20MS_CN.md)；五业务窗口变化帧的20/30ms验收仍独立待验。

## 实现

- x86 调用线程复用管道与请求缓冲区；Worker 每条连接循环收发完整帧。请求 ID 校验、512MiB 帧上限、完整读写与独立实例 ID 防止错配和旧句柄误用。断管后的 YOLO 句柄不会自动重载，应创建并加载新句柄。
- 空闲连接与执行请求分别计数。全局关闭、OCR 轮换和孤儿清理只等待执行请求；原有 OCR 轮换仍避开存活 YOLO 模型。
- Worker直接使用请求缓冲区的24位BMP视图，负stride表示自底向上图像。最近邻letterbox采用256项FP32归一化查表，表值仍以原来的`i/255.0f`生成；padding114与有效区域分开循环，格式分派移出逐像素循环。全部支持格式、正负stride和640输入均按位回归。坐标还原及原有按分数排序、类别无关NMS保持不变。
- ORT 静态输入张量包装及 FP32 静态输出缓冲区复用。每个 Session 由一个租约串行占用，异常也归还槽位。DirectML 保持顺序执行和关闭 memory pattern。
- TensorRT 模块动态加载：模型共享 engine，各槽独立 execution context、stream、pinned host input/output、device buffer 与工作区。异步 H2D / enqueueV3 / D2H 后仅同步本次 stream；没有逐帧显存申请或全设备同步。
- engine / timing cache 绑定模型 SHA-256、GPU UUID/能力、驱动、CUDA、TensorRT、FP32/FP16、workspace、TF32/aux stream 设置和算法版本。缓存文件另带 SHA-256，损坏时重建；加载阶段构建计时单列。
- x86 和 x64 共用池构建、CPU 物理核预算及 AUTO 逻辑。真实 BMP 校准记录优先；未命中时以五个并发调用做暂定合成校准，扫描预算内 1/2/4/6/8 个 CPU 内部线程。显式后端失败返回错误；只有 AUTO 选择其他后端。

## 构建

使用 VS2022 x64/Win32，与已有 ORT 1.24.4 / 静态 OpenCV 构建参数一致。

```powershell
python scripts/prepare_nvidia_headers.py
cmake -S . -B build-yolo-x64 -G "Visual Studio 17 2022" -A x64 `
  -DAIENGINE_WITH_ONNXRUNTIME=ON -DAIENGINE_BUILD_X64_DLL=ON `
  -DAIENGINE_BUILD_WORKER=OFF -DAIENGINE_BUILD_TENSORRT_MODULE=ON `
  -DAIENGINE_ONNXRUNTIME_DIR=third_party/runtime/ort-directml-1.24.4
cmake --build build-yolo-x64 --config Release --parallel 4
```

可从官方 TensorRT-10.13.3.9 SDK 准备头文件：`--tensorrt-sdk <SDK目录>`。模块编译无需 GPU，不静态链接 CUDA / TensorRT；GPU 功能验证必须在 NVIDIA 机器上进行。依赖包使用传统 TensorRT 10.13.3.9 Windows CUDA12 系列分发（官方 ZIP 名为 cuda-12.9，支持兼容 CUDA12 版本），固定搭配 CUDA runtime 12.8.90、NVRTC12.8.93 和 cuBLAS12.8.4.1。

## 部署

易语言目录：`CQ_X86.dll`、`CQ_AI_worker.exe`、可选 `CQ_YOLO_TensorRT.dll`。Python 目录：`CQ_AI_x64.dll`、现有 ORT / DirectML / VC 运行库、可选模块。NVIDIA 可选包的 `nvidia/` 放在 Worker / x64 DLL 旁，或设置 `CQ_AI_NVIDIA_DIR` 为其绝对路径。CPU / DirectML 无需此目录。

目标NVIDIA驱动须暴露CUDA Driver API≥12.8；GPU算力须≥7.5。`nvcuda.dll`来自系统驱动，不能从其他机器复制。独立x64进程首次构建还应把`CQ_AI_NVIDIA_DIR`添加到该进程PATH，供TensorRT按名称加载builder resource DLL；一键验收入口自动设置此进程路径。第一次加载可能需要较长构建时间；后续复用`%LOCALAPPDATA%/CQ_AI/tensorrt-v3`缓存。不要把冷启动算进稳定推理延迟。

## 业务校准

候选 ZIP 内附 `tools/run_yolo_acceptance.ps1`，解压独立工具包后执行 `powershell -File tools/run_yolo_acceptance.ps1 -RuntimeDir D:/业务程序目录 -Model D:/模型/best.onnx -ImagesDir D:/业务BMP -BusinessWindowsActive` 即可记录硬件、完成后端校准，并以推荐的 AUTO 参数跑五路30分钟稳定性。默认 validation 目录只有基线和测试图片；最终验收应改为五窗口实际变化帧。`-CalibrateOnly` 只做校准。

`--images-dir` 必须包含实际五窗口变化帧。工具对每个候选比较同一批图片，要求检测数量/类别一致、每个坐标差≤1px、分数差≤0.02；CPU 公共链对比要求 JSON 检测数据完全一致。FP16 任一图片失败就不发布该候选。Graph 只有成功捕获且两个调用模式的每轮 P95 都改善≥10%才可发布。

```powershell
python scripts/calibrate_yolo.py `
  --benchmark build-yolo-x86/Release/ai_engine_yolo_benchmark.exe `
  --model input/best.onnx --images-dir D:/业务BMP `
  --sessions 1,2,3,5 --cpu-threads 1,2,4,6,8 --devices 2,1,3 `
  --fp16 --graphs --preserve-worker --warmup 100 --samples 1000 --rounds 3 `
  --p95-limit-ms 20 `
  --output outpush/target-calibration
```

每个候选测持续调用与五路同时发起，输出每路/整体P50、P95、P99、最大值、超限比例、吞吐、CPU、内存及可用时的NVIDIA采样。以每轮每一路P95判定，先选满足门槛的候选，再比较P50；禁止总体平均掩盖单路超标。`--p95-limit-ms`、PowerShell的`-P95LimitMs`及汇总参数默认30，显式传20用于本次要求。工具生成`business-calibration.tsv`和推荐参数，两个模式均执行AUTO回读。校准键v6绑定嵌入运行库bundle哈希，不再因DirectML延迟解压或Worker重启变化；模型、设备/驱动、槽数、输入参数或图片标识变化仍使记录失效。

运行前设置 `CQ_AI_YOLO_CALIBRATION_FILE` 和 `CQ_AI_YOLO_WORKLOAD_ID` 为报告给出的值，再以推荐的正整数槽数加载 AUTO。内部线程可通过 `CQ_AI_YOLO_CPU_THREADS` 明确设置（1..8，受物理核预算限制）。生产中 FP16 / Graph 只从已验证记录启用；`CQ_AI_YOLO_VALIDATING=1` 是离线验证工具专用开关。OCR 仅支持 0..2。

状态追加 `precision`、`tensorrt_version`、`cuda_version`、`driver_version`、`driver_file_version`、`driver_binary_sha256`、`execution_slots`、`engine_cache_hit`、`engine_build_us`、`cuda_graph`、`selection_basis` 与校准键。`driver_version` 是 CUDA Driver API 能力版本；实际 nvcuda.dll 文件版本和 SHA-256 另列并参与 engine 缓存键，驱动升级但 CUDA 能力不变也会使缓存失效。状态缓冲区建议至少 8192 字节。

## 测量与稳定性

基准调用方用 QPC 包住 `YOLO_InferJson`，五个线程共享一个句柄。固定基线模型 SHA-256 为 `ca5407763a638b86d73e9f17c423ebec39c9b0c8c1ee7f33de64aa32c73177b8`，输入 BMP800×600，conf0.5，NMS0.45。报告包含实际 DLL / Worker / 模型哈希；正式采样关闭逐帧诊断与 ORT profiling。

```powershell
# 完整基准；另以 --mode simultaneous 重复。
./yolo_benchmark_x86.exe --model best.onnx --image 业务.bmp `
  --device 0 --sessions 5 --threads 0 --warmup 100 --samples 1000 --rounds 3 `
  --profile business-calibration.tsv --workload 报告中的业务标识 `
  --mode continuous --shutdown-worker 0 --output full-call.json
# 五路30分钟稳定性，使用变化帧目录。
./yolo_benchmark_x86.exe --model best.onnx --images-dir D:/业务BMP `
  --device 2 --sessions 5 --threads 1 --seconds 1800 --rounds 1 `
  --mode continuous --shutdown-worker 0 --output soak.json
```

诊断时在测试Worker启动及模型创建前设置 `CQ_AI_YOLO_TRACE_PREFIX` 为可写路径前缀；DLL / Worker 各输出带进程号的 CSV，请求 ID 关联 BMP、等待、预处理、执行、后处理、JSON 和通信耗时。普通TensorRT调用不创建或记录阶段CUDA event，H2D/GPU/D2H诊断字段为0；诊断模式保留四个event和阶段计时。图像复制和输出完成同步不变。`python scripts/summarize_yolo_trace.py <前缀> --output stages.json` 汇总排队及各阶段 P50/P95/P99。若目标机请求打包、连接、收发合计 P95>2ms，才进入共享内存图像槽优化；当前候选的测量工具会明确报告该门槛。

DirectML 节点分配诊断使用 `CQ_AI_ORT_PROFILE_PREFIX`，分析 ORT 输出中的 provider / memcpy 事件；`mixed_cpu_fallback=true` 仅表示允许回退，不能据此宣称发生了节点回退。

验收还需覆盖空结果、阈值边界、BMP 方向/padding、多模型、并发释放、断管、Worker 重启、缓存损坏和缺少 NVIDIA 依赖，并运行受影响的 x86/x64、OCR/CV、Python 回归。RTX2070 上的缓存重建、FP16 精度、Graph 和30分钟 GPU稳定性须记录真实测试结果。任一路完整调用未达标，候选仍不满足性能验收。

## 打包

统一入口 `scripts/package_e_language.ps1` 使用当前0.14.7/v23.7元数据，检查二进制、协议26、新版本五路报告和30分钟稳定性后打包并更新output。三文件ZIP、ABI3 TensorRT模块更新包（复用既有固定版本nvidia运行库）、基准和校准工具ZIP分别交付。源码、构建、实际DLL/Worker、文件SHA及回滚记录保存在outpush。HTML在二进制编译与哈希后生成，HTML/ZIP哈希另外记录。

`package_delivery.py --nvidia-only` 可提前生成独立NVIDIA包，后续统一入口以 `-SkipNvidiaPacking` 复用其已验证哈希。`package_python_x64.ps1`构建并离线安装新版0.14.7 Wheel验收，使用独立新目录。历史release/v23.5及原始哈希不变；当前source metadata与历史manifest分别验证。旧候选验证记录见 [历史验证](YOLO_VALIDATION_CN.md)，本版交付和实测见 [v23.7交付说明](V23_7_DELIVERY_CN.md)。
