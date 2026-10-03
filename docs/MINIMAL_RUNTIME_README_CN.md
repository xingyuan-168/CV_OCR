# CQ_AI 0.14.6 / v23.6 运行包说明

> 当前可用版为 **0.14.6 / v23.6 / Worker协议26**，交付状态为“功能验证通过、目标机性能待验”。最新三文件位于 `output/`，完整记录见 [v23.6交付说明](V23_6_DELIVERY_CN.md)。

本文描述 `v23.6` 当前可用版的部署、设备选择、缓存、构建和故障排查。多目标
返回、坐标和交付边界见 [v23.6交付说明](V23_6_DELIVERY_CN.md)。

## 1. 易语言运行目录

易语言 ZIP 根目录恰好包含：

```text
CQ_X86.dll
CQ_AI_worker.exe
易语言_DLL_API_说明.html
```

32 位易语言只加载 `CQ_X86.dll`。CV 在该 DLL 内执行；OCR 和 YOLO 通过协议
`26` 的长连接管道交给同目录 x64 Worker。运行不需要管理员权限，不写注册表或
系统目录，基础包无需CUDA/cuDNN，也不依赖系统 PATH 中的 ONNX Runtime。

历史成品的大小、成员和SHA以 [`release/v23.5/manifest.json`](../release/v23.5/manifest.json) 为准；最新成品依据本轮outpush清单。
该清单仅描述历史ZIP；最新版本在 `output/` 交付三文件，新版哈希与验证记录在outpush的v23.6清单。

## 2. API 与结果语义

公开头文件定义 60 个导出，项目版本为 `0.14.6`。错误文本接口无参数，返回
DLL 当前线程持有的只读文本：

```text
.DLL命令 AI_GetLastError, 文本型, "CQ_X86.dll", "AI_GetLastError", 公开
```

OCR支持0..2，YOLO支持以下0..3设备常量：

```text
AI_DEVICE_AUTO      ＝ 0
AI_DEVICE_DIRECTML  ＝ 1
AI_DEVICE_CPU       ＝ 2
AI_DEVICE_TENSORRT  ＝ 3  （仅YOLO）
```

OCR的设备3和所有超范围设备值返回参数错误。需要固定使用 CPU 时传 `2`。

### OCR 单次滤色与鲁棒性

`OCR_Recognize`、`OCR_FindOneText`、`OCR_FindMultiText` 和
`OCR_FindOneCoord` 的最后一个参数是可空 `color_filter`。空指针、空文本或
全空白文本启用自动预处理。显式规则格式为 `RRGGBB-RRGGBB`，多条规则以 `|`
分隔；最多 16 条、总长度最多 512 字节，非法规则返回
`AI_ERR_INVALID_ARGUMENT`。

- 显式滤色生成精确软掩膜、二值掩膜和通用抗锯齿恢复候选。恢复容差为
  `min(96, max(Δ+16, 2×Δ))`，只参与当前调用的候选竞争。
- 自动模式先走原图快速路径；检测为空、低置信、分框异常或首尾前景未覆盖时，
  再尝试灰度、Otsu 和自适应阈值候选。单次调用最多执行一个快速预处理候选和
  四个回退候选。
- 完整结果覆盖至少 92% 的有效前景列。异常小尺寸单行图可增加联合行框候选；
  多行、分栏和大间距标签按阅读顺序分框。
- TEXT、JSON 和 Find 接口使用同一最终候选。生产选择只依据图像统计、框几何
  和置信度，不依赖测试文件名、期望文本、专用颜色或固定坐标。

### 坐标原点

`CV_FindOne`、`CV_FindTransparentOne`、`CV_FindMultiText`、
`CV_FindTransparentMultiText`、`OCR_Recognize`、`OCR_FindOneText`、
`OCR_FindMultiText`、`OCR_FindOneCoord`、`YOLO_InferJson` 的末尾参数为
`origin_x/origin_y`。易语言业务封装可省略并传 `0,0`；直接 DLL 调用必须显式
传值。

偏移只作用于成功结果。CV 偏移 x/y，OCR 偏移 x/y/cx/cy，YOLO 偏移
x1/x2/cx 与 y1/y2/cy；宽高、分数、文本和序号不变。负起点合法，超出
`int32_t` 范围返回 `AI_ERR_INVALID_ARGUMENT`。

### 多目标紧凑文本

- `CV_FindMultiText`、`CV_FindTransparentMultiText`：`ID,x,y|ID,x,y`。
- `OCR_FindMultiText`：`ID,cx,cy|ID,cx,cy`。

ID 是输入列表零基序号。结果不含空格、JSON 字符或尾竖线。正常未命中和错误
均返回空文本；调用后通过 `AI_GetLastError()` 是否为空进行区分。

### 内存 ZIP 模板

`CV_LoadTemplateZipFromMemory(handle, zip_data, zip_size)` 读取完整标准 ZIP。
支持 Stored/Deflate；目录和非 BMP 项被忽略；UTF-8 标志文件名按 UTF-8 解释，
其他名称按 Windows 当前 ACP 解释。模板键为 BMP 基础文件名。

中央目录、边界、CRC、方法、路径、BMP 和解压限制全部通过后才替换句柄缓存。
限制为最多 4096 项、单 BMP 64 MiB、总 BMP 512 MiB、压缩比 200:1；加密、
ZIP64、多卷、路径穿越和重复 BMP 基础名会被拒绝。

## 3. 设备选择

### YOLO x86与x64

共用后端选择逻辑，优先读取通过真实BMP精度与性能验证的业务记录。未命中时执行五路短基准，在CPU、DirectML和TensorRT FP32间选择，并标明selection_basis。显式后端失败返回具体错误，只有AUTO允许选择其他后端。CPU按物理核预算和槽数选择内部线程，也可指定CQ_AI_YOLO_CPU_THREADS=1..8；单线程始终允许以保持正整数槽数语义，槽数大于物理核数会超订阅。

DirectML使用DXGI序号，TensorRT使用CUDA序号。`session_count`为执行槽容量，与业务线程和ORT内部线程分开；传20建立20槽，不建议盲目增加。每个DirectML Session串行，各TensorRT槽拥有独立context/stream/buffer。

校准键包含算法版本、模型SHA、尺寸、槽数、CPU、GPU、实际驱动、运行库及业务负载标识。FP16和Graph只读取通过业务门槛的记录；设备3需要独立NVIDIA包。部署和五路一键验收见 [YOLO优化说明](YOLO_OPTIMIZATION_CN.md)。

### OCR

OCR保留CPU、DirectML、AUTO和原有Worker选择行为。Python x64直接加载 `CQ_AI_x64.dll`，不创建Worker；YOLO的选择依据以模型句柄运行状态为准。

### DirectML 生命周期

DirectML 使用顺序执行模式；同一 Session 的 `Run` 由池租约串行化。ORT 可以
把 DirectML 不支持的少量图节点交给 CPU EP，状态字段
`mixed_cpu_fallback=true` 表示允许图级回退，不表示整个 Session 是 CPU。

Worker 的 OCR 长循环中，若相对首次 DML 请求的私有提交量增长 384 MiB（内存
计数不可用时最多 900 个请求），且没有存活 YOLO 句柄，Worker 会在当前响应
结束后轮换。x86 代理重放最近一次成功的 OCR 加载并重试请求；存在 YOLO 句柄
时延后轮换。Python x64 直连模式由宿主进程管理生命周期。

## 4. 内嵌运行库与私有缓存

Worker 的 XPRESS Huffman 资源包包含 13 项：

- ONNX Runtime DirectML `1.24.4` 与 DirectML `1.15.4`。
- x64 Visual C++ Runtime DLL。
- ONNX Runtime、DirectML、VC Runtime 的许可和第三方声明。
- 记录版本、Provider、大小与 SHA-256 的 `runtime-manifest.json`。

PP-OCRv6 tiny 检测模型、识别模型和字符表直接从 Worker 只读资源加载，不释放
为模型文件。YOLO 模型由调用方通过路径或内存提供。

缓存路径为：

```text
%LOCALAPPDATA%\CQ_AI\runtime\v23.6\
  ort-dml-1.24.4-<嵌入包SHA-256>\
```

CPU 基础缓存包含 `onnxruntime.dll` 和四个 VC Runtime DLL；首次 AUTO 或
DirectML 请求时按需增加 `DirectML.dll`。每次使用均核对大小和 SHA-256；损坏
时在同级临时目录重建，全部验证后替换。按运行库哈希命名的互斥锁协调多个
进程。删除该缓存不会影响正式包，下次运行会重新建立。

Worker 使用 `ORT_API_MANUAL_INIT`：先创建
`\\.\pipe\cq_ai_worker_v23_core_0145`，再验证缓存、限制 DLL 搜索目录、通过绝对
路径加载 `onnxruntime.dll` 并解析 `OrtGetApiBase`。请求 DirectML 时再验证并
加载 `DirectML.dll`。

## 5. 状态与诊断

状态 JSON 可报告：`requested`、`active`、`degraded`、
`mixed_cpu_fallback`、`device_id`、`adapter_name`、`selection_basis`、
`calibration_key`、`cpu_calibration_ms`、`directml_calibration_ms` 和 `reason`。
Worker 首次校准使用 `short_benchmark_10_percent_gate`，缓存命中使用
`cached_short_benchmark_10_percent_gate`。

诊断命令：

```cmd
CQ_AI_worker.exe --verify-embedded-runtime
CQ_AI_worker.exe --runtime-probe
CQ_AI_worker.exe --third-party-notices
```

- `--verify-embedded-runtime`：逐项解压并核对 SHA-256。
- `--runtime-probe`：建立缓存、加载 ORT，输出 Provider、路径、协议和包哈希。
- `--third-party-notices`：直接输出资源中的许可文本。

常见错误包括缓存空间不足、缓存目录不可写、内嵌包哈希不匹配、Windows 错误
126、DmlExecutionProvider 不可用、适配器越界或不支持 DirectX 12。失败后应
在同一线程立即读取 `AI_GetLastError()`。

## 6. 从源码构建

要求 Windows 10/11、Visual Studio 2022 Build Tools 和 64 位 Python 3.9+。
依赖脚本下载、校验并构建固定版本的 x86/x64 OpenCV 和运行库：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_dependencies.ps1
```

构建 x64 Worker：

```powershell
cmake -S . -B build-release-worker-x64 -G "Visual Studio 17 2022" -A x64 `
  -DAIENGINE_WITH_ONNXRUNTIME=ON `
  -DAIENGINE_ONNXRUNTIME_DIR="third_party/runtime/ort-directml-1.24.4" `
  -DAIENGINE_EMBED_OCR_ASSETS=ON `
  -DAIENGINE_BUILD_TESTS=OFF
cmake --build build-release-worker-x64 --config Release --parallel
```

构建 x86 DLL 与测试：

```powershell
cmake -S . -B build-release-x86 -G "Visual Studio 17 2022" -A Win32 `
  -DAIENGINE_BUILD_TESTS=ON `
  -DAIENGINE_WITH_OPENCV=ON `
  -DAIENGINE_OPENCV_DIR="third_party/opencv-5.0.0-static-mt/x86" `
  -DAIENGINE_WITH_ONNXRUNTIME=OFF
cmake --build build-release-x86 --config Release --parallel
Copy-Item build-release-worker-x64/Release/CQ_AI_worker.exe build-release-x86/Release/
ctest --test-dir build-release-x86 -C Release --output-on-failure
```

## 7. 打包、晋升与验证

```powershell
python scripts/generate_e_language_api_doc.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_python_x64.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_current_release.ps1 -AllowSourceCandidate
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_current_release.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_repository_hygiene.ps1 -CheckWorkingTree
```

统一入口验证PE位数、60个公开导出及stdcall别名、普通依赖、内嵌运行库和协议26；核对新版本正式五路及30分钟报告后生成HTML和ZIP。随后将现有output三文件备份到outpush/rollback，整组更新并按绝对路径加载最终DLL确认配套Worker。占用时保留原文件；替换失败时恢复原三文件。ZIP、可选NVIDIA包、工具、模型、哈希和日志保存在outpush，output只保留三文件。历史release/v23.5不改写。

## CV 多线程稳定性修复

OpenCV CV 匹配采用进程内有界工作区池：x86 同时计算上限为 2，x64 为 4；超额请求等待。空闲工作区缓存总预算分别为 64 MiB / 256 MiB，归还时超预算即释放该工作区缓冲。模板逐个处理，不再按历史模板尺寸或参与线程数无限保留评分矩阵，也不缓存跨调用结果及模板强引用。预算不覆盖正在计算的工作区、模板、返回文本或 OpenCV 临时分配，不代表进程内存峰值上限。

输入缓冲必须在调用完成前保持有效且不可被其他线程修改。共享句柄允许并发查询；清理/释放不会取消已取得模板快照的查询。不得从业务线程释放 DLL 返回文本。

`CV_FindMultiText` 参数错误分别说明 `handle`、`match_mode`、`min_score`、`color_bias`、`template_names`；缺失模板、BMP 格式、坐标溢出仍单独报告。`CV_*` 的 C++ 异常在 DLL 内转换为运行错误，文本接口返回空字符串；内存不足使用无需动态分配的后备错误文本。失败后在同一线程立即复制 `AI_GetLastError()`；下一次成功调用会清除错误。此机制不处理无效指针导致的访问违规。

同一帧的彩色通道变换和积分统计在本次调用内共享。仅在逐字节确认输入图像完全相同时复用图像预处理，变化立即失效；模板相关与结果仍每次重新计算。模板预处理随模板生命周期保存，最多保留一种图像尺寸的频域数据。工作区不持有模板强引用；模板资源不计入空闲工作区预算。候选峰值的稀疏检查与密集矩形最大值检查采用同一判定，不限制结果数量。支持 AVX2 时自动启用 SIMD，不支持时使用兼容路径。

依赖准备必须校验 IPPICV 下载哈希、生成配置中的 `HAVE_IPP` 和安装的静态库。仅设置 `WITH_IPP=ON` 不再视为依赖准备成功。

最终交付目录为 `output/`，仅包含 `CQ_X86.dll`、`CQ_AI_worker.exe` 和 `易语言_DLL_API_说明.html`；测试日志、哈希和内部候选保留在验证目录，不复制进 `output/`。本次不覆盖历史 `release/v23.5/` 包。验证与门禁误判分类见 [CV 稳定性验证](CV_STABILITY_VALIDATION_CN.md)。
