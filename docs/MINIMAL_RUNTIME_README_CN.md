# CQ_AI 0.14.5 / v23.5 运行包说明

`AI_Release()`不会终止Worker；易语言程序结束前必须调用 `AI_ShutdownWorker()`。宿主异常退出时 Worker 会监视宿主句柄并自动回收。v23.5 的多目标紧凑返回见 [交付说明](V23_5_COMPACT_MULTI_RESULT_REPORT_CN.md)；坐标原点、内存 ZIP 和 OCR 鲁棒性行为均属于当前基线。

## 1. 成品

最终目录包含两个运行文件和一份随包接口 HTML：

```text
CQ_X86.dll
CQ_AI_worker.exe
易语言_DLL_API_说明.html
```

当前 Release构建实测：

| 文件 | 位数 | 大小 |
| --- | --- | ---: |
| `CQ_X86.dll` | x86 | 9,059,840字节（约8.64 MiB） |
| `CQ_AI_worker.exe` | x64 | 24,429,568字节（约23.30 MiB） |
| `易语言_DLL_API_说明.html` | 文档 | 137,776字节（约135 KiB） |
| 合计 |  | 33,627,184字节（约32.07 MiB） |

不需要管理员权限，不安装 CUDA/cuDNN，不写注册表和系统目录，不依赖系统 PATH中的 ONNX Runtime。

## 2. 易语言接口与设备值

公开导出保持60个。三个多目标业务接口改用紧凑文本，函数名、参数和返回类型不变；九个坐标原点接口和 `CV_LoadTemplateZipFromMemory` 继续保留，标准 `AI_*` 接口、结构体和调用约定保持不变。错误文本接口仍是：

```text
.DLL命令 AI_GetLastError, 文本型, "CQ_X86.dll", "AI_GetLastError", 公开
```

三个设备常量统一用于 OCR和YOLO：

```text
AI_DEVICE_AUTO      ＝ 0
AI_DEVICE_DIRECTML  ＝ 1
AI_DEVICE_CPU       ＝ 2
```

旧代码传 `0`在 v21表示强制 CPU；v22中表示 AUTO。需要强制 CPU时必须传 `2`。设备值 `3`和其他值直接返回参数错误：

```text
Invalid runtime device 3; valid values are 0=AUTO, 1=DirectML, 2=CPU
```

## 2.1 OCR 单次调用滤色

模型加载接口保持原签名。v23 仅为 `OCR_Recognize`、`OCR_FindOneText`、`OCR_FindMultiText`、`OCR_FindOneCoord` 增加最后一个可空 `color_filter` 参数。传 `NULL`、空文本或全空白文本使用自动预处理；显式规则格式为 `RRGGBB-RRGGBB`，多条规则用 `|` 连接，目标颜色和 RGB 三通道容差均为 6 位十六进制。规则最多 16 条、总长度最多 512 字节，非法规则返回 `AI_ERR_INVALID_ARGUMENT`，详情通过 `AI_GetLastError()` 读取。滤色仅影响当前调用，不写入模型池，因此同一模型可连续使用不同颜色规则。

### 显式滤色

- 原有精确软掩膜和二值掩膜仍作为候选，规则含义及解析方式不变。
- 当前实现额外生成通用抗锯齿恢复候选。每个通道的基础容差为调用方传入的 `Δ`，恢复容差为 `min(96, max(Δ+16, 2×Δ))`；恢复范围只用于候选竞争，不回写调用参数。
- 软权重和二值结果都参与评估。恢复候选必须满足通用前景占比限制，不会因为某个文件名、文字、颜色值或坐标被强制选中。

### 自动预处理

- 原图仍是快速路径。结果空间完整且置信度正常时直接返回，不支付回退候选的推理成本。
- 检测为空、低置信、分框异常或首尾前景未覆盖时，才尝试灰度、正反极性 Otsu、正反极性自适应阈值等候选。
- 每次调用最多执行一个快速预处理候选和四个回退预处理候选，避免异常图导致无界推理。

### 完整性、联合框和输出语义

- 评分分别计算识别置信度和空间完整性。完整结果必须覆盖至少 92% 的有效前景列，并且首尾不能存在连续未覆盖文本区域；高置信但漏尾的候选不能提前结束。
- 只有检测框碎裂或扩边重叠的异常小尺寸单行图才增加整行联合识别候选。正常图、多行图、分栏和大间距独立标签继续使用原有分框。
- 原分框识别仍作为内部候选与联合框使用同一评分规则比较，不会因为生成联合框就直接丢弃。
- 联合候选对成员框的并集只扩边一次，不进行字符串前后缀删除，也不按重复汉字去重，因此不会把合法的 `人人`、`宫宫` 等文本改写掉。
- 联合候选获胜时，TEXT、JSON和 Find系列接口都使用同一最终识别结果；JSON及 Find返回成员框裁剪后的联合框。正常快速路径的分框和坐标语义不变。

上述逻辑全部依据图像统计、前景覆盖、检测框几何和置信度工作；生产代码不包含测试图片名、期望文本、专用颜色或固定坐标。

## 2.2 坐标原点与内存模板包

`CV_FindOne`、`CV_FindTransparentOne`、`CV_FindMultiText`、`CV_FindTransparentMultiText`、`OCR_Recognize`、`OCR_FindOneText`、`OCR_FindMultiText`、`OCR_FindOneCoord`、`YOLO_InferJson` 的末尾均有 `origin_x/origin_y`。易语言业务封装把两项声明为可空并默认传0；直接 DLL 调用必须显式传0。偏移只在最终结果生成阶段执行：CV偏移 `x/y`，OCR偏移 `x/y/cx/cy`，YOLO偏移 `x1/x2/cx` 与 `y1/y2/cy`，宽高和置信度不变。负起点合法；内部使用64位中间值，越界返回 `AI_ERR_INVALID_ARGUMENT`。

未命中时不会把起点作为伪坐标返回：三个多目标紧凑文本接口返回空文本，其他 JSON接口精确返回 `[]`；`CV_FindOne`、`CV_FindTransparentOne`、`OCR_FindOneText`、`OCR_FindOneCoord` 返回0且输出结构体全零。TEXT格式的 `OCR_Recognize` 文本不受起点影响。

### v23.5 多目标紧凑文本

- `CV_FindMultiText`、`CV_FindTransparentMultiText`：`ID,x,y|ID,x,y`。
- `OCR_FindMultiText`：`ID,cx,cy|ID,cx,cy`。
- ID是输入列表的零基序号，不因前项未命中而重排；相同目标的多次命中重复相同ID。
- 不添加空格、JSON符号、首尾竖线；未命中与错误均为空文本，调用后用 `AI_GetLastError()` 是否为空进行区分。

`CV_LoadTemplateZipFromMemory(handle, zip_data, zip_size)` 直接从易语言资源字节集读取标准 ZIP。支持 Stored/Deflate；目录和非BMP忽略；UTF-8标志文件名按UTF-8，否则按 Windows 当前 ACP；模板键仍为BMP基础文件名。整包通过中央目录、边界、CRC、方法、路径、BMP及解压限制检查后才原子替换句柄缓存，任何失败都会保留旧缓存。拒绝加密、ZIP64、多卷、路径穿越、重复BMP基础名和异常压缩比；默认上限为4096项、单BMP 64 MiB、总BMP 512 MiB、压缩比200:1。

## 2.3 运行设备选择

### AUTO

AUTO不允许一个池内混用不同 Provider，并从 v23开始按模型和本机硬件实测选择：

```text
创建完整 CPU候选池并短基准
  -> 创建完整 DirectML候选池并短基准
  -> DirectML中位耗时至少快10%：发布 DirectML池
  -> 否则发布 CPU池
  -> DirectML创建失败：发布 CPU池并保留失败原因
```

校准结果按模型内容、输入尺寸、Session数、CPU、显卡、驱动、ORT和内嵌资源包哈希缓存；
任一条件变化都会自动失效。显式 DirectML和显式 CPU不经过自动选择。

### DirectML

- 必须找到 `DmlExecutionProvider`、有效硬件适配器和 DirectX 12。
- 使用 `ORT_SEQUENTIAL`，关闭内存模式，同一 Session的 `Run`由池租约串行化。
- ORT可把 DML不支持的少量图节点交给 CPU EP；状态中的 `mixed_cpu_fallback=true`表示允许这种图级回退，不表示整个 Session退化为 CPU。
- 显式 DirectML失败时不整体回退 CPU。
- OCR长循环中若当前Worker相对首次DML请求的私有提交量增长384 MiB（内存计数不可用时最多900个请求），且没有存活的YOLO句柄，Worker会在当前响应完成后透明轮换；x86代理自动重放最近一次成功的OCR加载并重试请求。存在YOLO句柄时延后轮换，避免句柄失效。

### CPU

- 只使用 CPU Execution Provider。
- 不释放也不加载 `DirectML.dll`。
- 没有 DX12显卡也能运行。

## 3. Worker内嵌内容

Worker资源包使用 Windows Compression API的 XPRESS Huffman逐文件压缩，当前包含13项：

- `onnxruntime.dll`：Microsoft.ML.OnnxRuntime.DirectML 1.24.4。
- `DirectML.dll`：Microsoft.AI.DirectML 1.15.4，按需释放。
- `msvcp140.dll`、`msvcp140_1.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`。
- ORT、DirectML和 Visual C++再发行许可/通知。
- 自动生成的 `runtime-manifest.json`，包含版本、Provider、每项大小和 SHA-256。

PP-OCRv6检测模型、识别模型和字符集继续作为 Worker的只读资源直接从内存加载，不释放为模型文件。YOLO模型由调用方继续通过路径或内存传入。

`onnxruntime_providers_shared.dll`未纳入：PE依赖检查、CPU/DirectML Provider探测、OCR V6、`best.onnx`、`smc.onnx`、路径/内存加载及 DML Profiling验证均未加载该文件。若未来升级 ORT后生产路径实际需要它，打包清单必须据实重新加入。

DirectML官方未提供受支持的静态链接 DLL方案，因此本项目采用“压缩资源嵌入 + 私有缓存 + 绝对路径手动加载”，不使用内存 PE加载或自编译静态 ORT。

## 4. 私有缓存

缓存位置：

```text
%LOCALAPPDATA%\CQ_AI\runtime\v23.5\
  ort-dml-1.24.4-<嵌入包SHA-256>\
```

首次 CPU启动只释放：

```text
onnxruntime.dll
msvcp140.dll
msvcp140_1.dll
vcruntime140.dll
vcruntime140_1.dll
```

首次请求 AUTO或 DirectML时才补充：

```text
DirectML.dll
```

每次使用前按嵌入索引核对大小和 SHA-256。损坏时在同级随机临时目录重建，全部验证后原子替换。版本级命名互斥锁保证多个进程同时首次启动时不会互相覆盖。缓存被删除不影响程序，下次运行会自动重建。

旧版本缓存不由 Worker自动清理。

## 5. ORT手动加载

Worker编译时定义 `ORT_API_MANUAL_INIT`，普通 PE导入表不包含 ORT：

1. 先创建 v23.5 命名管道 `\\.\pipe\cq_ai_worker_v23_core_0145`。
2. 验证并建立 CPU基础缓存。
3. 限制 DLL搜索目录。
4. 绝对路径 `LoadLibraryExW(<缓存>\onnxruntime.dll)`。
5. 动态解析 `OrtGetApiBase`并调用 `Ort::InitApi()`。
6. 请求 DirectML时再验证/释放 `DirectML.dll`并解析 `OrtSessionOptionsAppendExecutionProvider_DML`。

因此运行库释放失败、哈希失败或 DLL加载错误都能通过管道传给 `CQ_X86.dll`，再由无参数 `AI_GetLastError()`直接读取。

## 6. 状态 JSON

OCR、YOLO句柄和全局状态字段含义一致：

```json
{
  "runtime_flavor": "core",
  "ort_version": "1.24.4",
  "ort_path": "C:\\Users\\...\\onnxruntime.dll",
  "available_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
  "requested": "auto",
  "active": "directml",
  "degraded": false,
  "mixed_cpu_fallback": true,
  "device_id": 0,
  "adapter_name": "Intel(R) UHD Graphics",
  "selection_basis": "calibrated",
  "calibration_key": "v23-cal7-...",
  "cpu_calibration_ms": 72.53,
  "directml_calibration_ms": 155.19,
  "reason": ""
}
```

AUTO因 DirectML失败改用 CPU时 `degraded=true`，`reason`保留完整失败原因；因实测
DirectML未快10%而选择 CPU属于正常性能决策，`selection_basis`和两种校准耗时会说明依据。

## 7. Worker诊断命令

```cmd
CQ_AI_worker.exe --verify-embedded-runtime
CQ_AI_worker.exe --runtime-probe
CQ_AI_worker.exe --third-party-notices
```

- `--verify-embedded-runtime`：在内存中逐项解压并核对 SHA-256。
- `--runtime-probe`：建立缓存、手动加载 ORT并输出 Provider、路径、协议和资源包哈希。
- `--third-party-notices`：从资源中输出许可文本，不产生旁文件。

常见错误包括：

```text
Embedded runtime package is corrupt
Runtime cache has insufficient free space
Cannot create runtime cache directory
Failed to extract DirectML.dll
SHA-256 mismatch for onnxruntime.dll
Failed to load embedded onnxruntime.dll: Windows error 126
DmlExecutionProvider is unavailable
DirectX 12 is unavailable on adapter 0
DirectML adapter index 1 is out of range
DirectML session creation failed
```

## 8. 构建

准备固定版本官方 SDK：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_dependencies.ps1
```

脚本从 NuGet官方下载并校验固定包哈希：

- `Microsoft.ML.OnnxRuntime.DirectML 1.24.4`
- `Microsoft.AI.DirectML 1.15.4`

构建 x64 Worker：

```powershell
cmake -S . -B build-codex-worker-v23 -G "Visual Studio 17 2022" -A x64 `
  -DAIENGINE_WITH_ONNXRUNTIME=ON `
  -DAIENGINE_ONNXRUNTIME_DIR="third_party/runtime/ort-directml-1.24.4" `
  -DAIENGINE_EMBED_OCR_ASSETS=ON `
  -DAIENGINE_BUILD_TESTS=OFF
cmake --build build-codex-worker-v23 --config Release --parallel
```

x86 DLL必须链接 `/MT`构建的静态 OpenCV。仓库旧的 `third_party/opencv-5.0.0-static/x86`是 `/MD`，不能用于 v23成品。当前验证配置使用：

```powershell
cmake -S third_party/opencv-5.0.0 `
  -B third_party/opencv-5.0.0-build-mt-x86 `
  -G "Visual Studio 17 2022" -A Win32 `
  -DCMAKE_INSTALL_PREFIX="third_party/opencv-5.0.0-static-mt/x86" `
  -DBUILD_SHARED_LIBS=OFF `
  -DBUILD_WITH_STATIC_CRT=ON `
  -DBUILD_LIST="core,imgproc" `
  -DBUILD_TESTS=OFF `
  -DBUILD_PERF_TESTS=OFF `
  -DBUILD_EXAMPLES=OFF `
  -DBUILD_opencv_apps=OFF `
  -DWITH_OPENCL=OFF
cmake --build third_party/opencv-5.0.0-build-mt-x86 `
  --config Release --target INSTALL --parallel
```

构建 x86 DLL：

```powershell
cmake -S . -B build-codex-x86-v23 -G "Visual Studio 17 2022" -A Win32 `
  -DAIENGINE_BUILD_TESTS=ON `
  -DAIENGINE_WITH_OPENCV=ON `
  -DAIENGINE_OPENCV_DIR="third_party/opencv-5.0.0-build-mt-x86" `
  -DAIENGINE_WITH_ONNXRUNTIME=OFF
cmake --build build-codex-x86-v23 --config Release --parallel
```

## 9. 打包与自动拒绝条件

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
```

脚本会拒绝：

- 公开导出不是 60/60。
- 九个坐标接口的 x86 修饰名参数字节数不匹配，或缺少 `_CV_LoadTemplateZipFromMemory@12`。
- 缺少 `_AI_GetLastError@0`或出现旧 `_AI_GetLastError@8`。
- x86 DLL不是只依赖 Windows系统 DLL。
- Worker普通导入表出现 ORT、DirectML或 VC运行库。
- Worker协议、ORT版本或 Provider探测不匹配。
- 内嵌资源或许可验证失败。
- 输出目录不是恰好包含两个运行文件和一份接口HTML。

成品输出：

```text
outpush\CQ_AI_e_language_v23.5\
outpush\CQ_AI_e_language_v23.5.zip
```

ZIP根目录包含 `CQ_X86.dll`、`CQ_AI_worker.exe` 与 `易语言_DLL_API_说明.html`；运行时只依赖前两个文件。

## 10. v23.5 发布验收

v23.5 正式发包前必须同时满足：

- 现有 CTest、OCR鲁棒性及坐标/ZIP回归全部通过。
- 三个多目标接口精确覆盖单项、多项、缺号ID、重复ID、负坐标、正负原点、空文本和无尾分隔符。
- 九个接口覆盖零、正、负起点和溢出；三个紧凑接口空结果为空文本，其他JSON空结果为 `[]`，单结果为 `0 + 全零结构体`。
- Stored/Deflate、中英文名、嵌套目录、透明模板、异常ZIP及失败后缓存回滚通过；不同句柄缓存隔离。
- OCR V6：CPU、DirectML、AUTO加载成功且相同输入结果一致。
- 显式滤色和自动模式均覆盖抗锯齿小单行回归；TEXT、JSON、Find结果一致，0、0.45、0.5、0.8四档最低置信度及重复运行稳定。
- 大图、多行、小尺寸样例、合法重复字符、空图和纯色图无非预期回归。
- 正常快速路径的P50/P95性能门禁通过。
- YOLO `best.onnx`和 `smc.onnx`：CPU和AUTO加载、推理、多句柄、多 Session及路径/内存加载通过。
- DirectML Profiling文件中存在 `DmlExecutionProvider`计算事件。
- CPU首次缓存无 `DirectML.dll`；DirectML请求后按需补充。
- 缓存缺失、单文件损坏和两个进程同时首次启动可自动恢复。
- Worker普通导入仅有 Windows系统库 `Cabinet.dll`、`bcrypt.dll`、`d3d12.dll`、`dxgi.dll`、`GDI32.dll`、`KERNEL32.dll`、`USER32.dll`。
- x86 DLL普通导入仅有 `KERNEL32.dll`。

性能数据只代表对应硬件、模型和输入。新的部署环境应使用最终 v23.5 成品执行业务验收，未执行的硬件环境不得写作已通过。
