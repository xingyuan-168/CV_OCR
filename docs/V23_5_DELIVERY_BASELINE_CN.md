# v23.5 单一交付基线

## 基线事实

| 项目版本 | 交付版本 | Worker 协议 | 公开导出 |
| --- | --- | --- | --- |
| `0.14.5` | `v23.5` | `25` | `60` |

事实来源如下：

- 公开函数、结构体、常量和版本：`include/ai_engine.h`。
- Worker 协议和命名管道：`src/worker_protocol.h`。
- 设备选择、缓存、OCR、CV、YOLO 行为：`src/` 与自动测试。
- 依赖版本和构建路径：`CMakeLists.txt` 与 `scripts/prepare_dependencies.ps1`。
- 正式包文件名、大小、成员和 SHA-256：`release/v23.5/manifest.json`。

## 正式交付

`release/v23.5/` 只包含：

- `CQ_AI_e_language_v23.5.zip`
- `cq_ai_engine-0.14.5-py3-none-win_amd64.whl`
- `manifest.json`

易语言 ZIP 根目录恰好包含 `CQ_X86.dll`、`CQ_AI_worker.exe` 和
`易语言_DLL_API_说明.html`。Python Wheel 为 `py3-none-win_amd64`，包含
`CQ_AI_x64.dll`、ONNX Runtime、DirectML、VC Runtime 及许可，不包含 EXE、
x86 DLL、YOLO 模型或外部配置。

## 公开行为

### 多目标紧凑文本

| 接口 | 返回格式 | 坐标 |
| --- | --- | --- |
| `CV_FindMultiText` | `ID,x,y|ID,x,y` | 模板左上角 |
| `CV_FindTransparentMultiText` | `ID,x,y|ID,x,y` | 模板左上角 |
| `OCR_FindMultiText` | `ID,cx,cy|ID,cx,cy` | 文本中心点 |

ID 是输入列表中的零基序号。未命中项不会改变后续 ID；同一目标多次命中会
重复该 ID。结果不含空格、JSON 符号或尾竖线。正常未命中与调用错误均返回空
文本，调用方使用 `AI_GetLastError()` 区分。

### 坐标原点和内存模板包

九个业务接口在最终结果阶段应用 `origin_x/origin_y`。CV 偏移 x/y，OCR 偏移
x/y/cx/cy，YOLO 偏移 x1/x2/cx 与 y1/y2/cy；宽高、置信度、文本和序号不变。
负起点合法，超出 `int32_t` 范围返回参数错误。

`CV_LoadTemplateZipFromMemory` 接收 Stored/Deflate 标准 ZIP。包在中央目录、
边界、CRC、压缩方法、路径、BMP 和解压限制全部通过后，才原子替换指定 CV
句柄的模板集；失败不会破坏可用模板。

### 设备选择

- 易语言 x86 的 OCR/YOLO 由单一 x64 Worker 执行。AUTO 对 CPU 与 DirectML
  各执行 2 次预热和 7 次采样，以中位耗时比较；DirectML 至少快 10% 才被
  选中。结果按模型、输入、Session、CPU、显卡、驱动和运行库哈希缓存。
- Python x64 直接加载 `CQ_AI_x64.dll`。AUTO 先配置 DirectML；初始化或 Session
  创建失败时重试 CPU。Python 进程不创建 Worker。
- 显式 DirectML 失败时不整体改用 CPU；显式 CPU 不加载 DirectML。

## 构建与验收

正式产物按以下顺序产生：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/prepare_dependencies.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_python_x64.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/promote_release.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_current_release.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check_repository_hygiene.ps1 -CheckWorkingTree
```

验收要求：

- x86 与 x64 的全部 CTest 通过。
- 公开头文件、x86 DLL、x64 DLL 的未修饰导出均为 60 个；易语言 stdcall 修饰名
  参数字节数符合头文件。
- 易语言 ZIP 的 PE 位数、依赖、协议、嵌入运行库和许可检查通过。
- Wheel 可在全新 64 位 Python 环境中离线安装，CV/OCR/YOLO 验收通过且不启动
  Worker。
- manifest 与两个正式包及 ZIP 成员的大小、SHA-256 完全一致。
- API HTML 可由生成器无差异重建；仓库不存在其他版本、生成缓存或临时产物。

性能结果只适用于执行验收的硬件、模型和输入。部署环境应使用正式成品完成
业务数据验证，并通过运行状态 JSON 确认实际 Provider。
