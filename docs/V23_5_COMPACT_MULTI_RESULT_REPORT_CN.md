# v23.5 多目标紧凑文本返回实施报告

## 版本与兼容边界

- 项目版本：`0.14.5`；交付版本：`v23.5`。
- Worker 协议：`25`；管道和单例后缀：`0145`；运行库缓存目录：`v23.5`。
- 公开导出保持 60 个，函数名、参数、返回类型和 stdcall 参数字节数不变。
- 标准 `AI_*` 接口、`OCR_Recognize` JSON 和 `YOLO_InferJson` 均不变。

## 三个变更接口

| 接口 | v23.5 返回格式 | 坐标含义 |
| --- | --- | --- |
| `CV_FindMultiText` | `ID,x,y|ID,x,y` | 模板左上角 |
| `CV_FindTransparentMultiText` | `ID,x,y|ID,x,y` | 模板左上角 |
| `OCR_FindMultiText` | `ID,cx,cy|ID,cx,cy` | 命中文本中心点 |

ID 是输入目标列表中的零基序号。前面的目标未命中时，后续目标不会重新编号；同一目标出现多次时重复输出相同 ID。结果沿用原匹配顺序，数字之间只使用英文逗号，结果之间只使用英文竖线，不含空格、JSON 符号或尾竖线。

`origin_x/origin_y` 仍只在最终输出阶段应用并支持负值。CV 偏移 x/y，OCR 偏移 cx/cy；计算超出 `int32_t` 范围时返回错误，不发生静默溢出。

未命中和调用错误都返回有效空文本 `""`。调用后 `AI_GetLastError()` 为空表示正常未命中，非空表示参数、BMP、模板、模型或 Worker 错误。

## 实现

- x86 本地 CV、x64 本地 OCR 和 Worker OCR 共用相同字段语义的紧凑序列化器。
- Worker 响应仍使用长度前缀文本，只有文本语义变化；协议号升级阻止新旧 DLL/Worker 混用。
- 返回缓冲区继续由当前线程的 DLL TLS 持有，调用方不得释放，并应在同线程下一次同类调用前复制或解析。
- Python 结构化 CV 结果只包含 `id/x/y`，结构化 OCR 结果只包含 `id/cx/cy`；原始文本方法不做转换。

## 验收门槛

- 精确覆盖单结果、多结果、缺号 ID、重复 ID、负坐标、正负原点、空结果及无尾分隔符。
- CPU、DirectML、AUTO 的 OCR 格式和坐标一致；普通与透明 CV 格式一致。
- `OCR_Recognize` JSON、`YOLO_InferJson` 和全部标准 `AI_*` JSON 回归不变。
- OCR鲁棒性、坐标/ZIP和全部 CTest 通过。
- 发布包只有 `CQ_X86.dll`、`CQ_AI_worker.exe`、`易语言_DLL_API_说明.html`，公开导出 60 个且无新增运行时依赖。

## 本机构建结果

- CTest：16/16 通过；包含 CPU、DirectML、AUTO 稳定性回归和坐标/ZIP 回归。
- 导出：60/60；Worker协议：25。
- `CQ_X86.dll`：9,059,840字节，SHA-256 `5218BB92475122B4D4057DE81B5E86CCD4AACABD53EAC239CE4FFEBA113BC9E0`。
- `CQ_AI_worker.exe`：24,429,568字节，SHA-256 `E16321A3BC57C83A78E6B51F5D6A5EAF8B58F3E19838AF1E978571757F455CF4`。
- `易语言_DLL_API_说明.html`：137,776字节，SHA-256 `898E23C4D1590ADBF642E4C16C9D30524E6E0D2BBDB110FB41111C776F1DF6F3`。
