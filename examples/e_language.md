# 易语言调用入口（v23.5）

当前正式交付对应项目 `0.14.5`、交付 `v23.5`、Worker 协议 `25`。32 位易语言程序把 `CQ_X86.dll`、`CQ_AI_worker.exe` 和随包 `易语言_DLL_API_说明.html` 放在自身 EXE 同目录；运行时只依赖前两个二进制：

```text
CQ_X86.dll
CQ_AI_worker.exe
```

现行完整 DLL 命令声明见 [e_language_onnx_module.txt](e_language_onnx_module.txt)，业务调用流程见 [e_language_business_demo.txt](e_language_business_demo.txt)。

## 设备常量

```text
AI_DEVICE_AUTO      ＝ 0
AI_DEVICE_DIRECTML  ＝ 1
AI_DEVICE_CPU       ＝ 2
```

- `AUTO`：分别用 CPU 与 DirectML 候选执行 2 次预热和 7 次采样；DirectML
  中位耗时至少快 10% 才被选中，结果按模型和本机硬件缓存。
- `DirectML`：必须成功使用 DirectML；失败不整体降级。
- `CPU`：纯 CPU，不加载 DirectML。

OCR 与 YOLO 使用相同的设备值。需要强制 CPU 时传 `2`。四个 OCR 识别/查找
命令的最后一个参数为可空滤色规则，空文本表示自动识别。

`AI_Release()`只释放模型资源，不终止 Worker。程序正常结束前调用 `AI_ShutdownWorker()`；它会等待 Worker真正退出。宿主异常结束时 v23.5 Worker会自动回收。不要使用 `taskkill /IM CQ_AI_worker.exe`。

## 坐标原点

九个业务识别子程序末尾提供可空 `起点_x`、`起点_y`。省略时业务封装传
`0,0`；直接 `.DLL命令` 调用必须显式传入这两个 `int32_t` 参数。只有成功识别
的坐标会增加起点：三个多目标文本接口未命中返回空文本，其他 JSON 接口未命中
返回精确 `[]`，单结果返回 `0` 且结构体全零。支持负值，超出 32 位坐标范围时
返回参数错误。

## 多目标紧凑返回

`CV_FindMultiText`、`CV_FindTransparentMultiText` 返回 `ID,x,y|ID,x,y`；`OCR_FindMultiText` 返回 `ID,cx,cy|ID,cx,cy`。ID 是输入列表中的零基序号，未命中的目标不会让后续 ID 重新编号，同一目标多次命中会重复输出同一个 ID。结果不含空格、JSON 符号或尾竖线；未命中和调用错误都返回空文本，错误必须通过 `AI_GetLastError()` 区分。

## 从图片与资源加载 ZIP 模板

先把模板 BMP 压缩成标准 ZIP 并加入易语言“图片与资源”，再把资源字节集的地址
和长度交给 `AI_业务加载CV内存模板包`。成功返回 BMP 数量，找图时使用包内
BMP 基础文件名。ZIP 支持 Stored/Deflate、目录和中英文名；非 BMP 被忽略，
重复基础文件名、损坏包或非法 BMP 会整体失败并保留当前可用模板。完整代码见
[e_language_business_demo.txt](e_language_business_demo.txt)。

## 错误文本

`AI_GetLastError` 无参数并直接返回文本，易语言不分配字节集或长度缓冲区：

```text
.DLL命令 AI_GetLastError, 文本型, "CQ_X86.dll", "AI_GetLastError", 公开
```

调用失败时应在同一线程立即读取：

```text
加载结果 ＝ OCR_LoadEmbeddedModel (AI_DEVICE_AUTO, 1)
如果真 (加载结果 ＜ 0)
    调试输出 (AI_GetLastError ())
```

返回文本由 DLL 当前线程内部持有，调用方不得释放。

## 图像和编码

- 易语言兼容接口 `CV_*`、`OCR_*`、`YOLO_*` 的文本型路径按 Windows 当前 ACP 传入。
- 模型、标签和 OCR 字符表的内存内容使用 UTF-8。
- 识别和检测接口接收完整、未压缩的 24 位 BGR BMP 文件字节。
- 标准 `AI_*` C ABI 的文本和路径严格使用 UTF-8。

完整参数、结构体、所有权和 60 个导出说明见 [易语言 DLL API 文档](../docs/易语言_DLL_API_说明.html)。
