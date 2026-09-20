# CQ_AI v23.5

CQ_AI 是面向 Windows 的 OpenCV、OCR 和 YOLO 视觉组件。当前仓库只维护一个交付基线：

| 项目版本 | 交付版本 | Worker 协议 | 公开导出 |
| --- | --- | --- | --- |
| `0.14.5` | `v23.5` | `25` | `60` |

32 位易语言使用 `CQ_X86.dll`，OCR/YOLO 请求通过命名管道交给同目录的
`CQ_AI_worker.exe`；CV 在 x86 DLL 内直接执行。64 位 Python Wheel 直接加载
`CQ_AI_x64.dll`，不启动 Worker。

## 当前交付物

`release/v23.5/` 是唯一正式交付目录：

- `CQ_AI_e_language_v23.5.zip`：易语言 x86 DLL、x64 Worker 和 API HTML。
- `cq_ai_engine-0.14.5-py3-none-win_amd64.whl`：Python x64 离线 Wheel。
- `manifest.json`：版本、协议、文件大小、SHA-256 和包内容约束。

验证正式交付物：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_current_release.ps1
```

## 主要能力

- CV：普通/透明模板匹配、单图和多图查找、内存 ZIP 模板库、坐标原点偏移。
- OCR：内置 PP-OCRv6 tiny、CPU/DirectML/AUTO、多行排序、滤色及鲁棒性回退。
- YOLO：路径或内存加载 ONNX、多模型句柄、Session 池及运行状态查询。
- 三个多目标易语言接口使用紧凑文本：CV 返回 `ID,x,y|...`，OCR 返回
  `ID,cx,cy|...`。

## 从源码构建

环境要求：Windows 10/11、Visual Studio 2022 Build Tools、64 位 Python 3.9+。
第三方 SDK 不提交仓库，由固定版本和 SHA-256 的脚本按需准备：

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

构建 x86 DLL 和测试：

```powershell
cmake -S . -B build-release-x86 -G "Visual Studio 17 2022" -A Win32 `
  -DAIENGINE_WITH_OPENCV=ON `
  -DAIENGINE_OPENCV_DIR="third_party/opencv-5.0.0-static-mt/x86" `
  -DAIENGINE_WITH_ONNXRUNTIME=OFF `
  -DAIENGINE_BUILD_TESTS=ON
cmake --build build-release-x86 --config Release --parallel
Copy-Item build-release-worker-x64/Release/CQ_AI_worker.exe build-release-x86/Release/
ctest --test-dir build-release-x86 -C Release --output-on-failure
```

生成可再生输出到被忽略的 `outpush/`，验收后晋升为唯一正式交付：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_python_x64.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/promote_release.ps1
```

晋升脚本根据成品重新计算大小与 SHA-256，经临时目录验证后替换
`release/v23.5/`。普通构建不会直接覆盖正式交付。

## 文档

- [项目与构建说明](docs/PROJECT_GUIDE_CN.md)
- [v23.5 运行和故障排查](docs/MINIMAL_RUNTIME_README_CN.md)
- [v23.5 交付基线](docs/V23_5_DELIVERY_BASELINE_CN.md)
- [易语言 DLL API](docs/易语言_DLL_API_说明.html)
- [Python 使用说明](python/README.md)

公开 ABI 以 `include/ai_engine.h` 为准，Worker 协议以
`src/worker_protocol.h` 为准，正式成品大小和哈希以
`release/v23.5/manifest.json` 为准。第三方许可可通过
`CQ_AI_worker.exe --third-party-notices` 查看，也包含在 Python Wheel 中。
