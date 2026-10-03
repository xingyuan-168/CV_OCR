# CQ_AI v23.6

> 当前可用版为 **0.14.6 / v23.6 / Worker协议26**，交付状态为“功能验证通过、目标机性能待验”。最新三文件位于 `output/`，完整记录见 [v23.6交付说明](docs/V23_6_DELIVERY_CN.md)。

CQ_AI 是面向 Windows 的 OpenCV、OCR 和 YOLO 视觉组件。

| 项目版本 | 交付版本 | Worker协议 | 公开函数 |
| --- | --- | --- | --- |
| `0.14.6` | `v23.6` | `26` | `60` |

32位易语言使用 `CQ_X86.dll`；OCR/YOLO由同目录 `CQ_AI_worker.exe` 处理，CV在DLL内执行。Python x64直接加载 `CQ_AI_x64.dll`。

## 当前交付物

`output/` 固定只放 `CQ_X86.dll`、`CQ_AI_worker.exe`、`易语言_DLL_API_说明.html`。三文件ZIP、含模块与运行库的可选NVIDIA包、诊断工具包、哈希清单及日志分别保存在 `outpush/`。TensorRT/FP16/Graph及五窗口20/30ms性能必须由E5-2696 v4＋RTX2070实测。

`release/v23.5/` 保留原来的ZIP、0.14.5 Wheel及manifest，成员和成品哈希不改变。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_current_release.ps1 -AllowSourceCandidate
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
cmake -S . -B build-cv-candidate-worker -G "Visual Studio 17 2022" -A x64 `
  -DAIENGINE_WITH_ONNXRUNTIME=ON `
  -DAIENGINE_ONNXRUNTIME_DIR="third_party/runtime/ort-directml-1.24.4" `
  -DAIENGINE_EMBED_OCR_ASSETS=ON `
  -DAIENGINE_BUILD_TESTS=OFF
cmake --build build-cv-candidate-worker --config Release --parallel
```

构建 x86 DLL 和测试：

```powershell
cmake -S . -B build-cv-candidate-x86 -G "Visual Studio 17 2022" -A Win32 `
  -DAIENGINE_WITH_OPENCV=ON `
  -DAIENGINE_OPENCV_DIR="third_party/opencv-5.0.0-static-mt/x86" `
  -DAIENGINE_WITH_ONNXRUNTIME=OFF `
  -DAIENGINE_BUILD_TESTS=ON
cmake --build build-cv-candidate-x86 --config Release --parallel
Copy-Item build-cv-candidate-worker/Release/CQ_AI_worker.exe build-cv-candidate-x86/Release/
ctest --test-dir build-cv-candidate-x86 -C Release --output-on-failure
```

生成交付前先完成 [v23.6验证步骤](docs/V23_6_DELIVERY_CN.md)。统一打包入口验证新DLL/Worker及新30分钟报告，生成三文件ZIP和工具包，再备份并整体更新 `output/`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_e_language.ps1
# Python采用独立新目录；不覆盖历史Wheel。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_python_x64.ps1
```

占用或验证失败会保留、恢复原配套文件；回滚三文件与原哈希保存在 `outpush/rollback/`。本轮不调用历史 `promote_release.ps1`。NVIDIA SDK准备、可选模块构建与真实BMP校准见 [YOLO说明](docs/YOLO_OPTIMIZATION_CN.md)。
