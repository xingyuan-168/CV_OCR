# CQ_AI Python x64 离线库

`cq-ai-engine` 是 CQ_AI `0.14.5` 的 Windows x64 离线 Wheel。安装后由 64 位
Python 直接加载包内 `CQ_AI_x64.dll`，OCR、YOLO 和 ONNX Runtime 都在当前
Python 进程内运行，不启动 `CQ_AI_worker.exe`。

## 离线安装

把 `cq_ai_engine-0.14.5-py3-none-win_amd64.whl` 复制到目标机，然后执行：

```powershell
python -m pip install --no-index .\cq_ai_engine-0.14.5-py3-none-win_amd64.whl
```

要求 Windows x64 和 64 位 Python 3.9 及以上。普通 BMP、CV、OCR、YOLO
调用没有第三方 Python 依赖；只有 `image_from_numpy()` 需要调用方自行安装
NumPy。

## 使用

可直接从顶层模块导入：

```python
from pathlib import Path

from ai_engine import AI_DEVICE_AUTO, Engine, image_from_bmp_bytes

image = image_from_bmp_bytes(Path("test.bmp").read_bytes())
with Engine() as engine:
    engine.ocr_load_embedded_models(runtime_device=AI_DEVICE_AUTO)
    print(engine.version())
    print(engine.ocr_recognize(image, min_confidence=0.5))
```

也可以从分发包命名空间导入：

```python
from cq_ai_engine import Engine
```

`Engine()` 默认从 Wheel 内的 `cq_ai_engine/_native` 加载 `CQ_AI_x64.dll` 和
同目录运行库，不读取当前工作目录中的 x86 DLL。需要调试指定 DLL 时，可显式
传入 `Engine(dll_path=...)`。

## 模型与设备

- 内置 PP-OCRv6 检测、识别模型和字符集。
- YOLO 模型不内置，使用路径或内存加载方法。
- 设备值为 `0=AUTO`、`1=DirectML`、`2=CPU`。
- x64 直连 AUTO 先配置 DirectML；初始化或 Session 创建失败时重试 CPU。它不
  执行易语言 Worker 的 2 次预热、7 次采样和 10% 性能门槛。
- 公开 C ABI 包含 60 个导出，具体签名以 `include/ai_engine.h` 为准。
- `shutdown_worker()` 是统一生命周期方法；x64 直连模式不创建 worker，因此调用成功但
  不执行进程操作。

Python x64 的 DirectML 生命周期由宿主进程管理。超长任务应监控当前 Python
进程内存，并在业务边界释放、重建 `Engine`；需要完全回收 DirectML 进程资源
时应重启宿主 Python 进程。

## Wheel 内容

单个 Wheel 内含：

- `CQ_AI_x64.dll`
- ONNX Runtime 1.24.4、DirectML 1.15.4 和所需 VC Runtime DLL
- ONNX Runtime、DirectML、VC Runtime 的许可与第三方声明

Wheel 不含 EXE、`CQ_X86.dll`、外部配置文件或 YOLO 模型。

## 构建

在已准备好仓库第三方源码和 SDK 的构建机上执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_python_x64.ps1
```

脚本会先校验当前 v23.5 正式交付，再构建 `/MT` x64 OpenCV、直连 DLL、
离线 Wheel，并验证 PE 位数、导出、依赖和 Wheel 内容。
