# YOLO 模型兼容与准备（0.14.7 / v23.7）

公开接口与返回格式保持不变。加载器依据模型元数据、输入输出和图信息识别检测头，不依据文件名中的版本号。推荐先运行检查工具。DLL 和 Worker 不依赖 Python。

| 模型与导出方式 | 输出 | 解码 |
| --- | --- | --- |
| 原版 YOLOv5 | `[1,N,5+C]` | objectness × 最大类别概率；跨类别 NMS 0.45 |
| V5u / V8 / YOLO11 / YOLO26 普通头 | `[1,4+C,N]` | 最大类别概率；跨类别 NMS 0.45 |
| YOLO26 原生端到端头 | `[1,K,6]` | xyxy、score、class；不再 NMS，不额外截断 |

共同边界为单输入、单检测输出、batch=1、方形 NCHW、FP32 输入输出。ORT 保留动态尺寸支持，使用实际正整数 `input_size`；静态模型传0自动读取或传相同尺寸。TensorRT 仅支持静态尺寸。分割、姿态、OBB、分类、多尺度未解码输出、外部权重和嵌入传统 NMS 的模型需从原始权重重新导出。模型不能超过512MiB。

用户的 `input/yolov26/yolo26n.onnx` 为640输入、80类、原生端到端输出，直接传 `input_size=0` 或640。配套bus图片以现有最近邻预处理、置信度0.5，CPU基线得到4人、1公交车。JPEG只在测试中转换为内存BMP；公开推理接口仍接收BMP字节集。

置信度相等时保留；最近邻、padding114、坐标还原、原点偏移、普通头跨类别NMS均沿用原行为。标签只取调用方提供的标签，未提供仍以数字回退。JSON字段、排序约定、空结果`[]`、线程内返回缓冲区、错误码集合不变。错误后立即在同一线程读取`AI_GetLastError()`。

## 检查和准备

解压工具包后，无Python也能检查：

```powershell
.\tools\ai_engine_yolo_model_inspect.exe D:/models/model.onnx
python tools/yolo_model_tool.py inspect D:/models/model.onnx
```

单类别V5与六列端到端输出可能有歧义。缺少明确格式时加载器拒绝猜测。确定模型来源后，为原版V5生成新副本：

```powershell
python tools/yolo_model_tool.py prepare D:/models/custom-v5.onnx --format objectness --classes 1 --layout BNC --output D:/prepared/custom-v5-cq.onnx
```

`prepare`离线需要Python、numpy、onnx与onnxruntime，验证基线为ORT1.24.4。它只新增`cq.yolo.schema=1`、`cq.yolo.format`、`cq.yolo.classes`、`cq.yolo.layout`元数据；核对所有非元数据字段未变，并比较零、padding、随机三种输入的原始张量输出完全相等。工具拒绝覆盖源文件、现有文件及项目`input/`、`output/`、`release/`。不能通过补标签修复错误检测头或恢复已被NMS过滤的候选框。

## 固定导出基线

原版V5使用官方v7.0导出器，FP32、batch1、静态、opset12、不启用NMS；V11/V26使用Ultralytics8.4.28，FP32、静态、opset18、`nms=False`。V26普通头另设`end2end=False`，原生端到端头设`end2end=True`。原版V5不要套用新版Ultralytics的命令参数。

仓库`scripts/export_yolo_reference_models.py`使用隔离环境并校验`configs/yolo-reference-sources.json`中的固定来源与SHA-256，生成模型和依赖清单。参考环境使用Python3.12、torch2.6.0+cpu、torchvision0.21.0+cpu、Ultralytics8.4.28、onnx1.19.1、ORT1.24.4、numpy2.2.6；V5额外依赖写入导出清单。源权重和导出ONNX的哈希分别记录。官方仓库许可证随参考来源适用，不将其Python实现嵌入DLL。

## 后端与部署

CPU和DirectML：将同一版本`CQ_X86.dll`与`CQ_AI_worker.exe`配套使用。公开60个函数、stdcall别名、协议26及正整数Session数量不变。五个业务线程可共享一个句柄；Session数是执行槽数，不是业务线程数或ORT内部线程数。

YOLO设备值0=AUTO、1=DirectML、2=CPU、3=TensorRT。`device_id`是从0开始的设备序号，CPU填0；DirectML使用DXGI列表，TensorRT使用CUDA列表，两个列表的0可能是不同显卡。AUTO将该序号分别用于每个候选后端。通过运行状态确认最终后端与设备；显式指定失败返回错误，仅AUTO允许选择其他后端。

TensorRT私有模块ABI升级为3，必须使用v23.7模块；NVIDIA运行库仍为TensorRT10.13.3.9/CUDA12.8，驱动至少支持CUDA12.8，GPU SM至少7.5。已有外部v23.6 NVIDIA归档中的`nvidia/`依赖可以复用，旧模块不可复用。模块放在Worker或x64 DLL旁；依赖放同目录`nvidia/`，或用`CQ_AI_NVIDIA_DIR`指定。新版engine与AUTO算法键已升级，旧缓存不作为新版验证记录。

CPU/DirectML已有本机真实模型证据；TensorRT仅完成构建和缺依赖错误检查，GPU执行、FP16、Graph及RTX2070性能仍待实机验证。FP16必须数量/类别一致、坐标误差≤1像素、分数差≤0.02；Graph需每轮P95改善≥10%才启用。本机证据与交付清单见[V23_7_DELIVERY_CN.md](V23_7_DELIVERY_CN.md)。
