# 当前模型资产

仓库仅保留 v23.5 构建和自动验收实际使用的模型：

- `ocr_ppocrv6/det.fp32.onnx`：内嵌 OCR 检测模型。
- `ocr_ppocrv6/rec.fp32.onnx`：内嵌 OCR 识别模型。
- `ocr_ppocrv6/ppocrv6_tiny_dict.txt`：OCR 字符表。
- `yolo/best.onnx`：YOLO 单模型和内存加载回归。
- `yolo/smc.onnx`、`yolo/smc.txt`：YOLO 多模型和标签回归。

其他精度、量化或模型转换中间产物不属于当前交付。模型文件随源码用于构建或测试，
不会作为独立模型包发布；正式二进制中的第三方许可说明随 Worker/Wheel 提供。
