#pragma once

#include "backends.h"

namespace ai {

// 创建基于 ONNX Runtime 的 YOLO 实现；如果配置中存在 yolo.model_path，则执行懒加载。
std::unique_ptr<YoloBackend> create_onnxruntime_yolo_backend(const Config& config, std::string* error);

// 创建基于 ONNX Runtime 的 OCR 实现；如果配置中存在 OCR 模型路径，则执行懒加载。
std::unique_ptr<OcrBackend> create_onnxruntime_ocr_backend(const Config& config, std::string* error);

} // namespace ai
