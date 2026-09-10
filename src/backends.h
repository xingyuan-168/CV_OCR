#pragma once

#include "ai_engine.h"
#include "config.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ai {

// OCR 单次识别的内部诊断信息。该结构不属于公开 C ABI，只用于区分
// “检测不到文字”和“纯识别整图输入”，并为自动预处理决定是否重试。
struct OcrRecognitionDiagnostics {
    bool detection_performed = false;
    bool detection_empty = false;
    bool whole_image_recognition = false;
    int32_t detected_box_count = 0;
    int32_t four_connected_box_count = 0;
    int32_t eight_connected_box_count = 0;
    int32_t merged_fragment_count = 0;
    bool used_eight_connectivity = false;
    bool used_merged_line_recognition = false;
    // When a conservative joint-line crop is evaluated, retain the original
    // split-box recognition as an internal competing hypothesis. This never
    // crosses the public ABI or Worker protocol boundary.
    std::vector<AIOcrLine> split_box_lines;
    int64_t detection_us = -1;
    int64_t recognition_us = -1;
    int64_t postprocess_us = -1;
};

// YOLO 后端抽象契约。
//
// 具体实现可以是 null/mock/ONNX Runtime 等。Engine 通过该接口持有后端，
// 从而保持公开 DLL API 稳定。
class YoloBackend {
public:
    virtual ~YoloBackend() = default;

    // 从路径加载模型，并使用配置中的预处理、标签等参数。
    virtual int32_t load_model(const char* model_path, const Config& config, std::string* error) = 0;

    // 从调用方已经持有的字节加载模型。
    virtual int32_t load_model_from_memory(const void* model_data, size_t model_size, const Config& config, std::string* error) = 0;

    // 使用本次调用提供的置信度和 NMS 阈值执行推理。
    virtual int32_t detect(
        const AIImage& image,
        float conf_threshold,
        float nms_threshold,
        std::vector<AIDetectBox>* output) = 0;

    virtual int32_t input_width() const { return 0; }
    virtual int32_t input_height() const { return 0; }

    // 释放模型/session 内存，同时保持后端对象可复用。
    virtual int32_t release_model() = 0;
};

// OCR 后端抽象契约。支持可选检测模型和必需识别模型，
// 因此 PP-OCR 纯识别模式可以完全跳过 det_model。
class OcrBackend {
public:
    virtual ~OcrBackend() = default;

    // 从路径/配置加载可选检测模型和必需识别模型。
    virtual int32_t load_models(const char* det_model_path, const char* rec_model_path, const Config& config, std::string* error) = 0;

    // 从内存加载可选检测模型和必需识别模型。
    virtual int32_t load_models_from_memory(
        const void* det_model_data,
        size_t det_model_size,
        const void* rec_model_data,
        size_t rec_model_size,
        const Config& config,
        std::string* error) = 0;

    // 执行 OCR，最多写入 max_output 行。
    virtual int32_t recognize(
        const AIImage& image,
        AIOcrLine* output,
        int32_t max_output,
        OcrRecognitionDiagnostics* diagnostics) = 0;

    virtual int64_t ocr_stage_latency_us(int32_t stage) const { (void)stage; return -1; }

    // 释放模型/session 内存，同时保持后端对象可复用。
    virtual int32_t release_models() = 0;
};

// 工厂函数根据配置键选择 null/mock/onnxruntime 后端。
std::unique_ptr<YoloBackend> create_yolo_backend(const Config& config, std::string* error);
std::unique_ptr<OcrBackend> create_ocr_backend(const Config& config, std::string* error);

} // namespace ai
