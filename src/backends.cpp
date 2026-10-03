#include "backends.h"
#if defined(_WIN32) && !defined(_M_IX86)
#include "backend_tensorrt.h"
#endif

#if defined(AIENGINE_WITH_ONNXRUNTIME)
#include "backend_onnxruntime.h"
#endif

#include "image_view.h"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace ai {

namespace {

// 将短 UTF-8 标签复制到固定大小 ABI 缓冲区，并保证以空字符结尾。
void copy_label(char* dst, int dst_size, const char* src) {
    if (dst == nullptr || dst_size <= 0) {
        return;
    }
    std::strncpy(dst, src, static_cast<size_t>(dst_size - 1));
    dst[dst_size - 1] = '\0';
}

// yolo.backend 为 null 或空时使用的占位 YOLO 后端。
// 它允许引擎在没有真实模型时完成初始化；如果尝试推理，则返回明确的“后端未配置”错误。
class NullYoloBackend final : public YoloBackend {
public:
    // null 后端不能从路径加载模型。
    int32_t load_model(const char*, const Config&, std::string*) override {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // null 后端不能从内存加载模型。
    int32_t load_model_from_memory(const void*, size_t, const Config&, std::string*) override {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // null 后端报告 YOLO 未配置。
    int32_t detect(const AIImage&, float, float, std::vector<AIDetectBox>* output) override {
        if (output != nullptr) output->clear();
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // 释放操作为空实现，重复清理也保持安全。
    int32_t release_model() override {
        return AI_OK;
    }
};

// 用于冒烟测试和 ABI 校验的确定性 YOLO 后端。
// 它不加载外部运行时，固定返回一个居中检测框。
class MockYoloBackend final : public YoloBackend {
public:
    // 路径加载只切换 loaded 状态，模型内容无关紧要。
    int32_t load_model(const char*, const Config& config, std::string* error) override {
        const int32_t status = configure(config, error);
        if (status < 0) return status;
        loaded_ = true;
        return AI_OK;
    }

    // 内存加载只检查调用方是否提供了非空字节。
    int32_t load_model_from_memory(const void* model_data, size_t model_size, const Config& config, std::string* error) override {
        if (model_data == nullptr || model_size == 0) {
            return AI_ERR_INVALID_ARGUMENT;
        }
        const int32_t status = configure(config, error);
        if (status < 0) return status;
        loaded_ = true;
        return AI_OK;
    }

    // 输出一个合成检测框，便于调用方验证结构体布局。
    int32_t detect(const AIImage& image, float conf_threshold, float, std::vector<AIDetectBox>* output) override {
        if (output == nullptr) return AI_ERR_INVALID_ARGUMENT;
        output->clear();
        if (!loaded_) {
            return AI_ERR_BACKEND_NOT_CONFIGURED;
        }
        const float w = static_cast<float>(image.width);
        const float h = static_cast<float>(image.height);
        if (conf_threshold > 0.99f) {
            return 0;
        }
        output->resize(static_cast<size_t>(mock_count_));
        for (int32_t i = 0; i < mock_count_; ++i) {
            AIDetectBox& box = (*output)[static_cast<size_t>(i)];
            const float offset = static_cast<float>(i % 10) * 0.001f;
            box.x1 = w * (0.20f + offset);
            box.y1 = h * (0.20f + offset);
            box.x2 = w * (0.70f + offset);
            box.y2 = h * (0.70f + offset);
            box.score = 0.99f;
            box.class_id = i % 3;
            copy_label(box.label, AIENGINE_MAX_LABEL, mock_label_.c_str());
        }
        return mock_count_;
    }

    // 将 mock 后端标记为未加载。
    int32_t release_model() override {
        loaded_ = false;
        return AI_OK;
    }

private:
    int32_t configure(const Config& config, std::string* error) {
        mock_count_ = config.get_int("yolo.mock_count", 1);
        mock_label_ = config.get_string("yolo.mock_label", "mock");
        if (mock_count_ < 0) {
            if (error != nullptr) *error = "yolo.mock_count must be non-negative";
            return AI_ERR_INVALID_ARGUMENT;
        }
        return AI_OK;
    }

    bool loaded_ = true;
    int32_t mock_count_ = 1;
    std::string mock_label_ = "mock";
};

// ocr.backend 为 null 或空时使用的占位 OCR 后端。
class NullOcrBackend final : public OcrBackend {
public:
    // null 后端不能从路径加载 OCR 模型。
    int32_t load_models(const char*, const char*, const Config&, std::string*) override {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // null 后端不能从内存加载 OCR 模型。
    int32_t load_models_from_memory(const void*, size_t, const void*, size_t, const Config&, std::string*) override {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // null 后端报告 OCR 未配置。
    int32_t recognize(
        const AIImage&,
        AIOcrLine*,
        int32_t,
        OcrRecognitionDiagnostics* diagnostics) override {
        if (diagnostics != nullptr) *diagnostics = OcrRecognitionDiagnostics{};
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }

    // 释放操作为空实现，重复清理也保持安全。
    int32_t release_models() override {
        return AI_OK;
    }
};

// 用于冒烟测试和 ABI 校验的确定性 OCR 后端。
// 它使用配置中的 ocr.mock_text 返回一条整图文本行。
class MockOcrBackend final : public OcrBackend {
public:
    // 保存后续识别使用的默认文本。
    explicit MockOcrBackend(std::string text) : text_(std::move(text)), lines_{text_} {}

    // 路径加载会从配置刷新 mock 文本。
    int32_t load_models(const char*, const char*, const Config& config, std::string*) override {
        configure(config);
        loaded_ = true;
        return AI_OK;
    }

    // 内存加载会检查识别模型字节是否存在。
    int32_t load_models_from_memory(const void*, size_t, const void* rec_model_data, size_t rec_model_size, const Config& config, std::string*) override {
        if (rec_model_data == nullptr || rec_model_size == 0) {
            return AI_ERR_INVALID_ARGUMENT;
        }
        configure(config);
        loaded_ = true;
        return AI_OK;
    }

    // 输出一条整图 OCR 结果，便于调用方验证结构体布局和文本解码。
    int32_t recognize(
        const AIImage& image,
        AIOcrLine* output,
        int32_t max_output,
        OcrRecognitionDiagnostics* diagnostics) override {
        if (diagnostics != nullptr) {
            *diagnostics = OcrRecognitionDiagnostics{};
            diagnostics->whole_image_recognition = true;
        }
        if (!loaded_) {
            return AI_ERR_BACKEND_NOT_CONFIGURED;
        }
        if (output == nullptr || max_output <= 0) {
            return AI_ERR_BUFFER_TOO_SMALL;
        }
        const int32_t count = std::min<int32_t>(max_output, static_cast<int32_t>(lines_.size()));
        for (int32_t i = 0; i < count; ++i) {
            const int32_t y0 = static_cast<int32_t>((static_cast<int64_t>(image.height) * i) / count);
            const int32_t y1 = static_cast<int32_t>((static_cast<int64_t>(image.height) * (i + 1)) / count);
            output[i].box = AIRect{0, y0, image.width, std::max<int32_t>(1, y1 - y0)};
            output[i].confidence = 0.99f;
            copy_label(output[i].text, AIENGINE_MAX_TEXT, lines_[static_cast<size_t>(i)].c_str());
        }
        return count;
    }

    // 将 mock 后端标记为未加载。
    int32_t release_models() override {
        loaded_ = false;
        return AI_OK;
    }

private:
    void configure(const Config& config) {
        text_ = config.get_string("ocr.mock_text", text_);
        lines_.clear();
        const std::string raw_lines = config.get_string("ocr.mock_lines", "");
        size_t start = 0;
        while (!raw_lines.empty() && start <= raw_lines.size()) {
            const size_t end = raw_lines.find('|', start);
            const std::string line = raw_lines.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!line.empty()) lines_.push_back(line);
            if (end == std::string::npos) break;
            start = end + 1;
        }
        if (lines_.empty()) lines_.push_back(text_);
    }

    std::string text_;
    std::vector<std::string> lines_;
    bool loaded_ = true;
};

} // namespace

// 创建 yolo.backend 指定的 YOLO 后端。
std::unique_ptr<YoloBackend> create_yolo_backend(const Config& config, std::string* error) {
    const std::string backend = config.get_string("yolo.backend", "null");
    if (backend == "null" || backend.empty()) {
        return std::make_unique<NullYoloBackend>();
    }
    if (backend == "mock") {
        return std::make_unique<MockYoloBackend>();
    }
    if (backend == "tensorrt") {
#if defined(_WIN32) && !defined(_M_IX86)
        return create_tensorrt_yolo_backend(config, error);
#else
        if (error) *error = "TensorRT requires the optional Windows x64 backend module";
        return nullptr;
#endif
    }
    if (backend == "onnxruntime") {
#if defined(AIENGINE_WITH_ONNXRUNTIME)
        return create_onnxruntime_yolo_backend(config, error);
#else
        if (error != nullptr) {
            *error = "yolo.backend=onnxruntime requires AIENGINE_WITH_ONNXRUNTIME=ON";
        }
        return nullptr;
#endif
    }

    if (error != nullptr) {
        *error = "Unsupported yolo.backend: " + backend;
    }
    return nullptr;
}

// 创建 ocr.backend 指定的 OCR 后端。
std::unique_ptr<OcrBackend> create_ocr_backend(const Config& config, std::string* error) {
    const std::string backend = config.get_string("ocr.backend", "null");
    if (backend == "null" || backend.empty()) {
        return std::make_unique<NullOcrBackend>();
    }
    if (backend == "mock") {
        return std::make_unique<MockOcrBackend>(config.get_string("ocr.mock_text", "MOCK"));
    }
    if (backend == "onnxruntime") {
#if defined(AIENGINE_WITH_ONNXRUNTIME)
        return create_onnxruntime_ocr_backend(config, error);
#else
        if (error != nullptr) {
            *error = "ocr.backend=onnxruntime requires AIENGINE_WITH_ONNXRUNTIME=ON";
        }
        return nullptr;
#endif
    }

    if (error != nullptr) {
        *error = "Unsupported ocr.backend: " + backend;
    }
    return nullptr;
}

} // namespace ai
