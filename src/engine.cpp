#include "engine.h"

#include "image_view.h"

#include <utility>

namespace ai {

namespace {

bool is_valid_runtime_device(int32_t runtime_device) {
    return runtime_device == AI_DEVICE_AUTO ||
        runtime_device == AI_DEVICE_DIRECTML ||
        runtime_device == AI_DEVICE_CPU;
}

// 将 API 指定的运行设备覆盖写入 Config。
bool apply_runtime_device(Config* config, int32_t runtime_device, std::string* error, bool yolo_only = false) {
    if (config == nullptr || (!is_valid_runtime_device(runtime_device) && !(yolo_only && runtime_device == AI_DEVICE_TENSORRT))) {
        if (error != nullptr) {
            *error = "Invalid runtime device " + std::to_string(runtime_device) +
                "; valid values are 0=AUTO, 1=DirectML, 2=CPU";
        }
        return false;
    }
    config->set_int("runtime.device", runtime_device);
    return true;
}

// 当调用方提供模型数据但基础配置仍为 "null" 时，自动启用真实后端；
// 这样 mock/null 配置仍可用于冒烟测试。
void default_backend_if_empty(Config* config, const char* key, const char* backend) {
    if (config == nullptr) {
        return;
    }
    const std::string current = config->get_string(key, "");
    if (current.empty() || current == "null") {
        config->set_string(key, backend);
    }
}

} // namespace

// 使用兼容入口初始化引擎，默认自动选择 GPU/CPU。
bool Engine::init(const char* config_path, std::string* error) {
    return init_ex(config_path, AI_DEVICE_AUTO, error);
}

// 按显式运行设备偏好初始化所有模块后端。
bool Engine::init_ex(const char* config_path, int32_t runtime_device, std::string* error) {
    // 初始化会同时创建两个模块后端，即使其中一个是 null 后端。
    // 后续模型加载调用可以独立替换任一后端。
    if (!config_.load_file(config_path, error)) {
        return false;
    }
    if (!apply_runtime_device(&config_, runtime_device, error)) {
        return false;
    }

    yolo_ = create_yolo_backend(config_, error);
    if (!yolo_) {
        return false;
    }

    ocr_ = create_ocr_backend(config_, error);
    if (!ocr_) {
        return false;
    }

    return true;
}

// 用从模型路径加载的后端替换 YOLO 后端。
int32_t Engine::yolo_load_model(const char* model_path, const char* config_path, int32_t runtime_device, std::string* error) {
    // 先构建替换后端。只有新模型加载成功后才替换当前后端，
    // 因此失败时仍保留原本可用的模型。
    std::lock_guard<std::mutex> lock(yolo_mutex_);

    Config next_config = config_;
    if (config_path != nullptr && config_path[0] != '\0') {
        if (!next_config.load_file(config_path, error)) {
            return AI_ERR_CONFIG;
        }
    }
    if (!apply_runtime_device(&next_config, runtime_device, error, true)) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (model_path != nullptr && model_path[0] != '\0') {
        default_backend_if_empty(&next_config, "yolo.backend", "onnxruntime");
    if (runtime_device == AI_DEVICE_TENSORRT) next_config.set_string("yolo.backend", "tensorrt");
    }

    std::unique_ptr<YoloBackend> next_yolo = create_yolo_backend(next_config, error);
    if (!next_yolo) {
        return AI_ERR_CONFIG;
    }

    const int32_t status = next_yolo->load_model(model_path, next_config, error);
    if (status < 0) {
        return status;
    }

    config_ = std::move(next_config);
    yolo_ = std::move(next_yolo);
    return status;
}

// 用调用方提供的模型字节替换 YOLO 后端。
int32_t Engine::yolo_load_model_from_memory(const void* model_data, int32_t model_size, const char* config_path, int32_t runtime_device, std::string* error) {
    // 当脚本内嵌/加密模型且不希望暴露磁盘 ONNX 路径时，内存加载是优先集成方式。
    Config next_config = config_;
    if (config_path != nullptr && config_path[0] != '\0') {
        if (!next_config.load_file(config_path, error)) {
            return AI_ERR_CONFIG;
        }
    }
    return yolo_load_model_from_memory_with_config(model_data, model_size, std::move(next_config), runtime_device, error);
}

// 用调用方提供的配置和内存模型字节替换 YOLO 后端。
int32_t Engine::yolo_load_model_from_memory_with_config(
    const void* model_data,
    int32_t model_size,
    Config next_config,
    int32_t runtime_device,
    std::string* error) {
    if (model_data == nullptr || model_size <= 0) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(yolo_mutex_);

    if (!apply_runtime_device(&next_config, runtime_device, error, true)) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    default_backend_if_empty(&next_config, "yolo.backend", "onnxruntime");
    if (runtime_device == AI_DEVICE_TENSORRT) next_config.set_string("yolo.backend", "tensorrt");

    std::unique_ptr<YoloBackend> next_yolo = create_yolo_backend(next_config, error);
    if (!next_yolo) {
        return AI_ERR_CONFIG;
    }

    const int32_t status = next_yolo->load_model_from_memory(model_data, static_cast<size_t>(model_size), next_config, error);
    if (status < 0) {
        return status;
    }

    config_ = std::move(next_config);
    yolo_ = std::move(next_yolo);
    return status;
}

// 完成 ABI 层图像/输出校验后执行 YOLO 检测。
int32_t Engine::yolo_detect(
    const AIImage& image,
    float conf_threshold,
    float nms_threshold,
    std::vector<AIDetectBox>* output) {
    // 进入后端专用代码前，先校验简单的 ABI 调用错误。
    if (!validate_image(image)) {
        return AI_ERR_IMAGE_FORMAT;
    }
    if (output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    if (conf_threshold < 0.0f || conf_threshold > 1.0f || nms_threshold < 0.0f || nms_threshold > 1.0f) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    return yolo_->detect(image, conf_threshold, nms_threshold, output);
}

int32_t Engine::yolo_input_width() const {
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    return yolo_ ? yolo_->input_width() : 0;
}

int32_t Engine::yolo_input_height() const {
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    return yolo_ ? yolo_->input_height() : 0;
}

// 仅释放当前 YOLO 后端资源。
int32_t Engine::yolo_release() {
    // 只释放 YOLO 资源；OCR 和配置保持不变。
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    if (!yolo_) {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }
    return yolo_->release_model();
}

// 用文件系统路径加载的模型替换 OCR 后端。
int32_t Engine::ocr_load_models(const char* det_model_path, const char* rec_model_path, const char* config_path, int32_t runtime_device, std::string* error) {
    // 与 YOLO 相同，先加载到替换后端，成功后再提交。
    std::lock_guard<std::mutex> lock(ocr_mutex_);

    Config next_config = config_;
    if (config_path != nullptr && config_path[0] != '\0') {
        if (!next_config.load_file(config_path, error)) {
            return AI_ERR_CONFIG;
        }
    }
    if (!apply_runtime_device(&next_config, runtime_device, error)) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (det_model_path != nullptr && det_model_path[0] != '\0') {
        next_config.set_string("ocr.det_model_path", det_model_path);
    }
    if (rec_model_path != nullptr && rec_model_path[0] != '\0') {
        next_config.set_string("ocr.rec_model_path", rec_model_path);
    }
    if ((rec_model_path != nullptr && rec_model_path[0] != '\0') || (det_model_path != nullptr && det_model_path[0] != '\0')) {
        default_backend_if_empty(&next_config, "ocr.backend", "onnxruntime");
    }

    std::unique_ptr<OcrBackend> next_ocr = create_ocr_backend(next_config, error);
    if (!next_ocr) {
        return AI_ERR_CONFIG;
    }

    const int32_t status = next_ocr->load_models(det_model_path, rec_model_path, next_config, error);
    if (status < 0) {
        return status;
    }

    config_ = std::move(next_config);
    ocr_ = std::move(next_ocr);
    return status;
}

// 用调用方提供的模型字节替换 OCR 后端。
int32_t Engine::ocr_load_models_from_memory(
    const void* det_model_data,
    int32_t det_model_size,
    const void* rec_model_data,
    int32_t rec_model_size,
    const char* config_path,
    int32_t runtime_device,
    std::string* error) {
    // 识别模型字节必填；检测模型字节可选，因为 PP-OCR 纯识别模式会把整图当作一行。
    Config next_config = config_;
    if (config_path != nullptr && config_path[0] != '\0') {
        if (!next_config.load_file(config_path, error)) {
            return AI_ERR_CONFIG;
        }
    }
    return ocr_load_models_from_memory_with_config(
        det_model_data,
        det_model_size,
        rec_model_data,
        rec_model_size,
        std::move(next_config),
        runtime_device,
        error);
}

// 用调用方提供的配置和内存模型字节替换 OCR 后端。
int32_t Engine::ocr_load_models_from_memory_with_config(
    const void* det_model_data,
    int32_t det_model_size,
    const void* rec_model_data,
    int32_t rec_model_size,
    Config next_config,
    int32_t runtime_device,
    std::string* error) {
    if (rec_model_data == nullptr || rec_model_size <= 0 || det_model_size < 0 || (det_model_data == nullptr && det_model_size > 0)) {
        return AI_ERR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ocr_mutex_);

    if (!apply_runtime_device(&next_config, runtime_device, error)) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    default_backend_if_empty(&next_config, "ocr.backend", "onnxruntime");

    std::unique_ptr<OcrBackend> next_ocr = create_ocr_backend(next_config, error);
    if (!next_ocr) {
        return AI_ERR_CONFIG;
    }

    const int32_t status = next_ocr->load_models_from_memory(
        det_model_data,
        static_cast<size_t>(det_model_size),
        rec_model_data,
        static_cast<size_t>(rec_model_size),
        next_config,
        error);
    if (status < 0) {
        return status;
    }

    config_ = std::move(next_config);
    ocr_ = std::move(next_ocr);
    return status;
}

// 完成 ABI 层图像/输出校验后执行 OCR 识别。
int32_t Engine::ocr_recognize(
    const AIImage& image,
    AIOcrLine* output,
    int32_t max_output,
    OcrRecognitionDiagnostics* diagnostics) {
    // 当前 OCR 后端自行决定走“检测+识别”还是“纯识别”；
    // Engine 只负责校验 ABI 层参数。
    if (!validate_image(image)) {
        return AI_ERR_IMAGE_FORMAT;
    }
    if (output == nullptr || max_output <= 0) {
        return AI_ERR_BUFFER_TOO_SMALL;
    }
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    return ocr_->recognize(image, output, max_output, diagnostics);
}

// 仅释放当前 OCR 后端资源。
int32_t Engine::ocr_release() {
    // 只释放 OCR 资源；YOLO 和配置保持不变。
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    if (!ocr_) {
        return AI_ERR_BACKEND_NOT_CONFIGURED;
    }
    return ocr_->release_models();
}

// 保存模块最近一次耗时，单位微秒。
void Engine::set_latency(int32_t module, int64_t latency_us) {
    // 耗时只是诊断数据，不参与模型状态同步，因此 relaxed 原子操作已经足够。
    switch (module) {
        case AI_MODULE_CV:
            cv_latency_us_.store(latency_us, std::memory_order_relaxed);
            break;
        case AI_MODULE_OCR:
            ocr_latency_us_.store(latency_us, std::memory_order_relaxed);
            break;
        case AI_MODULE_YOLO:
            yolo_latency_us_.store(latency_us, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

// 读取模块最近一次耗时；未知模块返回 -1。
int64_t Engine::get_latency(int32_t module) const {
    // 未知模块返回 -1，便于调用方区分“模块不存在”和合法的极快 0us 结果。
    switch (module) {
        case AI_MODULE_CV:
            return cv_latency_us_.load(std::memory_order_relaxed);
        case AI_MODULE_OCR:
            return ocr_latency_us_.load(std::memory_order_relaxed);
        case AI_MODULE_YOLO:
            return yolo_latency_us_.load(std::memory_order_relaxed);
        default:
            return -1;
    }
}

int64_t Engine::get_ocr_stage_latency_us(int32_t stage) const {
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    return ocr_ ? ocr_->ocr_stage_latency_us(stage) : -1;
}

} // namespace ai
