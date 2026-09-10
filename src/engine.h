#pragma once

#include "ai_engine.h"
#include "backends.h"
#include "config.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ai {

// 持有一个逻辑上的 ai_engine 实例。
//
// 导出的 DLL 函数会维护一个共享 Engine。本类把配置、后端生命周期、模型加载、
// 推理加锁和最近耗时记录从 C ABI 层中隔离出来。
class Engine {
public:
    // 从配置初始化 YOLO/OCR 后端，默认使用 CPU。
    bool init(const char* config_path, std::string* error);

    // 从配置初始化 YOLO/OCR 后端，并覆盖 runtime.device。
    bool init_ex(const char* config_path, int32_t runtime_device, std::string* error);

    // 用路径加载的模型替换当前 YOLO 后端。
    int32_t yolo_load_model(const char* model_path, const char* config_path, int32_t runtime_device, std::string* error);

    // 用内存复制的模型替换当前 YOLO 后端。
    int32_t yolo_load_model_from_memory(const void* model_data, int32_t model_size, const char* config_path, int32_t runtime_device, std::string* error);

    // 使用调用方已经构造好的配置，从内存加载 YOLO 模型。
    int32_t yolo_load_model_from_memory_with_config(
        const void* model_data,
        int32_t model_size,
        Config next_config,
        int32_t runtime_device,
        std::string* error);

    // 通过当前后端执行 YOLO 检测。
    int32_t yolo_detect(
        const AIImage& image,
        float conf_threshold,
        float nms_threshold,
        std::vector<AIDetectBox>* output);

    int32_t yolo_input_width() const;
    int32_t yolo_input_height() const;

    // 仅释放当前 YOLO 模型资源，不销毁 Engine。
    int32_t yolo_release();

    // 用路径中的检测/识别模型替换当前 OCR 后端。
    int32_t ocr_load_models(const char* det_model_path, const char* rec_model_path, const char* config_path, int32_t runtime_device, std::string* error);

    // 用内存复制的模型替换当前 OCR 后端。
    int32_t ocr_load_models_from_memory(
        const void* det_model_data,
        int32_t det_model_size,
        const void* rec_model_data,
        int32_t rec_model_size,
        const char* config_path,
        int32_t runtime_device,
        std::string* error);

    // 使用调用方已经构造好的配置，从内存加载 OCR 模型。
    int32_t ocr_load_models_from_memory_with_config(
        const void* det_model_data,
        int32_t det_model_size,
        const void* rec_model_data,
        int32_t rec_model_size,
        Config next_config,
        int32_t runtime_device,
        std::string* error);

    // 通过当前后端执行 OCR。
    int32_t ocr_recognize(
        const AIImage& image,
        AIOcrLine* output,
        int32_t max_output,
        OcrRecognitionDiagnostics* diagnostics = nullptr);

    // 仅释放当前 OCR 模型资源，不销毁 Engine。
    int32_t ocr_release();

    // 存取导出 C 函数记录的模块耗时。
    void set_latency(int32_t module, int64_t latency_us);
    int64_t get_latency(int32_t module) const;
    int64_t get_ocr_stage_latency_us(int32_t stage) const;

private:
    // 最近一次加载的配置。模型加载时会先复制并更新它，再替换新后端；
    // 因此重载失败不会破坏旧后端。
    Config config_;
    std::unique_ptr<YoloBackend> yolo_;
    std::unique_ptr<OcrBackend> ocr_;

    // 分开的锁让 YOLO 和 OCR 调用可以互不阻塞地执行。
    mutable std::mutex yolo_mutex_;
    mutable std::mutex ocr_mutex_;

    // 使用原子变量后，读取耗时不需要获取后端锁。
    std::atomic<int64_t> cv_latency_us_{0};
    std::atomic<int64_t> ocr_latency_us_{0};
    std::atomic<int64_t> yolo_latency_us_{0};
};

} // namespace ai
