#include "backend_tensorrt.h"
#include "tensorrt_module_api.h"
#include "runtime_status.h"
#include "yolo_diagnostics.h"
#include "yolo_decode.h"
#include "error.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <windows.h>

namespace ai {
namespace {
class TensorRTBackend final : public YoloBackend {
public:
    ~TensorRTBackend() override { release_model(); }
    int32_t load_model(const char* path, const Config& config, std::string* error) override {
        if (!path) return AI_ERR_INVALID_ARGUMENT;
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
        std::vector<char> model{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        return load_model_from_memory(model.data(), model.size(), config, error);
    }
    int32_t load_model_from_memory(const void* model, size_t length, const Config& config, std::string* error) override {
        release_model();
        if (!model || !length) return AI_ERR_INVALID_ARGUMENT;
        if ((config.get_string("yolo.precision","fp32")=="fp16" ||config.get_int("yolo.cuda_graph",0)) &&
            !config.get_int("yolo.validation_authorized",0))
            return fail(error,"FP16/CUDA Graph requires the validated YOLO pool selector",AI_ERR_CONFIG);
        HMODULE own = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&create_tensorrt_yolo_backend), &own);
        wchar_t filename[32768]{}; GetModuleFileNameW(own, filename, 32768);
        auto path = std::filesystem::path(filename).parent_path() / L"CQ_YOLO_TensorRT.dll";
        module_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module_) return fail(error, "Cannot load optional CQ_YOLO_TensorRT.dll (Windows error " + std::to_string(GetLastError()) + ")", AI_ERR_RUNTIME);
        auto get_api = reinterpret_cast<ai_trt::GetApi>(GetProcAddress(module_, "CQ_YOLO_TensorRT_GetApi"));
        api_ = get_api ? get_api(ai_trt::kAbi) : nullptr;
        if (!api_ || api_->abi != ai_trt::kAbi) return fail(error, "TensorRT module ABI mismatch", AI_ERR_RUNTIME);
        ai_trt::Options options;
        options.device = config.get_int("runtime.device_id", 0);
        options.fp16 = config.get_string("yolo.precision", "fp32") == "fp16";
        options.graph = config.get_int("yolo.cuda_graph", 0);
        // The shared selector only sets fp16/graph from validated business records,
        // or explicit offline verification options.
        const auto directory = std::filesystem::u8path(config.get_string("yolo.engine_cache", ""));
        options.cache_directory = directory.empty() ? nullptr : directory.c_str();
        char detail[4096]{};
        context_ = api_->create(model, length, &options, &info_, detail, sizeof(detail));
        if (!context_) return fail(error, detail, AI_ERR_RUNTIME);
        const int configured = config.get_int("yolo.input_width", 0);
        if (configured && (configured != info_.width || configured != info_.height))
            return fail(error, "TensorRT input shape does not match configured input_size", AI_ERR_INVALID_ARGUMENT);
        std::string text = config.get_string("yolo.labels_inline", "");
        if (text.empty()) {
            std::ifstream labels(std::filesystem::u8path(config.get_string("yolo.labels_path", "")), std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(labels), std::istreambuf_iterator<char>());
        }
        labels_.clear(); std::istringstream lines(text); std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) labels_.push_back(line);
        }
        RuntimeStatus status;
        status.requested = status.active = "tensorrt";
        status.runtime_flavor = "core"; status.device_id = options.device;
        status.adapter_name = info_.gpu; status.precision = info_.fp16 ? "fp16" : "fp32";
        status.tensorrt_version = info_.version; status.cuda_version = info_.cuda_version;
        status.driver_version = info_.driver_version; status.engine_cache_hit = info_.cache_hit != 0;
        status.driver_file_version = info_.driver_file_version; status.driver_binary_sha256 = info_.driver_binary_sha256;
        status.engine_cache_key = info_.cache_key; status.engine_build_us = info_.build_us;
        status.cuda_graph = info_.graph != 0; status.available_providers = {"TensorRT"};
        set_runtime_status(std::move(status));
        return AI_OK;
    }
    int32_t detect(const AIImage& image, float conf, float nms, std::vector<AIDetectBox>* output) override {
        if (!context_) return AI_ERR_BACKEND_NOT_CONFIGURED;
        output->clear();
        auto start = YoloClock::now();
        const auto letterbox = yolo_preprocess(image, info_.width, info_.height, api_->input(context_), columns_);
        yolo_timing.preprocess_us = elapsed_us(start);
        ai_trt::Timing timing{}; char error[4096]{};
        start = YoloClock::now();
        const auto status = api_->run(context_, &timing, error, sizeof(error));
        yolo_timing.run_us = elapsed_us(start);
        yolo_timing.h2d_us = timing.h2d_us; yolo_timing.gpu_us = timing.gpu_us; yolo_timing.d2h_us = timing.d2h_us;
        if (status < 0) { set_last_error(error); return AI_ERR_RUNTIME; }
        start = YoloClock::now();
        const auto result = yolo_decode(api_->output(context_), info_.attrs, info_.candidates,
            image, letterbox, conf, nms, labels_, candidates_, output);
        yolo_timing.postprocess_us = elapsed_us(start);
        return result;
    }
    int32_t release_model() override {
        if (context_ && api_) api_->destroy(context_);
        context_ = nullptr; api_ = nullptr;
        if (module_) FreeLibrary(module_);
        module_ = nullptr; return AI_OK;
    }
    int32_t input_width() const override { return info_.width; }
    int32_t input_height() const override { return info_.height; }
private:
    int32_t fail(std::string* error, const std::string& detail, int32_t code) {
        if (error) *error = detail; release_model(); return code;
    }
    HMODULE module_ = nullptr;
    const ai_trt::Api* api_ = nullptr;
    void* context_ = nullptr;
    ai_trt::Info info_;
    std::vector<int> columns_;
    std::vector<std::string> labels_;
    std::vector<RawDetection> candidates_;
};
}
std::unique_ptr<YoloBackend> create_tensorrt_yolo_backend(const Config&, std::string*) {
    return std::make_unique<TensorRTBackend>();
}
}
