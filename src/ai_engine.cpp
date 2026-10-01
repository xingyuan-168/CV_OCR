#include "ai_engine.h"

#include "config.h"
#include "compact_result_format.h"
#include "coordinate_offset.h"
#include "cv_fast.h"
#include "embedded_assets.h"
#include "engine.h"
#include "error.h"
#if defined(AIENGINE_CV_TEST_HOOKS)
#include "cv_test_hooks.h"
#endif
#include "image_view.h"
#include "runtime_status.h"
#include "timer.h"
#include "worker_protocol.h"
#include "ocr_color_filter.h"

#if defined(AIENGINE_WITH_OPENCV)
#include "memory_zip.h"
#include "cv_correlation.h"
#if defined(AIENGINE_CV_AVX2)
#include "cv_masked_score_avx2.h"
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <exception>
#include <new>
#include <stdexcept>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <limits>
#include <locale>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && defined(_M_IX86)
// MSVC 在 x86 下会把 __stdcall 导出修饰成 _Name@Bytes。易语言调用方需要未修饰名称，
// 因此这里用链接器指令额外发布干净别名。
#pragma comment(linker, "/EXPORT:AI_GetVersion=_AI_GetVersion@0")
#pragma comment(linker, "/EXPORT:AI_Init=_AI_Init@4")
#pragma comment(linker, "/EXPORT:AI_InitEx=_AI_InitEx@8")
#pragma comment(linker, "/EXPORT:AI_Release=_AI_Release@0")
#pragma comment(linker, "/EXPORT:AI_ShutdownWorker=_AI_ShutdownWorker@0")
#pragma comment(linker, "/EXPORT:AI_GetLastError=_AI_GetLastError@0")
#pragma comment(linker, "/EXPORT:AI_GetLastLatencyUs=_AI_GetLastLatencyUs@4")
#pragma comment(linker, "/EXPORT:AI_GetOcrStageLatencyUs=_AI_GetOcrStageLatencyUs@4")
#pragma comment(linker, "/EXPORT:AI_GetRuntimeStatusJson=_AI_GetRuntimeStatusJson@8")
#pragma comment(linker, "/EXPORT:AI_HasEmbeddedAssets=_AI_HasEmbeddedAssets@0")
#pragma comment(linker, "/EXPORT:AI_YoloCreate=_AI_YoloCreate@4")
#pragma comment(linker, "/EXPORT:AI_YoloDetect=_AI_YoloDetect@12")
#pragma comment(linker, "/EXPORT:AI_YoloLoadModel=_AI_YoloLoadModel@28")
#pragma comment(linker, "/EXPORT:AI_YoloLoadModelFromMemory=_AI_YoloLoadModelFromMemory@32")
#pragma comment(linker, "/EXPORT:AI_YoloLoadEmbeddedModel=_AI_YoloLoadEmbeddedModel@20")
#pragma comment(linker, "/EXPORT:AI_YoloInfer=_AI_YoloInfer@12")
#pragma comment(linker, "/EXPORT:AI_YoloRelease=_AI_YoloRelease@4")
#pragma comment(linker, "/EXPORT:AI_OcrRecognize=_AI_OcrRecognize@8")
#pragma comment(linker, "/EXPORT:AI_OcrLoadModels=_AI_OcrLoadModels@16")
#pragma comment(linker, "/EXPORT:AI_OcrLoadModelsFromMemory=_AI_OcrLoadModelsFromMemory@24")
#pragma comment(linker, "/EXPORT:AI_OcrLoadEmbeddedModels=_AI_OcrLoadEmbeddedModels@4")
#pragma comment(linker, "/EXPORT:AI_OcrRelease=_AI_OcrRelease@0")
#pragma comment(linker, "/EXPORT:AI_OcrRecognizeLine=_AI_OcrRecognizeLine@20")
#pragma comment(linker, "/EXPORT:AI_OcrRecognizeLines=_AI_OcrRecognizeLines@20")
#pragma comment(linker, "/EXPORT:AI_OcrFindText=_AI_OcrFindText@12")
#pragma comment(linker, "/EXPORT:AI_CvInit=_AI_CvInit@0")
#pragma comment(linker, "/EXPORT:AI_CvToGray=_AI_CvToGray@16")
#pragma comment(linker, "/EXPORT:AI_CvThreshold=_AI_CvThreshold@20")
#pragma comment(linker, "/EXPORT:AI_CvExtractTraceJson=_AI_CvExtractTraceJson@28")
#pragma comment(linker, "/EXPORT:AI_CvMeanColor=_AI_CvMeanColor@12")
#pragma comment(linker, "/EXPORT:AI_CvFindColor=_AI_CvFindColor@20")
#pragma comment(linker, "/EXPORT:AI_CvFindImage=_AI_CvFindImage@16")
#pragma comment(linker, "/EXPORT:AI_CvFindImages=_AI_CvFindImages@16")
#pragma comment(linker, "/EXPORT:AI_CvFindTransparentImage=_AI_CvFindTransparentImage@20")
#pragma comment(linker, "/EXPORT:AI_CvFindTransparentImages=_AI_CvFindTransparentImages@20")
#pragma comment(linker, "/EXPORT:CV_Create=_CV_Create@4")
#pragma comment(linker, "/EXPORT:CV_LoadTemplateDir=_CV_LoadTemplateDir@12")
#pragma comment(linker, "/EXPORT:CV_LoadTemplateZipFromMemory=_CV_LoadTemplateZipFromMemory@12")
#pragma comment(linker, "/EXPORT:CV_ClearTemplateCache=_CV_ClearTemplateCache@4")
#pragma comment(linker, "/EXPORT:CV_Release=_CV_Release@4")
#pragma comment(linker, "/EXPORT:CV_FindOne=_CV_FindOne@36")
#pragma comment(linker, "/EXPORT:CV_FindTransparentOne=_CV_FindTransparentOne@40")
#pragma comment(linker, "/EXPORT:CV_FindMultiText=_CV_FindMultiText@36")
#pragma comment(linker, "/EXPORT:CV_FindTransparentMultiText=_CV_FindTransparentMultiText@36")
#pragma comment(linker, "/EXPORT:OCR_LoadModelFromPath=_OCR_LoadModelFromPath@20")
#pragma comment(linker, "/EXPORT:OCR_LoadModelFromMemory=_OCR_LoadModelFromMemory@32")
#pragma comment(linker, "/EXPORT:OCR_LoadEmbeddedModel=_OCR_LoadEmbeddedModel@8")
#pragma comment(linker, "/EXPORT:OCR_LoadEmbeddedModelEx=_OCR_LoadEmbeddedModelEx@12")
#pragma comment(linker, "/EXPORT:OCR_Recognize=_OCR_Recognize@28")
#pragma comment(linker, "/EXPORT:OCR_FindOneText=_OCR_FindOneText@32")
#pragma comment(linker, "/EXPORT:OCR_FindMultiText=_OCR_FindMultiText@28")
#pragma comment(linker, "/EXPORT:OCR_FindOneCoord=_OCR_FindOneCoord@32")
#pragma comment(linker, "/EXPORT:OCR_Release=_OCR_Release@0")
#pragma comment(linker, "/EXPORT:YOLO_Create=_YOLO_Create@4")
#pragma comment(linker, "/EXPORT:YOLO_LoadModelFromPath=_YOLO_LoadModelFromPath@28")
#pragma comment(linker, "/EXPORT:YOLO_LoadModelFromMemory=_YOLO_LoadModelFromMemory@36")
#pragma comment(linker, "/EXPORT:YOLO_InferJson=_YOLO_InferJson@24")
#pragma comment(linker, "/EXPORT:YOLO_GetRuntimeStatusJson=_YOLO_GetRuntimeStatusJson@12")
#pragma comment(linker, "/EXPORT:YOLO_GetLastLatencyUs=_YOLO_GetLastLatencyUs@4")
#pragma comment(linker, "/EXPORT:YOLO_Release=_YOLO_Release@4")
#endif

namespace {

static_assert(sizeof(OCRTextResult) == 28, "OCRTextResult ABI must stay 28 bytes");
constexpr float kYoloInternalNmsThreshold = 0.45f;
using ai::coordinate::offset_cv_result;
using ai::coordinate::offset_ocr_coord_result;
using ai::coordinate::offset_ocr_line;
using ai::coordinate::offset_ocr_lines;
using ai::coordinate::offset_ocr_text_result;
using ai::coordinate::offset_yolo_boxes;

// 进程级共享引擎实例，由一个小互斥锁保护。具体模块工作在 Engine 内部再用
// YOLO/OCR 专用锁保护。
std::mutex g_engine_mutex;
std::shared_ptr<ai::Engine> g_engine;

struct OwnedImage {
    std::vector<uint8_t> pixels;
    AIImage view{};
};

struct BmpImageView {
    AIImage image{};
};

struct TemplateCcoeffStats {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    double sums[3]{};
    double energy = 0.0;
};

#if defined(AIENGINE_CV_TEST_HOOKS)
std::atomic<uint64_t> g_cv_live_templates{0};
thread_local int g_cv_fault_stage = 0;
thread_local int g_cv_fault_kind = 0;
thread_local double g_cv_times[6]{};
thread_local CVTestTransparentProfile g_cv_transparent_profile{};
thread_local bool g_cv_reference = false;
thread_local bool g_cv_force_scalar_masked = false;
std::atomic<bool> g_cv_pause{false};
std::atomic<bool> g_cv_paused{false};
#endif
void cv_test_fault(int stage) {
#if defined(AIENGINE_CV_TEST_HOOKS)
    if (stage == 1 && g_cv_pause.load()) {
        g_cv_paused = true;
        while (g_cv_pause.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (g_cv_fault_stage != stage) return;
    g_cv_fault_stage = 0;
    if (g_cv_fault_kind == 1) throw std::bad_alloc();
#if defined(AIENGINE_WITH_OPENCV)
    if (g_cv_fault_kind == 2) CV_Error(cv::Error::StsError, "injected OpenCV exception");
#endif
    if (g_cv_fault_kind == 3) throw std::runtime_error("injected standard exception");
    throw 7;
#else
    (void)stage;
#endif
}

struct TemplateEntry {
#if defined(AIENGINE_CV_TEST_HOOKS)
    TemplateEntry() { ++g_cv_live_templates; }
    ~TemplateEntry() { --g_cv_live_templates; }
#endif
    std::string key;
    OwnedImage color;
    OwnedImage gray;
    TemplateCcoeffStats color_stats;
#if defined(AIENGINE_WITH_OPENCV)
    mutable std::mutex mask_mutex;
    mutable std::mutex spectrum_mutex;
    mutable std::shared_ptr<const ai::cv_detail::TemplateSpectrum> spectrum;
    mutable std::unordered_map<uint64_t, std::shared_ptr<const cv::Mat>> transparent_masks;
    mutable std::unordered_map<uint32_t, std::shared_ptr<const cv::Mat>> transparent_candidate_templates;
    mutable std::unordered_map<uint32_t, bool> transparent_color_presence;
#endif
};

using TemplateMap = std::unordered_map<std::string, std::shared_ptr<const TemplateEntry>>;

struct CVContext {
    std::shared_mutex template_mutex;
    TemplateMap templates;
};

std::mutex g_cv_context_mutex;
std::unordered_map<int32_t, std::shared_ptr<CVContext>> g_cv_contexts;
int32_t g_next_cv_handle = 1;

struct YoloSessionSlot {
    std::unique_ptr<ai::Engine> engine;
    bool busy = false;
};

struct YoloModelPool {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<YoloSessionSlot> slots;
    std::shared_ptr<const std::vector<uint8_t>> model_bytes;
    ai::RuntimeStatus runtime;
    int32_t input_width = 0;
    int32_t input_height = 0;
    int32_t device_id = 0;
    int32_t session_count = 0;
    int32_t intra_op_threads = 0;
    float nms_threshold = kYoloInternalNmsThreshold;
    std::atomic<int64_t> last_latency_us{-1};

    int32_t detect(
        const AIImage& image,
        float conf,
        std::vector<AIDetectBox>* output) {
        const auto start = std::chrono::steady_clock::now();
        ai::Engine* engine = nullptr;
        size_t slot_index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return std::any_of(slots.begin(), slots.end(), [](const YoloSessionSlot& slot) { return !slot.busy; });
            });
            for (size_t i = 0; i < slots.size(); ++i) {
                if (!slots[i].busy) {
                    slots[i].busy = true;
                    engine = slots[i].engine.get();
                    slot_index = i;
                    break;
                }
            }
        }

        const int32_t status = engine == nullptr
            ? AI_ERR_RUNTIME
            : engine->yolo_detect(image, conf, nms_threshold, output);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (slot_index < slots.size()) slots[slot_index].busy = false;
        }
        cv.notify_one();
        const int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        last_latency_us.store(elapsed, std::memory_order_relaxed);
        return status;
    }
};

enum class YoloContextState {
    Empty,
    Loading,
    Loaded,
    Closing
};

struct YoloModelContext {
    std::mutex mutex;
    YoloContextState state = YoloContextState::Empty;
    std::shared_ptr<YoloModelPool> pool;
};

struct YoloRuntimeParams {
    int32_t input_size = 0;
    int32_t runtime_device = AI_DEVICE_CPU;
    int32_t device_id = 0;
    int32_t session_count = 1;
    int32_t intra_op_threads = 1;
};

std::mutex g_yolo_context_mutex;
std::unordered_map<int32_t, std::shared_ptr<YoloModelContext>> g_yolo_contexts;
int32_t g_next_yolo_handle = 1;
thread_local std::string g_ai_cv_json_result;
thread_local std::string g_ai_ocr_json_result;
thread_local std::string g_ai_yolo_json_result;
thread_local std::string g_compat_cv_json_result;
thread_local std::string g_compat_ocr_result;
thread_local std::string g_compat_ocr_json_result;
thread_local std::string g_yolo_json_result;

bool utf8_to_windows_acp(const std::string& utf8, std::string* acp) {
    if (acp == nullptr || utf8.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    acp->clear();
    if (utf8.empty()) return true;
#if defined(_WIN32)
    const int wide_size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (wide_size <= 0) return false;
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), wide.data(), wide_size) != wide_size) {
        return false;
    }
    const int acp_size = WideCharToMultiByte(CP_ACP, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    if (acp_size <= 0) return false;
    acp->resize(static_cast<size_t>(acp_size));
    return WideCharToMultiByte(
        CP_ACP, 0, wide.data(), wide_size, acp->data(), acp_size, nullptr, nullptr) == acp_size;
#else
    *acp = utf8;
    return true;
#endif
}

bool copy_valid_utf8(const char* text, std::string* utf8) {
    if (text == nullptr || utf8 == nullptr) return false;
    const size_t length = std::strlen(text);
    if (length > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    utf8->assign(text, length);
#if defined(_WIN32)
    if (length == 0) return true;
    const int input_size = static_cast<int>(length);
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, input_size, nullptr, 0) > 0;
#else
    return true;
#endif
}

bool windows_acp_to_utf8(const char* text, std::string* utf8) {
    if (text == nullptr || utf8 == nullptr) return false;
    const size_t length = std::strlen(text);
    if (length > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    utf8->assign(text, length);
#if defined(_WIN32)
    if (length == 0) return true;
    const int input_size = static_cast<int>(length);
    const int wide_size = MultiByteToWideChar(CP_ACP, 0, text, input_size, nullptr, 0);
    if (wide_size <= 0) return false;
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(CP_ACP, 0, text, input_size, wide.data(), wide_size) != wide_size) return false;
    const int utf8_size = WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), wide_size, nullptr, 0, nullptr, nullptr);
    if (utf8_size <= 0) return false;
    utf8->resize(static_cast<size_t>(utf8_size));
    return WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), wide_size, utf8->data(), utf8_size, nullptr, nullptr) == utf8_size;
#else
    return true;
#endif
}

std::vector<std::string> compat_text_candidates(const char* text) {
    std::vector<std::string> candidates;
    std::string acp;
    if (!windows_acp_to_utf8(text, &acp)) return candidates;
    candidates.push_back(std::move(acp));

    std::string utf8;
    if (copy_valid_utf8(text, &utf8) && utf8 != candidates.front()) {
        candidates.push_back(std::move(utf8));
    }
    return candidates;
}

std::filesystem::path process_executable_directory() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size()) {
        path.resize(length);
        return std::filesystem::path(path).parent_path();
    }
#endif
    std::error_code error;
    return std::filesystem::current_path(error);
}

std::filesystem::path resolve_utf8_api_path(const char* path) {
    if (path == nullptr || path[0] == '\0') return {};
    std::string utf8;
    if (!copy_valid_utf8(path, &utf8)) return {};
    try {
        std::filesystem::path value = std::filesystem::u8path(utf8);
        if (!value.is_absolute()) value = process_executable_directory() / value;
        return value.lexically_normal();
    } catch (...) {
        return {};
    }
}

std::filesystem::path resolve_compat_api_path(const char* path) {
    if (path == nullptr || path[0] == '\0') return {};
    const std::vector<std::string> candidates = compat_text_candidates(path);
    std::filesystem::path first;
    for (const std::string& candidate : candidates) {
        try {
            std::filesystem::path value = std::filesystem::u8path(candidate);
            if (!value.is_absolute()) value = process_executable_directory() / value;
            value = value.lexically_normal();
            if (first.empty()) first = value;
            std::error_code error;
            if (std::filesystem::exists(value, error) && !error) return value;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (...) {
        }
    }
    return first;
}

const char* runtime_device_name(int32_t device) {
    switch (device) {
        case AI_DEVICE_AUTO: return "auto";
        case AI_DEVICE_DIRECTML: return "directml";
        case AI_DEVICE_CPU: return "cpu";
        default: return "invalid";
    }
}

int32_t resolve_yolo_intra_threads(int32_t session_count) {
    const unsigned int hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned int per_session = std::max(1u, hardware / static_cast<unsigned int>(session_count));
    return static_cast<int32_t>(std::min(4u, per_session));
}

int32_t make_yolo_runtime_params(
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count,
    YoloRuntimeParams* output,
    std::string* error) {
    if (output == nullptr || (input_size != 0 && input_size != 320 && input_size != 640)) {
        if (error != nullptr) *error = "YOLO input_size must be 0, 320, or 640 (0 auto-detects static models)";
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (runtime_device != AI_DEVICE_AUTO &&
        runtime_device != AI_DEVICE_DIRECTML &&
        runtime_device != AI_DEVICE_CPU) {
        if (error != nullptr) {
            *error = "Invalid runtime device " +
                std::to_string(runtime_device) +
                "; valid values are 0=AUTO, 1=DirectML, 2=CPU";
        }
        return AI_ERR_INVALID_ARGUMENT;
    }
    if (device_id < 0 || session_count <= 0) {
        if (error != nullptr) *error = "YOLO device_id must be non-negative and session_count must be positive";
        return AI_ERR_INVALID_ARGUMENT;
    }
    output->input_size = input_size;
    output->runtime_device = runtime_device;
    output->device_id = device_id;
    output->session_count = session_count;
    output->intra_op_threads = resolve_yolo_intra_threads(session_count);
    return AI_OK;
}

std::shared_ptr<YoloModelContext> get_yolo_context(int32_t handle) {
    if (handle <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_yolo_context_mutex);
    const auto it = g_yolo_contexts.find(handle);
    return it == g_yolo_contexts.end() ? nullptr : it->second;
}

int32_t create_local_yolo_context(int32_t* out_handle) {
    if (out_handle == nullptr) return AI_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(g_yolo_context_mutex);
    if (g_next_yolo_handle <= 0 || g_next_yolo_handle == std::numeric_limits<int32_t>::max()) {
        return AI_ERR_RUNTIME;
    }
    const int32_t handle = g_next_yolo_handle++;
    g_yolo_contexts.emplace(handle, std::make_shared<YoloModelContext>());
    *out_handle = handle;
    return AI_OK;
}

void close_yolo_context(const std::shared_ptr<YoloModelContext>& context) {
    if (!context) return;
    std::lock_guard<std::mutex> lock(context->mutex);
    context->state = YoloContextState::Closing;
    context->pool.reset();
}

void clear_local_yolo_contexts() {
    std::unordered_map<int32_t, std::shared_ptr<YoloModelContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(g_yolo_context_mutex);
        contexts.swap(g_yolo_contexts);
    }
    for (auto& item : contexts) close_yolo_context(item.second);
}

bool read_binary_file(const char* path, std::shared_ptr<std::vector<uint8_t>>* output, std::string* error) {
    if (path == nullptr || path[0] == '\0' || output == nullptr) {
        if (error != nullptr) *error = "YOLO model path is empty";
        return false;
    }
    const std::filesystem::path resolved_path = resolve_utf8_api_path(path);
    if (resolved_path.empty()) {
        if (error != nullptr) *error = "YOLO model path is invalid";
        return false;
    }
    std::ifstream input(resolved_path, std::ios::binary);
    if (!input) {
        if (error != nullptr) *error = std::string("failed to open YOLO model: ") + path;
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0 || size > std::numeric_limits<int32_t>::max()) {
        if (error != nullptr) *error = "YOLO model file is empty or too large";
        return false;
    }
    auto bytes = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes->data()), size);
    if (!input) {
        if (error != nullptr) *error = "failed to read the complete YOLO model";
        return false;
    }
    *output = std::move(bytes);
    return true;
}

ai::Config apply_yolo_options(ai::Config config, const YoloRuntimeParams& options) {
    config.set_string("runtime.prefer_gpu", "true");
    config.set_int("runtime.device", options.runtime_device);
    config.set_int("runtime.device_id", options.device_id);
    config.set_int("runtime.session_count", options.session_count);
    config.set_int("runtime.intra_op_threads", options.intra_op_threads);
    const std::string backend = config.get_string("yolo.backend", "");
    if (backend.empty() || backend == "null") config.set_string("yolo.backend", "onnxruntime");
    config.set_int("yolo.input_width", options.input_size);
    config.set_int("yolo.input_height", options.input_size);
    config.set_string("yolo.nms_threshold", std::to_string(kYoloInternalNmsThreshold));
    config.set_string("ocr.backend", "null");
    return config;
}

std::shared_ptr<YoloModelPool> build_local_yolo_pool(
    const std::shared_ptr<const std::vector<uint8_t>>& model_bytes,
    ai::Config config,
    const YoloRuntimeParams& options,
    int32_t* result_status,
    std::string* error) {
    if (!model_bytes || model_bytes->empty()) {
        if (error != nullptr) *error = "YOLO model data is empty";
        if (result_status != nullptr) *result_status = AI_ERR_INVALID_ARGUMENT;
        return nullptr;
    }

    const bool real_onnx = config.get_string("yolo.backend", "onnxruntime") == "onnxruntime";
    std::vector<int32_t> candidates;
    if (real_onnx && options.runtime_device == AI_DEVICE_AUTO) {
        candidates = {AI_DEVICE_DIRECTML, AI_DEVICE_CPU};
    } else {
        candidates = {options.runtime_device};
    }

    std::vector<std::string> failures;
    int32_t last_status = AI_ERR_CONFIG;
    for (const int32_t candidate : candidates) {
        auto pool = std::make_shared<YoloModelPool>();
        pool->model_bytes = model_bytes;
        pool->device_id = options.device_id;
        pool->session_count = options.session_count;
        pool->intra_op_threads = options.intra_op_threads;
        pool->nms_threshold = kYoloInternalNmsThreshold;
        pool->slots.reserve(static_cast<size_t>(options.session_count));
        ai::RuntimeStatus provider_status{runtime_device_name(options.runtime_device), runtime_device_name(candidate), "", false};
        bool candidate_ok = true;
        std::string candidate_error;

        for (int32_t i = 0; i < options.session_count; ++i) {
            auto engine = std::make_unique<ai::Engine>();
            if (!engine->init_ex(nullptr, candidate, &candidate_error)) {
                last_status = AI_ERR_CONFIG;
                candidate_ok = false;
                break;
            }
            ai::Config slot_config = config;
            slot_config.set_int("runtime.device", candidate);
            const int32_t status = engine->yolo_load_model_from_memory_with_config(
                model_bytes->data(), static_cast<int32_t>(model_bytes->size()), std::move(slot_config), candidate, &candidate_error);
            if (status < 0) {
                last_status = status;
                candidate_ok = false;
                break;
            }
            const ai::RuntimeStatus slot_status = ai::get_thread_runtime_status();
            if (real_onnx && slot_status.active != runtime_device_name(candidate)) {
                candidate_error = "YOLO session pool resolved to mixed execution providers";
                last_status = AI_ERR_RUNTIME;
                candidate_ok = false;
                break;
            }
            if (i == 0) {
                if (real_onnx) provider_status = slot_status;
                pool->input_width = engine->yolo_input_width();
                pool->input_height = engine->yolo_input_height();
            } else if (pool->input_width != engine->yolo_input_width() ||
                       pool->input_height != engine->yolo_input_height()) {
                candidate_error = "YOLO session pool resolved to inconsistent input shapes";
                last_status = AI_ERR_RUNTIME;
                candidate_ok = false;
                break;
            }
            pool->slots.push_back(YoloSessionSlot{std::move(engine), false});
        }

        if (candidate_ok && static_cast<int32_t>(pool->slots.size()) == options.session_count) {
            provider_status.requested = runtime_device_name(options.runtime_device);
            provider_status.active = real_onnx ? runtime_device_name(candidate) : config.get_string("yolo.backend", "mock");
            provider_status.degraded =
                options.runtime_device == AI_DEVICE_AUTO &&
                candidate == AI_DEVICE_CPU;
            if (!failures.empty()) {
                std::ostringstream reason;
                for (size_t i = 0; i < failures.size(); ++i) {
                    if (i > 0) reason << "; ";
                    reason << failures[i];
                }
                provider_status.reason = reason.str();
            }
            pool->runtime = std::move(provider_status);
            if (result_status != nullptr) *result_status = AI_OK;
            return pool;
        }
        failures.push_back(std::string(runtime_device_name(candidate)) + ": " +
            (candidate_error.empty() ? "session creation failed" : candidate_error));
    }

    if (error != nullptr) {
        std::ostringstream message;
        for (size_t i = 0; i < failures.size(); ++i) {
            if (i > 0) message << "; ";
            message << failures[i];
        }
        *error = message.str();
    }
    if (result_status != nullptr) *result_status = last_status;
    return nullptr;
}

int32_t load_local_yolo_context(
    int32_t handle,
    const std::shared_ptr<const std::vector<uint8_t>>& model_bytes,
    ai::Config config,
    const YoloRuntimeParams* options,
    std::string* error) {
    if (options == nullptr) return AI_ERR_INVALID_ARGUMENT;
    const auto context = get_yolo_context(handle);
    if (!context) return AI_ERR_INVALID_HANDLE;
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        if (context->state == YoloContextState::Loaded) return AI_ERR_ALREADY_LOADED;
        if (context->state == YoloContextState::Loading) return AI_ERR_BUSY;
        if (context->state == YoloContextState::Closing) return AI_ERR_INVALID_HANDLE;
        context->state = YoloContextState::Loading;
    }

    config = apply_yolo_options(std::move(config), *options);
    int32_t build_status = AI_ERR_CONFIG;
    auto pool = build_local_yolo_pool(model_bytes, std::move(config), *options, &build_status, error);
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->state == YoloContextState::Closing) return AI_ERR_INVALID_HANDLE;
    if (!pool) {
        context->state = YoloContextState::Empty;
        return build_status;
    }
    context->pool = std::move(pool);
    context->state = YoloContextState::Loaded;
    return AI_OK;
}

std::shared_ptr<YoloModelPool> get_loaded_yolo_pool(int32_t handle, int32_t* status) {
    const auto context = get_yolo_context(handle);
    if (!context) {
        if (status != nullptr) *status = AI_ERR_INVALID_HANDLE;
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->state == YoloContextState::Loading) {
        if (status != nullptr) *status = AI_ERR_BUSY;
        return nullptr;
    }
    if (context->state != YoloContextState::Loaded || !context->pool) {
        if (status != nullptr) *status = AI_ERR_BACKEND_NOT_CONFIGURED;
        return nullptr;
    }
    if (status != nullptr) *status = AI_OK;
    return context->pool;
}

std::shared_ptr<ai::Engine> ensure_engine(const char* config_path, int32_t runtime_device, std::string* error);

// 返回当前共享引擎；若尚未 AI_Init/加载模型，则返回空。
std::shared_ptr<ai::Engine> get_engine() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return g_engine;
}

// 将 DLL 资源中的只读字节视图复制成字符串配置值，用于内置 labels/charset。
std::string embedded_asset_text(const ai::EmbeddedAsset& asset) {
    if (asset.data == nullptr || asset.size == 0) {
        return std::string();
    }
    return std::string(static_cast<const char*>(asset.data), asset.size);
}

// 构造内置模型共用的运行时配置，后续仍会由 Engine 统一校正 runtime.device。
ai::Config make_embedded_base_config(int32_t runtime_device) {
    ai::Config config;
    config.set_string("runtime.prefer_gpu", "true");
    config.set_int("runtime.device", runtime_device);
    config.set_int("runtime.device_id", 0);
    config.set_int("runtime.thread_count", 0);
    return config;
}

// 构造内置 PP-OCR 识别模型需要的完整配置。
ai::Config make_embedded_ocr_config(int32_t runtime_device, const std::string& charset_text) {
    ai::Config config = make_embedded_base_config(runtime_device);
    config.set_string("yolo.backend", "null");
    config.set_string("ocr.backend", "onnxruntime");
    config.set_string("ocr.rec_only", "false");
    config.set_int("ocr.input_height", 48);
    config.set_int("ocr.input_width", 320);
    config.set_string("ocr.channel_order", "bgr");
    config.set_int("ocr.det_input_width", 0);
    config.set_int("ocr.det_input_height", 0);
    config.set_string("ocr.det_binary_threshold", "0.2");
    config.set_string("ocr.det_box_score_threshold", "0.4");
    config.set_string("ocr.det_unclip_ratio", "1.4");
    config.set_int("ocr.det_min_area", 10);
    config.set_string("ocr.charset_inline", charset_text);
    return config;
}

void apply_ocr_runtime_options(ai::Config* config, const AIOcrRuntimeOptions* options, int32_t session_count) {
    if (config == nullptr) return;
    config->set_int("runtime.session_count", std::max<int32_t>(1, session_count));
    config->set_int("runtime.intra_op_threads", options == nullptr ? 0 : options->intra_op_threads);
    if (options == nullptr) return;
    if (options->det_input_width > 0) config->set_int("ocr.det_input_width", options->det_input_width);
    if (options->det_input_height > 0) config->set_int("ocr.det_input_height", options->det_input_height);
    if (options->det_binary_threshold > 0.0f) config->set_string("ocr.det_binary_threshold", std::to_string(options->det_binary_threshold));
    if (options->det_box_score_threshold > 0.0f) config->set_string("ocr.det_box_score_threshold", std::to_string(options->det_box_score_threshold));
    if (options->det_unclip_ratio > 0.0f) config->set_string("ocr.det_unclip_ratio", std::to_string(options->det_unclip_ratio));
}

// 向后兼容的引擎创建路径，默认使用 CPU 运行。
std::shared_ptr<ai::Engine> ensure_engine(const char* config_path, std::string* error) {
    return ensure_engine(config_path, AI_DEVICE_CPU, error);
}

// 懒创建全局引擎。这样即使调用方跳过 AI_Init，直接调用
// AI_YoloLoadModelFromMemory() 等模型加载函数也能工作。
std::shared_ptr<ai::Engine> ensure_engine(const char* config_path, int32_t runtime_device, std::string* error) {
    auto engine = get_engine();
    if (engine) {
        return engine;
    }

    auto next_engine = std::make_shared<ai::Engine>();
    if (!next_engine->init_ex(config_path, runtime_device, error)) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (!g_engine) {
        g_engine = std::move(next_engine);
    }
    return g_engine;
}

// 将内部状态码转换为可读的最近错误信息。
int32_t finish_status(int32_t status, const char* operation) {
    if (status == AI_OK || status > 0) {
        ai::set_last_error("");
        return status;
    }

    switch (status) {
        case AI_ERR_INVALID_ARGUMENT:
            ai::set_last_error(std::string(operation) + ": invalid argument");
            break;
        case AI_ERR_NOT_INITIALIZED:
            ai::set_last_error(std::string(operation) + ": engine is not initialized");
            break;
        case AI_ERR_BACKEND_NOT_CONFIGURED:
            ai::set_last_error(std::string(operation) + ": backend is not configured");
            break;
        case AI_ERR_IMAGE_FORMAT:
            ai::set_last_error(std::string(operation) + ": invalid image data or format");
            break;
        case AI_ERR_BUFFER_TOO_SMALL:
            ai::set_last_error(std::string(operation) + ": output buffer is too small");
            break;
        case AI_ERR_CONFIG:
            ai::set_last_error(std::string(operation) + ": config error");
            break;
        default:
            ai::set_last_error(std::string(operation) + ": runtime error");
            break;
    }
    return status;
}

// Every CV export catches C++ exceptions before they cross the C ABI.
int32_t cv_exception_status(const char* operation) noexcept {
    try { throw; }
    catch (const std::bad_alloc&) { ai::set_last_error_fallback(operation, "out of memory"); }
#if defined(AIENGINE_WITH_OPENCV)
    catch (const cv::Exception& error) { ai::set_last_error_fallback(operation, error.what()); }
#endif
    catch (const std::exception& error) { ai::set_last_error_fallback(operation, error.what()); }
    catch (...) { ai::set_last_error_fallback(operation, "unknown C++ exception"); }
    return AI_ERR_RUNTIME;
}

const char* cv_find_argument_error(const std::shared_ptr<CVContext>& context,
                                  int32_t mode, float min_score) noexcept {
    if (!context) return "invalid handle (missing or released)";
    if (mode != 0 && mode != 1) return "invalid match_mode: expected 0 or 1";
    if (!std::isfinite(min_score) || min_score < 0.0f || min_score > 1.0f)
        return "invalid min_score: expected a finite value in [0,1]";
    return nullptr;
}

// 与 finish_status() 类似，但在可用时保留后端提供的详细错误。
int32_t finish_status_with_detail(int32_t status, const char* operation, const std::string& detail) {
    if (status == AI_OK || status > 0) {
        ai::set_last_error("");
        return status;
    }
    if (!detail.empty()) {
        ai::set_last_error(std::string(operation) + ": " + detail);
        return status;
    }
    return finish_status(status, operation);
}

int32_t finish_proxy_status(int32_t status, const char* operation) {
    if (status == AI_OK || status > 0) {
        ai::set_last_error("");
        return status;
    }
    const std::string detail = ai::last_error();
    if (!detail.empty()) {
        const std::string prefix = std::string(operation) + ": ";
        ai::set_last_error(detail.rfind(prefix, 0) == 0 ? detail : prefix + detail);
        return status;
    }
    return finish_status(status, operation);
}

// 将 UTF-8 文本复制到固定大小 ABI 缓冲区。
void copy_c_string(char* dst, int32_t dst_size, const char* src) {
    if (dst == nullptr || dst_size <= 0) {
        return;
    }
    const char* safe_src = src == nullptr ? "" : src;
    std::strncpy(dst, safe_src, static_cast<size_t>(dst_size - 1));
    dst[dst_size - 1] = '\0';
}

// 为简单 JSON 输出转义 UTF-8 文本，同时保留非 ASCII 字节内容。
std::string json_escape(const char* text) {
    std::string escaped;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(text == nullptr ? "" : text);
    while (*p != '\0') {
        const unsigned char ch = *p++;
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char buffer[7]{};
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(ch));
                    escaped += buffer;
                } else {
                    escaped.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return escaped;
}

std::string providers_json(const std::vector<std::string>& providers) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i > 0) out << ',';
        out << '"' << json_escape(providers[i].c_str()) << '"';
    }
    out << ']';
    return out.str();
}

std::string runtime_status_fields(const ai::RuntimeStatus& status) {
    return std::string("\"runtime_flavor\":\"") + json_escape(status.runtime_flavor.c_str()) +
        "\",\"ort_version\":\"" + json_escape(status.ort_version.c_str()) +
        "\",\"ort_path\":\"" + json_escape(status.ort_path.c_str()) +
        "\",\"available_providers\":" + providers_json(status.available_providers) +
        ",\"requested\":\"" + json_escape(status.requested.c_str()) +
        "\",\"active\":\"" + json_escape(status.active.c_str()) +
        "\",\"degraded\":" + (status.degraded ? "true" : "false") +
        ",\"selection_basis\":\"" +
            json_escape(status.selection_basis.c_str()) +
        "\",\"calibration_key\":\"" +
            json_escape(status.calibration_key.c_str()) +
        "\",\"cpu_calibration_ms\":" +
            std::to_string(status.cpu_calibration_ms) +
        ",\"directml_calibration_ms\":" +
            std::to_string(status.directml_calibration_ms) +
        ",\"reason\":\"" + json_escape(status.reason.c_str()) + "\"";
}

// 将一条 AIOcrLine 追加为紧凑 JSON 对象。
void append_ocr_line_json(std::ostringstream& oss, const AIOcrLine& line) {
    oss << "{\"text\":\"" << json_escape(line.text)
        << "\",\"confidence\":" << (std::isfinite(line.confidence) ? line.confidence : 0.0f)
        << ",\"box\":{\"x\":" << line.box.x
        << ",\"y\":" << line.box.y
        << ",\"w\":" << line.box.w
        << ",\"h\":" << line.box.h
        << "}}";
}

std::string format_ocr_lines_json_array(const std::vector<AIOcrLine>& lines, int32_t count) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(lines.size())));
    oss << '[';
    for (int32_t i = 0; i < safe_count; ++i) {
        if (i > 0) oss << ',';
        append_ocr_line_json(oss, lines[static_cast<size_t>(i)]);
    }
    oss << ']';
    return oss.str();
}

// 将 OCR 结果格式化为换行文本或 JSON。
std::string format_ocr_output(const std::vector<AIOcrLine>& lines, int32_t count, int32_t output_format, bool single_line) {
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(lines.size())));
    if (output_format == AI_OCR_OUTPUT_TEXT) {
        std::string text;
        const int32_t limit = single_line ? std::min<int32_t>(safe_count, 1) : safe_count;
        for (int32_t i = 0; i < limit; ++i) text += lines[static_cast<size_t>(i)].text;
        return text;
    }

    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    if (single_line) {
        if (safe_count > 0) {
            append_ocr_line_json(oss, lines[0]);
        } else {
            oss << "{\"text\":\"\",\"confidence\":0,\"box\":{\"x\":0,\"y\":0,\"w\":0,\"h\":0}}";
        }
        return oss.str();
    }

    oss << "{\"lines\":[";
    for (int32_t i = 0; i < safe_count; ++i) {
        if (i > 0) {
            oss << ',';
        }
        append_ocr_line_json(oss, lines[static_cast<size_t>(i)]);
    }
    oss << "]}";
    return oss.str();
}

std::string format_compat_ocr_output(
    const std::vector<AIOcrLine>& lines,
    int32_t count,
    int32_t output_format) {
    if (output_format == AI_OCR_OUTPUT_JSON && count <= 0) return "[]";
    return format_ocr_output(lines, count, output_format, false);
}

int32_t filter_ocr_lines(AIOcrLine* lines, int32_t count, float min_confidence) {
    if (lines == nullptr || count < 0 || min_confidence < 0.0f || min_confidence > 1.0f) return AI_ERR_INVALID_ARGUMENT;
    int32_t written = 0;
    for (int32_t i = 0; i < count; ++i) {
        if (lines[i].confidence >= min_confidence) {
            if (written != i) lines[written] = lines[i];
            ++written;
        }
    }
    return written;
}

int32_t filter_ocr_lines(std::vector<AIOcrLine>* lines, float min_confidence) {
    if (lines == nullptr || min_confidence < 0.0f || min_confidence > 1.0f) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    lines->erase(std::remove_if(lines->begin(), lines->end(), [min_confidence](const AIOcrLine& line) {
        return line.confidence < min_confidence;
    }), lines->end());
    return static_cast<int32_t>(lines->size());
}

AIRect approximate_text_box(
    const AIOcrLine& line,
    const std::string& line_text,
    size_t target_byte_offset,
    const std::string& target);

int32_t recognize_ocr_all(
    const std::shared_ptr<ai::Engine>& engine,
    const AIImage& image,
    std::vector<AIOcrLine>* output,
    ai::OcrRecognitionDiagnostics* diagnostics = nullptr) {
    if (!engine || output == nullptr) return AI_ERR_INVALID_ARGUMENT;
    output->clear();

    int32_t capacity = 16;
    for (;;) {
        std::vector<AIOcrLine> attempt(static_cast<size_t>(capacity));
        const int32_t status = engine->ocr_recognize(
            image, attempt.data(), capacity, diagnostics);
        if (status < 0) return status;
        if (status > capacity) return AI_ERR_RUNTIME;
        attempt.resize(static_cast<size_t>(status));
        if (status < capacity) {
            *output = std::move(attempt);
            return status;
        }
        if (capacity > std::numeric_limits<int32_t>::max() / 2) return AI_ERR_RUNTIME;
        capacity *= 2;
    }
}

int32_t recognize_compat_ocr(
    const std::shared_ptr<ai::Engine>& engine,
    const AIImage& image,
    const char* color_filter,
    float min_confidence,
    std::vector<AIOcrLine>* output,
    std::string* error,
    const ai::OcrTargetGroups* target_groups = nullptr,
    ai::OcrPipelineSelection* selection = nullptr) {
    if (output == nullptr || min_confidence < 0.0f || min_confidence > 1.0f) return AI_ERR_INVALID_ARGUMENT;
    ai::OcrColorFilter parsed;
    std::string parse_error;
    if (!ai::parse_ocr_color_filter(color_filter, &parsed, &parse_error)) {
        if (error != nullptr) *error = parse_error;
        return AI_ERR_INVALID_ARGUMENT;
    }
    return ai::run_ocr_candidate_pipeline(
        image,
        parsed,
        min_confidence,
        target_groups,
        [engine](
            const AIImage& candidate,
            std::vector<AIOcrLine>* lines,
            ai::OcrRecognitionDiagnostics* diagnostics,
            std::string* attempt_error) {
            const int32_t status = recognize_ocr_all(
                engine, candidate, lines, diagnostics);
            if (status < 0 && attempt_error != nullptr) {
                *attempt_error = ai::last_error();
            }
            return status;
        },
        output,
        error,
        selection);
}

int32_t find_ocr_text_in_lines(
    const std::vector<AIOcrLine>& lines,
    const std::string& target,
    std::vector<AIOcrLine>* output,
    bool preserve_line_box = false) {
    if (target.empty() || output == nullptr) return AI_ERR_INVALID_ARGUMENT;
    output->clear();
    for (const AIOcrLine& line : lines) {
        const std::string line_text(line.text);
        size_t pos = line_text.find(target);
        while (pos != std::string::npos) {
            AIOcrLine result{};
            result.box = preserve_line_box
                ? line.box
                : approximate_text_box(line, line_text, pos, target);
            result.confidence = line.confidence;
            copy_c_string(result.text, AIENGINE_MAX_TEXT, target.c_str());
            output->push_back(result);
            pos = line_text.find(target, pos + std::max<size_t>(1, target.size()));
        }
    }
    if (output->size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return AI_ERR_RUNTIME;
    return static_cast<int32_t>(output->size());
}

int32_t find_ocr_text_all(
    const std::shared_ptr<ai::Engine>& engine,
    const AIImage& image,
    const char* target_utf8,
    float min_confidence,
    std::vector<AIOcrLine>* output) {
    if (target_utf8 == nullptr || target_utf8[0] == '\0' || min_confidence < 0.0f || min_confidence > 1.0f || output == nullptr) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    std::vector<AIOcrLine> lines;
    const int32_t status = recognize_ocr_all(engine, image, &lines);
    if (status < 0) return status;
    const int32_t filtered = filter_ocr_lines(&lines, min_confidence);
    if (filtered < 0) return filtered;

    const std::string target(target_utf8);
    return find_ocr_text_in_lines(lines, target, output);
}

int32_t cv_find_images_all(
    const AIImage& image,
    const AIImage* templates,
    int32_t template_count,
    int32_t alpha_threshold,
    float min_score,
    bool transparent,
    std::vector<AIImageMatch>* output) {
    if (output == nullptr) return AI_ERR_INVALID_ARGUMENT;
    output->clear();

    int32_t capacity = std::max<int32_t>(16, template_count);
    for (;;) {
        std::vector<AIImageMatch> attempt(static_cast<size_t>(capacity));
        const int32_t status = transparent
            ? ai::cv_find_transparent_images(image, templates, template_count, alpha_threshold, min_score, attempt.data(), capacity)
            : ai::cv_find_images(image, templates, template_count, min_score, attempt.data(), capacity);
        if (status < 0) return status;
        if (status > capacity) return AI_ERR_RUNTIME;
        attempt.resize(static_cast<size_t>(status));
        if (status < capacity) {
            *output = std::move(attempt);
            return status;
        }
        if (capacity > std::numeric_limits<int32_t>::max() / 2) return AI_ERR_RUNTIME;
        capacity *= 2;
    }
}

// 写入 C 字符串结果；当调用方缓冲区无法容纳结尾空字符时返回 AI_ERR_BUFFER_TOO_SMALL。
int32_t write_string_result(const std::string& value, char* output, int32_t output_size, int32_t success_return, const char* operation) {
    if (output == nullptr || output_size <= 0) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, operation);
    }

    const int32_t required = static_cast<int32_t>(value.size() + 1);
    const int32_t copy_size = std::min(output_size - 1, static_cast<int32_t>(value.size()));
    if (copy_size > 0) {
        std::memcpy(output, value.data(), static_cast<size_t>(copy_size));
    }
    output[copy_size] = '\0';

    if (output_size < required) {
        return finish_status(AI_ERR_BUFFER_TOO_SMALL, operation);
    }
    return finish_status(success_return, operation);
}

// 统计字节前缀中的 UTF-8 码点数量。它只用于估算 OCR 命中框，
// 因此遇到非法 UTF-8 时安全退化为按字节计数。
size_t utf8_codepoint_count(const std::string& value, size_t byte_count) {
    size_t count = 0;
    size_t i = 0;
    const size_t limit = std::min(byte_count, value.size());
    while (i < limit) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        size_t step = 1;
        if ((ch & 0x80u) == 0) {
            step = 1;
        } else if ((ch & 0xe0u) == 0xc0u) {
            step = 2;
        } else if ((ch & 0xf0u) == 0xe0u) {
            step = 3;
        } else if ((ch & 0xf8u) == 0xf0u) {
            step = 4;
        }
        if (i + step > limit) {
            step = 1;
        }
        i += step;
        ++count;
    }
    return count;
}

// 按 UTF-8 码点数量比例切分完整 OCR 行框，用于估算子串边界框。
AIRect approximate_text_box(const AIOcrLine& line, const std::string& line_text, size_t byte_pos, const std::string& target) {
    const size_t total_chars = utf8_codepoint_count(line_text, line_text.size());
    const size_t prefix_chars = utf8_codepoint_count(line_text, byte_pos);
    const size_t target_chars = std::max<size_t>(1, utf8_codepoint_count(target, target.size()));
    if (total_chars == 0 || line.box.w <= 0) {
        return line.box;
    }

    const int32_t start_x = line.box.x + static_cast<int32_t>((static_cast<int64_t>(line.box.w) * static_cast<int64_t>(prefix_chars)) / static_cast<int64_t>(total_chars));
    int32_t width = static_cast<int32_t>((static_cast<int64_t>(line.box.w) * static_cast<int64_t>(target_chars) + static_cast<int64_t>(total_chars) - 1) / static_cast<int64_t>(total_chars));
    width = std::max<int32_t>(1, width);
    const int32_t right = line.box.x + line.box.w;
    width = std::min<int32_t>(width, std::max<int32_t>(1, right - start_x));
    return AIRect{start_x, line.box.y, width, line.box.h};
}

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
}

int32_t read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(read_u32_le(p));
}

bool parse_bmp_view(const uint8_t* data, int32_t size, BmpImageView* out, bool force_top_down) {
    if (data == nullptr || size < 54 || out == nullptr) {
        return false;
    }
    if (data[0] != 'B' || data[1] != 'M') {
        return false;
    }

    const uint32_t pixel_offset = read_u32_le(data + 10);
    const uint32_t dib_size = read_u32_le(data + 14);
    if (dib_size < 40 || static_cast<uint64_t>(14) + dib_size > pixel_offset || pixel_offset >= static_cast<uint32_t>(size)) {
        return false;
    }

    const int32_t width = read_i32_le(data + 18);
    const int32_t raw_height = read_i32_le(data + 22);
    const uint16_t planes = read_u16_le(data + 26);
    const uint16_t bpp = read_u16_le(data + 28);
    const uint32_t compression = read_u32_le(data + 30);
    if (width <= 0 || raw_height == 0 || raw_height == std::numeric_limits<int32_t>::min() || planes != 1 || compression != 0 || (bpp != 24 && bpp != 32)) {
        return false;
    }

    const int64_t height64 = raw_height < 0 ? -static_cast<int64_t>(raw_height) : raw_height;
    const int32_t height = static_cast<int32_t>(height64);
    const int32_t channels = bpp == 24 ? 3 : 4;
    const int64_t row_stride = ((static_cast<int64_t>(width) * channels + 3) / 4) * 4;
    if (row_stride > std::numeric_limits<int32_t>::max() ||
        height64 > (static_cast<int64_t>(size) - pixel_offset) / row_stride) {
        return false;
    }

    uint8_t* pixels = const_cast<uint8_t*>(data + pixel_offset);
    int32_t stride = static_cast<int32_t>(row_stride);
    if (!force_top_down && raw_height > 0) {
        pixels += static_cast<std::ptrdiff_t>(height - 1) * stride;
        stride = -stride;
    }

    out->image = AIImage{pixels, width, height, stride, bpp == 24 ? AI_IMAGE_BGR24 : AI_IMAGE_BGRA32};
    return true;
}

bool read_file_bytes(const std::filesystem::path& path, std::vector<uint8_t>* bytes) {
    if (bytes == nullptr) {
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size <= 0 || size > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes->resize(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes->data()), size);
    return input.good();
}

size_t checked_image_bytes(int32_t width, int32_t height, size_t channels) {
    if (width <= 0 || height <= 0 || channels == 0 ||
        static_cast<size_t>(width) > static_cast<size_t>(INT32_MAX) / channels ||
        static_cast<size_t>(height) > SIZE_MAX / channels / static_cast<size_t>(width)) {
        throw std::length_error("image dimensions exceed addressable storage");
    }
    return static_cast<size_t>(width) * static_cast<size_t>(height) * channels;
}

OwnedImage copy_image_as_bgr(const AIImage& src) {
    OwnedImage out;
    out.pixels.resize(checked_image_bytes(src.width, src.height, 3));
    out.view = AIImage{out.pixels.data(), src.width, src.height, src.width * 3, AI_IMAGE_BGR24};
    const int src_channels = ai::channels_for_format(src.format);
    for (int32_t y = 0; y < src.height; ++y) {
        const uint8_t* src_row = ai::image_row_ptr(src, y);
        uint8_t* dst_row = ai::image_row_ptr(out.view, y);
        for (int32_t x = 0; x < src.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            ai::read_bgr(src_row + x * src_channels, src.format, &b, &g, &r);
            uint8_t* dst = dst_row + x * 3;
            dst[0] = b;
            dst[1] = g;
            dst[2] = r;
        }
    }
    return out;
}

OwnedImage make_gray_image(const AIImage& src) {
    OwnedImage out;
    out.pixels.resize(checked_image_bytes(src.width, src.height, 1));
    out.view = AIImage{out.pixels.data(), src.width, src.height, src.width, AI_IMAGE_GRAY8};
    const int src_channels = ai::channels_for_format(src.format);
    for (int32_t y = 0; y < src.height; ++y) {
        const uint8_t* src_row = ai::image_row_ptr(src, y);
        uint8_t* dst_row = ai::image_row_ptr(out.view, y);
        for (int32_t x = 0; x < src.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            ai::read_bgr(src_row + x * src_channels, src.format, &b, &g, &r);
            dst_row[x] = ai::bgr_to_gray(b, g, r);
        }
    }
    return out;
}

TemplateCcoeffStats make_ccoeff_stats(const OwnedImage& image) {
    TemplateCcoeffStats stats;
    stats.width = image.view.width;
    stats.height = image.view.height;
    stats.channels = image.view.format == AI_IMAGE_GRAY8 ? 1 : 3;
    const int64_t pixel_count = static_cast<int64_t>(stats.width) * stats.height;
    if (pixel_count <= 0) return stats;
    double square_sums[3]{};
    for (int32_t y = 0; y < stats.height; ++y) {
        const uint8_t* row = image.pixels.data() + static_cast<size_t>(y) * image.view.stride;
        for (int32_t x = 0; x < stats.width; ++x) {
            for (int32_t c = 0; c < stats.channels; ++c) {
                const double value = row[x * stats.channels + c];
                stats.sums[c] += value;
                square_sums[c] += value * value;
            }
        }
    }
    for (int32_t c = 0; c < stats.channels; ++c) {
        stats.energy += square_sums[c] -
            stats.sums[c] * stats.sums[c] / static_cast<double>(pixel_count);
    }
    return stats;
}

std::string normalize_key(const std::filesystem::path& path) {
    return path.filename().u8string();
}

std::shared_ptr<TemplateEntry> load_template_memory(
    const std::string& key,
    const uint8_t* data,
    int32_t size) {
    BmpImageView bmp;
    if (key.empty() || !parse_bmp_view(data, size, &bmp, false)) return nullptr;
    auto entry = std::make_shared<TemplateEntry>();
    entry->key = key;
    entry->color = copy_image_as_bgr(bmp.image);
    entry->gray = make_gray_image(entry->color.view);
    entry->color_stats = make_ccoeff_stats(entry->color);
    return entry;
}

std::shared_ptr<TemplateEntry> load_template_file(const std::filesystem::path& path) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, &bytes)) {
        return nullptr;
    }
    return load_template_memory(
        normalize_key(path), bytes.data(), static_cast<int32_t>(bytes.size()));
}

#if defined(AIENGINE_WITH_OPENCV)
std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool decode_zip_template_key(const ai::MemoryZipEntry& entry, std::string* key) {
    if (key == nullptr || entry.name_bytes.empty()) return false;
    std::string decoded;
    if (entry.utf8_name) {
        if (!copy_valid_utf8(entry.name_bytes.c_str(), &decoded)) return false;
    } else if (!windows_acp_to_utf8(entry.name_bytes.c_str(), &decoded)) {
        return false;
    }
    std::replace(decoded.begin(), decoded.end(), '\\', '/');
    try {
        *key = std::filesystem::u8path(decoded).filename().u8string();
    } catch (const std::bad_alloc&) {
            throw;
        } catch (...) {
        return false;
    }
    return !key->empty();
}
#endif

std::vector<std::string> split_pipe(const char* value) {
    std::vector<std::string> parts;
    if (value == nullptr) {
        return parts;
    }
    std::string text(value);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t pos = text.find('|', start);
        std::string part = text.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!part.empty()) {
            parts.push_back(part);
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return parts;
}

std::vector<std::vector<std::string>> compat_pipe_text_candidates(const char* value) {
    const std::vector<std::string> whole_candidates = compat_text_candidates(value);
    if (whole_candidates.empty()) return {};

    const auto split_checked = [](const std::string& text, std::vector<std::string>* parts) {
        if (parts == nullptr || text.empty() || text.front() == '|' || text.back() == '|' ||
            text.find("||") != std::string::npos) {
            return false;
        }
        *parts = split_pipe(text.c_str());
        return !parts->empty();
    };

    std::vector<std::string> primary;
    if (!split_checked(whole_candidates.front(), &primary)) return {};
    std::vector<std::vector<std::string>> result;
    result.reserve(primary.size());
    for (std::string& part : primary) result.push_back({std::move(part)});

    for (size_t candidate_index = 1; candidate_index < whole_candidates.size(); ++candidate_index) {
        std::vector<std::string> alternate;
        if (!split_checked(whole_candidates[candidate_index], &alternate) || alternate.size() != result.size()) continue;
        for (size_t i = 0; i < alternate.size(); ++i) {
            if (alternate[i] != result[i].front()) result[i].push_back(std::move(alternate[i]));
        }
    }
    return result;
}

std::shared_ptr<CVContext> get_cv_context(int32_t handle) {
    if (handle <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_cv_context_mutex);
    const auto it = g_cv_contexts.find(handle);
    return it == g_cv_contexts.end() ? nullptr : it->second;
}

std::shared_ptr<const TemplateEntry> find_template_candidates(
    const std::shared_ptr<CVContext>& context,
    const std::vector<std::string>& candidates,
    std::string* resolved_name) {
    if (!context) return nullptr;
    std::shared_lock<std::shared_mutex> lock(context->template_mutex);
    for (const std::string& candidate : candidates) {
        auto it = context->templates.find(candidate);
        if (it != context->templates.end()) {
            if (resolved_name != nullptr) *resolved_name = it->second->key;
            return it->second;
        }
        try {
            const std::string filename = std::filesystem::u8path(candidate).filename().u8string();
            it = context->templates.find(filename);
            if (it != context->templates.end()) {
                if (resolved_name != nullptr) *resolved_name = it->second->key;
                return it->second;
            }
        } catch (const std::bad_alloc&) {
            throw;
        } catch (...) {
        }
    }
    return nullptr;
}

std::shared_ptr<const TemplateEntry> find_template(
    const std::shared_ptr<CVContext>& context,
    const std::string& name,
    std::string* resolved_name = nullptr) {
    const std::vector<std::string> candidates = compat_text_candidates(name.c_str());
    return find_template_candidates(context, candidates, resolved_name);
}

void to_cv_result(const AIImageMatch& src, CVMatchResult* dst) {
    if (dst == nullptr) {
        return;
    }
    dst->x = src.box.x;
    dst->y = src.box.y;
    dst->w = src.box.w;
    dst->h = src.box.h;
    dst->sim = src.score;
    dst->template_index = src.template_index;
}

bool parse_rgb_hex(const char* text, uint32_t* rgb) {
    if (text == nullptr || rgb == nullptr) return false;
    if (text[0] == '#') {
        ++text;
    } else if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text += 2;
    }
    if (std::strlen(text) != 6) return false;
    uint32_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        const char ch = *p;
        uint32_t digit = 0;
        if (ch >= '0' && ch <= '9') digit = static_cast<uint32_t>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') digit = static_cast<uint32_t>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') digit = static_cast<uint32_t>(ch - 'A' + 10);
        else return false;
        value = (value << 4) | digit;
    }
    *rgb = value;
    return true;
}

constexpr float kCvInternalNmsIou = 0.30f;

bool valid_compat_color_bias(const char* color_bias, int32_t match_mode) {
    if (match_mode != 0 && match_mode != 1) return false;
    if (color_bias == nullptr || color_bias[0] == '\0') return true;
    const size_t len = std::strlen(color_bias);
    if ((match_mode == 1 && len != 2) || (match_mode == 0 && len != 6)) return false;
    for (const char* p = color_bias; *p != '\0'; ++p) {
        if (!std::isxdigit(static_cast<unsigned char>(*p))) return false;
    }
    return true;
}

int32_t compat_color_tolerance(const char* color_bias, int32_t match_mode) {
    if (color_bias == nullptr || color_bias[0] == '\0') return 0;
    const auto hex_byte = [](const char* p) {
        const auto hex = [](char ch) {
            if (ch >= '0' && ch <= '9') return ch - '0';
            if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
            return ch - 'A' + 10;
        };
        return hex(p[0]) * 16 + hex(p[1]);
    };
    if (match_mode == 1) return hex_byte(color_bias);
    return std::max({hex_byte(color_bias), hex_byte(color_bias + 2), hex_byte(color_bias + 4)});
}

std::string format_cv_multi_text(const std::vector<CVMatchResult>& matches) {
    std::vector<ai::CompactPointResult> results;
    results.reserve(matches.size());
    for (const CVMatchResult& match : matches) {
        results.push_back(ai::CompactPointResult{match.template_index, match.x, match.y});
    }
    return ai::format_compact_points(results);
}

std::string format_ai_cv_json(const std::vector<AIImageMatch>& matches) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << '[';
    for (size_t i = 0; i < matches.size(); ++i) {
        const AIImageMatch& match = matches[i];
        if (i > 0) oss << ',';
        oss << "{\"template_index\":" << match.template_index
            << ",\"x\":" << match.box.x
            << ",\"y\":" << match.box.y
            << ",\"w\":" << match.box.w
            << ",\"h\":" << match.box.h
            << ",\"score\":" << (std::isfinite(match.score) ? match.score : 0.0f)
            << '}';
    }
    oss << ']';
    return oss.str();
}

struct CVColorBias {
    bool enabled = false;
    int32_t b = 0;
    int32_t g = 0;
    int32_t r = 0;
    int32_t gray = 0;
};

int32_t hex_byte_value(const char* text) {
    const auto digit = [](char ch) {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        return ch - 'A' + 10;
    };
    return digit(text[0]) * 16 + digit(text[1]);
}

bool parse_cv_color_bias(const char* text, int32_t match_mode, CVColorBias* output) {
    if (output == nullptr || !valid_compat_color_bias(text, match_mode)) return false;
    *output = CVColorBias{};
    if (text == nullptr || text[0] == '\0') return true;
    output->enabled = true;
    if (match_mode == 1) {
        output->gray = hex_byte_value(text);
    } else {
        output->r = hex_byte_value(text);
        output->g = hex_byte_value(text + 2);
        output->b = hex_byte_value(text + 4);
    }
    return true;
}

float cv_match_iou(const CVMatchResult& lhs, const CVMatchResult& rhs) {
    const int32_t left = std::max(lhs.x, rhs.x);
    const int32_t top = std::max(lhs.y, rhs.y);
    const int32_t right = std::min(lhs.x + lhs.w, rhs.x + rhs.w);
    const int32_t bottom = std::min(lhs.y + lhs.h, rhs.y + rhs.h);
    const int64_t overlap = static_cast<int64_t>(std::max(0, right - left)) * std::max(0, bottom - top);
    const int64_t total = static_cast<int64_t>(lhs.w) * lhs.h + static_cast<int64_t>(rhs.w) * rhs.h - overlap;
    return total > 0 ? static_cast<float>(static_cast<double>(overlap) / static_cast<double>(total)) : 0.0f;
}

bool cv_result_screen_order(const CVMatchResult& lhs, const CVMatchResult& rhs) {
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.template_index != rhs.template_index) return lhs.template_index < rhs.template_index;
    return lhs.sim > rhs.sim;
}

#if defined(AIENGINE_WITH_OPENCV)

struct CVTilePeak {
    float score = -std::numeric_limits<float>::infinity();
    uint32_t index = UINT32_MAX;
};

struct CVVisiblePixel {
    uint32_t offset;
    uint8_t value[3];
};

struct CVTemplateThreadScratch {
    cv::Mat score;
    cv::Mat local_max;
    cv::Mat peak_kernel;
    std::vector<cv::Point> peak_points;
    std::vector<CVMatchResult> candidates;
    std::vector<CVMatchResult> selected;
    std::vector<CVTilePeak> tile_peaks;
    std::vector<CVTilePeak> tile_subpeaks;
    std::vector<uint32_t> tile_heap;
    std::vector<uint8_t> tile_suppressed;
    std::vector<CVVisiblePixel> visible_pixels;
    std::vector<uint32_t> visible_offsets;
    std::vector<uint32_t> visible_bgr;
};

// One heap node represents each 32x16 tile. Four 16x8 subpeaks let a stale
// node repair only the suppressed local regions when it reaches the root.
// A stale score is an upper bound, so the first valid root is the exact next
// global maximum; equal scores retain the original row-major order.
class CVTileMaxHeap {
public:
    static constexpr int kWidth = 32;
    static constexpr int kHeight = 16;

    CVTileMaxHeap(const cv::Mat& score, CVTemplateThreadScratch& scratch)
        : score_(score), scratch_(scratch),
          tile_columns_((score.cols + kWidth - 1) / kWidth) {
#if defined(AIENGINE_CV_AVX2)
        use_avx2_ = cv::checkHardwareSupport(CV_CPU_AVX2)
#if defined(AIENGINE_CV_TEST_HOOKS)
            && !g_cv_force_scalar_masked
#endif
            ;
#endif
#if defined(AIENGINE_CV_TEST_HOOKS)
        const auto started = std::chrono::steady_clock::now();
#endif
        if (score.total() > UINT32_MAX) {
            throw std::length_error("CV score matrix exceeds tile index range");
        }
        const size_t tile_count = static_cast<size_t>(tile_columns_) *
            static_cast<size_t>((score.rows + kHeight - 1) / kHeight);
        scratch_.tile_peaks.resize(tile_count);
        scratch_.tile_subpeaks.resize(tile_count * 4);
        scratch_.tile_heap.resize(tile_count);
        scratch_.tile_suppressed.resize(score.total());
        std::fill(scratch_.tile_suppressed.begin(), scratch_.tile_suppressed.end(), 0);
        for (uint32_t tile = 0; tile < tile_count; ++tile) {
            for (uint32_t sub = 0; sub < 4; ++sub) refresh_subtile(tile, sub);
            refresh_parent(tile);
            scratch_.tile_heap[tile] = tile;
        }
        for (size_t position = tile_count / 2; position > 0; --position) {
            sift_down(position - 1);
        }
#if defined(AIENGINE_CV_TEST_HOOKS)
        g_cv_transparent_profile.heap_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
#endif
    }

    std::pair<float, cv::Point> top() {
#if defined(AIENGINE_CV_TEST_HOOKS)
        const auto started = std::chrono::steady_clock::now();
#endif
        while (true) {
            const CVTilePeak& root = scratch_.tile_peaks[scratch_.tile_heap.front()];
            if (root.score < 0.0f || !scratch_.tile_suppressed[root.index]) break;
            const uint32_t tile = scratch_.tile_heap.front();
            for (uint32_t sub = 0; sub < 4; ++sub) {
                const CVTilePeak& subpeak = scratch_.tile_subpeaks[tile * 4 + sub];
                if (subpeak.score >= 0.0f && scratch_.tile_suppressed[subpeak.index]) {
                    refresh_subtile(tile, sub);
                }
            }
            refresh_parent(tile);
            sift_down(0);
        }
        const CVTilePeak& peak = scratch_.tile_peaks[scratch_.tile_heap.front()];
#if defined(AIENGINE_CV_TEST_HOOKS)
        g_cv_transparent_profile.heap_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
#endif
        if (peak.index == UINT32_MAX) return {peak.score, cv::Point{0, 0}};
        return {peak.score, cv::Point{
            static_cast<int>(peak.index % static_cast<uint32_t>(score_.cols)),
            static_cast<int>(peak.index / static_cast<uint32_t>(score_.cols))}};
    }

    void suppressed(const cv::Rect& rectangle) {
#if defined(AIENGINE_CV_TEST_HOOKS)
        const auto started = std::chrono::steady_clock::now();
#endif
        for (int y = rectangle.y; y < rectangle.y + rectangle.height; ++y) {
            const size_t row = static_cast<size_t>(y) * score_.cols;
            std::fill(scratch_.tile_suppressed.begin() + row + rectangle.x,
                scratch_.tile_suppressed.begin() + row + rectangle.x + rectangle.width, 1);
        }
#if defined(AIENGINE_CV_TEST_HOOKS)
        g_cv_transparent_profile.heap_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
#endif
    }

private:
    bool better(uint32_t left, uint32_t right) const {
        const CVTilePeak& lhs = scratch_.tile_peaks[left];
        const CVTilePeak& rhs = scratch_.tile_peaks[right];
        return lhs.score != rhs.score ? lhs.score > rhs.score : lhs.index < rhs.index;
    }

    void sift_down(size_t position) {
        const size_t count = scratch_.tile_heap.size();
        const uint32_t moving = scratch_.tile_heap[position];
        while (position < count / 2) {
            const size_t child = position * 2 + 1;
            size_t best = child;
            if (child + 1 < count &&
                better(scratch_.tile_heap[child + 1], scratch_.tile_heap[child])) {
                best = child + 1;
            }
            if (!better(scratch_.tile_heap[best], moving)) break;
            scratch_.tile_heap[position] = scratch_.tile_heap[best];
            position = best;
        }
        scratch_.tile_heap[position] = moving;
    }

    void refresh_parent(uint32_t tile) {
        CVTilePeak best;
        for (uint32_t sub = 0; sub < 4; ++sub) {
            const CVTilePeak& candidate = scratch_.tile_subpeaks[tile * 4 + sub];
            if (candidate.score > best.score ||
                (candidate.score == best.score && candidate.index < best.index)) best = candidate;
        }
        scratch_.tile_peaks[tile] = best;
    }

    void refresh_subtile(uint32_t tile, uint32_t sub) {
#if defined(AIENGINE_CV_TEST_HOOKS)
        ++g_cv_transparent_profile.tile_refreshes;
#endif
        const int start_x = static_cast<int>(tile % tile_columns_) * kWidth +
            static_cast<int>(sub % 2) * (kWidth / 2);
        const int start_y = static_cast<int>(tile / tile_columns_) * kHeight +
            static_cast<int>(sub / 2) * (kHeight / 2);
        const int end_x = std::min(start_x + kWidth / 2, score_.cols);
        const int end_y = std::min(start_y + kHeight / 2, score_.rows);
#if defined(AIENGINE_CV_AVX2)
        if (use_avx2_) {
            const auto peak = ai::cv_detail::tile_peak_avx2(
                score_.ptr<float>(), score_.step1(),
                scratch_.tile_suppressed.data(), score_.cols,
                start_x, start_y, end_x, end_y);
            scratch_.tile_subpeaks[tile * 4 + sub] = {peak.score, peak.index};
            return;
        }
#endif
        CVTilePeak best{-1.0f, static_cast<uint32_t>(
            static_cast<size_t>(start_y) * score_.cols + start_x)};
        for (int y = start_y; y < end_y; ++y) {
            const float* row = score_.ptr<float>(y);
            for (int x = start_x; x < end_x; ++x) {
                const uint32_t index = static_cast<uint32_t>(
                    static_cast<size_t>(y) * score_.cols + x);
                const float value = row[x];
                if (!scratch_.tile_suppressed[index] && value > best.score) best = {value, index};
            }
        }
        scratch_.tile_subpeaks[tile * 4 + sub] = best;
    }

    const cv::Mat& score_;
    CVTemplateThreadScratch& scratch_;
    int tile_columns_;
#if defined(AIENGINE_CV_AVX2)
    bool use_avx2_ = false;
#endif
};

struct CVWorkspace {
    OwnedImage color;
    OwnedImage gray;
    CVTemplateThreadScratch scratch;
    ai::cv_detail::CorrelationWorkspace correlation;
    bool correlation_image_valid = false;
    bool correlation_sums_valid = false;

    size_t retained_bytes() const noexcept {
        const auto mat_bytes = [](const cv::Mat& mat) -> size_t {
            return mat.data ? static_cast<size_t>(mat.datalimit - mat.datastart) : 0;
        };
        return correlation.retained_bytes() + color.pixels.capacity() + gray.pixels.capacity() +
            mat_bytes(scratch.score) + mat_bytes(scratch.local_max) + mat_bytes(scratch.peak_kernel) +
            scratch.peak_points.capacity() * sizeof(cv::Point) +
            (scratch.candidates.capacity() + scratch.selected.capacity()) * sizeof(CVMatchResult) +
            scratch.tile_peaks.capacity() * sizeof(CVTilePeak) +
            scratch.tile_subpeaks.capacity() * sizeof(CVTilePeak) +
            scratch.tile_heap.capacity() * sizeof(uint32_t) +
            scratch.tile_suppressed.capacity() * sizeof(uint8_t) +
            scratch.visible_pixels.capacity() * sizeof(CVVisiblePixel) +
            (scratch.visible_offsets.capacity() + scratch.visible_bgr.capacity()) * sizeof(uint32_t);
    }
    void discard() noexcept {
        correlation_image_valid = correlation_sums_valid = false;
        correlation.discard();
        std::vector<uint8_t>().swap(color.pixels);
        std::vector<uint8_t>().swap(gray.pixels);
        color.view = {}; gray.view = {};
        scratch.score.release(); scratch.local_max.release(); scratch.peak_kernel.release();
        std::vector<cv::Point>().swap(scratch.peak_points);
        std::vector<CVMatchResult>().swap(scratch.candidates);
        std::vector<CVMatchResult>().swap(scratch.selected);
        std::vector<CVTilePeak>().swap(scratch.tile_peaks);
        std::vector<CVTilePeak>().swap(scratch.tile_subpeaks);
        std::vector<uint32_t>().swap(scratch.tile_heap);
        std::vector<uint8_t>().swap(scratch.tile_suppressed);
        std::vector<CVVisiblePixel>().swap(scratch.visible_pixels);
        std::vector<uint32_t>().swap(scratch.visible_offsets);
        std::vector<uint32_t>().swap(scratch.visible_bgr);
    }
};

constexpr size_t kCvWorkspaceCount = sizeof(void*) == 4 ? 2 : 4;
constexpr size_t kCvIdleBudget = (sizeof(void*) == 4 ? 64u : 256u) * 1024u * 1024u;
struct CVWorkspacePool {
    std::mutex mutex;
    std::condition_variable available;
    std::array<CVWorkspace, kCvWorkspaceCount> workspaces;
    std::array<bool, kCvWorkspaceCount> busy{};
    size_t idle_bytes = 0;
    size_t active = 0;
    size_t peak_active = 0;
    size_t discarded = 0;
};
CVWorkspacePool g_cv_pool;

class CVWorkspaceLease {
public:
    CVWorkspaceLease() {
        std::unique_lock<std::mutex> lock(g_cv_pool.mutex);
        g_cv_pool.available.wait(lock, [] {
            return g_cv_pool.active < kCvWorkspaceCount;
        });
        for (size_t i = 0; i < kCvWorkspaceCount; ++i) {
            if (!g_cv_pool.busy[i]) { index_ = i; break; }
        }
        g_cv_pool.busy[index_] = true;
        g_cv_pool.idle_bytes -= get().retained_bytes();
        ++g_cv_pool.active;
        g_cv_pool.peak_active = std::max(g_cv_pool.peak_active, g_cv_pool.active);
    }
    CVWorkspaceLease(const CVWorkspaceLease&) = delete;
    CVWorkspaceLease& operator=(const CVWorkspaceLease&) = delete;
    ~CVWorkspaceLease() noexcept {
        {
            std::lock_guard<std::mutex> lock(g_cv_pool.mutex);
            if (std::uncaught_exceptions() > exceptions_ ||
                get().retained_bytes() > kCvIdleBudget - g_cv_pool.idle_bytes) {
                get().discard();
                ++g_cv_pool.discarded;
            }
            g_cv_pool.idle_bytes += get().retained_bytes();
            g_cv_pool.busy[index_] = false;
            --g_cv_pool.active;
        }
        g_cv_pool.available.notify_one();
    }
    CVWorkspace& get() noexcept { return g_cv_pool.workspaces[index_]; }
private:
    size_t index_ = 0;
    int exceptions_ = std::uncaught_exceptions();
};

std::once_flag g_cv_runtime_init_once;

void ensure_cv_runtime_initialized() {
    std::call_once(g_cv_runtime_init_once, [] {
        cv::setUseOptimized(true);
        // A CV context is designed for caller-managed concurrency. Keeping each
        // small template match single-threaded avoids nested scheduling jitter.
        cv::setNumThreads(1);
    });
}

bool refresh_bgr_cache(const AIImage& source, OwnedImage* output) {
    const size_t bytes = checked_image_bytes(source.width, source.height, 3);
    const bool reusable_bgr = source.format == AI_IMAGE_BGR24 &&
        output->view.width == source.width && output->view.height == source.height &&
        output->view.format == AI_IMAGE_BGR24 && output->pixels.size() == bytes;
    if (reusable_bgr) {
        bool identical = true;
        for (int32_t y = 0; y < source.height; ++y) {
            const uint8_t* src = ai::image_row_ptr(source, y);
            const uint8_t* cached = output->pixels.data() + static_cast<size_t>(y) * source.width * 3;
            if (std::memcmp(src, cached, static_cast<size_t>(source.width) * 3) != 0) {
                identical = false;
                break;
            }
        }
        if (identical) return false;
    }
    output->pixels.resize(bytes);
    output->view = AIImage{output->pixels.data(), source.width, source.height, source.width * 3, AI_IMAGE_BGR24};
    const int channels = ai::channels_for_format(source.format);
    for (int32_t y = 0; y < source.height; ++y) {
        const uint8_t* src = ai::image_row_ptr(source, y);
        uint8_t* dst = output->pixels.data() + static_cast<size_t>(y) * source.width * 3;
        if (source.format == AI_IMAGE_BGR24) {
            std::memcpy(dst, src, static_cast<size_t>(source.width) * 3);
            continue;
        }
        for (int32_t x = 0; x < source.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            ai::read_bgr(src + x * channels, source.format, &b, &g, &r);
            dst[x * 3] = b;
            dst[x * 3 + 1] = g;
            dst[x * 3 + 2] = r;
        }
    }
    return true;
}

void make_gray_reuse(const OwnedImage& source, OwnedImage* output) {
    const size_t pixels = checked_image_bytes(source.view.width, source.view.height, 1);
    output->pixels.resize(pixels);
    output->view = AIImage{output->pixels.data(), source.view.width, source.view.height, source.view.width, AI_IMAGE_GRAY8};
    cv::Mat color(source.view.height, source.view.width, CV_8UC3,
        const_cast<uint8_t*>(source.pixels.data()), static_cast<size_t>(source.view.stride));
    cv::Mat gray(source.view.height, source.view.width, CV_8UC1, output->pixels.data(), output->view.stride);
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
}

cv::Mat owned_image_mat(const OwnedImage& image) {
    const int type = image.view.format == AI_IMAGE_GRAY8 ? CV_8UC1 : CV_8UC3;
    return cv::Mat(image.view.height, image.view.width, type,
        const_cast<uint8_t*>(image.pixels.data()), static_cast<size_t>(image.view.stride));
}

bool make_transparent_mask(const OwnedImage& color, int32_t match_mode, uint32_t transparent_rgb, cv::Mat* mask) {
    if (mask == nullptr) return false;
    const int channels = match_mode == 1 ? 1 : 3;
    mask->create(color.view.height, color.view.width, CV_MAKETYPE(CV_8U, channels));
    const uint8_t transparent_r = static_cast<uint8_t>((transparent_rgb >> 16) & 0xffu);
    const uint8_t transparent_g = static_cast<uint8_t>((transparent_rgb >> 8) & 0xffu);
    const uint8_t transparent_b = static_cast<uint8_t>(transparent_rgb & 0xffu);
    int32_t visible = 0;
    for (int32_t y = 0; y < color.view.height; ++y) {
        const uint8_t* src = color.pixels.data() + static_cast<size_t>(y) * color.view.stride;
        uint8_t* dst = mask->ptr<uint8_t>(y);
        for (int32_t x = 0; x < color.view.width; ++x) {
            const bool is_transparent = src[x * 3] == transparent_b &&
                src[x * 3 + 1] == transparent_g && src[x * 3 + 2] == transparent_r;
            const uint8_t value = is_transparent ? 0 : 255;
            for (int32_t c = 0; c < channels; ++c) dst[x * channels + c] = value;
            if (!is_transparent) ++visible;
        }
    }
    return visible > 0;
}

std::shared_ptr<const cv::Mat> get_transparent_mask(
    const std::shared_ptr<const TemplateEntry>& entry,
    int32_t match_mode,
    uint32_t transparent_rgb) {
    const uint64_t key = (static_cast<uint64_t>(transparent_rgb) << 1) |
        static_cast<uint64_t>(match_mode == 1 ? 1 : 0);
    {
        std::lock_guard<std::mutex> lock(entry->mask_mutex);
        const auto found = entry->transparent_masks.find(key);
        if (found != entry->transparent_masks.end()) return found->second;
    }
    auto created = std::make_shared<cv::Mat>();
    if (!make_transparent_mask(entry->color, match_mode, transparent_rgb, created.get())) return nullptr;
    std::lock_guard<std::mutex> lock(entry->mask_mutex);
    const auto inserted = entry->transparent_masks.emplace(key, created);
    return inserted.first->second;
}

bool template_contains_transparent_color(
    const std::shared_ptr<const TemplateEntry>& entry,
    uint32_t transparent_rgb) {
    {
        std::lock_guard<std::mutex> lock(entry->mask_mutex);
        const auto found = entry->transparent_color_presence.find(transparent_rgb);
        if (found != entry->transparent_color_presence.end()) return found->second;
    }
    const uint8_t transparent_r = static_cast<uint8_t>((transparent_rgb >> 16) & 0xffu);
    const uint8_t transparent_g = static_cast<uint8_t>((transparent_rgb >> 8) & 0xffu);
    const uint8_t transparent_b = static_cast<uint8_t>(transparent_rgb & 0xffu);
    bool present = false;
    for (int32_t y = 0; y < entry->color.view.height && !present; ++y) {
        const uint8_t* row = entry->color.pixels.data() + static_cast<size_t>(y) * entry->color.view.stride;
        for (int32_t x = 0; x < entry->color.view.width; ++x) {
            if (row[x * 3] == transparent_b && row[x * 3 + 1] == transparent_g && row[x * 3 + 2] == transparent_r) {
                present = true;
                break;
            }
        }
    }
    std::lock_guard<std::mutex> lock(entry->mask_mutex);
    const auto inserted = entry->transparent_color_presence.emplace(transparent_rgb, present);
    return inserted.first->second;
}

std::shared_ptr<const cv::Mat> get_transparent_candidate_template(
    const std::shared_ptr<const TemplateEntry>& entry,
    uint32_t transparent_rgb) {
    {
        std::lock_guard<std::mutex> lock(entry->mask_mutex);
        const auto found = entry->transparent_candidate_templates.find(transparent_rgb);
        if (found != entry->transparent_candidate_templates.end()) return found->second;
    }
    const auto mask = get_transparent_mask(entry, 1, transparent_rgb);
    if (!mask) return nullptr;
    auto created = std::make_shared<cv::Mat>(owned_image_mat(entry->gray).clone());
    const cv::Scalar visible_mean = cv::mean(*created, *mask);
    created->setTo(visible_mean, *mask == 0);
    std::lock_guard<std::mutex> lock(entry->mask_mutex);
    const auto inserted = entry->transparent_candidate_templates.emplace(transparent_rgb, created);
    return inserted.first->second;
}

bool passes_color_bias(
    const cv::Mat& image,
    const cv::Mat& templ,
    const cv::Mat* mask,
    int32_t origin_x,
    int32_t origin_y,
    const CVColorBias& bias,
    float min_score) {
    if (!bias.enabled) return true;
    uint64_t total_diff = 0;
    int64_t compared_values = 0;
    const int channels = image.channels();
    for (int32_t y = 0; y < templ.rows; ++y) {
        const uint8_t* image_row = image.ptr<uint8_t>(origin_y + y) + static_cast<size_t>(origin_x) * channels;
        const uint8_t* templ_row = templ.ptr<uint8_t>(y);
        const uint8_t* mask_row = mask == nullptr ? nullptr : mask->ptr<uint8_t>(y);
        for (int32_t x = 0; x < templ.cols; ++x) {
            if (mask_row != nullptr && mask_row[x * mask->channels()] == 0) continue;
            if (channels == 1) {
                total_diff += static_cast<uint64_t>(std::max(0,
                    std::abs(static_cast<int>(image_row[x]) - static_cast<int>(templ_row[x])) - bias.gray));
                ++compared_values;
            } else {
                const int32_t tolerance[3] = {bias.b, bias.g, bias.r};
                for (int32_t c = 0; c < 3; ++c) {
                    total_diff += static_cast<uint64_t>(std::max(0,
                        std::abs(static_cast<int>(image_row[x * 3 + c]) - static_cast<int>(templ_row[x * 3 + c])) - tolerance[c]));
                    ++compared_values;
                }
            }
        }
    }
    if (compared_values <= 0) return false;
    const double tolerance_score = 1.0 - static_cast<double>(total_diff) /
        (static_cast<double>(compared_values) * 255.0);
    return tolerance_score >= min_score;
}

float exact_ccoeff_score_at(
    const cv::Mat& image,
    const cv::Mat& templ,
    const TemplateCcoeffStats& stats,
    int32_t origin_x,
    int32_t origin_y) {
    const int32_t channels = image.channels();
    const int64_t pixel_count = static_cast<int64_t>(templ.cols) * templ.rows;
    if (channels != templ.channels() || channels != stats.channels || pixel_count <= 0 || stats.energy <= 1e-12) {
        return -1.0f;
    }
    bool identical = true;
    const size_t row_bytes = static_cast<size_t>(templ.cols) * channels;
    for (int32_t y = 0; y < templ.rows; ++y) {
        const uint8_t* image_row = image.ptr<uint8_t>(origin_y + y) +
            static_cast<size_t>(origin_x) * channels;
        if (std::memcmp(image_row, templ.ptr<uint8_t>(y), row_bytes) != 0) {
            identical = false;
            break;
        }
    }
    if (identical) return 1.0f;
    uint64_t image_sums[3]{};
    uint64_t image_square_sums[3]{};
    uint64_t dot_products[3]{};
    for (int32_t y = 0; y < templ.rows; ++y) {
        const uint8_t* image_row = image.ptr<uint8_t>(origin_y + y) +
            static_cast<size_t>(origin_x) * channels;
        const uint8_t* templ_row = templ.ptr<uint8_t>(y);
        for (int32_t x = 0; x < templ.cols; ++x) {
            for (int32_t c = 0; c < channels; ++c) {
                const uint32_t image_value = image_row[x * channels + c];
                const uint32_t template_value = templ_row[x * channels + c];
                image_sums[c] += image_value;
                image_square_sums[c] += image_value * image_value;
                dot_products[c] += image_value * template_value;
            }
        }
    }
    double numerator = 0.0;
    double image_energy = 0.0;
    for (int32_t c = 0; c < channels; ++c) {
        numerator += static_cast<double>(dot_products[c]) -
            static_cast<double>(image_sums[c]) * stats.sums[c] / static_cast<double>(pixel_count);
        image_energy += static_cast<double>(image_square_sums[c]) -
            static_cast<double>(image_sums[c]) * image_sums[c] / static_cast<double>(pixel_count);
    }
    if (image_energy <= 1e-12) return -1.0f;
    const double score = numerator / std::sqrt(stats.energy * image_energy);
    if (!std::isfinite(score)) return -1.0f;
    if (score > 1.0 - 1e-6) return 1.0f;
    return static_cast<float>(std::max(-1.0, std::min(1.0, score)));
}

float exact_masked_ccoeff_score_at(
    const cv::Mat& image,
    const cv::Mat& templ,
    const cv::Mat& mask,
    int32_t origin_x,
    int32_t origin_y) {
    const int32_t channels = image.channels();
    if (channels != templ.channels() || mask.channels() != channels) return -1.0f;
    uint64_t image_sums[3]{};
    uint64_t template_sums[3]{};
    uint64_t image_square_sums[3]{};
    uint64_t template_square_sums[3]{};
    uint64_t dot_products[3]{};
    int64_t visible_pixels = 0;
    bool identical = true;
    for (int32_t y = 0; y < templ.rows; ++y) {
        const uint8_t* image_row = image.ptr<uint8_t>(origin_y + y) +
            static_cast<size_t>(origin_x) * channels;
        const uint8_t* templ_row = templ.ptr<uint8_t>(y);
        const uint8_t* mask_row = mask.ptr<uint8_t>(y);
        for (int32_t x = 0; x < templ.cols; ++x) {
            if (mask_row[x * channels] == 0) continue;
            ++visible_pixels;
            for (int32_t c = 0; c < channels; ++c) {
                const uint32_t image_value = image_row[x * channels + c];
                const uint32_t template_value = templ_row[x * channels + c];
                identical = identical && image_value == template_value;
                image_sums[c] += image_value;
                template_sums[c] += template_value;
                image_square_sums[c] += image_value * image_value;
                template_square_sums[c] += template_value * template_value;
                dot_products[c] += image_value * template_value;
            }
        }
    }
    if (visible_pixels <= 0) return -1.0f;
    if (identical) return 1.0f;
    double numerator = 0.0;
    double image_energy = 0.0;
    double template_energy = 0.0;
    for (int32_t c = 0; c < channels; ++c) {
        numerator += static_cast<double>(dot_products[c]) -
            static_cast<double>(image_sums[c]) * template_sums[c] / static_cast<double>(visible_pixels);
        image_energy += static_cast<double>(image_square_sums[c]) -
            static_cast<double>(image_sums[c]) * image_sums[c] / static_cast<double>(visible_pixels);
        template_energy += static_cast<double>(template_square_sums[c]) -
            static_cast<double>(template_sums[c]) * template_sums[c] / static_cast<double>(visible_pixels);
    }
    if (image_energy <= 1e-12 || template_energy <= 1e-12) return -1.0f;
    const double score = numerator / std::sqrt(image_energy * template_energy);
    if (!std::isfinite(score)) return -1.0f;
    if (score > 1.0 - 1e-6) return 1.0f;
    return static_cast<float>(std::max(-1.0, std::min(1.0, score)));
}

struct CVMaskedTemplateStats {
    uint64_t sums[3]{};
    uint64_t square_sums[3]{};
    double energy = 0.0;
};

CVMaskedTemplateStats prepare_masked_score(
    const cv::Mat& templ,
    const cv::Mat& mask,
    size_t image_stride,
    std::vector<CVVisiblePixel>* visible) {
    visible->clear();
    CVMaskedTemplateStats stats;
    const int channels = templ.channels();
    for (int y = 0; y < templ.rows; ++y) {
        const uint8_t* templ_row = templ.ptr<uint8_t>(y);
        const uint8_t* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < templ.cols; ++x) {
            if (mask_row[x * channels] == 0) continue;
            const size_t offset = static_cast<size_t>(y) * image_stride +
                static_cast<size_t>(x) * channels;
            if (offset > UINT32_MAX) throw std::length_error("CV image exceeds visible pixel offset range");
            CVVisiblePixel pixel{static_cast<uint32_t>(offset), {0, 0, 0}};
            for (int c = 0; c < channels; ++c) {
                const uint32_t value = templ_row[x * channels + c];
                pixel.value[c] = static_cast<uint8_t>(value);
                stats.sums[c] += value;
                stats.square_sums[c] += value * value;
            }
            visible->push_back(pixel);
        }
    }
    const double count = static_cast<double>(visible->size());
    if (count > 0.0) {
        for (int c = 0; c < channels; ++c) {
            stats.energy += static_cast<double>(stats.square_sums[c]) -
                static_cast<double>(stats.sums[c]) * stats.sums[c] / count;
        }
    }
    return stats;
}

float exact_masked_ccoeff_score_at_prepared(
    const cv::Mat& image,
    const std::vector<CVVisiblePixel>& visible,
    const CVMaskedTemplateStats& stats,
    int32_t origin_x,
    int32_t origin_y,
    bool use_avx2,
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& template_bgr) {
    if (visible.empty()) return -1.0f;
    const int channels = image.channels();
    uint64_t image_sums[3]{};
    uint64_t image_square_sums[3]{};
    uint64_t dot_products[3]{};
    bool identical = true;
    const uint8_t* image_origin = image.ptr<uint8_t>(origin_y) +
        static_cast<size_t>(origin_x) * channels;
#if defined(AIENGINE_CV_AVX2)
    if (use_avx2) {
        const auto sums = ai::cv_detail::masked_accumulate_avx2(
            image_origin, offsets.data(), template_bgr.data(), visible.size());
        for (int c = 0; c < 3; ++c) {
            image_sums[c] = sums.sums[c];
            image_square_sums[c] = sums.squares[c];
            dot_products[c] = sums.dots[c];
        }
        identical = sums.identical;
    } else
#else
    (void)use_avx2; (void)offsets; (void)template_bgr;
#endif
    if (channels == 3 && visible.size() <= 65535) {
        uint32_t sum_b = 0, sum_g = 0, sum_r = 0;
        uint32_t square_b = 0, square_g = 0, square_r = 0;
        uint32_t dot_b = 0, dot_g = 0, dot_r = 0;
        uint32_t difference = 0;
        for (const CVVisiblePixel& pixel : visible) {
            const uint8_t* value = image_origin + pixel.offset;
            const uint32_t b = value[0], g = value[1], r = value[2];
            difference |= (b ^ pixel.value[0]) | (g ^ pixel.value[1]) |
                (r ^ pixel.value[2]);
            sum_b += b; sum_g += g; sum_r += r;
            square_b += b * b; square_g += g * g; square_r += r * r;
            dot_b += b * pixel.value[0];
            dot_g += g * pixel.value[1];
            dot_r += r * pixel.value[2];
        }
        identical = difference == 0;
        image_sums[0] = sum_b; image_sums[1] = sum_g; image_sums[2] = sum_r;
        image_square_sums[0] = square_b;
        image_square_sums[1] = square_g;
        image_square_sums[2] = square_r;
        dot_products[0] = dot_b; dot_products[1] = dot_g; dot_products[2] = dot_r;
    } else {
        for (const CVVisiblePixel& pixel : visible) {
            const uint8_t* image_pixel = image_origin + pixel.offset;
            for (int c = 0; c < channels; ++c) {
                const uint32_t value = image_pixel[c];
                const uint32_t template_value = pixel.value[c];
                identical &= value == template_value;
                image_sums[c] += value;
                image_square_sums[c] += value * value;
                dot_products[c] += value * template_value;
            }
        }
    }
    if (identical) return 1.0f;
    const double count = static_cast<double>(visible.size());
    double numerator = 0.0;
    double image_energy = 0.0;
    for (int c = 0; c < channels; ++c) {
        numerator += static_cast<double>(dot_products[c]) -
            static_cast<double>(image_sums[c]) * stats.sums[c] / count;
        image_energy += static_cast<double>(image_square_sums[c]) -
            static_cast<double>(image_sums[c]) * image_sums[c] / count;
    }
    if (image_energy <= 1e-12 || stats.energy <= 1e-12) return -1.0f;
    const double score = numerator / std::sqrt(image_energy * stats.energy);
    if (!std::isfinite(score)) return -1.0f;
    if (score > 1.0 - 1e-6) return 1.0f;
    return static_cast<float>(std::max(-1.0, std::min(1.0, score)));
}

void collect_local_peaks(
    const cv::Mat& score,
    int32_t template_width,
    int32_t template_height,
    float nms_iou,
    float candidate_floor,
    cv::Mat* local_max,
    cv::Mat* peak_kernel,
    std::vector<cv::Point>* points) {
    points->clear();
    const int32_t radius_x = std::max<int32_t>(1,
        static_cast<int32_t>(std::floor(template_width * nms_iou)));
    const int32_t radius_y = std::max<int32_t>(1,
        static_cast<int32_t>(std::floor(template_height * nms_iou)));
    const cv::Size kernel_size{radius_x * 2 + 1, radius_y * 2 + 1};
    // Sparse threshold crossings do not need a full-frame morphological pass.
    // This is the same rectangular maximum predicate (including its tolerance),
    // with an early rejection, not a cap on the number of candidates.
    for (int y=0;y<score.rows;++y) {
        const auto* row=score.ptr<float>(y);
        for (int x=0;x<score.cols;++x)
            if (row[x]>=candidate_floor && std::isfinite(row[x])) points->push_back({x,y});
    }
    if (points->size()<=score.total()/32
#if defined(AIENGINE_CV_TEST_HOOKS)
        && !g_cv_reference
#endif
    ) {
        size_t kept=0;
        for(const auto point:*points) {
            const float value=score.at<float>(point);
            if(point.x>0 && std::abs(value-score.at<float>(point.y,point.x-1))<=1e-7f) continue;
            if(point.y>0 && std::abs(value-score.at<float>(point.y-1,point.x))<=1e-7f) continue;
            bool peak=true;
            for(int y=std::max(0,point.y-radius_y);y<=std::min(score.rows-1,point.y+radius_y)&&peak;++y) {
                const auto* row=score.ptr<float>(y);
                for(int x=std::max(0,point.x-radius_x);x<=std::min(score.cols-1,point.x+radius_x);++x)
                    if(value+1e-6f<row[x]) { peak=false; break; }
            }
            if(peak) (*points)[kept++]=point;
        }
        points->resize(kept);
        return;
    }
    points->clear();
    if (peak_kernel->size() != kernel_size) {
        *peak_kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    }
    cv::dilate(score, *local_max, *peak_kernel);
    for (int32_t y = 0; y < score.rows; ++y) {
        const float* score_row = score.ptr<float>(y);
        const float* max_row = local_max->ptr<float>(y);
        const float* previous_row = y > 0 ? score.ptr<float>(y - 1) : nullptr;
        for (int32_t x = 0; x < score.cols; ++x) {
            const float value = score_row[x];
            if (!std::isfinite(value) || value < candidate_floor || value + 1e-6f < max_row[x]) continue;
            if (x > 0 && std::abs(value - score_row[x - 1]) <= 1e-7f) continue;
            if (previous_row != nullptr && std::abs(value - previous_row[x]) <= 1e-7f) continue;
            points->push_back(cv::Point{x, y});
        }
    }
}

int32_t find_cv_matches_opencv(
    const AIImage& big_image,
    const std::vector<std::shared_ptr<const TemplateEntry>>& entries,
    int32_t match_mode,
    bool transparent,
    uint32_t transparent_rgb,
    const CVColorBias& color_bias,
    float min_score,
    float nms_iou,
    bool best_only,
    std::vector<CVMatchResult>* output) {
    if (output == nullptr || entries.empty() || (match_mode != 0 && match_mode != 1) ||
        !std::isfinite(min_score) || min_score < 0.0f || min_score > 1.0f ||
        !std::isfinite(nms_iou) || nms_iou < 0.0f || nms_iou > 1.0f) {
        return AI_ERR_INVALID_ARGUMENT;
    }
    ensure_cv_runtime_initialized();
    output->clear();
    CVWorkspaceLease lease;
    auto& workspace = lease.get();
#if defined(AIENGINE_CV_TEST_HOOKS)
    std::fill(std::begin(g_cv_times), std::end(g_cv_times), 0);
    g_cv_transparent_profile = {};
    auto prof_start = std::chrono::steady_clock::now();
#endif
    cv_test_fault(1);
    if (refresh_bgr_cache(big_image, &workspace.color)) {
        workspace.correlation_image_valid = workspace.correlation_sums_valid = false;
    }
    bool gray_ready=false;
    const auto prepare_gray=[&] {
        if (!gray_ready) { make_gray_reuse(workspace.color,&workspace.gray); gray_ready=true; }
    };
    if (match_mode==1) prepare_gray();
    const cv::Mat image = owned_image_mat(match_mode == 1 ? workspace.gray : workspace.color);
    cv::Mat candidate_image;
    bool correlation_ready = false;
#if defined(AIENGINE_CV_TEST_HOOKS)
    bool correlation_prepared_now = false;
#endif
#if defined(AIENGINE_CV_TEST_HOOKS)
    g_cv_times[0] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prof_start).count();
#endif
    std::atomic<int32_t> match_error{AI_OK};
    const auto match_entry = [&](size_t template_index) {
        if (match_error.load(std::memory_order_relaxed) < 0) return;
        const auto& entry = entries[template_index];
        const cv::Mat templ = owned_image_mat(match_mode == 1 ? entry->gray : entry->color);
        cv::Mat candidate_templ = templ;
        if (templ.cols > image.cols || templ.rows > image.rows) return;
        CVTemplateThreadScratch& scratch = workspace.scratch;

        std::shared_ptr<const cv::Mat> mask_owner;
        std::shared_ptr<const cv::Mat> candidate_template_owner;
        const cv::Mat* mask = nullptr;
        const cv::Mat* candidate_mask = nullptr;
        bool use_transparent_mask = transparent && template_contains_transparent_color(entry, transparent_rgb);
        const bool optimized_transparent_one = use_transparent_mask && best_only
#if defined(AIENGINE_CV_TEST_HOOKS)
            && !g_cv_reference
#endif
            ;
        int method = best_only ? cv::TM_CCORR_NORMED : cv::TM_CCOEFF_NORMED;
        if (use_transparent_mask) {
            prepare_gray(); candidate_image=owned_image_mat(workspace.gray);
            mask_owner = get_transparent_mask(entry, match_mode, transparent_rgb);
            if (!mask_owner) {
                match_error.store(AI_ERR_INVALID_ARGUMENT, std::memory_order_relaxed);
                return;
            }
            mask = mask_owner.get();
            candidate_template_owner = get_transparent_candidate_template(entry, transparent_rgb);
            if (!candidate_template_owner) {
                match_error.store(AI_ERR_INVALID_ARGUMENT, std::memory_order_relaxed);
                return;
            }
            candidate_templ = *candidate_template_owner;
        }

        std::vector<CVMatchResult>& candidates = scratch.candidates;
        candidates.clear();
        const cv::Mat& candidate_source = use_transparent_mask ? candidate_image : image;
        cv_test_fault(2);
        if (!use_transparent_mask && match_mode == 0 && image.total() <= 512u * 1024u
#if defined(AIENGINE_CV_TEST_HOOKS)
            && !g_cv_reference
#endif
        ) {
            if (!correlation_ready) {
                // Exact byte equality of the owned frame permits reuse of its
                // transform/statistics. Correlation, peaks and results are still
                // recomputed for every request; no template owner is retained.
                if (!workspace.correlation_image_valid || (!best_only && !workspace.correlation_sums_valid)) {
                    workspace.correlation.prepare(image,!best_only);
                    workspace.correlation_image_valid = true;
                    workspace.correlation_sums_valid = !best_only;
#if defined(AIENGINE_CV_TEST_HOOKS)
                    correlation_prepared_now = true;
#endif
                }
                correlation_ready = true;
            }
            const auto geometry = ai::cv_detail::spectrum_geometry(image.size());
            std::shared_ptr<const ai::cv_detail::TemplateSpectrum> spectrum;
            {
                std::lock_guard<std::mutex> lock(entry->spectrum_mutex);
                if (!entry->spectrum || entry->spectrum->geometry != geometry) {
                    entry->spectrum = std::make_shared<const ai::cv_detail::TemplateSpectrum>(
                        ai::cv_detail::make_spectrum(templ, geometry));
                }
                spectrum = entry->spectrum;
            }
            workspace.correlation.match(templ, *spectrum, entry->color_stats.sums,
                entry->color_stats.energy, !best_only, scratch.score,
                best_only ? -1.0f : std::max(0.0f,min_score-2e-6f));
#if defined(AIENGINE_CV_TEST_HOOKS)
            g_cv_times[1]=correlation_prepared_now ? workspace.correlation.timings[0] : 0;
            g_cv_times[2]=correlation_prepared_now ? workspace.correlation.timings[1] : 0;
            g_cv_times[3]+=workspace.correlation.timings[2];
            g_cv_times[4]+=workspace.correlation.timings[3];
#endif
        } else if (candidate_mask == nullptr) {
#if defined(AIENGINE_CV_TEST_HOOKS)
            const auto rough_started = std::chrono::steady_clock::now();
#endif
            cv::matchTemplate(candidate_source, candidate_templ, scratch.score, method);
#if defined(AIENGINE_CV_TEST_HOOKS)
            if (optimized_transparent_one) {
                g_cv_transparent_profile.rough_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - rough_started).count();
            }
#endif
        } else {
            cv::matchTemplate(candidate_source, candidate_templ, scratch.score, method, *candidate_mask);
        }
        // The tile scan treats NaNs as -1. Other matching paths still need a
        // materialized finite matrix for OpenCV peak operations.
        if (!optimized_transparent_one) cv::patchNaNs(scratch.score, -1.0);
        const auto exact_color_score = [&](int x,int y) {
            return correlation_ready && !best_only
                ? workspace.correlation.exact_at(image,templ,entry->color_stats.sums,entry->color_stats.energy,x,y)
                : exact_ccoeff_score_at(image,templ,entry->color_stats,x,y);
        };
#if defined(AIENGINE_CV_TEST_HOOKS)
        auto prof_post = std::chrono::steady_clock::now();
#endif
        CVMaskedTemplateStats masked_stats;
        bool masked_avx2 = false;
        if (optimized_transparent_one) {
#if defined(AIENGINE_CV_TEST_HOOKS)
            const auto started = std::chrono::steady_clock::now();
#endif
            masked_stats = prepare_masked_score(templ, *mask, image.step,
                &scratch.visible_pixels);
#if defined(AIENGINE_CV_AVX2)
            masked_avx2 = match_mode == 0 && scratch.visible_pixels.size() <= 65535 &&
                image.total() <= static_cast<size_t>(INT32_MAX / 3) &&
                cv::checkHardwareSupport(CV_CPU_AVX2)
#if defined(AIENGINE_CV_TEST_HOOKS)
                && !g_cv_force_scalar_masked
#endif
                ;
            if (masked_avx2) {
                scratch.visible_offsets.resize(scratch.visible_pixels.size());
                scratch.visible_bgr.resize(scratch.visible_pixels.size());
                for (size_t i = 0; i < scratch.visible_pixels.size(); ++i) {
                    const CVVisiblePixel& pixel = scratch.visible_pixels[i];
                    scratch.visible_offsets[i] = pixel.offset;
                    scratch.visible_bgr[i] = static_cast<uint32_t>(pixel.value[0]) |
                        (static_cast<uint32_t>(pixel.value[1]) << 8) |
                        (static_cast<uint32_t>(pixel.value[2]) << 16);
                }
            }
#endif
#if defined(AIENGINE_CV_TEST_HOOKS)
            g_cv_transparent_profile.prepare_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
#endif
        }
        std::optional<CVTileMaxHeap> tile_selector;
        if (optimized_transparent_one) tile_selector.emplace(scratch.score, scratch);
        const auto masked_score_at = [&](int x, int y) {
            if (!optimized_transparent_one) {
                return exact_masked_ccoeff_score_at(image, templ, *mask, x, y);
            }
#if defined(AIENGINE_CV_TEST_HOOKS)
            const auto started = std::chrono::steady_clock::now();
            ++g_cv_transparent_profile.candidates;
#endif
            const float result = exact_masked_ccoeff_score_at_prepared(
                image, scratch.visible_pixels, masked_stats, x, y, masked_avx2,
                scratch.visible_offsets, scratch.visible_bgr);
#if defined(AIENGINE_CV_TEST_HOOKS)
            g_cv_transparent_profile.exact_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
#endif
            return result;
        };
        bool best_match_complete = false;
        cv::Point best_point;
        if (best_only) {
            double best_score = -1.0;
            if (tile_selector) {
                const auto peak = tile_selector->top();
                best_score = peak.first;
                best_point = peak.second;
            } else {
                cv::minMaxLoc(scratch.score, nullptr, &best_score, nullptr, &best_point);
            }
            float final_score = static_cast<float>(best_score);
            if (use_transparent_mask && std::isfinite(best_score)) {
                final_score = masked_score_at(best_point.x, best_point.y);
            } else if (match_mode == 0 && std::isfinite(best_score)) {
                final_score = exact_color_score(best_point.x, best_point.y);
            } else if (best_score > 1.0 - 1e-6) {
                final_score = 1.0f;
            }
            if (std::isfinite(final_score) && final_score >= min_score &&
                passes_color_bias(image, templ, mask, best_point.x, best_point.y, color_bias, min_score)) {
                candidates.push_back(CVMatchResult{best_point.x, best_point.y, templ.cols, templ.rows,
                    final_score, static_cast<int32_t>(template_index)});
            }
            // FindOne returns one valid target. Once the coarse global maximum
            // passes the exact score and color checks, scanning every remaining
            // local peak cannot improve the caller-visible single-result contract.
            best_match_complete = !candidates.empty();
        }
        if (!best_match_complete) {
            candidates.clear();
            const float candidate_floor = use_transparent_mask
                ? std::min(min_score, 0.30f)
                : std::max(0.0f, min_score - 1e-6f);
            const auto evaluate_point = [&](const cv::Point& point, float rough_score) {
                float score = rough_score;
                if (use_transparent_mask) {
                    score = masked_score_at(point.x, point.y);
                } else if (match_mode == 0) {
                    score = exact_color_score(point.x, point.y);
                } else if (score > 1.0f - 1e-6f) {
                    score = 1.0f;
                }
                if (!std::isfinite(score) || score < min_score) return false;
                if (!passes_color_bias(image, templ, mask, point.x, point.y, color_bias, min_score)) return false;
                candidates.push_back(CVMatchResult{point.x, point.y, templ.cols, templ.rows, score,
                    static_cast<int32_t>(template_index)});
                return true;
            };
            const int32_t radius_x = std::max<int32_t>(1,
                static_cast<int32_t>(std::floor(templ.cols * nms_iou)));
            const int32_t radius_y = std::max<int32_t>(1,
                static_cast<int32_t>(std::floor(templ.rows * nms_iou)));
            const auto suppress_peak = [&](const cv::Point& point) {
                const int32_t left = std::max(0, point.x - radius_x);
                const int32_t top = std::max(0, point.y - radius_y);
                const int32_t right = std::min(scratch.score.cols, point.x + radius_x + 1);
                const int32_t bottom = std::min(scratch.score.rows, point.y + radius_y + 1);
                const cv::Rect rectangle(left, top, right - left, bottom - top);
                if (tile_selector) {
                    tile_selector->suppressed(rectangle);
                } else {
                    scratch.score(rectangle).setTo(-1.0f);
                }
            };
            if (best_only) {
                suppress_peak(best_point);
                while (candidates.empty()) {
                    double rough_score = -1.0;
                    cv::Point point;
                    if (tile_selector) {
                        const auto peak = tile_selector->top();
                        rough_score = peak.first;
                        point = peak.second;
                    } else {
                        cv::minMaxLoc(scratch.score, nullptr, &rough_score, nullptr, &point);
                    }
                    if (!std::isfinite(rough_score) || rough_score < candidate_floor) break;
                    const bool accepted = evaluate_point(point, static_cast<float>(rough_score));
                    suppress_peak(point);
                    if (accepted) break;
                }
            } else {
                collect_local_peaks(scratch.score, templ.cols, templ.rows, nms_iou,
                    candidate_floor,
                    &scratch.local_max, &scratch.peak_kernel, &scratch.peak_points);
                for (const cv::Point& point : scratch.peak_points) {
                    evaluate_point(point, scratch.score.at<float>(point));
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const CVMatchResult& lhs, const CVMatchResult& rhs) {
            if (lhs.sim != rhs.sim) return lhs.sim > rhs.sim;
            return cv_result_screen_order(lhs, rhs);
        });
        std::vector<CVMatchResult>& selected = scratch.selected;
        selected.clear();
        selected.reserve(candidates.size());
        for (const CVMatchResult& candidate : candidates) {
            const bool suppressed = std::any_of(selected.begin(), selected.end(), [&](const CVMatchResult& previous) {
                return cv_match_iou(previous, candidate) > nms_iou;
            });
            if (!suppressed) selected.push_back(candidate);
            if (best_only && !selected.empty()) break;
        }
        output->insert(output->end(), selected.begin(), selected.end());
#if defined(AIENGINE_CV_TEST_HOOKS)
        g_cv_times[5]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prof_post).count();
#endif
    };
    for (size_t template_index = 0; template_index < entries.size(); ++template_index) {
        match_entry(template_index);
    }
    const int32_t error = match_error.load(std::memory_order_relaxed);
    if (error < 0) return error;
    std::sort(output->begin(), output->end(), cv_result_screen_order);
    return static_cast<int32_t>(output->size());
}

#else

int32_t find_cv_matches_opencv(
    const AIImage& big_image,
    const std::vector<std::shared_ptr<const TemplateEntry>>& entries,
    int32_t match_mode,
    bool transparent,
    uint32_t,
    const CVColorBias& color_bias,
    float min_score,
    float,
    bool best_only,
    std::vector<CVMatchResult>* output) {
    if (output == nullptr || entries.empty()) return AI_ERR_INVALID_ARGUMENT;
    OwnedImage big_gray;
    const AIImage* image = &big_image;
    std::vector<AIImage> templates;
    templates.reserve(entries.size());
    for (const auto& entry : entries) templates.push_back(match_mode == 1 ? entry->gray.view : entry->color.view);
    if (match_mode == 1) {
        big_gray = make_gray_image(big_image);
        image = &big_gray.view;
    }
    int64_t capacity = 0;
    for (const AIImage& templ : templates) {
        if (templ.width <= image->width && templ.height <= image->height) {
            capacity += static_cast<int64_t>(image->width - templ.width + 1) * (image->height - templ.height + 1);
        }
    }
    if (best_only) capacity = 1;
    if (capacity <= 0) {
        output->clear();
        return 0;
    }
    capacity = std::min<int64_t>(capacity, std::numeric_limits<int32_t>::max());
    std::vector<AIImageMatch> raw(static_cast<size_t>(capacity));
    const int32_t tolerance = color_bias.enabled
        ? (match_mode == 1 ? color_bias.gray : std::max({color_bias.b, color_bias.g, color_bias.r})) : 0;
    const int32_t status = transparent
        ? ai::cv_find_transparent_images_with_tolerance(*image, templates.data(), static_cast<int32_t>(templates.size()), 0,
            tolerance, min_score, raw.data(), static_cast<int32_t>(raw.size()))
        : ai::cv_find_images_with_tolerance(*image, templates.data(), static_cast<int32_t>(templates.size()),
            tolerance, min_score, raw.data(), static_cast<int32_t>(raw.size()));
    if (status < 0) return status;
    output->resize(static_cast<size_t>(status));
    for (int32_t i = 0; i < status; ++i) to_cv_result(raw[static_cast<size_t>(i)], &(*output)[static_cast<size_t>(i)]);
    std::sort(output->begin(), output->end(), cv_result_screen_order);
    return status;
}

#endif

int32_t find_cv_matches_multi(
    const AIImage& big_image,
    const std::vector<std::shared_ptr<const TemplateEntry>>& entries,
    int32_t match_mode,
    bool transparent,
    uint32_t transparent_rgb,
    const CVColorBias& color_bias,
    float min_score,
    float nms_iou,
    bool best_only,
    std::vector<CVMatchResult>* output) {
    return find_cv_matches_opencv(big_image, entries, match_mode, transparent, transparent_rgb,
        color_bias, min_score, nms_iou, best_only, output);
}

uint32_t pixel_rgb_at(const AIImage& image, int32_t x, int32_t y) {
    const int channels = ai::channels_for_format(image.format);
    const uint8_t* pixel = ai::image_row_ptr(image, y) + x * channels;
    uint8_t b = 0, g = 0, r = 0;
    ai::read_bgr(pixel, image.format, &b, &g, &r);
    return (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
}

bool corners_match_rgb(const AIImage& image, uint32_t rgb) {
    if (!ai::validate_image(image)) {
        return false;
    }
    return pixel_rgb_at(image, 0, 0) == rgb &&
        pixel_rgb_at(image, image.width - 1, 0) == rgb &&
        pixel_rgb_at(image, 0, image.height - 1) == rgb &&
        pixel_rgb_at(image, image.width - 1, image.height - 1) == rgb;
}

OwnedImage make_transparent_template(const AIImage& src, uint32_t transparent_rgb) {
    OwnedImage out;
    out.pixels.resize(checked_image_bytes(src.width, src.height, 4));
    out.view = AIImage{out.pixels.data(), src.width, src.height, src.width * 4, AI_IMAGE_BGRA32};
    const int src_channels = ai::channels_for_format(src.format);
    for (int32_t y = 0; y < src.height; ++y) {
        const uint8_t* src_row = ai::image_row_ptr(src, y);
        uint8_t* dst_row = ai::image_row_ptr(out.view, y);
        for (int32_t x = 0; x < src.width; ++x) {
            uint8_t b = 0, g = 0, r = 0;
            ai::read_bgr(src_row + x * src_channels, src.format, &b, &g, &r);
            uint8_t* dst = dst_row + x * 4;
            dst[0] = b;
            dst[1] = g;
            dst[2] = r;
            const uint32_t rgb = (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
            dst[3] = rgb == transparent_rgb ? 0 : 255;
        }
    }
    return out;
}

std::string read_text_file(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return std::string();
    }
    const std::filesystem::path resolved_path = resolve_utf8_api_path(path);
    if (resolved_path.empty()) return std::string();
    std::ifstream input(resolved_path, std::ios::binary);
    if (!input) {
        return std::string();
    }
    std::ostringstream oss;
    oss << input.rdbuf();
    return oss.str();
}

ai::Config make_ocr_api_config(const char* keys_path, int32_t device, int32_t session_count) {
    ai::Config config;
    config.set_string("runtime.prefer_gpu", "true");
    config.set_int("runtime.device", device);
    config.set_int("runtime.thread_count", session_count > 0 ? session_count : 1);
    config.set_string("yolo.backend", "null");
    config.set_string("ocr.backend", "onnxruntime");
    config.set_string("ocr.rec_only", "false");
    config.set_int("ocr.input_height", 48);
    config.set_int("ocr.input_width", 320);
    config.set_string("ocr.channel_order", "bgr");
    if (keys_path != nullptr && keys_path[0] != '\0') {
        const std::filesystem::path resolved_path = resolve_utf8_api_path(keys_path);
        config.set_string("ocr.charset_path", resolved_path.empty() ? keys_path : resolved_path.u8string());
    }
    return config;
}

std::string format_yolo_json(const std::vector<AIDetectBox>& boxes, int32_t count) {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << '[';
    const int32_t safe_count = std::max<int32_t>(0, std::min<int32_t>(count, static_cast<int32_t>(boxes.size())));
    for (int32_t i = 0; i < safe_count; ++i) {
        const AIDetectBox& box = boxes[static_cast<size_t>(i)];
        if (i > 0) {
            oss << ',';
        }
        const float x1 = std::isfinite(box.x1) ? box.x1 : 0.0f;
        const float y1 = std::isfinite(box.y1) ? box.y1 : 0.0f;
        const float x2 = std::isfinite(box.x2) ? box.x2 : 0.0f;
        const float y2 = std::isfinite(box.y2) ? box.y2 : 0.0f;
        const float score = std::isfinite(box.score) ? box.score : 0.0f;
        oss << "{\"x1\":" << x1
            << ",\"y1\":" << y1
            << ",\"x2\":" << x2
            << ",\"y2\":" << y2
            << ",\"cx\":" << ((x1 + x2) * 0.5f)
            << ",\"cy\":" << ((y1 + y2) * 0.5f)
            << ",\"score\":" << score
            << ",\"class_id\":" << box.class_id
            << ",\"label\":\"" << json_escape(box.label[0] == '\0' ? std::to_string(box.class_id).c_str() : box.label) << "\"}";
    }
    oss << ']';
    return oss.str();
}

struct OcrTargetMatch {
    int32_t target_index = 0;
    std::string target;
    AIOcrLine line{};
};

std::string format_ocr_target_text(const std::vector<OcrTargetMatch>& matches) {
    std::vector<ai::CompactPointResult> results;
    results.reserve(matches.size());
    for (const OcrTargetMatch& match : matches) {
        results.push_back(ai::CompactPointResult{
            match.target_index,
            match.line.box.x + match.line.box.w / 2,
            match.line.box.y + match.line.box.h / 2});
    }
    return ai::format_compact_points(results);
}

void fill_ocr_text_result(const AIOcrLine& line, OCRTextResult* out) {
    if (out == nullptr) {
        return;
    }
    out->x = line.box.x;
    out->y = line.box.y;
    out->w = line.box.w;
    out->h = line.box.h;
    out->cx = line.box.x + line.box.w / 2;
    out->cy = line.box.y + line.box.h / 2;
    out->score = line.confidence;
}

void fill_ocr_coord_result(const AIOcrLine& line, int32_t target_index, OCRCoordResult* out) {
    if (out == nullptr) {
        return;
    }
    out->x = line.box.x + line.box.w / 2;
    out->y = line.box.y + line.box.h / 2;
    out->w = line.box.w;
    out->h = line.box.h;
    out->target_index = target_index;
}

#if defined(_WIN32) && defined(_M_IX86)
extern "C" IMAGE_DOS_HEADER __ImageBase;
std::mutex g_worker_start_mutex;
struct ProxyYoloIdentity {
    uint64_t worker_instance = 0;
    int32_t remote_handle = 0;
    bool loading = false;
    ai_worker::RuntimeFlavor flavor = ai_worker::RuntimeFlavor::Core;
    int32_t requested_device = AI_DEVICE_AUTO;
    int32_t active_device = AI_DEVICE_AUTO;
    std::string fallback_reason;
};
std::mutex g_proxy_yolo_mutex;
std::unordered_map<int32_t, ProxyYoloIdentity> g_proxy_yolo_handles;
int32_t g_next_proxy_yolo_handle = 1;
std::mutex g_proxy_ocr_mutex;
std::mutex g_proxy_ocr_load_mutex;
struct ProxyOcrLoadRecipe {
    bool valid = false;
    ai_worker::RuntimeFlavor flavor = ai_worker::RuntimeFlavor::Core;
    uint32_t command = 0;
    std::vector<uint8_t> payload;
};
ProxyOcrLoadRecipe g_proxy_ocr_load_recipe;
uint64_t g_proxy_ocr_load_generation = 0;
ai_worker::RuntimeFlavor g_proxy_ocr_flavor = ai_worker::RuntimeFlavor::Core;
int32_t g_proxy_ocr_requested_device = AI_DEVICE_AUTO;
int32_t g_proxy_ocr_active_device = AI_DEVICE_AUTO;
std::string g_proxy_ocr_fallback_reason;

struct ProxyRuntimeCandidate {
    ai_worker::RuntimeFlavor flavor;
    int32_t device;
};

std::vector<ProxyRuntimeCandidate> proxy_runtime_candidates(int32_t requested_device) {
    switch (requested_device) {
        case AI_DEVICE_AUTO:
            return {{ai_worker::RuntimeFlavor::Core, AI_DEVICE_AUTO}};
        case AI_DEVICE_DIRECTML:
            return {{ai_worker::RuntimeFlavor::Core, AI_DEVICE_DIRECTML}};
        case AI_DEVICE_CPU:
            return {{ai_worker::RuntimeFlavor::Core, AI_DEVICE_CPU}};
        default:
            return {};
    }
}

void proxy_append_i32(std::vector<uint8_t>* out, int32_t value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void proxy_append_f32(std::vector<uint8_t>* out, float value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void proxy_append_u64(std::vector<uint8_t>* out, uint64_t value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    out->insert(out->end(), p, p + sizeof(value));
}

void proxy_append_yolo_params(
    std::vector<uint8_t>* out,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
    proxy_append_i32(out, input_size);
    proxy_append_i32(out, runtime_device);
    proxy_append_i32(out, device_id);
    proxy_append_i32(out, session_count);
}

bool get_proxy_yolo_identity(int32_t handle, ProxyYoloIdentity* identity) {
    if (handle <= 0 || identity == nullptr) return false;
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    if (it == g_proxy_yolo_handles.end() || it->second.remote_handle <= 0) return false;
    *identity = it->second;
    return true;
}

bool proxy_yolo_is_empty(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    return it != g_proxy_yolo_handles.end() && it->second.remote_handle == 0 && !it->second.loading;
}

bool proxy_yolo_is_loading(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    return it != g_proxy_yolo_handles.end() && it->second.loading;
}

int32_t begin_proxy_yolo_load(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    if (it == g_proxy_yolo_handles.end()) return AI_ERR_INVALID_HANDLE;
    if (it->second.loading) return AI_ERR_BUSY;
    if (it->second.remote_handle != 0) return AI_ERR_ALREADY_LOADED;
    it->second.loading = true;
    return AI_OK;
}

void cancel_proxy_yolo_load(int32_t handle) {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    if (it != g_proxy_yolo_handles.end() && it->second.remote_handle == 0) {
        it->second.loading = false;
    }
}

bool set_proxy_yolo_identity(int32_t handle, ProxyYoloIdentity identity) {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    const auto it = g_proxy_yolo_handles.find(handle);
    if (it == g_proxy_yolo_handles.end() || !it->second.loading) return false;
    identity.loading = false;
    it->second = std::move(identity);
    return true;
}

void proxy_append_yolo_identity(std::vector<uint8_t>* out, const ProxyYoloIdentity& identity) {
    proxy_append_u64(out, identity.worker_instance);
    proxy_append_i32(out, identity.remote_handle);
}

void clear_proxy_yolo_handles() {
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    g_proxy_yolo_handles.clear();
}

void proxy_append_bytes(std::vector<uint8_t>* out, const void* data, int32_t size) {
    proxy_append_i32(out, size);
    if (data != nullptr && size > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        out->insert(out->end(), p, p + size);
    }
}

void proxy_append_string(std::vector<uint8_t>* out, const char* value) {
    const char* safe = value == nullptr ? "" : value;
    proxy_append_bytes(out, safe, static_cast<int32_t>(std::strlen(safe)));
}

void proxy_append_string_candidates(std::vector<uint8_t>* out, const std::vector<std::string>& candidates) {
    proxy_append_i32(out, static_cast<int32_t>(candidates.size()));
    for (const std::string& candidate : candidates) proxy_append_string(out, candidate.c_str());
}

bool proxy_read_string(const std::vector<uint8_t>& payload, std::string* out) {
    if (payload.size() < sizeof(int32_t) || out == nullptr) return false;
    int32_t size = 0;
    std::memcpy(&size, payload.data(), sizeof(size));
    if (size < 0 || payload.size() < sizeof(int32_t) + static_cast<size_t>(size)) return false;
    out->assign(reinterpret_cast<const char*>(payload.data() + sizeof(int32_t)), static_cast<size_t>(size));
    return true;
}

bool proxy_read_sized_bytes(const std::vector<uint8_t>& payload, void* output, size_t output_size) {
    if (output == nullptr || payload.size() < sizeof(int32_t)) return false;
    int32_t size = 0;
    std::memcpy(&size, payload.data(), sizeof(size));
    if (size < 0 || static_cast<size_t>(size) != output_size ||
        payload.size() != sizeof(int32_t) + output_size) {
        return false;
    }
    std::memcpy(output, payload.data() + sizeof(int32_t), output_size);
    return true;
}

std::filesystem::path module_path() {
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        reinterpret_cast<HMODULE>(&__ImageBase), path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) return {};
    path.resize(length);
    return std::filesystem::path(path);
}

std::string proxy_compat_absolute_path(const char* path) {
    const std::filesystem::path value = resolve_compat_api_path(path);
    return value.empty() ? std::string() : value.u8string();
}

const wchar_t* worker_file_name(ai_worker::RuntimeFlavor) {
    return L"CQ_AI_worker.exe";
}

std::filesystem::path worker_executable_path(ai_worker::RuntimeFlavor flavor, bool* used_override) {
    const std::filesystem::path base_dir = module_path().parent_path();
    if (used_override != nullptr) *used_override = false;
    return base_dir / worker_file_name(flavor);
}

bool start_worker_process(ai_worker::RuntimeFlavor flavor) {
    bool ignored_override = false;
    const std::filesystem::path exe =
        worker_executable_path(flavor, &ignored_override);
    const DWORD attrs = GetFileAttributesW(exe.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        ai::set_last_error(
            std::string(ai_worker::runtime_flavor_name(flavor)) +
            " worker is missing; expected exactly beside CQ_X86.dll: " +
            exe.u8string());
        return false;
    }
    const std::filesystem::path work_dir = exe.parent_path();
    STARTUPINFOW si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    std::wstring command_line = L"\"" + exe.wstring() + L"\" --owner-pid=" +
        std::to_wstring(GetCurrentProcessId());
    std::vector<wchar_t> command_buffer(command_line.begin(), command_line.end());
    command_buffer.push_back(L'\0');
    const BOOL ok = CreateProcessW(
        exe.c_str(),
        command_buffer.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        work_dir.c_str(),
        &si,
        &pi);
    bool started = ok != FALSE;
    if (ok) {
        CloseHandle(pi.hThread);
        if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
            DWORD exit_code = 0;
            GetExitCodeProcess(pi.hProcess, &exit_code);
            if (exit_code == 0) {
                // A healthy Worker may own the singleton while all currently
                // published pipe instances are briefly busy.  The duplicate
                // process exits successfully; keep retrying the existing
                // Worker's pipe instead of reporting a false startup failure.
                started = true;
            } else if (exit_code == 0xC0000135u) {
                started = false;
                ai::set_last_error(
                    std::string(ai_worker::runtime_flavor_name(flavor)) +
                    " worker could not start because a Windows system dependency is missing "
                    "(Windows status 0xC0000135)");
            } else {
                started = false;
                std::ostringstream message;
                message << ai_worker::runtime_flavor_name(flavor)
                        << " worker exited during startup with Windows status 0x"
                        << std::hex << std::uppercase << exit_code;
                ai::set_last_error(message.str());
            }
        }
        CloseHandle(pi.hProcess);
    } else {
        ai::set_last_error(
            std::string("failed to start ") + ai_worker::runtime_flavor_name(flavor) +
            " worker (Windows error " +
            std::to_string(GetLastError()) + "), path: " + exe.u8string());
    }
    return started;
}

HANDLE open_pipe_server_process(HANDLE pipe) {
    ULONG process_id = 0;
    if (pipe == INVALID_HANDLE_VALUE || !GetNamedPipeServerProcessId(pipe, &process_id) || process_id == 0) {
        return nullptr;
    }
    return OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
}

std::string worker_disconnect_error(
    ai_worker::RuntimeFlavor flavor,
    HANDLE process,
    const char* phase,
    DWORD pipe_error) {
    DWORD exit_code = STILL_ACTIVE;
    if (process != nullptr) {
        WaitForSingleObject(process, 500);
        GetExitCodeProcess(process, &exit_code);
    }
    if (exit_code != STILL_ACTIVE) {
        std::ostringstream message;
        message << ai_worker::runtime_flavor_name(flavor) << " worker exited while "
                << phase << " with Windows status 0x"
                << std::hex << std::uppercase << exit_code;
        if (exit_code == 0xC0000135u) {
            message << "; a required runtime DLL is missing";
        } else {
            message << "; verify the embedded runtime package and private cache";
        }
        return message.str();
    }
    return std::string("worker closed the pipe while ") + phase +
        " (Windows error " + std::to_string(pipe_error) + ")";
}

bool proxy_request(
    ai_worker::RuntimeFlavor flavor,
    uint32_t command,
    const std::vector<uint8_t>& payload,
    int32_t* status,
    std::vector<uint8_t>* response,
    bool allow_start = true,
    bool wait_for_shutdown = false) {
    if (status == nullptr || response == nullptr) return false;
    ai::set_last_error("");
    response->clear();

    HANDLE pipe = INVALID_HANDLE_VALUE;
    const int max_attempts = allow_start ? 30 : 1;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (WaitNamedPipeA(ai_worker::pipe_name(flavor), 250)) {
            pipe = CreateFileA(ai_worker::pipe_name(flavor), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            if (pipe != INVALID_HANDLE_VALUE) break;
        }
        if (allow_start && (attempt == 0 || attempt == 10)) {
            std::lock_guard<std::mutex> lock(g_worker_start_mutex);
            if (WaitNamedPipeA(ai_worker::pipe_name(flavor), 50)) {
                pipe = CreateFileA(ai_worker::pipe_name(flavor), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
                if (pipe != INVALID_HANDLE_VALUE) break;
            } else {
                if (!start_worker_process(flavor)) {
                    *status = AI_ERR_RUNTIME;
                    return false;
                }
            }
        }
        Sleep(100);
    }
    if (pipe == INVALID_HANDLE_VALUE) {
        if (ai::last_error().empty()) {
            ai::set_last_error(
                std::string(ai_worker::runtime_flavor_name(flavor)) +
                " worker did not create the v23 named pipe or exited during startup");
        }
        *status = AI_ERR_RUNTIME;
        return false;
    }
    HANDLE worker_process = open_pipe_server_process(pipe);

    ai_worker::Header header{ai_worker::kMagic, ai_worker::kVersion, command, static_cast<uint32_t>(payload.size())};
    DWORD written = 0;
    BOOL ok = WriteFile(pipe, &header, sizeof(header), &written, nullptr);
    if (ok && !payload.empty()) {
        ok = WriteFile(pipe, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr);
    }
    if (!ok) {
        const DWORD error_code = GetLastError();
        CloseHandle(pipe);
        ai::set_last_error(worker_disconnect_error(flavor, worker_process, "writing the request", error_code));
        if (worker_process != nullptr) CloseHandle(worker_process);
        *status = AI_ERR_RUNTIME;
        return false;
    }

    ai_worker::ResponseHeader rh{};
    DWORD got = 0;
    ok = ReadFile(pipe, &rh, sizeof(rh), &got, nullptr);
    if (!ok) {
        const DWORD error_code = GetLastError();
        CloseHandle(pipe);
        ai::set_last_error(worker_disconnect_error(flavor, worker_process, "waiting for the response", error_code));
        if (worker_process != nullptr) CloseHandle(worker_process);
        *status = AI_ERR_RUNTIME;
        return false;
    }
    if (got != sizeof(rh)) {
        CloseHandle(pipe);
        if (worker_process != nullptr) CloseHandle(worker_process);
        ai::set_last_error("worker returned a truncated response header");
        *status = AI_ERR_RUNTIME;
        return false;
    }
    if (rh.magic != ai_worker::kMagic || rh.version != ai_worker::kVersion) {
        CloseHandle(pipe);
        if (worker_process != nullptr) CloseHandle(worker_process);
        ai::set_last_error(
            "worker protocol mismatch: expected version " + std::to_string(ai_worker::kVersion) +
            ", got " + std::to_string(rh.version));
        *status = AI_ERR_RUNTIME;
        return false;
    }

    response->resize(rh.payload_size);
    uint32_t total = 0;
    while (total < rh.payload_size) {
        got = 0;
        ok = ReadFile(pipe, response->data() + total, rh.payload_size - total, &got, nullptr);
        if (!ok || got == 0) {
            const DWORD error_code = GetLastError();
            CloseHandle(pipe);
            ai::set_last_error(worker_disconnect_error(flavor, worker_process, "reading the response payload", error_code));
            if (worker_process != nullptr) CloseHandle(worker_process);
            *status = AI_ERR_RUNTIME;
            return false;
        }
        total += got;
    }
    const int32_t response_status = rh.status;
    CloseHandle(pipe);
    *status = response_status;
    if (wait_for_shutdown && command == ai_worker::CMD_SHUTDOWN && response_status >= 0) {
        if (worker_process == nullptr) {
            ai::set_last_error("Worker shutdown was acknowledged but its process handle was unavailable");
            *status = AI_ERR_RUNTIME;
        } else {
            const DWORD wait_result = WaitForSingleObject(worker_process, 5000);
            if (wait_result == WAIT_TIMEOUT) {
                ai::set_last_error("Worker shutdown acknowledged but process did not exit within 5000 ms");
                *status = AI_ERR_RUNTIME;
            } else if (wait_result == WAIT_FAILED) {
                ai::set_last_error(
                    "Worker shutdown was acknowledged but waiting for process exit failed (Windows error " +
                    std::to_string(GetLastError()) + ")");
                *status = AI_ERR_RUNTIME;
            }
        }
    }
    if (worker_process != nullptr) CloseHandle(worker_process);
    if (*status < 0) {
        std::string error;
        if (proxy_read_string(*response, &error) && !error.empty()) {
            ai::set_last_error(error);
        } else {
            ai::set_last_error(
                "worker returned error status " + std::to_string(*status) + " without details");
        }
    }
    return true;
}

bool proxy_request_if_running(
    ai_worker::RuntimeFlavor flavor,
    uint32_t command,
    const std::vector<uint8_t>& payload,
    int32_t* status,
    std::vector<uint8_t>* response,
    bool wait_for_shutdown = false) {
    ai::set_last_error("");
    if (!WaitNamedPipeA(ai_worker::pipe_name(flavor), 0)) {
        if (status != nullptr) *status = AI_OK;
        if (response != nullptr) response->clear();
        return true;
    }
    return proxy_request(flavor, command, payload, status, response, false, wait_for_shutdown);
}

ai_worker::RuntimeFlavor proxy_ocr_flavor() {
    std::lock_guard<std::mutex> lock(g_proxy_ocr_mutex);
    return g_proxy_ocr_flavor;
}

std::string proxy_failure_entry(const ProxyRuntimeCandidate& candidate, const std::string& detail) {
    return std::string(ai_worker::runtime_flavor_name(candidate.flavor)) + "/" +
        runtime_device_name(candidate.device) + ": " +
        (detail.empty() ? "runtime request failed without details" : detail);
}

std::string proxy_join_failures(const std::vector<std::string>& failures) {
    std::ostringstream out;
    for (size_t i = 0; i < failures.size(); ++i) {
        if (i != 0) out << "; ";
        out << failures[i];
    }
    return out.str();
}

void proxy_replace_json_string(std::string* json, const char* field, const std::string& value) {
    if (json == nullptr || field == nullptr) return;
    const std::string prefix = std::string("\"") + field + "\":\"";
    const size_t value_begin = json->find(prefix);
    if (value_begin == std::string::npos) return;
    const size_t begin = value_begin + prefix.size();
    const size_t end = json->find('"', begin);
    if (end == std::string::npos) return;
    json->replace(begin, end - begin, json_escape(value.c_str()));
}

void proxy_replace_json_bool(std::string* json, const char* field, bool value) {
    if (json == nullptr || field == nullptr) return;
    const std::string prefix = std::string("\"") + field + "\":";
    const size_t value_begin = json->find(prefix);
    if (value_begin == std::string::npos) return;
    const size_t begin = value_begin + prefix.size();
    if (json->compare(begin, 4, "true") == 0) {
        json->replace(begin, 4, value ? "true" : "false");
    } else if (json->compare(begin, 5, "false") == 0) {
        json->replace(begin, 5, value ? "true" : "false");
    }
}

void proxy_rewrite_runtime_status(
    std::string* json,
    int32_t requested_device,
    int32_t active_device,
    const std::string& fallback_reason) {
    // AUTO routing is resolved inside the single Worker. Its returned status
    // already contains the real active Provider and selection reason.
    if (active_device == AI_DEVICE_AUTO) return;
    proxy_replace_json_string(json, "requested", runtime_device_name(requested_device));
    proxy_replace_json_string(json, "active", runtime_device_name(active_device));
    const bool degraded =
        requested_device == AI_DEVICE_AUTO &&
        active_device == AI_DEVICE_CPU;
    proxy_replace_json_bool(json, "degraded", degraded);
    if (!fallback_reason.empty()) proxy_replace_json_string(json, "reason", fallback_reason);
}

bool proxy_load_ocr_candidates(
    int32_t requested_device,
    uint32_t command,
    const std::function<std::vector<uint8_t>(const ProxyRuntimeCandidate&)>& make_payload,
    int32_t* status,
    std::vector<uint8_t>* response) {
    const std::vector<ProxyRuntimeCandidate> candidates = proxy_runtime_candidates(requested_device);
    if (candidates.empty() || status == nullptr || response == nullptr) {
        if (status != nullptr) *status = AI_ERR_INVALID_ARGUMENT;
        ai::set_last_error(
            "Invalid runtime device " + std::to_string(requested_device) +
            "; valid values are 0=AUTO, 1=DirectML, 2=CPU");
        return false;
    }

    // OCR has one active pool across all runtime flavors. Serialize cross-worker
    // replacements so a successful new pool can be published before the old one
    // is released.
    std::lock_guard<std::mutex> load_lock(g_proxy_ocr_load_mutex);
    ai_worker::RuntimeFlavor previous_flavor = ai_worker::RuntimeFlavor::Core;
    {
        std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
        previous_flavor = g_proxy_ocr_flavor;
    }
    std::vector<std::string> failures;
    int32_t last_status = AI_ERR_RUNTIME;
    for (const ProxyRuntimeCandidate& candidate : candidates) {
        const std::vector<uint8_t> payload = make_payload(candidate);
        const bool requested = proxy_request(candidate.flavor, command, payload, status, response);
        if (requested && *status >= 0) {
            {
                std::lock_guard<std::mutex> lock(g_proxy_ocr_mutex);
                g_proxy_ocr_flavor = candidate.flavor;
                g_proxy_ocr_requested_device = requested_device;
                g_proxy_ocr_active_device = candidate.device;
                g_proxy_ocr_fallback_reason = proxy_join_failures(failures);
                g_proxy_ocr_load_recipe.valid = true;
                g_proxy_ocr_load_recipe.flavor = candidate.flavor;
                g_proxy_ocr_load_recipe.command = command;
                g_proxy_ocr_load_recipe.payload = payload;
                ++g_proxy_ocr_load_generation;
            }
            if (previous_flavor != candidate.flavor) {
                int32_t ignored_status = AI_OK;
                std::vector<uint8_t> ignored_response;
                proxy_request_if_running(
                    previous_flavor,
                    ai_worker::CMD_OCR_RELEASE,
                    {},
                    &ignored_status,
                    &ignored_response);
            }
            return true;
        }
        if (*status < 0) last_status = *status;
        failures.push_back(proxy_failure_entry(candidate, ai::last_error()));
    }
    *status = last_status;
    ai::set_last_error(proxy_join_failures(failures));
    return false;
}

bool proxy_ocr_request(
    uint32_t command,
    const std::vector<uint8_t>& payload,
    int32_t* status,
    std::vector<uint8_t>* response) {
    ai_worker::RuntimeFlavor flavor = ai_worker::RuntimeFlavor::Core;
    uint64_t observed_generation = 0;
    bool can_restore = false;
    {
        std::lock_guard<std::mutex> lock(g_proxy_ocr_mutex);
        flavor = g_proxy_ocr_flavor;
        observed_generation = g_proxy_ocr_load_generation;
        can_restore = g_proxy_ocr_load_recipe.valid;
    }

    const bool requested =
        proxy_request(flavor, command, payload, status, response);
    if (requested && *status >= 0) return true;
    const bool recoverable = can_restore &&
        (!requested || *status == AI_ERR_BACKEND_NOT_CONFIGURED ||
         *status == AI_ERR_BUSY);
    if (!recoverable) return requested;

    std::lock_guard<std::mutex> load_lock(g_proxy_ocr_load_mutex);
    ProxyOcrLoadRecipe recipe;
    uint64_t current_generation = 0;
    {
        std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
        recipe = g_proxy_ocr_load_recipe;
        current_generation = g_proxy_ocr_load_generation;
    }
    if (!recipe.valid) return requested;

    if (current_generation == observed_generation) {
        bool restored = false;
        for (int attempt = 0; attempt < 150 && !restored; ++attempt) {
            std::vector<uint8_t> load_response;
            const bool load_requested = proxy_request(
                recipe.flavor,
                recipe.command,
                recipe.payload,
                status,
                &load_response);
            restored = load_requested && *status >= 0;
            if (!restored) {
                *response = std::move(load_response);
                Sleep(100);
            }
        }
        if (!restored) return false;
        {
            std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
            ++g_proxy_ocr_load_generation;
            flavor = g_proxy_ocr_flavor;
        }
    } else {
        std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
        flavor = g_proxy_ocr_flavor;
    }
    const bool retried =
        proxy_request(flavor, command, payload, status, response);
    if (retried && *status == 0) {
        // A freshly created DirectML Worker can occasionally return an empty
        // first inference while its provider resources finish warming. Retry
        // only at this post-restore boundary; ordinary empty-image requests
        // and the stable fast path are unchanged.
        return proxy_request(flavor, command, payload, status, response);
    }
    return retried;
}

bool proxy_create_remote_yolo(
    const ProxyRuntimeCandidate& candidate,
    int32_t requested_device,
    const std::string& fallback_reason,
    ProxyYoloIdentity* identity) {
    if (identity == nullptr) return false;
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    bool requested = proxy_request(candidate.flavor, ai_worker::CMD_YOLO_CREATE, {}, &status, &response);
    for (int retry = 0; retry < 2 && (!requested || status == AI_ERR_BUSY); ++retry) {
        Sleep(100);
        requested = proxy_request(candidate.flavor, ai_worker::CMD_YOLO_CREATE, {}, &status, &response);
    }
    if (!requested || status < 0 || response.size() != sizeof(uint64_t) + sizeof(int32_t)) {
        if (status >= 0) ai::set_last_error("worker returned an invalid YOLO identity");
        return false;
    }
    std::memcpy(&identity->worker_instance, response.data(), sizeof(identity->worker_instance));
    std::memcpy(
        &identity->remote_handle,
        response.data() + sizeof(identity->worker_instance),
        sizeof(identity->remote_handle));
    identity->flavor = candidate.flavor;
    identity->requested_device = requested_device;
    identity->active_device = candidate.device;
    identity->fallback_reason = fallback_reason;
    return true;
}

void proxy_release_remote_yolo(const ProxyYoloIdentity& identity) {
    if (identity.remote_handle <= 0) return;
    std::vector<uint8_t> request;
    proxy_append_yolo_identity(&request, identity);
    int32_t ignored_status = AI_OK;
    std::vector<uint8_t> ignored_response;
    proxy_request(identity.flavor, ai_worker::CMD_YOLO_RELEASE, request, &ignored_status, &ignored_response, false);
}

bool proxy_load_yolo_candidates(
    int32_t local_handle,
    int32_t requested_device,
    uint32_t command,
    const std::function<std::vector<uint8_t>(const ProxyYoloIdentity&, const ProxyRuntimeCandidate&)>& make_payload,
    int32_t* status,
    std::vector<uint8_t>* response) {
    const std::vector<ProxyRuntimeCandidate> candidates = proxy_runtime_candidates(requested_device);
    if (candidates.empty() || status == nullptr || response == nullptr) {
        if (status != nullptr) *status = AI_ERR_INVALID_ARGUMENT;
        ai::set_last_error(
            "Invalid runtime device " + std::to_string(requested_device) +
            "; valid values are 0=AUTO, 1=DirectML, 2=CPU");
        return false;
    }
    const int32_t handle_state = begin_proxy_yolo_load(local_handle);
    if (handle_state != AI_OK) {
        *status = handle_state;
        if (handle_state == AI_ERR_ALREADY_LOADED) {
            ai::set_last_error("YOLO handle already has a loaded model");
        } else if (handle_state == AI_ERR_BUSY) {
            ai::set_last_error("YOLO handle is already loading a model");
        } else {
            ai::set_last_error("YOLO handle is invalid");
        }
        return false;
    }

    std::vector<std::string> failures;
    int32_t last_status = AI_ERR_RUNTIME;
    for (const ProxyRuntimeCandidate& candidate : candidates) {
        ProxyYoloIdentity identity{};
        const std::string previous_failures = proxy_join_failures(failures);
        if (!proxy_create_remote_yolo(candidate, requested_device, previous_failures, &identity)) {
            last_status = AI_ERR_RUNTIME;
            failures.push_back(proxy_failure_entry(candidate, ai::last_error()));
            continue;
        }
        const std::vector<uint8_t> payload = make_payload(identity, candidate);
        const bool requested = proxy_request(candidate.flavor, command, payload, status, response);
        if (requested && *status >= 0) {
            identity.fallback_reason = previous_failures;
            if (!set_proxy_yolo_identity(local_handle, identity)) {
                proxy_release_remote_yolo(identity);
                *status = AI_ERR_INVALID_HANDLE;
                ai::set_last_error("YOLO handle was released while the model was loading");
                return false;
            }
            return true;
        }
        const std::string detail = ai::last_error();
        if (*status < 0) last_status = *status;
        proxy_release_remote_yolo(identity);
        failures.push_back(proxy_failure_entry(candidate, detail));
    }
    cancel_proxy_yolo_load(local_handle);
    *status = last_status;
    ai::set_last_error(proxy_join_failures(failures));
    return false;
}
#endif

} // namespace

// 返回 DLL 内部持有的静态版本字符串。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_GetVersion(void) {
#if defined(AIENGINE_BINARY_NAME)
    return AIENGINE_BINARY_NAME "/0.14.5";
#else
    return "CQ_X86/0.14.5";
#endif
}

// 默认初始化入口，按 AUTO 策略选择 GPU/CPU。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_Init(const char* config_path) {
    return AI_InitEx(config_path, AI_DEVICE_AUTO);
}

// 使用显式运行设备创建/替换全局引擎。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_InitEx(const char* config_path, int32_t runtime_device) {
    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_InitEx", "config_path must be valid UTF-8");
    }
    auto engine = std::make_shared<ai::Engine>();
    std::string error;
    if (!engine->init_ex(config_path, runtime_device, &error)) {
        ai::set_last_error(error.empty() ? "AI_Init 失败" : error);
        return AI_ERR_CONFIG;
    }

    {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_engine = std::move(engine);
    }
    ai::set_last_error("");
    return AI_OK;
}

// 释放全局引擎状态，并清空当前线程错误文本。
AIENGINE_EXPORT void AIENGINE_CALL AI_Release(void) {
#if defined(_WIN32) && defined(_M_IX86)
    int32_t status = AI_OK;
    std::vector<uint8_t> response;
    proxy_request_if_running(
        ai_worker::RuntimeFlavor::Core,
        ai_worker::CMD_RELEASE,
        {},
        &status,
        &response);
    clear_proxy_yolo_handles();
    {
        std::lock_guard<std::mutex> ocr_lock(g_proxy_ocr_mutex);
        g_proxy_ocr_flavor = ai_worker::RuntimeFlavor::Core;
        g_proxy_ocr_requested_device = AI_DEVICE_AUTO;
        g_proxy_ocr_active_device = AI_DEVICE_AUTO;
        g_proxy_ocr_fallback_reason.clear();
        g_proxy_ocr_load_recipe = ProxyOcrLoadRecipe{};
        ++g_proxy_ocr_load_generation;
    }
#else
    clear_local_yolo_contexts();
#endif
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    g_engine.reset();
    ai::set_last_error("");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_ShutdownWorker(void) {
#if defined(_WIN32) && defined(_M_IX86)
    int32_t first_error = AI_OK;
    std::string first_detail;
    int32_t status = AI_OK;
    std::vector<uint8_t> response;
    proxy_request_if_running(
        ai_worker::RuntimeFlavor::Core,
        ai_worker::CMD_SHUTDOWN,
        {},
        &status,
        &response,
        true);
    if (status < 0) {
        first_error = status;
        first_detail = ai::last_error();
    }
    clear_proxy_yolo_handles();
    {
        std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
        g_proxy_ocr_load_recipe = ProxyOcrLoadRecipe{};
        ++g_proxy_ocr_load_generation;
    }
    if (!first_detail.empty()) ai::set_last_error(first_detail);
    return finish_proxy_status(first_error, "AI_ShutdownWorker");
#else
    return AI_OK;
#endif
}

// 返回当前线程持有的错误文本。调用方不分配、不释放，也不传缓冲区长度。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_GetLastError(void) {
    return ai::last_error_c_str();
}

// 从全局引擎读取模块最近一次耗时。
AIENGINE_EXPORT int64_t AIENGINE_CALL AI_GetLastLatencyUs(int32_t module) {
#if defined(_WIN32) && defined(_M_IX86)
    if (module == AI_MODULE_OCR || module == AI_MODULE_YOLO) {
        std::vector<uint8_t> payload;
        proxy_append_i32(&payload, module);
        const std::vector<ai_worker::RuntimeFlavor> flavors{
            module == AI_MODULE_OCR
                ? proxy_ocr_flavor()
                : ai_worker::RuntimeFlavor::Core};
        for (const ai_worker::RuntimeFlavor flavor : flavors) {
            int32_t status = AI_OK;
            std::vector<uint8_t> response;
            if (proxy_request_if_running(flavor, ai_worker::CMD_GET_LAST_LATENCY, payload, &status, &response) &&
                status >= 0 && response.size() == sizeof(int64_t)) {
                int64_t latency = -1;
                std::memcpy(&latency, response.data(), sizeof(latency));
                if (latency >= 0) return latency;
            }
        }
        return -1;
    }
#endif
    const auto engine = get_engine();
    if (!engine) {
        return -1;
    }
    return engine->get_latency(module);
}

AIENGINE_EXPORT int64_t AIENGINE_CALL AI_GetOcrStageLatencyUs(int32_t stage) {
    if (stage < AI_OCR_STAGE_DETECTION || stage > AI_OCR_STAGE_POSTPROCESS) return -1;
#if defined(_WIN32) && defined(_M_IX86)
    std::vector<uint8_t> payload;
    proxy_append_i32(&payload, stage);
    int32_t status = AI_OK;
    std::vector<uint8_t> response;
    if (proxy_request_if_running(proxy_ocr_flavor(), ai_worker::CMD_GET_OCR_STAGE_LATENCY, payload, &status, &response) &&
        status >= 0 && response.size() == sizeof(int64_t)) {
        int64_t latency = -1;
        std::memcpy(&latency, response.data(), sizeof(latency));
        return latency;
    }
    return -1;
#else
    const auto engine = get_engine();
    return engine ? engine->get_ocr_stage_latency_us(stage) : -1;
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_GetRuntimeStatusJson(char* buffer, int32_t buffer_size) {
#if defined(_WIN32) && defined(_M_IX86)
    int32_t worker_status = AI_OK;
    std::vector<uint8_t> response;
    const bool requested = proxy_request_if_running(
        proxy_ocr_flavor(), ai_worker::CMD_RUNTIME_STATUS, {}, &worker_status, &response);
    if (!requested || worker_status < 0) {
        return finish_proxy_status(worker_status, "AI_GetRuntimeStatusJson");
    }
    if (!response.empty()) {
        std::string json;
        if (proxy_read_string(response, &json)) {
            {
                std::lock_guard<std::mutex> lock(g_proxy_ocr_mutex);
                proxy_rewrite_runtime_status(
                    &json,
                    g_proxy_ocr_requested_device,
                    g_proxy_ocr_active_device,
                    g_proxy_ocr_fallback_reason);
            }
            return write_string_result(json, buffer, buffer_size, AI_OK, "AI_GetRuntimeStatusJson");
        }
        return finish_status_with_detail(AI_ERR_RUNTIME, "AI_GetRuntimeStatusJson", "worker returned an invalid runtime status payload");
    }
#endif
    const ai::RuntimeStatus status = ai::get_runtime_status();
    const std::string json = std::string("{") + runtime_status_fields(status) + "}";
    return write_string_result(json, buffer, buffer_size, AI_OK, "AI_GetRuntimeStatusJson");
}

// 判断当前 DLL 是否已经编译进默认 YOLO/OCR 模型资源。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_HasEmbeddedAssets(void) {
    const bool ok = ai::has_embedded_assets();
    ai::set_last_error(ok ? "" : "AI_HasEmbeddedAssets: embedded assets are not available");
    return ok ? 1 : 0;
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloCreate(int32_t* out_handle) {
    return YOLO_Create(out_handle);
}

// 从 DLL 内置资源加载默认 YOLO 模型，不再需要外部模型路径和配置文件。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadEmbeddedModel(
    int32_t handle,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
    ai::EmbeddedAsset model_asset;
    ai::EmbeddedAsset labels_asset;
    if (!ai::get_embedded_asset(ai::EmbeddedAssetId::YoloModel, &model_asset) ||
        !ai::get_embedded_asset(ai::EmbeddedAssetId::YoloLabels, &labels_asset)) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_YoloLoadEmbeddedModel", "DLL does not contain embedded YOLO assets");
    }

    return YOLO_LoadModelFromMemory(
        handle,
        model_asset.data,
        static_cast<int32_t>(model_asset.size),
        labels_asset.data,
        static_cast<int32_t>(labels_asset.size),
        input_size,
        runtime_device,
        device_id,
        session_count);
}

// 从 DLL 内置资源加载默认 PP-OCR 识别模型，不再需要外部模型路径和配置文件。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadEmbeddedModels(int32_t runtime_device) {
    return OCR_LoadEmbeddedModel(runtime_device, 1);
}

AIENGINE_EXPORT const char* AIENGINE_CALL AI_YoloDetect(
    int32_t handle,
    const AIImage* image,
    float conf) {
    return AI_YoloInfer(handle, image, conf);
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadModel(
    int32_t handle,
    const char* model_path,
    const char* config_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
#if defined(_WIN32) && defined(_M_IX86)
    ai::Config config;
    std::string error;
    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModel", "config_path must be valid UTF-8");
    }
    if (!config.load_file(config_path, &error)) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_YoloLoadModel", error);
    }
    YoloRuntimeParams params;
    const int32_t params_status = make_yolo_runtime_params(
        input_size, runtime_device, device_id, session_count, &params, &error);
    if (params_status < 0) return finish_status_with_detail(params_status, "AI_YoloLoadModel", error);
    const std::filesystem::path model_resolved = resolve_utf8_api_path(model_path);
    const std::string labels_value = config.get_string("yolo.labels_path", "");
    const std::filesystem::path labels_resolved = resolve_utf8_api_path(labels_value.c_str());
    if (model_resolved.empty() || (!labels_value.empty() && labels_resolved.empty())) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModel", "model or labels path must be valid UTF-8");
    }
    const std::string model_utf8 = model_resolved.u8string();
    const std::string labels_utf8 = labels_resolved.empty() ? std::string() : labels_resolved.u8string();
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_load_yolo_candidates(
        handle,
        runtime_device,
        ai_worker::CMD_YOLO_LOAD_PATH,
        [&](const ProxyYoloIdentity& identity, const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_yolo_identity(&request, identity);
            proxy_append_string(&request, model_utf8.c_str());
            proxy_append_string(&request, labels_utf8.c_str());
            proxy_append_yolo_params(&request, input_size, candidate.device, device_id, session_count);
            return request;
        },
        &status,
        &response);
    return finish_proxy_status(status, "AI_YoloLoadModel");
#else
    std::string error;
    YoloRuntimeParams params;
    const int32_t params_status = make_yolo_runtime_params(
        input_size, runtime_device, device_id, session_count, &params, &error);
    if (params_status < 0) return finish_status_with_detail(params_status, "AI_YoloLoadModel", error);
    ai::Config config;
    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModel", "config_path must be valid UTF-8");
    }
    if (!config.load_file(config_path, &error)) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_YoloLoadModel", error);
    }
    std::shared_ptr<std::vector<uint8_t>> model;
    if (config.get_string("yolo.backend", "onnxruntime") == "mock" && (model_path == nullptr || model_path[0] == '\0')) {
        model = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(1), static_cast<uint8_t>(1));
    } else if (!read_binary_file(model_path, &model, &error)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModel", error);
    }
    const int32_t status = load_local_yolo_context(handle, model, std::move(config), &params, &error);
    return finish_status_with_detail(status, "AI_YoloLoadModel", error);
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloLoadModelFromMemory(
    int32_t handle,
    const void* model_data,
    int32_t model_size,
    const char* config_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
    if (model_data == nullptr || model_size <= 0) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModelFromMemory");
    }
    ai::Config config;
    std::string error;
    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_YoloLoadModelFromMemory", "config_path must be valid UTF-8");
    }
    if (!config.load_file(config_path, &error)) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_YoloLoadModelFromMemory", error);
    }
#if defined(_WIN32) && defined(_M_IX86)
    const std::string labels = read_text_file(config.get_string("yolo.labels_path", "").c_str());
    return YOLO_LoadModelFromMemory(
        handle, model_data, model_size, labels.data(), static_cast<int32_t>(labels.size()),
        input_size, runtime_device, device_id, session_count);
#else
    YoloRuntimeParams params;
    const int32_t params_status = make_yolo_runtime_params(
        input_size, runtime_device, device_id, session_count, &params, &error);
    if (params_status < 0) return finish_status_with_detail(params_status, "AI_YoloLoadModelFromMemory", error);
    auto model = std::make_shared<std::vector<uint8_t>>(
        static_cast<const uint8_t*>(model_data), static_cast<const uint8_t*>(model_data) + model_size);
    const int32_t status = load_local_yolo_context(handle, model, std::move(config), &params, &error);
    return finish_status_with_detail(status, "AI_YoloLoadModelFromMemory", error);
#endif
}

AIENGINE_EXPORT const char* AIENGINE_CALL AI_YoloInfer(
    int32_t handle,
    const AIImage* image,
    float conf) {
    g_ai_yolo_json_result.clear();
    if (image == nullptr || conf < 0.0f || conf > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "AI_YoloInfer");
        return g_ai_yolo_json_result.c_str();
    }
#if defined(_WIN32) && defined(_M_IX86)
    finish_status_with_detail(AI_ERR_CONFIG, "AI_YoloInfer", "raw AIImage YOLO inference is available in the x64 DLL; x86 callers must use YOLO_InferJson");
    return g_ai_yolo_json_result.c_str();
#else
    int32_t lookup_status = AI_OK;
    const auto pool = get_loaded_yolo_pool(handle, &lookup_status);
    if (!pool) {
        finish_status(lookup_status, "AI_YoloInfer");
        return g_ai_yolo_json_result.c_str();
    }
    std::vector<AIDetectBox> results;
    const int32_t status = pool->detect(*image, conf, &results);
    if (status < 0 || status != static_cast<int32_t>(results.size())) {
        finish_status(status < 0 ? status : AI_ERR_RUNTIME, "AI_YoloInfer");
        return g_ai_yolo_json_result.c_str();
    }
    g_ai_yolo_json_result = format_yolo_json(results, status);
    finish_status(status, "AI_YoloInfer");
    return g_ai_yolo_json_result.c_str();
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL AI_YoloRelease(int32_t handle) {
    return YOLO_Release(handle);
}

// 执行 OCR，并将全部文本行直接序列化为 UTF-8 JSON 数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_OcrRecognize(const AIImage* image, float min_confidence) {
    g_ai_ocr_json_result.clear();
    const auto engine = get_engine();
    if (!engine) {
        finish_status(AI_ERR_NOT_INITIALIZED, "AI_OcrRecognize");
        return g_ai_ocr_json_result.c_str();
    }
    if (image == nullptr || min_confidence < 0.0f || min_confidence > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrRecognize");
        return g_ai_ocr_json_result.c_str();
    }

    std::vector<AIOcrLine> results;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = recognize_ocr_all(engine, *image, &results);
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    if (status < 0) {
        finish_status(status, "AI_OcrRecognize");
        return g_ai_ocr_json_result.c_str();
    }
    status = filter_ocr_lines(&results, min_confidence);
    if (status < 0) {
        finish_status(status, "AI_OcrRecognize");
        return g_ai_ocr_json_result.c_str();
    }
    g_ai_ocr_json_result = format_ocr_lines_json_array(results, status);
    finish_status(status, "AI_OcrRecognize");
    return g_ai_ocr_json_result.c_str();
}

// 基于路径加载 OCR 模型；纯识别模式下 det_model_path 可以为空。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadModels(const char* det_model_path, const char* rec_model_path, const char* config_path, int32_t runtime_device) {
    std::string error;
    const auto engine = ensure_engine(nullptr, runtime_device, &error);
    if (!engine) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_OcrLoadModels", error);
    }

    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_OcrLoadModels", "config_path must be valid UTF-8");
    }
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    const std::filesystem::path det_resolved = resolve_utf8_api_path(det_model_path);
    const std::filesystem::path rec_resolved = resolve_utf8_api_path(rec_model_path);
    if ((det_model_path != nullptr && det_model_path[0] != '\0' && det_resolved.empty()) ||
        (rec_model_path != nullptr && rec_model_path[0] != '\0' && rec_resolved.empty())) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrLoadModels");
    }
    const std::string det_utf8 = det_resolved.empty() ? std::string() : det_resolved.u8string();
    const std::string rec_utf8 = rec_resolved.empty() ? std::string() : rec_resolved.u8string();
    {
        ai::ScopedLatency timer(&latency);
        status = engine->ocr_load_models(
            det_utf8.empty() ? nullptr : det_utf8.c_str(),
            rec_utf8.empty() ? nullptr : rec_utf8.c_str(),
            config_path, runtime_device, &error);
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    return finish_status_with_detail(status, "AI_OcrLoadModels", error);
}

// 基于内存加载 OCR 模型；识别模型必填，检测模型可选。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrLoadModelsFromMemory(
    const void* det_model_data,
    int32_t det_model_size,
    const void* rec_model_data,
    int32_t rec_model_size,
    const char* config_path,
    int32_t runtime_device) {
    std::string error;
    const auto engine = ensure_engine(nullptr, runtime_device, &error);
    if (!engine) {
        return finish_status_with_detail(AI_ERR_CONFIG, "AI_OcrLoadModelsFromMemory", error);
    }

    std::string config_utf8;
    if (config_path != nullptr && config_path[0] != '\0' && !copy_valid_utf8(config_path, &config_utf8)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "AI_OcrLoadModelsFromMemory", "config_path must be valid UTF-8");
    }
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = engine->ocr_load_models_from_memory(
            det_model_data,
            det_model_size,
            rec_model_data,
            rec_model_size,
            config_path,
            runtime_device,
            &error);
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    return finish_status_with_detail(status, "AI_OcrLoadModelsFromMemory", error);
}

// 释放 OCR 模型/session 状态。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRelease(void) {
#if defined(_WIN32) && defined(_M_IX86)
    std::lock_guard<std::mutex> load_lock(g_proxy_ocr_load_mutex);
    int32_t worker_status = AI_OK;
    std::vector<uint8_t> response;
    proxy_request_if_running(proxy_ocr_flavor(), ai_worker::CMD_OCR_RELEASE, {}, &worker_status, &response);
    {
        std::lock_guard<std::mutex> state_lock(g_proxy_ocr_mutex);
        g_proxy_ocr_load_recipe = ProxyOcrLoadRecipe{};
        ++g_proxy_ocr_load_generation;
    }
    const auto engine = get_engine();
    const int32_t local_status = engine ? engine->ocr_release() : AI_OK;
    return worker_status < 0
        ? finish_proxy_status(worker_status, "AI_OcrRelease")
        : finish_status(local_status, "AI_OcrRelease");
#else
    const auto engine = get_engine();
    if (!engine) {
        return finish_status(AI_ERR_NOT_INITIALIZED, "AI_OcrRelease");
    }

    const int32_t status = engine->ocr_release();
    return finish_status(status, "AI_OcrRelease");
#endif
}

// 识别一行，并向 output 写入文本或 JSON。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRecognizeLine(const AIImage* image, int32_t output_format, float min_confidence, char* output, int32_t output_size) {
    const auto engine = get_engine();
    if (!engine) {
        return finish_status(AI_ERR_NOT_INITIALIZED, "AI_OcrRecognizeLine");
    }
    if (image == nullptr || min_confidence < 0.0f || min_confidence > 1.0f) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrRecognizeLine");
    }
    if (output_format != AI_OCR_OUTPUT_TEXT && output_format != AI_OCR_OUTPUT_JSON) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrRecognizeLine");
    }

    std::vector<AIOcrLine> lines(64);
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = engine->ocr_recognize(*image, lines.data(), static_cast<int32_t>(lines.size()));
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    if (status < 0) {
        return finish_status(status, "AI_OcrRecognizeLine");
    }

    status = filter_ocr_lines(lines.data(), status, min_confidence);
    if (status < 0) return finish_status(status, "AI_OcrRecognizeLine");
    const std::string formatted = format_ocr_output(lines, status, output_format, true);
    return write_string_result(formatted, output, output_size, status > 0 ? 1 : 0, "AI_OcrRecognizeLine");
}

// 识别全部行，并写入换行分隔文本或 JSON。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_OcrRecognizeLines(const AIImage* image, int32_t output_format, float min_confidence, char* output, int32_t output_size) {
    const auto engine = get_engine();
    if (!engine) {
        return finish_status(AI_ERR_NOT_INITIALIZED, "AI_OcrRecognizeLines");
    }
    if (image == nullptr || min_confidence < 0.0f || min_confidence > 1.0f) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrRecognizeLines");
    }
    if (output_format != AI_OCR_OUTPUT_TEXT && output_format != AI_OCR_OUTPUT_JSON) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrRecognizeLines");
    }

    std::vector<AIOcrLine> lines(64);
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = engine->ocr_recognize(*image, lines.data(), static_cast<int32_t>(lines.size()));
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    if (status < 0) {
        return finish_status(status, "AI_OcrRecognizeLines");
    }

    status = filter_ocr_lines(lines.data(), status, min_confidence);
    if (status < 0) return finish_status(status, "AI_OcrRecognizeLines");
    const std::string formatted = format_ocr_output(lines, status, output_format, false);
    return write_string_result(formatted, output, output_size, status, "AI_OcrRecognizeLines");
}

// 在识别文本中搜索目标字符串，并将全部匹配直接序列化为 UTF-8 JSON 数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_OcrFindText(const AIImage* image, const char* target_utf8, float min_confidence) {
    g_ai_ocr_json_result.clear();
    const auto engine = get_engine();
    if (!engine) {
        finish_status(AI_ERR_NOT_INITIALIZED, "AI_OcrFindText");
        return g_ai_ocr_json_result.c_str();
    }
    std::string validated_target;
    if (image == nullptr || target_utf8 == nullptr || target_utf8[0] == '\0' ||
        !copy_valid_utf8(target_utf8, &validated_target) ||
        min_confidence < 0.0f || min_confidence > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "AI_OcrFindText");
        return g_ai_ocr_json_result.c_str();
    }

    std::vector<AIOcrLine> results;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = find_ocr_text_all(engine, *image, validated_target.c_str(), min_confidence, &results);
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    if (status < 0) {
        finish_status(status, "AI_OcrFindText");
        return g_ai_ocr_json_result.c_str();
    }
    g_ai_ocr_json_result = format_ocr_lines_json_array(results, status);
    finish_status(status, "AI_OcrFindText");
    return g_ai_ocr_json_result.c_str();
}

// 无状态 CV 初始化入口。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvInit(void) {
    ai::set_last_error("");
    return AI_OK;
}

// 将图像/ROI 转为 GRAY8。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvToGray(const AIImage* image, const AIRect* roi, uint8_t* output, int32_t output_stride) {
    if (image == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvToGray");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_to_gray(*image, roi, output, output_stride);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvToGray");
}

// 使用固定阈值将图像/ROI 转为二值掩码。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvThreshold(const AIImage* image, const AIRect* roi, int32_t threshold, uint8_t* output, int32_t output_stride) {
    if (image == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvThreshold");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_threshold(*image, roi, threshold, output, output_stride);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvThreshold");
}

// 提取前景笔画中心线并写入 JSON。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvExtractTraceJson(
    const AIImage* image,
    const AIRect* roi,
    int32_t threshold,
    int32_t invert,
    int32_t max_points,
    char* output,
    int32_t output_size) {
    if (image == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvExtractTraceJson");
    }

    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    std::string formatted;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_extract_trace_json(*image, roi, threshold, invert != 0, max_points, &formatted);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    if (status < 0) {
        return finish_status(status, "AI_CvExtractTraceJson");
    }

    return write_string_result(formatted, output, output_size, status, "AI_CvExtractTraceJson");
}

// 计算 ROI 内 B/G/R 均值和灰度范围。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvMeanColor(const AIImage* image, const AIRect* roi, AIColorStats* output) {
    if (image == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvMeanColor");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_mean_color(*image, roi, output);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvMeanColor");
}

// 按通道容差查找接近 target_bgr 的像素。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindColor(const AIImage* image, const AIRect* roi, uint32_t target_bgr, int32_t tolerance, AIColorFindResult* output) {
    if (image == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvFindColor");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_find_color(*image, roi, target_bgr, tolerance, output);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvFindColor");
}

// 查找最佳非透明模板匹配。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindImage(const AIImage* image, const AIImage* templ, float min_score, AIImageMatch* output) {
    if (image == nullptr || templ == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvFindImage");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_find_image(*image, *templ, min_score, output);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvFindImage");
}

// 查找多个非透明模板匹配并返回 UTF-8 JSON 数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_CvFindImages(
    const AIImage* image,
    const AIImage* templates,
    int32_t template_count,
    float min_score) {
    g_ai_cv_json_result.clear();
    if (image == nullptr) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvFindImages");
        return g_ai_cv_json_result.c_str();
    }
    const auto engine = get_engine();
    std::vector<AIImageMatch> results;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = cv_find_images_all(*image, templates, template_count, 255, min_score, false, &results);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    if (status < 0) {
        finish_status(status, "AI_CvFindImages");
        return g_ai_cv_json_result.c_str();
    }
    g_ai_cv_json_result = format_ai_cv_json(results);
    finish_status(status, "AI_CvFindImages");
    return g_ai_cv_json_result.c_str();
}

// 查找最佳透明模板匹配。
AIENGINE_EXPORT int32_t AIENGINE_CALL AI_CvFindTransparentImage(
    const AIImage* image,
    const AIImage* templ,
    int32_t alpha_threshold,
    float min_score,
    AIImageMatch* output) {
    if (image == nullptr || templ == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvFindTransparentImage");
    }
    const auto engine = get_engine();
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = ai::cv_find_transparent_image(*image, *templ, alpha_threshold, min_score, output);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    return finish_status(status, "AI_CvFindTransparentImage");
}

// 查找多个透明模板匹配并返回 UTF-8 JSON 数组。
AIENGINE_EXPORT const char* AIENGINE_CALL AI_CvFindTransparentImages(
    const AIImage* image,
    const AIImage* templates,
    int32_t template_count,
    int32_t alpha_threshold,
    float min_score) {
    g_ai_cv_json_result.clear();
    if (image == nullptr) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "AI_CvFindTransparentImages");
        return g_ai_cv_json_result.c_str();
    }
    const auto engine = get_engine();
    std::vector<AIImageMatch> results;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = cv_find_images_all(*image, templates, template_count, alpha_threshold, min_score, true, &results);
    }
    if (engine) {
        engine->set_latency(AI_MODULE_CV, latency);
    }
    if (status < 0) {
        finish_status(status, "AI_CvFindTransparentImages");
        return g_ai_cv_json_result.c_str();
    }
    g_ai_cv_json_result = format_ai_cv_json(results);
    finish_status(status, "AI_CvFindTransparentImages");
    return g_ai_cv_json_result.c_str();
}

#if defined(AIENGINE_CV_TEST_HOOKS)
CV_TEST_EXPORT void CVTest_LastTimes(double* values) { std::copy(std::begin(g_cv_times),std::end(g_cv_times),values); }
CV_TEST_EXPORT void CVTest_TransparentProfile(CVTestTransparentProfile* output) {
    if (output != nullptr) *output = g_cv_transparent_profile;
}
CV_TEST_EXPORT void CVTest_Reference(int enabled) { g_cv_reference=enabled!=0; }
CV_TEST_EXPORT void CVTest_ForceScalarMasked(int enabled) { g_cv_force_scalar_masked=enabled!=0; }
CV_TEST_EXPORT void CVTest_Fault(int stage, int kind) {
    g_cv_fault_stage = stage; g_cv_fault_kind = kind;
}
CV_TEST_EXPORT void CVTest_Pause(int enabled) {
    if (enabled) g_cv_paused = false;
    g_cv_pause = enabled != 0;
}
CV_TEST_EXPORT int CVTest_Paused() { return g_cv_paused.load() ? 1 : 0; }
CV_TEST_EXPORT CVTestStats CVTest_Stats() {
#if defined(AIENGINE_WITH_OPENCV)
    std::lock_guard<std::mutex> lock(g_cv_pool.mutex);
    return {kCvWorkspaceCount, kCvIdleBudget, g_cv_pool.active, g_cv_pool.peak_active,
            g_cv_pool.idle_bytes, g_cv_pool.discarded, g_cv_live_templates.load()};
#else
    return {0, 0, 0, 0, 0, 0, g_cv_live_templates.load()};
#endif
}
CV_TEST_EXPORT void CVTest_OverBudget() {
#if defined(AIENGINE_WITH_OPENCV)
    CVWorkspaceLease lease;
    lease.get().color.pixels.resize(kCvIdleBudget + 1);
#endif
}
#endif

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_Create(int32_t* out_handle) try {
    if (out_handle != nullptr) *out_handle = 0;
    if (out_handle == nullptr) return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_Create");
    std::lock_guard<std::mutex> lock(g_cv_context_mutex);
    if (g_next_cv_handle <= 0 || g_next_cv_handle == std::numeric_limits<int32_t>::max()) {
        return finish_status(AI_ERR_RUNTIME, "CV_Create");
    }
    const int32_t handle = g_next_cv_handle++;
    g_cv_contexts.emplace(handle, std::make_shared<CVContext>());
    *out_handle = handle;
    return finish_status(AI_OK, "CV_Create");
}
catch (...) {
    if (out_handle != nullptr) *out_handle = 0;
    return cv_exception_status("CV_Create");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_LoadTemplateDir(int32_t handle, const char* dir_path, int32_t recursive) try {
    const auto context = get_cv_context(handle);
    if (!context || dir_path == nullptr || dir_path[0] == '\0') {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_LoadTemplateDir");
    }

    const std::filesystem::path root = resolve_compat_api_path(dir_path);
    std::error_code path_error;
    if (root.empty() || !std::filesystem::is_directory(root, path_error) || path_error) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_LoadTemplateDir", "template directory not found");
    }

    TemplateMap next;
    auto load_one = [&](const std::filesystem::directory_entry& entry) {
        if (!entry.is_regular_file()) {
            return;
        }
        const std::filesystem::path path = entry.path();
        const std::string ext = path.extension().u8string();
        if (ext != ".bmp" && ext != ".BMP") {
            return;
        }
        std::shared_ptr<TemplateEntry> templ = load_template_file(path);
        if (templ) {
            next[templ->key] = templ;
        }
    };

    try {
        if (recursive != 0) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
                load_one(entry);
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                load_one(entry);
            }
        }
    } catch (...) {
        throw;
    }

    const int32_t count = static_cast<int32_t>(next.size());
    {
        std::unique_lock<std::shared_mutex> lock(context->template_mutex);
        context->templates = std::move(next);
    }
    return finish_status(count, "CV_LoadTemplateDir");
}
catch (...) {
    return cv_exception_status("CV_LoadTemplateDir");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_LoadTemplateZipFromMemory(
    int32_t handle,
    const uint8_t* zip_data,
    int32_t zip_size) try {
    const auto context = get_cv_context(handle);
    if (!context || zip_data == nullptr || zip_size <= 0) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_LoadTemplateZipFromMemory");
    }
#if !defined(AIENGINE_WITH_OPENCV)
    return finish_status_with_detail(
        AI_ERR_BACKEND_NOT_CONFIGURED,
        "CV_LoadTemplateZipFromMemory",
        "memory ZIP template loading requires the static OpenCV build");
#else
    std::vector<ai::MemoryZipEntry> archive_entries;
    std::string zip_error;
    if (!ai::extract_memory_zip(zip_data, zip_size, &archive_entries, &zip_error)) {
        return finish_status_with_detail(
            AI_ERR_INVALID_ARGUMENT, "CV_LoadTemplateZipFromMemory", zip_error);
    }

    TemplateMap next;
    std::unordered_map<std::string, std::string> canonical_names;
    for (const ai::MemoryZipEntry& archive_entry : archive_entries) {
        std::string key;
        if (!decode_zip_template_key(archive_entry, &key)) {
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_LoadTemplateZipFromMemory",
                "ZIP contains an invalid template file name");
        }
        std::string extension;
        try {
            extension = ascii_lower(std::filesystem::u8path(key).extension().u8string());
        } catch (const std::bad_alloc&) {
            throw;
        } catch (...) {
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_LoadTemplateZipFromMemory",
                "ZIP contains an invalid template path");
        }
        if (extension != ".bmp") continue;

        const std::string canonical = ascii_lower(key);
        const auto inserted = canonical_names.emplace(canonical, key);
        if (!inserted.second) {
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_LoadTemplateZipFromMemory",
                std::string("duplicate BMP file name in ZIP: ") + key);
        }
        if (archive_entry.data.empty() ||
            archive_entry.data.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return finish_status_with_detail(
                AI_ERR_IMAGE_FORMAT,
                "CV_LoadTemplateZipFromMemory",
                std::string("invalid BMP template in ZIP: ") + key);
        }
        std::shared_ptr<TemplateEntry> templ = load_template_memory(
            key,
            archive_entry.data.data(),
            static_cast<int32_t>(archive_entry.data.size()));
        if (!templ) {
            return finish_status_with_detail(
                AI_ERR_IMAGE_FORMAT,
                "CV_LoadTemplateZipFromMemory",
                std::string("invalid BMP template in ZIP: ") + key);
        }
        next.emplace(key, std::move(templ));
    }
    if (next.empty()) {
        return finish_status_with_detail(
            AI_ERR_INVALID_ARGUMENT,
            "CV_LoadTemplateZipFromMemory",
            "ZIP does not contain any BMP templates");
    }

    const int32_t count = static_cast<int32_t>(next.size());
    {
        std::unique_lock<std::shared_mutex> lock(context->template_mutex);
        context->templates = std::move(next);
    }
    return finish_status(count, "CV_LoadTemplateZipFromMemory");
#endif
}
catch (...) {
    return cv_exception_status("CV_LoadTemplateZipFromMemory");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_ClearTemplateCache(int32_t handle) try {
    const auto context = get_cv_context(handle);
    if (!context) return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_ClearTemplateCache");
    std::unique_lock<std::shared_mutex> lock(context->template_mutex);
    context->templates.clear();
    return finish_status(AI_OK, "CV_ClearTemplateCache");
}
catch (...) {
    return cv_exception_status("CV_ClearTemplateCache");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_Release(int32_t handle) try {
    std::lock_guard<std::mutex> lock(g_cv_context_mutex);
    const auto it = g_cv_contexts.find(handle);
    if (it == g_cv_contexts.end()) return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_Release");
    g_cv_contexts.erase(it);
    return finish_status(AI_OK, "CV_Release");
}
catch (...) {
    return cv_exception_status("CV_Release");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_FindOne(
    int32_t handle,
    const char* template_name,
    const uint8_t* big_data,
    int32_t big_size,
    float min_score,
    int32_t match_mode,
    CVMatchResult* output,
    int32_t origin_x,
    int32_t origin_y) try {
    if (output != nullptr) *output = CVMatchResult{};
    const auto context = get_cv_context(handle);
    if (const char* reason = cv_find_argument_error(context, match_mode, min_score)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindOne", reason);
    }
    if (template_name == nullptr || template_name[0] == '\0' || output == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_FindOne");
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        return finish_status(AI_ERR_IMAGE_FORMAT, "CV_FindOne");
    }
    const auto templ = find_template(context, template_name);
    if (!templ) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindOne", std::string("template not found: ") + template_name);
    }
    const std::vector<std::shared_ptr<const TemplateEntry>> templates{templ};
    const CVColorBias no_bias{};
    std::vector<CVMatchResult> matches;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = find_cv_matches_opencv(big.image, templates, match_mode, false, 0, no_bias,
            min_score, kCvInternalNmsIou, true, &matches);
    }
    const auto engine = get_engine();
    if (engine) engine->set_latency(AI_MODULE_CV, latency);
    if (status > 0) {
        CVMatchResult shifted = matches.front();
        if (!offset_cv_result(&shifted, origin_x, origin_y)) {
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT, "CV_FindOne", "coordinate origin causes int32 overflow");
        }
        *output = shifted;
    }
    return finish_status(status, "CV_FindOne");
}
catch (...) {
    if (output != nullptr) *output = CVMatchResult{};
    return cv_exception_status("CV_FindOne");
}

AIENGINE_EXPORT int32_t AIENGINE_CALL CV_FindTransparentOne(
    int32_t handle,
    const char* template_name,
    const uint8_t* big_data,
    int32_t big_size,
    float min_score,
    int32_t match_mode,
    const char* transparent_rgb,
    CVMatchResult* output,
    int32_t origin_x,
    int32_t origin_y) try {
    if (output != nullptr) *output = CVMatchResult{};
    const auto context = get_cv_context(handle);
    if (const char* reason = cv_find_argument_error(context, match_mode, min_score)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentOne", reason);
    }
    uint32_t rgb = 0;
    if (template_name == nullptr || template_name[0] == '\0' || output == nullptr || !parse_rgb_hex(transparent_rgb, &rgb)) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentOne");
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        return finish_status(AI_ERR_IMAGE_FORMAT, "CV_FindTransparentOne");
    }
    const auto templ = find_template(context, template_name);
    if (!templ) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentOne", std::string("template not found: ") + template_name);
    }
    const std::vector<std::shared_ptr<const TemplateEntry>> templates{templ};
    const CVColorBias no_bias{};
    std::vector<CVMatchResult> matches;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = find_cv_matches_opencv(big.image, templates, match_mode, true, rgb, no_bias,
            min_score, kCvInternalNmsIou, true, &matches);
    }
    const auto engine = get_engine();
    if (engine) engine->set_latency(AI_MODULE_CV, latency);
    if (status > 0) {
        CVMatchResult shifted = matches.front();
        if (!offset_cv_result(&shifted, origin_x, origin_y)) {
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_FindTransparentOne",
                "coordinate origin causes int32 overflow");
        }
        *output = shifted;
    }
    return finish_status(status, "CV_FindTransparentOne");
}
catch (...) {
    if (output != nullptr) *output = CVMatchResult{};
    return cv_exception_status("CV_FindTransparentOne");
}

AIENGINE_EXPORT const char* AIENGINE_CALL CV_FindMultiText(
    int32_t handle,
    const char* template_names,
    const uint8_t* big_data,
    int32_t big_size,
    const char* color_bias,
    float min_score,
    int32_t match_mode,
    int32_t origin_x,
    int32_t origin_y) try {
    g_compat_cv_json_result.clear();
    const auto context = get_cv_context(handle);
    if (const char* reason = cv_find_argument_error(context, match_mode, min_score)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindMultiText", reason);
        return g_compat_cv_json_result.c_str();
    }
    CVColorBias bias{};
    if (!parse_cv_color_bias(color_bias, match_mode, &bias)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindMultiText", "invalid color_bias: expected hexadecimal text (color=6 digits, gray=2 digits), or empty");
        return g_compat_cv_json_result.c_str();
    }
    const auto name_candidates = compat_pipe_text_candidates(template_names);
    if (name_candidates.empty()) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindMultiText", "invalid template_names: empty or malformed pipe-separated list");
        return g_compat_cv_json_result.c_str();
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        finish_status(AI_ERR_IMAGE_FORMAT, "CV_FindMultiText");
        return g_compat_cv_json_result.c_str();
    }
    std::vector<std::shared_ptr<const TemplateEntry>> template_entries;
    template_entries.reserve(name_candidates.size());
    for (const auto& candidates : name_candidates) {
        const auto templ = find_template_candidates(context, candidates, nullptr);
        if (!templ) {
            finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindMultiText", std::string("template not found: ") + candidates.front());
            return g_compat_cv_json_result.c_str();
        }
        template_entries.push_back(templ);
    }
    std::vector<CVMatchResult> matches;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = find_cv_matches_multi(big.image, template_entries, match_mode, false, 0, bias,
            min_score, kCvInternalNmsIou, false, &matches);
    }
    const auto engine = get_engine();
    if (engine) engine->set_latency(AI_MODULE_CV, latency);
    if (status < 0) {
        finish_status(status, "CV_FindMultiText");
        return g_compat_cv_json_result.c_str();
    }
    for (CVMatchResult& match : matches) {
        if (!offset_cv_result(&match, origin_x, origin_y)) {
            finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_FindMultiText",
                "coordinate origin causes int32 overflow");
            return g_compat_cv_json_result.c_str();
        }
    }
    cv_test_fault(3);
    const std::string compact_text = format_cv_multi_text(matches);
    if (!utf8_to_windows_acp(compact_text, &g_compat_cv_json_result)) {
        g_compat_cv_json_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "CV_FindMultiText", "failed to convert compact result text to the current Windows ANSI code page");
        return g_compat_cv_json_result.c_str();
    }
    finish_status(status, "CV_FindMultiText");
    return g_compat_cv_json_result.c_str();
}
catch (...) {
    g_compat_cv_json_result.clear();
    cv_exception_status("CV_FindMultiText");
    return "";
}

AIENGINE_EXPORT const char* AIENGINE_CALL CV_FindTransparentMultiText(
    int32_t handle,
    const char* template_names,
    const uint8_t* big_data,
    int32_t big_size,
    const char* color_bias,
    float min_score,
    const char* transparent_rgb,
    int32_t origin_x,
    int32_t origin_y) try {
    g_compat_cv_json_result.clear();
    const auto context = get_cv_context(handle);
    if (const char* reason = cv_find_argument_error(context, 0, min_score)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentMultiText", reason);
        return g_compat_cv_json_result.c_str();
    }
    CVColorBias bias{};
    if (!parse_cv_color_bias(color_bias, 0, &bias)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentMultiText", "invalid color_bias: expected hexadecimal text (color=6 digits, gray=2 digits), or empty");
        return g_compat_cv_json_result.c_str();
    }
    uint32_t rgb = 0;
    if (!parse_rgb_hex(transparent_rgb, &rgb)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentMultiText", "invalid transparent_rgb: expected 6 hexadecimal digits");
        return g_compat_cv_json_result.c_str();
    }
    const auto name_candidates = compat_pipe_text_candidates(template_names);
    if (name_candidates.empty()) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentMultiText", "invalid template_names: empty or malformed pipe-separated list");
        return g_compat_cv_json_result.c_str();
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        finish_status(AI_ERR_IMAGE_FORMAT, "CV_FindTransparentMultiText");
        return g_compat_cv_json_result.c_str();
    }
    std::vector<std::shared_ptr<const TemplateEntry>> template_entries;
    template_entries.reserve(name_candidates.size());
    for (const auto& candidates : name_candidates) {
        const auto templ = find_template_candidates(context, candidates, nullptr);
        if (!templ) {
            finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "CV_FindTransparentMultiText", std::string("template not found: ") + candidates.front());
            return g_compat_cv_json_result.c_str();
        }
        template_entries.push_back(templ);
    }
    std::vector<CVMatchResult> matches;
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = find_cv_matches_multi(big.image, template_entries, 0, true, rgb, bias,
            min_score, kCvInternalNmsIou, false, &matches);
    }
    const auto engine = get_engine();
    if (engine) engine->set_latency(AI_MODULE_CV, latency);
    if (status < 0) {
        finish_status(status, "CV_FindTransparentMultiText");
        return g_compat_cv_json_result.c_str();
    }
    for (CVMatchResult& match : matches) {
        if (!offset_cv_result(&match, origin_x, origin_y)) {
            finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "CV_FindTransparentMultiText",
                "coordinate origin causes int32 overflow");
            return g_compat_cv_json_result.c_str();
        }
    }
    cv_test_fault(3);
    const std::string compact_text = format_cv_multi_text(matches);
    if (!utf8_to_windows_acp(compact_text, &g_compat_cv_json_result)) {
        g_compat_cv_json_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "CV_FindTransparentMultiText", "failed to convert compact result text to the current Windows ANSI code page");
        return g_compat_cv_json_result.c_str();
    }
    finish_status(status, "CV_FindTransparentMultiText");
    return g_compat_cv_json_result.c_str();
}
catch (...) {
    g_compat_cv_json_result.clear();
    cv_exception_status("CV_FindTransparentMultiText");
    return "";
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadModelFromPath(
    const char* det_path,
    const char* rec_path,
    const char* keys_path,
    int32_t device,
    int32_t session_count) {
#if defined(_WIN32) && defined(_M_IX86)
    const std::string det_abs = proxy_compat_absolute_path(det_path);
    const std::string rec_abs = proxy_compat_absolute_path(rec_path);
    const std::string keys_abs = proxy_compat_absolute_path(keys_path);
    if ((det_path != nullptr && det_path[0] != '\0' && det_abs.empty()) || rec_abs.empty() || keys_abs.empty()) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_LoadModelFromPath", "OCR model or charset path is invalid");
    }
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_load_ocr_candidates(
        device,
        ai_worker::CMD_OCR_LOAD_PATH,
        [&](const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_string(&request, det_abs.c_str());
            proxy_append_string(&request, rec_abs.c_str());
            proxy_append_string(&request, keys_abs.c_str());
            proxy_append_i32(&request, candidate.device);
            proxy_append_i32(&request, session_count);
            return request;
        },
        &status,
        &response);
    return finish_proxy_status(status, "OCR_LoadModelFromPath");
#else
    std::string error;
    const auto engine = ensure_engine(nullptr, device, &error);
    if (!engine) {
        return finish_status_with_detail(AI_ERR_CONFIG, "OCR_LoadModelFromPath", error);
    }
    const std::filesystem::path det_resolved = resolve_compat_api_path(det_path);
    const std::filesystem::path rec_resolved = resolve_compat_api_path(rec_path);
    const std::filesystem::path keys_resolved = resolve_compat_api_path(keys_path);
    if ((det_path != nullptr && det_path[0] != '\0' && det_resolved.empty()) || rec_resolved.empty() || keys_resolved.empty()) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_LoadModelFromPath", "OCR model or charset path is invalid");
    }
    const std::string keys_utf8 = keys_resolved.u8string();
    ai::Config config = make_ocr_api_config(keys_utf8.c_str(), device, session_count);
    std::vector<uint8_t> det_bytes;
    std::vector<uint8_t> rec_bytes;
    const bool det_ok = det_resolved.empty() || read_file_bytes(det_resolved, &det_bytes);
    if (!det_ok || !read_file_bytes(rec_resolved, &rec_bytes)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_LoadModelFromPath", "failed to read OCR model file");
    }
    const int32_t status = engine->ocr_load_models_from_memory_with_config(
        det_bytes.data(), static_cast<int32_t>(det_bytes.size()),
        rec_bytes.data(), static_cast<int32_t>(rec_bytes.size()),
        std::move(config), device, &error);
    return finish_status_with_detail(status, "OCR_LoadModelFromPath", error);
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadModelFromMemory(
    const void* det_data,
    int32_t det_size,
    const void* rec_data,
    int32_t rec_size,
    const void* keys_data,
    int32_t keys_size,
    int32_t device,
    int32_t session_count) {
#if defined(_WIN32) && defined(_M_IX86)
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_load_ocr_candidates(
        device,
        ai_worker::CMD_OCR_LOAD_MEMORY,
        [&](const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_bytes(&request, det_data, det_size);
            proxy_append_bytes(&request, rec_data, rec_size);
            proxy_append_bytes(&request, keys_data, keys_size);
            proxy_append_i32(&request, candidate.device);
            proxy_append_i32(&request, session_count);
            return request;
        },
        &status,
        &response);
    return finish_proxy_status(status, "OCR_LoadModelFromMemory");
#else
    std::string error;
    const auto engine = ensure_engine(nullptr, device, &error);
    if (!engine) {
        return finish_status_with_detail(AI_ERR_CONFIG, "OCR_LoadModelFromMemory", error);
    }
    ai::Config config = make_ocr_api_config(nullptr, device, session_count);
    if (keys_data != nullptr && keys_size > 0) {
        config.set_string("ocr.charset_inline", std::string(static_cast<const char*>(keys_data), static_cast<size_t>(keys_size)));
    }
    const int32_t status = engine->ocr_load_models_from_memory_with_config(
        det_data, det_size, rec_data, rec_size, std::move(config), device, &error);
    return finish_status_with_detail(status, "OCR_LoadModelFromMemory", error);
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadEmbeddedModel(int32_t device, int32_t session_count) {
    return OCR_LoadEmbeddedModelEx(device, session_count, nullptr);
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_LoadEmbeddedModelEx(
    int32_t device,
    int32_t session_count,
    const AIOcrRuntimeOptions* options) {
    if (session_count <= 0) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_LoadEmbeddedModelEx");
    }
    if (options != nullptr && (options->intra_op_threads < 0 || options->det_input_width < 0 || options->det_input_height < 0 ||
        options->det_binary_threshold < 0.0f || options->det_binary_threshold > 1.0f ||
        options->det_box_score_threshold < 0.0f || options->det_box_score_threshold > 1.0f || options->det_unclip_ratio < 0.0f ||
        ((options->det_input_width == 0) != (options->det_input_height == 0)))) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_LoadEmbeddedModelEx");
    }
#if defined(_WIN32) && defined(_M_IX86)
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    const AIOcrRuntimeOptions effective_options = options == nullptr ? AIOcrRuntimeOptions{} : *options;
    proxy_load_ocr_candidates(
        device,
        ai_worker::CMD_OCR_LOAD_EMBEDDED,
        [&](const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_i32(&request, candidate.device);
            proxy_append_i32(&request, session_count);
            proxy_append_i32(&request, effective_options.det_input_width);
            proxy_append_i32(&request, effective_options.det_input_height);
            proxy_append_i32(&request, effective_options.intra_op_threads);
            proxy_append_f32(&request, effective_options.det_binary_threshold);
            proxy_append_f32(&request, effective_options.det_box_score_threshold);
            proxy_append_f32(&request, effective_options.det_unclip_ratio);
            return request;
        },
        &status,
        &response);
    std::string worker_error;
    if (status < 0) proxy_read_string(response, &worker_error);
    if (status < 0 && worker_error.empty()) {
        return finish_proxy_status(status, "OCR_LoadEmbeddedModelEx");
    }
    return finish_status_with_detail(status, "OCR_LoadEmbeddedModelEx", worker_error);
#else
    ai::EmbeddedAsset det_asset;
    ai::EmbeddedAsset rec_asset;
    ai::EmbeddedAsset charset_asset;
    if (!ai::get_embedded_asset(ai::EmbeddedAssetId::OcrDetModel, &det_asset) ||
        !ai::get_embedded_asset(ai::EmbeddedAssetId::OcrRecModel, &rec_asset) ||
        !ai::get_embedded_asset(ai::EmbeddedAssetId::OcrCharset, &charset_asset)) {
        return finish_status_with_detail(AI_ERR_CONFIG, "OCR_LoadEmbeddedModelEx", "module does not contain embedded PP-OCRv6 assets");
    }
    std::string error;
    const auto engine = ensure_engine(nullptr, device, &error);
    if (!engine) {
        return finish_status_with_detail(AI_ERR_CONFIG, "OCR_LoadEmbeddedModel", error);
    }
    ai::Config config = make_embedded_ocr_config(device, embedded_asset_text(charset_asset));
    apply_ocr_runtime_options(&config, options, session_count);
    int64_t latency = 0;
    int32_t status = AI_ERR_RUNTIME;
    {
        ai::ScopedLatency timer(&latency);
        status = engine->ocr_load_models_from_memory_with_config(
            det_asset.data,
            static_cast<int32_t>(det_asset.size),
            rec_asset.data,
            static_cast<int32_t>(rec_asset.size),
            std::move(config),
            device,
            &error);
    }
    engine->set_latency(AI_MODULE_OCR, latency);
    return finish_status_with_detail(status, "OCR_LoadEmbeddedModelEx", error);
#endif
}

AIENGINE_EXPORT const char* AIENGINE_CALL OCR_Recognize(
    const uint8_t* big_data,
    int32_t big_size,
    int32_t output_format,
    float min_confidence,
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y) {
    g_compat_ocr_result.clear();
    ai::OcrColorFilter parsed_filter;
    std::string filter_error;
    if (!ai::parse_ocr_color_filter(color_filter, &parsed_filter, &filter_error)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_Recognize", filter_error);
        return g_compat_ocr_result.c_str();
    }
#if defined(_WIN32) && defined(_M_IX86)
    std::vector<uint8_t> request;
    proxy_append_bytes(&request, big_data, big_size);
    proxy_append_i32(&request, output_format);
    proxy_append_f32(&request, min_confidence);
    proxy_append_string(&request, color_filter == nullptr ? "" : color_filter);
    proxy_append_i32(&request, origin_x);
    proxy_append_i32(&request, origin_y);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_ocr_request(
        ai_worker::CMD_OCR_RECOGNIZE, request, &status, &response);
    std::string text;
    const bool decoded = proxy_read_string(response, &text);
    if (status < 0) {
        finish_proxy_status(status, "OCR_Recognize");
        return g_compat_ocr_result.c_str();
    }
    if (!decoded) {
        finish_status_with_detail(
            AI_ERR_RUNTIME,
            "OCR_Recognize",
            "worker returned an invalid OCR text payload");
        return g_compat_ocr_result.c_str();
    }
    if (!utf8_to_windows_acp(text, &g_compat_ocr_result)) {
        g_compat_ocr_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "OCR_Recognize", "failed to convert worker UTF-8 OCR text to the current Windows ANSI code page");
        return g_compat_ocr_result.c_str();
    }
    finish_status(status, "OCR_Recognize");
    return g_compat_ocr_result.c_str();
#else
    if (output_format != AI_OCR_OUTPUT_TEXT && output_format != AI_OCR_OUTPUT_JSON ||
        min_confidence < 0.0f || min_confidence > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_Recognize");
        return g_compat_ocr_result.c_str();
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        finish_status(AI_ERR_IMAGE_FORMAT, "OCR_Recognize");
        return g_compat_ocr_result.c_str();
    }
    const auto engine = get_engine();
    std::vector<AIOcrLine> lines;
    std::string error;
    const int32_t status = recognize_compat_ocr(engine, big.image, color_filter, min_confidence, &lines, &error);
    if (status < 0) {
        finish_status_with_detail(status, "OCR_Recognize", error);
        return g_compat_ocr_result.c_str();
    }
    if (output_format == AI_OCR_OUTPUT_JSON && !offset_ocr_lines(&lines, origin_x, origin_y)) {
        finish_status_with_detail(
            AI_ERR_INVALID_ARGUMENT, "OCR_Recognize", "coordinate origin causes int32 overflow");
        return g_compat_ocr_result.c_str();
    }
    const std::string utf8_result = format_compat_ocr_output(
        lines, static_cast<int32_t>(lines.size()), output_format);
    if (!utf8_to_windows_acp(utf8_result, &g_compat_ocr_result)) {
        g_compat_ocr_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "OCR_Recognize", "failed to convert UTF-8 OCR text to the current Windows ANSI code page");
        return g_compat_ocr_result.c_str();
    }
    finish_status(static_cast<int32_t>(lines.size()), "OCR_Recognize");
    return g_compat_ocr_result.c_str();
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_FindOneText(
    const uint8_t* big_data,
    int32_t big_size,
    const char* target_utf8,
    float min_confidence,
    OCRTextResult* output,
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y) {
    if (output != nullptr) std::memset(output, 0, sizeof(*output));
    const std::vector<std::string> target_candidates = compat_text_candidates(target_utf8);
    if (output == nullptr || target_candidates.empty() || target_candidates.front().empty() ||
        min_confidence < 0.0f || min_confidence > 1.0f) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneText");
    }
    ai::OcrColorFilter parsed_filter;
    std::string filter_error;
    if (!ai::parse_ocr_color_filter(color_filter, &parsed_filter, &filter_error)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneText", filter_error);
    }
#if defined(_WIN32) && defined(_M_IX86)
    std::vector<uint8_t> request;
    proxy_append_bytes(&request, big_data, big_size);
    proxy_append_string_candidates(&request, target_candidates);
    proxy_append_f32(&request, min_confidence);
    proxy_append_string(&request, color_filter == nullptr ? "" : color_filter);
    proxy_append_i32(&request, origin_x);
    proxy_append_i32(&request, origin_y);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_ocr_request(
        ai_worker::CMD_OCR_FIND_ONE, request, &status, &response);
    if (status > 0 && !proxy_read_sized_bytes(response, output, sizeof(OCRTextResult))) {
        return finish_status_with_detail(
            AI_ERR_RUNTIME,
            "OCR_FindOneText",
            "worker returned an invalid OCRTextResult payload");
    }
    return finish_proxy_status(status > 0 ? 1 : status, "OCR_FindOneText");
#else
    if (output == nullptr) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneText");
    }
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        return finish_status(AI_ERR_IMAGE_FORMAT, "OCR_FindOneText");
    }
    const auto engine = get_engine();
    std::vector<AIOcrLine> recognized_lines;
    std::string error;
    const ai::OcrTargetGroups selection_targets{target_candidates};
    ai::OcrPipelineSelection selection;
    int32_t status = recognize_compat_ocr(
        engine,
        big.image,
        color_filter,
        min_confidence,
        &recognized_lines,
        &error,
        &selection_targets,
        &selection);
    std::vector<AIOcrLine> lines;
    for (size_t i = 0; status >= 0 && i < target_candidates.size(); ++i) {
        status = find_ocr_text_in_lines(
            recognized_lines,
            target_candidates[i],
            &lines,
            selection.used_merged_line_recognition);
        if (status > 0) break;
    }
    if (status > 0) {
        fill_ocr_text_result(lines.front(), output);
        if (!offset_ocr_text_result(output, origin_x, origin_y)) {
            *output = OCRTextResult{};
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "OCR_FindOneText",
                "coordinate origin causes int32 overflow");
        }
    }
    return finish_status_with_detail(status > 0 ? 1 : status, "OCR_FindOneText", error);
#endif
}

AIENGINE_EXPORT const char* AIENGINE_CALL OCR_FindMultiText(
    const uint8_t* big_data,
    int32_t big_size,
    const char* targets_utf8,
    float min_confidence,
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y) {
    g_compat_ocr_json_result.clear();
    if (big_data == nullptr || big_size <= 0 || targets_utf8 == nullptr || targets_utf8[0] == '\0' ||
        min_confidence < 0.0f || min_confidence > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindMultiText");
        return g_compat_ocr_json_result.c_str();
    }
    const std::vector<std::vector<std::string>> target_candidates = compat_pipe_text_candidates(targets_utf8);
    if (target_candidates.empty()) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindMultiText");
        return g_compat_ocr_json_result.c_str();
    }
    ai::OcrColorFilter parsed_filter;
    std::string filter_error;
    if (!ai::parse_ocr_color_filter(color_filter, &parsed_filter, &filter_error)) {
        finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_FindMultiText", filter_error);
        return g_compat_ocr_json_result.c_str();
    }
#if defined(_WIN32) && defined(_M_IX86)
    std::vector<uint8_t> request;
    proxy_append_bytes(&request, big_data, big_size);
    proxy_append_i32(&request, static_cast<int32_t>(target_candidates.size()));
    for (const auto& candidates : target_candidates) proxy_append_string_candidates(&request, candidates);
    proxy_append_f32(&request, min_confidence);
    proxy_append_string(&request, color_filter == nullptr ? "" : color_filter);
    proxy_append_i32(&request, origin_x);
    proxy_append_i32(&request, origin_y);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_ocr_request(
        ai_worker::CMD_OCR_FIND_MULTI, request, &status, &response);
    if (status < 0) {
        finish_proxy_status(status, "OCR_FindMultiText");
        return g_compat_ocr_json_result.c_str();
    }
    std::string compact_text;
    if (!proxy_read_string(response, &compact_text) || !utf8_to_windows_acp(compact_text, &g_compat_ocr_json_result)) {
        g_compat_ocr_json_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "OCR_FindMultiText", "failed to decode or convert worker OCR compact result text");
        return g_compat_ocr_json_result.c_str();
    }
    finish_status(status, "OCR_FindMultiText");
    return g_compat_ocr_json_result.c_str();
#else
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        finish_status(AI_ERR_IMAGE_FORMAT, "OCR_FindMultiText");
        return g_compat_ocr_json_result.c_str();
    }
    std::vector<OcrTargetMatch> collected;
    if (target_candidates.empty()) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindMultiText");
        return g_compat_ocr_json_result.c_str();
    }
    const auto engine = get_engine();
    std::vector<AIOcrLine> recognized_lines;
    std::string error;
    ai::OcrPipelineSelection selection;
    int32_t status = recognize_compat_ocr(
        engine,
        big.image,
        color_filter,
        min_confidence,
        &recognized_lines,
        &error,
        &target_candidates,
        &selection);
    if (status < 0) {
        finish_status_with_detail(status, "OCR_FindMultiText", error);
        return g_compat_ocr_json_result.c_str();
    }
    for (size_t target_index = 0; target_index < target_candidates.size(); ++target_index) {
        std::vector<AIOcrLine> matches;
        std::string matched_target;
        for (const std::string& target : target_candidates[target_index]) {
            status = find_ocr_text_in_lines(
                recognized_lines,
                target,
                &matches,
                selection.used_merged_line_recognition);
            if (status < 0 || status > 0) {
                matched_target = target;
                break;
            }
        }
        if (status < 0) {
            finish_status(status, "OCR_FindMultiText");
            return g_compat_ocr_json_result.c_str();
        }
        for (const AIOcrLine& line : matches) {
            AIOcrLine shifted = line;
            if (!offset_ocr_line(&shifted, origin_x, origin_y)) {
                finish_status_with_detail(
                    AI_ERR_INVALID_ARGUMENT,
                    "OCR_FindMultiText",
                    "coordinate origin causes int32 overflow");
                return g_compat_ocr_json_result.c_str();
            }
            collected.push_back(OcrTargetMatch{
                static_cast<int32_t>(target_index), matched_target, shifted});
        }
    }
    const int32_t count = static_cast<int32_t>(collected.size());
    const std::string compact_text = format_ocr_target_text(collected);
    if (!utf8_to_windows_acp(compact_text, &g_compat_ocr_json_result)) {
        g_compat_ocr_json_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "OCR_FindMultiText", "failed to convert compact result text to the current Windows ANSI code page");
        return g_compat_ocr_json_result.c_str();
    }
    finish_status(count, "OCR_FindMultiText");
    return g_compat_ocr_json_result.c_str();
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_FindOneCoord(
    const uint8_t* big_data,
    int32_t big_size,
    const char* target_utf8,
    float min_confidence,
    OCRCoordResult* output,
    const char* color_filter,
    int32_t origin_x,
    int32_t origin_y) {
    if (output != nullptr) *output = OCRCoordResult{};
    const std::vector<std::string> target_candidates = compat_text_candidates(target_utf8);
    if (output == nullptr || target_candidates.empty() || target_candidates.front().empty() ||
        min_confidence < 0.0f || min_confidence > 1.0f) {
        return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneCoord");
    }
    ai::OcrColorFilter parsed_filter;
    std::string filter_error;
    if (!ai::parse_ocr_color_filter(color_filter, &parsed_filter, &filter_error)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneCoord", filter_error);
    }
#if defined(_WIN32) && defined(_M_IX86)
    std::vector<uint8_t> request;
    proxy_append_bytes(&request, big_data, big_size);
    proxy_append_string_candidates(&request, target_candidates);
    proxy_append_f32(&request, min_confidence);
    proxy_append_string(&request, color_filter == nullptr ? "" : color_filter);
    proxy_append_i32(&request, origin_x);
    proxy_append_i32(&request, origin_y);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_ocr_request(
        ai_worker::CMD_OCR_FIND_ONE_COORD, request, &status, &response);
    if (status > 0 && !proxy_read_sized_bytes(response, output, sizeof(OCRCoordResult))) {
        return finish_status_with_detail(AI_ERR_RUNTIME, "OCR_FindOneCoord", "worker returned an invalid OCRCoordResult payload");
    }
    return finish_proxy_status(status > 0 ? 1 : status, "OCR_FindOneCoord");
#else
    if (output == nullptr) return finish_status(AI_ERR_INVALID_ARGUMENT, "OCR_FindOneCoord");
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        return finish_status(AI_ERR_IMAGE_FORMAT, "OCR_FindOneCoord");
    }
    const auto engine = get_engine();
    std::vector<AIOcrLine> recognized_lines;
    std::string error;
    const ai::OcrTargetGroups selection_targets{target_candidates};
    ai::OcrPipelineSelection selection;
    int32_t status = recognize_compat_ocr(
        engine,
        big.image,
        color_filter,
        min_confidence,
        &recognized_lines,
        &error,
        &selection_targets,
        &selection);
    std::vector<AIOcrLine> lines;
    for (size_t i = 0; status >= 0 && i < target_candidates.size(); ++i) {
        status = find_ocr_text_in_lines(
            recognized_lines,
            target_candidates[i],
            &lines,
            selection.used_merged_line_recognition);
        if (status > 0) break;
    }
    if (status > 0) {
        fill_ocr_coord_result(lines.front(), 0, output);
        if (!offset_ocr_coord_result(output, origin_x, origin_y)) {
            *output = OCRCoordResult{};
            return finish_status_with_detail(
                AI_ERR_INVALID_ARGUMENT,
                "OCR_FindOneCoord",
                "coordinate origin causes int32 overflow");
        }
    }
    return finish_status_with_detail(status > 0 ? 1 : status, "OCR_FindOneCoord", error);
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL OCR_Release(void) {
    return AI_OcrRelease();
}

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_Create(int32_t* out_handle) {
    if (out_handle == nullptr) return finish_status(AI_ERR_INVALID_ARGUMENT, "YOLO_Create");
#if defined(_WIN32) && defined(_M_IX86)
    std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
    if (g_next_proxy_yolo_handle <= 0 || g_next_proxy_yolo_handle == std::numeric_limits<int32_t>::max()) {
        return finish_status(AI_ERR_RUNTIME, "YOLO_Create");
    }
    const int32_t local_handle = g_next_proxy_yolo_handle++;
    g_proxy_yolo_handles.emplace(local_handle, ProxyYoloIdentity{});
    *out_handle = local_handle;
    return finish_status(AI_OK, "YOLO_Create");
#else
    return finish_status(create_local_yolo_context(out_handle), "YOLO_Create");
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_LoadModelFromPath(
    int32_t handle,
    const char* model_path,
    const char* labels_path,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
    std::string error;
    YoloRuntimeParams params;
    const int32_t params_status = make_yolo_runtime_params(
        input_size, runtime_device, device_id, session_count, &params, &error);
    if (params_status < 0) return finish_status_with_detail(params_status, "YOLO_LoadModelFromPath", error);
#if defined(_WIN32) && defined(_M_IX86)
    const std::string model_abs = proxy_compat_absolute_path(model_path);
    const std::string labels_abs = proxy_compat_absolute_path(labels_path);
    if (model_abs.empty() || (labels_path != nullptr && labels_path[0] != '\0' && labels_abs.empty())) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "YOLO_LoadModelFromPath", "YOLO model or labels path is invalid");
    }
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_load_yolo_candidates(
        handle,
        runtime_device,
        ai_worker::CMD_YOLO_LOAD_PATH,
        [&](const ProxyYoloIdentity& identity, const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_yolo_identity(&request, identity);
            proxy_append_string(&request, model_abs.c_str());
            proxy_append_string(&request, labels_abs.c_str());
            proxy_append_yolo_params(&request, input_size, candidate.device, device_id, session_count);
            return request;
        },
        &status,
        &response);
    return finish_proxy_status(status, "YOLO_LoadModelFromPath");
#else
    const std::filesystem::path model_resolved = resolve_compat_api_path(model_path);
    std::vector<uint8_t> model_bytes;
    if (model_resolved.empty() || !read_file_bytes(model_resolved, &model_bytes)) {
        return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "YOLO_LoadModelFromPath", "failed to read YOLO model file");
    }
    auto model = std::make_shared<std::vector<uint8_t>>(std::move(model_bytes));
    ai::Config config;
    config.set_string("yolo.backend", "onnxruntime");
    if (labels_path != nullptr && labels_path[0] != '\0') {
        const std::filesystem::path resolved_path = resolve_compat_api_path(labels_path);
        if (resolved_path.empty()) {
            return finish_status_with_detail(AI_ERR_INVALID_ARGUMENT, "YOLO_LoadModelFromPath", "YOLO labels path is invalid");
        }
        config.set_string("yolo.labels_path", resolved_path.u8string());
    }
    const int32_t status = load_local_yolo_context(handle, model, std::move(config), &params, &error);
    return finish_status_with_detail(status, "YOLO_LoadModelFromPath", error);
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_LoadModelFromMemory(
    int32_t handle,
    const void* model_data,
    int32_t model_size,
    const void* labels_data,
    int32_t labels_size,
    int32_t input_size,
    int32_t runtime_device,
    int32_t device_id,
    int32_t session_count) {
    std::string error;
    YoloRuntimeParams params;
    const int32_t params_status = make_yolo_runtime_params(
        input_size, runtime_device, device_id, session_count, &params, &error);
    if (params_status < 0 || model_data == nullptr || model_size <= 0 || labels_size < 0 || (labels_size > 0 && labels_data == nullptr)) {
        return finish_status_with_detail(params_status < 0 ? params_status : AI_ERR_INVALID_ARGUMENT, "YOLO_LoadModelFromMemory", error);
    }
#if defined(_WIN32) && defined(_M_IX86)
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_load_yolo_candidates(
        handle,
        runtime_device,
        ai_worker::CMD_YOLO_LOAD_MEMORY,
        [&](const ProxyYoloIdentity& identity, const ProxyRuntimeCandidate& candidate) {
            std::vector<uint8_t> request;
            proxy_append_yolo_identity(&request, identity);
            proxy_append_bytes(&request, model_data, model_size);
            proxy_append_bytes(&request, labels_data, labels_size);
            proxy_append_yolo_params(&request, input_size, candidate.device, device_id, session_count);
            return request;
        },
        &status,
        &response);
    return finish_proxy_status(status, "YOLO_LoadModelFromMemory");
#else
    auto model = std::make_shared<std::vector<uint8_t>>(
        static_cast<const uint8_t*>(model_data), static_cast<const uint8_t*>(model_data) + model_size);
    ai::Config config;
    config.set_string("yolo.backend", "onnxruntime");
    if (labels_data != nullptr && labels_size > 0) {
        config.set_string("yolo.labels_inline", std::string(static_cast<const char*>(labels_data), static_cast<size_t>(labels_size)));
    }
    const int32_t status = load_local_yolo_context(handle, model, std::move(config), &params, &error);
    return finish_status_with_detail(status, "YOLO_LoadModelFromMemory", error);
#endif
}

AIENGINE_EXPORT const char* AIENGINE_CALL YOLO_InferJson(
    int32_t handle,
    const uint8_t* big_data,
    int32_t big_size,
    float conf,
    int32_t origin_x,
    int32_t origin_y) {
    g_yolo_json_result.clear();
    if (big_data == nullptr || big_size <= 0 || conf < 0.0f || conf > 1.0f) {
        finish_status(AI_ERR_INVALID_ARGUMENT, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
#if defined(_WIN32) && defined(_M_IX86)
    ProxyYoloIdentity identity{};
    if (!get_proxy_yolo_identity(handle, &identity)) {
        finish_status(AI_ERR_INVALID_HANDLE, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
    std::vector<uint8_t> request;
    proxy_append_yolo_identity(&request, identity);
    proxy_append_bytes(&request, big_data, big_size);
    proxy_append_f32(&request, conf);
    proxy_append_i32(&request, origin_x);
    proxy_append_i32(&request, origin_y);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_request(identity.flavor, ai_worker::CMD_YOLO_INFER_JSON, request, &status, &response);
    if (status >= 0) {
        std::string utf8_json;
        if (!proxy_read_string(response, &utf8_json) || !utf8_to_windows_acp(utf8_json, &g_yolo_json_result)) {
            g_yolo_json_result.clear();
            finish_status_with_detail(AI_ERR_RUNTIME, "YOLO_InferJson", "failed to convert worker UTF-8 JSON to the current Windows ANSI code page");
            return g_yolo_json_result.c_str();
        }
        finish_status(status, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
    finish_proxy_status(status, "YOLO_InferJson");
    return g_yolo_json_result.c_str();
#else
    BmpImageView big;
    if (!parse_bmp_view(big_data, big_size, &big, false)) {
        finish_status(AI_ERR_IMAGE_FORMAT, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
    int32_t lookup_status = AI_OK;
    const auto pool = get_loaded_yolo_pool(handle, &lookup_status);
    if (!pool) {
        finish_status(lookup_status, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
    std::vector<AIDetectBox> results;
    const int32_t status = pool->detect(big.image, conf, &results);
    if (status < 0 || status != static_cast<int32_t>(results.size())) {
        finish_status(status < 0 ? status : AI_ERR_RUNTIME, "YOLO_InferJson");
        return g_yolo_json_result.c_str();
    }
    if (!offset_yolo_boxes(&results, origin_x, origin_y)) {
        finish_status_with_detail(
            AI_ERR_INVALID_ARGUMENT,
            "YOLO_InferJson",
            "coordinate origin causes int32 overflow");
        return g_yolo_json_result.c_str();
    }
    const std::string utf8_json = format_yolo_json(results, status);
    if (!utf8_to_windows_acp(utf8_json, &g_yolo_json_result)) {
        g_yolo_json_result.clear();
        finish_status_with_detail(AI_ERR_RUNTIME, "YOLO_InferJson", "failed to convert UTF-8 JSON to the current Windows ANSI code page");
        return g_yolo_json_result.c_str();
    }
    finish_status(status, "YOLO_InferJson");
    return g_yolo_json_result.c_str();
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_GetRuntimeStatusJson(int32_t handle, char* output, int32_t output_size) {
#if defined(_WIN32) && defined(_M_IX86)
    ProxyYoloIdentity identity{};
    if (!get_proxy_yolo_identity(handle, &identity)) return finish_status(AI_ERR_INVALID_HANDLE, "YOLO_GetRuntimeStatusJson");
    std::vector<uint8_t> request;
    proxy_append_yolo_identity(&request, identity);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_request(identity.flavor, ai_worker::CMD_YOLO_RUNTIME_STATUS, request, &status, &response);
    if (status < 0) return finish_proxy_status(status, "YOLO_GetRuntimeStatusJson");
    std::string text;
    if (!proxy_read_string(response, &text)) return finish_status(AI_ERR_RUNTIME, "YOLO_GetRuntimeStatusJson");
    proxy_rewrite_runtime_status(
        &text, identity.requested_device, identity.active_device, identity.fallback_reason);
    return write_string_result(text, output, output_size, AI_OK, "YOLO_GetRuntimeStatusJson");
#else
    int32_t lookup_status = AI_OK;
    const auto pool = get_loaded_yolo_pool(handle, &lookup_status);
    if (!pool) return finish_status(lookup_status, "YOLO_GetRuntimeStatusJson");
    const std::string json = std::string("{\"handle\":") + std::to_string(handle) +
        "," + runtime_status_fields(pool->runtime) +
        ",\"device_id\":" + std::to_string(pool->device_id) +
        ",\"session_count\":" + std::to_string(pool->session_count) +
        ",\"intra_op_threads\":" + std::to_string(pool->intra_op_threads) +
        ",\"input_width\":" + std::to_string(pool->input_width) +
        ",\"input_height\":" + std::to_string(pool->input_height) + "}";
    return write_string_result(json, output, output_size, AI_OK, "YOLO_GetRuntimeStatusJson");
#endif
}

AIENGINE_EXPORT int64_t AIENGINE_CALL YOLO_GetLastLatencyUs(int32_t handle) {
#if defined(_WIN32) && defined(_M_IX86)
    ProxyYoloIdentity identity{};
    if (!get_proxy_yolo_identity(handle, &identity)) return -1;
    std::vector<uint8_t> request;
    proxy_append_yolo_identity(&request, identity);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_request(identity.flavor, ai_worker::CMD_YOLO_LAST_LATENCY, request, &status, &response);
    int64_t latency = -1;
    if (status >= 0 && response.size() == sizeof(latency)) std::memcpy(&latency, response.data(), sizeof(latency));
    return latency;
#else
    int32_t lookup_status = AI_OK;
    const auto pool = get_loaded_yolo_pool(handle, &lookup_status);
    return pool ? pool->last_latency_us.load(std::memory_order_relaxed) : -1;
#endif
}

AIENGINE_EXPORT int32_t AIENGINE_CALL YOLO_Release(int32_t handle) {
#if defined(_WIN32) && defined(_M_IX86)
    ProxyYoloIdentity identity{};
    if (proxy_yolo_is_loading(handle)) {
        return finish_status_with_detail(
            AI_ERR_BUSY,
            "YOLO_Release",
            "YOLO handle is loading a model");
    }
    if (proxy_yolo_is_empty(handle)) {
        std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
        g_proxy_yolo_handles.erase(handle);
        return finish_status(AI_OK, "YOLO_Release");
    }
    if (!get_proxy_yolo_identity(handle, &identity)) return finish_status(AI_ERR_INVALID_HANDLE, "YOLO_Release");
    std::vector<uint8_t> request;
    proxy_append_yolo_identity(&request, identity);
    int32_t status = AI_ERR_RUNTIME;
    std::vector<uint8_t> response;
    proxy_request(identity.flavor, ai_worker::CMD_YOLO_RELEASE, request, &status, &response, false);
    {
        std::lock_guard<std::mutex> lock(g_proxy_yolo_mutex);
        g_proxy_yolo_handles.erase(handle);
    }
    return finish_proxy_status(status, "YOLO_Release");
#else
    std::shared_ptr<YoloModelContext> context;
    {
        std::lock_guard<std::mutex> lock(g_yolo_context_mutex);
        const auto it = g_yolo_contexts.find(handle);
        if (it == g_yolo_contexts.end()) return finish_status(AI_ERR_INVALID_HANDLE, "YOLO_Release");
        context = it->second;
        g_yolo_contexts.erase(it);
    }
    close_yolo_context(context);
    return finish_status(AI_OK, "YOLO_Release");
#endif
}
